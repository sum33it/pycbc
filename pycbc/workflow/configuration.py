# Copyright (C) 2013,2017 Ian Harry, Duncan Brown
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This module provides a wrapper to the ConfigParser utilities for pycbc
workflow construction. This module is described in the page here:
https://ldas-jobs.ligo.caltech.edu/~cbc/docs/pycbc/ahope/initialization_inifile.html
"""

import re
import os
import logging
import stat
import shutil
import subprocess
from shutil import which
import urllib.parse
import hashlib

from pycbc.types.config import InterpolatingConfigParser

logger = logging.getLogger('pycbc.workflow.configuration')

# NOTE urllib is weird. For some reason it only allows known schemes and will
# give *wrong* results, rather then failing, if you use something like gsiftp
# We can add schemes explicitly, as below, but be careful with this!
urllib.parse.uses_relative.append('osdf')
urllib.parse.uses_netloc.append('osdf')


def hash_compare(filename_1, filename_2, chunk_size=None, max_chunks=None):
    """
    Calculate the sha1 hash of a file, or of part of a file

    Parameters
    ----------
    filename_1 : string or path
        the first file to be hashed / compared
    filename_2 : string or path
        the second file to be hashed / compared
    chunk_size : integer
        This size of chunks to be read in and hashed. If not given, will read
        the whole file (may be slow for large files).
    max_chunks: integer
        This many chunks to be compared. If all chunks so far have been the
        same, then just assume its the same file. Default 10

    Returns
    -------
    hash : string
        The hexdigest() after a sha1 hash of (part of) the file
    """

    if max_chunks is None and chunk_size is not None:
        max_chunks = 10
    elif chunk_size is None:
        max_chunks = 1

    with open(filename_1, 'rb') as f1:
        with open(filename_2, 'rb') as f2:
            for _ in range(max_chunks):
                h1 = hashlib.sha1(f1.read(chunk_size)).hexdigest()
                h2 = hashlib.sha1(f2.read(chunk_size)).hexdigest()
                if h1 != h2:
                    return False
    return True


def resolve_url_http(url, u, filename):
    """Helper function used by `resolve_url()` to handle HTTP and HTTPS URLs.
    """

    # Would like to move ciecplib import to top using import_optional, but
    # it needs to be available when documentation runs in the CI, and I
    # can't get it to install in the GitHub CI
    import ciecplib

    headers = None

    if u.netloc == 'git.ligo.org':
        # We need to do two ugly special things to download a file from
        # git.ligo.org. First, we need to pass a per-user GitLab Personal Access
        # Token via the headers. Second, we need to translate the raw-download
        # URL scheme to a different scheme that uses GitLab's REST API.
        re_match = re.match(
            'https://git.ligo.org/([^ ]+(?<!/-))/(?:-/)?raw/([^ /]+)/(.+)', url
        )
        assert re_match, f'Failed to parse URL {url} for git.ligo.org special case'
        project = re_match.group(1)
        ref = re_match.group(2)
        file_path = re_match.group(3)
        logging.info(
            'Special-case download from git.ligo.org: '
            'file %s from ref %s of project %s',
            file_path,
            ref,
            project
        )
        project = urllib.parse.quote(project, safe='')
        ref = urllib.parse.quote(ref, safe='')
        # GitLab's REST API does not seem to like double slashes
        file_path = os.path.normpath(file_path)
        file_path = urllib.parse.quote(file_path, safe='')
        # FIXME This code will break once GitLab changes their mind about the
        # URL format. As a different approach, we could rely on the gitlab
        # Python module to (arguably) make this more robust. For now I prefer
        # to avoid adding one more dependency, and stick with ciecplib.
        url = f'https://git.ligo.org/api/v4/projects/{project}/repository/files/{file_path}/raw?ref={ref}&lfs=true'
        # Get the user's Personal Access Token and pass it as a header
        xdg_config_home = (
            os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config')
        )
        with open(f'{xdg_config_home}/pycbc/git-ligo-org.pat', 'rb') as pat_fh:
            headers = {
                'PRIVATE-TOKEN': pat_fh.read().decode('ascii').strip()
            }

    # Make the scitokens logger a little quieter
    # (it is called through ciecpclib)
    curr_level = logging.getLogger().level
    logging.getLogger('scitokens').setLevel(curr_level + 10)

    with ciecplib.Session() as s:
        r = s.get(url, allow_redirects=True, headers=headers)
        r.raise_for_status()
    content = r.content

    with open(filename, "wb") as output_fp:
        output_fp.write(r.content)


def resolve_url(
    url,
    directory=None,
    permissions=None,
    copy_to_cwd=True,
    hash_max_chunks=None,
    hash_chunk_size=None,
):
    """Resolves a URL to a local file, and returns the path to that file.

    If a URL is given, the file will be copied to the current working
    directory. If a local file path is given, the file will only be copied
    to the current working directory if ``copy_to_cwd`` is ``True``
    (the default).
    """

    u = urllib.parse.urlparse(url)

    # determine whether the file exists locally
    islocal = u.scheme == "" or u.scheme == "file"

    if not islocal or copy_to_cwd:
        # create the name of the destination file
        if directory is None:
            directory = os.getcwd()
        filename = os.path.join(directory, os.path.basename(u.path))
    else:
        filename = u.path

    if islocal:
        # check that the file exists
        if not os.path.isfile(u.path):
            errmsg = "Cannot open file %s from URL %s" % (u.path, url)
            raise ValueError(errmsg)
        # for regular files, make a direct copy if requested
        elif copy_to_cwd:
            if os.path.isfile(filename):
                # check to see if src and dest are the same file
                same_file = hash_compare(
                    u.path,
                    filename,
                    chunk_size=hash_chunk_size,
                    max_chunks=hash_max_chunks
                )
                if not same_file:
                    shutil.copy(u.path, filename)
            else:
                shutil.copy(u.path, filename)

    elif u.scheme in ("http", "https"):
        resolve_url_http(url, u, filename)

    elif u.scheme == "osdf":
        # OSDF will require a scitoken to be present and stashcp to be
        # available. Thanks Dunky for the code here!
        cmd = [
            which("pelican") or "pelican",
            "object",
            "get",
            u.scheme + "://" + u.path,
            filename
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            # Print information about the failure
            logging.error(
                'Command %s failed with the following output:', err.cmd
            )
            logging.error(err.stderr.decode())
            logging.error(err.stdout.decode())
            raise

    else:
        # TODO: We could support other schemes as needed
        errmsg = "Unknown URL scheme: %s\n" % (u.scheme)
        errmsg += "Currently supported are: file, http, and https."
        raise ValueError(errmsg)

    if not os.path.isfile(filename):
        errmsg = "Error trying to create file %s from %s" % (filename, url)
        raise ValueError(errmsg)

    if permissions:
        if os.access(filename, os.W_OK):
            os.chmod(filename, permissions)
        else:
            # check that the file has at least the permissions requested
            s = os.stat(filename)[stat.ST_MODE]
            if (s & permissions) != permissions:
                errmsg = "Could not change permissions on %s (read-only)" % url
                raise ValueError(errmsg)

    return filename


def add_workflow_command_line_group(parser):
    """
    The standard way of initializing a ConfigParser object in workflow will be
    to do it from the command line. This is done by giving a

    --local-config-files filea.ini fileb.ini filec.ini

    command. You can also set config file override commands on the command
    line. This will be most useful when setting (for example) start and
    end times, or active ifos. This is done by

    --config-overrides section1:option1:value1 section2:option2:value2 ...

    This can also be given as

    --config-overrides section1:option1

    where the value will be left as ''.

    To remove a configuration option, use the command line argument

    --config-delete section1:option1

    which will delete option1 from [section1] or

    --config-delete section1

    to delete all of the options in [section1]

    Deletes are implemented before overrides.

    This function returns an argparse OptionGroup to ensure these options are
    parsed correctly and can then be sent directly to initialize an
    WorkflowConfigParser.

    Parameters
    -----------
    parser : argparse.ArgumentParser instance
        The initialized argparse instance to add the workflow option group to.
    """
    workflowArgs = parser.add_argument_group(
        "Configuration", "Options needed for parsing " "config file(s)."
    )
    workflowArgs.add_argument(
        "--config-files",
        nargs="+",
        action="store",
        metavar="CONFIGFILE",
        help="List of config files to be used in " "analysis.",
    )
    workflowArgs.add_argument(
        "--config-overrides",
        nargs="*",
        action="store",
        metavar="SECTION:OPTION:VALUE",
        help="List of section,option,value combinations to "
        "add into the configuration file. Normally the gps "
        "start and end times might be provided this way, "
        "and user specific locations (ie. output directories). "
        "This can also be provided as SECTION:OPTION or "
        "SECTION:OPTION: both of which indicate that the "
        "corresponding value is left blank.",
    )
    workflowArgs.add_argument(
        "--config-delete",
        nargs="*",
        action="store",
        metavar="SECTION:OPTION",
        help="List of section,option combinations to delete "
        "from the configuration file. This can also be "
        "provided as SECTION which deletes the enture section"
        " from the configuration file or SECTION:OPTION "
        "which deletes a specific option from a given "
        "section.",
    )


class WorkflowConfigParser(InterpolatingConfigParser):
    """
    This is a sub-class of InterpolatingConfigParser, which lets
    us add a few additional helper features that are useful in workflows.
    """

    def __init__(
        self,
        configFiles=None,
        overrideTuples=None,
        parsedFilePath=None,
        deleteTuples=None,
        copy_to_cwd=False,
    ):
        """
        Initialize an WorkflowConfigParser. This reads the input configuration
        files, overrides values if necessary and performs the interpolation.
        See https://ldas-jobs.ligo.caltech.edu/~cbc/docs/pycbc/ahope/initialization_inifile.html

        Parameters
        -----------
        configFiles : Path to .ini file, or list of paths
            The file(s) to be read in and parsed.
        overrideTuples : List of (section, option, value) tuples
            Add the (section, option, value) triplets provided
            in this list to the provided .ini file(s). If the section, option
            pair is already present, it will be overwritten.
        parsedFilePath : Path, optional (default=None)
            If given, write the parsed .ini file back to disk at this location.
        deleteTuples : List of (section, option) tuples
            Delete the (section, option) pairs provided
            in this list from provided .ini file(s). If the section only
            is provided, the entire section will be deleted.
        copy_to_cwd : bool, optional
            Copy the configuration files to the current working directory if
            they are not already there, even if they already exist locally.
            If False, files will only be copied to the current working
            directory if they are remote. Default is False.

        Returns
        --------
        WorkflowConfigParser
            Initialized WorkflowConfigParser instance.
        """
        if configFiles is not None:
            configFiles = [
                resolve_url(cFile, copy_to_cwd=copy_to_cwd)
                for cFile in configFiles
            ]

        InterpolatingConfigParser.__init__(
            self,
            configFiles,
            overrideTuples,
            parsedFilePath,
            deleteTuples,
            skip_extended=True,
        )
        # expand executable which statements
        self.perform_exe_expansion()

        # Resolve any URLs needing resolving
        self.curr_resolved_files = {}
        self.resolve_urls()

        # Check for any substitutions that can be made
        self.perform_extended_interpolation()

    def perform_exe_expansion(self):
        """
        This function will look through the executables section of the
        ConfigParser object and replace any values using macros with full paths.

        For any values that look like

        ${which:lalapps_tmpltbank}

        will be replaced with the equivalent of which(lalapps_tmpltbank)

        Otherwise values will be unchanged.
        """
        # Only works on executables section
        if self.has_section("executables"):
            for option, value in self.items("executables"):
                # Check the value
                newStr = self.interpolate_exe(value)
                if newStr != value:
                    self.set("executables", option, newStr)

    def interpolate_exe(self, testString):
        """
        Replace testString with a path to an executable based on the format.

        If this looks like

        ${which:lalapps_tmpltbank}

        it will return the equivalent of which(lalapps_tmpltbank)

        Otherwise it will return an unchanged string.

        Parameters
        -----------
        testString : string
            The input string

        Returns
        --------
        newString : string
            The output string.
        """
        # First check if any interpolation is needed and abort if not
        testString = testString.strip()
        if not (testString.startswith("${") and testString.endswith("}")):
            return testString

        # This may not be an exe interpolation, so even if it has ${ ... } form
        # I may not have to do anything
        newString = testString

        # Strip the ${ and }
        testString = testString[2:-1]

        testList = testString.split(":")

        # Maybe we can add a few different possibilities for substitution
        if len(testList) == 2:
            if testList[0] == "which":
                newString = which(testList[1])
                if not newString:
                    errmsg = "Cannot find exe %s in your path " % (testList[1])
                    errmsg += "and you specified ${which:%s}." % (testList[1])
                    raise ValueError(errmsg)

        return newString

    def section_to_cli(self, section, skip_opts=None):
        """Converts a section into a command-line string.

        For example:

        .. code::

            [section_name]
            foo =
            bar = 10

        yields: `'--foo --bar 10'`.

        Parameters
        ----------
        section : str
            The name of the section to convert.
        skip_opts : list, optional
            List of options to skip. Default (None) results in all options
            in the section being converted.

        Returns
        -------
        str :
            The options as a command-line string.
        """
        if skip_opts is None:
            skip_opts = []
        read_opts = [
            opt for opt in self.options(section) if opt not in skip_opts
        ]
        opts = []
        for opt in read_opts:
            opts.append("--{}".format(opt))
            val = self.get(section, opt)
            if val != "":
                opts.append(val)
        return " ".join(opts)

    def get_cli_option(self, section, option_name, **kwds):
        """Return option using CLI action parsing

        Parameters
        ----------
        section: str
            Section to find option to parse
        option_name: str
            Name of the option to parse from the config file
        kwds: keywords
            Additional keywords are passed directly to the argument parser.

        Returns
        -------
        value:
            The parsed value for this option
        """
        import argparse

        optstr = self.section_to_cli(section)
        parser = argparse.ArgumentParser()
        name = "--" + option_name.replace("_", "-")
        parser.add_argument(name, **kwds)
        args, _ = parser.parse_known_args(optstr.split())
        return getattr(args, option_name)

    def resolve_urls(self):
        """
        This function will look through all sections of the
        ConfigParser object and replace any URLs that are given the resolve
        magic flag with a path on the local drive.

        Specifically for any values that look like

        ${resolve:https://git.ligo.org/detchar/SOME_GATING_FILE.txt}

        the file will be replaced with the output of resolve_url(URL)

        Otherwise values will be unchanged.
        """
        # Only works on executables section
        for section in self.sections():
            for option, value in self.items(section):
                # Check the value
                value_l = value.split(' ')
                new_str_l = [self.resolve_file_url(val) for val in value_l]
                new_str = ' '.join(new_str_l)
                if new_str is not None and new_str != value:
                    self.set(section, option, new_str)

    def resolve_file_url(self, test_string):
        """
        Replace test_string with a path to an executable based on the format.

        If this looks like

        ${which:lalapps_tmpltbank}

        it will return the equivalent of which(lalapps_tmpltbank)

        Otherwise it will return an unchanged string.

        Parameters
        -----------
        test_string : string
            The input string

        Returns
        --------
        new_string : string
            The output string.
        """
        # First check if any interpolation is needed and abort if not
        test_string = test_string.strip()
        if not (test_string.startswith("${") and test_string.endswith("}")):
            return test_string

        # This may not be a "resolve" interpolation, so even if it has
        # ${ ... } form I may not have to do anything

        # Strip the ${ and }
        test_string_strip = test_string[2:-1]

        test_list = test_string_strip.split(":", 1)

        if len(test_list) == 2:
            if test_list[0] == "resolve":
                curr_lfn = os.path.basename(test_list[1])
                if curr_lfn in self.curr_resolved_files:
                    return self.curr_resolved_files[curr_lfn]
                local_url = resolve_url(test_list[1])
                self.curr_resolved_files[curr_lfn] = local_url
                return local_url

        return test_string
