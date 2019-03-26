#!/usr/bin/env python
"""
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Sync lldb and related source from a local machine to a remote machine.

This facilitates working on the lldb sourcecode on multiple machines
and multiple OS types, verifying changes across all.
"""

import argparse
import io
import importlib
import json
import os.path
import re
import sys

# Add the local lib directory to the python path.
LOCAL_LIB_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "lib")
sys.path.append(LOCAL_LIB_PATH)

import transfer.transfer_spec


DOTRC_BASE_FILENAME = ".syncsourcerc"


class Configuration(object):
    """Provides chaining configuration lookup."""

    def __init__(self, rcdata_configs):
        self.__rcdata_configs = rcdata_configs

    def get_value(self, key):
        """
        Return the first value in the parent chain that has the key.

        The traversal starts from the most derived configuration (i.e.
        child) and works all the way up the parent chain.

        @return the value of the first key in the parent chain that
        contains a value for the given key.
        """
        for config in self.__rcdata_configs:
            if key in config:
                return config[key]
        return None

    def __getitem__(self, key):
        value = self.get_value(key)
        if value:
            return value
        else:
            raise KeyError(key)


def parse_args():
    """@return options parsed from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", "-c", action="store", default="default",
        help="specify configuration name to use")
    parser.add_argument(
        "--default-excludes", action="store", default="*.git,*.svn,*.pyc",
        help=("comma-separated list of default file patterns to exclude "
              "from each source directory and to protect from deletion "
              "on each destination directory; if starting with forward "
              "slash, it only matches at the top of the base directory"))
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="do a dry run of the transfer operation, don't really transfer")
    parser.add_argument(
        "--rc-file", "-r", action="store",
        help="specify the sync-source rc file to use for configurations")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="turn on verbose output")
    return parser.parse_args()


def read_rcfile(filename):
    """Returns the json-parsed contents of the input file."""

    # First parse file contents, removing all comments but
    # preserving the line count.
    regex = re.compile(r"#.*$")

    comment_stripped_file = io.StringIO()
    with open(filename, "r") as json_file:
        for line in json_file:
            comment_stripped_file.write(regex.sub("", line))
    return json.load(io.StringIO(comment_stripped_file.getvalue()))


def find_appropriate_rcfile(options):
    # Use an options-specified rcfile if specified.
    if options.rc_file and len(options.rc_file) > 0:
        if not os.path.isfile(options.rc_file):
            # If it doesn't exist, error out here.
            raise "rcfile '{}' specified but doesn't exist".format(
                options.rc_file)
        return options.rc_file

    # Check if current directory .sync-sourcerc exists.  If so, use it.
    local_rc_filename = os.path.abspath(DOTRC_BASE_FILENAME)
    if os.path.isfile(local_rc_filename):
        return local_rc_filename

    # Check if home directory .sync-sourcerc exists.  If so, use it.
    homedir_rc_filename = os.path.abspath(
        os.path.join(os.path.expanduser("~"), DOTRC_BASE_FILENAME))
    if os.path.isfile(homedir_rc_filename):
        return homedir_rc_filename

    # Nothing matched.  We don't have an rc filename candidate.
    return None


def get_configuration(options, rcdata, config_name):
    rcdata_configs = []
    next_config_name = config_name
    while next_config_name:
        # Find the next rcdata configuration for the given name.
        rcdata_config = next(
            config for config in rcdata["configurations"]
            if config["name"] == next_config_name)

        # See if we found it.
        if rcdata_config:
            # This is our next configuration to use in the chain.
            rcdata_configs.append(rcdata_config)

            # If we have a parent, check that next.
            if "parent" in rcdata_config:
                next_config_name = rcdata_config["parent"]
            else:
                next_config_name = None
        else:
            raise "failed to find specified parent config '{}'".format(
                next_config_name)
    return Configuration(rcdata_configs)


def create_transfer_agent(options, configuration):
    transfer_class_spec = configuration.get_value("transfer_class")
    if options.verbose:
        print("specified transfer class: '{}'".format(transfer_class_spec))

    # Load the module (possibly package-qualified).
    components = transfer_class_spec.split(".")
    module = importlib.import_module(".".join(components[:-1]))

    # Create the class name we need to load.
    clazz = getattr(module, components[-1])
    return clazz(options, configuration)


def sync_configured_sources(options, configuration, default_excludes):
    # Look up the transfer method.
    transfer_agent = create_transfer_agent(options, configuration)

    # For each configured dir_names source, do the following transfer:
    #   1. Start with base_dir + {source-dir-name}_dir
    #   2. Copy all files recursively, but exclude
    #      all dirs specified by source_excludes:
    #      skip all base_dir + {source-dir-name}_dir +
    #      {source-dir-name}_dir excludes.
    source_dirs = configuration.get_value("source")
    source_excludes = configuration.get_value("source_excludes")
    dest_dirs = configuration.get_value("dest")

    source_base_dir = source_dirs["base_dir"]
    dest_base_dir = dest_dirs["base_dir"]
    dir_ids = configuration.get_value("dir_names")
    transfer_specs = []

    for dir_id in dir_ids:
        dir_key = "{}_dir".format(dir_id)

        # Build the source dir (absolute) that we're copying from.
        # Defaults the base-relative source dir to the source id (e.g. lldb)
        rel_source_dir = source_dirs.get(dir_key, dir_id)
        transfer_source_dir = os.path.expanduser(
            os.path.join(source_base_dir, rel_source_dir))

        # Exclude dirs do two things:
        # 1) stop items from being copied on the source side, and
        # 2) protect things from being deleted on the dest side.
        #
        # In both cases, they are specified relative to the base
        # directory on either the source or dest side.
        #
        # Specifying a leading '/' in the directory will limit it to
        # be rooted in the base directory.  i.e. "/.git" will only
        # match {base-dir}/.git, not {base-dir}/subdir/.git, but
        # ".svn" will match {base-dir}/.svn and
        # {base-dir}/subdir/.svn.
        #
        # If excludes are specified for this dir_id, then pass along
        # the excludes.  These are relative to the dir_id directory
        # source, and get passed along that way as well.
        transfer_source_excludes = []

        # Add the source excludes for this dir.
        skip_defaults = False
        if source_excludes and dir_key in source_excludes:
            transfer_source_excludes.extend(source_excludes[dir_key])
            if "<no-defaults>" in source_excludes[dir_key]:
                skip_defaults = True
                transfer_source_excludes.remove("<no-defaults>")

        if not skip_defaults and default_excludes is not None:
            transfer_source_excludes.extend(list(default_excludes))

        # Build the destination-base-relative dest dir into which
        # we'll be syncing.  Relative directory defaults to the
        # dir id
        rel_dest_dir = dest_dirs.get(dir_key, dir_id)
        transfer_dest_dir = os.path.join(dest_base_dir, rel_dest_dir)

        # Add the exploded paths to the list that we'll ask the
        # transfer agent to transfer for us.
        transfer_specs.append(
            transfer.transfer_spec.TransferSpec(
                transfer_source_dir,
                transfer_source_excludes,
                transfer_dest_dir))

    # Do the transfer.
    if len(transfer_specs) > 0:
        transfer_agent.transfer(transfer_specs, options.dry_run)
    else:
        raise Exception("nothing to transfer, bad configuration?")


def main():
    """Drives the main program."""
    options = parse_args()

    if options.default_excludes and len(options.default_excludes) > 0:
        default_excludes = options.default_excludes.split(",")
    else:
        default_excludes = []

    # Locate the rc filename to load, then load it.
    rc_filename = find_appropriate_rcfile(options)
    if rc_filename:
        if options.verbose:
            print("reading rc data from file '{}'".format(rc_filename))
        rcdata = read_rcfile(rc_filename)
    else:
        sys.stderr.write("no rcfile specified, cannot guess configuration")
        exit(1)

    # Find configuration.
    configuration = get_configuration(options, rcdata, options.config_name)
    if not configuration:
        sys.stderr.write("failed to find configuration for {}".format(
            options.config_data))
        exit(2)

    # Kick off the transfer.
    sync_configured_sources(options, configuration, default_excludes)

if __name__ == "__main__":
    main()
