syncsource.py

OVERVIEW

The syncsource.py utility transfers groups of files between
computers. The primary use case is to enable developing LLVM project
software on one machine, transfer it efficiently to other machines ---
possibly of other architectures --- and test it there. syncsource.py
supports configurable, named source-to-destination mappings and has a
transfer agent plug-in architecture. The current distribution provides
an rsync-over-ssh transfer agent.

The primary benefits of using syncsource.py are:

* Provides a simple, reliable way to get a mirror copy of primary-
  machine files onto several different destinations without concern
  of compromising the patch during testing on different machines.

* Handles directory-mapping differences between two machines.  For
  LLDB, this is helpful when going between OS X and any other non-OS X
  target system.

EXAMPLE WORKFLOW

This utility was developed in the context of working on the LLDB
project.  Below we show the transfers we'd like to have happen,
and the configuration that supports it.

Workflow Example:

* Develop on OS X (primary machine)
* Test candidate changes on OS X.
* Test candidate changes on a Linux machine (machine-name: lldb-linux).
* Test candidate changes on a FreeBSD machine (machine-name: lldb-freebsd).
* Do check-ins from OS X machine.

Requirements:

* OS X machine requires the lldb source layout: lldb, lldb/llvm,
  lldb/llvm/tools/clang. Note this is different than the canonical
  llvm, llvm/tools/clang, llvm/tools/lldb layout that we'll want on
  the Linux and FreeBSD test machines.

* Linux machine requires the llvm, llvm/tools/clang and
  llvm/tools/lldb layout.

* FreeBSD machine requires the same layout as the llvm machine.

syncsource.py configuration in ~/.syncsourcerc:

# This is my configuration with a comment.  Configuration
# files are JSON-based.
{ "configurations": [
    # Here we have a collection of named configuration blocks.
    # Configuration blocks can chain back to a parent by name.
    {
        # Every block has a name so it can be referenced from
        # the command line or chained back to by a child block
        # for sharing.
        "name": "base_tot_settings",

        # This directive lists the "directory ids" that we'll care
        # about.  If your local repository has additional directories
        # for other projects that need to ride along, add them here.
        # For defaulting purposes, it makes sense to name the
        # directory IDs as the most likely name for the directory
        # itself.  For stock LLDB from top of tree, we generally only
        # care about lldb, llvm and clang.
        "dir_names": [ "llvm", "clang", "lldb" ],

        # This section describes where the source files live on
        # the primary machine.  There should always be a base_dir
        # entry, which indicates where in the local filesystem the
        # projects are rooted.  For each dir in dir_names, there
        # should be either:
        # 1. an entry named {dir-id}_dir (e.g. llvm_dir), which
        #    specifies the source directory for the given dir id
        #    relative to the base_dir entry, OR
        # 2. no entry, in which case the directory is assumed to
        #    be the same as {dir-id}.  In the example below, the
        #    base_dir-relative directory for the "lldb" dir-id is
        #    defaulted to being "lldb".  That's exactly what
        #    we need in an OS X-style lldb dir layout.
        "source": {
            "base_dir": "~/work/lldb-tot",
            "llvm_dir": "lldb/llvm",
            "clang_dir": "lldb/llvm/tools/clang"
        },

        # source_excludes covers any exclusions that:
        # * should be applied when copying files from the source
        # * should be excluded from deletion on the destination
        #
        # By default, ".git", ".svn" and ".pyc" are added to
        # all dir-id exclusions.  The default excludes can be
        # controlled by the syncsource.py --default-excludes
        # option.
        #
        # Below, I have transfer of the lldb dir skip everything
        # rooted at "/llvm" below the lldb dir.  This is
        # because we want the source OS X lldb to move to
        # a destination of {some-dest-root}/llvm/tools/lldb, and
        # not have the OS-X-inverted llvm copy over with the lldb
        # transfer portion.  We'll see the complete picture of
        # how this works when we get to specifying destinations
        # later on in the config.
        #
        # We also exclude the "/build" and "/llvm-build" dir rooted in
        # the OS X-side sources.  The Xcode configuration on this
        # OS X machine will dump lldb builds in the /build directory
        # relative to the lldb dir, and it will build llvm+clang in
        # the /llvm-build dir relative to the lldb dir.
        #
        # Note the first forward slash in "/build" indicates to the
        # transfer agent that we only want to exclude the
        # ~/work/lldb-tot/lldb/build dir, not just any file or
        # directory named "build" somewhere underneath the lldb
        # directory.  Without the leading forward slash, any file
        # or directory called build anywhere underneath the lldb dir
        # will be excluded, which is definitely not what we want here.
        #
        # For the llvm dir, we do a source-side exclude for
        # "/tools/clang".  We manage the clang transfer as a separate
        # entity, so we don't want the movement of llvm to also move
        # clang.
        #
        # The llvm_dir exclusion of "/tools/lldb" is the first example
        # of an exclude targeting a requirement of the destination
        # side.  Normally the transfer agent will delete anything on
        # the destination that is not present on the source.  It is
        # trying to mirror, and ensure both sides have the same
        # content.  The source side of llvm on OS X does not have a
        # "/tools/lldb", so at first this exclude looks non-sensical.
        # But on the canonical destination layout, lldb lives in
        # {some-dest-root}/llvm/tools/lldb.  Without this exclude,
        # the transfer agent would blow away the tools/lldb directory
        # on the destination every time we transfer, and then have to
        # copy the lldb dir all over again.  For rsync+ssh, that
        # totally would defeat the huge transfer efficiencies gained
        # by using rsync in the first place.
        #
        # Note the overloading of both source and dest style excludes
        # ultimately comes from the rsync-style exclude mechanism.
        # If it wasn't for that, I would have separated source and
        # dest excludes out better.
        "source_excludes": {
            "lldb_dir": ["/llvm", "/build", "/llvm-build"],
            "llvm_dir": ["/tools/lldb", "/tools/clang"]
        }
    },

    # Top of tree public, common settings for all destinations.
    {
        # The name for this config block.
        "name": "common_tot",

        # Here is our first chaining back to a parent config block.
        # Any settings in "common_tot" not specified here are going
        # to be retrieved from the parent.
        "parent": "base_tot_settings",

        # The transfer agent class to use.  Right now, the only one
        # available is this one here that uses rsync over ssh.
        # If some other mechanism is needed to reach this destination,
        # it can be specified here in full [[package.]module.]class form.
        "transfer_class": "transfer.rsync.RsyncOverSsh",

        # Specifies the destination-root-relative directories.
        # Here our desination is rooted at:
        # {some-yet-to-be-specified-destination-root} + "base_dir".
        # In other words, each destination will have some kind of root
        # for all relative file placement.  We'll see those defined
        # later, as they can change per destination machine.
        # The block below describes the settings relative to that
        # destination root.
        #
        # As before, each dir-id used in this configuration is
        # expected to have either:
        # 1. an entry named {dir-id}_dir (e.g. llvm_dir), which
        #    specifies the destination directory for the given dir id
        #    relative to the dest_root+base_dir entries, OR
        # 2. no entry, in which case the directory is assumed to
        #    be the same as {dir-id}.  In the example below, the
        #    dest_root+base_dir-relative directory for the "llvm" dir-id is
        #    defaulted to being "llvm".  That's exactly what
        #    we need in a canonical llvm/clang/lldb setup on
        #    Linux/FreeBSD.
        #
        #    Note we see the untangling of the OS X lldb-centric
        #    directory structure to the canonical llvm,
        #    llvm/tools/clang, llvm/tools/lldb structure below.
        #    We are mapping lldb into a subdirectory of the llvm
        #    directory.
        #
        #    The transfer logic figures out which directories to copy
        #    first by finding the shortest destination absolute path
        #    and doing them in that order.  For our case, this ensures
        #    llvm is copied over before lldb or clang.
        "dest": {
            "base_dir": "work/mirror/git",
            "lldb_dir": "llvm/tools/lldb",
            "clang_dir": "llvm/tools/clang"
        }
    },

    # Describe the lldb-linux destination.  With this,
    # we're done with the mapping for transfer setup
    # for the lldb-linux box.  This configuration can
    # be used either by:
    # 1. having a parent "default" blockthat points to this one,
    #    which then gets used by default, or
    # 2. using the --configuration/-c CONFIG option to
    #    specify using this name on the syncsource.py command line.
    {
        "name": "lldb-linux"
        "parent": "common_tot",

        # The ssh block is understood by the rsync+ssh transfer
        # agent.  Other agents would probably require different
        # agent-specific details that they could read from
        # other blocks.
        "ssh": {
            # This specifies the host name (or IP address) as would
            # be used as the target for an ssh command.
            "dest_host": "lldb-linux.example.com",

            # root_dir specifies the global root directory for
            # this destination.  All destinations on this target
            # will be in a directory that is built from
            # root_dir + base_dir + {dir_id}_dir.
            "root_dir" : "/home/tfiala",

            # The ssh user is specified here.
            "user": "tfiala",

            # The ssh port is specified here.
            "port": 22
        }
    },

    # Describe the lldb-freebsd destination.
    # Very similar to the lldb-linux one.
    {
        "name": "lldb-freebsd"
        "parent": "common_tot",
        "ssh": {
            "dest_host": "lldb-freebsd.example.com",
            # Specify a different destination-specific root dir here.
            "root_dir" : "/mnt/ssd02/fialato",
            "user": "fialato",
            # The ssh port is specified here.
            "port": 2022
        }
    },

    # If a block named "default" exists, and if no configuration
    # is specified on the command line, then the default block
    # will be used.  Use this block to point to the most common
    # transfer destination you would use.
    {
        "name": "default",
        "parent": "lldb-linux"
    }
]
}

Using it

Now that we have a .syncsourcerc file set up, we can do a transfer.
The .syncsourcerc file will be searched for as follows, using the
first one that is found:

* First check the --rc-file RCFILE option.  If this is specified
  and doesn't exist, it will raise an error and quit.

* Check if the current directory has a .syncsourcerc file.  If so,
  use that.

* Use the .syncsourcerc file from the user's home directory.

Run the command:
python /path/to/syncsource.rc -c {configuration-name}

The -c {configuration-name} can be left off, in which case a
configuration with the name 'default' will be used.

After that, the transfer will occur.  With the rsync-over-ssh
transfer agent, one rsync per dir-id will be used.  rsync output
is redirected to the console.

FEEDBACK

Feel free to pass feedback along to Todd Fiala (todd.fiala@gmail.com).
