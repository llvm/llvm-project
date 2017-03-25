====================================
Getting Started with the LLVM System
====================================

.. contents::
   :local:

Overview
========

Welcome to LLVM! In order to get started, you first need to know some basic
information.

First, LLVM comes in three pieces. The first piece is the LLVM suite. This
contains all of the tools, libraries, and header files needed to use LLVM.  It
contains an assembler, disassembler, bitcode analyzer and bitcode optimizer.  It
also contains basic regression tests that can be used to test the LLVM tools and
the Clang front end.

The second piece is the `Clang <http://clang.llvm.org/>`_ front end.  This
component compiles C, C++, Objective C, and Objective C++ code into LLVM
bitcode. Once compiled into LLVM bitcode, a program can be manipulated with the
LLVM tools from the LLVM suite.

There is a third, optional piece called Test Suite.  It is a suite of programs
with a testing harness that can be used to further test LLVM's functionality
and performance.

Getting Started Quickly (A Summary)
===================================

The LLVM Getting Started documentation may be out of date.  So, the `Clang
Getting Started <http://clang.llvm.org/get_started.html>`_ page might also be a
good place to start.

Here's the short story for getting up and running quickly with LLVM:

#. Read the documentation.
#. Read the documentation.
#. Remember that you were warned twice about reading the documentation.

   * In particular, the *relative paths specified are important*.

#. Checkout LLVM:

   * ``cd where-you-want-llvm-to-live``
   * ``svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm``

#. Checkout Clang:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/tools``
   * ``svn co http://llvm.org/svn/llvm-project/cfe/trunk clang``

#. Checkout LLD linker **[Optional]**:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/tools``
   * ``svn co http://llvm.org/svn/llvm-project/lld/trunk lld``

#. Checkout Polly Loop Optimizer **[Optional]**:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/tools``
   * ``svn co http://llvm.org/svn/llvm-project/polly/trunk polly``

#. Checkout Compiler-RT (required to build the sanitizers) **[Optional]**:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/projects``
   * ``svn co http://llvm.org/svn/llvm-project/compiler-rt/trunk compiler-rt``

#. Checkout Libomp (required for OpenMP support) **[Optional]**:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/projects``
   * ``svn co http://llvm.org/svn/llvm-project/openmp/trunk openmp``

#. Checkout libcxx and libcxxabi **[Optional]**:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/projects``
   * ``svn co http://llvm.org/svn/llvm-project/libcxx/trunk libcxx``
   * ``svn co http://llvm.org/svn/llvm-project/libcxxabi/trunk libcxxabi``

#. Get the Test Suite Source Code **[Optional]**

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/projects``
   * ``svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite``

#. Configure and build LLVM and Clang:

   *Warning:* Make sure you've checked out *all of* the source code 
   before trying to configure with cmake.  cmake does not pickup newly
   added source directories in incremental builds. 

   The build uses `CMake <CMake.html>`_. LLVM requires CMake 3.4.3 to build. It
   is generally recommended to use a recent CMake, especially if you're
   generating Ninja build files. This is because the CMake project is constantly
   improving the quality of the generators, and the Ninja generator gets a lot
   of attention.

   * ``cd where you want to build llvm``
   * ``mkdir build``
   * ``cd build``
   * ``cmake -G <generator> [options] <path to llvm sources>``

     Some common generators are:

     * ``Unix Makefiles`` --- for generating make-compatible parallel makefiles.
     * ``Ninja`` --- for generating `Ninja <https://ninja-build.org>`_
       build files. Most llvm developers use Ninja.
     * ``Visual Studio`` --- for generating Visual Studio projects and
       solutions.
     * ``Xcode`` --- for generating Xcode projects.

     Some Common options:

     * ``-DCMAKE_INSTALL_PREFIX=directory`` --- Specify for *directory* the full
       pathname of where you want the LLVM tools and libraries to be installed
       (default ``/usr/local``).

     * ``-DCMAKE_BUILD_TYPE=type`` --- Valid options for *type* are Debug,
       Release, RelWithDebInfo, and MinSizeRel. Default is Debug.

     * ``-DLLVM_ENABLE_ASSERTIONS=On`` --- Compile with assertion checks enabled
       (default is Yes for Debug builds, No for all other build types).

   * Run your build tool of choice!

     * The default target (i.e. ``make``) will build all of LLVM

     * The ``check-all`` target (i.e. ``make check-all``) will run the
       regression tests to ensure everything is in working order.

     * CMake will generate build targets for each tool and library, and most
       LLVM sub-projects generate their own ``check-<project>`` target.

     * Running a serial build will be *slow*.  Make sure you run a 
       parallel build; for ``make``, use ``make -j``.  

   * For more information see `CMake <CMake.html>`_

   * If you get an "internal compiler error (ICE)" or test failures, see
     `below`_.

Consult the `Getting Started with LLVM`_ section for detailed information on
configuring and compiling LLVM.  Go to `Directory Layout`_ to learn about the 
layout of the source code tree.

Requirements
============

Before you begin to use the LLVM system, review the requirements given below.
This may save you some trouble by knowing ahead of time what hardware and
software you will need.

Hardware
--------

LLVM is known to work on the following host platforms:

================== ===================== =============
OS                 Arch                  Compilers
================== ===================== =============
Linux              x86\ :sup:`1`         GCC, Clang
Linux              amd64                 GCC, Clang
Linux              ARM\ :sup:`4`         GCC, Clang
Linux              PowerPC               GCC, Clang
Solaris            V9 (Ultrasparc)       GCC
FreeBSD            x86\ :sup:`1`         GCC, Clang
FreeBSD            amd64                 GCC, Clang
MacOS X\ :sup:`2`  PowerPC               GCC
MacOS X            x86                   GCC, Clang
Cygwin/Win32       x86\ :sup:`1, 3`      GCC
Windows            x86\ :sup:`1`         Visual Studio
Windows x64        x86-64                Visual Studio
================== ===================== =============

.. note::

  #. Code generation supported for Pentium processors and up
  #. Code generation supported for 32-bit ABI only
  #. To use LLVM modules on Win32-based system, you may configure LLVM
     with ``-DBUILD_SHARED_LIBS=On``.
  #. MCJIT not working well pre-v7, old JIT engine not supported any more.

Note that Debug builds require a lot of time and disk space.  An LLVM-only build
will need about 1-3 GB of space.  A full build of LLVM and Clang will need around
15-20 GB of disk space.  The exact space requirements will vary by system.  (It
is so large because of all the debugging information and the fact that the 
libraries are statically linked into multiple tools).  

If you you are space-constrained, you can build only selected tools or only 
selected targets.  The Release build requires considerably less space.

The LLVM suite *may* compile on other platforms, but it is not guaranteed to do
so.  If compilation is successful, the LLVM utilities should be able to
assemble, disassemble, analyze, and optimize LLVM bitcode.  Code generation
should work as well, although the generated native code may not work on your
platform.

Software
--------

Compiling LLVM requires that you have several software packages installed. The
table below lists those required packages. The Package column is the usual name
for the software package that LLVM depends on. The Version column provides
"known to work" versions of the package. The Notes column describes how LLVM
uses the package and provides other details.

=========================================================== ============ ==========================================
Package                                                     Version      Notes
=========================================================== ============ ==========================================
`GNU Make <http://savannah.gnu.org/projects/make>`_         3.79, 3.79.1 Makefile/build processor
`GCC <http://gcc.gnu.org/>`_                                >=4.8.0      C/C++ compiler\ :sup:`1`
`python <http://www.python.org/>`_                          >=2.7        Automated test suite\ :sup:`2`
`zlib <http://zlib.net>`_                                   >=1.2.3.4    Compression library\ :sup:`3`
=========================================================== ============ ==========================================

.. note::

   #. Only the C and C++ languages are needed so there's no need to build the
      other languages for LLVM's purposes. See `below` for specific version
      info.
   #. Only needed if you want to run the automated test suite in the
      ``llvm/test`` directory.
   #. Optional, adds compression / uncompression capabilities to selected LLVM
      tools.

Additionally, your compilation host is expected to have the usual plethora of
Unix utilities. Specifically:

* **ar** --- archive library builder
* **bzip2** --- bzip2 command for distribution generation
* **bunzip2** --- bunzip2 command for distribution checking
* **chmod** --- change permissions on a file
* **cat** --- output concatenation utility
* **cp** --- copy files
* **date** --- print the current date/time
* **echo** --- print to standard output
* **egrep** --- extended regular expression search utility
* **find** --- find files/dirs in a file system
* **grep** --- regular expression search utility
* **gzip** --- gzip command for distribution generation
* **gunzip** --- gunzip command for distribution checking
* **install** --- install directories/files
* **mkdir** --- create a directory
* **mv** --- move (rename) files
* **ranlib** --- symbol table builder for archive libraries
* **rm** --- remove (delete) files and directories
* **sed** --- stream editor for transforming output
* **sh** --- Bourne shell for make build scripts
* **tar** --- tape archive for distribution generation
* **test** --- test things in file system
* **unzip** --- unzip command for distribution checking
* **zip** --- zip command for distribution generation

.. _below:
.. _check here:

Host C++ Toolchain, both Compiler and Standard Library
------------------------------------------------------

LLVM is very demanding of the host C++ compiler, and as such tends to expose
bugs in the compiler. We are also planning to follow improvements and
developments in the C++ language and library reasonably closely. As such, we
require a modern host C++ toolchain, both compiler and standard library, in
order to build LLVM.

For the most popular host toolchains we check for specific minimum versions in
our build systems:

* Clang 3.1
* GCC 4.8
* Visual Studio 2015 (Update 3)

Anything older than these toolchains *may* work, but will require forcing the
build system with a special option and is not really a supported host platform.
Also note that older versions of these compilers have often crashed or
miscompiled LLVM.

For less widely used host toolchains such as ICC or xlC, be aware that a very
recent version may be required to support all of the C++ features used in LLVM.

We track certain versions of software that are *known* to fail when used as
part of the host toolchain. These even include linkers at times.

**GNU ld 2.16.X**. Some 2.16.X versions of the ld linker will produce very long
warning messages complaining that some "``.gnu.linkonce.t.*``" symbol was
defined in a discarded section. You can safely ignore these messages as they are
erroneous and the linkage is correct.  These messages disappear using ld 2.17.

**GNU binutils 2.17**: Binutils 2.17 contains `a bug
<http://sourceware.org/bugzilla/show_bug.cgi?id=3111>`__ which causes huge link
times (minutes instead of seconds) when building LLVM.  We recommend upgrading
to a newer version (2.17.50.0.4 or later).

**GNU Binutils 2.19.1 Gold**: This version of Gold contained `a bug
<http://sourceware.org/bugzilla/show_bug.cgi?id=9836>`__ which causes
intermittent failures when building LLVM with position independent code.  The
symptom is an error about cyclic dependencies.  We recommend upgrading to a
newer version of Gold.

Getting a Modern Host C++ Toolchain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section mostly applies to Linux and older BSDs. On Mac OS X, you should
have a sufficiently modern Xcode, or you will likely need to upgrade until you
do. Windows does not have a "system compiler", so you must install either Visual
Studio 2015 or a recent version of mingw64. FreeBSD 10.0 and newer have a modern
Clang as the system compiler.

However, some Linux distributions and some other or older BSDs sometimes have
extremely old versions of GCC. These steps attempt to help you upgrade you
compiler even on such a system. However, if at all possible, we encourage you
to use a recent version of a distribution with a modern system compiler that
meets these requirements. Note that it is tempting to to install a prior
version of Clang and libc++ to be the host compiler, however libc++ was not
well tested or set up to build on Linux until relatively recently. As
a consequence, this guide suggests just using libstdc++ and a modern GCC as the
initial host in a bootstrap, and then using Clang (and potentially libc++).

The first step is to get a recent GCC toolchain installed. The most common
distribution on which users have struggled with the version requirements is
Ubuntu Precise, 12.04 LTS. For this distribution, one easy option is to install
the `toolchain testing PPA`_ and use it to install a modern GCC. There is
a really nice discussions of this on the `ask ubuntu stack exchange`_. However,
not all users can use PPAs and there are many other distributions, so it may be
necessary (or just useful, if you're here you *are* doing compiler development
after all) to build and install GCC from source. It is also quite easy to do
these days.

.. _toolchain testing PPA:
  https://launchpad.net/~ubuntu-toolchain-r/+archive/test
.. _ask ubuntu stack exchange:
  http://askubuntu.com/questions/271388/how-to-install-gcc-4-8-in-ubuntu-12-04-from-the-terminal

Easy steps for installing GCC 4.8.2:

.. code-block:: console

  % wget https://ftp.gnu.org/gnu/gcc/gcc-4.8.2/gcc-4.8.2.tar.bz2
  % wget https://ftp.gnu.org/gnu/gcc/gcc-4.8.2/gcc-4.8.2.tar.bz2.sig
  % wget https://ftp.gnu.org/gnu/gnu-keyring.gpg
  % signature_invalid=`gpg --verify --no-default-keyring --keyring ./gnu-keyring.gpg gcc-4.8.2.tar.bz2.sig`
  % if [ $signature_invalid ]; then echo "Invalid signature" ; exit 1 ; fi
  % tar -xvjf gcc-4.8.2.tar.bz2
  % cd gcc-4.8.2
  % ./contrib/download_prerequisites
  % cd ..
  % mkdir gcc-4.8.2-build
  % cd gcc-4.8.2-build
  % $PWD/../gcc-4.8.2/configure --prefix=$HOME/toolchains --enable-languages=c,c++
  % make -j$(nproc)
  % make install

For more details, check out the excellent `GCC wiki entry`_, where I got most
of this information from.

.. _GCC wiki entry:
  http://gcc.gnu.org/wiki/InstallingGCC

Once you have a GCC toolchain, configure your build of LLVM to use the new
toolchain for your host compiler and C++ standard library. Because the new
version of libstdc++ is not on the system library search path, you need to pass
extra linker flags so that it can be found at link time (``-L``) and at runtime
(``-rpath``). If you are using CMake, this invocation should produce working
binaries:

.. code-block:: console

  % mkdir build
  % cd build
  % CC=$HOME/toolchains/bin/gcc CXX=$HOME/toolchains/bin/g++ \
    cmake .. -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$HOME/toolchains/lib64 -L$HOME/toolchains/lib64"

If you fail to set rpath, most LLVM binaries will fail on startup with a message
from the loader similar to ``libstdc++.so.6: version `GLIBCXX_3.4.20' not
found``. This means you need to tweak the -rpath linker flag.

When you build Clang, you will need to give *it* access to modern C++11
standard library in order to use it as your new host in part of a bootstrap.
There are two easy ways to do this, either build (and install) libc++ along
with Clang and then use it with the ``-stdlib=libc++`` compile and link flag,
or install Clang into the same prefix (``$HOME/toolchains`` above) as GCC.
Clang will look within its own prefix for libstdc++ and use it if found. You
can also add an explicit prefix for Clang to look in for a GCC toolchain with
the ``--gcc-toolchain=/opt/my/gcc/prefix`` flag, passing it to both compile and
link commands when using your just-built-Clang to bootstrap.

.. _Getting Started with LLVM:

Getting Started with LLVM
=========================

The remainder of this guide is meant to get you up and running with LLVM and to
give you some basic information about the LLVM environment.

The later sections of this guide describe the `general layout`_ of the LLVM
source tree, a `simple example`_ using the LLVM tool chain, and `links`_ to find
more information about LLVM or to get help via e-mail.

Terminology and Notation
------------------------

Throughout this manual, the following names are used to denote paths specific to
the local system and working environment.  *These are not environment variables
you need to set but just strings used in the rest of this document below*.  In
any of the examples below, simply replace each of these names with the
appropriate pathname on your local system.  All these paths are absolute:

``SRC_ROOT``

  This is the top level directory of the LLVM source tree.

``OBJ_ROOT``

  This is the top level directory of the LLVM object tree (i.e. the tree where
  object files and compiled programs will be placed.  It can be the same as
  SRC_ROOT).

Unpacking the LLVM Archives
---------------------------

If you have the LLVM distribution, you will need to unpack it before you can
begin to compile it.  LLVM is distributed as a set of two files: the LLVM suite
and the LLVM GCC front end compiled for your platform.  There is an additional
test suite that is optional.  Each file is a TAR archive that is compressed with
the gzip program.

The files are as follows, with *x.y* marking the version number:

``llvm-x.y.tar.gz``

  Source release for the LLVM libraries and tools.

``llvm-test-x.y.tar.gz``

  Source release for the LLVM test-suite.

.. _checkout:

Checkout LLVM from Subversion
-----------------------------

If you have access to our Subversion repository, you can get a fresh copy of the
entire source code.  All you need to do is check it out from Subversion as
follows:

* ``cd where-you-want-llvm-to-live``
* Read-Only: ``svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm``
* Read-Write: ``svn co https://user@llvm.org/svn/llvm-project/llvm/trunk llvm``

This will create an '``llvm``' directory in the current directory and fully
populate it with the LLVM source code, Makefiles, test directories, and local
copies of documentation files.

If you want to get a specific release (as opposed to the most recent revision),
you can checkout it from the '``tags``' directory (instead of '``trunk``'). The
following releases are located in the following subdirectories of the '``tags``'
directory:

* Release 3.4: **RELEASE_34/final**
* Release 3.3: **RELEASE_33/final**
* Release 3.2: **RELEASE_32/final**
* Release 3.1: **RELEASE_31/final**
* Release 3.0: **RELEASE_30/final**
* Release 2.9: **RELEASE_29/final**
* Release 2.8: **RELEASE_28**
* Release 2.7: **RELEASE_27**
* Release 2.6: **RELEASE_26**
* Release 2.5: **RELEASE_25**
* Release 2.4: **RELEASE_24**
* Release 2.3: **RELEASE_23**
* Release 2.2: **RELEASE_22**
* Release 2.1: **RELEASE_21**
* Release 2.0: **RELEASE_20**
* Release 1.9: **RELEASE_19**
* Release 1.8: **RELEASE_18**
* Release 1.7: **RELEASE_17**
* Release 1.6: **RELEASE_16**
* Release 1.5: **RELEASE_15**
* Release 1.4: **RELEASE_14**
* Release 1.3: **RELEASE_13**
* Release 1.2: **RELEASE_12**
* Release 1.1: **RELEASE_11**
* Release 1.0: **RELEASE_1**

If you would like to get the LLVM test suite (a separate package as of 1.4), you
get it from the Subversion repository:

.. code-block:: console

  % cd llvm/projects
  % svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite

By placing it in the ``llvm/projects``, it will be automatically configured by
the LLVM cmake configuration.

Git Mirror
----------

Git mirrors are available for a number of LLVM subprojects. These mirrors sync
automatically with each Subversion commit and contain all necessary git-svn
marks (so, you can recreate git-svn metadata locally). Note that right now
mirrors reflect only ``trunk`` for each project. You can do the read-only Git
clone of LLVM via:

.. code-block:: console

  % git clone http://llvm.org/git/llvm.git

If you want to check out clang too, run:

.. code-block:: console

  % cd llvm/tools
  % git clone http://llvm.org/git/clang.git

If you want to check out compiler-rt (required to build the sanitizers), run:

.. code-block:: console

  % cd llvm/projects
  % git clone http://llvm.org/git/compiler-rt.git

If you want to check out libomp (required for OpenMP support), run:

.. code-block:: console

  % cd llvm/projects
  % git clone http://llvm.org/git/openmp.git

If you want to check out libcxx and libcxxabi (optional), run:

.. code-block:: console

  % cd llvm/projects
  % git clone http://llvm.org/git/libcxx.git
  % git clone http://llvm.org/git/libcxxabi.git

If you want to check out the Test Suite Source Code (optional), run:

.. code-block:: console

  % cd llvm/projects
  % git clone http://llvm.org/git/test-suite.git

Since the upstream repository is in Subversion, you should use ``git
pull --rebase`` instead of ``git pull`` to avoid generating a non-linear history
in your clone.  To configure ``git pull`` to pass ``--rebase`` by default on the
master branch, run the following command:

.. code-block:: console

  % git config branch.master.rebase true

Sending patches with Git
^^^^^^^^^^^^^^^^^^^^^^^^

Please read `Developer Policy <DeveloperPolicy.html#one-off-patches>`_, too.

Assume ``master`` points the upstream and ``mybranch`` points your working
branch, and ``mybranch`` is rebased onto ``master``.  At first you may check
sanity of whitespaces:

.. code-block:: console

  % git diff --check master..mybranch

The easiest way to generate a patch is as below:

.. code-block:: console

  % git diff master..mybranch > /path/to/mybranch.diff

It is a little different from svn-generated diff. git-diff-generated diff has
prefixes like ``a/`` and ``b/``. Don't worry, most developers might know it
could be accepted with ``patch -p1 -N``.

But you may generate patchset with git-format-patch. It generates by-each-commit
patchset. To generate patch files to attach to your article:

.. code-block:: console

  % git format-patch --no-attach master..mybranch -o /path/to/your/patchset

If you would like to send patches directly, you may use git-send-email or
git-imap-send. Here is an example to generate the patchset in Gmail's [Drafts].

.. code-block:: console

  % git format-patch --attach master..mybranch --stdout | git imap-send

Then, your .git/config should have [imap] sections.

.. code-block:: ini

  [imap]
        host = imaps://imap.gmail.com
        user = your.gmail.account@gmail.com
        pass = himitsu!
        port = 993
        sslverify = false
  ; in English
        folder = "[Gmail]/Drafts"
  ; example for Japanese, "Modified UTF-7" encoded.
        folder = "[Gmail]/&Tgtm+DBN-"
  ; example for Traditional Chinese
        folder = "[Gmail]/&g0l6Pw-"

.. _developers-work-with-git-svn:

For developers to work with git-svn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up clone from which you can submit code using ``git-svn``, run:

.. code-block:: console

  % git clone http://llvm.org/git/llvm.git
  % cd llvm
  % git svn init https://llvm.org/svn/llvm-project/llvm/trunk --username=<username>
  % git config svn-remote.svn.fetch :refs/remotes/origin/master
  % git svn rebase -l  # -l avoids fetching ahead of the git mirror.

  # If you have clang too:
  % cd tools
  % git clone http://llvm.org/git/clang.git
  % cd clang
  % git svn init https://llvm.org/svn/llvm-project/cfe/trunk --username=<username>
  % git config svn-remote.svn.fetch :refs/remotes/origin/master
  % git svn rebase -l

Likewise for compiler-rt, libomp and test-suite.

To update this clone without generating git-svn tags that conflict with the
upstream Git repo, run:

.. code-block:: console

  % git fetch && (cd tools/clang && git fetch)  # Get matching revisions of both trees.
  % git checkout master
  % git svn rebase -l
  % (cd tools/clang &&
     git checkout master &&
     git svn rebase -l)

Likewise for compiler-rt, libomp and test-suite.

This leaves your working directories on their master branches, so you'll need to
``checkout`` each working branch individually and ``rebase`` it on top of its
parent branch.

For those who wish to be able to update an llvm repo/revert patches easily using
git-svn, please look in the directory for the scripts ``git-svnup`` and
``git-svnrevert``.

To perform the aforementioned update steps go into your source directory and
just type ``git-svnup`` or ``git svnup`` and everything will just work.

If one wishes to revert a commit with git-svn, but do not want the git hash to
escape into the commit message, one can use the script ``git-svnrevert`` or
``git svnrevert`` which will take in the git hash for the commit you want to
revert, look up the appropriate svn revision, and output a message where all
references to the git hash have been replaced with the svn revision.

To commit back changes via git-svn, use ``git svn dcommit``:

.. code-block:: console

  % git svn dcommit

Note that git-svn will create one SVN commit for each Git commit you have pending,
so squash and edit each commit before executing ``dcommit`` to make sure they all
conform to the coding standards and the developers' policy.

On success, ``dcommit`` will rebase against the HEAD of SVN, so to avoid conflict,
please make sure your current branch is up-to-date (via fetch/rebase) before
proceeding.

The git-svn metadata can get out of sync after you mess around with branches and
``dcommit``. When that happens, ``git svn dcommit`` stops working, complaining
about files with uncommitted changes. The fix is to rebuild the metadata:

.. code-block:: console

  % rm -rf .git/svn
  % git svn rebase -l

Please, refer to the Git-SVN manual (``man git-svn``) for more information.

For developers to work with a git monorepo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   This set-up is using unofficial mirror hosted on GitHub, use with caution.

To set up a clone of all the llvm projects using a unified repository:

.. code-block:: console

  % export TOP_LEVEL_DIR=`pwd`
  % git clone https://github.com/llvm-project/llvm-project/
  % cd llvm-project
  % git config branch.master.rebase true

You can configure various build directory from this clone, starting with a build
of LLVM alone:

.. code-block:: console

  % cd $TOP_LEVEL_DIR
  % mkdir llvm-build && cd llvm-build
  % cmake -GNinja ../llvm-project/llvm

Or lldb:

.. code-block:: console

  % cd $TOP_LEVEL_DIR
  % mkdir lldb-build && cd lldb-build
  % cmake -GNinja ../llvm-project/llvm -DLLVM_ENABLE_PROJECTS=lldb

Or a combination of multiple projects:

.. code-block:: console

  % cd $TOP_LEVEL_DIR
  % mkdir clang-build && cd clang-build
  % cmake -GNinja ../llvm-project/llvm -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi"

A helper script is provided in ``llvm/utils/git-svn/git-llvm``. After you add it
to your path, you can push committed changes upstream with ``git llvm push``.

.. code-block:: console

  % export PATH=$PATH:$TOP_LEVEL_DIR/llvm-project/llvm/utils/git-svn/
  % git llvm push

While this is using SVN under the hood, it does not require any interaction from
you with git-svn.
After a few minutes, ``git pull`` should get back the changes as they were
committed. Note that a current limitation is that ``git`` does not directly
record file rename, and thus it is propagated to SVN as a combination of
delete-add instead of a file rename.

The SVN revision of each monorepo commit can be found in the commit notes.  git
does not fetch notes by default. The following commands will fetch the notes and
configure git to fetch future notes. Use ``git notes show $commit`` to look up
the SVN revision of a git commit. The notes show up ``git log``, and searching
the log is currently the recommended way to look up the git commit for a given
SVN revision.

.. code-block:: console

  % git config --add remote.origin.fetch +refs/notes/commits:refs/notes/commits
  % git fetch

If you are using `arc` to interact with Phabricator, you need to manually put it
at the root of the checkout:

.. code-block:: console

  % cd $TOP_LEVEL_DIR
  % cp llvm/.arcconfig ./
  % mkdir -p .git/info/
  % echo .arcconfig >> .git/info/exclude


Local LLVM Configuration
------------------------

Once checked out from the Subversion repository, the LLVM suite source code must
be configured before being built. This process uses CMake.
Unlinke the normal ``configure`` script, CMake
generates the build files in whatever format you request as well as various
``*.inc`` files, and ``llvm/include/Config/config.h``.

Variables are passed to ``cmake`` on the command line using the format
``-D<variable name>=<value>``. The following variables are some common options
used by people developing LLVM.

+-------------------------+----------------------------------------------------+
| Variable                | Purpose                                            |
+=========================+====================================================+
| CMAKE_C_COMPILER        | Tells ``cmake`` which C compiler to use. By        |
|                         | default, this will be /usr/bin/cc.                 |
+-------------------------+----------------------------------------------------+
| CMAKE_CXX_COMPILER      | Tells ``cmake`` which C++ compiler to use. By      |
|                         | default, this will be /usr/bin/c++.                |
+-------------------------+----------------------------------------------------+
| CMAKE_BUILD_TYPE        | Tells ``cmake`` what type of build you are trying  |
|                         | to generate files for. Valid options are Debug,    |
|                         | Release, RelWithDebInfo, and MinSizeRel. Default   |
|                         | is Debug.                                          |
+-------------------------+----------------------------------------------------+
| CMAKE_INSTALL_PREFIX    | Specifies the install directory to target when     |
|                         | running the install action of the build files.     |
+-------------------------+----------------------------------------------------+
| LLVM_TARGETS_TO_BUILD   | A semicolon delimited list controlling which       |
|                         | targets will be built and linked into llc. This is |
|                         | equivalent to the ``--enable-targets`` option in   |
|                         | the configure script. The default list is defined  |
|                         | as ``LLVM_ALL_TARGETS``, and can be set to include |
|                         | out-of-tree targets. The default value includes:   |
|                         | ``AArch64, AMDGPU, ARM, BPF, Hexagon, Mips,        |
|                         | MSP430, NVPTX, PowerPC, Sparc, SystemZ, X86,       |
|                         | XCore``.                                           |
+-------------------------+----------------------------------------------------+
| LLVM_ENABLE_DOXYGEN     | Build doxygen-based documentation from the source  |
|                         | code This is disabled by default because it is     |
|                         | slow and generates a lot of output.                |
+-------------------------+----------------------------------------------------+
| LLVM_ENABLE_SPHINX      | Build sphinx-based documentation from the source   |
|                         | code. This is disabled by default because it is    |
|                         | slow and generates a lot of output. Sphinx version |
|                         | 1.5 or later recommended.                          |
+-------------------------+----------------------------------------------------+
| LLVM_BUILD_LLVM_DYLIB   | Generate libLLVM.so. This library contains a       |
|                         | default set of LLVM components that can be         |
|                         | overridden with ``LLVM_DYLIB_COMPONENTS``. The     |
|                         | default contains most of LLVM and is defined in    |
|                         | ``tools/llvm-shlib/CMakelists.txt``.               |
+-------------------------+----------------------------------------------------+
| LLVM_OPTIMIZED_TABLEGEN | Builds a release tablegen that gets used during    |
|                         | the LLVM build. This can dramatically speed up     |
|                         | debug builds.                                      |
+-------------------------+----------------------------------------------------+

To configure LLVM, follow these steps:

#. Change directory into the object root directory:

   .. code-block:: console

     % cd OBJ_ROOT

#. Run the ``cmake``:

   .. code-block:: console

     % cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=prefix=/install/path
       [other options] SRC_ROOT

Compiling the LLVM Suite Source Code
------------------------------------

Unlike with autotools, with CMake your build type is defined at configuration.
If you want to change your build type, you can re-run cmake with the following
invocation:

   .. code-block:: console

     % cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=type SRC_ROOT

Between runs, CMake preserves the values set for all options. CMake has the
following build types defined:

Debug

  These builds are the default. The build system will compile the tools and
  libraries unoptimized, with debugging information, and asserts enabled.

Release

  For these builds, the build system will compile the tools and libraries
  with optimizations enabled and not generate debug info. CMakes default
  optimization level is -O3. This can be configured by setting the
  ``CMAKE_CXX_FLAGS_RELEASE`` variable on the CMake command line.

RelWithDebInfo

  These builds are useful when debugging. They generate optimized binaries with
  debug information. CMakes default optimization level is -O2. This can be
  configured by setting the ``CMAKE_CXX_FLAGS_RELWITHDEBINFO`` variable on the
  CMake command line.

Once you have LLVM configured, you can build it by entering the *OBJ_ROOT*
directory and issuing the following command:

.. code-block:: console

  % make

If the build fails, please `check here`_ to see if you are using a version of
GCC that is known not to compile LLVM.

If you have multiple processors in your machine, you may wish to use some of the
parallel build options provided by GNU Make.  For example, you could use the
command:

.. code-block:: console

  % make -j2

There are several special targets which are useful when working with the LLVM
source code:

``make clean``

  Removes all files generated by the build.  This includes object files,
  generated C/C++ files, libraries, and executables.

``make install``

  Installs LLVM header files, libraries, tools, and documentation in a hierarchy
  under ``$PREFIX``, specified with ``CMAKE_INSTALL_PREFIX``, which
  defaults to ``/usr/local``.

``make docs-llvm-html``

  If configured with ``-DLLVM_ENABLE_SPHINX=On``, this will generate a directory
  at ``OBJ_ROOT/docs/html`` which contains the HTML formatted documentation.

Cross-Compiling LLVM
--------------------

It is possible to cross-compile LLVM itself. That is, you can create LLVM
executables and libraries to be hosted on a platform different from the platform
where they are built (a Canadian Cross build). To generate build files for
cross-compiling CMake provides a variable ``CMAKE_TOOLCHAIN_FILE`` which can
define compiler flags and variables used during the CMake test operations.

The result of such a build is executables that are not runnable on on the build
host but can be executed on the target. As an example the following CMake
invocation can generate build files targeting iOS. This will work on Mac OS X
with the latest Xcode:

.. code-block:: console

  % cmake -G "Ninja" -DCMAKE_OSX_ARCHITECTURES="armv7;armv7s;arm64"
    -DCMAKE_TOOLCHAIN_FILE=<PATH_TO_LLVM>/cmake/platforms/iOS.cmake
    -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_RUNTIME=Off -DLLVM_INCLUDE_TESTS=Off
    -DLLVM_INCLUDE_EXAMPLES=Off -DLLVM_ENABLE_BACKTRACES=Off [options]
    <PATH_TO_LLVM>

Note: There are some additional flags that need to be passed when building for
iOS due to limitations in the iOS SDK.

Check :doc:`HowToCrossCompileLLVM` and `Clang docs on how to cross-compile in general
<http://clang.llvm.org/docs/CrossCompilation.html>`_ for more information
about cross-compiling.

The Location of LLVM Object Files
---------------------------------

The LLVM build system is capable of sharing a single LLVM source tree among
several LLVM builds.  Hence, it is possible to build LLVM for several different
platforms or configurations using the same source tree.

* Change directory to where the LLVM object files should live:

  .. code-block:: console

    % cd OBJ_ROOT

* Run ``cmake``:

  .. code-block:: console

    % cmake -G "Unix Makefiles" SRC_ROOT

The LLVM build will create a structure underneath *OBJ_ROOT* that matches the
LLVM source tree. At each level where source files are present in the source
tree there will be a corresponding ``CMakeFiles`` directory in the *OBJ_ROOT*.
Underneath that directory there is another directory with a name ending in
``.dir`` under which you'll find object files for each source.

For example:

  .. code-block:: console

    % cd llvm_build_dir
    % find lib/Support/ -name APFloat*
    lib/Support/CMakeFiles/LLVMSupport.dir/APFloat.cpp.o

Optional Configuration Items
----------------------------

If you're running on a Linux system that supports the `binfmt_misc
<http://en.wikipedia.org/wiki/binfmt_misc>`_
module, and you have root access on the system, you can set your system up to
execute LLVM bitcode files directly. To do this, use commands like this (the
first command may not be required if you are already using the module):

.. code-block:: console

  % mount -t binfmt_misc none /proc/sys/fs/binfmt_misc
  % echo ':llvm:M::BC::/path/to/lli:' > /proc/sys/fs/binfmt_misc/register
  % chmod u+x hello.bc   (if needed)
  % ./hello.bc

This allows you to execute LLVM bitcode files directly.  On Debian, you can also
use this command instead of the 'echo' command above:

.. code-block:: console

  % sudo update-binfmts --install llvm /path/to/lli --magic 'BC'

.. _Program Layout:
.. _general layout:

Directory Layout
================

One useful source of information about the LLVM source base is the LLVM `doxygen
<http://www.doxygen.org/>`_ documentation available at 
`<http://llvm.org/doxygen/>`_.  The following is a brief introduction to code
layout:

``llvm/examples``
-----------------

Simple examples using the LLVM IR and JIT.

``llvm/include``
----------------

Public header files exported from the LLVM library. The three main subdirectories:

``llvm/include/llvm``

  All LLVM-specific header files, and  subdirectories for different portions of 
  LLVM: ``Analysis``, ``CodeGen``, ``Target``, ``Transforms``, etc...

``llvm/include/llvm/Support``

  Generic support libraries provided with LLVM but not necessarily specific to 
  LLVM. For example, some C++ STL utilities and a Command Line option processing 
  library store header files here.

``llvm/include/llvm/Config``

  Header files configured by the ``configure`` script.
  They wrap "standard" UNIX and C header files.  Source code can include these
  header files which automatically take care of the conditional #includes that
  the ``configure`` script generates.

``llvm/lib``
------------

Most source files are here. By putting code in libraries, LLVM makes it easy to 
share code among the `tools`_.

``llvm/lib/IR/``

  Core LLVM source files that implement core classes like Instruction and 
  BasicBlock.

``llvm/lib/AsmParser/``

  Source code for the LLVM assembly language parser library.

``llvm/lib/Bitcode/``

  Code for reading and writing bitcode.

``llvm/lib/Analysis/``

  A variety of program analyses, such as Call Graphs, Induction Variables, 
  Natural Loop Identification, etc.

``llvm/lib/Transforms/``

  IR-to-IR program transformations, such as Aggressive Dead Code Elimination, 
  Sparse Conditional Constant Propagation, Inlining, Loop Invariant Code Motion, 
  Dead Global Elimination, and many others.

``llvm/lib/Target/``

  Files describing target architectures for code generation.  For example, 
  ``llvm/lib/Target/X86`` holds the X86 machine description.

``llvm/lib/CodeGen/``

  The major parts of the code generator: Instruction Selector, Instruction 
  Scheduling, and Register Allocation.

``llvm/lib/MC/``

  (FIXME: T.B.D.)  ....?

``llvm/lib/ExecutionEngine/``

  Libraries for directly executing bitcode at runtime in interpreted and 
  JIT-compiled scenarios.

``llvm/lib/Support/``

  Source code that corresponding to the header files in ``llvm/include/ADT/``
  and ``llvm/include/Support/``.

``llvm/projects``
-----------------

Projects not strictly part of LLVM but shipped with LLVM. This is also the 
directory for creating your own LLVM-based projects which leverage the LLVM
build system.

``llvm/test``
-------------

Feature and regression tests and other sanity checks on LLVM infrastructure. These
are intended to run quickly and cover a lot of territory without being exhaustive.

``test-suite``
--------------

A comprehensive correctness, performance, and benchmarking test suite for LLVM. 
Comes in a separate Subversion module because not every LLVM user is interested 
in such a comprehensive suite. For details see the :doc:`Testing Guide
<TestingGuide>` document.

.. _tools:

``llvm/tools``
--------------

Executables built out of the libraries
above, which form the main part of the user interface.  You can always get help
for a tool by typing ``tool_name -help``.  The following is a brief introduction
to the most important tools.  More detailed information is in
the `Command Guide <CommandGuide/index.html>`_.

``bugpoint``

  ``bugpoint`` is used to debug optimization passes or code generation backends
  by narrowing down the given test case to the minimum number of passes and/or
  instructions that still cause a problem, whether it is a crash or
  miscompilation. See `<HowToSubmitABug.html>`_ for more information on using
  ``bugpoint``.

``llvm-ar``

  The archiver produces an archive containing the given LLVM bitcode files,
  optionally with an index for faster lookup.

``llvm-as``

  The assembler transforms the human readable LLVM assembly to LLVM bitcode.

``llvm-dis``

  The disassembler transforms the LLVM bitcode to human readable LLVM assembly.

``llvm-link``

  ``llvm-link``, not surprisingly, links multiple LLVM modules into a single
  program.

``lli``

  ``lli`` is the LLVM interpreter, which can directly execute LLVM bitcode
  (although very slowly...). For architectures that support it (currently x86,
  Sparc, and PowerPC), by default, ``lli`` will function as a Just-In-Time
  compiler (if the functionality was compiled in), and will execute the code
  *much* faster than the interpreter.

``llc``

  ``llc`` is the LLVM backend compiler, which translates LLVM bitcode to a
  native code assembly file or to C code (with the ``-march=c`` option).

``opt``

  ``opt`` reads LLVM bitcode, applies a series of LLVM to LLVM transformations
  (which are specified on the command line), and outputs the resultant
  bitcode.   '``opt -help``'  is a good way to get a list of the
  program transformations available in LLVM.

  ``opt`` can also  run a specific analysis on an input LLVM bitcode
  file and print  the results.  Primarily useful for debugging
  analyses, or familiarizing yourself with what an analysis does.

``llvm/utils``
--------------

Utilities for working with LLVM source code; some are part of the build process
because they are code generators for parts of the infrastructure.


``codegen-diff``

  ``codegen-diff`` finds differences between code that LLC
  generates and code that LLI generates. This is useful if you are
  debugging one of them, assuming that the other generates correct output. For
  the full user manual, run ```perldoc codegen-diff'``.

``emacs/``

   Emacs and XEmacs syntax highlighting  for LLVM   assembly files and TableGen 
   description files.  See the ``README`` for information on using them.

``getsrcs.sh``

  Finds and outputs all non-generated source files,
  useful if one wishes to do a lot of development across directories
  and does not want to find each file. One way to use it is to run,
  for example: ``xemacs `utils/getsources.sh``` from the top of the LLVM source
  tree.

``llvmgrep``

  Performs an ``egrep -H -n`` on each source file in LLVM and
  passes to it a regular expression provided on ``llvmgrep``'s command
  line. This is an efficient way of searching the source base for a
  particular regular expression.

``makellvm``

  Compiles all files in the current directory, then
  compiles and links the tool that is the first argument. For example, assuming
  you are in  ``llvm/lib/Target/Sparc``, if ``makellvm`` is in your
  path,  running ``makellvm llc`` will make a build of the current
  directory, switch to directory ``llvm/tools/llc`` and build it, causing a
  re-linking of LLC.

``TableGen/``

  Contains the tool used to generate register
  descriptions, instruction set descriptions, and even assemblers from common
  TableGen description files.

``vim/``

  vim syntax-highlighting for LLVM assembly files
  and TableGen description files. See the    ``README`` for how to use them.

.. _simple example:

An Example Using the LLVM Tool Chain
====================================

This section gives an example of using LLVM with the Clang front end.

Example with clang
------------------

#. First, create a simple C file, name it 'hello.c':

   .. code-block:: c

     #include <stdio.h>

     int main() {
       printf("hello world\n");
       return 0;
     }

#. Next, compile the C file into a native executable:

   .. code-block:: console

     % clang hello.c -o hello

   .. note::

     Clang works just like GCC by default.  The standard -S and -c arguments
     work as usual (producing a native .s or .o file, respectively).

#. Next, compile the C file into an LLVM bitcode file:

   .. code-block:: console

     % clang -O3 -emit-llvm hello.c -c -o hello.bc

   The -emit-llvm option can be used with the -S or -c options to emit an LLVM
   ``.ll`` or ``.bc`` file (respectively) for the code.  This allows you to use
   the `standard LLVM tools <CommandGuide/index.html>`_ on the bitcode file.

#. Run the program in both forms. To run the program, use:

   .. code-block:: console

      % ./hello

   and

   .. code-block:: console

     % lli hello.bc

   The second examples shows how to invoke the LLVM JIT, :doc:`lli
   <CommandGuide/lli>`.

#. Use the ``llvm-dis`` utility to take a look at the LLVM assembly code:

   .. code-block:: console

     % llvm-dis < hello.bc | less

#. Compile the program to native assembly using the LLC code generator:

   .. code-block:: console

     % llc hello.bc -o hello.s

#. Assemble the native assembly language file into a program:

   .. code-block:: console

     % /opt/SUNWspro/bin/cc -xarch=v9 hello.s -o hello.native   # On Solaris

     % gcc hello.s -o hello.native                              # On others

#. Execute the native code program:

   .. code-block:: console

     % ./hello.native

   Note that using clang to compile directly to native code (i.e. when the
   ``-emit-llvm`` option is not present) does steps 6/7/8 for you.

Common Problems
===============

If you are having problems building or using LLVM, or if you have any other
general questions about LLVM, please consult the `Frequently Asked
Questions <FAQ.html>`_ page.

.. _links:

Links
=====

This document is just an **introduction** on how to use LLVM to do some simple
things... there are many more interesting and complicated things that you can do
that aren't documented here (but we'll gladly accept a patch if you want to
write something up!).  For more information about LLVM, check out:

* `LLVM Homepage <http://llvm.org/>`_
* `LLVM Doxygen Tree <http://llvm.org/doxygen/>`_
* `Starting a Project that Uses LLVM <http://llvm.org/docs/Projects.html>`_
