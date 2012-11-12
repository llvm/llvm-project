.. _getting_started:

====================================
Getting Started with the LLVM System  
====================================

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
#. Checkout LLVM:

   * ``cd where-you-want-llvm-to-live``
   * ``svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm``

#. Checkout Clang:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/tools``
   * ``svn co http://llvm.org/svn/llvm-project/cfe/trunk clang``

#. Checkout Compiler-RT:

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/projects``
   * ``svn co http://llvm.org/svn/llvm-project/compiler-rt/trunk compiler-rt``

#. Get the Test Suite Source Code **[Optional]**

   * ``cd where-you-want-llvm-to-live``
   * ``cd llvm/projects``
   * ``svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite``

#. Configure and build LLVM and Clang:

   * ``cd where-you-want-to-build-llvm``
   * ``mkdir build`` (for building without polluting the source dir)
   * ``cd build``
   * ``../llvm/configure [options]``
     Some common options:

     * ``--prefix=directory`` ---

       Specify for *directory* the full pathname of where you want the LLVM
       tools and libraries to be installed (default ``/usr/local``).

     * ``--enable-optimized`` ---

       Compile with optimizations enabled (default is NO).

     * ``--enable-assertions`` ---

       Compile with assertion checks enabled (default is YES).

   * ``make [-j]`` --- The ``-j`` specifies the number of jobs (commands) to run
     simultaneously.  This builds both LLVM and Clang for Debug+Asserts mode.
     The --enabled-optimized configure option is used to specify a Release
     build.

   * ``make check-all`` --- This run the regression tests to ensure everything
     is in working order.
  
   * ``make update`` --- This command is used to update all the svn repositories
     at once, rather then having to ``cd`` into the individual repositories and
     running ``svn update``.

   * It is also possible to use CMake instead of the makefiles. With CMake it is
     also possible to generate project files for several IDEs: Eclipse CDT4,
     CodeBlocks, Qt-Creator (use the CodeBlocks generator), KDevelop3.

   * If you get an "internal compiler error (ICE)" or test failures, see
     `below`.

Consult the `Getting Started with LLVM`_ section for detailed information on
configuring and compiling LLVM.  See `Setting Up Your Environment`_ for tips
that simplify working with the Clang front end and LLVM tools.  Go to `Program
Layout`_ to learn about the layout of the source code tree.

Requirements
============

Before you begin to use the LLVM system, review the requirements given below.
This may save you some trouble by knowing ahead of time what hardware and
software you will need.

Hardware
--------

LLVM is known to work on the following platforms:

+-----------------+----------------------+-------------------------+
|OS               |  Arch                | Compilers               |
+=================+======================+=========================+
|AuroraUX         | x86\ :sup:`1`        | GCC                     |
+-----------------+----------------------+-------------------------+
|Linux            | x86\ :sup:`1`        | GCC                     |
+-----------------+----------------------+-------------------------+
|Linux            | amd64                | GCC                     |
+-----------------+----------------------+-------------------------+
|Solaris          | V9 (Ultrasparc)      | GCC                     |
+-----------------+----------------------+-------------------------+
|FreeBSD          | x86\ :sup:`1`        | GCC                     |
+-----------------+----------------------+-------------------------+
|FreeBSD          | amd64                | GCC                     |
+-----------------+----------------------+-------------------------+
|MacOS X\ :sup:`2`| PowerPC              | GCC                     |
+-----------------+----------------------+-------------------------+
|MacOS X\ :sup:`9`| x86                  | GCC                     |
+-----------------+----------------------+-------------------------+
|Cygwin/Win32     | x86\ :sup:`1, 8, 11` | GCC 3.4.X, binutils 2.20|
+-----------------+----------------------+-------------------------+

LLVM has partial support for the following platforms:

+-------------------+----------------------+-------------------------------------------+
|OS                 |  Arch                | Compilers                                 |
+===================+======================+===========================================+
| Windows           | x86\ :sup:`1`        | Visual Studio 2000 or higher\ :sup:`4,5`  |
+-------------------+----------------------+-------------------------------------------+
| AIX\ :sup:`3,4`   | PowerPC              | GCC                                       |
+-------------------+----------------------+-------------------------------------------+
| Linux\ :sup:`3,5` | PowerPC              | GCC                                       |
+-------------------+----------------------+-------------------------------------------+
| Linux\ :sup:`7`   | Alpha                | GCC                                       |
+-------------------+----------------------+-------------------------------------------+
| Linux\ :sup:`7`   | Itanium (IA-64)      | GCC                                       |
+-------------------+----------------------+-------------------------------------------+
| HP-UX\ :sup:`7`   | Itanium (IA-64)      | HP aCC                                    |
+-------------------+----------------------+-------------------------------------------+
| Windows x64       | x86-64               | mingw-w64's GCC-4.5.x\ :sup:`12`          |
+-------------------+----------------------+-------------------------------------------+

.. note::

  Code generation supported for Pentium processors and up

  #. Code generation supported for Pentium processors and up
  #. Code generation supported for 32-bit ABI only
  #. No native code generation
  #. Build is not complete: one or more tools do not link or function
  #. The GCC-based C/C++ frontend does not build
  #. The port is done using the MSYS shell.
  #. Native code generation exists but is not complete.
  #. Binutils 2.20 or later is required to build the assembler generated by LLVM properly.
  #. Xcode 2.5 and gcc 4.0.1 (Apple Build 5370) will trip internal LLVM assert
     messages when compiled for Release at optimization levels greater than 0
     (i.e., ``-O1`` and higher).  Add ``OPTIMIZE_OPTION="-O0"`` to the build
     command line if compiling for LLVM Release or bootstrapping the LLVM
     toolchain.
  #. For MSYS/MinGW on Windows, be sure to install the MSYS version of the perl
     package, and be sure it appears in your path before any Windows-based
     versions such as Strawberry Perl and ActivePerl, as these have
     Windows-specifics that will cause the build to fail.
  #. To use LLVM modules on Win32-based system, you may configure LLVM
     with ``--enable-shared``.

  #. To compile SPU backend, you need to add ``LDFLAGS=-Wl,--stack,16777216`` to
     configure.

Note that you will need about 1-3 GB of space for a full LLVM build in Debug
mode, depending on the system (it is so large because of all the debugging
information and the fact that the libraries are statically linked into multiple
tools).  If you do not need many of the tools and you are space-conscious, you
can pass ``ONLY_TOOLS="tools you need"`` to make.  The Release build requires
considerably less space.

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

+--------------------------------------------------------------+-----------------+---------------------------------------------+
| Package                                                      | Version         | Notes                                       |
+==============================================================+=================+=============================================+
| `GNU Make <http://savannah.gnu.org/projects/make>`_          | 3.79, 3.79.1    | Makefile/build processor                    |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `GCC <http://gcc.gnu.org/>`_                                 | 3.4.2           | C/C++ compiler\ :sup:`1`                    |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `TeXinfo <http://www.gnu.org/software/texinfo/>`_            | 4.5             | For building the CFE                        |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `SVN <http://subversion.tigris.org/project_packages.html>`_  | >=1.3           | Subversion access to LLVM\ :sup:`2`         |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `DejaGnu <http://savannah.gnu.org/projects/dejagnu>`_        | 1.4.2           | Automated test suite\ :sup:`3`              |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `tcl <http://www.tcl.tk/software/tcltk/>`_                   | 8.3, 8.4        | Automated test suite\ :sup:`3`              |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `expect <http://expect.nist.gov/>`_                          | 5.38.0          | Automated test suite\ :sup:`3`              |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `perl <http://www.perl.com/download.csp>`_                   | >=5.6.0         | Utilities                                   |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `GNU M4 <http://savannah.gnu.org/projects/m4>`_              | 1.4             | Macro processor for configuration\ :sup:`4` |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `GNU Autoconf <http://www.gnu.org/software/autoconf/>`_      | 2.60            | Configuration script builder\ :sup:`4`      |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `GNU Automake <http://www.gnu.org/software/automake/>`_      | 1.9.6           | aclocal macro generator\ :sup:`4`           |
+--------------------------------------------------------------+-----------------+---------------------------------------------+
| `libtool <http://savannah.gnu.org/projects/libtool>`_        | 1.5.22          | Shared library manager\ :sup:`4`            |
+--------------------------------------------------------------+-----------------+---------------------------------------------+

.. note::

   #. Only the C and C++ languages are needed so there's no need to build the
      other languages for LLVM's purposes. See `below` for specific version
      info.
   #. You only need Subversion if you intend to build from the latest LLVM
      sources. If you're working from a release distribution, you don't need
      Subversion.
   #. Only needed if you want to run the automated test suite in the
      ``llvm/test`` directory.
   #. If you want to make changes to the configure scripts, you will need GNU
      autoconf (2.60), and consequently, GNU M4 (version 1.4 or higher). You
      will also need automake (1.9.6). We only use aclocal from that package.

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

Broken versions of GCC and other tools
--------------------------------------

LLVM is very demanding of the host C++ compiler, and as such tends to expose
bugs in the compiler.  In particular, several versions of GCC crash when trying
to compile LLVM.  We routinely use GCC 4.2 (and higher) or Clang.  Other
versions of GCC will probably work as well.  GCC versions listed here are known
to not work.  If you are using one of these versions, please try to upgrade your
GCC to something more recent.  If you run into a problem with a version of GCC
not listed here, please `let us know <mailto:llvmdev@cs.uiuc.edu>`_.  Please use
the "``gcc -v``" command to find out which version of GCC you are using.

**GCC versions prior to 3.0**: GCC 2.96.x and before had several problems in the
STL that effectively prevent it from compiling LLVM.

**GCC 3.2.2 and 3.2.3**: These versions of GCC fails to compile LLVM with a
bogus template error.  This was fixed in later GCCs.

**GCC 3.3.2**: This version of GCC suffered from a `serious bug
<http://gcc.gnu.org/PR13392>`_ which causes it to crash in the
"``convert_from_eh_region_ranges_1``" GCC function.

**Cygwin GCC 3.3.3**: The version of GCC 3.3.3 commonly shipped with Cygwin does
not work.

**SuSE GCC 3.3.3**: The version of GCC 3.3.3 shipped with SuSE 9.1 (and possibly
others) does not compile LLVM correctly (it appears that exception handling is
broken in some cases).  Please download the FSF 3.3.3 or upgrade to a newer
version of GCC.

**GCC 3.4.0 on linux/x86 (32-bit)**: GCC miscompiles portions of the code
generator, causing an infinite loop in the llvm-gcc build when built with
optimizations enabled (i.e. a release build).

**GCC 3.4.2 on linux/x86 (32-bit)**: GCC miscompiles portions of the code
generator at -O3, as with 3.4.0.  However gcc 3.4.2 (unlike 3.4.0) correctly
compiles LLVM at -O2.  A work around is to build release LLVM builds with
"``make ENABLE_OPTIMIZED=1 OPTIMIZE_OPTION=-O2 ...``"

**GCC 3.4.x on X86-64/amd64**: GCC `miscompiles portions of LLVM
<http://llvm.org/PR1056>`__.

**GCC 3.4.4 (CodeSourcery ARM 2005q3-2)**: this compiler miscompiles LLVM when
building with optimizations enabled.  It appears to work with "``make
ENABLE_OPTIMIZED=1 OPTIMIZE_OPTION=-O1``" or build a debug build.

**IA-64 GCC 4.0.0**: The IA-64 version of GCC 4.0.0 is known to miscompile LLVM.

**Apple Xcode 2.3**: GCC crashes when compiling LLVM at -O3 (which is the
default with ENABLE_OPTIMIZED=1.  To work around this, build with
"``ENABLE_OPTIMIZED=1 OPTIMIZE_OPTION=-O2``".

**GCC 4.1.1**: GCC fails to build LLVM with template concept check errors
compiling some files.  At the time of this writing, GCC mainline (4.2) did not
share the problem.

**GCC 4.1.1 on X86-64/amd64**: GCC `miscompiles portions of LLVM
<http://llvm.org/PR1063>`__ when compiling llvm itself into 64-bit code.  LLVM
will appear to mostly work but will be buggy, e.g. failing portions of its
testsuite.

**GCC 4.1.2 on OpenSUSE**: Seg faults during libstdc++ build and on x86_64
platforms compiling md5.c gets a mangled constant.

**GCC 4.1.2 (20061115 (prerelease) (Debian 4.1.1-21)) on Debian**: Appears to
miscompile parts of LLVM 2.4. One symptom is ValueSymbolTable complaining about
symbols remaining in the table on destruction.

**GCC 4.1.2 20071124 (Red Hat 4.1.2-42)**: Suffers from the same symptoms as the
previous one. It appears to work with ENABLE_OPTIMIZED=0 (the default).

**Cygwin GCC 4.3.2 20080827 (beta) 2**: Users `reported
<http://llvm.org/PR4145>`_ various problems related with link errors when using
this GCC version.

**Debian GCC 4.3.2 on X86**: Crashes building some files in LLVM 2.6.

**GCC 4.3.3 (Debian 4.3.3-10) on ARM**: Miscompiles parts of LLVM 2.6 when
optimizations are turned on. The symptom is an infinite loop in
``FoldingSetImpl::RemoveNode`` while running the code generator.

**SUSE 11 GCC 4.3.4**: Miscompiles LLVM, causing crashes in ValueHandle logic.

**GCC 4.3.5 and GCC 4.4.5 on ARM**: These can miscompile ``value >> 1`` even at
``-O0``. A test failure in ``test/Assembler/alignstack.ll`` is one symptom of
the problem.

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

.. _Setting Up Your Environment:

Setting Up Your Environment
---------------------------

In order to compile and use LLVM, you may need to set some environment
variables.

``LLVM_LIB_SEARCH_PATH=/path/to/your/bitcode/libs``

  [Optional] This environment variable helps LLVM linking tools find the
  locations of your bitcode libraries. It is provided only as a convenience
  since you can specify the paths using the -L options of the tools and the
  C/C++ front-end will automatically use the bitcode files installed in its
  ``lib`` directory.

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

``llvm-gcc-4.2-x.y.source.tar.gz``

  Source release of the llvm-gcc-4.2 front end.  See README.LLVM in the root
  directory for build instructions.

``llvm-gcc-4.2-x.y-platform.tar.gz``

  Binary release of the llvm-gcc-4.2 front end for a specific platform.

Checkout LLVM from Subversion
-----------------------------

If you have access to our Subversion repository, you can get a fresh copy of the
entire source code.  All you need to do is check it out from Subversion as
follows:

* ``cd where-you-want-llvm-to-live``
* Read-Only: ``svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm``
* Read-Write:``svn co https://user@llvm.org/svn/llvm-project/llvm/trunk llvm``

This will create an '``llvm``' directory in the current directory and fully
populate it with the LLVM source code, Makefiles, test directories, and local
copies of documentation files.

If you want to get a specific release (as opposed to the most recent revision),
you can checkout it from the '``tags``' directory (instead of '``trunk``'). The
following releases are located in the following subdirectories of the '``tags``'
directory:

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

.. code-block:: bash

  % cd llvm/projects
  % svn co http://llvm.org/svn/llvm-project/test-suite/trunk test-suite

By placing it in the ``llvm/projects``, it will be automatically configured by
the LLVM configure script as well as automatically updated when you run ``svn
update``.

GIT mirror
----------

GIT mirrors are available for a number of LLVM subprojects. These mirrors sync
automatically with each Subversion commit and contain all necessary git-svn
marks (so, you can recreate git-svn metadata locally). Note that right now
mirrors reflect only ``trunk`` for each project. You can do the read-only GIT
clone of LLVM via:

.. code-block:: bash

  % git clone http://llvm.org/git/llvm.git

If you want to check out clang too, run:

.. code-block:: bash

  % git clone http://llvm.org/git/llvm.git
  % cd llvm/tools
  % git clone http://llvm.org/git/clang.git

Since the upstream repository is in Subversion, you should use ``git
pull --rebase`` instead of ``git pull`` to avoid generating a non-linear history
in your clone.  To configure ``git pull`` to pass ``--rebase`` by default on the
master branch, run the following command:

.. code-block:: bash

  % git config branch.master.rebase true

Sending patches with Git
^^^^^^^^^^^^^^^^^^^^^^^^

Please read `Developer Policy <DeveloperPolicy.html#one-off-patches>`_, too.

Assume ``master`` points the upstream and ``mybranch`` points your working
branch, and ``mybranch`` is rebased onto ``master``.  At first you may check
sanity of whitespaces:

.. code-block:: bash

  % git diff --check master..mybranch

The easiest way to generate a patch is as below:

.. code-block:: bash

  % git diff master..mybranch > /path/to/mybranch.diff

It is a little different from svn-generated diff. git-diff-generated diff has
prefixes like ``a/`` and ``b/``. Don't worry, most developers might know it
could be accepted with ``patch -p1 -N``.

But you may generate patchset with git-format-patch. It generates by-each-commit
patchset. To generate patch files to attach to your article:

.. code-block:: bash

  % git format-patch --no-attach master..mybranch -o /path/to/your/patchset

If you would like to send patches directly, you may use git-send-email or
git-imap-send. Here is an example to generate the patchset in Gmail's [Drafts].

.. code-block:: bash

  % git format-patch --attach master..mybranch --stdout | git imap-send

Then, your .git/config should have [imap] sections.

.. code-block:: bash

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

For developers to work with git-svn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up clone from which you can submit code using ``git-svn``, run:

.. code-block:: bash

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

To update this clone without generating git-svn tags that conflict with the
upstream git repo, run:

.. code-block:: bash

  % git fetch && (cd tools/clang && git fetch)  # Get matching revisions of both trees.
  % git checkout master
  % git svn rebase -l
  % (cd tools/clang &&
     git checkout master &&
     git svn rebase -l)

This leaves your working directories on their master branches, so you'll need to
``checkout`` each working branch individually and ``rebase`` it on top of its
parent branch.  (Note: This script is intended for relative newbies to git.  If
you have more experience, you can likely improve on it.)

The git-svn metadata can get out of sync after you mess around with branches and
``dcommit``. When that happens, ``git svn dcommit`` stops working, complaining
about files with uncommitted changes. The fix is to rebuild the metadata:

.. code-block:: bash

  % rm -rf .git/svn
  % git svn rebase -l

Local LLVM Configuration
------------------------

Once checked out from the Subversion repository, the LLVM suite source code must
be configured via the ``configure`` script.  This script sets variables in the
various ``*.in`` files, most notably ``llvm/Makefile.config`` and
``llvm/include/Config/config.h``.  It also populates *OBJ_ROOT* with the
Makefiles needed to begin building LLVM.

The following environment variables are used by the ``configure`` script to
configure the build system:

+------------+-----------------------------------------------------------+
| Variable   | Purpose                                                   |
+============+===========================================================+
| CC         | Tells ``configure`` which C compiler to use.  By default, |
|            | ``configure`` will look for the first GCC C compiler in   |
|            | ``PATH``.  Use this variable to override ``configure``\'s |
|            | default behavior.                                         |
+------------+-----------------------------------------------------------+
| CXX        | Tells ``configure`` which C++ compiler to use.  By        |
|            | default, ``configure`` will look for the first GCC C++    |
|            | compiler in ``PATH``.  Use this variable to override      |
|            | ``configure``'s default behavior.                         |
+------------+-----------------------------------------------------------+

The following options can be used to set or enable LLVM specific options:

``--enable-optimized``

  Enables optimized compilation (debugging symbols are removed and GCC
  optimization flags are enabled). Note that this is the default setting if you
  are using the LLVM distribution. The default behavior of an Subversion
  checkout is to use an unoptimized build (also known as a debug build).

``--enable-debug-runtime``

  Enables debug symbols in the runtime libraries. The default is to strip debug
  symbols from the runtime libraries.

``--enable-jit``

  Compile the Just In Time (JIT) compiler functionality.  This is not available
  on all platforms.  The default is dependent on platform, so it is best to
  explicitly enable it if you want it.

``--enable-targets=target-option``

  Controls which targets will be built and linked into llc. The default value
  for ``target_options`` is "all" which builds and links all available targets.
  The value "host-only" can be specified to build only a native compiler (no
  cross-compiler targets available). The "native" target is selected as the
  target of the build host. You can also specify a comma separated list of
  target names that you want available in llc. The target names use all lower
  case. The current set of targets is:

    ``arm, cpp, hexagon, mblaze, mips, mipsel, msp430, powerpc, ptx, sparc, spu,
    x86, x86_64, xcore``.

``--enable-doxygen``

  Look for the doxygen program and enable construction of doxygen based
  documentation from the source code. This is disabled by default because
  generating the documentation can take a long time and producess 100s of
  megabytes of output.

``--with-udis86``

  LLVM can use external disassembler library for various purposes (now it's used
  only for examining code produced by JIT). This option will enable usage of
  `udis86 <http://udis86.sourceforge.net/>`_ x86 (both 32 and 64 bits)
  disassembler library.

To configure LLVM, follow these steps:

#. Change directory into the object root directory:

   .. code-block:: bash

     % cd OBJ_ROOT

#. Run the ``configure`` script located in the LLVM source tree:

   .. code-block:: bash

     % SRC_ROOT/configure --prefix=/install/path [other options]

Compiling the LLVM Suite Source Code
------------------------------------

Once you have configured LLVM, you can build it.  There are three types of
builds:

Debug Builds

  These builds are the default when one is using an Subversion checkout and
  types ``gmake`` (unless the ``--enable-optimized`` option was used during
  configuration).  The build system will compile the tools and libraries with
  debugging information.  To get a Debug Build using the LLVM distribution the
  ``--disable-optimized`` option must be passed to ``configure``.

Release (Optimized) Builds

  These builds are enabled with the ``--enable-optimized`` option to
  ``configure`` or by specifying ``ENABLE_OPTIMIZED=1`` on the ``gmake`` command
  line.  For these builds, the build system will compile the tools and libraries
  with GCC optimizations enabled and strip debugging information from the
  libraries and executables it generates.  Note that Release Builds are default
  when using an LLVM distribution.

Profile Builds

  These builds are for use with profiling.  They compile profiling information
  into the code for use with programs like ``gprof``.  Profile builds must be
  started by specifying ``ENABLE_PROFILING=1`` on the ``gmake`` command line.

Once you have LLVM configured, you can build it by entering the *OBJ_ROOT*
directory and issuing the following command:

.. code-block:: bash

  % gmake

If the build fails, please `check here`_ to see if you are using a version of
GCC that is known not to compile LLVM.

If you have multiple processors in your machine, you may wish to use some of the
parallel build options provided by GNU Make.  For example, you could use the
command:

.. code-block:: bash

  % gmake -j2

There are several special targets which are useful when working with the LLVM
source code:

``gmake clean``

  Removes all files generated by the build.  This includes object files,
  generated C/C++ files, libraries, and executables.

``gmake dist-clean``

  Removes everything that ``gmake clean`` does, but also removes files generated
  by ``configure``.  It attempts to return the source tree to the original state
  in which it was shipped.

``gmake install``

  Installs LLVM header files, libraries, tools, and documentation in a hierarchy
  under ``$PREFIX``, specified with ``./configure --prefix=[dir]``, which
  defaults to ``/usr/local``.

``gmake -C runtime install-bytecode``

  Assuming you built LLVM into $OBJDIR, when this command is run, it will
  install bitcode libraries into the GCC front end's bitcode library directory.
  If you need to update your bitcode libraries, this is the target to use once
  you've built them.

Please see the `Makefile Guide <MakefileGuide.html>`_ for further details on
these ``make`` targets and descriptions of other targets available.

It is also possible to override default values from ``configure`` by declaring
variables on the command line.  The following are some examples:

``gmake ENABLE_OPTIMIZED=1``

  Perform a Release (Optimized) build.

``gmake ENABLE_OPTIMIZED=1 DISABLE_ASSERTIONS=1``

  Perform a Release (Optimized) build without assertions enabled.
 
``gmake ENABLE_OPTIMIZED=0``

  Perform a Debug build.

``gmake ENABLE_PROFILING=1``

  Perform a Profiling build.

``gmake VERBOSE=1``

  Print what ``gmake`` is doing on standard output.

``gmake TOOL_VERBOSE=1``

  Ask each tool invoked by the makefiles to print out what it is doing on 
  the standard output. This also implies ``VERBOSE=1``.

Every directory in the LLVM object tree includes a ``Makefile`` to build it and
any subdirectories that it contains.  Entering any directory inside the LLVM
object tree and typing ``gmake`` should rebuild anything in or below that
directory that is out of date.

Cross-Compiling LLVM
--------------------

It is possible to cross-compile LLVM itself. That is, you can create LLVM
executables and libraries to be hosted on a platform different from the platform
where they are build (a Canadian Cross build). To configure a cross-compile,
supply the configure script with ``--build`` and ``--host`` options that are
different. The values of these options must be legal target triples that your
GCC compiler supports.

The result of such a build is executables that are not runnable on on the build
host (--build option) but can be executed on the compile host (--host option).

The Location of LLVM Object Files
---------------------------------

The LLVM build system is capable of sharing a single LLVM source tree among
several LLVM builds.  Hence, it is possible to build LLVM for several different
platforms or configurations using the same source tree.

This is accomplished in the typical autoconf manner:

* Change directory to where the LLVM object files should live:

  .. code-block:: bash

    % cd OBJ_ROOT

* Run the ``configure`` script found in the LLVM source directory:

  .. code-block:: bash

    % SRC_ROOT/configure

The LLVM build will place files underneath *OBJ_ROOT* in directories named after
the build type:

Debug Builds with assertions enabled (the default)

  Tools

    ``OBJ_ROOT/Debug+Asserts/bin``

  Libraries

    ``OBJ_ROOT/Debug+Asserts/lib``

Release Builds

  Tools

    ``OBJ_ROOT/Release/bin``

  Libraries

    ``OBJ_ROOT/Release/lib``

Profile Builds

  Tools

    ``OBJ_ROOT/Profile/bin``

  Libraries

    ``OBJ_ROOT/Profile/lib``

Optional Configuration Items
----------------------------

If you're running on a Linux system that supports the `binfmt_misc
<http://www.tat.physik.uni-tuebingen.de/~rguenth/linux/binfmt_misc.html>`_
module, and you have root access on the system, you can set your system up to
execute LLVM bitcode files directly. To do this, use commands like this (the
first command may not be required if you are already using the module):

.. code-block:: bash

  % mount -t binfmt_misc none /proc/sys/fs/binfmt_misc
  % echo ':llvm:M::BC::/path/to/lli:' > /proc/sys/fs/binfmt_misc/register
  % chmod u+x hello.bc   (if needed)
  % ./hello.bc

This allows you to execute LLVM bitcode files directly.  On Debian, you can also
use this command instead of the 'echo' command above:

.. code-block:: bash

  % sudo update-binfmts --install llvm /path/to/lli --magic 'BC'

.. _Program Layout:
.. _general layout:

Program Layout
==============

One useful source of information about the LLVM source base is the LLVM `doxygen
<http://www.doxygen.org/>`_ documentation available at
`<http://llvm.org/doxygen/>`_.  The following is a brief introduction to code
layout:

``llvm/examples``
-----------------

This directory contains some simple examples of how to use the LLVM IR and JIT.

``llvm/include``
----------------

This directory contains public header files exported from the LLVM library. The
three main subdirectories of this directory are:

``llvm/include/llvm``

  This directory contains all of the LLVM specific header files.  This directory
  also has subdirectories for different portions of LLVM: ``Analysis``,
  ``CodeGen``, ``Target``, ``Transforms``, etc...

``llvm/include/llvm/Support``

  This directory contains generic support libraries that are provided with LLVM
  but not necessarily specific to LLVM. For example, some C++ STL utilities and
  a Command Line option processing library store their header files here.

``llvm/include/llvm/Config``

  This directory contains header files configured by the ``configure`` script.
  They wrap "standard" UNIX and C header files.  Source code can include these
  header files which automatically take care of the conditional #includes that
  the ``configure`` script generates.

``llvm/lib``
------------

This directory contains most of the source files of the LLVM system. In LLVM,
almost all code exists in libraries, making it very easy to share code among the
different `tools`_.

``llvm/lib/VMCore/``

  This directory holds the core LLVM source files that implement core classes
  like Instruction and BasicBlock.

``llvm/lib/AsmParser/``

  This directory holds the source code for the LLVM assembly language parser
  library.

``llvm/lib/BitCode/``

  This directory holds code for reading and write LLVM bitcode.

``llvm/lib/Analysis/``

  This directory contains a variety of different program analyses, such as
  Dominator Information, Call Graphs, Induction Variables, Interval
  Identification, Natural Loop Identification, etc.

``llvm/lib/Transforms/``

  This directory contains the source code for the LLVM to LLVM program
  transformations, such as Aggressive Dead Code Elimination, Sparse Conditional
  Constant Propagation, Inlining, Loop Invariant Code Motion, Dead Global
  Elimination, and many others.

``llvm/lib/Target/``

  This directory contains files that describe various target architectures for
  code generation.  For example, the ``llvm/lib/Target/X86`` directory holds the
  X86 machine description while ``llvm/lib/Target/ARM`` implements the ARM
  backend.
    
``llvm/lib/CodeGen/``

  This directory contains the major parts of the code generator: Instruction
  Selector, Instruction Scheduling, and Register Allocation.

``llvm/lib/MC/``

  (FIXME: T.B.D.)

``llvm/lib/Debugger/``

  This directory contains the source level debugger library that makes it
  possible to instrument LLVM programs so that a debugger could identify source
  code locations at which the program is executing.

``llvm/lib/ExecutionEngine/``

  This directory contains libraries for executing LLVM bitcode directly at
  runtime in both interpreted and JIT compiled fashions.

``llvm/lib/Support/``

  This directory contains the source code that corresponds to the header files
  located in ``llvm/include/ADT/`` and ``llvm/include/Support/``.

``llvm/projects``
-----------------

This directory contains projects that are not strictly part of LLVM but are
shipped with LLVM. This is also the directory where you should create your own
LLVM-based projects. See ``llvm/projects/sample`` for an example of how to set
up your own project.

``llvm/runtime``
----------------

This directory contains libraries which are compiled into LLVM bitcode and used
when linking programs with the Clang front end.  Most of these libraries are
skeleton versions of real libraries; for example, libc is a stripped down
version of glibc.

Unlike the rest of the LLVM suite, this directory needs the LLVM GCC front end
to compile.

``llvm/test``
-------------

This directory contains feature and regression tests and other basic sanity
checks on the LLVM infrastructure. These are intended to run quickly and cover a
lot of territory without being exhaustive.

``test-suite``
--------------

This is not a directory in the normal llvm module; it is a separate Subversion
module that must be checked out (usually to ``projects/test-suite``).  This
module contains a comprehensive correctness, performance, and benchmarking test
suite for LLVM. It is a separate Subversion module because not every LLVM user
is interested in downloading or building such a comprehensive test suite. For
further details on this test suite, please see the `Testing
Guide <TestingGuide.html>`_ document.

.. _tools:

``llvm/tools``
--------------

The **tools** directory contains the executables built out of the libraries
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
  (which are specified on the command line), and then outputs the resultant
  bitcode.  The '``opt -help``' command is a good way to get a list of the
  program transformations available in LLVM.

  ``opt`` can also be used to run a specific analysis on an input LLVM bitcode
  file and print out the results.  It is primarily useful for debugging
  analyses, or familiarizing yourself with what an analysis does.

``llvm/utils``
--------------

This directory contains utilities for working with LLVM source code, and some of
the utilities are actually required as part of the build process because they
are code generators for parts of LLVM infrastructure.


``codegen-diff``

  ``codegen-diff`` is a script that finds differences between code that LLC
  generates and code that LLI generates. This is a useful tool if you are
  debugging one of them, assuming that the other generates correct output. For
  the full user manual, run ```perldoc codegen-diff'``.

``emacs/``

  The ``emacs`` directory contains syntax-highlighting files which will work
  with Emacs and XEmacs editors, providing syntax highlighting support for LLVM
  assembly files and TableGen description files. For information on how to use
  the syntax files, consult the ``README`` file in that directory.

``getsrcs.sh``

  The ``getsrcs.sh`` script finds and outputs all non-generated source files,
  which is useful if one wishes to do a lot of development across directories
  and does not want to individually find each file. One way to use it is to run,
  for example: ``xemacs `utils/getsources.sh``` from the top of your LLVM source
  tree.

``llvmgrep``

  This little tool performs an ``egrep -H -n`` on each source file in LLVM and
  passes to it a regular expression provided on ``llvmgrep``'s command
  line. This is a very efficient way of searching the source base for a
  particular regular expression.

``makellvm``

  The ``makellvm`` script compiles all files in the current directory and then
  compiles and links the tool that is the first argument. For example, assuming
  you are in the directory ``llvm/lib/Target/Sparc``, if ``makellvm`` is in your
  path, simply running ``makellvm llc`` will make a build of the current
  directory, switch to directory ``llvm/tools/llc`` and build it, causing a
  re-linking of LLC.

``TableGen/``

  The ``TableGen`` directory contains the tool used to generate register
  descriptions, instruction set descriptions, and even assemblers from common
  TableGen description files.

``vim/``

  The ``vim`` directory contains syntax-highlighting files which will work with
  the VIM editor, providing syntax highlighting support for LLVM assembly files
  and TableGen description files. For information on how to use the syntax
  files, consult the ``README`` file in that directory.

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

   .. code-block:: bash

     % clang hello.c -o hello

   .. note::

     Clang works just like GCC by default.  The standard -S and -c arguments
     work as usual (producing a native .s or .o file, respectively).

#. Next, compile the C file into a LLVM bitcode file:

   .. code-block:: bash

     % clang -O3 -emit-llvm hello.c -c -o hello.bc

   The -emit-llvm option can be used with the -S or -c options to emit an LLVM
   ``.ll`` or ``.bc`` file (respectively) for the code.  This allows you to use
   the `standard LLVM tools <CommandGuide/index.html>`_ on the bitcode file.

#. Run the program in both forms. To run the program, use:

   .. code-block:: bash

      % ./hello
 
   and

   .. code-block:: bash

     % lli hello.bc

   The second examples shows how to invoke the LLVM JIT, `lli
   <CommandGuide/html/lli.html>`_.

#. Use the ``llvm-dis`` utility to take a look at the LLVM assembly code:

   .. code-block:: bash

     % llvm-dis < hello.bc | less

#. Compile the program to native assembly using the LLC code generator:

   .. code-block:: bash

     % llc hello.bc -o hello.s

#. Assemble the native assembly language file into a program:

   .. code-block:: bash

     **Solaris:** % /opt/SUNWspro/bin/cc -xarch=v9 hello.s -o hello.native

     **Others:**  % gcc hello.s -o hello.native

#. Execute the native code program:

   .. code-block:: bash

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
