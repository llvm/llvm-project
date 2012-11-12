.. _makefile_guide:

===================
LLVM Makefile Guide
===================

.. contents::
   :local:

Introduction
============

This document provides *usage* information about the LLVM makefile system. While
loosely patterned after the BSD makefile system, LLVM has taken a departure from
BSD in order to implement additional features needed by LLVM.  Although makefile
systems, such as ``automake``, were attempted at one point, it has become clear
that the features needed by LLVM and the ``Makefile`` norm are too great to use
a more limited tool. Consequently, LLVM requires simply GNU Make 3.79, a widely
portable makefile processor. LLVM unabashedly makes heavy use of the features of
GNU Make so the dependency on GNU Make is firm. If you're not familiar with
``make``, it is recommended that you read the `GNU Makefile Manual
<http://www.gnu.org/software/make/manual/make.html>`_.

While this document is rightly part of the `LLVM Programmer's
Manual <ProgrammersManual.html>`_, it is treated separately here because of the
volume of content and because it is often an early source of bewilderment for
new developers.

General Concepts
================

The LLVM Makefile System is the component of LLVM that is responsible for
building the software, testing it, generating distributions, checking those
distributions, installing and uninstalling, etc. It consists of a several files
throughout the source tree. These files and other general concepts are described
in this section.

Projects
--------

The LLVM Makefile System is quite generous. It not only builds its own software,
but it can build yours too. Built into the system is knowledge of the
``llvm/projects`` directory. Any directory under ``projects`` that has both a
``configure`` script and a ``Makefile`` is assumed to be a project that uses the
LLVM Makefile system.  Building software that uses LLVM does not require the
LLVM Makefile System nor even placement in the ``llvm/projects``
directory. However, doing so will allow your project to get up and running
quickly by utilizing the built-in features that are used to compile LLVM. LLVM
compiles itself using the same features of the makefile system as used for
projects.

For complete details on setting up your projects configuration, simply mimic the
``llvm/projects/sample`` project. Or for further details, consult the
`Projects <Projects.html>`_ page.

Variable Values
---------------

To use the makefile system, you simply create a file named ``Makefile`` in your
directory and declare values for certain variables.  The variables and values
that you select determine what the makefile system will do. These variables
enable rules and processing in the makefile system that automatically Do The
Right Thing&trade;.

Including Makefiles
-------------------

Setting variables alone is not enough. You must include into your Makefile
additional files that provide the rules of the LLVM Makefile system. The various
files involved are described in the sections that follow.

``Makefile``
^^^^^^^^^^^^

Each directory to participate in the build needs to have a file named
``Makefile``. This is the file first read by ``make``. It has three
sections:

#. Settable Variables --- Required that must be set first.
#. ``include $(LEVEL)/Makefile.common`` --- include the LLVM Makefile system.
#. Override Variables --- Override variables set by the LLVM Makefile system.

.. _$(LEVEL)/Makefile.common:

``Makefile.common``
^^^^^^^^^^^^^^^^^^^

Every project must have a ``Makefile.common`` file at its top source
directory. This file serves three purposes:

#. It includes the project's configuration makefile to obtain values determined
   by the ``configure`` script. This is done by including the
   `$(LEVEL)/Makefile.config`_ file.

#. It specifies any other (static) values that are needed throughout the
   project. Only values that are used in all or a large proportion of the
   project's directories should be placed here.

#. It includes the standard rules for the LLVM Makefile system,
   `$(LLVM_SRC_ROOT)/Makefile.rules`_.  This file is the *guts* of the LLVM
   ``Makefile`` system.

.. _$(LEVEL)/Makefile.config:

``Makefile.config``
^^^^^^^^^^^^^^^^^^^

Every project must have a ``Makefile.config`` at the top of its *build*
directory. This file is **generated** by the ``configure`` script from the
pattern provided by the ``Makefile.config.in`` file located at the top of the
project's *source* directory. The contents of this file depend largely on what
configuration items the project uses, however most projects can get what they
need by just relying on LLVM's configuration found in
``$(LLVM_OBJ_ROOT)/Makefile.config``.

.. _$(LLVM_SRC_ROOT)/Makefile.rules:

``Makefile.rules``
^^^^^^^^^^^^^^^^^^

This file, located at ``$(LLVM_SRC_ROOT)/Makefile.rules`` is the heart of the
LLVM Makefile System. It provides all the logic, dependencies, and rules for
building the targets supported by the system. What it does largely depends on
the values of ``make`` `variables`_ that have been set *before*
``Makefile.rules`` is included.

Comments
^^^^^^^^

User ``Makefile``\s need not have comments in them unless the construction is
unusual or it does not strictly follow the rules and patterns of the LLVM
makefile system. Makefile comments are invoked with the pound (``#``) character.
The ``#`` character and any text following it, to the end of the line, are
ignored by ``make``.

Tutorial
========

This section provides some examples of the different kinds of modules you can
build with the LLVM makefile system. In general, each directory you provide will
build a single object although that object may be composed of additionally
compiled components.

Libraries
---------

Only a few variable definitions are needed to build a regular library.
Normally, the makefile system will build all the software into a single
``libname.o`` (pre-linked) object. This means the library is not searchable and
that the distinction between compilation units has been dissolved. Optionally,
you can ask for a shared library (.so) or archive library (.a) built.  Archive
libraries are the default. For example:

.. code-block:: makefile

  LIBRARYNAME = mylib
  SHARED_LIBRARY = 1
  ARCHIVE_LIBRARY = 1

says to build a library named ``mylib`` with both a shared library
(``mylib.so``) and an archive library (``mylib.a``) version. The contents of all
the libraries produced will be the same, they are just constructed differently.
Note that you normally do not need to specify the sources involved. The LLVM
Makefile system will infer the source files from the contents of the source
directory.

The ``LOADABLE_MODULE=1`` directive can be used in conjunction with
``SHARED_LIBRARY=1`` to indicate that the resulting shared library should be
openable with the ``dlopen`` function and searchable with the ``dlsym`` function
(or your operating system's equivalents). While this isn't strictly necessary on
Linux and a few other platforms, it is required on systems like HP-UX and
Darwin. You should use ``LOADABLE_MODULE`` for any shared library that you
intend to be loaded into an tool via the ``-load`` option. See the
`WritingAnLLVMPass.html <WritingAnLLVMPass.html#makefile>`_ document for an
example of why you might want to do this.

Bitcode Modules
^^^^^^^^^^^^^^^

In some situations, it is desirable to build a single bitcode module from a
variety of sources, instead of an archive, shared library, or bitcode
library. Bitcode modules can be specified in addition to any of the other types
of libraries by defining the `MODULE_NAME`_ variable. For example:

.. code-block:: makefile

  LIBRARYNAME = mylib
  BYTECODE_LIBRARY = 1
  MODULE_NAME = mymod

will build a module named ``mymod.bc`` from the sources in the directory. This
module will be an aggregation of all the bitcode modules derived from the
sources. The example will also build a bitcode archive containing a bitcode
module for each compiled source file. The difference is subtle, but important
depending on how the module or library is to be linked.

Loadable Modules
^^^^^^^^^^^^^^^^

In some situations, you need to create a loadable module. Loadable modules can
be loaded into programs like ``opt`` or ``llc`` to specify additional passes to
run or targets to support.  Loadable modules are also useful for debugging a
pass or providing a pass with another package if that pass can't be included in
LLVM.

LLVM provides complete support for building such a module. All you need to do is
use the ``LOADABLE_MODULE`` variable in your ``Makefile``. For example, to build
a loadable module named ``MyMod`` that uses the LLVM libraries ``LLVMSupport.a``
and ``LLVMSystem.a``, you would specify:

.. code-block:: makefile

  LIBRARYNAME := MyMod
  LOADABLE_MODULE := 1
  LINK_COMPONENTS := support system

Use of the ``LOADABLE_MODULE`` facility implies several things:

#. There will be no "``lib``" prefix on the module. This differentiates it from
    a standard shared library of the same name.

#. The `SHARED_LIBRARY`_ variable is turned on.

#. The `LINK_LIBS_IN_SHARED`_ variable is turned on.

A loadable module is loaded by LLVM via the facilities of libtool's libltdl
library which is part of ``lib/System`` implementation.

Tools
-----

For building executable programs (tools), you must provide the name of the tool
and the names of the libraries you wish to link with the tool. For example:

.. code-block:: makefile

  TOOLNAME = mytool
  USEDLIBS = mylib
  LINK_COMPONENTS = support system

says that we are to build a tool name ``mytool`` and that it requires three
libraries: ``mylib``, ``LLVMSupport.a`` and ``LLVMSystem.a``.

Note that two different variables are use to indicate which libraries are
linked: ``USEDLIBS`` and ``LLVMLIBS``. This distinction is necessary to support
projects. ``LLVMLIBS`` refers to the LLVM libraries found in the LLVM object
directory. ``USEDLIBS`` refers to the libraries built by your project. In the
case of building LLVM tools, ``USEDLIBS`` and ``LLVMLIBS`` can be used
interchangeably since the "project" is LLVM itself and ``USEDLIBS`` refers to
the same place as ``LLVMLIBS``.

Also note that there are two different ways of specifying a library: with a
``.a`` suffix and without. Without the suffix, the entry refers to the re-linked
(.o) file which will include *all* symbols of the library.  This is
useful, for example, to include all passes from a library of passes.  If the
``.a`` suffix is used then the library is linked as a searchable library (with
the ``-l`` option). In this case, only the symbols that are unresolved *at
that point* will be resolved from the library, if they exist. Other
(unreferenced) symbols will not be included when the ``.a`` syntax is used. Note
that in order to use the ``.a`` suffix, the library in question must have been
built with the ``ARCHIVE_LIBRARY`` option set.

JIT Tools
^^^^^^^^^

Many tools will want to use the JIT features of LLVM.  To do this, you simply
specify that you want an execution 'engine', and the makefiles will
automatically link in the appropriate JIT for the host or an interpreter if none
is available:

.. code-block:: makefile

  TOOLNAME = my_jit_tool
  USEDLIBS = mylib
  LINK_COMPONENTS = engine

Of course, any additional libraries may be listed as other components.  To get a
full understanding of how this changes the linker command, it is recommended
that you:

.. code-block:: bash

  % cd examples/Fibonacci
  % make VERBOSE=1

Targets Supported
=================

This section describes each of the targets that can be built using the LLVM
Makefile system. Any target can be invoked from any directory but not all are
applicable to a given directory (e.g. "check", "dist" and "install" will always
operate as if invoked from the top level directory).

================= ===============      ==================
Target Name       Implied Targets      Target Description
================= ===============      ==================
``all``           \                    Compile the software recursively. Default target.
``all-local``     \                    Compile the software in the local directory only.
``check``         \                    Change to the ``test`` directory in a project and run the test suite there.
``check-local``   \                    Run a local test suite. Generally this is only defined in the  ``Makefile`` of the project's ``test`` directory.
``clean``         \                    Remove built objects recursively.
``clean-local``   \                    Remove built objects from the local directory only.
``dist``          ``all``              Prepare a source distribution tarball.
``dist-check``    ``all``              Prepare a source distribution tarball and check that it builds.
``dist-clean``    ``clean``            Clean source distribution tarball temporary files.
``install``       ``all``              Copy built objects to installation directory.
``preconditions`` ``all``              Check to make sure configuration and makefiles are up to date.
``printvars``     ``all``              Prints variables defined by the makefile system (for debugging).
``tags``          \                    Make C and C++ tags files for emacs and vi.
``uninstall``     \                    Remove built objects from installation directory.
================= ===============      ==================

.. _all:

``all`` (default)
-----------------

When you invoke ``make`` with no arguments, you are implicitly instructing it to
seek the ``all`` target (goal). This target is used for building the software
recursively and will do different things in different directories.  For example,
in a ``lib`` directory, the ``all`` target will compile source files and
generate libraries. But, in a ``tools`` directory, it will link libraries and
generate executables.

``all-local``
-------------

This target is the same as `all`_ but it operates only on the current directory
instead of recursively.

``check``
---------

This target can be invoked from anywhere within a project's directories but
always invokes the `check-local`_ target in the project's ``test`` directory, if
it exists and has a ``Makefile``. A warning is produced otherwise.  If
`TESTSUITE`_ is defined on the ``make`` command line, it will be passed down to
the invocation of ``make check-local`` in the ``test`` directory. The intended
usage for this is to assist in running specific suites of tests. If
``TESTSUITE`` is not set, the implementation of ``check-local`` should run all
normal tests.  It is up to the project to define what different values for
``TESTSUTE`` will do. See the `Testing Guide <TestingGuide.html>`_ for further
details.

``check-local``
---------------

This target should be implemented by the ``Makefile`` in the project's ``test``
directory. It is invoked by the ``check`` target elsewhere.  Each project is
free to define the actions of ``check-local`` as appropriate for that
project. The LLVM project itself uses dejagnu to run a suite of feature and
regresson tests. Other projects may choose to use dejagnu or any other testing
mechanism.

``clean``
---------

This target cleans the build directory, recursively removing all things that the
Makefile builds. The cleaning rules have been made guarded so they shouldn't go
awry (via ``rm -f $(UNSET_VARIABLE)/*`` which will attempt to erase the entire
directory structure.

``clean-local``
---------------

This target does the same thing as ``clean`` but only for the current (local)
directory.

``dist``
--------

This target builds a distribution tarball. It first builds the entire project
using the ``all`` target and then tars up the necessary files and compresses
it. The generated tarball is sufficient for a casual source distribution, but
probably not for a release (see ``dist-check``).

``dist-check``
--------------

This target does the same thing as the ``dist`` target but also checks the
distribution tarball. The check is made by unpacking the tarball to a new
directory, configuring it, building it, installing it, and then verifying that
the installation results are correct (by comparing to the original build).  This
target can take a long time to run but should be done before a release goes out
to make sure that the distributed tarball can actually be built into a working
release.

``dist-clean``
--------------

This is a special form of the ``clean`` clean target. It performs a normal
``clean`` but also removes things pertaining to building the distribution.

``install``
-----------

This target finalizes shared objects and executables and copies all libraries,
headers, executables and documentation to the directory given with the
``--prefix`` option to ``configure``.  When completed, the prefix directory will
have everything needed to **use** LLVM.

The LLVM makefiles can generate complete **internal** documentation for all the
classes by using ``doxygen``. By default, this feature is **not** enabled
because it takes a long time and generates a massive amount of data (>100MB). If
you want this feature, you must configure LLVM with the --enable-doxygen switch
and ensure that a modern version of doxygen (1.3.7 or later) is available in
your ``PATH``. You can download doxygen from `here
<http://www.stack.nl/~dimitri/doxygen/download.html#latestsrc>`_.

``preconditions``
-----------------

This utility target checks to see if the ``Makefile`` in the object directory is
older than the ``Makefile`` in the source directory and copies it if so. It also
reruns the ``configure`` script if that needs to be done and rebuilds the
``Makefile.config`` file similarly. Users may overload this target to ensure
that sanity checks are run *before* any building of targets as all the targets
depend on ``preconditions``.

``printvars``
-------------

This utility target just causes the LLVM makefiles to print out some of the
makefile variables so that you can double check how things are set.

``reconfigure``
---------------

This utility target will force a reconfigure of LLVM or your project. It simply
runs ``$(PROJ_OBJ_ROOT)/config.status --recheck`` to rerun the configuration
tests and rebuild the configured files. This isn't generally useful as the
makefiles will reconfigure themselves whenever its necessary.

``spotless``
------------

.. warning::

  Use with caution!

This utility target, only available when ``$(PROJ_OBJ_ROOT)`` is not the same as
``$(PROJ_SRC_ROOT)``, will completely clean the ``$(PROJ_OBJ_ROOT)`` directory
by removing its content entirely and reconfiguring the directory. This returns
the ``$(PROJ_OBJ_ROOT)`` directory to a completely fresh state. All content in
the directory except configured files and top-level makefiles will be lost.

``tags``
--------

This target will generate a ``TAGS`` file in the top-level source directory. It
is meant for use with emacs, XEmacs, or ViM. The TAGS file provides an index of
symbol definitions so that the editor can jump you to the definition
quickly.

``uninstall``
-------------

This target is the opposite of the ``install`` target. It removes the header,
library and executable files from the installation directories. Note that the
directories themselves are not removed because it is not guaranteed that LLVM is
the only thing installing there (e.g. ``--prefix=/usr``).

.. _variables:

Variables
=========

Variables are used to tell the LLVM Makefile System what to do and to obtain
information from it. Variables are also used internally by the LLVM Makefile
System. Variable names that contain only the upper case alphabetic letters and
underscore are intended for use by the end user. All other variables are
internal to the LLVM Makefile System and should not be relied upon nor
modified. The sections below describe how to use the LLVM Makefile
variables.

Control Variables
-----------------

Variables listed in the table below should be set *before* the inclusion of
`$(LEVEL)/Makefile.common`_.  These variables provide input to the LLVM make
system that tell it what to do for the current directory.

``BUILD_ARCHIVE``
    If set to any value, causes an archive (.a) library to be built.

``BUILT_SOURCES``
    Specifies a set of source files that are generated from other source
    files. These sources will be built before any other target processing to
    ensure they are present.

``BYTECODE_LIBRARY``
    If set to any value, causes a bitcode library (.bc) to be built.

``CONFIG_FILES``
    Specifies a set of configuration files to be installed.

``DEBUG_SYMBOLS``
    If set to any value, causes the build to include debugging symbols even in
    optimized objects, libraries and executables. This alters the flags
    specified to the compilers and linkers. Debugging isn't fun in an optimized
    build, but it is possible.

``DIRS``
    Specifies a set of directories, usually children of the current directory,
    that should also be made using the same goal. These directories will be
    built serially.

``DISABLE_AUTO_DEPENDENCIES``
    If set to any value, causes the makefiles to **not** automatically generate
    dependencies when running the compiler. Use of this feature is discouraged
    and it may be removed at a later date.

``ENABLE_OPTIMIZED``
    If set to 1, causes the build to generate optimized objects, libraries and
    executables. This alters the flags specified to the compilers and
    linkers. Generally debugging won't be a fun experience with an optimized
    build.

``ENABLE_PROFILING``
    If set to 1, causes the build to generate both optimized and profiled
    objects, libraries and executables. This alters the flags specified to the
    compilers and linkers to ensure that profile data can be collected from the
    tools built. Use the ``gprof`` tool to analyze the output from the profiled
    tools (``gmon.out``).

``DISABLE_ASSERTIONS``
    If set to 1, causes the build to disable assertions, even if building a
    debug or profile build.  This will exclude all assertion check code from the
    build. LLVM will execute faster, but with little help when things go
    wrong.

``EXPERIMENTAL_DIRS``
    Specify a set of directories that should be built, but if they fail, it
    should not cause the build to fail. Note that this should only be used
    temporarily while code is being written.

``EXPORTED_SYMBOL_FILE``
    Specifies the name of a single file that contains a list of the symbols to
    be exported by the linker. One symbol per line.

``EXPORTED_SYMBOL_LIST``
    Specifies a set of symbols to be exported by the linker.

``EXTRA_DIST``
    Specifies additional files that should be distributed with LLVM. All source
    files, all built sources, all Makefiles, and most documentation files will
    be automatically distributed. Use this variable to distribute any files that
    are not automatically distributed.

``KEEP_SYMBOLS``
    If set to any value, specifies that when linking executables the makefiles
    should retain debug symbols in the executable. Normally, symbols are
    stripped from the executable.

``LEVEL`` (required)
    Specify the level of nesting from the top level. This variable must be set
    in each makefile as it is used to find the top level and thus the other
    makefiles.

``LIBRARYNAME``
    Specify the name of the library to be built. (Required For Libraries)

``LINK_COMPONENTS``
    When specified for building a tool, the value of this variable will be
    passed to the ``llvm-config`` tool to generate a link line for the
    tool. Unlike ``USEDLIBS`` and ``LLVMLIBS``, not all libraries need to be
    specified. The ``llvm-config`` tool will figure out the library dependencies
    and add any libraries that are needed. The ``USEDLIBS`` variable can still
    be used in conjunction with ``LINK_COMPONENTS`` so that additional
    project-specific libraries can be linked with the LLVM libraries specified
    by ``LINK_COMPONENTS``.

.. _LINK_LIBS_IN_SHARED:

``LINK_LIBS_IN_SHARED``
    By default, shared library linking will ignore any libraries specified with
    the `LLVMLIBS`_ or `USEDLIBS`_. This prevents shared libs from including
    things that will be in the LLVM tool the shared library will be loaded
    into. However, sometimes it is useful to link certain libraries into your
    shared library and this option enables that feature.

.. _LLVMLIBS:

``LLVMLIBS``
    Specifies the set of libraries from the LLVM ``$(ObjDir)`` that will be
    linked into the tool or library.

``LOADABLE_MODULE``
    If set to any value, causes the shared library being built to also be a
    loadable module. Loadable modules can be opened with the dlopen() function
    and searched with dlsym (or the operating system's equivalent). Note that
    setting this variable without also setting ``SHARED_LIBRARY`` will have no
    effect.

.. _MODULE_NAME:

``MODULE_NAME``
    Specifies the name of a bitcode module to be created. A bitcode module can
    be specified in conjunction with other kinds of library builds or by
    itself. It constructs from the sources a single linked bitcode file.

``NO_INSTALL``
    Specifies that the build products of the directory should not be installed
    but should be built even if the ``install`` target is given.  This is handy
    for directories that build libraries or tools that are only used as part of
    the build process, such as code generators (e.g.  ``tblgen``).

``OPTIONAL_DIRS``
    Specify a set of directories that may be built, if they exist, but its not
    an error for them not to exist.

``PARALLEL_DIRS``
    Specify a set of directories to build recursively and in parallel if the
    ``-j`` option was used with ``make``.

.. _SHARED_LIBRARY:

``SHARED_LIBRARY``
    If set to any value, causes a shared library (``.so``) to be built in
    addition to any other kinds of libraries. Note that this option will cause
    all source files to be built twice: once with options for position
    independent code and once without. Use it only where you really need a
    shared library.

``SOURCES`` (optional)
    Specifies the list of source files in the current directory to be
    built. Source files of any type may be specified (programs, documentation,
    config files, etc.). If not specified, the makefile system will infer the
    set of source files from the files present in the current directory.

``SUFFIXES``
    Specifies a set of filename suffixes that occur in suffix match rules.  Only
    set this if your local ``Makefile`` specifies additional suffix match
    rules.

``TARGET``
    Specifies the name of the LLVM code generation target that the current
    directory builds. Setting this variable enables additional rules to build
    ``.inc`` files from ``.td`` files. 

.. _TESTSUITE:

``TESTSUITE``
    Specifies the directory of tests to run in ``llvm/test``.

``TOOLNAME``
    Specifies the name of the tool that the current directory should build.

``TOOL_VERBOSE``
    Implies ``VERBOSE`` and also tells each tool invoked to be verbose. This is
    handy when you're trying to see the sub-tools invoked by each tool invoked
    by the makefile. For example, this will pass ``-v`` to the GCC compilers
    which causes it to print out the command lines it uses to invoke sub-tools
    (compiler, assembler, linker).

.. _USEDLIBS:

``USEDLIBS``
    Specifies the list of project libraries that will be linked into the tool or
    library.

``VERBOSE``
    Tells the Makefile system to produce detailed output of what it is doing
    instead of just summary comments. This will generate a LOT of output.

Override Variables
------------------

Override variables can be used to override the default values provided by the
LLVM makefile system. These variables can be set in several ways:

* In the environment (e.g. setenv, export) --- not recommended.
* On the ``make`` command line --- recommended.
* On the ``configure`` command line.
* In the Makefile (only *after* the inclusion of `$(LEVEL)/Makefile.common`_).

The override variables are given below:

``AR`` (defaulted)
    Specifies the path to the ``ar`` tool.

``PROJ_OBJ_DIR``
    The directory into which the products of build rules will be placed.  This
    might be the same as `PROJ_SRC_DIR`_ but typically is not.

.. _PROJ_SRC_DIR:

``PROJ_SRC_DIR``
    The directory which contains the source files to be built.

``BUILD_EXAMPLES``
    If set to 1, build examples in ``examples`` and (if building Clang)
    ``tools/clang/examples`` directories.

``BZIP2`` (configured)
    The path to the ``bzip2`` tool.

``CC`` (configured)
    The path to the 'C' compiler.

``CFLAGS``
    Additional flags to be passed to the 'C' compiler.

``CXX``
    Specifies the path to the C++ compiler.

``CXXFLAGS``
    Additional flags to be passed to the C++ compiler.

``DATE`` (configured)
    Specifies the path to the ``date`` program or any program that can generate
    the current date and time on its standard output.

``DOT`` (configured)
    Specifies the path to the ``dot`` tool or ``false`` if there isn't one.

``ECHO`` (configured)
    Specifies the path to the ``echo`` tool for printing output.

``EXEEXT`` (configured)
    Provides the extension to be used on executables built by the makefiles.
    The value may be empty on platforms that do not use file extensions for
    executables (e.g. Unix).

``INSTALL`` (configured)
    Specifies the path to the ``install`` tool.

``LDFLAGS`` (configured)
    Allows users to specify additional flags to pass to the linker.

``LIBS`` (configured)
    The list of libraries that should be linked with each tool.

``LIBTOOL`` (configured)
    Specifies the path to the ``libtool`` tool. This tool is renamed ``mklib``
    by the ``configure`` script.

``LLVMAS`` (defaulted)
    Specifies the path to the ``llvm-as`` tool.

``LLVMCC``
    Specifies the path to the LLVM capable compiler.

``LLVMCXX``
    Specifies the path to the LLVM C++ capable compiler.

``LLVMGCC`` (defaulted)
    Specifies the path to the LLVM version of the GCC 'C' Compiler.

``LLVMGXX`` (defaulted)
    Specifies the path to the LLVM version of the GCC C++ Compiler.

``LLVMLD`` (defaulted)
    Specifies the path to the LLVM bitcode linker tool

``LLVM_OBJ_ROOT`` (configured)
    Specifies the top directory into which the output of the build is placed.

``LLVM_SRC_ROOT`` (configured)
    Specifies the top directory in which the sources are found.

``LLVM_TARBALL_NAME`` (configured)
    Specifies the name of the distribution tarball to create. This is configured
    from the name of the project and its version number.

``MKDIR`` (defaulted)
    Specifies the path to the ``mkdir`` tool that creates directories.

``ONLY_TOOLS``
    If set, specifies the list of tools to build.

``PLATFORMSTRIPOPTS``
    The options to provide to the linker to specify that a stripped (no symbols)
    executable should be built.

``RANLIB`` (defaulted)
    Specifies the path to the ``ranlib`` tool.

``RM`` (defaulted)
    Specifies the path to the ``rm`` tool.

``SED`` (defaulted)
    Specifies the path to the ``sed`` tool.

``SHLIBEXT`` (configured)
    Provides the filename extension to use for shared libraries.

``TBLGEN`` (defaulted)
    Specifies the path to the ``tblgen`` tool.

``TAR`` (defaulted)
    Specifies the path to the ``tar`` tool.

``ZIP`` (defaulted)
    Specifies the path to the ``zip`` tool.

Readable Variables
------------------

Variables listed in the table below can be used by the user's Makefile but
should not be changed. Changing the value will generally cause the build to go
wrong, so don't do it.

``bindir``
    The directory into which executables will ultimately be installed. This
    value is derived from the ``--prefix`` option given to ``configure``.

``BuildMode``
    The name of the type of build being performed: Debug, Release, or
    Profile.

``bytecode_libdir``
    The directory into which bitcode libraries will ultimately be installed.
    This value is derived from the ``--prefix`` option given to ``configure``.

``ConfigureScriptFLAGS``
    Additional flags given to the ``configure`` script when reconfiguring.

``DistDir``
    The *current* directory for which a distribution copy is being made.

.. _Echo:

``Echo``
    The LLVM Makefile System output command. This provides the ``llvm[n]``
    prefix and starts with ``@`` so the command itself is not printed by
    ``make``.

``EchoCmd``
    Same as `Echo`_ but without the leading ``@``.

``includedir``
    The directory into which include files will ultimately be installed.  This
    value is derived from the ``--prefix`` option given to ``configure``.

``libdir``
    The directory into which native libraries will ultimately be installed.
    This value is derived from the ``--prefix`` option given to
    ``configure``.

``LibDir``
    The configuration specific directory into which libraries are placed before
    installation.

``MakefileConfig``
    Full path of the ``Makefile.config`` file.

``MakefileConfigIn``
    Full path of the ``Makefile.config.in`` file.

``ObjDir``
    The configuration and directory specific directory where build objects
    (compilation results) are placed.

``SubDirs``
    The complete list of sub-directories of the current directory as
    specified by other variables.

``Sources``
    The complete list of source files.

``sysconfdir``
    The directory into which configuration files will ultimately be
    installed. This value is derived from the ``--prefix`` option given to
    ``configure``.

``ToolDir``
    The configuration specific directory into which executables are placed
    before they are installed.

``TopDistDir``
    The top most directory into which the distribution files are copied.

``Verb``
    Use this as the first thing on your build script lines to enable or disable
    verbose mode. It expands to either an ``@`` (quiet mode) or nothing (verbose
    mode).

Internal Variables
------------------

Variables listed below are used by the LLVM Makefile System and considered
internal. You should not use these variables under any circumstances.

.. code-block:: makefile

    Archive
    AR.Flags
    BaseNameSources
    BCCompile.C
    BCCompile.CXX
    BCLinkLib
    C.Flags
    Compile.C
    CompileCommonOpts
    Compile.CXX
    ConfigStatusScript
    ConfigureScript
    CPP.Flags
    CPP.Flags 
    CXX.Flags
    DependFiles
    DestArchiveLib
    DestBitcodeLib
    DestModule
    DestSharedLib
    DestTool
    DistAlways
    DistCheckDir
    DistCheckTop
    DistFiles
    DistName
    DistOther
    DistSources
    DistSubDirs
    DistTarBZ2
    DistTarGZip
    DistZip
    ExtraLibs
    FakeSources
    INCFiles
    InternalTargets
    LD.Flags
    LibName.A
    LibName.BC
    LibName.LA
    LibName.O
    LibTool.Flags
    Link
    LinkModule
    LLVMLibDir
    LLVMLibsOptions
    LLVMLibsPaths
    LLVMToolDir
    LLVMUsedLibs
    LocalTargets
    Module
    ObjectsBC
    ObjectsLO
    ObjectsO
    ObjMakefiles
    ParallelTargets
    PreConditions
    ProjLibsOptions
    ProjLibsPaths
    ProjUsedLibs
    Ranlib
    RecursiveTargets
    SrcMakefiles
    Strip
    StripWarnMsg
    TableGen
    TDFiles
    ToolBuildPath
    TopLevelTargets
    UserTargets
