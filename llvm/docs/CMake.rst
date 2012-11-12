.. _building-with-cmake:

========================
Building LLVM with CMake
========================

.. contents::
   :local:

Introduction
============

`CMake <http://www.cmake.org/>`_ is a cross-platform build-generator tool. CMake
does not build the project, it generates the files needed by your build tool
(GNU make, Visual Studio, etc) for building LLVM.

If you are really anxious about getting a functional LLVM build, go to the
`Quick start`_ section. If you are a CMake novice, start on `Basic CMake usage`_
and then go back to the `Quick start`_ once you know what you are doing. The
`Options and variables`_ section is a reference for customizing your build. If
you already have experience with CMake, this is the recommended starting point.

.. _Quick start:

Quick start
===========

We use here the command-line, non-interactive CMake interface.

#. `Download <http://www.cmake.org/cmake/resources/software.html>`_ and install
   CMake. Version 2.8 is the minimum required.

#. Open a shell. Your development tools must be reachable from this shell
   through the PATH environment variable.

#. Create a directory for containing the build. It is not supported to build
   LLVM on the source directory. cd to this directory:

   .. code-block:: bash

     $ mkdir mybuilddir
     $ cd mybuilddir

#. Execute this command on the shell replacing `path/to/llvm/source/root` with
   the path to the root of your LLVM source tree:

   .. code-block:: bash

     $ cmake path/to/llvm/source/root

   CMake will detect your development environment, perform a series of test and
   generate the files required for building LLVM. CMake will use default values
   for all build parameters. See the `Options and variables`_ section for
   fine-tuning your build

   This can fail if CMake can't detect your toolset, or if it thinks that the
   environment is not sane enough. On this case make sure that the toolset that
   you intend to use is the only one reachable from the shell and that the shell
   itself is the correct one for you development environment. CMake will refuse
   to build MinGW makefiles if you have a POSIX shell reachable through the PATH
   environment variable, for instance. You can force CMake to use a given build
   tool, see the `Usage`_ section.

.. _Basic CMake usage:
.. _Usage:

Basic CMake usage
=================

This section explains basic aspects of CMake, mostly for explaining those
options which you may need on your day-to-day usage.

CMake comes with extensive documentation in the form of html files and on the
cmake executable itself. Execute ``cmake --help`` for further help options.

CMake requires to know for which build tool it shall generate files (GNU make,
Visual Studio, Xcode, etc). If not specified on the command line, it tries to
guess it based on you environment. Once identified the build tool, CMake uses
the corresponding *Generator* for creating files for your build tool. You can
explicitly specify the generator with the command line option ``-G "Name of the
generator"``. For knowing the available generators on your platform, execute

.. code-block:: bash

  $ cmake --help

This will list the generator's names at the end of the help text. Generator's
names are case-sensitive. Example:

.. code-block:: bash

  $ cmake -G "Visual Studio 9 2008" path/to/llvm/source/root

For a given development platform there can be more than one adequate
generator. If you use Visual Studio "NMake Makefiles" is a generator you can use
for building with NMake. By default, CMake chooses the more specific generator
supported by your development environment. If you want an alternative generator,
you must tell this to CMake with the ``-G`` option.

.. todo::

  Explain variables and cache. Move explanation here from #options section.

.. _Options and variables:

Options and variables
=====================

Variables customize how the build will be generated. Options are boolean
variables, with possible values ON/OFF. Options and variables are defined on the
CMake command line like this:

.. code-block:: bash

  $ cmake -DVARIABLE=value path/to/llvm/source

You can set a variable after the initial CMake invocation for changing its
value. You can also undefine a variable:

.. code-block:: bash

  $ cmake -UVARIABLE path/to/llvm/source

Variables are stored on the CMake cache. This is a file named ``CMakeCache.txt``
on the root of the build directory. Do not hand-edit it.

Variables are listed here appending its type after a colon. It is correct to
write the variable and the type on the CMake command line:

.. code-block:: bash

  $ cmake -DVARIABLE:TYPE=value path/to/llvm/source

Frequently-used CMake variables
-------------------------------

Here are listed some of the CMake variables that are used often, along with a
brief explanation and LLVM-specific notes. For full documentation, check the
CMake docs or execute ``cmake --help-variable VARIABLE_NAME``.

**CMAKE_BUILD_TYPE**:STRING
  Sets the build type for ``make`` based generators. Possible values are
  Release, Debug, RelWithDebInfo and MinSizeRel. On systems like Visual Studio
  the user sets the build type with the IDE settings.

**CMAKE_INSTALL_PREFIX**:PATH
  Path where LLVM will be installed if "make install" is invoked or the
  "INSTALL" target is built.

**LLVM_LIBDIR_SUFFIX**:STRING
  Extra suffix to append to the directory where libraries are to be
  installed. On a 64-bit architecture, one could use ``-DLLVM_LIBDIR_SUFFIX=64``
  to install libraries to ``/usr/lib64``.

**CMAKE_C_FLAGS**:STRING
  Extra flags to use when compiling C source files.

**CMAKE_CXX_FLAGS**:STRING
  Extra flags to use when compiling C++ source files.

**BUILD_SHARED_LIBS**:BOOL
  Flag indicating is shared libraries will be built. Its default value is
  OFF. Shared libraries are not supported on Windows and not recommended in the
  other OSes.

.. _LLVM-specific variables:

LLVM-specific variables
-----------------------

**LLVM_TARGETS_TO_BUILD**:STRING
  Semicolon-separated list of targets to build, or *all* for building all
  targets. Case-sensitive. For Visual C++ defaults to *X86*. On the other cases
  defaults to *all*. Example: ``-DLLVM_TARGETS_TO_BUILD="X86;PowerPC"``.

**LLVM_BUILD_TOOLS**:BOOL
  Build LLVM tools. Defaults to ON. Targets for building each tool are generated
  in any case. You can build an tool separately by invoking its target. For
  example, you can build *llvm-as* with a makefile-based system executing *make
  llvm-as* on the root of your build directory.

**LLVM_INCLUDE_TOOLS**:BOOL
  Generate build targets for the LLVM tools. Defaults to ON. You can use that
  option for disabling the generation of build targets for the LLVM tools.

**LLVM_BUILD_EXAMPLES**:BOOL
  Build LLVM examples. Defaults to OFF. Targets for building each example are
  generated in any case. See documentation for *LLVM_BUILD_TOOLS* above for more
  details.

**LLVM_INCLUDE_EXAMPLES**:BOOL
  Generate build targets for the LLVM examples. Defaults to ON. You can use that
  option for disabling the generation of build targets for the LLVM examples.

**LLVM_BUILD_TESTS**:BOOL
  Build LLVM unit tests. Defaults to OFF. Targets for building each unit test
  are generated in any case. You can build a specific unit test with the target
  *UnitTestNameTests* (where at this time *UnitTestName* can be ADT, Analysis,
  ExecutionEngine, JIT, Support, Transform, VMCore; see the subdirectories of
  *unittests* for an updated list.) It is possible to build all unit tests with
  the target *UnitTests*.

**LLVM_INCLUDE_TESTS**:BOOL
  Generate build targets for the LLVM unit tests. Defaults to ON. You can use
  that option for disabling the generation of build targets for the LLVM unit
  tests.

**LLVM_APPEND_VC_REV**:BOOL
  Append version control revision info (svn revision number or git revision id)
  to LLVM version string (stored in the PACKAGE_VERSION macro). For this to work
  cmake must be invoked before the build. Defaults to OFF.

**LLVM_ENABLE_THREADS**:BOOL
  Build with threads support, if available. Defaults to ON.

**LLVM_ENABLE_ASSERTIONS**:BOOL
  Enables code assertions. Defaults to OFF if and only if ``CMAKE_BUILD_TYPE``
  is *Release*.

**LLVM_ENABLE_PIC**:BOOL
  Add the ``-fPIC`` flag for the compiler command-line, if the compiler supports
  this flag. Some systems, like Windows, do not need this flag. Defaults to ON.

**LLVM_ENABLE_WARNINGS**:BOOL
  Enable all compiler warnings. Defaults to ON.

**LLVM_ENABLE_PEDANTIC**:BOOL
  Enable pedantic mode. This disable compiler specific extensions, is
  possible. Defaults to ON.

**LLVM_ENABLE_WERROR**:BOOL
  Stop and fail build, if a compiler warning is triggered. Defaults to OFF.

**LLVM_BUILD_32_BITS**:BOOL
  Build 32-bits executables and libraries on 64-bits systems. This option is
  available only on some 64-bits unix systems. Defaults to OFF.

**LLVM_TARGET_ARCH**:STRING
  LLVM target to use for native code generation. This is required for JIT
  generation. It defaults to "host", meaning that it shall pick the architecture
  of the machine where LLVM is being built. If you are cross-compiling, set it
  to the target architecture name.

**LLVM_TABLEGEN**:STRING
  Full path to a native TableGen executable (usually named ``tblgen``). This is
  intended for cross-compiling: if the user sets this variable, no native
  TableGen will be created.

**LLVM_LIT_ARGS**:STRING
  Arguments given to lit.  ``make check`` and ``make clang-test`` are affected.
  By default, ``'-sv --no-progress-bar'`` on Visual C++ and Xcode, ``'-sv'`` on
  others.

**LLVM_LIT_TOOLS_DIR**:PATH
  The path to GnuWin32 tools for tests. Valid on Windows host.  Defaults to "",
  then Lit seeks tools according to %PATH%.  Lit can find tools(eg. grep, sort,
  &c) on LLVM_LIT_TOOLS_DIR at first, without specifying GnuWin32 to %PATH%.

**LLVM_ENABLE_FFI**:BOOL
  Indicates whether LLVM Interpreter will be linked with Foreign Function
  Interface library. If the library or its headers are installed on a custom
  location, you can set the variables FFI_INCLUDE_DIR and
  FFI_LIBRARY_DIR. Defaults to OFF.

**LLVM_EXTERNAL_{CLANG,LLD,POLLY}_SOURCE_DIR**:PATH
  Path to ``{Clang,lld,Polly}``\'s source directory. Defaults to
  ``tools/{clang,lld,polly}``. ``{Clang,lld,Polly}`` will not be built when it
  is empty or it does not point valid path.

**LLVM_USE_OPROFILE**:BOOL
  Enable building OProfile JIT support. Defaults to OFF

**LLVM_USE_INTEL_JITEVENTS**:BOOL
  Enable building support for Intel JIT Events API. Defaults to OFF

Executing the test suite
========================

Testing is performed when the *check* target is built. For instance, if you are
using makefiles, execute this command while on the top level of your build
directory:

.. code-block:: bash

  $ make check

On Visual Studio, you may run tests to build the project "check".

Cross compiling
===============

See `this wiki page <http://www.vtk.org/Wiki/CMake_Cross_Compiling>`_ for
generic instructions on how to cross-compile with CMake. It goes into detailed
explanations and may seem daunting, but it is not. On the wiki page there are
several examples including toolchain files. Go directly to `this section
<http://www.vtk.org/Wiki/CMake_Cross_Compiling#Information_how_to_set_up_various_cross_compiling_toolchains>`_
for a quick solution.

Also see the `LLVM-specific variables`_ section for variables used when
cross-compiling.

Embedding LLVM in your project
==============================

The most difficult part of adding LLVM to the build of a project is to determine
the set of LLVM libraries corresponding to the set of required LLVM
features. What follows is an example of how to obtain this information:

.. code-block:: cmake

  # A convenience variable:
  set(LLVM_ROOT "" CACHE PATH "Root of LLVM install.")

  # A bit of a sanity check:
  if( NOT EXISTS ${LLVM_ROOT}/include/llvm )
  message(FATAL_ERROR "LLVM_ROOT (${LLVM_ROOT}) is not a valid LLVM install")
  endif()

  # We incorporate the CMake features provided by LLVM:
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_ROOT}/share/llvm/cmake")
  include(LLVMConfig)

  # Now set the header and library paths:
  include_directories( ${LLVM_INCLUDE_DIRS} )
  link_directories( ${LLVM_LIBRARY_DIRS} )
  add_definitions( ${LLVM_DEFINITIONS} )

  # Let's suppose we want to build a JIT compiler with support for
  # binary code (no interpreter):
  llvm_map_components_to_libraries(REQ_LLVM_LIBRARIES jit native)

  # Finally, we link the LLVM libraries to our executable:
  target_link_libraries(mycompiler ${REQ_LLVM_LIBRARIES})

This assumes that LLVM_ROOT points to an install of LLVM. The procedure works
too for uninstalled builds although we need to take care to add an
`include_directories` for the location of the headers on the LLVM source
directory (if we are building out-of-source.)

Alternativaly, you can utilize CMake's ``find_package`` functionality. Here is
an equivalent variant of snippet shown above:

.. code-block:: cmake

  find_package(LLVM)

  if( NOT LLVM_FOUND )
    message(FATAL_ERROR "LLVM package can't be found. Set CMAKE_PREFIX_PATH variable to LLVM's installation prefix.")
  endif()

  include_directories( ${LLVM_INCLUDE_DIRS} )
  link_directories( ${LLVM_LIBRARY_DIRS} )

  llvm_map_components_to_libraries(REQ_LLVM_LIBRARIES jit native)

  target_link_libraries(mycompiler ${REQ_LLVM_LIBRARIES})

Developing LLVM pass out of source
----------------------------------

It is possible to develop LLVM passes against installed LLVM.  An example of
project layout provided below:

.. code-block:: bash

  <project dir>/
      |
      CMakeLists.txt
      <pass name>/
          |
          CMakeLists.txt
          Pass.cpp
          ...

Contents of ``<project dir>/CMakeLists.txt``:

.. code-block:: cmake

  find_package(LLVM)

  # Define add_llvm_* macro's.
  include(AddLLVM)

  add_definitions(${LLVM_DEFINITIONS})
  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIRS})

  add_subdirectory(<pass name>)

Contents of ``<project dir>/<pass name>/CMakeLists.txt``:

.. code-block:: cmake

  add_llvm_loadable_module(LLVMPassname
    Pass.cpp
    )

When you are done developing your pass, you may wish to integrate it
into LLVM source tree. You can achieve it in two easy steps:

#. Copying ``<pass name>`` folder into ``<LLVM root>/lib/Transform`` directory.

#. Adding ``add_subdirectory(<pass name>)`` line into
   ``<LLVM root>/lib/Transform/CMakeLists.txt``.

Compiler/Platform specific topics
=================================

Notes for specific compilers and/or platforms.

Microsoft Visual C++
--------------------

**LLVM_COMPILER_JOBS**:STRING
  Specifies the maximum number of parallell compiler jobs to use per project
  when building with msbuild or Visual Studio. Only supported for Visual Studio
  2008 and Visual Studio 2010 CMake generators. 0 means use all
  processors. Default is 0.
