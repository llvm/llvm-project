==================================================================
Getting Started with the LLVM System using Microsoft Visual Studio
==================================================================

.. contents::
   :local:


Overview
========
Welcome to LLVM on Windows! This document only covers LLVM on Windows using
Visual Studio, not mingw or cygwin. In order to get started, you first need to
know some basic information.

There are many different projects that compose LLVM. The first is the LLVM
suite. This contains all of the tools, libraries, and header files needed to
use LLVM. It contains an assembler, disassembler,
bitcode analyzer and bitcode optimizer. It also contains a test suite that can
be used to test the LLVM tools.

Another useful project on Windows is `Clang <http://clang.llvm.org/>`_.
Clang is a C family ([Objective]C/C++) compiler. Clang mostly works on
Windows, but does not currently understand all of the Microsoft extensions
to C and C++. Because of this, clang cannot parse the C++ standard library
included with Visual Studio, nor parts of the Windows Platform SDK. However,
most standard C programs do compile. Clang can be used to emit bitcode,
directly emit object files or even linked executables using Visual Studio's
``link.exe``.

The large LLVM test suite cannot be run on the Visual Studio port at this
time.

Most of the tools build and work.  ``bugpoint`` does build, but does
not work.

Additional information about the LLVM directory structure and tool chain
can be found on the main :doc:`GettingStarted` page.


Requirements
============
Before you begin to use the LLVM system, review the requirements given
below.  This may save you some trouble by knowing ahead of time what hardware
and software you will need.

Hardware
--------
Any system that can adequately run Visual Studio 2013 is fine. The LLVM
source tree and object files, libraries and executables will consume
approximately 3GB.

Software
--------
You will need Visual Studio 2013 or higher.

You will also need the `CMake <http://www.cmake.org/>`_ build system since it
generates the project files you will use to build with.

If you would like to run the LLVM tests you will need `Python
<http://www.python.org/>`_. Version 2.7 and newer are known to work. You will
need `GnuWin32 <http://gnuwin32.sourceforge.net/>`_ tools, too.

Do not install the LLVM directory tree into a path containing spaces (e.g.
``C:\Documents and Settings\...``) as the configure step will fail.


Getting Started
===============
Here's the short story for getting up and running quickly with LLVM:

1. Read the documentation.
2. Seriously, read the documentation.
3. Remember that you were warned twice about reading the documentation.
4. Get the Source Code

   * With the distributed files:

      1. ``cd <where-you-want-llvm-to-live>``
      2. ``gunzip --stdout llvm-VERSION.tar.gz | tar -xvf -``
         (*or use WinZip*)
      3. ``cd llvm``

   * With anonymous Subversion access:

      1. ``cd <where-you-want-llvm-to-live>``
      2. ``svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm``
      3. ``cd llvm``

5. Use `CMake <http://www.cmake.org/>`_ to generate up-to-date project files:

   * Once CMake is installed then the simplest way is to just start the
     CMake GUI, select the directory where you have LLVM extracted to, and
     the default options should all be fine.  One option you may really
     want to change, regardless of anything else, might be the
     ``CMAKE_INSTALL_PREFIX`` setting to select a directory to INSTALL to
     once compiling is complete, although installation is not mandatory for
     using LLVM.  Another important option is ``LLVM_TARGETS_TO_BUILD``,
     which controls the LLVM target architectures that are included on the
     build.
   * See the :doc:`LLVM CMake guide <CMake>` for detailed information about
     how to configure the LLVM build.
   * CMake generates project files for all build types. To select a specific
     build type, use the Configuration manager from the VS IDE or the 
     ``/property:Configuration`` command line option when using MSBuild.

6. Start Visual Studio

   * In the directory you created the project files will have an ``llvm.sln``
     file, just double-click on that to open Visual Studio.

7. Build the LLVM Suite:

   * The projects may still be built individually, but to build them all do
     not just select all of them in batch build (as some are meant as
     configuration projects), but rather select and build just the
     ``ALL_BUILD`` project to build everything, or the ``INSTALL`` project,
     which first builds the ``ALL_BUILD`` project, then installs the LLVM
     headers, libs, and other useful things to the directory set by the
     ``CMAKE_INSTALL_PREFIX`` setting when you first configured CMake.
   * The Fibonacci project is a sample program that uses the JIT. Modify the
     project's debugging properties to provide a numeric command line argument
     or run it from the command line.  The program will print the
     corresponding fibonacci value.

8. Test LLVM in Visual Studio:

   * If ``%PATH%`` does not contain GnuWin32, you may specify
     ``LLVM_LIT_TOOLS_DIR`` on CMake for the path to GnuWin32.
   * You can run LLVM tests by merely building the project "check". The test
     results will be shown in the VS output window.

9. Test LLVM on the command line:

   * The LLVM tests can be run by changing directory to the llvm source
     directory and running:

     .. code-block:: bat

        C:\..\llvm> python ..\build\bin\llvm-lit --param build_config=Win32 --param build_mode=Debug --param llvm_site_config=../build/test/lit.site.cfg test

     This example assumes that Python is in your PATH variable, you
     have built a Win32 Debug version of llvm with a standard out of
     line build. You should not see any unexpected failures, but will
     see many unsupported tests and expected failures.

     A specific test or test directory can be run with:

     .. code-block:: bat

        C:\..\llvm> python ..\build\bin\llvm-lit --param build_config=Win32 --param build_mode=Debug --param llvm_site_config=../build/test/lit.site.cfg test/path/to/test


An Example Using the LLVM Tool Chain
====================================

1. First, create a simple C file, name it '``hello.c``':

   .. code-block:: c

      #include <stdio.h>
      int main() {
        printf("hello world\n");
        return 0;
      }

2. Next, compile the C file into an LLVM bitcode file:

   .. code-block:: bat

      C:\..> clang -c hello.c -emit-llvm -o hello.bc

   This will create the result file ``hello.bc`` which is the LLVM bitcode
   that corresponds the compiled program and the library facilities that
   it required.  You can execute this file directly using ``lli`` tool,
   compile it to native assembly with the ``llc``, optimize or analyze it
   further with the ``opt`` tool, etc.

   Alternatively you can directly output an executable with clang with:

   .. code-block:: bat

      C:\..> clang hello.c -o hello.exe

   The ``-o hello.exe`` is required because clang currently outputs ``a.out``
   when neither ``-o`` nor ``-c`` are given.

3. Run the program using the just-in-time compiler:

   .. code-block:: bat

      C:\..> lli hello.bc

4. Use the ``llvm-dis`` utility to take a look at the LLVM assembly code:

   .. code-block:: bat

      C:\..> llvm-dis < hello.bc | more

5. Compile the program to object code using the LLC code generator:

   .. code-block:: bat

      C:\..> llc -filetype=obj hello.bc

6. Link to binary using Microsoft link:

   .. code-block:: bat

      C:\..> link hello.obj -defaultlib:libcmt

7. Execute the native code program:

   .. code-block:: bat

      C:\..> hello.exe


Common Problems
===============
If you are having problems building or using LLVM, or if you have any other
general questions about LLVM, please consult the :doc:`Frequently Asked Questions
<FAQ>` page.


Links
=====
This document is just an **introduction** to how to use LLVM to do some simple
things... there are many more interesting and complicated things that you can
do that aren't documented here (but we'll gladly accept a patch if you want to
write something up!).  For more information about LLVM, check out:

* `LLVM homepage <http://llvm.org/>`_
* `LLVM doxygen tree <http://llvm.org/doxygen/>`_

