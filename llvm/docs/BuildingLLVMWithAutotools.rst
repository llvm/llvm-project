====================================
Building LLVM With Autotools
====================================

.. contents::
   :local:

.. warning::

    Building LLVM with autoconf is deprecated as of 3.8. The autoconf build
    system will be removed in 3.9. Please migrate to using CMake. For more
    information see: `Building LLVM with CMake <CMake.html>`_

Overview
========

This document details how to use the LLVM autotools based build system to
configure and build LLVM from source. The normal developer process using CMake
is detailed `here <GettingStarted.html#check-here>`_.

A Quick Summary
---------------

#. Configure and build LLVM and Clang:

   * ``cd where-you-want-to-build-llvm``
   * ``mkdir build`` (for building without polluting the source dir)
   * ``cd build``
   * ``../llvm/configure [options]``
     Some common options:

     * ``--prefix=directory`` --- Specify for *directory* the full pathname of
       where you want the LLVM tools and libraries to be installed (default
       ``/usr/local``).

     * ``--enable-optimized`` --- Compile with optimizations enabled (default
       is NO).

     * ``--enable-assertions`` --- Compile with assertion checks enabled
       (default is YES).

   * ``make [-j]`` --- The ``-j`` specifies the number of jobs (commands) to run
     simultaneously.  This builds both LLVM and Clang for Debug+Asserts mode.
     The ``--enable-optimized`` configure option is used to specify a Release
     build.

   * ``make check-all`` --- This run the regression tests to ensure everything
     is in working order.

   * If you get an "internal compiler error (ICE)" or test failures, see
     `here <GettingStarted.html#check-here>`_.

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
|            | ``configure`` will check ``PATH`` for ``clang`` and GCC C |
|            | compilers (in this order).  Use this variable to override |
|            | ``configure``\'s  default behavior.                       |
+------------+-----------------------------------------------------------+
| CXX        | Tells ``configure`` which C++ compiler to use.  By        |
|            | default, ``configure`` will check ``PATH`` for            |
|            | ``clang++`` and GCC C++ compilers (in this order).  Use   |
|            | this variable to override  ``configure``'s default        |
|            | behavior.                                                 |
+------------+-----------------------------------------------------------+

The following options can be used to set or enable LLVM specific options:

``--enable-optimized``

  Enables optimized compilation (debugging symbols are removed and GCC
  optimization flags are enabled). Note that this is the default setting if you
  are using the LLVM distribution. The default behavior of a Subversion
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
  The "host" target is selected as the target of the build host. You can also
  specify a comma separated list of target names that you want available in llc.
  The target names use all lower case. The current set of targets is:

    ``aarch64, arm, arm64, cpp, hexagon, mips, mipsel, mips64, mips64el, msp430,
    powerpc, nvptx, r600, sparc, systemz, x86, x86_64, xcore``.

``--enable-doxygen``

  Look for the doxygen program and enable construction of doxygen based
  documentation from the source code. This is disabled by default because
  generating the documentation can take a long time and producess 100s of
  megabytes of output.

To configure LLVM, follow these steps:

#. Change directory into the object root directory:

   .. code-block:: console

     % cd OBJ_ROOT

#. Run the ``configure`` script located in the LLVM source tree:

   .. code-block:: console

     % $LLVM_SRC_DIR/configure --prefix=/install/path [other options]

Compiling the LLVM Suite Source Code
------------------------------------

Once you have configured LLVM, you can build it.  There are three types of
builds:

Debug Builds

  These builds are the default when one is using a Subversion checkout and
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

.. code-block:: console

  % gmake

If the build fails, please `check here <GettingStarted.html#check-here>`_
to see if you are using a version of GCC that is known not to compile LLVM.

If you have multiple processors in your machine, you may wish to use some of the
parallel build options provided by GNU Make.  For example, you could use the
command:

.. code-block:: console

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
  under ``$PREFIX``, specified with ``$LLVM_SRC_DIR/configure --prefix=[dir]``, which
  defaults to ``/usr/local``.

``gmake -C runtime install-bytecode``

  Assuming you built LLVM into $OBJDIR, when this command is run, it will
  install bitcode libraries into the GCC front end's bitcode library directory.
  If you need to update your bitcode libraries, this is the target to use once
  you've built them.

Please see the `CMake <CMake.html>`_ documentation for further details on
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

This does not apply to building the documentation.
LLVM's (non-Doxygen) documentation is produced with the
`Sphinx <http://sphinx-doc.org/>`_ documentation generation system.
There are some HTML documents that have not yet been converted to the new
system (which uses the easy-to-read and easy-to-write
`reStructuredText <http://sphinx-doc.org/rest.html>`_ plaintext markup
language).
The generated documentation is built in the ``$LLVM_SRC_DIR/docs`` directory using
a special makefile.
For instructions on how to install Sphinx, see
`Sphinx Introduction for LLVM Developers
<http://lld.llvm.org/sphinx_intro.html>`_.
After following the instructions there for installing Sphinx, build the LLVM
HTML documentation by doing the following:

.. code-block:: console

  $ cd $LLVM_SRC_DIR/docs
  $ make -f Makefile.sphinx

This creates a ``_build/html`` sub-directory with all of the HTML files, not
just the generated ones.
This directory corresponds to ``llvm.org/docs``.
For example, ``_build/html/SphinxQuickstartTemplate.html`` corresponds to
``llvm.org/docs/SphinxQuickstartTemplate.html``.
The :doc:`SphinxQuickstartTemplate` is useful when creating a new document.

Cross-Compiling LLVM
--------------------

It is possible to cross-compile LLVM itself. That is, you can create LLVM
executables and libraries to be hosted on a platform different from the platform
where they are built (a Canadian Cross build). To configure a cross-compile,
supply the configure script with ``--build`` and ``--host`` options that are
different. The values of these options must be legal target triples that your
GCC compiler supports.

The result of such a build is executables that are not runnable on on the build
host (--build option) but can be executed on the compile host (--host option).

Check :doc:`HowToCrossCompileLLVM` and `Clang docs on how to cross-compile in general
<http://clang.llvm.org/docs/CrossCompilation.html>`_ for more information
about cross-compiling.

The Location of LLVM Object Files
---------------------------------

The LLVM build system is capable of sharing a single LLVM source tree among
several LLVM builds.  Hence, it is possible to build LLVM for several different
platforms or configurations using the same source tree.

This is accomplished in the typical autoconf manner:

* Change directory to where the LLVM object files should live:

  .. code-block:: console

    % cd OBJ_ROOT

* Run the ``configure`` script found in the LLVM source directory:

  .. code-block:: console

    % $LLVM_SRC_DIR/configure

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
