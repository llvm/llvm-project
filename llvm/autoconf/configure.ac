dnl === configure.ac --------------------------------------------------------===
dnl                     The LLVM Compiler Infrastructure
dnl
dnl This file is distributed under the University of Illinois Open Source
dnl License. See LICENSE.TXT for details.
dnl
dnl===-----------------------------------------------------------------------===
dnl This is the LLVM configuration script. It is processed by the autoconf
dnl program to produce a script named configure. This script contains the
dnl configuration checks that LLVM needs in order to support multiple platforms.
dnl This file is composed of 10 sections per the recommended organization of
dnl autoconf input defined in the autoconf documentation. As this file evolves,
dnl please keep the various types of checks within their sections. The sections
dnl are as follows:
dnl
dnl SECTION 1: Initialization & Setup
dnl SECTION 2: Architecture, target, and host checks
dnl SECTION 3: Command line arguments for the configure script.
dnl SECTION 4: Check for programs we need and that they are the right version
dnl SECTION 5: Check for libraries
dnl SECTION 6: Check for header files
dnl SECTION 7: Check for types and structures
dnl SECTION 8: Check for specific functions needed
dnl SECTION 9: Additional checks, variables, etc.
dnl SECTION 10: Specify the output files and generate it
dnl
dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 1: Initialization & Setup
dnl===
dnl===-----------------------------------------------------------------------===
dnl Initialize autoconf and define the package name, version number and
dnl address for reporting bugs.
AC_INIT([LLVM],[3.1svn],[http://llvm.org/bugs/])
AC_DEFINE([LLVM_VERSION_MAJOR], [3], [Major version of the LLVM API])
AC_DEFINE([LLVM_VERSION_MINOR], [1], [Minor version of the LLVM API])

dnl Provide a copyright substitution and ensure the copyright notice is included
dnl in the output of --version option of the generated configure script.
AC_SUBST(LLVM_COPYRIGHT,["Copyright (c) 2003-2012 University of Illinois at Urbana-Champaign."])
AC_COPYRIGHT([Copyright (c) 2003-2012 University of Illinois at Urbana-Champaign.])

dnl Indicate that we require autoconf 2.60 or later.
AC_PREREQ(2.60)

dnl Verify that the source directory is valid. This makes sure that we are
dnl configuring LLVM and not some other package (it validates --srcdir argument)
AC_CONFIG_SRCDIR([lib/VMCore/Module.cpp])

dnl Place all of the extra autoconf files into the config subdirectory. Tell
dnl various tools where the m4 autoconf macros are.
AC_CONFIG_AUX_DIR([autoconf])

dnl Quit if the source directory has already been configured.
dnl NOTE: This relies upon undocumented autoconf behavior.
if test ${srcdir} != "." ; then
  if test -f ${srcdir}/include/llvm/Config/config.h ; then
    AC_MSG_ERROR([Already configured in ${srcdir}])
  fi
fi

dnl We need to check for the compiler up here to avoid anything else
dnl starting with a different one.
AC_PROG_CC(clang llvm-gcc gcc)
AC_PROG_CXX(clang++ llvm-g++ g++)
AC_PROG_CPP

dnl Configure all of the projects present in our source tree. While we could
dnl just AC_CONFIG_SUBDIRS on the set of directories in projects that have a
dnl configure script, that usage of the AC_CONFIG_SUBDIRS macro is deprecated.
dnl Instead we match on the known projects.

dnl
dnl One tricky part of doing this is that some projects depend upon other
dnl projects.  For example, several projects rely upon the LLVM test suite.
dnl We want to configure those projects first so that their object trees are
dnl created before running the configure scripts of projects that depend upon
dnl them.
dnl

dnl Several projects use llvm-gcc, so configure that first
if test -d ${srcdir}/projects/llvm-gcc ; then
  AC_CONFIG_SUBDIRS([projects/llvm-gcc])
fi

dnl Several projects use the LLVM test suite, so configure it next.
if test -d ${srcdir}/projects/test-suite ; then
  AC_CONFIG_SUBDIRS([projects/test-suite])
fi

dnl llvm-test is the old name of the test-suite, kept here for backwards
dnl compatibility
if test -d ${srcdir}/projects/llvm-test ; then
  AC_CONFIG_SUBDIRS([projects/llvm-test])
fi

dnl Some projects use poolalloc; configure that next
if test -d ${srcdir}/projects/poolalloc ; then
  AC_CONFIG_SUBDIRS([projects/poolalloc])
fi

if test -d ${srcdir}/projects/llvm-poolalloc ; then
  AC_CONFIG_SUBDIRS([projects/llvm-poolalloc])
fi

dnl Check for all other projects
for i in `ls ${srcdir}/projects`
do
  if test -d ${srcdir}/projects/${i} ; then
    case ${i} in
      sample)       AC_CONFIG_SUBDIRS([projects/sample])    ;;
      privbracket)  AC_CONFIG_SUBDIRS([projects/privbracket]) ;;
      llvm-stacker) AC_CONFIG_SUBDIRS([projects/llvm-stacker]) ;;
      llvm-reopt)   AC_CONFIG_SUBDIRS([projects/llvm-reopt]);;
      llvm-java)    AC_CONFIG_SUBDIRS([projects/llvm-java]) ;;
      llvm-tv)      AC_CONFIG_SUBDIRS([projects/llvm-tv])   ;;
      safecode)     AC_CONFIG_SUBDIRS([projects/safecode]) ;;
      llvm-kernel)  AC_CONFIG_SUBDIRS([projects/llvm-kernel]) ;;
      compiler-rt)       ;;
      llvm-gcc)       ;;
      test-suite)     ;;
      llvm-test)      ;;
      poolalloc)      ;;
      llvm-poolalloc) ;;
      *)
        AC_MSG_WARN([Unknown project (${i}) won't be configured automatically])
        ;;
    esac
  fi
done

dnl Disable the build of polly, even if it is checked out into tools/polly.
AC_ARG_ENABLE(polly,
              AS_HELP_STRING([--enable-polly],
                             [Use polly if available (default is YES)]),,
                             enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_POLLY,[1]) ;;
  no)  AC_SUBST(ENABLE_POLLY,[0]) ;;
  default) AC_SUBST(ENABLE_POLLY,[1]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-polly. Use "yes" or "no"]) ;;
esac


dnl Check if polly is checked out into tools/polly and configure it if
dnl available.
if (test -d ${srcdir}/tools/polly) && (test $ENABLE_POLLY -eq 1) ; then
  AC_SUBST(LLVM_HAS_POLLY,1)
  AC_CONFIG_SUBDIRS([tools/polly])
fi

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 2: Architecture, target, and host checks
dnl===
dnl===-----------------------------------------------------------------------===

dnl Check the target for which we're compiling and the host that will do the
dnl compilations. This will tell us which LLVM compiler will be used for
dnl compiling SSA into object code. This needs to be done early because
dnl following tests depend on it.
AC_CANONICAL_TARGET

dnl Determine the platform type and cache its value. This helps us configure
dnl the System library to the correct build platform.
AC_CACHE_CHECK([type of operating system we're going to host on],
               [llvm_cv_os_type],
[case $host in
  *-*-aix*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="AIX"
    llvm_cv_platform_type="Unix" ;;
  *-*-irix*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="IRIX"
    llvm_cv_platform_type="Unix" ;;
  *-*-cygwin*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Cygwin"
    llvm_cv_platform_type="Unix" ;;
  *-*-darwin*)
    llvm_cv_link_all_option="-Wl,-all_load"
    llvm_cv_no_link_all_option="-Wl,-noall_load"
    llvm_cv_os_type="Darwin"
    llvm_cv_platform_type="Unix" ;;
  *-*-minix*)
    llvm_cv_link_all_option="-Wl,-all_load"
    llvm_cv_no_link_all_option="-Wl,-noall_load"
    llvm_cv_os_type="Minix"
    llvm_cv_platform_type="Unix" ;;
  *-*-freebsd* | *-*-kfreebsd-gnu)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="FreeBSD"
    llvm_cv_platform_type="Unix" ;;
  *-*-openbsd*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="OpenBSD"
    llvm_cv_platform_type="Unix" ;;
  *-*-netbsd*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="NetBSD"
    llvm_cv_platform_type="Unix" ;;
  *-*-dragonfly*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="DragonFly"
    llvm_cv_platform_type="Unix" ;;
  *-*-hpux*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="HP-UX"
    llvm_cv_platform_type="Unix" ;;
  *-*-interix*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Interix"
    llvm_cv_platform_type="Unix" ;;
  *-*-linux*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Linux"
    llvm_cv_platform_type="Unix" ;;
  *-*-gnu*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="GNU"
    llvm_cv_platform_type="Unix" ;;
  *-*-solaris*)
    llvm_cv_link_all_option="-Wl,-z,allextract"
    llvm_cv_no_link_all_option="-Wl,-z,defaultextract"
    llvm_cv_os_type="SunOS"
    llvm_cv_platform_type="Unix" ;;
  *-*-auroraux*)
    llvm_cv_link_all_option="-Wl,-z,allextract"
    llvm_cv_link_all_option="-Wl,-z,defaultextract"
    llvm_cv_os_type="AuroraUX"
    llvm_cv_platform_type="Unix" ;;
  *-*-win32*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Win32"
    llvm_cv_platform_type="Win32" ;;
  *-*-mingw*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="MingW"
    llvm_cv_platform_type="Win32" ;;
  *-*-haiku*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Haiku"
    llvm_cv_platform_type="Unix" ;;
  *-unknown-eabi*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Freestanding"
    llvm_cv_platform_type="Unix" ;;
  *-unknown-elf*)
    llvm_cv_link_all_option="-Wl,--whole-archive"
    llvm_cv_no_link_all_option="-Wl,--no-whole-archive"
    llvm_cv_os_type="Freestanding"
    llvm_cv_platform_type="Unix" ;;
  *)
    llvm_cv_link_all_option=""
    llvm_cv_no_link_all_option=""
    llvm_cv_os_type="Unknown"
    llvm_cv_platform_type="Unknown" ;;
esac])

AC_CACHE_CHECK([type of operating system we're going to target],
               [llvm_cv_target_os_type],
[case $target in
  *-*-aix*)
    llvm_cv_target_os_type="AIX" ;;
  *-*-irix*)
    llvm_cv_target_os_type="IRIX" ;;
  *-*-cygwin*)
    llvm_cv_target_os_type="Cygwin" ;;
  *-*-darwin*)
    llvm_cv_target_os_type="Darwin" ;;
  *-*-minix*)
    llvm_cv_target_os_type="Minix" ;;
  *-*-freebsd* | *-*-kfreebsd-gnu)
    llvm_cv_target_os_type="FreeBSD" ;;
  *-*-openbsd*)
    llvm_cv_target_os_type="OpenBSD" ;;
  *-*-netbsd*)
    llvm_cv_target_os_type="NetBSD" ;;
  *-*-dragonfly*)
    llvm_cv_target_os_type="DragonFly" ;;
  *-*-hpux*)
    llvm_cv_target_os_type="HP-UX" ;;
  *-*-interix*)
    llvm_cv_target_os_type="Interix" ;;
  *-*-linux*)
    llvm_cv_target_os_type="Linux" ;;
  *-*-gnu*)
    llvm_cv_target_os_type="GNU" ;;
  *-*-solaris*)
    llvm_cv_target_os_type="SunOS" ;;
  *-*-auroraux*)
    llvm_cv_target_os_type="AuroraUX" ;;
  *-*-win32*)
    llvm_cv_target_os_type="Win32" ;;
  *-*-mingw*)
    llvm_cv_target_os_type="MingW" ;;
  *-*-haiku*)
    llvm_cv_target_os_type="Haiku" ;;
  *-*-rtems*)
    llvm_cv_target_os_type="RTEMS" ;;
  *-*-nacl*)
    llvm_cv_target_os_type="NativeClient" ;;
  *-unknown-eabi*)
    llvm_cv_target_os_type="Freestanding" ;;
  *)
    llvm_cv_target_os_type="Unknown" ;;
esac])

dnl Make sure we aren't attempting to configure for an unknown system
if test "$llvm_cv_os_type" = "Unknown" ; then
  AC_MSG_ERROR([Operating system is unknown, configure can't continue])
fi

dnl Set the "OS" Makefile variable based on the platform type so the
dnl makefile can configure itself to specific build hosts
AC_SUBST(OS,$llvm_cv_os_type)
AC_SUBST(HOST_OS,$llvm_cv_os_type)
AC_SUBST(TARGET_OS,$llvm_cv_target_os_type)

dnl Set the LINKALL and NOLINKALL Makefile variables based on the platform
AC_SUBST(LINKALL,$llvm_cv_link_all_option)
AC_SUBST(NOLINKALL,$llvm_cv_no_link_all_option)

dnl Set the "LLVM_ON_*" variables based on llvm_cv_platform_type
dnl This is used by lib/Support to determine the basic kind of implementation
dnl to use.
case $llvm_cv_platform_type in
  Unix)
    AC_DEFINE([LLVM_ON_UNIX],[1],[Define if this is Unixish platform])
    AC_SUBST(LLVM_ON_UNIX,[1])
    AC_SUBST(LLVM_ON_WIN32,[0])
    ;;
  Win32)
    AC_DEFINE([LLVM_ON_WIN32],[1],[Define if this is Win32ish platform])
    AC_SUBST(LLVM_ON_UNIX,[0])
    AC_SUBST(LLVM_ON_WIN32,[1])
    ;;
esac

dnl Determine what our target architecture is and configure accordingly.
dnl This will allow Makefiles to make a distinction between the hardware and
dnl the OS.
AC_CACHE_CHECK([target architecture],[llvm_cv_target_arch],
[case $target in
  i?86-*)                 llvm_cv_target_arch="x86" ;;
  amd64-* | x86_64-*)     llvm_cv_target_arch="x86_64" ;;
  sparc*-*)               llvm_cv_target_arch="Sparc" ;;
  powerpc*-*)             llvm_cv_target_arch="PowerPC" ;;
  arm*-*)                 llvm_cv_target_arch="ARM" ;;
  mips-*)                 llvm_cv_target_arch="Mips" ;;
  mipsel-*)               llvm_cv_target_arch="Mips" ;;
  xcore-*)                llvm_cv_target_arch="XCore" ;;
  msp430-*)               llvm_cv_target_arch="MSP430" ;;
  hexagon-*)              llvm_cv_target_arch="Hexagon" ;;
  mblaze-*)               llvm_cv_target_arch="MBlaze" ;;
  ptx-*)                  llvm_cv_target_arch="PTX" ;;
  *)                      llvm_cv_target_arch="Unknown" ;;
esac])

if test "$llvm_cv_target_arch" = "Unknown" ; then
  AC_MSG_WARN([Configuring LLVM for an unknown target archicture])
fi

dnl Determine the LLVM native architecture for the target
case "$llvm_cv_target_arch" in
    x86)     LLVM_NATIVE_ARCH="X86" ;;
    x86_64)  LLVM_NATIVE_ARCH="X86" ;;
    *)       LLVM_NATIVE_ARCH="$llvm_cv_target_arch" ;;
esac

dnl Define a substitution, ARCH, for the target architecture
AC_SUBST(ARCH,$llvm_cv_target_arch)

dnl Check for the endianness of the target
AC_C_BIGENDIAN(AC_SUBST([ENDIAN],[big]),AC_SUBST([ENDIAN],[little]))

dnl Check for build platform executable suffix if we're cross-compiling
if test "$cross_compiling" = yes; then
  AC_SUBST(LLVM_CROSS_COMPILING, [1])
  AC_BUILD_EXEEXT
  ac_build_prefix=${build_alias}-
  AC_CHECK_PROG(BUILD_CXX, ${ac_build_prefix}g++, ${ac_build_prefix}g++)
  if test -z "$BUILD_CXX"; then
     AC_CHECK_PROG(BUILD_CXX, g++, g++)
     if test -z "$BUILD_CXX"; then
       AC_CHECK_PROG(BUILD_CXX, c++, c++, , , /usr/ucb/c++)
     fi
  fi
else
  AC_SUBST(LLVM_CROSS_COMPILING, [0])
fi

dnl Check to see if there's a .svn or .git directory indicating that this
dnl build is being done from a checkout. This sets up several defaults for
dnl the command line switches. When we build with a checkout directory,
dnl we get a debug with assertions turned on. Without, we assume a source
dnl release and we get an optimized build without assertions.
dnl See --enable-optimized and --enable-assertions below
if test -d ".svn" -o -d "${srcdir}/.svn" -o -d ".git" -o -d "${srcdir}/.git"; then
  cvsbuild="yes"
  optimize="no"
  AC_SUBST(CVSBUILD,[[CVSBUILD=1]])
else
  cvsbuild="no"
  optimize="yes"
fi

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 3: Command line arguments for the configure script.
dnl===
dnl===-----------------------------------------------------------------------===

dnl --enable-libcpp : check whether or not to use libc++ on the command line
AC_ARG_ENABLE(libcpp,
              AS_HELP_STRING([--enable-libcpp],
                             [Use libc++ if available (default is NO)]),,
                             enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_LIBCPP,[1]) ;;
  no)  AC_SUBST(ENABLE_LIBCPP,[0]) ;;
  default) AC_SUBST(ENABLE_LIBCPP,[0]);;
  *) AC_MSG_ERROR([Invalid setting for --enable-libcpp. Use "yes" or "no"]) ;;
esac

dnl --enable-optimized : check whether they want to do an optimized build:
AC_ARG_ENABLE(optimized, AS_HELP_STRING(
 --enable-optimized,[Compile with optimizations enabled (default is NO)]),,enableval=$optimize)
if test ${enableval} = "no" ; then
  AC_SUBST(ENABLE_OPTIMIZED,[[]])
else
  AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]])
fi

dnl --enable-profiling : check whether they want to do a profile build:
AC_ARG_ENABLE(profiling, AS_HELP_STRING(
 --enable-profiling,[Compile with profiling enabled (default is NO)]),,enableval="no")
if test ${enableval} = "no" ; then
  AC_SUBST(ENABLE_PROFILING,[[]])
else
  AC_SUBST(ENABLE_PROFILING,[[ENABLE_PROFILING=1]])
fi

dnl --enable-assertions : check whether they want to turn on assertions or not:
AC_ARG_ENABLE(assertions,AS_HELP_STRING(
  --enable-assertions,[Compile with assertion checks enabled (default is YES)]),, enableval="yes")
if test ${enableval} = "yes" ; then
  AC_SUBST(DISABLE_ASSERTIONS,[[]])
else
  AC_SUBST(DISABLE_ASSERTIONS,[[DISABLE_ASSERTIONS=1]])
fi

dnl --enable-expensive-checks : check whether they want to turn on expensive debug checks:
AC_ARG_ENABLE(expensive-checks,AS_HELP_STRING(
  --enable-expensive-checks,[Compile with expensive debug checks enabled (default is NO)]),, enableval="no")
if test ${enableval} = "yes" ; then
  AC_SUBST(ENABLE_EXPENSIVE_CHECKS,[[ENABLE_EXPENSIVE_CHECKS=1]])
  AC_SUBST(EXPENSIVE_CHECKS,[[yes]])
else
  AC_SUBST(ENABLE_EXPENSIVE_CHECKS,[[]])
  AC_SUBST(EXPENSIVE_CHECKS,[[no]])
fi

dnl --enable-debug-runtime : should runtime libraries have debug symbols?
AC_ARG_ENABLE(debug-runtime,
   AS_HELP_STRING(--enable-debug-runtime,[Build runtime libs with debug symbols (default is NO)]),,enableval=no)
if test ${enableval} = "no" ; then
  AC_SUBST(DEBUG_RUNTIME,[[]])
else
  AC_SUBST(DEBUG_RUNTIME,[[DEBUG_RUNTIME=1]])
fi

dnl --enable-debug-symbols : should even optimized compiler libraries
dnl have debug symbols?
AC_ARG_ENABLE(debug-symbols,
   AS_HELP_STRING(--enable-debug-symbols,[Build compiler with debug symbols (default is NO if optimization is on and YES if it's off)]),,enableval=no)
if test ${enableval} = "no" ; then
  AC_SUBST(DEBUG_SYMBOLS,[[]])
else
  AC_SUBST(DEBUG_SYMBOLS,[[DEBUG_SYMBOLS=1]])
fi

dnl --enable-jit: check whether they want to enable the jit
AC_ARG_ENABLE(jit,
  AS_HELP_STRING(--enable-jit,
                 [Enable Just In Time Compiling (default is YES)]),,
  enableval=default)
if test ${enableval} = "no"
then
  AC_SUBST(JIT,[[]])
else
  case "$llvm_cv_target_arch" in
    x86)         AC_SUBST(TARGET_HAS_JIT,1) ;;
    Sparc)       AC_SUBST(TARGET_HAS_JIT,0) ;;
    PowerPC)     AC_SUBST(TARGET_HAS_JIT,1) ;;
    x86_64)      AC_SUBST(TARGET_HAS_JIT,1) ;;
    ARM)         AC_SUBST(TARGET_HAS_JIT,1) ;;
    Mips)        AC_SUBST(TARGET_HAS_JIT,1) ;;
    XCore)       AC_SUBST(TARGET_HAS_JIT,0) ;;
    MSP430)      AC_SUBST(TARGET_HAS_JIT,0) ;;
    Hexagon)     AC_SUBST(TARGET_HAS_JIT,0) ;;
    MBlaze)      AC_SUBST(TARGET_HAS_JIT,0) ;;
    PTX)         AC_SUBST(TARGET_HAS_JIT,0) ;;
    *)           AC_SUBST(TARGET_HAS_JIT,0) ;;
  esac
fi

dnl Allow enablement of building and installing docs
AC_ARG_ENABLE(docs,
              AS_HELP_STRING([--enable-docs],
                             [Build documents (default is YES)]),,
                             enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_DOCS,[1]) ;;
  no)  AC_SUBST(ENABLE_DOCS,[0]) ;;
  default) AC_SUBST(ENABLE_DOCS,[1]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-docs. Use "yes" or "no"]) ;;
esac

dnl Allow enablement of doxygen generated documentation
AC_ARG_ENABLE(doxygen,
              AS_HELP_STRING([--enable-doxygen],
                             [Build doxygen documentation (default is NO)]),,
                             enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_DOXYGEN,[1]) ;;
  no)  AC_SUBST(ENABLE_DOXYGEN,[0]) ;;
  default) AC_SUBST(ENABLE_DOXYGEN,[0]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-doxygen. Use "yes" or "no"]) ;;
esac

dnl Allow disablement of threads
AC_ARG_ENABLE(threads,
              AS_HELP_STRING([--enable-threads],
                             [Use threads if available (default is YES)]),,
                             enableval=default)
case "$enableval" in
  yes) AC_SUBST(LLVM_ENABLE_THREADS,[1]) ;;
  no)  AC_SUBST(LLVM_ENABLE_THREADS,[0]) ;;
  default) AC_SUBST(LLVM_ENABLE_THREADS,[1]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-threads. Use "yes" or "no"]) ;;
esac
AC_DEFINE_UNQUOTED([LLVM_ENABLE_THREADS],$LLVM_ENABLE_THREADS,
                   [Define if threads enabled])

dnl Allow disablement of pthread.h
AC_ARG_ENABLE(pthreads,
              AS_HELP_STRING([--enable-pthreads],
                             [Use pthreads if available (default is YES)]),,
                             enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_PTHREADS,[1]) ;;
  no)  AC_SUBST(ENABLE_PTHREADS,[0]) ;;
  default) AC_SUBST(ENABLE_PTHREADS,[1]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-pthreads. Use "yes" or "no"]) ;;
esac

dnl Allow building without position independent code
AC_ARG_ENABLE(pic,
  AS_HELP_STRING([--enable-pic],
                 [Build LLVM with Position Independent Code (default is YES)]),,
                 enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_PIC,[1]) ;;
  no)  AC_SUBST(ENABLE_PIC,[0]) ;;
  default) AC_SUBST(ENABLE_PIC,[1]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-pic. Use "yes" or "no"]) ;;
esac
AC_DEFINE_UNQUOTED([ENABLE_PIC],$ENABLE_PIC,
                   [Define if position independent code is enabled])

dnl Allow building a shared library and linking tools against it.
AC_ARG_ENABLE(shared,
  AS_HELP_STRING([--enable-shared],
                 [Build a shared library and link tools against it (default is NO)]),,
                 enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_SHARED,[1]) ;;
  no)  AC_SUBST(ENABLE_SHARED,[0]) ;;
  default) AC_SUBST(ENABLE_SHARED,[0]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-shared. Use "yes" or "no"]) ;;
esac

dnl Allow libstdc++ is embedded in LLVM.dll.
AC_ARG_ENABLE(embed-stdcxx,
  AS_HELP_STRING([--enable-embed-stdcxx],
                 [Build a shared library with embedded libstdc++ for Win32 DLL (default is NO)]),,
                 enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_EMBED_STDCXX,[1]) ;;
  no)  AC_SUBST(ENABLE_EMBED_STDCXX,[0]) ;;
  default) AC_SUBST(ENABLE_EMBED_STDCXX,[0]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-embed-stdcxx. Use "yes" or "no"]) ;;
esac

dnl Enable embedding timestamp information into build.
AC_ARG_ENABLE(timestamps,
  AS_HELP_STRING([--enable-timestamps],
                 [Enable embedding timestamp information in build (default is YES)]),,
                 enableval=default)
case "$enableval" in
  yes) AC_SUBST(ENABLE_TIMESTAMPS,[1]) ;;
  no)  AC_SUBST(ENABLE_TIMESTAMPS,[0]) ;;
  default) AC_SUBST(ENABLE_TIMESTAMPS,[1]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-timestamps. Use "yes" or "no"]) ;;
esac
AC_DEFINE_UNQUOTED([ENABLE_TIMESTAMPS],$ENABLE_TIMESTAMPS,
                   [Define if timestamp information (e.g., __DATE___) is allowed])

dnl Allow specific targets to be specified for building (or not)
TARGETS_TO_BUILD=""
AC_ARG_ENABLE([targets],AS_HELP_STRING([--enable-targets],
    [Build specific host targets: all or target1,target2,... Valid targets are:
     host, x86, x86_64, sparc, powerpc, arm, mips, spu, hexagon,
     xcore, msp430, ptx, and cpp (default=all)]),,
    enableval=all)
if test "$enableval" = host-only ; then
  enableval=host
fi
case "$enableval" in
  all) TARGETS_TO_BUILD="X86 Sparc PowerPC ARM Mips CellSPU XCore MSP430 CppBackend MBlaze PTX Hexagon" ;;
  *)for a_target in `echo $enableval|sed -e 's/,/ /g' ` ; do
      case "$a_target" in
        x86)      TARGETS_TO_BUILD="X86 $TARGETS_TO_BUILD" ;;
        x86_64)   TARGETS_TO_BUILD="X86 $TARGETS_TO_BUILD" ;;
        sparc)    TARGETS_TO_BUILD="Sparc $TARGETS_TO_BUILD" ;;
        powerpc)  TARGETS_TO_BUILD="PowerPC $TARGETS_TO_BUILD" ;;
        arm)      TARGETS_TO_BUILD="ARM $TARGETS_TO_BUILD" ;;
        mips)     TARGETS_TO_BUILD="Mips $TARGETS_TO_BUILD" ;;
        mipsel)   TARGETS_TO_BUILD="Mips $TARGETS_TO_BUILD" ;;
        spu)      TARGETS_TO_BUILD="CellSPU $TARGETS_TO_BUILD" ;;
        xcore)    TARGETS_TO_BUILD="XCore $TARGETS_TO_BUILD" ;;
        msp430)   TARGETS_TO_BUILD="MSP430 $TARGETS_TO_BUILD" ;;
        cpp)      TARGETS_TO_BUILD="CppBackend $TARGETS_TO_BUILD" ;;
        hexagon)  TARGETS_TO_BUILD="Hexagon $TARGETS_TO_BUILD" ;;
        mblaze)   TARGETS_TO_BUILD="MBlaze $TARGETS_TO_BUILD" ;;
        ptx)      TARGETS_TO_BUILD="PTX $TARGETS_TO_BUILD" ;;
        host) case "$llvm_cv_target_arch" in
            x86)         TARGETS_TO_BUILD="X86 $TARGETS_TO_BUILD" ;;
            x86_64)      TARGETS_TO_BUILD="X86 $TARGETS_TO_BUILD" ;;
            Sparc)       TARGETS_TO_BUILD="Sparc $TARGETS_TO_BUILD" ;;
            PowerPC)     TARGETS_TO_BUILD="PowerPC $TARGETS_TO_BUILD" ;;
            ARM)         TARGETS_TO_BUILD="ARM $TARGETS_TO_BUILD" ;;
            Mips)        TARGETS_TO_BUILD="Mips $TARGETS_TO_BUILD" ;;
            MBlaze)      TARGETS_TO_BUILD="MBlaze $TARGETS_TO_BUILD" ;;
            CellSPU|SPU) TARGETS_TO_BUILD="CellSPU $TARGETS_TO_BUILD" ;;
            XCore)       TARGETS_TO_BUILD="XCore $TARGETS_TO_BUILD" ;;
            MSP430)      TARGETS_TO_BUILD="MSP430 $TARGETS_TO_BUILD" ;;
            Hexagon)     TARGETS_TO_BUILD="Hexagon $TARGETS_TO_BUILD" ;;
            PTX)         TARGETS_TO_BUILD="PTX $TARGETS_TO_BUILD" ;;
            *)       AC_MSG_ERROR([Can not set target to build]) ;;
          esac ;;
        *) AC_MSG_ERROR([Unrecognized target $a_target]) ;;
      esac
  done
  ;;
esac
AC_SUBST(TARGETS_TO_BUILD,$TARGETS_TO_BUILD)

dnl Determine whether we are building LLVM support for the native architecture.
dnl If so, define LLVM_NATIVE_ARCH to that LLVM target.
for a_target in $TARGETS_TO_BUILD; do
  if test "$a_target" = "$LLVM_NATIVE_ARCH"; then
    AC_DEFINE_UNQUOTED(LLVM_NATIVE_ARCH, $LLVM_NATIVE_ARCH,
      [LLVM architecture name for the native architecture, if available])
    LLVM_NATIVE_TARGET="LLVMInitialize${LLVM_NATIVE_ARCH}Target"
    LLVM_NATIVE_TARGETINFO="LLVMInitialize${LLVM_NATIVE_ARCH}TargetInfo"
    LLVM_NATIVE_TARGETMC="LLVMInitialize${LLVM_NATIVE_ARCH}TargetMC"
    LLVM_NATIVE_ASMPRINTER="LLVMInitialize${LLVM_NATIVE_ARCH}AsmPrinter"
    if test -f ${srcdir}/lib/Target/${LLVM_NATIVE_ARCH}/AsmParser/Makefile ; then
      LLVM_NATIVE_ASMPARSER="LLVMInitialize${LLVM_NATIVE_ARCH}AsmParser"
    fi
    if test -f ${srcdir}/lib/Target/${LLVM_NATIVE_ARCH}/Disassembler/Makefile ; then
      LLVM_NATIVE_DISASSEMBLER="LLVMInitialize${LLVM_NATIVE_ARCH}Disassembler"
    fi
    AC_DEFINE_UNQUOTED(LLVM_NATIVE_TARGET, $LLVM_NATIVE_TARGET,
      [LLVM name for the native Target init function, if available])
    AC_DEFINE_UNQUOTED(LLVM_NATIVE_TARGETINFO, $LLVM_NATIVE_TARGETINFO,
      [LLVM name for the native TargetInfo init function, if available])
    AC_DEFINE_UNQUOTED(LLVM_NATIVE_TARGETMC, $LLVM_NATIVE_TARGETMC,
      [LLVM name for the native target MC init function, if available])
    AC_DEFINE_UNQUOTED(LLVM_NATIVE_ASMPRINTER, $LLVM_NATIVE_ASMPRINTER,
      [LLVM name for the native AsmPrinter init function, if available])
    if test -f ${srcdir}/lib/Target/${LLVM_NATIVE_ARCH}/AsmParser/Makefile ; then
      AC_DEFINE_UNQUOTED(LLVM_NATIVE_ASMPARSER, $LLVM_NATIVE_ASMPARSER,
       [LLVM name for the native AsmParser init function, if available])
    fi
    if test -f ${srcdir}/lib/Target/${LLVM_NATIVE_ARCH}/Disassembler/Makefile ; then
      AC_DEFINE_UNQUOTED(LLVM_NATIVE_DISASSEMBLER, $LLVM_NATIVE_DISASSEMBLER,
       [LLVM name for the native Disassembler init function, if available])
    fi
  fi
done

dnl Build the LLVM_TARGET and LLVM_... macros for Targets.def and the individual
dnl target feature def files.
LLVM_ENUM_TARGETS=""
LLVM_ENUM_ASM_PRINTERS=""
LLVM_ENUM_ASM_PARSERS=""
LLVM_ENUM_DISASSEMBLERS=""
for target_to_build in $TARGETS_TO_BUILD; do
  LLVM_ENUM_TARGETS="LLVM_TARGET($target_to_build) $LLVM_ENUM_TARGETS"
  if test -f ${srcdir}/lib/Target/${target_to_build}/*AsmPrinter.cpp ; then
    LLVM_ENUM_ASM_PRINTERS="LLVM_ASM_PRINTER($target_to_build) $LLVM_ENUM_ASM_PRINTERS";
  fi
  if test -f ${srcdir}/lib/Target/${target_to_build}/AsmParser/Makefile ; then
    LLVM_ENUM_ASM_PARSERS="LLVM_ASM_PARSER($target_to_build) $LLVM_ENUM_ASM_PARSERS";
  fi
  if test -f ${srcdir}/lib/Target/${target_to_build}/Disassembler/Makefile ; then
    LLVM_ENUM_DISASSEMBLERS="LLVM_DISASSEMBLER($target_to_build) $LLVM_ENUM_DISASSEMBLERS";
  fi
done
AC_SUBST(LLVM_ENUM_TARGETS)
AC_SUBST(LLVM_ENUM_ASM_PRINTERS)
AC_SUBST(LLVM_ENUM_ASM_PARSERS)
AC_SUBST(LLVM_ENUM_DISASSEMBLERS)

dnl Override the option to use for optimized builds.
AC_ARG_WITH(optimize-option,
  AS_HELP_STRING([--with-optimize-option],
                 [Select the compiler options to use for optimized builds]),,
                 withval=default)
AC_MSG_CHECKING([optimization flags])
case "$withval" in
  default)
    case "$llvm_cv_os_type" in
    FreeBSD) optimize_option=-O2 ;;
    MingW) optimize_option=-O2 ;;
    *)     optimize_option=-O3 ;;
    esac ;;
  *) optimize_option="$withval" ;;
esac
AC_SUBST(OPTIMIZE_OPTION,$optimize_option)
AC_MSG_RESULT([$optimize_option])

dnl Specify extra build options
AC_ARG_WITH(extra-options,
  AS_HELP_STRING([--with-extra-options],
                 [Specify additional options to compile LLVM with]),,
                 withval=default)
case "$withval" in
  default) EXTRA_OPTIONS= ;;
  *) EXTRA_OPTIONS=$withval ;;
esac
AC_SUBST(EXTRA_OPTIONS,$EXTRA_OPTIONS)

dnl Specify extra linker build options
AC_ARG_WITH(extra-ld-options,
  AS_HELP_STRING([--with-extra-ld-options],
                 [Specify additional options to link LLVM with]),,
                 withval=default)
case "$withval" in
  default) EXTRA_LD_OPTIONS= ;;
  *) EXTRA_LD_OPTIONS=$withval ;;
esac
AC_SUBST(EXTRA_LD_OPTIONS,$EXTRA_LD_OPTIONS)

dnl Allow specific bindings to be specified for building (or not)
AC_ARG_ENABLE([bindings],AS_HELP_STRING([--enable-bindings],
    [Build specific language bindings: all,auto,none,{binding-name} (default=auto)]),,
    enableval=default)
BINDINGS_TO_BUILD=""
case "$enableval" in
  yes | default | auto) BINDINGS_TO_BUILD="auto" ;;
  all ) BINDINGS_TO_BUILD="ocaml" ;;
  none | no) BINDINGS_TO_BUILD="" ;;
  *)for a_binding in `echo $enableval|sed -e 's/,/ /g' ` ; do
      case "$a_binding" in
        ocaml) BINDINGS_TO_BUILD="ocaml $BINDINGS_TO_BUILD" ;;
        *) AC_MSG_ERROR([Unrecognized binding $a_binding]) ;;
      esac
  done
  ;;
esac

dnl Allow the ocaml libdir to be overridden. This could go in a configure
dnl script for bindings/ocaml/configure, except that its auto value depends on
dnl OCAMLC, which is found here to support tests.
AC_ARG_WITH([ocaml-libdir],
  [AS_HELP_STRING([--with-ocaml-libdir],
    [Specify install location for ocaml bindings (default is stdlib)])],
  [],
  [withval=auto])
case "$withval" in
  auto) with_ocaml_libdir="$withval" ;;
  /* | [[A-Za-z]]:[[\\/]]*) with_ocaml_libdir="$withval" ;;
  *) AC_MSG_ERROR([Invalid path for --with-ocaml-libdir. Provide full path]) ;;
esac

AC_ARG_WITH(clang-srcdir,
  AS_HELP_STRING([--with-clang-srcdir],
    [Directory to the out-of-tree Clang source]),,
    withval="-")
case "$withval" in
  -) clang_src_root="" ;;
  /* | [[A-Za-z]]:[[\\/]]*) clang_src_root="$withval" ;;
  *) clang_src_root="$ac_pwd/$withval" ;;
esac
AC_SUBST(CLANG_SRC_ROOT,[$clang_src_root])

AC_ARG_WITH(clang-resource-dir,
  AS_HELP_STRING([--with-clang-resource-dir],
    [Relative directory from the Clang binary for resource files]),,
    withval="")
AC_DEFINE_UNQUOTED(CLANG_RESOURCE_DIR,"$withval",
                   [Relative directory for resource files])

AC_ARG_WITH(c-include-dirs,
  AS_HELP_STRING([--with-c-include-dirs],
    [Colon separated list of directories clang will search for headers]),,
    withval="")
AC_DEFINE_UNQUOTED(C_INCLUDE_DIRS,"$withval",
                   [Directories clang will search for headers])

# Clang normally uses the system c++ headers and libraries. With this option,
# clang will use the ones provided by a gcc installation instead. This option should
# be passed the same value that was used with --prefix when configuring gcc.
AC_ARG_WITH(gcc-toolchain,
  AS_HELP_STRING([--with-gcc-toolchain],
    [Directory where gcc is installed.]),,
    withval="")
AC_DEFINE_UNQUOTED(GCC_INSTALL_PREFIX,"$withval",
                   [Directory where gcc is installed.])

AC_ARG_WITH(default-sysroot,
  AS_HELP_STRING([--with-default-sysroot],
    [Add --sysroot=<path> to all compiler invocations.]),,
    withval="")
AC_DEFINE_UNQUOTED(DEFAULT_SYSROOT,"$withval",
                   [Default <path> to all compiler invocations for --sysroot=<path>.])

dnl Allow linking of LLVM with GPLv3 binutils code.
AC_ARG_WITH(binutils-include,
  AS_HELP_STRING([--with-binutils-include],
    [Specify path to binutils/include/ containing plugin-api.h file for gold plugin.]),,
  withval=default)
case "$withval" in
  default) WITH_BINUTILS_INCDIR=default ;;
  /* | [[A-Za-z]]:[[\\/]]*)      WITH_BINUTILS_INCDIR=$withval ;;
  *) AC_MSG_ERROR([Invalid path for --with-binutils-include. Provide full path]) ;;
esac
if test "x$WITH_BINUTILS_INCDIR" != xdefault ; then
  AC_SUBST(BINUTILS_INCDIR,$WITH_BINUTILS_INCDIR)
  if test ! -f "$WITH_BINUTILS_INCDIR/plugin-api.h"; then
     echo "$WITH_BINUTILS_INCDIR/plugin-api.h"
     AC_MSG_ERROR([Invalid path to directory containing plugin-api.h.]);
  fi
fi

dnl Specify the URL where bug reports should be submitted.
AC_ARG_WITH(bug-report-url,
  AS_HELP_STRING([--with-bug-report-url],
    [Specify the URL where bug reports should be submitted (default=http://llvm.org/bugs/)]),,
    withval="http://llvm.org/bugs/")
AC_DEFINE_UNQUOTED(BUG_REPORT_URL,"$withval",
                   [Bug report URL.])

dnl --enable-libffi : check whether the user wants to turn off libffi:
AC_ARG_ENABLE(libffi,AS_HELP_STRING(
  --enable-libffi,[Check for the presence of libffi (default is NO)]),
  [case "$enableval" in
    yes) llvm_cv_enable_libffi="yes" ;;
    no)  llvm_cv_enable_libffi="no"  ;;
    *) AC_MSG_ERROR([Invalid setting for --enable-libffi. Use "yes" or "no"]) ;;
  esac],
  llvm_cv_enable_libffi=no)

AC_ARG_WITH(internal-prefix,
  AS_HELP_STRING([--with-internal-prefix],
    [Installation directory for internal files]),,
    withval="")
AC_SUBST(INTERNAL_PREFIX,[$withval])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 4: Check for programs we need and that they are the right version
dnl===
dnl===-----------------------------------------------------------------------===

AC_PROG_NM
AC_SUBST(NM)

dnl Check for the tools that the makefiles require
AC_CHECK_GNU_MAKE
AC_PROG_LN_S
AC_PATH_PROG(CMP, [cmp], [cmp])
AC_PATH_PROG(CP, [cp], [cp])
AC_PATH_PROG(DATE, [date], [date])
AC_PATH_PROG(FIND, [find], [find])
AC_PATH_PROG(GREP, [grep], [grep])
AC_PATH_PROG(MKDIR,[mkdir],[mkdir])
AC_PATH_PROG(MV,   [mv],   [mv])
AC_PROG_RANLIB
AC_CHECK_TOOL(AR, ar, false)
AC_PATH_PROG(RM,   [rm],   [rm])
AC_PATH_PROG(SED,  [sed],  [sed])
AC_PATH_PROG(TAR,  [tar],  [gtar])
AC_PATH_PROG(BINPWD,[pwd],  [pwd])

dnl Looking for misc. graph plotting software
AC_PATH_PROG(GRAPHVIZ, [Graphviz], [echo Graphviz])
if test "$GRAPHVIZ" != "echo Graphviz" ; then
  AC_DEFINE([HAVE_GRAPHVIZ],[1],[Define if the Graphviz program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    GRAPHVIZ=`echo $GRAPHVIZ | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_GRAPHVIZ],"$GRAPHVIZ${EXEEXT}",
   [Define to path to Graphviz program if found or 'echo Graphviz' otherwise])
fi
AC_PATH_PROG(DOT, [dot], [echo dot])
if test "$DOT" != "echo dot" ; then
  AC_DEFINE([HAVE_DOT],[1],[Define if the dot program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    DOT=`echo $DOT | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_DOT],"$DOT${EXEEXT}",
   [Define to path to dot program if found or 'echo dot' otherwise])
fi
AC_PATH_PROG(FDP, [fdp], [echo fdp])
if test "$FDP" != "echo fdp" ; then
  AC_DEFINE([HAVE_FDP],[1],[Define if the neat program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    FDP=`echo $FDP | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_FDP],"$FDP${EXEEXT}",
   [Define to path to fdp program if found or 'echo fdp' otherwise])
fi
AC_PATH_PROG(NEATO, [neato], [echo neato])
if test "$NEATO" != "echo neato" ; then
  AC_DEFINE([HAVE_NEATO],[1],[Define if the neat program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    NEATO=`echo $NEATO | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_NEATO],"$NEATO${EXEEXT}",
   [Define to path to neato program if found or 'echo neato' otherwise])
fi
AC_PATH_PROG(TWOPI, [twopi], [echo twopi])
if test "$TWOPI" != "echo twopi" ; then
  AC_DEFINE([HAVE_TWOPI],[1],[Define if the neat program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    TWOPI=`echo $TWOPI | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_TWOPI],"$TWOPI${EXEEXT}",
   [Define to path to twopi program if found or 'echo twopi' otherwise])
fi
AC_PATH_PROG(CIRCO, [circo], [echo circo])
if test "$CIRCO" != "echo circo" ; then
  AC_DEFINE([HAVE_CIRCO],[1],[Define if the neat program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    CIRCO=`echo $CIRCO | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_CIRCO],"$CIRCO${EXEEXT}",
   [Define to path to circo program if found or 'echo circo' otherwise])
fi
AC_PATH_PROGS(GV, [gv gsview32], [echo gv])
if test "$GV" != "echo gv" ; then
  AC_DEFINE([HAVE_GV],[1],[Define if the gv program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    GV=`echo $GV | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_GV],"$GV${EXEEXT}",
   [Define to path to gv program if found or 'echo gv' otherwise])
fi
AC_PATH_PROG(DOTTY, [dotty], [echo dotty])
if test "$DOTTY" != "echo dotty" ; then
  AC_DEFINE([HAVE_DOTTY],[1],[Define if the dotty program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    DOTTY=`echo $DOTTY | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_DOTTY],"$DOTTY${EXEEXT}",
   [Define to path to dotty program if found or 'echo dotty' otherwise])
fi
AC_PATH_PROG(XDOT_PY, [xdot.py], [echo xdot.py])
if test "$XDOT_PY" != "echo xdot.py" ; then
  AC_DEFINE([HAVE_XDOT_PY],[1],[Define if the xdot.py program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" = "MingW" ; then
    XDOT_PY=`echo $XDOT_PY | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_XDOT_PY],"$XDOT_PY${EXEEXT}",
   [Define to path to xdot.py program if found or 'echo xdot.py' otherwise])
fi

dnl Find the install program
AC_PROG_INSTALL
dnl Prepend src dir to install path dir if it's a relative path
dnl This is a hack for installs that take place in something other
dnl than the top level.
case "$INSTALL" in
 [[\\/$]]* | ?:[[\\/]]* ) ;;
 *)  INSTALL="\\\$(TOPSRCDIR)/$INSTALL" ;;
esac

dnl Checks for documentation and testing tools that we can do without. If these
dnl are not found then they are set to "true" which always succeeds but does
dnl nothing. This just lets the build output show that we could have done
dnl something if the tool was available.
AC_PATH_PROG(BZIP2, [bzip2])
AC_PATH_PROG(CAT, [cat])
AC_PATH_PROG(DOXYGEN, [doxygen])
AC_PATH_PROG(GROFF, [groff])
AC_PATH_PROG(GZIPBIN, [gzip])
AC_PATH_PROG(POD2HTML, [pod2html])
AC_PATH_PROG(POD2MAN, [pod2man])
AC_PATH_PROG(PDFROFF, [pdfroff])
AC_PATH_PROG(RUNTEST, [runtest])
DJ_AC_PATH_TCLSH
AC_PATH_PROG(ZIP, [zip])
AC_PATH_PROGS(OCAMLC, [ocamlc])
AC_PATH_PROGS(OCAMLOPT, [ocamlopt])
AC_PATH_PROGS(OCAMLDEP, [ocamldep])
AC_PATH_PROGS(OCAMLDOC, [ocamldoc])
AC_PATH_PROGS(GAS, [gas as])

dnl Get the version of the linker in use.
AC_LINK_GET_VERSION

dnl Determine whether the linker supports the -R option.
AC_LINK_USE_R

dnl Determine whether the linker supports the -export-dynamic option.
AC_LINK_EXPORT_DYNAMIC

dnl Determine whether the linker supports the --version-script option.
AC_LINK_VERSION_SCRIPT

dnl Check for libtool and the library that has dlopen function (which must come
dnl before the AC_PROG_LIBTOOL check in order to enable dlopening libraries with
dnl libtool).
AC_LIBTOOL_DLOPEN
AC_LIB_LTDL

AC_MSG_CHECKING([tool compatibility])

dnl Ensure that compilation tools are GCC or a GNU compatible compiler such as
dnl ICC; we use GCC specific options in the makefiles so the compiler needs
dnl to support those options.
dnl "icc" emits gcc signatures
dnl "icc -no-gcc" emits no gcc signature BUT is still compatible
ICC=no
IXX=no
case $CC in
  icc*|icpc*)
    ICC=yes
    IXX=yes
    ;;
   *)
    ;;
esac

if test "$GCC" != "yes" && test "$ICC" != "yes"
then
  AC_MSG_ERROR([gcc|icc required but not found])
fi

dnl Ensure that compilation tools are compatible with GCC extensions
if test "$GXX" != "yes" && test "$IXX" != "yes"
then
  AC_MSG_ERROR([g++|clang++|icc required but not found])
fi

dnl Verify that GCC is version 3.0 or higher
if test "$GCC" = "yes"
then
  AC_COMPILE_IFELSE([[#if !defined(__GNUC__) || __GNUC__ < 3
#error Unsupported GCC version
#endif
]], [], [AC_MSG_ERROR([gcc 3.x required, but you have a lower version])])
fi

dnl Check for GNU Make.  We use its extensions, so don't build without it
if test -z "$llvm_cv_gnu_make_command"
then
  AC_MSG_ERROR([GNU Make required but not found])
fi

dnl Tool compatibility is okay if we make it here.
AC_MSG_RESULT([ok])

dnl Check optional compiler flags.
AC_MSG_CHECKING([optional compiler flags])
CXX_FLAG_CHECK(NO_VARIADIC_MACROS, [-Wno-variadic-macros])
CXX_FLAG_CHECK(NO_MISSING_FIELD_INITIALIZERS, [-Wno-missing-field-initializers])
CXX_FLAG_CHECK(COVERED_SWITCH_DEFAULT, [-Wcovered-switch-default])
AC_MSG_RESULT([$NO_VARIADIC_MACROS $NO_MISSING_FIELD_INITIALIZERS $COVERED_SWITCH_DEFAULT])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 5: Check for libraries
dnl===
dnl===-----------------------------------------------------------------------===

AC_CHECK_LIB(m,sin)
if test "$llvm_cv_os_type" = "MingW" ; then
  AC_CHECK_LIB(imagehlp, main)
  AC_CHECK_LIB(psapi, main)
fi

dnl dlopen() is required for plugin support.
AC_SEARCH_LIBS(dlopen,dl,AC_DEFINE([HAVE_DLOPEN],[1],
               [Define if dlopen() is available on this platform.]),
               AC_MSG_WARN([dlopen() not found - disabling plugin support]))

dnl libffi is optional; used to call external functions from the interpreter
if test "$llvm_cv_enable_libffi" = "yes" ; then
  AC_SEARCH_LIBS(ffi_call,ffi,AC_DEFINE([HAVE_FFI_CALL],[1],
                 [Define if libffi is available on this platform.]),
                 AC_MSG_ERROR([libffi not found - configure without --enable-libffi to compile without it]))
fi

dnl mallinfo is optional; the code can compile (minus features) without it
AC_SEARCH_LIBS(mallinfo,malloc,AC_DEFINE([HAVE_MALLINFO],[1],
               [Define if mallinfo() is available on this platform.]))

dnl pthread locking functions are optional - but llvm will not be thread-safe
dnl without locks.
if test "$LLVM_ENABLE_THREADS" -eq 1 && test "$ENABLE_PTHREADS" -eq 1 ; then
  AC_CHECK_LIB(pthread, pthread_mutex_init)
  AC_SEARCH_LIBS(pthread_mutex_lock,pthread,
                 AC_DEFINE([HAVE_PTHREAD_MUTEX_LOCK],[1],
                           [Have pthread_mutex_lock]))
  AC_SEARCH_LIBS(pthread_rwlock_init,pthread,
                 AC_DEFINE([HAVE_PTHREAD_RWLOCK_INIT],[1],
                 [Have pthread_rwlock_init]))
  AC_SEARCH_LIBS(pthread_getspecific,pthread,
                 AC_DEFINE([HAVE_PTHREAD_GETSPECIFIC],[1],
                 [Have pthread_getspecific]))
fi

dnl Allow extra x86-disassembler library
AC_ARG_WITH(udis86,
  AS_HELP_STRING([--with-udis86=<path>],
    [Use udis86 external x86 disassembler library]),
    [
      AC_SUBST(USE_UDIS86, [1])
      case "$withval" in
        /usr/lib|yes) ;;
        *) LDFLAGS="$LDFLAGS -L${withval}" ;;
      esac
      AC_CHECK_LIB(udis86, ud_init, [], [
        echo "Error! You need to have libudis86 around."
        exit -1
      ])
    ],
    AC_SUBST(USE_UDIS86, [0]))
AC_DEFINE_UNQUOTED([USE_UDIS86],$USE_UDIS86,
                   [Define if use udis86 library])

dnl Allow OProfile support for JIT output.
AC_ARG_WITH(oprofile,
  AS_HELP_STRING([--with-oprofile=<prefix>],
    [Tell OProfile >= 0.9.4 how to symbolize JIT output]),
    [
      AC_SUBST(USE_OPROFILE, [1])
      case "$withval" in
        /usr|yes) llvm_cv_oppath=/usr/lib/oprofile ;;
        no) llvm_cv_oppath=
            AC_SUBST(USE_OPROFILE, [0]) ;;
        *) llvm_cv_oppath="${withval}/lib/oprofile"
           CPPFLAGS="-I${withval}/include";;
      esac
      case $llvm_cv_os_type in
        Linux)
          if test -n "$llvm_cv_oppath" ; then
            LIBS="$LIBS -lopagent -L${llvm_cv_oppath} -Wl,-rpath,${llvm_cv_oppath}"
            dnl Work around http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=537744:
            dnl libbfd is not included properly in libopagent in some Debian
            dnl versions.  If libbfd isn't found at all, we assume opagent works
            dnl anyway.
            AC_SEARCH_LIBS(bfd_init, bfd, [], [])
            AC_SEARCH_LIBS(op_open_agent, opagent, [], [
              echo "Error! You need to have libopagent around."
              exit -1
            ])
            AC_CHECK_HEADER([opagent.h], [], [
              echo "Error! You need to have opagent.h around."
              exit -1
              ])
          fi ;;
        *)
          AC_MSG_ERROR([OProfile support is available on Linux only.]) ;;
      esac 
    ],
    [
      AC_SUBST(USE_OPROFILE, [0])
    ])
AC_DEFINE_UNQUOTED([LLVM_USE_OPROFILE],$USE_OPROFILE,
                   [Define if we have the oprofile JIT-support library])

dnl Enable support for Intel JIT Events API.
AC_ARG_WITH(intel-jitevents,
  AS_HELP_STRING([--with-intel-jitevents=<vtune-amplifier-dir>],
    [Specify location of run-time support library for Intel JIT API (default=/opt/intel/vtune_amplifier_xe_2011)]),
    [
      case $llvm_cv_os_type in
        Linux|Win32|Cygwin|MingW) ;;
        *)
          AC_MSG_ERROR([
            Intel JIT API support is available on Linux and Windows only."]) ;;
      esac

      AC_SUBST(USE_INTEL_JITEVENTS, [1])
      case "$llvm_cv_target_arch" in
        x86)    llvm_intel_jitevents_archdir="lib32";;
        x86_64) llvm_intel_jitevents_archdir="lib64";;
        *)      echo "Target architecture $llvm_cv_target_arch does not support Intel JIT Events API"
                exit -1;;
      esac
      INTEL_JITEVENTS_INCDIR="/opt/intel/vtune_amplifier_xe_2011/include"
      INTEL_JITEVENTS_LIBDIR="/opt/intel/vtune_amplifier_xe_2011/$llvm_intel_jitevents_archdir"
      case "$withval" in
        /* | [[A-Za-z]]:[[\\/]]*) INTEL_JITEVENTS_INCDIR=$withval/include
                                  INTEL_JITEVENTS_LIBDIR=$withval/$llvm_intel_jitevents_archdir ;;
        *) ;;
      esac

      AC_SUBST(INTEL_JITEVENTS_INCDIR)
      AC_SUBST(INTEL_JITEVENTS_LIBDIR)

      LIBS="$LIBS -L${INTEL_JITEVENTS_LIBDIR}"
      CPPFLAGS="$CPPFLAGS -I$INTEL_JITEVENTS_INCDIR"

      AC_SEARCH_LIBS(iJIT_IsProfilingActive, jitprofiling, [], [
        echo "Error! Cannot find libjitprofiling.a. Please check path specified in flag --with-intel-jitevents"
        exit -1
      ])
      AC_CHECK_HEADER([jitprofiling.h], [], [
        echo "Error! Cannot find jitprofiling.h. Please check path specified in flag --with-intel-jitevents"
        exit -1
      ])

    ],
    [
      AC_SUBST(USE_INTEL_JITEVENTS, [0])
    ])
AC_DEFINE_UNQUOTED([LLVM_USE_INTEL_JITEVENTS],$USE_INTEL_JITEVENTS,
                   [Define if we have the Intel JIT API runtime support library])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 6: Check for header files
dnl===
dnl===-----------------------------------------------------------------------===

dnl First, use autoconf provided macros for specific headers that we need
dnl We don't check for ancient stuff or things that are guaranteed to be there
dnl by the C++ standard. We always use the <cfoo> versions of <foo.h> C headers.
dnl Generally we're looking for POSIX headers.
AC_HEADER_DIRENT
AC_HEADER_MMAP_ANONYMOUS
AC_HEADER_STAT
AC_HEADER_SYS_WAIT
AC_HEADER_TIME

AC_CHECK_HEADERS([dlfcn.h execinfo.h fcntl.h inttypes.h limits.h link.h])
AC_CHECK_HEADERS([malloc.h setjmp.h signal.h stdint.h termios.h unistd.h])
AC_CHECK_HEADERS([utime.h windows.h])
AC_CHECK_HEADERS([sys/mman.h sys/param.h sys/resource.h sys/time.h sys/uio.h])
AC_CHECK_HEADERS([sys/types.h sys/ioctl.h malloc/malloc.h mach/mach.h])
AC_CHECK_HEADERS([valgrind/valgrind.h])
AC_CHECK_HEADERS([fenv.h])
if test "$LLVM_ENABLE_THREADS" -eq 1 && test "$ENABLE_PTHREADS" -eq 1 ; then
  AC_CHECK_HEADERS(pthread.h,
                   AC_SUBST(HAVE_PTHREAD, 1),
                   AC_SUBST(HAVE_PTHREAD, 0))
else
  AC_SUBST(HAVE_PTHREAD, 0)
fi

dnl Try to find ffi.h.
if test "$llvm_cv_enable_libffi" = "yes" ; then
  AC_CHECK_HEADERS([ffi.h ffi/ffi.h])
fi

dnl Try to find Darwin specific crash reporting libraries.
AC_CHECK_HEADERS([CrashReporterClient.h])

dnl Try to find Darwin specific crash reporting global.
AC_MSG_CHECKING([__crashreporter_info__])
AC_LINK_IFELSE(
  AC_LANG_SOURCE(
    [[extern const char *__crashreporter_info__;
      int main() {
        __crashreporter_info__ = "test";
        return 0;
      }
    ]]),
  AC_MSG_RESULT(yes)
  AC_DEFINE(HAVE_CRASHREPORTER_INFO, 1, Can use __crashreporter_info__),
  AC_MSG_RESULT(no)
  AC_DEFINE(HAVE_CRASHREPORTER_INFO, 0,
            Define if __crashreporter_info__ exists.))

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 7: Check for types and structures
dnl===
dnl===-----------------------------------------------------------------------===

AC_HUGE_VAL_CHECK
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_DEFINE_UNQUOTED([RETSIGTYPE],[void],[Define as the return type of signal handlers (`int' or `void').])
AC_STRUCT_TM
AC_CHECK_TYPES([int64_t],,AC_MSG_ERROR([Type int64_t required but not found]))
AC_CHECK_TYPES([uint64_t],,
         AC_CHECK_TYPES([u_int64_t],,
         AC_MSG_ERROR([Type uint64_t or u_int64_t required but not found])))

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 8: Check for specific functions needed
dnl===
dnl===-----------------------------------------------------------------------===

AC_CHECK_FUNCS([backtrace ceilf floorf roundf rintf nearbyintf getcwd ])
AC_CHECK_FUNCS([powf fmodf strtof round ])
AC_CHECK_FUNCS([getpagesize getrusage getrlimit setrlimit gettimeofday ])
AC_CHECK_FUNCS([isatty mkdtemp mkstemp ])
AC_CHECK_FUNCS([mktemp posix_spawn pread realpath sbrk setrlimit strdup ])
AC_CHECK_FUNCS([strerror strerror_r setenv ])
AC_CHECK_FUNCS([strtoll strtoq sysconf malloc_zone_statistics ])
AC_CHECK_FUNCS([setjmp longjmp sigsetjmp siglongjmp writev])
AC_C_PRINTF_A
AC_FUNC_RAND48

dnl Check the declaration "Secure API" on Windows environments.
AC_CHECK_DECLS([strerror_s])

dnl Check symbols in libgcc.a for JIT on Mingw.
if test "$llvm_cv_os_type" = "MingW" ; then
  AC_CHECK_LIB(gcc,_alloca,AC_DEFINE([HAVE__ALLOCA],[1],[Have host's _alloca]))
  AC_CHECK_LIB(gcc,__alloca,AC_DEFINE([HAVE___ALLOCA],[1],[Have host's __alloca]))
  AC_CHECK_LIB(gcc,__chkstk,AC_DEFINE([HAVE___CHKSTK],[1],[Have host's __chkstk]))
  AC_CHECK_LIB(gcc,___chkstk,AC_DEFINE([HAVE____CHKSTK],[1],[Have host's ___chkstk]))

  AC_CHECK_LIB(gcc,__ashldi3,AC_DEFINE([HAVE___ASHLDI3],[1],[Have host's __ashldi3]))
  AC_CHECK_LIB(gcc,__ashrdi3,AC_DEFINE([HAVE___ASHRDI3],[1],[Have host's __ashrdi3]))
  AC_CHECK_LIB(gcc,__divdi3,AC_DEFINE([HAVE___DIVDI3],[1],[Have host's __divdi3]))
  AC_CHECK_LIB(gcc,__fixdfdi,AC_DEFINE([HAVE___FIXDFDI],[1],[Have host's __fixdfdi]))
  AC_CHECK_LIB(gcc,__fixsfdi,AC_DEFINE([HAVE___FIXSFDI],[1],[Have host's __fixsfdi]))
  AC_CHECK_LIB(gcc,__floatdidf,AC_DEFINE([HAVE___FLOATDIDF],[1],[Have host's __floatdidf]))
  AC_CHECK_LIB(gcc,__lshrdi3,AC_DEFINE([HAVE___LSHRDI3],[1],[Have host's __lshrdi3]))
  AC_CHECK_LIB(gcc,__moddi3,AC_DEFINE([HAVE___MODDI3],[1],[Have host's __moddi3]))
  AC_CHECK_LIB(gcc,__udivdi3,AC_DEFINE([HAVE___UDIVDI3],[1],[Have host's __udivdi3]))
  AC_CHECK_LIB(gcc,__umoddi3,AC_DEFINE([HAVE___UMODDI3],[1],[Have host's __umoddi3]))

  AC_CHECK_LIB(gcc,__main,AC_DEFINE([HAVE___MAIN],[1],[Have host's __main]))
  AC_CHECK_LIB(gcc,__cmpdi2,AC_DEFINE([HAVE___CMPDI2],[1],[Have host's __cmpdi2]))
fi

dnl Check Win32 API EnumerateLoadedModules.
if test "$llvm_cv_os_type" = "MingW" ; then
  AC_MSG_CHECKING([whether EnumerateLoadedModules() accepts new decl])
  AC_COMPILE_IFELSE([[#include <windows.h>
#include <imagehlp.h>
extern void foo(PENUMLOADED_MODULES_CALLBACK);
extern void foo(BOOL(CALLBACK*)(PCSTR,ULONG_PTR,ULONG,PVOID));]],
[
  AC_MSG_RESULT([yes])
  llvm_cv_win32_elmcb_pcstr="PCSTR"
],
[
  AC_MSG_RESULT([no])
  llvm_cv_win32_elmcb_pcstr="PSTR"
])
  AC_DEFINE_UNQUOTED([WIN32_ELMCB_PCSTR],$llvm_cv_win32_elmcb_pcstr,[Type of 1st arg on ELM Callback])
fi

dnl Check for variations in the Standard C++ library and STL. These macros are
dnl provided by LLVM in the autoconf/m4 directory.
AC_FUNC_ISNAN
AC_FUNC_ISINF

dnl Check for mmap support.We also need to know if /dev/zero is required to
dnl be opened for allocating RWX memory.
dnl Make sure we aren't attempting to configure for an unknown system
if test "$llvm_cv_platform_type" = "Unix" ; then
  AC_FUNC_MMAP
  AC_FUNC_MMAP_FILE
  AC_NEED_DEV_ZERO_FOR_MMAP

  if test "$ac_cv_func_mmap_fixed_mapped" = "no"
  then
    AC_MSG_WARN([mmap() of a fixed address required but not supported])
  fi
  if test "$ac_cv_func_mmap_file" = "no"
  then
    AC_MSG_WARN([mmap() of files required but not found])
  fi
fi

dnl atomic builtins are required for threading support.
AC_MSG_CHECKING(for GCC atomic builtins)
dnl Since we'll be using these atomic builtins in C++ files we should test
dnl the C++ compiler.
AC_LANG_PUSH([C++])
AC_LINK_IFELSE(
  AC_LANG_SOURCE(
    [[int main() {
        volatile unsigned long val = 1;
        __sync_synchronize();
        __sync_val_compare_and_swap(&val, 1, 0);
        __sync_add_and_fetch(&val, 1);
        __sync_sub_and_fetch(&val, 1);
        return 0;
      }
    ]]),
  AC_LANG_POP([C++])
  AC_MSG_RESULT(yes)
  AC_DEFINE(LLVM_HAS_ATOMICS, 1, Has gcc/MSVC atomic intrinsics),
  AC_MSG_RESULT(no)
  AC_DEFINE(LLVM_HAS_ATOMICS, 0, Has gcc/MSVC atomic intrinsics)
  AC_MSG_WARN([LLVM will be built thread-unsafe because atomic builtins are missing]))

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 9: Additional checks, variables, etc.
dnl===
dnl===-----------------------------------------------------------------------===

dnl Handle 32-bit linux systems running a 64-bit kernel.
dnl This has to come after section 4 because it invokes the compiler.
if test "$llvm_cv_os_type" = "Linux" -a "$llvm_cv_target_arch" = "x86_64" ; then
  AC_IS_LINUX_MIXED
  if test "$llvm_cv_linux_mixed" = "yes"; then
    llvm_cv_target_arch="x86"
    ARCH="x86"
  fi
fi

dnl Check whether __dso_handle is present
AC_CHECK_FUNCS([__dso_handle])

dnl Propagate the shared library extension that the libltdl checks did to
dnl the Makefiles so we can use it there too
AC_SUBST(SHLIBEXT,$libltdl_cv_shlibext)

dnl Propagate the run-time library path variable that the libltdl
dnl checks found to the Makefiles so we can use it there too
AC_SUBST(SHLIBPATH_VAR,$libltdl_cv_shlibpath_var)

dnl Translate the various configuration directories and other basic
dnl information into substitutions that will end up in Makefile.config.in
dnl that these configured values can be used by the makefiles
if test "${prefix}" = "NONE" ; then
  prefix="/usr/local"
fi
eval LLVM_PREFIX="${prefix}";
eval LLVM_BINDIR="${prefix}/bin";
eval LLVM_LIBDIR="${prefix}/lib";
eval LLVM_DATADIR="${prefix}/share/llvm";
eval LLVM_DOCSDIR="${prefix}/share/doc/llvm";
eval LLVM_ETCDIR="${prefix}/etc/llvm";
eval LLVM_INCLUDEDIR="${prefix}/include";
eval LLVM_INFODIR="${prefix}/info";
eval LLVM_MANDIR="${prefix}/man";
LLVM_CONFIGTIME=`date`
AC_SUBST(LLVM_PREFIX)
AC_SUBST(LLVM_BINDIR)
AC_SUBST(LLVM_LIBDIR)
AC_SUBST(LLVM_DATADIR)
AC_SUBST(LLVM_DOCSDIR)
AC_SUBST(LLVM_ETCDIR)
AC_SUBST(LLVM_INCLUDEDIR)
AC_SUBST(LLVM_INFODIR)
AC_SUBST(LLVM_MANDIR)
AC_SUBST(LLVM_CONFIGTIME)

dnl Disable embedding timestamps in the build directory, with ENABLE_TIMESTAMPS.
if test "${ENABLE_TIMESTAMPS}" = "0"; then
  LLVM_CONFIGTIME="(timestamp not enabled)"
fi

dnl Place the various directories into the config.h file as #defines so that we
dnl can know about the installation paths within LLVM.
AC_DEFINE_UNQUOTED(LLVM_PREFIX,"$LLVM_PREFIX",
                   [Installation prefix directory])
AC_DEFINE_UNQUOTED(LLVM_BINDIR, "$LLVM_BINDIR",
                   [Installation directory for binary executables])
AC_DEFINE_UNQUOTED(LLVM_LIBDIR, "$LLVM_LIBDIR",
                   [Installation directory for libraries])
AC_DEFINE_UNQUOTED(LLVM_DATADIR, "$LLVM_DATADIR",
                   [Installation directory for data files])
AC_DEFINE_UNQUOTED(LLVM_DOCSDIR, "$LLVM_DOCSDIR",
                   [Installation directory for documentation])
AC_DEFINE_UNQUOTED(LLVM_ETCDIR, "$LLVM_ETCDIR",
                   [Installation directory for config files])
AC_DEFINE_UNQUOTED(LLVM_INCLUDEDIR, "$LLVM_INCLUDEDIR",
                   [Installation directory for include files])
AC_DEFINE_UNQUOTED(LLVM_INFODIR, "$LLVM_INFODIR",
                   [Installation directory for .info files])
AC_DEFINE_UNQUOTED(LLVM_MANDIR, "$LLVM_MANDIR",
                   [Installation directory for man pages])
AC_DEFINE_UNQUOTED(LLVM_CONFIGTIME, "$LLVM_CONFIGTIME",
                   [Time at which LLVM was configured])
AC_DEFINE_UNQUOTED(LLVM_DEFAULT_TARGET_TRIPLE, "$target",
                   [Target triple LLVM will generate code for by default])

dnl Determine which bindings to build.
if test "$BINDINGS_TO_BUILD" = auto ; then
  BINDINGS_TO_BUILD=""
  if test "x$OCAMLC" != x -a "x$OCAMLDEP" != x ; then
    BINDINGS_TO_BUILD="ocaml $BINDINGS_TO_BUILD"
  fi
fi
AC_SUBST(BINDINGS_TO_BUILD,$BINDINGS_TO_BUILD)

dnl This isn't really configurey, but it avoids having to repeat the list in
dnl other files.
AC_SUBST(ALL_BINDINGS,ocaml)

dnl Do any work necessary to ensure that bindings have what they need.
binding_prereqs_failed=0
for a_binding in $BINDINGS_TO_BUILD ; do
  case "$a_binding" in
  ocaml)
    if test "x$OCAMLC" = x ; then
      AC_MSG_WARN([--enable-bindings=ocaml specified, but ocamlc not found. Try configure OCAMLC=/path/to/ocamlc])
      binding_prereqs_failed=1
    fi
    if test "x$OCAMLDEP" = x ; then
      AC_MSG_WARN([--enable-bindings=ocaml specified, but ocamldep not found. Try configure OCAMLDEP=/path/to/ocamldep])
      binding_prereqs_failed=1
    fi
    if test "x$OCAMLOPT" = x ; then
      AC_MSG_WARN([--enable-bindings=ocaml specified, but ocamlopt not found. Try configure OCAMLOPT=/path/to/ocamlopt])
      dnl ocamlopt is optional!
    fi
    if test "x$with_ocaml_libdir" != xauto ; then
      AC_SUBST(OCAML_LIBDIR,$with_ocaml_libdir)
    else
      ocaml_stdlib="`"$OCAMLC" -where`"
      if test "$LLVM_PREFIX" '<' "$ocaml_stdlib" -a "$ocaml_stdlib" '<' "$LLVM_PREFIX~"
      then
        # ocaml stdlib is beneath our prefix; use stdlib
        AC_SUBST(OCAML_LIBDIR,$ocaml_stdlib)
      else
        # ocaml stdlib is outside our prefix; use libdir/ocaml
        AC_SUBST(OCAML_LIBDIR,$LLVM_LIBDIR/ocaml)
      fi
    fi
    ;;
  esac
done
if test "$binding_prereqs_failed" = 1 ; then
  AC_MSG_ERROR([Prequisites for bindings not satisfied. Fix them or use configure --disable-bindings.])
fi

dnl Determine whether the compiler supports -fvisibility-inlines-hidden.
AC_CXX_USE_VISIBILITY_INLINES_HIDDEN

dnl Determine linker rpath flag
if test "$llvm_cv_link_use_r" = "yes" ; then
  RPATH="-Wl,-R"
else
  RPATH="-Wl,-rpath"
fi
AC_SUBST(RPATH)

dnl Determine linker rdynamic flag
if test "$llvm_cv_link_use_export_dynamic" = "yes" ; then
  RDYNAMIC="-Wl,-export-dynamic"
else
  RDYNAMIC=""
fi
AC_SUBST(RDYNAMIC)

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 10: Specify the output files and generate it
dnl===
dnl===-----------------------------------------------------------------------===

dnl Configure header files
dnl WARNING: dnl If you add or remove any of the following config headers, then
dnl you MUST also update Makefile so that the variable FilesToConfig
dnl contains the same list of files as AC_CONFIG_HEADERS below. This ensures the
dnl files can be updated automatically when their *.in sources change.
AC_CONFIG_HEADERS([include/llvm/Config/config.h include/llvm/Config/llvm-config.h])
AH_TOP([#ifndef CONFIG_H
#define CONFIG_H])
AH_BOTTOM([#endif])

AC_CONFIG_FILES([include/llvm/Config/Targets.def])
AC_CONFIG_FILES([include/llvm/Config/AsmPrinters.def])
AC_CONFIG_FILES([include/llvm/Config/AsmParsers.def])
AC_CONFIG_FILES([include/llvm/Config/Disassemblers.def])
AC_CONFIG_HEADERS([include/llvm/Support/DataTypes.h])

dnl Configure the makefile's configuration data
AC_CONFIG_FILES([Makefile.config])

dnl Configure the RPM spec file for LLVM
AC_CONFIG_FILES([llvm.spec])

dnl Configure doxygen's configuration file
AC_CONFIG_FILES([docs/doxygen.cfg])

dnl Configure clang, if present
if test "${clang_src_root}" = ""; then
  clang_src_root="$srcdir/tools/clang"
fi
if test -f ${clang_src_root}/README.txt; then
  dnl Use variables to stay under 80 columns.
  configh="include/clang/Config/config.h"
  doxy="docs/doxygen.cfg"
  AC_CONFIG_HEADERS([tools/clang/${configh}:${clang_src_root}/${configh}.in])
  AC_CONFIG_FILES([tools/clang/${doxy}:${clang_src_root}/${doxy}.in])
fi

dnl OCaml findlib META file
AC_CONFIG_FILES([bindings/ocaml/llvm/META.llvm])

dnl Do special configuration of Makefiles
AC_CONFIG_COMMANDS([setup],,[llvm_src="${srcdir}"])
AC_CONFIG_MAKEFILE(Makefile)
AC_CONFIG_MAKEFILE(Makefile.common)
AC_CONFIG_MAKEFILE(examples/Makefile)
AC_CONFIG_MAKEFILE(lib/Makefile)
AC_CONFIG_MAKEFILE(runtime/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile.tests)
AC_CONFIG_MAKEFILE(unittests/Makefile)
AC_CONFIG_MAKEFILE(tools/Makefile)
AC_CONFIG_MAKEFILE(utils/Makefile)
AC_CONFIG_MAKEFILE(projects/Makefile)
AC_CONFIG_MAKEFILE(bindings/Makefile)
AC_CONFIG_MAKEFILE(bindings/ocaml/Makefile.ocaml)

dnl Finally, crank out the output
AC_OUTPUT
