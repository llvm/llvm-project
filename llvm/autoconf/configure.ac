dnl === configure.ac --------------------------------------------------------===
dnl                     The LLVM Compiler Infrastructure
dnl
dnl This file was developed by the LLVM research group and is distributed under
dnl the University of Illinois Open Source License. See LICENSE.TXT for details.
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
dnl email address for reporting bugs.
AC_INIT([[llvm]],[[1.4]],[llvmbugs@cs.uiuc.edu])

dnl Provide a copyright substitution and ensure the copyright notice is included
dnl in the output of --version option of the generated configure script.
AC_SUBST(LLVM_COPYRIGHT,["Copyright (c) 2003-2005 U University of Illinois at Urbana-Champaign."])
AC_COPYRIGHT($LLVM_COPYRIGHT)

dnl Indicate that we require autoconf 2.59 or later. Ths is needed because we
dnl use some autoconf macros only available in 2.59.
AC_PREREQ(2.59)

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

dnl Configure all of the projects present in our source tree. While we could
dnl just AC_CONFIG_SUBDIRS on the set of directories in projects that have a
dnl configure script, that usage of the AC_CONFIG_SUBDIRS macro is deprecated.
dnl Instead we match on the known projects.
for i in `ls ${srcdir}/projects`
do
  if test -d ${srcdir}/projects/${i} ; then
    case ${i} in
      "CVS") ;;
      "sample")       AC_CONFIG_SUBDIRS([projects/sample])    ;;
      "Stacker")      AC_CONFIG_SUBDIRS([projects/Stacker])   ;;
      "llvm-test")    AC_CONFIG_SUBDIRS([projects/llvm-test]) ;;
      "llvm-reopt")   AC_CONFIG_SUBDIRS([projects/llvm-reopt]);;
      "llvm-gcc")     AC_CONFIG_SUBDIRS([projects/llvm-gcc])  ;;
      "Java")         AC_CONFIG_SUBDIRS([projects/Java])      ;;
      "llvm-tv")      AC_CONFIG_SUBDIRS([projects/llvm-tv])   ;;
      "llvm-fefw")    AC_CONFIG_SUBDIRS([projects/llvm-fefw]) ;;
      "poolalloc")    AC_CONFIG_SUBDIRS([projects/poolalloc]) ;;
      *)              
        AC_MSG_WARN([Unknown project (${i}) won't be configured automatically])
        ;;
    esac
  fi
done

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
AC_CACHE_CHECK([support for generic build operating system],
               [llvm_cv_platform_type],
[case $build in
  *-*-aix*) llvm_cv_platform_type="AIX" ;;
  *-*-cygwin*) llvm_cv_platform_type="Cygwin" ;;
  *-*-darwin*) llvm_cv_platform_type="Darwin" ;;
  *-*-freebsd*) llvm_cv_platform_type="FreeBSD" ;;
  *-*-interix*) llvm_cv_platform_type="Interix" ;;
  *-*-linux*) llvm_cv_platform_type="Linux" ;;
  *-*-solaris*) llvm_cv_platform_type="SunOS" ;;
  *-*-win32*) llvm_cv_platform_type="Win32" ;;
  *-*-mingw*) llvm_cv_platform_type="Win32" ;;
  *) llvm_cv_platform_type="Unknown" ;;
esac])

dnl Make sure we aren't attempting to configure for an unknown system
if test "$llvm_cv_platform_type" = "Unknown" ; then
  AC_MSG_ERROR([Operating system platform is unknown, configure can't continue])
fi

dnl Set the "OS" Makefile variable based on the platform type so the
dnl makefile can configure itself to specific build hosts
AC_SUBST(OS,$llvm_cv_platform_type)

dnl Make a link from lib/System/platform to lib/System/$llvm_cv_platform_type
dnl This helps the #inclusion of the system specific include files
dnl for the operating system abstraction library, lib/System.
AC_CONFIG_LINKS(lib/System/platform:lib/System/$llvm_cv_platform_type)

dnl If we are targetting a Sparc machine running Solaris, pretend that it is
dnl V9, since that is all that we support at the moment, and autoconf will only
dnl tell us we're a sparc.
case $target in
  sparc*-*-solaris*)  AC_SUBST(target,[[sparcv9-sun-solaris2.8]])
  ;;
esac

dnl Determine what our target architecture is and configure accordingly.
dnl This will allow Makefiles to make a distinction between the hardware and
dnl the OS.
AC_CACHE_CHECK([target architecture],[llvm_cv_target_arch],
[case $target in
  i*86-*)      llvm_cv_target_arch="x86" ;;
  sparc*-*)    llvm_cv_target_arch="Sparc" ;;
  powerpc*-*)  llvm_cv_target_arch="PowerPC" ;;
  *)           llvm_cv_target_arch="Unknown" ;;
esac])

if test "$llvm_cv_target_arch" = "Unknown" ; then
  AC_MSG_WARN([Configuring LLVM for an unknown target archicture])
fi

dnl Define a substitution, ARCH, for the target architecture
AC_SUBST(ARCH,$llvm_cv_target_arch)

dnl Check for the endianness of the target
AC_C_BIGENDIAN(AC_SUBST([ENDIAN],[big]),AC_SUBST([ENDIAN],[little]))

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 3: Command line arguments for the configure script.
dnl===
dnl===-----------------------------------------------------------------------===

dnl --enable-optimized : check whether they want to do an optimized build:
AC_ARG_ENABLE(optimized,
              AS_HELP_STRING([--enable-optimized,Compile with optimizations enabled (default is NO)]),,enableval=no)
if test ${enableval} = "no" ; then
  AC_SUBST(ENABLE_OPTIMIZED,[[]])
else
  AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]])
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
    x86)     AC_SUBST(JIT,[[TARGET_HAS_JIT=1]]) ;;
    Sparc)   AC_SUBST(JIT,[[TARGET_HAS_JIT=1]]) ;;
    PowerPC) AC_SUBST(JIT,[[TARGET_HAS_JIT=1]]) ;;
    *)       AC_SUBST(JIT,[[]]) ;;
  esac
fi

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

dnl Find the LLVM GCC-based C/C++ front end
AC_ARG_WITH(llvmgccdir,
  AS_HELP_STRING(--with-llvmgccdir,[Location of LLVM GCC front-end]),
  AC_SUBST(LLVMGCCDIR,[$withval]))

dnl Provide some defaults for LLVMGCC for UIUC environments.
if test "x$LLVMGCCDIR" = x ; then
  case "$llvm_cv_platform_type" in
    Linux)
      if test -d /home/vadve/lattner/local/x86/llvm-gcc ; then
        AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/x86/llvm-gcc/])
      fi
      ;;
    SunOS)
      if test -d /home/vadve/lattner/local/sparc/llvm-gcc ; then
        AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/sparc/llvm-gcc/])
      fi
      ;;
  esac
fi

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 4: Check for programs we need and that they are the right version
dnl===
dnl===-----------------------------------------------------------------------===

dnl Check for compilation tools
AC_PROG_CPP
AC_PROG_CC(gcc)
AC_PROG_CXX(g++)
AC_PROG_FLEX
AC_PROG_BISON

dnl Check for the tools that the makefiles require
AC_CHECK_GNU_MAKE
AC_PROG_LN_S
AC_PATH_PROG(DATE, [date], [date])
AC_PATH_PROG(FIND, [find], [find])
AC_PATH_PROG(GREP, [grep], [grep])
AC_PATH_PROG(MKDIR,[mkdir],[mkdir])
AC_PATH_PROG(MV,   [mv],   [mv])
AC_PATH_PROG(PAX,  [pax],  [pax])
AC_PROG_RANLIB
AC_PATH_PROG(RM,   [rm],   [rm])
AC_PATH_PROG(SED,  [sed],  [sed])
AC_PATH_PROG(TAR,  [tar],  [gtar])

dnl Find the install program
AC_PROG_INSTALL

dnl Checks for documentation and testing tools that we can do without. If these
dnl are not found then they are set to "true" which always succeeds but does
dnl nothing. This just lets the build output show that we could have done 
dnl something if the tool was available. 
AC_PATH_PROG(BZIP2,[bzip2],[true bzip2])
AC_PATH_PROG(DOT,[dot],[true dot])
AC_PATH_PROG(DOXYGEN,[doxygen],[true doxygen])
AC_PATH_PROG(ETAGS,[etags],[true etags])
AC_PATH_PROG(GROFF,[groff],[true groff])
AC_PATH_PROG(GZIP,[gzip],[true gzip])
AC_PATH_PROG(POD2HTML,[pod2html],[true pod2html])
AC_PATH_PROG(POD2MAN,[pod2man],[true pod2man])
AC_PATH_PROG(PYTHON,[python],[true python])
AC_PATH_PROG(QMTEST,[qmtest],[true qmtest])
AC_PATH_PROG(RUNTEST,[runtest],[true runtest])
DJ_AC_PATH_TCLSH
AC_PATH_PROG(ZIP,[zip],[true zip])

dnl Determine if the linker supports the -R option.
AC_LINK_USE_R

dnl Check for libtool and the library that has dlopen function (which must come
dnl before the AC_PROG_LIBTOOL check in order to enable dlopening libraries with
dnl libtool).
AC_LIBTOOL_DLOPEN
AC_LIB_LTDL
AC_PROG_LIBTOOL

dnl Check if we know how to tell etags we are using C++:
etags_version=`$ETAGS --version 2>&1`
case "$etags_version" in
	*[Ee]xuberant*) ETAGSFLAGS="--language-force=c++" ;;
	*GNU\ Emacs*) ETAGSFLAGS="-l c++" ;;
	*) ETAGSFLAGS="" ;;
esac
AC_SUBST(ETAGSFLAGS,$ETAGSFLAGS)

AC_MSG_CHECKING([tool compatibility])

dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GCC" != "yes"
then
  AC_MSG_ERROR([gcc required but not found])
fi
dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GXX" != "yes"
then
  AC_MSG_ERROR([g++ required but not found])
fi

dnl Verify that GCC is version 3.0 or higher
gccmajor=`$CC --version | head -n 1 | awk '{print $NF;}' | cut -d. -f1`
if test "$gccmajor" -lt "3"
then
  AC_MSG_ERROR([gcc 3.x required, but you have a lower version])
fi

dnl Check for GNU Make.  We use its extensions, so don't build without it
if test -z "$llvm_cv_gnu_make_command"
then
  AC_MSG_ERROR([GNU Make required but not found])
fi

dnl Tool compatibility is okay if we make it here.
AC_MSG_RESULT([ok])

dnl Generate warnings for things not found but not strictly needed
if test "$PYTHON" = "false" ; then
  AC_MSG_WARN([Python is required for the test suite, but it was not found])
fi
if test "$QMTEST" = "false" ; then
  AC_MSG_WARN([QMTest is required for the test suite, but it was not found])
fi
if test "$RUNTEST" = "false" ; then
  AC_MSG_WARN([runtest (Deja-Gnu) is required for the test sute, but it was not found])
fi
pyversion=`$PYTHON -V 2>&1 | cut -d\  -f2`
pymajor=`echo $pyversion | cut -d. -f1`
pyminor=`echo $pyversion | cut -d. -f2`
if test "$pymajor" -ge "2" ; then
  if test "$pymajor" -eq "2" ; then
    if test "$pyminor" -lt "2" ; then
      AC_MSG_WARN([QMTest requires Python 2.2 or later])
    fi
  fi
else
  AC_MSG_WARN([QMTest requires Python 2.2 or later])
fi

AC_MSG_RESULT([])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 5: Check for libraries
dnl===
dnl===-----------------------------------------------------------------------===

dnl libelf is for sparc only; we can ignore it if we don't have it
AC_CHECK_LIB(elf, elf_begin)

dnl lt_dlopen may be required for plugin support.
AC_SEARCH_LIBS(lt_dlopen,ltdl,AC_DEFINE([HAVE_LT_DLOPEN],[1],
              [Define if lt_dlopen() is available on this platform]),
              AC_MSG_WARN([lt_dlopen() not found - plugin support might 
                           not be available]))

dnl dlopen() is required for plugin support.
AC_SEARCH_LIBS(dlopen,dl,AC_DEFINE([HAVE_DLOPEN],[1],
               [Define if dlopen() is available on this platform.]),
               AC_MSG_WARN([dlopen() not found - disabling plugin support]))

dnl mallinfo is optional; the code can compile (minus features) without it
AC_SEARCH_LIBS(mallinfo,malloc,AC_DEFINE([HAVE_MALLINFO],[1],
               [Define if mallinfo() is available on this platform.]))

dnl pthread locking functions are optional - but llvm will not be thread-safe
dnl without locks.
AC_SEARCH_LIBS(pthread_mutex_lock,pthread,
               AC_DEFINE([HAVE_PTHREAD_MUTEX_LOCK],[1],
                         [Have pthread_mutex_lock]))

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 6: Check for header files
dnl===
dnl===-----------------------------------------------------------------------===

dnl We don't check for ancient stuff or things that are guaranteed to be there
dnl by the C++ standard. We always use the <cfoo> versions of <foo.h> C headers.
AC_HEADER_STDC

dnl However, we do need some system specific header files. Generally we're
dnl looking for POSIX headers.
AC_HEADER_SYS_WAIT
AC_CHECK_HEADERS([sys/time.h sys/mman.h sys/resource.h sys/time.h sys/types.h])
AC_CHECK_HEADERS([dlfcn.h execinfo.h fcntl.h limits.h link.h])
AC_CHECK_HEADERS([malloc.h unistd.h windows.h])
AC_HEADER_TIME
AC_HEADER_MMAP_ANONYMOUS

dnl Check for things that need to be included in public headers, and so
dnl for which we may not have access to a HAVE_* preprocessor #define.
dnl (primarily used in DataTypes.h)
AC_CHECK_HEADERS([inttypes.h stdint.h])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 7: Check for types and structures
dnl===
dnl===-----------------------------------------------------------------------===

AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SIGNAL
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

AC_CHECK_FUNCS([backtrace getcwd gettimeofday isatty getrusage mkstemp]) 
AC_CHECK_FUNCS([strdup strtoq strtoll])
AC_C_PRINTF_A
AC_FUNC_ALLOCA
AC_FUNC_RAND48

dnl Check for variations in the Standard C++ library and STL. These macros are
dnl provided by LLVM in the autoconf/m4 directory.
AC_CXX_HAVE_HASH_MAP
AC_CXX_HAVE_HASH_SET
AC_CXX_HAVE_STD_ITERATOR
AC_CXX_HAVE_BI_ITERATOR
AC_CXX_HAVE_FWD_ITERATOR
AC_FUNC_ISNAN
AC_FUNC_ISINF

dnl Check for mmap and mprotect support. We need both to do the JIT and for
dnl bytecode loading, etc.
AC_FUNC_MMAP
AC_FUNC_MMAP_FILE
AC_CHECK_FUNC(mprotect,,
              AC_MSG_ERROR([Function mprotect() required but not found]))

if test "$ac_cv_func_mmap_fixed_mapped" = "no"
then
  AC_MSG_WARN([mmap() required but not found])
fi
if test "$ac_cv_func_mmap_file" = "no"
then
  AC_MSG_WARN([mmap() of files required but not found])
fi

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 9: Additional checks, variables, etc.
dnl===
dnl===-----------------------------------------------------------------------===

dnl Figure out if there's an executable llvm-gcc in the llvmgccdir.
AC_CACHE_CHECK([for llvm-gcc],[llvm_cv_llvmgcc],
[llvm_cv_llvmgcc='llvmgcc_not_found'
if test -d "$LLVMGCCDIR" ; then
  if test -x "$LLVMGCCDIR/bin/gcc" ; then
    llvm_cv_llvmgcc="$LLVMGCCDIR/bin/gcc"
  fi
fi])

dnl See if the llvmgcc executable can compile to LLVM assembly
AC_CACHE_CHECK([whether llvm-gcc is sane],[llvm_cv_llvmgcc_sanity],
[llvm_cv_llvmgcc_sanity="no"
if test -x "$llvm_cv_llvmgcc" ; then
  cp /dev/null conftest.c
  "$llvm_cv_llvmgcc" -S -o - conftest.c | grep implementation > /dev/null 2>&1
  if test $? -eq 0 ; then
    llvm_cv_llvmgcc_sanity="yes"
  fi
  rm conftest.c
fi])

dnl Since we have a sane llvmgcc, identify it and its sub-tools 
if test "$llvm_cv_llvmgcc_sanity" = "yes" ; then
  AC_SUBST(LLVMGCC,"$llvm_cv_llvmgcc")
  llvmcc1path=`"$llvm_cv_llvmgcc" --print-prog-name=cc1`
  AC_SUBST(LLVMCC1,$llvmcc1path)
  llvmcc1pluspath=`"$llvm_cv_llvmgcc" --print-prog-name=cc1plus`
  AC_SUBST(LLVMCC1PLUS,$llvmcc1pluspath)
fi

dnl Propagate the shared library extension that the libltdl checks did to 
dnl the Makefiles so we can use it there too
AC_SUBST(SHLIBEXT,$libltdl_cv_shlibext)

# Translate the various configuration directories and other basic
# information into substitutions that will end up in Makefile.config.in 
# that these configured values can be used by the makefiles
eval LLVM_PREFIX="${prefix}";
eval LLVM_BINDIR="${prefix}/bin";
eval LLVM_LIBDIR="${prefix}/lib";
eval LLVM_DATADIR="${prefix}/share/llvm";
eval LLVM_DOCSDIR="${prefix}/docs/llvm";
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

# Place the various directores into the config.h file as #defines so that we
# can know about the installation paths within LLVM.
AC_DEFINE_UNQUOTED(LLVM_PREFIX,"$LLVM_PREFIX", 
                   [Installation prefix directory])
AC_DEFINE_UNQUOTED(LLVM_BINDIR, "$LLVM_BINDIR", 
                   [Installation directory for binary executables])
AC_DEFINE_UNQUOTED(LLVM_LIBDIR, "$LLVM_LIBDIR", 
                   [Installation directory for libraries])
AC_DEFINE_UNQUOTED(LLVM_DATADIR, "$LLVM_DATADIR", 
                   [Installation directory for data files])
AC_DEFINE_UNQUOTED(LLVM_DATADIR, "$LLVM_DOCSDIR", 
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

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 10: Specify the output files and generate it
dnl===
dnl===-----------------------------------------------------------------------===

dnl Configure header files
AC_CONFIG_HEADERS(include/llvm/Config/config.h)

AC_CONFIG_HEADERS([include/llvm/Support/DataTypes.h])
AC_CONFIG_HEADERS([include/llvm/ADT/hash_map])
AC_CONFIG_HEADERS([include/llvm/ADT/hash_set])
AC_CONFIG_HEADERS([include/llvm/Support/ThreadSupport.h])
AC_CONFIG_HEADERS([include/llvm/ADT/iterator])

dnl Configure the makefile's configuration data
AC_CONFIG_FILES([Makefile.config])

dnl Configure llvmc's configuration files
AC_CONFIG_FILES([tools/llvmc/st tools/llvmc/cpp tools/llvmc/ll tools/llvmc/c])

dnl Configure doxygen's configuration file
AC_CONFIG_FILES([docs/doxygen.cfg])

dnl Do special configuration of Makefiles
AC_CONFIG_MAKEFILE(Makefile)
AC_CONFIG_MAKEFILE(Makefile.common)
AC_CONFIG_MAKEFILE(examples/Makefile)
AC_CONFIG_MAKEFILE(lib/Makefile)
AC_CONFIG_MAKEFILE(runtime/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile.tests)
AC_CONFIG_MAKEFILE(tools/Makefile)
AC_CONFIG_MAKEFILE(utils/Makefile)
AC_CONFIG_MAKEFILE(projects/Makefile)

dnl Finally, crank out the output
AC_OUTPUT

dnl Warn loudly if llvm-gcc was not obviously working
if test "$llvm_cv_llvmgcc_sanity" = "no" ; then
 AC_MSG_WARN([***** llvm C/C++ front end was not found, or does not])
 AC_MSG_WARN([***** appear to be working.])
 AC_MSG_WARN([***** ])
 AC_MSG_WARN([***** Please check configure's --with-llvmgccdir option.])
 AC_MSG_WARN([***** Runtime libraries (in llvm/runtime) will not be built,])
 AC_MSG_WARN([***** but you should be able to build the llvm tools.])
fi
