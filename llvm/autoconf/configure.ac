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
AC_INIT([[llvm]],[[1.8]],[llvmbugs@cs.uiuc.edu])

dnl Provide a copyright substitution and ensure the copyright notice is included
dnl in the output of --version option of the generated configure script.
AC_SUBST(LLVM_COPYRIGHT,["Copyright (c) 2003-2005 University of Illinois at Urbana-Champaign."])
AC_COPYRIGHT([Copyright (c) 2003-2005 University of Illinois at Urbana-Champaign.])

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
      "privbracket")  AC_CONFIG_SUBDIRS([projects/privbracket]) ;;
      "llvm-test")    AC_CONFIG_SUBDIRS([projects/llvm-test]) ;;
      "llvm-reopt")   AC_CONFIG_SUBDIRS([projects/llvm-reopt]);;
      "llvm-gcc")     AC_CONFIG_SUBDIRS([projects/llvm-gcc])  ;;
      "llvm-java")    AC_CONFIG_SUBDIRS([projects/llvm-java]) ;;
      "llvm-tv")      AC_CONFIG_SUBDIRS([projects/llvm-tv])   ;;
      "llvm-poolalloc") AC_CONFIG_SUBDIRS([projects/llvm-poolalloc]) ;;
      "llvm-kernel")  AC_CONFIG_SUBDIRS([projects/llvm-kernel]) ;;
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
AC_CACHE_CHECK([type of operating system we're going to host on],
               [llvm_cv_os_type],
[case $host in
  *-*-aix*) 
    llvm_cv_os_type="AIX" 
    llvm_cv_platform_type="Unix" ;;
  *-*-cygwin*) 
    llvm_cv_os_type="Cygwin" 
    llvm_cv_platform_type="Unix" ;;
  *-*-darwin*) 
    llvm_cv_os_type="Darwin"
    llvm_cv_platform_type="Unix" ;;
  *-*-freebsd*) 
    llvm_cv_os_type="FreeBSD" 
    llvm_cv_platform_type="Unix" ;;
  *-*-openbsd*) 
    llvm_cv_os_type="OpenBSD" 
    llvm_cv_platform_type="Unix" ;;
  *-*-hpux*) 
    llvm_cv_os_type="HP-UX" 
    llvm_cv_platform_type="Unix" ;;
  *-*-interix*) 
    llvm_cv_os_type="Interix"
    llvm_cv_platform_type="Unix" ;;
  *-*-linux*) 
    llvm_cv_os_type="Linux"
    llvm_cv_platform_type="Unix" ;;
  *-*-solaris*) 
    llvm_cv_os_type="SunOS"
    llvm_cv_platform_type="Unix" ;;
  *-*-win32*) 
    llvm_cv_os_type="Win32"
    llvm_cv_platform_type="Win32" ;;
  *-*-mingw*) 
    llvm_cv_os_type="MingW" 
    llvm_cv_platform_type="Win32" ;;
  *) 
    llvm_cv_os_type="Unknown" 
    llvm_cv_platform_type="Unknown" ;;
esac])

dnl Make sure we aren't attempting to configure for an unknown system
if test "$llvm_cv_os_type" = "Unknown" ; then
  AC_MSG_ERROR([Operating system is unknown, configure can't continue])
fi

dnl Set the "OS" Makefile variable based on the platform type so the
dnl makefile can configure itself to specific build hosts
AC_SUBST(OS,$llvm_cv_os_type)

dnl Set the "LLVM_ON_*" variables based on llvm_cvs_platform_type
dnl This is used by lib/System to determine the basic kind of implementation
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
  alpha*-*)               llvm_cv_target_arch="Alpha" ;;
  ia64-*)                 llvm_cv_target_arch="IA64" ;;
  *)                      llvm_cv_target_arch="Unknown" ;;
esac])

if test "$llvm_cv_target_arch" = "Unknown" ; then
  AC_MSG_WARN([Configuring LLVM for an unknown target archicture])
fi

dnl Define a substitution, ARCH, for the target architecture
AC_SUBST(ARCH,$llvm_cv_target_arch)

dnl Check for the endianness of the target
AC_C_BIGENDIAN(AC_SUBST([ENDIAN],[big]),AC_SUBST([ENDIAN],[little]))

dnl Check for build platform executable suffix if we're crosscompiling
if test "$cross_compiling" = yes; then
  AC_SUBST(LLVM_CROSS_COMPILING, [1])  
  AC_BUILD_EXEEXT
else
  AC_SUBST(LLVM_CROSS_COMPILING, [0])
fi

dnl Check to see if there's a "CVS" directory indicating that this build is
dnl being done from a CVS checkout. This sets up several defaults for the
dnl command line switches. When we build with a CVS directory, we get a 
dnl debug with assertions turned on. Without, we assume a source release and we
dnl get an optimized build without assertions. See --enable-optimized and
dnl --enable-assertions below
if test -d "CVS" -o -d "${srcdir}/CVS"; then
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

dnl --enable-optimized : check whether they want to do an optimized build:
AC_ARG_ENABLE(optimized, AS_HELP_STRING(
 [--enable-optimized,Compile with optimizations enabled (default is NO)]),,enableval=$optimize)
if test ${enableval} = "no" ; then
  AC_SUBST(ENABLE_OPTIMIZED,[[]])
else
  AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]])
fi

dnl --enable-assertions : check whether they want to turn on assertions or not:
AC_ARG_ENABLE(assertions,AS_HELP_STRING(
  [--enable-assertions,Compile with assertion checks enabled (default is YES)]),, enableval="yes")
if test ${enableval} = "yes" ; then
  AC_SUBST(DISABLE_ASSERTIONS,[[]])
else
  AC_SUBST(DISABLE_ASSERTIONS,[[DISABLE_ASSERTIONS=1]])
fi

dnl --enable-debug-runtime : should runtime libraries have debug symbols?
AC_ARG_ENABLE(debug-runtime,
   AS_HELP_STRING([--enable-debug-runtime,Build runtime libs with debug symbols (default is NO)]),,enableval=no)
if test ${enableval} = "no" ; then
  AC_SUBST(DEBUG_RUNTIME,[[]])
else
  AC_SUBST(DEBUG_RUNTIME,[[DEBUG_RUNTIME=1]])
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
    x86)     AC_SUBST(TARGET_HAS_JIT,1) ;;
    Sparc)   AC_SUBST(TARGET_HAS_JIT,1) ;;
    PowerPC) AC_SUBST(TARGET_HAS_JIT,1) ;;
    x86_64)  AC_SUBST(TARGET_HAS_JIT,0) ;;
    Alpha)   AC_SUBST(TARGET_HAS_JIT,1) ;;
    IA64)    AC_SUBST(TARGET_HAS_JIT,0) ;;
    *)       AC_SUBST(TARGET_HAS_JIT,0) ;;
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

dnl Allow disablement of threads
AC_ARG_ENABLE(threads,
              AS_HELP_STRING([--enable-threads],
                             [Use threads if available (default is YES)]),,
                             enableval=yes)
case "$enableval" in
  yes) AC_SUBST(ENABLE_THREADS,[1]) ;;
  no)  AC_SUBST(ENABLE_THREADS,[0]) ;;
  *) AC_MSG_ERROR([Invalid setting for --enable-threads. Use "yes" or "no"]) ;;
esac
AC_DEFINE_UNQUOTED([ENABLE_THREADS],$ENABLE_THREADS,[Define if threads enabled])

dnl Allow specific targets to be specified for building (or not)
TARGETS_TO_BUILD=""
AC_ARG_ENABLE([targets],AS_HELP_STRING([--enable-targets],
    [Build specific host targets: all,host-only,{target-name} (default=all)]),,
    enableval=all)
case "$enableval" in
  all) TARGETS_TO_BUILD="X86 Sparc PowerPC Alpha IA64 ARM" ;;
  host-only)
    case "$llvm_cv_target_arch" in
      x86)     TARGETS_TO_BUILD="X86" ;;
      x86_64)  TARGETS_TO_BUILD="X86" ;;
      Sparc)   TARGETS_TO_BUILD="Sparc" ;;
      PowerPC) TARGETS_TO_BUILD="PowerPC" ;;
      Alpha)   TARGETS_TO_BUILD="Alpha" ;;
      IA64)    TARGETS_TO_BUILD="IA64" ;;
      *)       AC_MSG_ERROR([Can not set target to build]) ;;
    esac 
    ;;
  *)for a_target in `echo $enableval|sed -e 's/,/ /g' ` ; do
      case "$a_target" in
        x86)     TARGETS_TO_BUILD="X86 $TARGETS_TO_BUILD" ;;
        x86_64)  TARGETS_TO_BUILD="X86 $TARGETS_TO_BUILD" ;;
        sparc)   TARGETS_TO_BUILD="Sparc $TARGETS_TO_BUILD" ;;
        powerpc) TARGETS_TO_BUILD="PowerPC $TARGETS_TO_BUILD" ;;
        alpha)   TARGETS_TO_BUILD="Alpha $TARGETS_TO_BUILD" ;;
        ia64)    TARGETS_TO_BUILD="IA64 $TARGETS_TO_BUILD" ;;
        *) AC_MSG_ERROR([Unrecognized target $a_target]) ;;
      esac
  done 
  ;;
esac
TARGETS_TO_BUILD="CBackend $TARGETS_TO_BUILD"
AC_SUBST(TARGETS_TO_BUILD,$TARGETS_TO_BUILD)

dnl Allow a specific llvm-gcc/llvm-g++ pair to be used with this LLVM config.
AC_ARG_WITH(llvmgccdir,
  AS_HELP_STRING([--with-llvmgccdir],
    [Specify location of llvm-gcc install dir (default searches PATH)]),,
    withval=default)
case "$withval" in
  default) WITH_LLVMGCCDIR=default ;;
  /* | [[A-Za-z]]:[[\\/]]*)      WITH_LLVMGCCDIR=$withval ;;
  *) AC_MSG_ERROR([Invalid path for --with-llvmgccdir. Provide full path]) ;;
esac

dnl Specify extra build options
AC_ARG_WITH(extra-options,
  AS_HELP_STRING([--with-extra-options],
                 [Specify addtional options to compile LLVM with]),,
                 extraopts=default)
case "$extraopts" in
  default) EXTRA_OPTIONS= ;;
  *) EXTRA_OPTIONS=$extraopts ;;
esac
AC_SUBST(EXTRA_OPTIONS,$EXTRA_OPTIONS)

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
AC_PATH_PROG(CMP, [cmp], [cmp])
AC_PATH_PROG(CP, [cp], [cp])
AC_PATH_PROG(DATE, [date], [date])
AC_PATH_PROG(FIND, [find], [find])
AC_PATH_PROG(GREP, [grep], [grep])
AC_PATH_PROG(MKDIR,[mkdir],[mkdir])
AC_PATH_PROG(MV,   [mv],   [mv])
AC_PROG_RANLIB
AC_PATH_PROG(RM,   [rm],   [rm])
AC_PATH_PROG(SED,  [sed],  [sed])
AC_PATH_PROG(TAR,  [tar],  [gtar])

dnl Looking for misc. graph plotting software
AC_PATH_PROG(GRAPHVIZ, [Graphviz], [echo Graphviz])
if test "$GRAPHVIZ" != "echo Graphviz" ; then
  AC_DEFINE([HAVE_GRAPHVIZ],[1],[Define if the Graphviz program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" == "MingW" ; then
    GRAPHVIZ=`echo $GRAPHVIZ | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_GRAPHVIZ],"$GRAPHVIZ${EXEEXT}",
   [Define to path to Graphviz program if found or 'echo Graphviz' otherwise])
fi
AC_PATH_PROG(DOT, [dot], [echo dot])
if test "$DOT" != "echo dot" ; then
  AC_DEFINE([HAVE_DOT],[1],[Define if the dot program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" == "MingW" ; then
    DOT=`echo $DOT | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_DOT],"$DOT${EXEEXT}",
   [Define to path to dot program if found or 'echo dot' otherwise])
fi
AC_PATH_PROGS(GV, [gv gsview32], [echo gv])
if test "$GV" != "echo gv" ; then
  AC_DEFINE([HAVE_GV],[1],[Define if the gv program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" == "MingW" ; then
    GV=`echo $GV | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_GV],"$GV${EXEEXT}",
   [Define to path to gv program if found or 'echo gv' otherwise])
fi
AC_PATH_PROG(DOTTY, [dotty], [echo dotty])
if test "$DOTTY" != "echo dotty" ; then
  AC_DEFINE([HAVE_DOTTY],[1],[Define if the dotty program is available])
  dnl If we're targeting for mingw we should emit windows paths, not msys
  if test "$llvm_cv_os_type" == "MingW" ; then
    DOTTY=`echo $DOTTY | sed 's/^\/\([[A-Za-z]]\)\//\1:\//' `
  fi
  AC_DEFINE_UNQUOTED([LLVM_PATH_DOTTY],"$DOTTY${EXEEXT}",
   [Define to path to dotty program if found or 'echo dotty' otherwise])
fi

dnl Look for a sufficiently recent version of Perl.
LLVM_PROG_PERL([5.006])
AC_SUBST(PERL)
if test x"$PERL" = xnone; then
   AC_SUBST(HAVE_PERL,0)
else
   AC_SUBST(HAVE_PERL,1)
fi

dnl Find the install program
AC_PROG_INSTALL

dnl Checks for documentation and testing tools that we can do without. If these
dnl are not found then they are set to "true" which always succeeds but does
dnl nothing. This just lets the build output show that we could have done 
dnl something if the tool was available. 
AC_PATH_PROG(BZIP2,[bzip2],[echo "Skipped: bzip2 not found"])
AC_PATH_PROG(DOXYGEN,[doxygen],[echo "Skipped: doxygen not found"])
AC_PATH_PROG(ETAGS,[etags],[echo "Skipped: etags not found"])
AC_PATH_PROG(GROFF,[groff],[echo "Skipped: groff not found"])
AC_PATH_PROG(GZIP,[gzip],[echo "Skipped: gzip not found"])
AC_PATH_PROG(POD2HTML,[pod2html],[echo "Skipped: pod2html not found"])
AC_PATH_PROG(POD2MAN,[pod2man],[echo "Skipped: pod2man not found"])
AC_PATH_PROG(RUNTEST,[runtest],[echo "Skipped: runtest not found"])
DJ_AC_PATH_TCLSH
AC_PATH_PROG(ZIP,[zip],[echo "Skipped: zip not found"])

dnl Determine if the linker supports the -R option.
AC_LINK_USE_R

dnl Check for libtool and the library that has dlopen function (which must come
dnl before the AC_PROG_LIBTOOL check in order to enable dlopening libraries with
dnl libtool).
AC_LIBTOOL_DLOPEN
AC_LIB_LTDL
AC_PROG_LIBTOOL

if test "$lt_cv_dlopen_self" = "yes" ; then
  AC_DEFINE([CAN_DLOPEN_SELF],[1],
              [Define if dlopen(0) will open the symbols of the program])
fi

dnl Check if we know how to tell etags we are using C++:
etags_version=`$ETAGS --version 2>&1`
case "$etags_version" in
	*[Ee]xuberant*) ETAGSFLAGS="--language-force=c++" ;;
	*GNU\ Emacs*) ETAGSFLAGS="-l c++" ;;
	*) ETAGSFLAGS="" ;;
esac
AC_SUBST(ETAGSFLAGS,$ETAGSFLAGS)

if test "$WITH_LLVMGCCDIR" = "default" ; then
  LLVMGCC="llvm-gcc${EXEEXT}"
  LLVMGXX="llvm-g++${EXEEXT}"
  AC_PATH_PROG(LLVMGCC, $LLVMGCC, [])
  AC_PATH_PROG(LLVMGXX, $LLVMGXX, [])
else
  LLVMGCC="$WITH_LLVMGCCDIR/bin/llvm-gcc${EXEEXT}"
  LLVMGXX="$WITH_LLVMGCCDIR/bin/llvm-g++${EXEEXT}"
  AC_SUBST(LLVMGCC,$LLVMGCC)
  AC_SUBST(LLVMGXX,$LLVMGXX)
fi
  
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

dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GXX" != "yes" && test "$IXX" != "yes"
then
  AC_MSG_ERROR([g++|icc required but not found])
fi

dnl Verify that GCC is version 3.0 or higher
if test "$GCC" = "yes"
then
  gccmajor=`$CC --version | head -n 1 | sed 's/[[^0-9]]*\([[0-9.]]\).*/\1/'` 
  if test "$gccmajor" -lt "3"
  then
    AC_MSG_ERROR([gcc 3.x required, but you have a lower version])
  fi
fi

dnl Check for GNU Make.  We use its extensions, so don't build without it
if test -z "$llvm_cv_gnu_make_command"
then
  AC_MSG_ERROR([GNU Make required but not found])
fi

dnl Tool compatibility is okay if we make it here.
AC_MSG_RESULT([ok])

dnl===-----------------------------------------------------------------------===
dnl===
dnl=== SECTION 5: Check for libraries
dnl===
dnl===-----------------------------------------------------------------------===

dnl libelf is for sparc only; we can ignore it if we don't have it
AC_CHECK_LIB(elf, elf_begin)
AC_CHECK_LIB(m,sin)
if test "$llvm_cv_os_type" == "MingW" ; then
  AC_CHECK_LIB(imagehlp, main)
  AC_CHECK_LIB(psapi, main)
fi

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
if test "$ENABLE_THREADS" -eq 1 ; then
  AC_CHECK_LIB(pthread,pthread_mutex_init)
  AC_SEARCH_LIBS(pthread_mutex_lock,pthread,
                 AC_DEFINE([HAVE_PTHREAD_MUTEX_LOCK],[1],
                           [Have pthread_mutex_lock]))
fi

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
AC_HEADER_STDC
AC_HEADER_SYS_WAIT
AC_HEADER_TIME

AC_CHECK_HEADERS([dlfcn.h execinfo.h fcntl.h inttypes.h limits.h link.h])
AC_CHECK_HEADERS([malloc.h setjmp.h signal.h stdint.h unistd.h utime.h])
AC_CHECK_HEADERS([windows.h sys/mman.h sys/param.h sys/resource.h sys/time.h])
AC_CHECK_HEADERS([sys/types.h malloc/malloc.h])
if test "$ENABLE_THREADS" -eq 1 ; then
  AC_CHECK_HEADERS(pthread.h)
fi

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

AC_CHECK_FUNCS([backtrace ceilf floorf roundf rintf nearbyintf getcwd ])
AC_CHECK_FUNCS([getpagesize getrusage gettimeofday isatty mkdtemp mkstemp ])
AC_CHECK_FUNCS([mktemp realpath sbrk setrlimit strdup strerror strerror_r ])
AC_CHECK_FUNCS([strtoll strtoq sysconf malloc_zone_statistics ])
AC_CHECK_FUNCS([_setjmp _longjmp setjmp longjmp sigsetjmp siglongjmp])
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
dnl bytecode loading, etc. We also need to know if /dev/zero is required to
dnl be opened for allocating RWX memory.
AC_FUNC_MMAP
AC_FUNC_MMAP_FILE
AC_NEED_DEV_ZERO_FOR_MMAP
AC_CHECK_FUNC(mprotect,,
              AC_MSG_ERROR([Function mprotect() required but not found]))

if test "$ac_cv_func_mmap_fixed_mapped" = "no"
then
  AC_MSG_WARN([mmap() of a fixed address required but not supported])
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

dnl See if the llvm-gcc executable can compile to LLVM assembly
AC_CACHE_CHECK([whether llvm-gcc is sane],[llvm_cv_llvmgcc_sanity],
[llvm_cv_llvmgcc_sanity="no"
if test -x "$LLVMGCC" ; then
  cp /dev/null conftest.c
  "$LLVMGCC" -emit-llvm -S -o - conftest.c | grep implementation > /dev/null 2>&1
  if test $? -eq 0 ; then
    llvm_cv_llvmgcc_sanity="yes"
  fi
  rm conftest.c
fi])

dnl Since we have a sane llvm-gcc, identify it and its sub-tools 
if test "$llvm_cv_llvmgcc_sanity" = "yes" ; then
  llvmcc1path=`"$LLVMGCC" --print-prog-name=cc1`
  AC_SUBST(LLVMCC1,$llvmcc1path)
  llvmcc1pluspath=`"$LLVMGCC" --print-prog-name=cc1plus`
  AC_SUBST(LLVMCC1PLUS,$llvmcc1pluspath)
  llvmgccdir=`echo "$llvmcc1path" | sed 's,/libexec/.*,,'`
  AC_SUBST(LLVMGCCDIR,$llvmgccdir)
  llvmgccversion=[`"$LLVMGCC" -dumpversion 2>&1 | sed 's/^\([0-9.]*\).*/\1/'`]
  llvmgccmajvers=[`echo $llvmgccversion | sed 's/^\([0-9]\).*/\1/'`]
  AC_SUBST(LLVMGCC_VERSION,$llvmgccversion)
  AC_SUBST(LLVMGCC_MAJVERS,$llvmgccmajvers)
fi

dnl Propagate the shared library extension that the libltdl checks did to 
dnl the Makefiles so we can use it there too
AC_SUBST(SHLIBEXT,$libltdl_cv_shlibext)

# Translate the various configuration directories and other basic
# information into substitutions that will end up in Makefile.config.in 
# that these configured values can be used by the makefiles
if test "${prefix}" == "NONE" ; then
  prefix="/usr/local"
fi
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
dnl WARNING: dnl If you add or remove any of the following config headers, then
dnl you MUST also update Makefile.rules so that the variable FilesToConfig
dnl contains the same list of files as AC_CONFIG_HEADERS below. This ensures the
dnl files can be updated automatically when their *.in sources change.
AC_CONFIG_HEADERS([include/llvm/Config/config.h])
AC_CONFIG_HEADERS([include/llvm/Support/DataTypes.h])
AC_CONFIG_HEADERS([include/llvm/ADT/hash_map])
AC_CONFIG_HEADERS([include/llvm/ADT/hash_set])
AC_CONFIG_HEADERS([include/llvm/ADT/iterator])

dnl Configure the makefile's configuration data
AC_CONFIG_FILES([Makefile.config])

dnl Configure doxygen's configuration file
AC_CONFIG_FILES([docs/doxygen.cfg])

dnl Do the first stage of configuration for llvm-config.in.
AC_CONFIG_FILES([tools/llvm-config/llvm-config.in])

dnl Do special configuration of Makefiles
AC_CONFIG_COMMANDS([setup],,[llvm_src="${srcdir}"])
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
 AC_MSG_WARN([***** llvm-gcc/llvm-g++ was not found, or does not appear to be ])
 AC_MSG_WARN([***** working. Please make sure you have llvmgcc and llvmg++ in])
 AC_MSG_WARN([***** your path before configuring LLVM. The runtime libraries])
 AC_MSG_WARN([***** (llvm/runtime) will not be built but you should be able to])
 AC_MSG_WARN([***** build the llvm tools.])
fi
