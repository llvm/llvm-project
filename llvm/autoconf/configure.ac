dnl Autoconf requirements
dnl AC_INIT(package, version, bug-report-address)
dnl information on the package
dnl checks for programs
dnl checks for libraries
dnl checks for header files
dnl checks for types
dnl checks for structures
dnl checks for compiler characteristics
dnl checks for library functions
dnl checks for system services
dnl AC_CONFIG_FILES([file...])
dnl AC_OUTPUT

dnl **************************************************************************
dnl * Initialize
dnl **************************************************************************
AC_INIT([[[LLVM]]],[[[1.1]]],[llvmbugs@cs.uiuc.edu])

dnl Place all of the extra autoconf files into the config subdirectory
AC_CONFIG_AUX_DIR([autoconf])

dnl Quit if the source directory has already been configured.
dnl NOTE: This relies upon undocumented autoconf behavior.
if test ${srcdir} != "."
then
	if test -f ${srcdir}/include/Config/config.h
	then
		AC_MSG_ERROR([Already configured in ${srcdir}])
	fi
fi

dnl
dnl Configure all of the projects present in our source tree.
dnl
for i in `ls ${srcdir}/projects`
do
  if test ${i} != "CVS"
  then
    if test -f ${srcdir}/projects/${i}/configure
    then
      AC_CONFIG_SUBDIRS(projects/${i})
    fi
  fi
done

dnl Configure a header file
AC_CONFIG_HEADERS(include/Config/config.h)
AC_CONFIG_MAKEFILE(Makefile)
AC_CONFIG_MAKEFILE(Makefile.common)
AC_CONFIG_MAKEFILE(lib/Makefile)
AC_CONFIG_MAKEFILE(runtime/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile)
AC_CONFIG_MAKEFILE(test/Makefile.tests)
AC_CONFIG_MAKEFILE(test/QMTest/llvm.py)
AC_CONFIG_MAKEFILE(test/QMTest/llvmdb.py)
AC_CONFIG_MAKEFILE(test/Programs/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/Makefile.programs)
AC_CONFIG_MAKEFILE(test/Programs/TEST.aa.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.dsgraph.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.micro.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.aa.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.example.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.nightly.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.buildrepo.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.jit.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.nightly.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.dsgraph.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.jit.report)
AC_CONFIG_MAKEFILE(test/Programs/TEST.typesafe.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/TEST.dsgraph.gnuplot)
AC_CONFIG_MAKEFILE(test/Programs/TEST.micro.Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/External/SPEC/Makefile.spec)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Makefile.multisrc)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/analyzer/test.in)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/distray/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/distray/test.in)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/fourinarow/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/fourinarow/test.in)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/mason/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/mason/test.in)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/neural/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/neural/test.in)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/pcompress2/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/pcompress2/test.in)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/pifft/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/MultiSource/Benchmarks/FreeBench/pifft/test.in)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Makefile)
AC_CONFIG_MAKEFILE(test/Programs/SingleSource/Makefile.singlesrc)
AC_CONFIG_MAKEFILE(tools/Makefile)
AC_CONFIG_MAKEFILE(utils/Makefile)
AC_CONFIG_MAKEFILE(projects/Makefile)

dnl **************************************************************************
dnl * Determine which system we are building on
dnl **************************************************************************

dnl Check the install program (needs to be done before canonical stuff)
AC_PROG_INSTALL

dnl Check which host for which we're compiling.  This will tell us which LLVM
dnl compiler will be used for compiling SSA into object code.
AC_CANONICAL_TARGET

dnl Set the "OS" Makefile variable based on the system we are building on.
dnl We will use the build machine information to set some variables.
case $build in
	*-*-linux*)
             AC_SUBST(OS,[Linux])
			 if test -d /home/vadve/lattner/local/x86/llvm-gcc
			 then
			   AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/x86/llvm-gcc/])
			 fi
	         ;;

	*-*-solaris*)
             AC_SUBST(OS,[SunOS])
			 if test -d /home/vadve/lattner/local/sparc/llvm-gcc
			 then
	           AC_SUBST(LLVMGCCDIR,[/home/vadve/lattner/local/sparc/llvm-gcc/])
			 fi
	         ;;

	*-*-darwin*)
             AC_SUBST(OS,[Darwin])
	         ;;

	*)       AC_SUBST(OS,[Unknown])
	         ;;
esac

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
case $target in
	i*86-*)           AC_SUBST(ARCH,[x86])
	                  ;;
	sparc*-*)         AC_SUBST(ARCH,[Sparc])
	                  ;;
	powerpc*-*)       AC_SUBST(ARCH,[PowerPC])
	                  ;;
	*)                AC_SUBST(ARCH,[Unknown])
	                  ;;
esac

dnl **************************************************************************
dnl * Check for programs.
dnl **************************************************************************

dnl Check for compilation tools
AC_PROG_CXX
AC_PROG_CC(gcc)
AC_PROG_CPP

dnl Ensure that compilation tools are GCC; we use GCC specific extensions
if test "$GCC" != "yes"
then
	AC_MSG_ERROR([gcc required but not found])
fi

if test "$GXX" != "yes"
then
	AC_MSG_ERROR([g++ required but not found])
fi

dnl Verify that GCC is version 3.0 or higher
gccmajor=`$CC --version | head -n 1 | awk '{print $NF;}' | cut -d. -f1`
if test "$gccmajor" -lt "3"
then
	AC_MSG_ERROR([gcc 3.x required])
fi

dnl Check for GNU Make.  We use its extensions to, so don't build without it
CHECK_GNU_MAKE
if test -z "$_cv_gnu_make_command"
then
	AC_MSG_ERROR([GNU Make required but not found])
fi

dnl Check for compiler-compiler tools (reminds me of Little Caesar's Pizza)
AC_PROG_FLEX
AC_PROG_BISON

dnl Check for libtool
AC_PROG_LIBTOOL

dnl Check for our special programs
AC_PATH_PROG(RPWD,[pwd],[false])
if test ${RPWD} = "false"
then
	AC_MSG_ERROR([pwd required but not found])
fi

AC_PATH_PROG(AR,[ar],[false])
if test ${AR} = "false"
then
	AC_MSG_ERROR([ar required but not found])
fi

AC_PATH_PROG(SED,[sed],[false])
if test ${SED} = "false"
then
	AC_MSG_ERROR([sed required but not found])
fi

AC_PATH_PROG(RM,[rm],[false])
if test ${RM} = "false"
then
	AC_MSG_ERROR([rm required but not found])
fi

AC_PATH_PROG(ECHO,[echo],[false])
if test ${ECHO} = "false"
then
	AC_MSG_ERROR([echo required but not found])
fi

AC_PATH_PROG(MKDIR,[mkdir],[false])
if test ${MKDIR} = "false"
then
	AC_MSG_ERROR([mkdir required but not found])
fi

AC_PATH_PROG(DATE,[date],[false])
if test ${DATE} = "false"
then
	AC_MSG_ERROR([date required but not found])
fi

AC_PATH_PROG(MV,[mv],[false])
if test ${MV} = "false"
then
	AC_MSG_ERROR([mv required but not found])
fi

AC_PATH_PROG(DOT,[dot],[false])

AC_PATH_PROG(ETAGS,[etags],[false])

AC_PATH_PROG(PYTHON,[python],[false])
if test ${PYTHON} = "false"
then
	AC_MSG_WARN([python required but not found])
fi

AC_PATH_PROG(QMTEST,[qmtest],[false])
if test ${QMTEST} = "false"
then
	AC_MSG_WARN([qmtest required but not found])
fi

dnl Verify that the version of python available is high enough for qmtest
pyversion=`$PYTHON -V 2>&1 | cut -d\  -f2`
pymajor=`echo $pyversion | cut -d. -f1`
pyminor=`echo $pyversion | cut -d. -f2`

if test "$pymajor" -ge "2"
then
	if test "$pymajor" -eq "2"
	then
		if test "$pyminor" -lt "2"
		then
			AC_MSG_WARN([Python 2.2 or greater required for qmtest])
		fi
	fi
else
	AC_MSG_WARN([Python 2.2 or greater required for qmtest])
fi

dnl Verify that the source directory is valid
AC_CONFIG_SRCDIR(["Makefile.config.in"])

dnl **************************************************************************
dnl * Check for libraries.
dnl **************************************************************************

dnl libelf is for sparc only; we can ignore it if we don't have it
AC_CHECK_LIB(elf, elf_begin)

dnl dlopen() is required for plugin support.
AC_SEARCH_LIBS(dlopen,dl,AC_DEFINE([HAVE_DLOPEN],[1],[Define if dlopen() is available on this platform.]),AC_MSG_WARN([dlopen() not found - disabling plugin support]))

dnl mallinfo is optional; the code can compile (minus features) without it
AC_SEARCH_LIBS(mallinfo,malloc,AC_DEFINE([HAVE_MALLINFO],[1],[Define if mallinfo() is available on this platform.]))

dnl pthread locking functions are optional - but llvm will not be thread-safe
dnl without locks.
AC_SEARCH_LIBS(pthread_mutex_lock,pthread,AC_DEFINE(HAVE_PTHREAD_MUTEX_LOCK,1,[Define if PThread mutexes (e.g., pthread_mutex_lock) are available in the system's thread library.]))

dnl
dnl The math libraries are used by the test code, but not by the actual LLVM
dnl code.
dnl
dnl AC_CHECK_LIB(m, cos)

dnl **************************************************************************
dnl * Checks for header files.
dnl *	Chances are, if the standard C or POSIX type header files are missing,
dnl *	then LLVM just isn't going to compile.  However, it is possible that
dnl *	the necessary functions/macros will be included from other
dnl *	(non-standard and non-obvious) header files.
dnl *
dnl *	So, we'll be gracious, give it a chance, and try to go on without
dnl *	them.
dnl **************************************************************************
AC_HEADER_STDC
AC_HEADER_SYS_WAIT

dnl Check for ANSI C/POSIX header files
AC_CHECK_HEADERS(assert.h fcntl.h limits.h sys/time.h unistd.h errno.h signal.h math.h)

dnl Check for system specific header files
AC_CHECK_HEADERS(malloc.h sys/mman.h sys/resource.h)

dnl Check for header files associated with dlopen and friends
AC_CHECK_HEADERS(dlfcn.h link.h)

dnl **************************************************************************
dnl * Checks for typedefs, structures, and compiler characteristics.
dnl **************************************************************************

dnl Check for const and inline keywords
AC_C_CONST
AC_C_INLINE

dnl Check for machine endian-ness
AC_C_BIGENDIAN(AC_DEFINE([ENDIAN_BIG],[],[Define if the machine is Big-Endian]),AC_DEFINE([ENDIAN_LITTLE],[],[Define if the machine is Little-Endian]))

dnl Check for types
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_TYPES([int64_t],,AC_MSG_ERROR([Type int64_t required but not found]))
AC_CHECK_TYPES([uint64_t],,AC_MSG_ERROR([Type uint64_t required but not found]))
AC_HEADER_TIME
AC_STRUCT_TM

dnl Check for various C features
AC_C_PRINTF_A

dnl Check for C++ extensions
AC_CXX_HAVE_HASH_MAP
AC_CXX_HAVE_HASH_SET
AC_CXX_HAVE_EXT_SLIST
AC_CXX_HAVE_STD_ITERATOR
AC_CXX_HAVE_BI_ITERATOR
AC_CXX_HAVE_FWD_ITERATOR

dnl **************************************************************************
dnl * Checks for library functions.
dnl **************************************************************************
AC_FUNC_ALLOCA
AC_PROG_GCC_TRADITIONAL
AC_FUNC_MEMCMP
AC_FUNC_MMAP
AC_FUNC_MMAP_FILE
if test ${ac_cv_func_mmap_file} = "no"
then
	AC_MSG_ERROR([mmap() of files required but not found])
fi
AC_HEADER_MMAP_ANONYMOUS
AC_TYPE_SIGNAL
AC_CHECK_FUNCS(getcwd gettimeofday strcspn strdup strerror strspn strstr strtod strtol strtoq strtoll)

dnl
dnl Need to check mmap for MAP_PRIVATE, MAP_ANONYMOUS, MAP_ANON, MAP_FIXED
dnl MAP_FIXED is only needed for Sparc
dnl MAP_ANON is used for Sparc and BSD
dnl Everyone should have MAP_PRIVATE
dnl

dnl Check for certain functions (even if we've already found them) so that we
dnl can quit with an error if they are unavailable.
dnl
dnl As the code is made more portable (i.e. less reliant on these functions,
dnl these checks should go away.
AC_CHECK_FUNC(mmap,,AC_MSG_ERROR([Function mmap() required but not found]))
AC_CHECK_FUNC(mprotect,,AC_MSG_ERROR([Function mprotect() required but not found]))

dnl Determine if the linker supports the -R option.
AC_LINK_USE_R()

dnl **************************************************************************
dnl * Enable various compile-time options
dnl **************************************************************************

dnl Purify Option
AC_ARG_ENABLE(purify,AC_HELP_STRING([--enable-purify],[Compile with purify (default is NO)]),,enableval="no")
if test ${enableval} = "no"
then
	AC_SUBST(ENABLE_PURIFY,[[]])
else
	AC_SUBST(ENABLE_PURIFY,[[ENABLE_PURIFY=1]])
fi

dnl Optimized Option
AC_ARG_ENABLE(optimized,AC_HELP_STRING([--enable-optimized],[Compile with optimizations enabled (default is NO)]),,enableval=no)
if test ${enableval} = "no"
then
	AC_SUBST(ENABLE_OPTIMIZED,[[]])
else
	AC_SUBST(ENABLE_OPTIMIZED,[[ENABLE_OPTIMIZED=1]])
fi

dnl Spec Benchmarks
AC_ARG_ENABLE(spec2000,AC_HELP_STRING([--enable-spec],[Compile SPEC 2000 benchmarks (default is NO)]),,enableval=no)
if test ${enableval} = "no"
then
	if test -d /home/vadve/shared/benchmarks/speccpu2000/benchspec
	then
		AC_SUBST(SPEC_ROOT,[/home/vadve/shared/benchmarks/speccpu2000/benchspec])
		AC_SUBST(USE_SPEC,[[USE_SPEC=1]])
	else
		AC_SUBST(USE_SPEC,[[]])
		AC_SUBST(SPEC_ROOT,[])
	fi
else
	if test ${enableval} = ""
	then
		AC_SUBST(SPEC_ROOT,[/home/vadve/shared/benchmarks/speccpu2000/benchspec])
	else
		AC_SUBST(SPEC_ROOT,[${enableval}])
	fi
	AC_SUBST(USE_SPEC,[[USE_SPEC=1]])
fi

dnl Precompiled Bytecode Option
AC_ARG_ENABLE(precompiled_bytecode,AC_HELP_STRING([--enable-precompiled_bytecode],[Use pre-compiled bytecode (default is NO)]),,enableval=no)
if test ${enableval} = "no"
then
	AC_SUBST(UPB,[[]])
else
	AC_SUBST(UPB,[[USE_PRECOMPILED_BYTECODE=1]])
fi


dnl LLC Diff Option
AC_ARG_ENABLE(llc_diffs,AC_HELP_STRING([--enable-llc_diffs],[Enable LLC Diffs when testing (default is YES)]),,enableval=yes)
if test ${enableval} = "no"
then
	AC_SUBST(DISABLE_LLC_DIFFS,[DISABLE_LLC_DIFFS:=1])
else
	AC_SUBST(DISABLE_LLC_DIFFS,[[]])
fi

dnl JIT Option
AC_ARG_ENABLE(jit,AC_HELP_STRING([--enable-jit],[Enable Just In Time Compiling (default is YES)]),,enableval=default)

if test ${enableval} = "no"
then
	AC_SUBST(JIT,[[]])
else
	case $target in
		*i*86*)
			AC_SUBST(JIT,[[TARGET_HAS_JIT=1]])
			;;
		*sparc*)
			AC_SUBST(JIT,[[TARGET_HAS_JIT=1]])
			;;
		*)
			AC_SUBST(JIT,[[]])
			;;
	esac
fi

dnl **************************************************************************
dnl * Set the location of various third-party software packages
dnl **************************************************************************

dnl Location of the LLVM C front end
AC_ARG_WITH(llvmgccdir,AC_HELP_STRING([--with-llvmgccdir],[Location of LLVM GCC front-end]),AC_SUBST(LLVMGCCDIR,[$withval]))

AC_MSG_CHECKING([for llvm-gcc])
LLVM_GCC_CHECK=no
if test -d "$LLVMGCCDIR"
then
	if test -x "$LLVMGCCDIR/bin/gcc"
	then
		LLVM_GCC_CHECK="$LLVMGCCDIR/bin/gcc"
	fi
fi
llvmgccwarn=no
AC_MSG_RESULT($LLVM_GCC_CHECK)
if test "$LLVM_GCC_CHECK" = "no"
then
    llvmgccwarn=yes
fi

AC_MSG_CHECKING([whether llvm-gcc is sane])
LLVM_GCC_SANE=no
if test -x "$LLVM_GCC_CHECK"
then
	cp /dev/null conftest.c
    "$LLVM_GCC_CHECK" -S -o - conftest.c | grep implementation > /dev/null 2>&1
	if test $? -eq 0
	then
		LLVM_GCC_SANE=yes
	fi
	rm conftest.c
fi
AC_MSG_RESULT($LLVM_GCC_SANE)
if test "$LLVM_GCC_SANE" = "no"
then
	llvmgccwarn=yes
fi

dnl Location of the bytecode repository
AC_ARG_WITH(bcrepos,AC_HELP_STRING([--with-bcrepos],[Location of Bytecode Repository]),AC_SUBST(BCR,[$withval]),AC_SUBST(BCR,[/home/vadve/lattner/LLVMPrograms]))

dnl Location of PAPI
AC_ARG_WITH(papi,AC_HELP_STRING([--with-papi],[Location of PAPI]),AC_SUBST(PAPIDIR,[$withval]),AC_SUBST(PAPIDIR,[/home/vadve/shared/Sparc/papi-2.3.4.1]))

dnl Location of the purify program
AC_ARG_WITH(purify,AC_HELP_STRING([--with-purify],[Location of purify program]),AC_SUBST(PURIFY,[$withval]))

dnl **************************************************************************
dnl * Configure other software packages (via AC_CONFIG_SUBDIRS)
dnl **************************************************************************

dnl **************************************************************************
dnl * Create the output files
dnl **************************************************************************
AC_OUTPUT(Makefile.config)

if test $llvmgccwarn = yes
then
	AC_MSG_WARN([***** llvm C/C++ front end was not found, or does not])
	AC_MSG_WARN([***** appear to be working.])
	AC_MSG_WARN([***** ])
	AC_MSG_WARN([***** Please check configure's --with-llvmgccdir option.])
	AC_MSG_WARN([***** Runtime libraries (in llvm/runtime) will not be built,])
	AC_MSG_WARN([***** but you should be able to build the llvm tools.])
fi
