AC_INIT([isl], [0.23], [isl-development@googlegroups.com])
AC_CONFIG_AUX_DIR([.])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])])
AC_SUBST(versioninfo)
versioninfo=23:0:0

if test "x$prefix" != "xNONE"; then
	prefix_wd=`cd $prefix && pwd`
	srcdir_wd=`cd $srcdir && pwd`
	wd=`pwd`
	if test "x$prefix_wd" = "x$srcdir_wd"; then
		AC_MSG_ERROR(Installation in source directory not supported)
	fi
	if test "x$prefix_wd" = "x$wd"; then
		AC_MSG_ERROR(Installation in build directory not supported)
	fi
fi

AC_PROG_CC
AC_PROG_CXX
AX_PROG_CC_FOR_BUILD

AX_CC_MAXOPT
AX_GCC_WARN_UNUSED_RESULT
AX_C___ATTRIBUTE__

AX_CXX_COMPILE_STDCXX_11_NO_OVERRIDE

AC_PROG_GREP
AC_PROG_LIBTOOL
AC_PROG_SED
AM_PATH_PYTHON([2.5], [], [:])
AM_CONDITIONAL([HAVE_PYTHON], [test "$PYTHON" != :])

AC_CHECK_PROG(CYGPATH, cygpath, cygpath, [])
AC_CHECK_PROG(PERL, perl, perl, [])
AC_CHECK_PROG(PDFLATEX, pdflatex, pdflatex, [])
AC_CHECK_PROG(POD2HTML, pod2html, pod2html, [])

AC_SUBST(OS_SRCDIR)
if test "$host_os" = "mingw32" -a -n "$CYGPATH"; then
	OS_SRCDIR=`$CYGPATH -m "$srcdir"`
else
	OS_SRCDIR="$srcdir"
fi

AM_CONDITIONAL(GENERATE_DOC, test -n "$PERL" -a -n "$PDFLATEX" -a -n "$POD2HTML")

AX_CREATE_STDINT_H(include/isl/stdint.h)

AC_ARG_WITH([int],
	    [AS_HELP_STRING([--with-int=gmp|imath|imath-32],
			    [Which package to use to represent
				multi-precision integers [default=gmp]])],
	    [], [with_int=gmp])
case "$with_int" in
gmp|imath|imath-32)
	;;
*)
	AC_MSG_ERROR(
	    [bad value ${withval} for --with-int (use gmp, imath or imath-32)])
esac

AC_SUBST(MP_CPPFLAGS)
AC_SUBST(MP_CFLAGS)
AC_SUBST(MP_LDFLAGS)
AC_SUBST(MP_LIBS)
case "$with_int" in
gmp)
	AX_DETECT_GMP
	;;
imath|imath-32)
	AX_DETECT_IMATH
	;;
esac
if test "x$with_int" = "ximath-32" -a "x$GCC" = "xyes"; then
	MP_CFLAGS="-std=gnu99 $MP_CFLAGS"
fi

AM_CONDITIONAL(IMATH_FOR_MP, test x$with_int = ximath -o x$with_int = ximath-32)
AM_CONDITIONAL(GMP_FOR_MP, test x$with_int = xgmp)

AM_CONDITIONAL(HAVE_CXX11, test "x$HAVE_CXX11" = "x1")
AM_CONDITIONAL(SMALL_INT_OPT, test "x$with_int" == "ximath-32")
AS_IF([test "x$with_int" == "ximath-32"], [
	AC_DEFINE([USE_SMALL_INT_OPT], [], [Use small integer optimization])
])

AC_CHECK_DECLS(ffs,[],[],[#include <strings.h>])
AC_CHECK_DECLS(__builtin_ffs,[],[],[])
AC_CHECK_DECLS([_BitScanForward],[],[],[#include <intrin.h>])
if test "x$ac_cv_have_decl_ffs" = xno -a \
		"x$ac_cv_have_decl___builtin_ffs" = xno -a \
		"x$ac_cv_have_decl__BitScanForward" = xno; then
	AC_MSG_ERROR([No ffs implementation found])
fi
AC_CHECK_DECLS([strcasecmp,strncasecmp],[],[],[#include <strings.h>])
AC_CHECK_DECLS([_stricmp,_strnicmp],[],[],[#include <string.h>])
if test "x$ac_cv_have_decl_strcasecmp" = xno -a \
		"x$ac_cv_have_decl__stricmp" = xno; then
	AC_MSG_ERROR([No strcasecmp implementation found])
fi
if test "x$ac_cv_have_decl_strncasecmp" = xno -a \
		"x$ac_cv_have_decl__strnicmp" = xno; then
	AC_MSG_ERROR([No strncasecmp implementation found])
fi
AC_CHECK_DECLS([snprintf,_snprintf],[],[],[#include <stdio.h>])
if test "x$ac_cv_have_decl_snprintf" = xno -a \
		"x$ac_cv_have_decl__snprintf" = xno; then
	AC_MSG_ERROR([No snprintf implementation found])
fi

AX_SUBMODULE(clang,system|no,no)
AM_CONDITIONAL(HAVE_CLANG, test $with_clang = system)
AM_CONDITIONAL(HAVE_CPP_ISL_H,
    [(test $with_clang = system -a "x$HAVE_CXX11" = "x1") || \
     test -f $srcdir/include/isl/cpp.h])

AX_SET_WARNING_FLAGS

AC_SUBST(WARNING_FLAGS)

PACKAGE_CFLAGS="$MP_CPPFLAGS"
PACKAGE_LDFLAGS="$MP_LDFLAGS"
PACKAGE_LIBS="-lisl $MP_LIBS"
AX_CREATE_PKGCONFIG_INFO

AX_DETECT_GIT_HEAD

AH_BOTTOM([#include <isl_config_post.h>])
AC_CONFIG_HEADERS(isl_config.h)
AC_CONFIG_FILES(isl_srcdir.c)
AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES(doc/Makefile)
if test $with_clang = system; then
	AC_CONFIG_SUBDIRS(interface)
fi
AC_CONFIG_FILES([bound_test.sh], [chmod +x bound_test.sh])
AC_CONFIG_FILES([codegen_test.sh], [chmod +x codegen_test.sh])
AC_CONFIG_FILES([pip_test.sh], [chmod +x pip_test.sh])
AC_CONFIG_FILES([flow_test.sh], [chmod +x flow_test.sh])
AC_CONFIG_FILES([schedule_test.sh], [chmod +x schedule_test.sh])
AC_OUTPUT
