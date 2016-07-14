AC_INIT([ppcg], [0.04], [isl-development@googlegroups.com])
AC_CONFIG_AUX_DIR([.])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign])
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])])

AC_PROG_CC
AC_PROG_LIBTOOL
PKG_PROG_PKG_CONFIG

AX_CHECK_OPENMP
AX_CHECK_OPENCL
if test $HAVE_OPENCL = yes; then
	extra_tests="$extra_tests opencl_test.sh"
fi

AX_SUBMODULE(isl,build|bundled|system,bundled)
AM_CONDITIONAL(BUNDLED_ISL, test $with_isl = bundled)

AC_SUBST(ISL_CFLAGS)
AC_SUBST(ISL_LIBS)
case "$with_isl" in
bundled)
	ISL_CFLAGS="-I\$(top_srcdir)/isl/include -I\$(top_builddir)/isl/include"
	ISL_CFLAGS="$ISL_CFLAGS"
	ppcg_configure_args="$ppcg_configure_args --with-isl-builddir=../isl"
	ppcg_configure_args="$ppcg_configure_args --with-isl=build"
	;;
build)
	ISL_BUILDDIR=`echo @abs_builddir@ | $with_isl_builddir/config.status --file=-`
	ppcg_configure_args="$ppcg_configure_args --with-isl-builddir=$ISL_BUILDDIR"
	ISL_CFLAGS="-I$isl_srcdir/include -I$ISL_BUILDDIR/include"
	ISL_CFLAGS="$ISL_CFLAGS"
	ISL_LIBS="$with_isl_builddir/libisl.la"
	;;
system)
	PKG_CHECK_MODULES([ISL], [isl])
esac

AX_SUBMODULE(pet,bundled|system,bundled)
AM_CONDITIONAL(BUNDLED_PET, test $with_pet = bundled)

AC_SUBST(PET_CFLAGS)
AC_SUBST(PET_LIBS)
case "$with_pet" in
bundled)
	PET_CFLAGS="$PET_CFLAGS -I\$(top_srcdir)/pet/include"
	;;
system)
	PKG_CHECK_MODULES([PET], [pet])
	;;
esac

AC_SUBST(POLYBENCH_DIR)
AC_SUBST(extra_tests)
AC_ARG_WITH([polybench],
	[AS_HELP_STRING([--with-polybench=DIR], [PolyBench location])],
	[
	if test -f "$with_polybench/utilities/benchmark_list"; then
		POLYBENCH_DIR=$with_polybench
		extra_tests="$extra_tests polybench_test.sh"
	fi
	])

AX_DETECT_GIT_HEAD
echo '#define GIT_HEAD_ID "'$GIT_HEAD_ID'"' > gitversion.h

AC_CONFIG_FILES(Makefile)
AC_CONFIG_FILES([polybench_test.sh], [chmod +x polybench_test.sh])
AC_CONFIG_FILES([opencl_test.sh], [chmod +x opencl_test.sh])
if test $with_isl = bundled; then
	AC_CONFIG_SUBDIRS(isl)
fi
if test $with_pet = bundled; then
	AC_CONFIG_SUBDIRS(pet)
fi
AC_CONFIG_COMMANDS_POST([
	dnl pass on arguments to subdir configures, but don't
	dnl add them to config.status
	ac_configure_args="$ac_configure_args $ppcg_configure_args"
])
AC_OUTPUT
