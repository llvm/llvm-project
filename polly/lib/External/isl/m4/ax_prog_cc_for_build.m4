# ===========================================================================
#   https://www.gnu.org/software/autoconf-archive/ax_prog_cc_for_build.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_PROG_CC_FOR_BUILD
#
# DESCRIPTION
#
#   This macro searches for a C compiler that generates native executables,
#   that is a C compiler that surely is not a cross-compiler. This can be
#   useful if you have to generate source code at compile-time like for
#   example GCC does.
#
#   The macro sets the CC_FOR_BUILD and CPP_FOR_BUILD macros to anything
#   needed to compile or link (CC_FOR_BUILD) and preprocess (CPP_FOR_BUILD).
#   The value of these variables can be overridden by the user by specifying
#   a compiler with an environment variable (like you do for standard CC).
#
#   It also sets BUILD_EXEEXT and BUILD_OBJEXT to the executable and object
#   file extensions for the build platform, and GCC_FOR_BUILD to `yes' if
#   the compiler we found is GCC. All these variables but GCC_FOR_BUILD are
#   substituted in the Makefile.
#
# LICENSE
#
#   Copyright (c) 2008 Paolo Bonzini <bonzini@gnu.org>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved. This file is offered as-is, without any
#   warranty.

#serial 18

AU_ALIAS([AC_PROG_CC_FOR_BUILD], [AX_PROG_CC_FOR_BUILD])
AC_DEFUN([AX_PROG_CC_FOR_BUILD], [dnl
AC_REQUIRE([AC_PROG_CC])dnl
AC_REQUIRE([AC_PROG_CPP])dnl
AC_REQUIRE([AC_CANONICAL_BUILD])dnl

dnl Use the standard macros, but make them use other variable names
dnl
pushdef([ac_cv_prog_CPP], ac_cv_build_prog_CPP)dnl
pushdef([ac_cv_prog_cc_c89], ac_cv_build_prog_cc_c89)dnl
pushdef([ac_cv_prog_gcc], ac_cv_build_prog_gcc)dnl
pushdef([ac_cv_prog_cc_works], ac_cv_build_prog_cc_works)dnl
pushdef([ac_cv_prog_cc_cross], ac_cv_build_prog_cc_cross)dnl
pushdef([ac_cv_prog_cc_g], ac_cv_build_prog_cc_g)dnl
pushdef([ac_cv_c_compiler_gnu], ac_cv_build_c_compiler_gnu)dnl
pushdef([ac_cv_exeext], ac_cv_build_exeext)dnl
pushdef([ac_cv_objext], ac_cv_build_objext)dnl
pushdef([ac_exeext], ac_build_exeext)dnl
pushdef([ac_objext], ac_build_objext)dnl
pushdef([CC], CC_FOR_BUILD)dnl
pushdef([CCDEPMODE], CCDEPMODE_FOR_BUILD)dnl
pushdef([CPP], CPP_FOR_BUILD)dnl
pushdef([GCC], GCC_FOR_BUILD)dnl
pushdef([CFLAGS], CFLAGS_FOR_BUILD)dnl
pushdef([CPPFLAGS], CPPFLAGS_FOR_BUILD)dnl
pushdef([EXEEXT], BUILD_EXEEXT)dnl
pushdef([LDFLAGS], LDFLAGS_FOR_BUILD)dnl
pushdef([OBJEXT], BUILD_OBJEXT)dnl
pushdef([host], build)dnl
pushdef([host_alias], build_alias)dnl
pushdef([host_cpu], build_cpu)dnl
pushdef([host_vendor], build_vendor)dnl
pushdef([host_os], build_os)dnl
pushdef([ac_cv_host], ac_cv_build)dnl
pushdef([ac_cv_host_alias], ac_cv_build_alias)dnl
pushdef([ac_cv_host_cpu], ac_cv_build_cpu)dnl
pushdef([ac_cv_host_vendor], ac_cv_build_vendor)dnl
pushdef([ac_cv_host_os], ac_cv_build_os)dnl
pushdef([ac_tool_prefix], ac_build_tool_prefix)dnl
pushdef([am_cv_CC_dependencies_compiler_type], am_cv_build_CC_dependencies_compiler_type)dnl
pushdef([am__fastdepCC_FALSE], am__fastdepCC_build_FALSE)dnl
pushdef([am__fastdepCC_TRUE], am__fastdepCC_build_TRUE)dnl
pushdef([am_cv_prog_cc_c_o], am_cv_build_prog_cc_c_o)dnl
pushdef([cross_compiling], cross_compiling_build)dnl

cross_compiling_build=no

ac_build_tool_prefix=
AS_IF([test -n "$build"],      [ac_build_tool_prefix="$build-"],
      [test -n "$build_alias"],[ac_build_tool_prefix="$build_alias-"])

AC_LANG_PUSH([C])
AC_PROG_CC
_AC_COMPILER_EXEEXT
_AC_COMPILER_OBJEXT
AC_PROG_CPP

dnl Restore the old definitions
dnl
popdef([cross_compiling])dnl
popdef([am_cv_prog_cc_c_o])dnl
popdef([am__fastdepCC_TRUE])dnl
popdef([am__fastdepCC_FALSE])dnl
popdef([am_cv_CC_dependencies_compiler_type])dnl
popdef([ac_tool_prefix])dnl
popdef([ac_cv_host_os])dnl
popdef([ac_cv_host_vendor])dnl
popdef([ac_cv_host_cpu])dnl
popdef([ac_cv_host_alias])dnl
popdef([ac_cv_host])dnl
popdef([host_os])dnl
popdef([host_vendor])dnl
popdef([host_cpu])dnl
popdef([host_alias])dnl
popdef([host])dnl
popdef([OBJEXT])dnl
popdef([LDFLAGS])dnl
popdef([EXEEXT])dnl
popdef([CPPFLAGS])dnl
popdef([CFLAGS])dnl
popdef([GCC])dnl
popdef([CPP])dnl
popdef([CCDEPMODE])dnl
popdef([CC])dnl
popdef([ac_objext])dnl
popdef([ac_exeext])dnl
popdef([ac_cv_objext])dnl
popdef([ac_cv_exeext])dnl
popdef([ac_cv_c_compiler_gnu])dnl
popdef([ac_cv_prog_cc_g])dnl
popdef([ac_cv_prog_cc_cross])dnl
popdef([ac_cv_prog_cc_works])dnl
popdef([ac_cv_prog_cc_c89])dnl
popdef([ac_cv_prog_gcc])dnl
popdef([ac_cv_prog_CPP])dnl

dnl restore global variables ac_ext, ac_cpp, ac_compile,
dnl ac_link, ac_compiler_gnu (dependant on the current
dnl language after popping):
AC_LANG_POP([C])

dnl Finally, set Makefile variables
dnl
AC_SUBST(BUILD_EXEEXT)dnl
AC_SUBST(BUILD_OBJEXT)dnl
AC_SUBST([CFLAGS_FOR_BUILD])dnl
AC_SUBST([CPPFLAGS_FOR_BUILD])dnl
AC_SUBST([LDFLAGS_FOR_BUILD])dnl
])
