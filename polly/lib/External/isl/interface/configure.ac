AC_INIT([isl-interface], [0], [isl-development@googlegroups.com])
AC_CONFIG_AUX_DIR([.])
AC_CONFIG_MACRO_DIR([../m4])
AM_INIT_AUTOMAKE([foreign])
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])])

AC_PROG_CXX
AX_PROG_CXX_FOR_BUILD

# extract_interface needs to be run on the build system.
# Since this is the only target that is being built,
# simply use the build compiler throughout.
# This ensures that the clang configure checks are
# performed using the right compiler.
CXX="$CXX_FOR_BUILD"
CXXCPP="$CXXCPP_FOR_BUILD"
EXEEXT="$BUILD_EXEEXT"
OBJEXT="$BUILD_OBJEXT"

AX_CXX_COMPILE_STDCXX_11_NO_OVERRIDE

AC_DISABLE_SHARED
LT_INIT

AX_DETECT_CLANG

AC_SUBST([CONFIG_STATUS_DEPENDENCIES], [$LLVM_CONFIG])
AC_CONFIG_HEADERS(isl_config.h)
AC_CONFIG_FILES(Makefile)

AC_OUTPUT
