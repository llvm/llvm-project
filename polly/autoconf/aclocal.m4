# generated automatically by aclocal 1.11.1 -*- Autoconf -*-

# Copyright (C) 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
# 2005, 2006, 2007, 2008, 2009  Free Software Foundation, Inc.
# This file is free software; the Free Software Foundation
# gives unlimited permission to copy and/or distribute it,
# with or without modifications, as long as this notice is preserved.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY, to the extent permitted by law; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.

dnl find_lib_and_headers(name, verify-header, library-name, requirded?)
dnl Export
dnl         name_inc in -I"include-path" form
dnl         name_lib in -l"library-name" form
dnl         name_ld  in -L"library-path" form
dnl         name_found set to "yes" if found

AC_DEFUN([find_lib_and_headers],
[
  AC_LANG_PUSH(C++)
  OLD_CXXFLAGS=$CXXFLAGS;
  OLD_LDFLAGS=$LDFLAGS;
  OLD_LIBS=$LIBS;

  LIBS="$LIBS -l$3";

  # Get include path and lib path
  AC_ARG_WITH([$1],
    [AS_HELP_STRING([--with-$1], [prefix of $1 ])],
      [given_inc_path="$withval/include"; CXXFLAGS="-I$given_inc_path $CXXFLAGS";
       given_lib_path="$withval/lib"; LDFLAGS="-L$given_lib_path $LDFLAGS"],
      [given_inc_path=inc_not_give_$1;
       given_lib_path=lib_not_give_$1]
    )
  # Check for library and headers works
  AC_MSG_CHECKING([for $1: $2 in $given_inc_path, and lib$3 in $given_lib_path])

  # try to compile a file that includes a header of the library
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <$2>]], [[;]])],
    [AC_MSG_RESULT([ok])
    AC_SUBST([$1_found],["yes"])
    AS_IF([test "x$given_inc_path" != "xinc_not_give_$1"],
      [AC_SUBST([$1_inc],["-I$given_inc_path"])])
    AC_SUBST([$1_lib],["-l$3"])
    AS_IF([test "x$given_lib_path" != "xlib_not_give_$1"],
      [AC_SUBST([$1_ld],["-L$given_lib_path"])])],
    [AS_IF([test "x$4" = "xrequired"],
      [AC_MSG_ERROR([$1 required but not found])],
      [AC_MSG_RESULT([not found])])]
  )

  # reset original CXXFLAGS
  CXXFLAGS=$OLD_CXXFLAGS
  LDFLAGS=$OLD_LDFLAGS;
  LIBS=$OLD_LIBS
  AC_LANG_POP(C++)
])

#
# Provide the arguments and other processing needed for an LLVM project
#
AC_DEFUN([LLVM_CONFIG_PROJECT],
  [AC_ARG_WITH([llvmsrc],
    AS_HELP_STRING([--with-llvmsrc],[Location of LLVM Source Code]),
    [llvm_src="$withval"],[llvm_src="]$1["])
  AC_SUBST(LLVM_SRC,$llvm_src)
  AC_ARG_WITH([llvmobj],
    AS_HELP_STRING([--with-llvmobj],[Location of LLVM Object Code]),
    [llvm_obj="$withval"],[llvm_obj="]$2["])
  AC_SUBST(LLVM_OBJ,$llvm_obj)
  AC_CONFIG_COMMANDS([setup],,[llvm_src="${LLVM_SRC}"])
])

