dnl acinclude.m4 -- configure macros

dnl Copyright 2009 The Go Authors. All rights reserved.
dnl Use of this source code is governed by a BSD-style
dnl license that can be found in the LICENSE file.

dnl Go support--this could be in autoconf.
dnl This version is probably autoconf 2.64 specific.

AC_LANG_DEFINE([Go], [go], [GO], [],
[ac_ext=go
ac_compile='$GOC -c $GOCFLAGS conftest.$ac_ext >&AS_MESSAGE_LOG_FD'
ac_link='$GOC -o conftest$ac_exeext $GOCFLAGS $LDFLAGS conftest.$ac_ext $LIBS >&AS_MESSAGE_LOG_FD'
ac_compile_gnu=yes
])

AU_DEFUN([AC_LANG_GO], [AC_LANG(Go)])

m4_define([AC_LANG_PROGRAM(Go)],
[package main
$1
func main() {
$2
}])

m4_define([AC_LANG_IO_PROGRAM(Go)],
[AC_LANG_PROGRAM([import "os"],
[if f, err := os.Open("conftest.out", os.O_WRONLY), err != nil {
	os.Exit(1);
 }
 if err := f.Close(); err != nil {
	os.Exit(1);
 }
 os.Exit(0);
])])

m4_define([AC_LANG_CALL(Go)],
[AC_LANG_PROGRAM([$1
m4_if([$2], [main], ,
[func $2();])],[$2();])])

m4_define([AC_LANG_FUNC_LINK_TRY(Go)],
[AC_LANG_PROGRAM(
[func $1() int;
var f := $1;
], [return f();])])

m4_define([AC_LANG_BOOL_COMPILE_TRY(Go)],
[AC_LANG_PROGRAM([$1], [var test_array @<:@1 - 2 * !($2)@:>@;
test_array @<:@0@:>@ = 0
])])

m4_define([AC_LANG_INT_SAVE(Go)],
[AC_LANG_PROGRAM([$1
import os
func longval() long { return $2 }
func ulongval() ulong { return $2 }],
[panic("unimplemented")])])

AC_DEFUN([AC_LANG_COMPILER(Go)],
[AC_REQUIRE([AC_PROG_GO])])

AN_MAKEVAR([GOC], [AC_PROG_GO])
AN_PROGRAM([gccgo], [AC_PROG_GO])
AC_DEFUN([AC_PROG_GO],
[AC_LANG_PUSH(Go)dnl
AC_ARG_VAR([GOC],   [Go compiler command])dnl
AC_ARG_VAR([GOCFLAGS], [Go compiler flags])dnl
_AC_ARG_VAR_LDFLAGS()dnl
m4_ifval([$1],
      [AC_CHECK_TOOLS(GOC, [$1])],
[AC_CHECK_TOOL(GOC, gccgo)
if test -z "$GOC"; then
  if test -n "$ac_tool_prefix"; then
    AC_CHECK_PROG(GOC, [${ac_tool_prefix}gccgo], [$ac_tool_prefix}gccgo])
  fi
fi
if test -z "$GOC"; then
  AC_CHECK_PROG(GOC, gccgo, gccgo, , , gccgo)
fi
])

# Provide some information about the compiler.
_AS_ECHO_LOG([checking for _AC_LANG compiler version])
set X $ac_compile
ac_compiler=$[2]
_AC_DO_LIMIT([$ac_compiler --version >&AS_MESSAGE_LOG_FD])
m4_expand_once([_AC_COMPILER_EXEEXT])[]dnl
m4_expand_once([_AC_COMPILER_OBJEXT])[]dnl
GOCFLAGS="-g -O2"
AC_LANG_POP(Go)dnl
])# AC_PROG_GO
