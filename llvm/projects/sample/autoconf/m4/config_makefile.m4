#
# Configure a Makefile without clobbering it if it exists and is not out of
# date.  This macro is unique to LLVM.
#
AC_DEFUN([AC_CONFIG_MAKEFILE],
[AC_CONFIG_COMMANDS($1,
  [${llvm_src}/autoconf/mkinstalldirs `dirname $1`
   ${SHELL} ${llvm_src}/autoconf/install-sh -m 0644 -c ${srcdir}/$1 $1])
])
