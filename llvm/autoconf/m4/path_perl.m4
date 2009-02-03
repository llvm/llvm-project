dnl Check for a reasonable version of Perl.
dnl   $1 - Minimum Perl version.  Typically 5.006.
dnl 
AC_DEFUN([LLVM_PROG_PERL], [
AC_PATH_PROG(PERL, [perl], [none])
if test "$PERL" != "none"; then
  AC_MSG_CHECKING(for Perl $1 or newer)
  if $PERL -e 'use $1;' 2>&1 > /dev/null; then
    AC_MSG_RESULT(yes)
  else
    PERL=none
    AC_MSG_RESULT(not found)
  fi
fi
])

