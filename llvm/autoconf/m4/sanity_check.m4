dnl Check a program for version sanity. The test runs a program, passes it an
dnl argument to make it print out some identification string, and filters that 
dnl output with a regular expression. If the output is non-empty, the program
dnl passes the sanity check.
dnl   $1 - Name or full path of the program to run
dnl   $2 - Argument to pass to print out identification string
dnl   $3 - grep RE to match identification string
dnl   $4 - set to 1 to make errors only a warning
AC_DEFUN([CHECK_PROGRAM_SANITY],
[
AC_MSG_CHECKING([sanity for program ]$1)
sanity="0"
sanity_path=`which $1 2>/dev/null`
if test "$?" -eq 0 -a -x "$sanity_path" ; then
  sanity=`$1 $2 2>&1 | grep "$3"`
  if test -z "$sanity" ; then
    AC_MSG_RESULT([no])
    sanity="0"
    if test "$4" -eq 1 ; then
      AC_MSG_WARN([Program ]$1[ failed to pass sanity check.])
    else
      AC_MSG_ERROR([Program ]$1[ failed to pass sanity check.])
    fi
  else
    AC_MSG_RESULT([yes])
    sanity="1"
  fi
else
  AC_MSG_RESULT([not found])
fi
])
