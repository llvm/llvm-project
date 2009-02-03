#
# Check for FLEX.  
#
# This macro verifies that flex is installed.  If successful, then
# 1) $LEX is set to "flex" (to emulate lex calls)
# 2) BISON is set to bison
AC_DEFUN([AC_PROG_FLEX],
[AC_CACHE_CHECK(,
ac_cv_has_flex,
[AC_PROG_LEX()
])
if test "$LEX" != "flex"; then
  AC_MSG_ERROR([flex not found but required])
else
  AC_SUBST(FLEX,[flex],[location of flex])
fi
])
