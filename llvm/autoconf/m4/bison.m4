#
# Check for Bison. 
#
# This macro verifies that Bison is installed.  If successful, then
# 1) YACC is set to bison -y (to emulate YACC calls)
# 2) BISON is set to bison
#
AC_DEFUN([AC_PROG_BISON],
[AC_CACHE_CHECK([],[llvm_cv_has_bison],[AC_PROG_YACC()])
if test "$YACC" != "bison -y"; then
  AC_SUBST(BISON,[])
  AC_MSG_WARN([bison not found, can't rebuild grammars])
else
  AC_SUBST(BISON,[bison])
fi])
