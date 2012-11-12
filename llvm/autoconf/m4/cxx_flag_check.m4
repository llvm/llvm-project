AC_DEFUN([CXX_FLAG_CHECK],
  [AC_SUBST($1, `$CXX -Werror $2 -fsyntax-only -xc /dev/null 2>/dev/null && echo $2`)])
