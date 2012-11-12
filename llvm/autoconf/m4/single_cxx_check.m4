dnl AC_SINGLE_CXX_CHECK(CACHEVAR, FUNCTION, HEADER, PROGRAM)
dnl                     $1,       $2,       $3,     $4,     
dnl 
AC_DEFUN([AC_SINGLE_CXX_CHECK],
 [AC_CACHE_CHECK([for $2 in $3], [$1],
  [AC_LANG_PUSH([C++])
   AC_COMPILE_IFELSE(AC_LANG_PROGRAM([#include $3],[$4]),[$1=yes],[$1=no])
  AC_LANG_POP([C++])])
 ])

