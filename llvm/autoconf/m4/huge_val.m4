#
# This function determins if the HUGE_VAL macro is compilable with the 
# -pedantic switch or not. XCode < 2.4.1 doesn't get it right.
#
AC_DEFUN([AC_HUGE_VAL_CHECK],[
  AC_CACHE_CHECK([for HUGE_VAL sanity], [ac_cv_huge_val_sanity],[
    AC_LANG_PUSH([C++])
    ac_save_CXXFLAGS=$CXXFLAGS
    CXXFLAGS="$CXXFLAGS -pedantic"
    AC_RUN_IFELSE(
      AC_LANG_PROGRAM(
        [#include <math.h>],
        [double x = HUGE_VAL; return x != x; ]),
      [ac_cv_huge_val_sanity=yes],[ac_cv_huge_val_sanity=no],
      [ac_cv_huge_val_sanity=yes])
    CXXFLAGS=$ac_save_CXXFLAGS
    AC_LANG_POP([C++])
    ])
  AC_SUBST(HUGE_VAL_SANITY,$ac_cv_huge_val_sanity)
])
