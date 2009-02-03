#
# This function determines if the isnan function is available on this
# platform.
#
AC_DEFUN([AC_FUNC_ISNAN],[
AC_SINGLE_CXX_CHECK([ac_cv_func_isnan_in_math_h],   
                    [isnan], [<math.h>],
                    [float f; isnan(f);])

if test "$ac_cv_func_isnan_in_math_h" = "yes" ; then
  AC_DEFINE([HAVE_ISNAN_IN_MATH_H],1,[Set to 1 if the isnan function is found in <math.h>])
fi

AC_SINGLE_CXX_CHECK([ac_cv_func_isnan_in_cmath],    
                    [isnan], [<cmath>],
                    [float f; isnan(f);])
if test "$ac_cv_func_isnan_in_cmath" = "yes" ; then
  AC_DEFINE([HAVE_ISNAN_IN_CMATH],1,[Set to 1 if the isnan function is found in <cmath>])
fi

AC_SINGLE_CXX_CHECK([ac_cv_func_std_isnan_in_cmath],
                    [std::isnan], [<cmath>],
                    [float f; std::isnan(f);])
if test "$ac_cv_func_std_isnan_in_cmath" = "yes" ; then
  AC_DEFINE([HAVE_STD_ISNAN_IN_CMATH],1,[Set to 1 if the std::isnan function is found in <cmath>])
fi
])
