#
# This function determins if the isinf function isavailable on this
# platform.
#
AC_DEFUN([AC_FUNC_ISINF],[
AC_SINGLE_CXX_CHECK([ac_cv_func_isinf_in_math_h],   
                    [isinf], [<math.h>],
                    [float f; isinf(f);])
if test "$ac_cv_func_isinf_in_math_h" = "yes" ; then 
  AC_DEFINE([HAVE_ISINF_IN_MATH_H],1,[Set to 1 if the isinf function is found in <math.h>])
fi

AC_SINGLE_CXX_CHECK([ac_cv_func_isinf_in_cmath],    
                    [isinf], [<cmath>],
                    [float f; isinf(f);])
if test "$ac_cv_func_isinf_in_cmath" = "yes" ; then
  AC_DEFINE([HAVE_ISINF_IN_CMATH],1,[Set to 1 if the isinf function is found in <cmath>])
fi

AC_SINGLE_CXX_CHECK([ac_cv_func_std_isinf_in_cmath],
                    [std::isinf], [<cmath>],
                    [float f; std::isinf(f);])
if test "$ac_cv_func_std_isinf_in_cmath" = "yes" ; then 
  AC_DEFINE([HAVE_STD_ISINF_IN_CMATH],1,[Set to 1 if the std::isinf function is found in <cmath>])
fi

AC_SINGLE_CXX_CHECK([ac_cv_func_finite_in_ieeefp_h],
                    [finite], [<ieeefp.h>],
                    [float f; finite(f);])
if test "$ac_cv_func_finite_in_ieeefp_h" = "yes" ; then
  AC_DEFINE([HAVE_FINITE_IN_IEEEFP_H],1,[Set to 1 if the finite function is found in <ieeefp.h>])
fi

])


