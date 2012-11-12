#
# This function determins if the srand48,drand48,lrand48 functions are
# available on this platform.
#
AC_DEFUN([AC_FUNC_RAND48],[
AC_SINGLE_CXX_CHECK([ac_cv_func_rand48],   
                    [srand48/lrand48/drand48], [<stdlib.h>],
                    [srand48(0);lrand48();drand48();])
if test "$ac_cv_func_rand48" = "yes" ; then
AC_DEFINE([HAVE_RAND48],1,[Define to 1 if srand48/lrand48/drand48 exist in <stdlib.h>])
fi
])
