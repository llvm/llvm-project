# Check if OpenCL is available and that it supports a CPU device.
# The check for a CPU device is the same check that is performed
# by opencl_create_device in ocl_utilities.c
AC_DEFUN([AX_CHECK_OPENCL], [
	AC_SUBST(HAVE_OPENCL)
	HAVE_OPENCL=no
	AC_CHECK_HEADER([CL/opencl.h], [
		AC_CHECK_LIB([OpenCL], [clGetPlatformIDs], [
			SAVE_LIBS=$LIBS
			LIBS="$LIBS -lOpenCL"
			AC_MSG_CHECKING([for OpenCL CPU device])
			AC_RUN_IFELSE([AC_LANG_PROGRAM(
				[[#include <CL/opencl.h>]], [[
	cl_platform_id platform;
	cl_device_id dev;

	if (clGetPlatformIDs(1, &platform, NULL) < 0)
		return 1;
	if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL) < 0)
		return 1;
				]])], [HAVE_OPENCL=yes])
			AC_MSG_RESULT($HAVE_OPENCL)
			LIBS=$SAVE_LIBS
			])])
])
