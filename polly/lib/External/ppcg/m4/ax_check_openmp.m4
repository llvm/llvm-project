# Check if $CC supports openmp.
AC_DEFUN([AX_CHECK_OPENMP], [
	AC_SUBST(HAVE_OPENMP)
	HAVE_OPENMP=no
	AC_MSG_CHECKING([for OpenMP support by $CC])
	echo | $CC -x c - -fsyntax-only -fopenmp -Werror >/dev/null 2>/dev/null
	if test $? -eq 0; then
		HAVE_OPENMP=yes
	fi
])
