# Check if $CXX does or can be made to support C++11 by adding switches.
# If $CXX explicitly selects a language standard, then
# refrain from overriding this choice.
AC_DEFUN([AX_CXX_COMPILE_STDCXX_11_NO_OVERRIDE], [dnl
	AC_PROG_GREP
	echo $CXX | $GREP -e "-std=" > /dev/null 2> /dev/null
	if test $? -eq 0; then
		_AX_CXX_COMPILE_STDCXX_11_DEFAULT
	else
		AX_CXX_COMPILE_STDCXX_11([noext], [optional])
	fi
])

# Check if $CXX supports C++11 by default (without adding switches).
# This is a trimmed down version of AX_CXX_COMPILE_STDCXX_11
# that reuses its _AX_CXX_COMPILE_STDCXX_testbody_11.
AC_DEFUN([_AX_CXX_COMPILE_STDCXX_11_DEFAULT], [dnl
  AC_LANG_PUSH([C++])dnl
  ac_success=no
  AC_CACHE_CHECK(whether $CXX supports C++11 features by default,
  ax_cv_cxx_compile_cxx11,
  [AC_COMPILE_IFELSE([AC_LANG_SOURCE([_AX_CXX_COMPILE_STDCXX_testbody_11])],
    [ax_cv_cxx_compile_cxx11=yes],
    [ax_cv_cxx_compile_cxx11=no])])
  if test x$ax_cv_cxx_compile_cxx11 = xyes; then
    ac_success=yes
  fi
  AC_LANG_POP([C++])
  if test x$ac_success = xno; then
    HAVE_CXX11=0
  else
    HAVE_CXX11=1
    AC_DEFINE(HAVE_CXX11,1,
              [define if the compiler supports basic C++11 syntax])
  fi
  AC_SUBST(HAVE_CXX11)
])
