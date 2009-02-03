#
# Determine if the compiler accepts -fvisibility-inlines-hidden
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_CXX_USE_VISIBILITY_INLINES_HIDDEN],
[AC_CACHE_CHECK([for compiler -fvisibility-inlines-hidden option],
                [llvm_cv_cxx_visibility_inlines_hidden],
[ AC_LANG_PUSH([C++])
  oldcxxflags="$CXXFLAGS"
  CXXFLAGS="$CXXFLAGS -fvisibility-inlines-hidden"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM()],
    [llvm_cv_cxx_visibility_inlines_hidden=yes],[llvm_cv_cxx_visibility_inlines_hidden=no])
  CXXFLAGS="$oldcxxflags"
  AC_LANG_POP([C++])
])
if test "$llvm_cv_cxx_visibility_inlines_hidden" = yes ; then
  AC_SUBST([ENABLE_VISIBILITY_INLINES_HIDDEN],[1])
else
  AC_SUBST([ENABLE_VISIBILITY_INLINES_HIDDEN],[0])
fi
])
