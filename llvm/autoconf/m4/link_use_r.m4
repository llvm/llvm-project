#
# Determine if the system can handle the -R option being passed to the linker.
#
# This macro is specific to LLVM.
#
AC_DEFUN([AC_LINK_USE_R],
[AC_CACHE_CHECK([for compiler -Wl,-R<path> option],[llvm_cv_link_use_r],
[ AC_LANG_PUSH([C])
  oldcflags="$CFLAGS"
  CFLAGS="$CFLAGS -Wl,-R."
  AC_LINK_IFELSE([AC_LANG_PROGRAM([[]],[[int main() { return 0; }]])],
    [llvm_cv_link_use_r=yes],[llvm_cv_link_use_r=no])
  CFLAGS="$oldcflags"
  AC_LANG_POP([C])
])
if test "$llvm_cv_link_use_r" = yes ; then
  AC_DEFINE([HAVE_LINK_R],[1],[Define if you can use -Wl,-R. to pass -R. to the linker, in order to add the current directory to the dynamic linker search path.])
  fi
])
