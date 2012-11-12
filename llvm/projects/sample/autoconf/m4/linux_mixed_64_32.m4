#
# Some Linux machines run a 64-bit kernel with a 32-bit userspace. 'uname -m'
# shows these as x86_64. Ask the system 'gcc' what it thinks.
#
AC_DEFUN([AC_IS_LINUX_MIXED],
[AC_CACHE_CHECK(for 32-bit userspace on 64-bit system,llvm_cv_linux_mixed,
[ AC_LANG_PUSH([C])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
      [[#ifndef __x86_64__
       error: Not x86-64 even if uname says so!
      #endif
      ]])],
      [llvm_cv_linux_mixed=no],
      [llvm_cv_linux_mixed=yes])
  AC_LANG_POP([C])
])
])
