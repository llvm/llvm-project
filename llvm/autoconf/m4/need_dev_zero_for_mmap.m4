#
# When allocating RWX memory, check whether we need to use /dev/zero
# as the file descriptor or not.
#
AC_DEFUN([AC_NEED_DEV_ZERO_FOR_MMAP],
[AC_CACHE_CHECK([if /dev/zero is needed for mmap],
ac_cv_need_dev_zero_for_mmap,
[if test "$llvm_cv_os_type" = "Interix" ; then
   ac_cv_need_dev_zero_for_mmap=yes
 else
   ac_cv_need_dev_zero_for_mmap=no
 fi
])
if test "$ac_cv_need_dev_zero_for_mmap" = yes; then
  AC_DEFINE([NEED_DEV_ZERO_FOR_MMAP],[1],
   [Define if /dev/zero should be used when mapping RWX memory, or undefine if its not necessary])
fi])
