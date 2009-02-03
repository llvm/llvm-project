#
# Check for the ability to mmap a file.  
#
AC_DEFUN([AC_FUNC_MMAP_FILE],
[AC_CACHE_CHECK(for mmap of files,
ac_cv_func_mmap_file,
[ AC_LANG_PUSH([C])
  AC_RUN_IFELSE([
    AC_LANG_PROGRAM([[
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
]],[[
  int fd;
  fd = creat ("foo",0777); 
  fd = (int) mmap (0, 1, PROT_READ, MAP_SHARED, fd, 0);
  unlink ("foo"); 
  return (fd != (int) MAP_FAILED);]])],
  [ac_cv_func_mmap_file=yes],[ac_cv_func_mmap_file=no],[ac_cv_func_mmap_file=no])
  AC_LANG_POP([C])
])
if test "$ac_cv_func_mmap_file" = yes; then
   AC_DEFINE([HAVE_MMAP_FILE],[],[Define if mmap() can map files into memory])
   AC_SUBST(MMAP_FILE,[yes])
fi
])
