#
# Check for anonymous mmap macros.  This is modified from
# http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_ext_slist.html
#
AC_DEFUN([AC_HEADER_MMAP_ANONYMOUS],
[AC_CACHE_CHECK(for MAP_ANONYMOUS vs. MAP_ANON,
ac_cv_header_mmap_anon,
[ AC_LANG_PUSH([C])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
    [[#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>]],
  [[mmap (0, 1, PROT_READ, MAP_ANONYMOUS, -1, 0); return (0);]])],
  ac_cv_header_mmap_anon=yes, 
  ac_cv_header_mmap_anon=no)
  AC_LANG_POP([C])
])
if test "$ac_cv_header_mmap_anon" = yes; then
   AC_DEFINE([HAVE_MMAP_ANONYMOUS],[1],[Define if mmap() uses MAP_ANONYMOUS to map anonymous pages, or undefine if it uses MAP_ANON])
fi
])
