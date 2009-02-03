#
# Check for bidirectional iterator extension.  This is modified from
# http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_ext_hash_set.html
#
AC_DEFUN([AC_CXX_HAVE_BI_ITERATOR],
[AC_CACHE_CHECK(whether the compiler has the bidirectional iterator,
ac_cv_cxx_have_bi_iterator,
[AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <iterator>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[bidirectional_iterator<int,int> t; return 0;]])],[ac_cv_cxx_have_bi_iterator=yes],[ac_cv_cxx_have_bi_iterator=no])
  AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_have_bi_iterator" = yes
then
  AC_DEFINE(HAVE_BI_ITERATOR,1,[Have bi-directional iterator])
else
  AC_DEFINE(HAVE_BI_ITERATOR,0,[Does not have bi-directional iterator])
fi
])
