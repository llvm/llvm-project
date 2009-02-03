# Check for hash_set extension.  This is modified from
# http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_ext_hash_set.html
AC_DEFUN([AC_CXX_HAVE_STD_EXT_HASH_SET],
[AC_CACHE_CHECK([whether the compiler has <ext/hash_set> defining template class std::hash_set],
 ac_cv_cxx_have_std_ext_hash_set,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ext/hash_set>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[hash_set<int> t; ]])],[ac_cv_cxx_have_std_ext_hash_set=yes],[ac_cv_cxx_have_std_ext_hash_set=no])
  AC_LANG_POP([C++])])
 if test "$ac_cv_cxx_have_std_ext_hash_set" = yes
 then
   AC_DEFINE(HAVE_STD_EXT_HASH_SET,1,[Have hash_set in std namespace])
 else
   AC_DEFINE(HAVE_STD_EXT_HASH_SET,0,[Does not have hash_set in std namespace])
 fi
 ])

AC_DEFUN([AC_CXX_HAVE_GNU_EXT_HASH_SET],
[AC_CACHE_CHECK(
 [whether the compiler has <ext/hash_set> defining template class __gnu_cxx::hash_set],
 ac_cv_cxx_have_gnu_ext_hash_set,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ext/hash_set>
#ifdef HAVE_NAMESPACES
using namespace __gnu_cxx;
#endif]], [[hash_set<int> t; ]])],[ac_cv_cxx_have_gnu_ext_hash_set=yes],[ac_cv_cxx_have_gnu_ext_hash_set=no])
  AC_LANG_POP([C++])])
 if test "$ac_cv_cxx_have_gnu_ext_hash_set" = yes
 then
   AC_DEFINE(HAVE_GNU_EXT_HASH_SET,1,[Have hash_set in gnu namespace])
 else
   AC_DEFINE(HAVE_GNU_EXT_HASH_SET,0,[Does not have hash_set in gnu namespace])
 fi
 ])

AC_DEFUN([AC_CXX_HAVE_GLOBAL_HASH_SET],
[AC_CACHE_CHECK([whether the compiler has <hash_set> defining template class ::hash_set],
 ac_cv_cxx_have_global_hash_set,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <hash_set>]], [[hash_set<int> t; return 0;]])],[ac_cv_cxx_have_global_hash_set=yes],[ac_cv_cxx_have_global_hash_set=no])
  AC_LANG_POP([C++])])
 if test "$ac_cv_cxx_have_global_hash_set" = yes
 then
   AC_DEFINE(HAVE_GLOBAL_HASH_SET,1,[Have hash_set in global namespace])
 else
   AC_DEFINE(HAVE_GLOBAL_HASH_SET,0,[Does not have hash_set in global namespace])
 fi
 ])

AC_DEFUN([AC_CXX_HAVE_HASH_SET],
[AC_CXX_HAVE_STD_EXT_HASH_SET
 AC_CXX_HAVE_GNU_EXT_HASH_SET
 AC_CXX_HAVE_GLOBAL_HASH_SET])


