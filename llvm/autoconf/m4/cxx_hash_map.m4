# Check for hash_map extension.  This is from
# http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_ext_hash_map.html
AC_DEFUN([AC_CXX_HAVE_STD_EXT_HASH_MAP],
[AC_CACHE_CHECK([whether the compiler has <ext/hash_map> defining template class std::hash_map],
 ac_cv_cxx_have_std_ext_hash_map,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ext/hash_map>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[hash_map<int, int> t;]])],[ac_cv_cxx_have_std_ext_hash_map=yes],[ac_cv_cxx_have_std_ext_hash_map=no])
  AC_LANG_POP([C++])])
 if test "$ac_cv_cxx_have_std_ext_hash_map" = yes
 then
   AC_DEFINE(HAVE_STD_EXT_HASH_MAP,1,[Have ext/hash_map>])
 else
   AC_DEFINE(HAVE_STD_EXT_HASH_MAP,0,[Does not have ext/hash_map>])
 fi
 ])

AC_DEFUN([AC_CXX_HAVE_GNU_EXT_HASH_MAP],
[AC_CACHE_CHECK([whether the compiler has <ext/hash_map> defining template class __gnu_cxx::hash_map],
 ac_cv_cxx_have_gnu_ext_hash_map,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <ext/hash_map>
#ifdef HAVE_NAMESPACES
using namespace __gnu_cxx;
#endif]], [[hash_map<int,int> t; ]])],[ac_cv_cxx_have_gnu_ext_hash_map=yes],[ac_cv_cxx_have_gnu_ext_hash_map=no])
  AC_LANG_POP([C++])])
 if test "$ac_cv_cxx_have_gnu_ext_hash_map" = yes
 then
   AC_DEFINE(HAVE_GNU_EXT_HASH_MAP,1,[Have ext/hash_map])
 else
   AC_DEFINE(HAVE_GNU_EXT_HASH_MAP,0,[Does not have ext/hash_map])
 fi
 ])

AC_DEFUN([AC_CXX_HAVE_GLOBAL_HASH_MAP],
[AC_CACHE_CHECK([whether the compiler has <hash_map> defining template class ::hash_map],
 ac_cv_cxx_have_global_hash_map,
 [AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <hash_map>]], [[hash_map<int,int> t; ]])],[ac_cv_cxx_have_global_hash_map=yes],[ac_cv_cxx_have_global_hash_map=no])
  AC_LANG_POP([C++])])
 if test "$ac_cv_cxx_have_global_hash_map" = yes
 then
   AC_DEFINE(HAVE_GLOBAL_HASH_MAP,1,[Have <hash_map>])
 else
   AC_DEFINE(HAVE_GLOBAL_HASH_MAP,0,[Does not have <hash_map>])
 fi
 ])

AC_DEFUN([AC_CXX_HAVE_HASH_MAP],
[AC_CXX_HAVE_STD_EXT_HASH_MAP
 AC_CXX_HAVE_GNU_EXT_HASH_MAP
 AC_CXX_HAVE_GLOBAL_HASH_MAP])


