# Check for C++ namespace support.  This is from
# http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_namespaces.html
#
AC_DEFUN([AC_CXX_NAMESPACES],
[AC_CACHE_CHECK(whether the compiler implements namespaces,
ac_cv_cxx_namespaces,
[AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
   [[namespace Outer { namespace Inner { int i = 0; }}]],
   [[using namespace Outer::Inner; return i;]])], 
   ac_cv_cxx_namespaces=yes, 
   ac_cv_cxx_namespaces=no)
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_namespaces" = yes; then
  AC_DEFINE(HAVE_NAMESPACES,,[define if the compiler implements namespaces])
fi
])

