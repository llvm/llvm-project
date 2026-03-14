/* define if your compiler has __attribute__ */
#cmakedefine HAVE___ATTRIBUTE__ /**/

/* most gcc compilers know a function __attribute__((__warn_unused_result__)) */
#define GCC_WARN_UNUSED_RESULT @GCC_WARN_UNUSED_RESULT@


/* Define to 1 if you have the declaration of `ffs', and to 0 if you don't. */
#define HAVE_DECL_FFS @HAVE_DECL_FFS@

/* Define to 1 if you have the declaration of `__builtin_ffs', and to 0 if you
   don't. */
#define HAVE_DECL___BUILTIN_FFS @HAVE_DECL___BUILTIN_FFS@

/* Define to 1 if you have the declaration of `_BitScanForward', and to 0 if
   you don't. */
#define HAVE_DECL__BITSCANFORWARD @HAVE_DECL__BITSCANFORWARD@


/* Define to 1 if you have the declaration of `strcasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRCASECMP @HAVE_DECL_STRCASECMP@

/* Define to 1 if you have the declaration of `_stricmp', and to 0 if you
   don't. */
#define HAVE_DECL__STRICMP @HAVE_DECL__STRICMP@


/* Define to 1 if you have the declaration of `strncasecmp', and to 0 if you
   don't. */
#define HAVE_DECL_STRNCASECMP @HAVE_DECL_STRNCASECMP@

/* Define to 1 if you have the declaration of `_strnicmp', and to 0 if you
   don't. */
#define HAVE_DECL__STRNICMP @HAVE_DECL__STRNICMP@


/* Define to 1 if you have the declaration of `snprintf', and to 0 if you
   don't. */
#define HAVE_DECL_SNPRINTF @HAVE_DECL_SNPRINTF@

/* Define to 1 if you have the declaration of `_snprintf', and to 0 if you
   don't. */
#define HAVE_DECL__SNPRINTF @HAVE_DECL__SNPRINTF@


/* use gmp to implement isl_int */
#cmakedefine USE_GMP_FOR_MP

/* use imath to implement isl_int */
#cmakedefine USE_IMATH_FOR_MP

/* Use small integer optimization */
#cmakedefine USE_SMALL_INT_OPT

#include <isl_config_post.h>
