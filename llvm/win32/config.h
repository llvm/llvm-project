/* This file is appended to config.h.in to form the Windows version of
 * config.h */

#define PACKAGE_NAME "LLVM (win32 vc8.0)" 
#define PACKAGE_VERSION 2.4
#define PACKAGE_STRING "llvm 2.6svn"
#define LLVM_HOSTTRIPLE "i686-pc-win32"
#define HAVE_WINDOWS_H 1 
#define HAVE_LIMITS_H 1 
#define HAVE_SYS_STAT_H 1 
#define HAVE_STDLIB_H 1 
#define HAVE_STDIO_H 1 
#define HAVE_STRING_H 1 
#define HAVE_CEILF 1 
#define HAVE_FLOORF 1 
#define SHLIBEXT ".lib" 
#define error_t int 
#define HAVE_ERRNO_H 1 
#define LTDL_DLOPEN_DEPLIBS 1 
#define LTDL_OBJDIR "_libs" 
#define LTDL_SHLIBPATH_VAR "PATH" 
#define LTDL_SHLIB_EXT ".dll" 
#define LTDL_SYSSEARCHPATH "" 
#define LLVM_ON_WIN32 1 

#define strtoll _strtoi64
#define strtoull _strtoui64
#define stricmp _stricmp
#define strdup _strdup

