#ifndef CONFIG_H
#define CONFIG_H

// Include this header only under the llvm source tree.
// This is a private header.

/* Exported configuration */
#include "llvm/Config/llvm-config.h"

/* Bug report URL. */
#define BUG_REPORT_URL "https://github.com/llvm/llvm-project/issues/"

/* Define to 1 to enable backtraces, and to 0 otherwise. */
#define ENABLE_BACKTRACES 1

/* Define to 1 to enable crash overrides, and to 0 otherwise. */
#define ENABLE_CRASH_OVERRIDES 1

/* Define to 1 to enable crash memory dumps, and to 0 otherwise. */
#define LLVM_ENABLE_CRASH_DUMPS 0

/* Define to 1 to prefer forward slashes on Windows, and to 0 prefer
   backslashes. */
#define LLVM_WINDOWS_PREFER_FORWARD_SLASH 0

/* Define to 1 if you have the `backtrace' function. */
/* #undef HAVE_BACKTRACE */

#define BACKTRACE_HEADER <backtrace.h>

/* Define to 1 if you have the <CrashReporterClient.h> header file. */
/* #undef HAVE_CRASHREPORTERCLIENT_H */

/* can use __crashreporter_info__ */
#define HAVE_CRASHREPORTER_INFO 0

/* Define to 1 if you have the declaration of `arc4random', and to 0 if you
   don't. */
#define HAVE_DECL_ARC4RANDOM 0

/* Define to 1 if you have the declaration of `FE_ALL_EXCEPT', and to 0 if you
   don't. */
#define HAVE_DECL_FE_ALL_EXCEPT 1

/* Define to 1 if you have the declaration of `FE_INEXACT', and to 0 if you
   don't. */
#define HAVE_DECL_FE_INEXACT 1

/* Define to 1 if you have the declaration of `strerror_s', and to 0 if you
   don't. */
#define HAVE_DECL_STRERROR_S 1

/* Define if dlopen() is available on this platform. */
/* #undef HAVE_DLOPEN */

/* Define to 1 if we can register EH frames on this platform. */
/* #undef HAVE_REGISTER_FRAME */

/* Define to 1 if we can deregister EH frames on this platform. */
/* #undef HAVE_DEREGISTER_FRAME */

/* Define if __unw_add_dynamic_fde() is available on this platform. */
/* #undef HAVE_UNW_ADD_DYNAMIC_FDE */

/* Define if libffi is available on this platform. */
/* #undef HAVE_FFI_CALL */

/* Define to 1 if you have the <ffi/ffi.h> header file. */
/* #undef HAVE_FFI_FFI_H */

/* Define to 1 if you have the <ffi.h> header file. */
/* #undef HAVE_FFI_H */

/* Define to 1 if you have the `futimens' function. */
/* #undef HAVE_FUTIMENS */

/* Define to 1 if you have the `futimes' function. */
/* #undef HAVE_FUTIMES */

/* Define to 1 if you have the `getpagesize' function. */
/* #undef HAVE_GETPAGESIZE */

/* Define to 1 if you have the `getrusage' function. */
/* #undef HAVE_GETRUSAGE */

/* Define to 1 if you have the `isatty' function. */
/* #undef HAVE_ISATTY */

/* Define to 1 if you have the `edit' library (-ledit). */
/* #undef HAVE_LIBEDIT */

/* Define to 1 if you have the `pfm' library (-lpfm). */
/* #undef HAVE_LIBPFM */

/* Define to 1 if the `perf_branch_entry' struct has field cycles. */
/* #undef LIBPFM_HAS_FIELD_CYCLES */

/* Define to 1 if you have the `psapi' library (-lpsapi). */
/* #undef HAVE_LIBPSAPI */

/* Define to 1 if you have the `pthread' library (-lpthread). */
/* #undef HAVE_LIBPTHREAD */

/* Define to 1 if you have the `pthread_getname_np' function. */
/* #undef HAVE_PTHREAD_GETNAME_NP */

/* Define to 1 if you have the `pthread_setname_np' function. */
/* #undef HAVE_PTHREAD_SETNAME_NP */

/* Define to 1 if you have the `pthread_get_name_np' function. */
/* #undef HAVE_PTHREAD_GET_NAME_NP */

/* Define to 1 if you have the `pthread_set_name_np' function. */
/* #undef HAVE_PTHREAD_SET_NAME_NP */

/* Define to 1 if you have the <mach/mach.h> header file. */
/* #undef HAVE_MACH_MACH_H */

/* Define to 1 if you have the `mallctl' function. */
/* #undef HAVE_MALLCTL */

/* Define to 1 if you have the `mallinfo' function. */
/* #undef HAVE_MALLINFO */

/* Define to 1 if you have the `mallinfo2' function. */
/* #undef HAVE_MALLINFO2 */

/* Define to 1 if you have the <malloc/malloc.h> header file. */
/* #undef HAVE_MALLOC_MALLOC_H */

/* Define to 1 if you have the `malloc_zone_statistics' function. */
/* #undef HAVE_MALLOC_ZONE_STATISTICS */

/* Define to 1 if you have the `posix_spawn' function. */
/* #undef HAVE_POSIX_SPAWN */

/* Define to 1 if you have the `pread' function. */
/* #undef HAVE_PREAD */

/* Define to 1 if you have the <pthread.h> header file. */
/* #undef HAVE_PTHREAD_H */

/* Have pthread_mutex_lock */
/* #undef HAVE_PTHREAD_MUTEX_LOCK */

/* Have pthread_rwlock_init */
/* #undef HAVE_PTHREAD_RWLOCK_INIT */

/* Define to 1 if you have the `sbrk' function. */
/* #undef HAVE_SBRK */

/* Define to 1 if you have the `setenv' function. */
/* #undef HAVE_SETENV */

/* Define to 1 if you have the `sigaltstack' function. */
/* #undef HAVE_SIGALTSTACK */

/* Define to 1 if you have the `strerror_r' function. */
/* #undef HAVE_STRERROR_R */

/* Define to 1 if you have the `sysconf' function. */
/* #undef HAVE_SYSCONF */

/* Define to 1 if you have the <sys/mman.h> header file. */
/* #undef HAVE_SYS_MMAN_H */

/* Define to 1 if you have the <sys/ioctl.h> header file. */
/* #undef HAVE_SYS_IOCTL_H */

/* Define to 1 if stat struct has st_mtimespec member .*/
/* #undef HAVE_STRUCT_STAT_ST_MTIMESPEC_TV_NSEC */

/* Define to 1 if stat struct has st_mtim member. */
/* #undef HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC */

/* Define to 1 if you have the <unistd.h> header file. */
/* #undef HAVE_UNISTD_H */

/* Define to 1 if you have the <valgrind/valgrind.h> header file. */
/* #undef HAVE_VALGRIND_VALGRIND_H */

/* Have host's _alloca */
/* #undef HAVE__ALLOCA */

/* Define to 1 if you have the `_chsize_s' function. */
#define HAVE__CHSIZE_S 1

/* Define to 1 if you have the `_Unwind_Backtrace' function. */
/* #undef HAVE__UNWIND_BACKTRACE */

/* Have host's __alloca */
/* #undef HAVE___ALLOCA */

/* Have host's __ashldi3 */
/* #undef HAVE___ASHLDI3 */

/* Have host's __ashrdi3 */
/* #undef HAVE___ASHRDI3 */

/* Have host's __chkstk */
#define HAVE___CHKSTK 1

/* Have host's __chkstk_ms */
/* #undef HAVE___CHKSTK_MS */

/* Have host's __cmpdi2 */
/* #undef HAVE___CMPDI2 */

/* Have host's __divdi3 */
/* #undef HAVE___DIVDI3 */

/* Have host's __fixdfdi */
/* #undef HAVE___FIXDFDI */

/* Have host's __fixsfdi */
/* #undef HAVE___FIXSFDI */

/* Have host's __floatdidf */
/* #undef HAVE___FLOATDIDF */

/* Have host's __lshrdi3 */
/* #undef HAVE___LSHRDI3 */

/* Have host's __main */
/* #undef HAVE___MAIN */

/* Have host's __moddi3 */
/* #undef HAVE___MODDI3 */

/* Have host's __udivdi3 */
/* #undef HAVE___UDIVDI3 */

/* Have host's __umoddi3 */
/* #undef HAVE___UMODDI3 */

/* Have host's ___chkstk */
/* #undef HAVE____CHKSTK */

/* Have host's ___chkstk_ms */
/* #undef HAVE____CHKSTK_MS */

/* Define if ICU library is available */
#define HAVE_ICU 0

/* Define if iconv library is available */
#define HAVE_ICONV 0

/* Linker version detected at compile time. */
/* #undef HOST_LINK_VERSION */

/* Define if overriding target triple is enabled */
/* #undef LLVM_TARGET_TRIPLE_ENV */

/* Whether tools show host and target info when invoked with --version */
#define LLVM_VERSION_PRINTER_SHOW_HOST_TARGET_INFO 1

/* Whether tools show optional build config flags when invoked with --version */
#define LLVM_VERSION_PRINTER_SHOW_BUILD_CONFIG 1

/* Define if libxml2 is supported on this platform. */
/* #undef LLVM_ENABLE_LIBXML2 */

/* Define to the extension used for shared libraries, say, ".so". */
#define LTDL_SHLIB_EXT ".dll"

/* Define to the extension used for plugin libraries, say, ".so". */
#define LLVM_PLUGIN_EXT ".dll"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://github.com/llvm/llvm-project/issues/"

/* Define to the full name of this package. */
#define PACKAGE_NAME "LLVM"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "LLVM 22.0.0git"

/* Define to the version of this package. */
#define PACKAGE_VERSION "22.0.0git"

/* Define to the vendor of this package. */
/* #undef PACKAGE_VENDOR */

/* Define to a function implementing stricmp */
#define stricmp _stricmp

/* Define to a function implementing strdup */
#define strdup _strdup

/* Whether GlobalISel rule coverage is being collected */
#define LLVM_GISEL_COV_ENABLED 0

/* Define to the default GlobalISel coverage file prefix */
/* #undef LLVM_GISEL_COV_PREFIX */

/* Whether Timers signpost passes in Xcode Instruments */
#define LLVM_SUPPORT_XCODE_SIGNPOSTS 0

/* #undef HAVE_PROC_PID_RUSAGE */

/* #undef HAVE_BUILTIN_THREAD_POINTER */

/* #undef HAVE_GETAUXVAL */

#endif
