/*===------- llvm/Config/config.h - llvm configuration ------------*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This is a manual port of config.h.cmake for the symbols that do not change
   based on platform. Those that do change should not be defined here and
   instead use Bazel cc_library defines. Some attempt has been made to extract
   such symbols that do vary based on platform (for the platforms we care about)
   into Bazel defines, but it is by no means complete, so if you see something
   that looks wrong, it probably is. */

#ifndef CONFIG_H
#define CONFIG_H

// Include this header only under the llvm source tree.
// This is a private header.

/* Exported configuration */
#include "llvm/Config/llvm-config.h"

/* Bug report URL. */
#define BUG_REPORT_URL "https://bugs.llvm.org/"

/* Define to 1 to enable backtraces, and to 0 otherwise. */
#define ENABLE_BACKTRACES 1

/* Define to 1 to enable crash overrides, and to 0 otherwise. */
#define ENABLE_CRASH_OVERRIDES 1

/* Define to 1 to enable crash memory dumps, and to 0 otherwise. */
#define LLVM_ENABLE_CRASH_DUMPS 0

/* Define to 1 if you have the `backtrace' function. */
/* HAVE_BACKTRACE defined in Bazel */

/* BACKTRACE_HEADER defined in Bazel */

/* Define to 1 if you have the <CrashReporterClient.h> header file. */
/* HAVE_CRASHREPORTERCLIENT_H defined in Bazel */

/* can use __crashreporter_info__ */
/* HAVE_CRASHREPORTER_INFO  defined in Bazel */

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
#define HAVE_DECL_STRERROR_S 0

/* Define to 1 if you have the DIA SDK installed, and to 0 if you don't. */
#define LLVM_ENABLE_DIA_SDK 0

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define if dlopen() is available on this platform. */
#define HAVE_DLOPEN 1

/* Define if dladdr() is available on this platform. */
/* #undef HAVE_DLADDR */

/* Define to 1 if we can register EH frames on this platform. */
#define HAVE_REGISTER_FRAME 1

/* Define to 1 if we can deregister EH frames on this platform. */
#define HAVE_DEREGISTER_FRAME 1

/* Define to 1 if you have the <errno.h> header file. */
#define HAVE_ERRNO_H 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the <fenv.h> header file. */
#define HAVE_FENV_H 1

/* Define if libffi is available on this platform. */
/* #undef HAVE_FFI_CALL */

/* Define to 1 if you have the <ffi/ffi.h> header file. */
/* #undef HAVE_FFI_FFI_H */

/* Define to 1 if you have the <ffi.h> header file. */
/* #undef HAVE_FFI_H */

/* Define to 1 if you have the `futimens' function. */
#define HAVE_FUTIMENS 1

/* Define to 1 if you have the `futimes' function. */
#define HAVE_FUTIMES 1

/* Define to 1 if you have the `getpagesize' function. */
#define HAVE_GETPAGESIZE 1

/* Define to 1 if you have the `getrlimit' function. */
#define HAVE_GETRLIMIT 1

/* Define to 1 if you have the `getrusage' function. */
#define HAVE_GETRUSAGE 1

/* Define to 1 if you have the `isatty' function. */
#define HAVE_ISATTY 1

/* Define to 1 if you have the `edit' library (-ledit). */
/* #undef HAVE_LIBEDIT */

/* Define to 1 if you have the `pfm' library (-lpfm). */
/* #undef HAVE_LIBPFM */

/* Define to 1 if the `perf_branch_entry' struct has field cycles. */
/* #undef LIBPFM_HAS_FIELD_CYCLES */

/* Define to 1 if you have the `psapi' library (-lpsapi). */
/* #undef HAVE_LIBPSAPI */

/* Define to 1 if you have the `pthread' library (-lpthread). */
#define HAVE_LIBPTHREAD 1

/* Define to 1 if you have the `pthread_getname_np' function. */
#define HAVE_PTHREAD_GETNAME_NP 1

/* Define to 1 if you have the `pthread_setname_np' function. */
#define HAVE_PTHREAD_SETNAME_NP 1

/* Define to 1 if you have the <link.h> header file. */
/* HAVE_LINK_H defined in Bazel */

/* Define to 1 if you have the `lseek64' function. */
/* HAVE_LSEEK64 defined in Bazel */

/* Define to 1 if you have the <mach/mach.h> header file. */
/* HAVE_MACH_MACH_H defined in Bazel */

/* Define to 1 if you have the `mallctl' function. */
/* #undef HAVE_MALLCTL */

/* Define to 1 if you have the `mallinfo' function. */
/* HAVE_MALLINFO defined in Bazel */

/* Define to 1 if you have the <malloc/malloc.h> header file. */
/* HAVE_MALLOC_MALLOC_H defined in Bazel */

/* Define to 1 if you have the `malloc_zone_statistics' function. */
/* HAVE_MALLOC_ZONE_STATISTICS defined in Bazel */

/* Define to 1 if you have the `posix_fallocate' function. */
/* HAVE_POSIX_FALLOCATE defined in Bazel */

/* Define to 1 if you have the `posix_spawn' function. */
#define HAVE_POSIX_SPAWN 1

/* Define to 1 if you have the `pread' function. */
#define HAVE_PREAD 1

/* Have pthread_getspecific */
#define HAVE_PTHREAD_GETSPECIFIC 1

/* Define to 1 if you have the <pthread.h> header file. */
#define HAVE_PTHREAD_H 1

/* Have pthread_mutex_lock */
#define HAVE_PTHREAD_MUTEX_LOCK 1

/* Have pthread_rwlock_init */
#define HAVE_PTHREAD_RWLOCK_INIT 1

/* Define to 1 if you have the `sbrk' function. */
/* HAVE_SBRK defined in Bazel */

/* Define to 1 if you have the `setenv' function. */
#define HAVE_SETENV 1

/* Define to 1 if you have the `setrlimit' function. */
#define HAVE_SETRLIMIT 1

/* Define to 1 if you have the `sigaltstack' function. */
#define HAVE_SIGALTSTACK 1

/* Define to 1 if you have the <signal.h> header file. */
#define HAVE_SIGNAL_H 1

/* Define to 1 if you have the `strerror' function. */
#define HAVE_STRERROR 1

/* Define to 1 if you have the `strerror_r' function. */
/* HAVE_STRERROR_R defined in Bazel */

/* Define to 1 if you have the `sysconf' function. */
#define HAVE_SYSCONF 1

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#define HAVE_SYS_IOCTL_H 1

/* Define to 1 if you have the <sys/mman.h> header file. */
#define HAVE_SYS_MMAN_H 1

/* Define to 1 if you have the <sys/param.h> header file. */
#define HAVE_SYS_PARAM_H 1

/* Define to 1 if you have the <sys/resource.h> header file. */
#define HAVE_SYS_RESOURCE_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if stat struct has st_mtimespec member .*/
/* #undef HAVE_STRUCT_STAT_ST_MTIMESPEC_TV_NSEC */

/* Define to 1 if stat struct has st_mtim member. */
/* HAVE_STRUCT_STAT_ST_MTIM_TV_NSEC defined in Bazel */

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define if the setupterm() function is supported this platform. */
/* LLVM_ENABLE_TERMINFO defined in Bazel */

/* Define to 1 if you have the <termios.h> header file. */
#define HAVE_TERMIOS_H 1

/* Define to 1 if you have the <unistd.h> header file. */
/* HAVE_UNISTD_H defined in Bazel */

/* Define to 1 if you have the <valgrind/valgrind.h> header file. */
/* #undef HAVE_VALGRIND_VALGRIND_H */

/* Have host's _alloca */
/* #undef HAVE__ALLOCA */

/* Define to 1 if you have the `_chsize_s' function. */
/* #undef HAVE__CHSIZE_S */

/* Define to 1 if you have the `_Unwind_Backtrace' function. */
#define HAVE__UNWIND_BACKTRACE 1

/* Have host's __alloca */
/* #undef HAVE___ALLOCA */

/* Have host's __ashldi3 */
/* #undef HAVE___ASHLDI3 */

/* Have host's __ashrdi3 */
/* #undef HAVE___ASHRDI3 */

/* Have host's __chkstk */
/* #undef HAVE___CHKSTK */

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

/* Linker version detected at compile time. */
/* #undef HOST_LINK_VERSION */

/* Target triple LLVM will generate code for by default */
/* Doesn't use `cmakedefine` because it is allowed to be empty. */
/* LLVM_DEFAULT_TARGET_TRIPLE defined in Bazel */

/* Define if zlib compression is available */
/* LLVM_ENABLE_ZLIB defined in Bazel */

/* Define if overriding target triple is enabled */
/* #undef LLVM_TARGET_TRIPLE_ENV */

/* LLVM version information */
/* #undef LLVM_VERSION_INFO */

/* Whether tools show host and target info when invoked with --version */
#define LLVM_VERSION_PRINTER_SHOW_HOST_TARGET_INFO 1

/* Define if libxml2 is supported on this platform. */
/* #undef LLVM_ENABLE_LIBXML2 */

/* Define to the extension used for shared libraries, say, ".so". */
/* LTDL_SHLIB_EXT defined in Bazel */

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "https://bugs.llvm.org/"

/* Define to the full name of this package. */
#define PACKAGE_NAME "LLVM"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING PACKAGE_NAME " " LLVM_VERSION_STRING

/* Define to the version of this package. */
#define PACKAGE_VERSION LLVM_VERSION_STRING

/* Define to the vendor of this package. */
/* #undef PACKAGE_VENDOR */

/* Define as the return type of signal handlers (`int' or `void'). */
#define RETSIGTYPE void

/* Define if std::is_trivially_copyable is supported */
#define HAVE_STD_IS_TRIVIALLY_COPYABLE 1

/* Define to a function implementing stricmp */
/* stricmp defined in Bazel */

/* Define to a function implementing strdup */
/* strdup defined in Bazel */

/* Whether GlobalISel rule coverage is being collected */
#define LLVM_GISEL_COV_ENABLED 0

/* Define to the default GlobalISel coverage file prefix */
/* #undef LLVM_GISEL_COV_PREFIX */

#endif
