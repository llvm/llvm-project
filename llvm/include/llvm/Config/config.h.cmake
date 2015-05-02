/* include/llvm/Config/config.h.cmake corresponding to config.h.in. */

#ifndef CONFIG_H
#define CONFIG_H

/* Exported configuration */
#include "llvm/Config/llvm-config.h"

/* Bug report URL. */
#define BUG_REPORT_URL "${BUG_REPORT_URL}"

/* Define if you want backtraces on crash */
#cmakedefine ENABLE_BACKTRACES

/* Define to enable crash overrides */
#cmakedefine ENABLE_CRASH_OVERRIDES

/* Define to disable C++ atexit */
#cmakedefine DISABLE_LLVM_DYLIB_ATEXIT

/* Define if position independent code is enabled */
#cmakedefine ENABLE_PIC

/* Define if timestamp information (e.g., __DATE__) is allowed */
#cmakedefine ENABLE_TIMESTAMPS ${ENABLE_TIMESTAMPS}

/* Define to 1 if you have the `arc4random' function. */
#cmakedefine HAVE_DECL_ARC4RANDOM ${HAVE_DECL_ARC4RANDOM}

/* Define to 1 if you have the `backtrace' function. */
#cmakedefine HAVE_BACKTRACE ${HAVE_BACKTRACE}

/* Define to 1 if you have the `bcopy' function. */
#undef HAVE_BCOPY

/* Define to 1 if you have the `ceilf' function. */
#cmakedefine HAVE_CEILF ${HAVE_CEILF}

/* Define to 1 if you have the `closedir' function. */
#cmakedefine HAVE_CLOSEDIR ${HAVE_CLOSEDIR}

/* Define to 1 if you have the <cxxabi.h> header file. */
#cmakedefine HAVE_CXXABI_H ${HAVE_CXXABI_H}

/* Define to 1 if you have the <CrashReporterClient.h> header file. */
#undef HAVE_CRASHREPORTERCLIENT_H

/* can use __crashreporter_info__ */
#undef HAVE_CRASHREPORTER_INFO

/* Define to 1 if you have the declaration of `strerror_s', and to 0 if you
   don't. */
#cmakedefine01 HAVE_DECL_STRERROR_S

/* Define to 1 if you have the <dirent.h> header file, and it defines `DIR'.
   */
#cmakedefine HAVE_DIRENT_H ${HAVE_DIRENT_H}

/* Define if you have the GNU dld library. */
#undef HAVE_DLD

/* Define to 1 if you have the `dlerror' function. */
#cmakedefine HAVE_DLERROR ${HAVE_DLERROR}

/* Define to 1 if you have the <dlfcn.h> header file. */
#cmakedefine HAVE_DLFCN_H ${HAVE_DLFCN_H}

/* Define if dlopen() is available on this platform. */
#cmakedefine HAVE_DLOPEN ${HAVE_DLOPEN}

/* Define if you have the _dyld_func_lookup function. */
#undef HAVE_DYLD

/* Define to 1 if you have the <errno.h> header file. */
#cmakedefine HAVE_ERRNO_H ${HAVE_ERRNO_H}

/* Define to 1 if you have the <execinfo.h> header file. */
#cmakedefine HAVE_EXECINFO_H ${HAVE_EXECINFO_H}

/* Define to 1 if you have the <fcntl.h> header file. */
#cmakedefine HAVE_FCNTL_H ${HAVE_FCNTL_H}

/* Define to 1 if you have the <fenv.h> header file. */
#cmakedefine HAVE_FENV_H ${HAVE_FENV_H}

/* Define if libffi is available on this platform. */
#cmakedefine HAVE_FFI_CALL ${HAVE_FFI_CALL}

/* Define to 1 if you have the <ffi/ffi.h> header file. */
#cmakedefine HAVE_FFI_FFI_H ${HAVE_FFI_FFI_H}

/* Define to 1 if you have the <ffi.h> header file. */
#cmakedefine HAVE_FFI_H ${HAVE_FFI_H}

/* Set to 1 if the finite function is found in <ieeefp.h> */
#cmakedefine HAVE_FINITE_IN_IEEEFP_H ${HAVE_FINITE_IN_IEEEFP_H}

/* Define to 1 if you have the `floorf' function. */
#cmakedefine HAVE_FLOORF ${HAVE_FLOORF}

/* Define to 1 if you have the `log' function. */
#cmakedefine HAVE_LOG ${HAVE_LOG}

/* Define to 1 if you have the `log2' function. */
#cmakedefine HAVE_LOG2 ${HAVE_LOG2}

/* Define to 1 if you have the `log10' function. */
#cmakedefine HAVE_LOG10 ${HAVE_LOG10}

/* Define to 1 if you have the `exp' function. */
#cmakedefine HAVE_EXP ${HAVE_EXP}

/* Define to 1 if you have the `exp2' function. */
#cmakedefine HAVE_EXP2 ${HAVE_EXP2}

/* Define to 1 if you have the `exp10' function. */
#cmakedefine HAVE_EXP10 ${HAVE_EXP10}

/* Define to 1 if you have the `fmodf' function. */
#cmakedefine HAVE_FMODF ${HAVE_FMODF}

/* Define to 1 if you have the `futimes' function. */
#cmakedefine HAVE_FUTIMES ${HAVE_FUTIMES}

/* Define to 1 if you have the `futimens' function */
#cmakedefine HAVE_FUTIMENS ${HAVE_FUTIMENS}

/* Define to 1 if you have the `getcwd' function. */
#cmakedefine HAVE_GETCWD ${HAVE_GETCWD}

/* Define to 1 if you have the `getpagesize' function. */
#cmakedefine HAVE_GETPAGESIZE ${HAVE_GETPAGESIZE}

/* Define to 1 if you have the `getrlimit' function. */
#cmakedefine HAVE_GETRLIMIT ${HAVE_GETRLIMIT}

/* Define to 1 if you have the `getrusage' function. */
#cmakedefine HAVE_GETRUSAGE ${HAVE_GETRUSAGE}

/* Define to 1 if you have the `gettimeofday' function. */
#cmakedefine HAVE_GETTIMEOFDAY ${HAVE_GETTIMEOFDAY}

/* Define to 1 if the system has the type `int64_t'. */
#cmakedefine HAVE_INT64_T ${HAVE_INT64_T}

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine HAVE_INTTYPES_H ${HAVE_INTTYPES_H}

/* Define to 1 if you have the `isatty' function. */
#cmakedefine HAVE_ISATTY 1

/* Set to 1 if the isinf function is found in <cmath> */
#cmakedefine HAVE_ISINF_IN_CMATH ${HAVE_ISINF_IN_CMATH}

/* Set to 1 if the isinf function is found in <math.h> */
#cmakedefine HAVE_ISINF_IN_MATH_H ${HAVE_ISINF_IN_MATH_H}

/* Set to 1 if the isnan function is found in <cmath> */
#cmakedefine HAVE_ISNAN_IN_CMATH ${HAVE_ISNAN_IN_CMATH}

/* Set to 1 if the isnan function is found in <math.h> */
#cmakedefine HAVE_ISNAN_IN_MATH_H ${HAVE_ISNAN_IN_MATH_H}

/* Define if you have the libdl library or equivalent. */
#cmakedefine HAVE_LIBDL ${HAVE_LIBDL}

/* Define to 1 if you have the `imagehlp' library (-limagehlp). */
#cmakedefine HAVE_LIBIMAGEHLP ${HAVE_LIBIMAGEHLP}

/* Define to 1 if you have the `m' library (-lm). */
#undef HAVE_LIBM

/* Define to 1 if you have the `psapi' library (-lpsapi). */
#cmakedefine HAVE_LIBPSAPI ${HAVE_LIBPSAPI}

/* Define to 1 if you have the `pthread' library (-lpthread). */
#cmakedefine HAVE_LIBPTHREAD ${HAVE_LIBPTHREAD}

/* Define to 1 if you have the `shell32' library (-lshell32). */
#cmakedefine HAVE_LIBSHELL32 ${HAVE_LIBSHELL32}

/* Define to 1 if you have the 'z' library (-lz). */
#cmakedefine HAVE_LIBZ ${HAVE_LIBZ}

/* Define to 1 if you have the 'edit' library (-ledit). */
#cmakedefine HAVE_LIBEDIT ${HAVE_LIBEDIT}

/* Define to 1 if you have the <limits.h> header file. */
#cmakedefine HAVE_LIMITS_H ${HAVE_LIMITS_H}

/* Define to 1 if you have the <link.h> header file. */
#cmakedefine HAVE_LINK_H ${HAVE_LINK_H}

/* Define if you can use -rdynamic. */
#define HAVE_LINK_EXPORT_DYNAMIC 1

/* Define if you can use -Wl,-R. to pass -R. to the linker, in order to add
   the current directory to the dynamic linker search path. */
#undef HAVE_LINK_R

/* Define to 1 if you have the `longjmp' function. */
#cmakedefine HAVE_LONGJMP ${HAVE_LONGJMP}

/* Define to 1 if you have the <mach/mach.h> header file. */
#cmakedefine HAVE_MACH_MACH_H ${HAVE_MACH_MACH_H}

/* Define to 1 if you have the <mach-o/dyld.h> header file. */
#cmakedefine HAVE_MACH_O_DYLD_H ${HAVE_MACH_O_DYLD_H}

/* Define if mallinfo() is available on this platform. */
#cmakedefine HAVE_MALLINFO ${HAVE_MALLINFO}

/* Define to 1 if you have the <malloc.h> header file. */
#cmakedefine HAVE_MALLOC_H ${HAVE_MALLOC_H}

/* Define to 1 if you have the <malloc/malloc.h> header file. */
#cmakedefine HAVE_MALLOC_MALLOC_H ${HAVE_MALLOC_MALLOC_H}

/* Define to 1 if you have the `malloc_zone_statistics' function. */
#cmakedefine HAVE_MALLOC_ZONE_STATISTICS ${HAVE_MALLOC_ZONE_STATISTICS}

/* Define to 1 if you have the `mkdtemp' function. */
#cmakedefine HAVE_MKDTEMP ${HAVE_MKDTEMP}

/* Define to 1 if you have the `mkstemp' function. */
#cmakedefine HAVE_MKSTEMP ${HAVE_MKSTEMP}

/* Define to 1 if you have the `mktemp' function. */
#cmakedefine HAVE_MKTEMP ${HAVE_MKTEMP}

/* Define to 1 if you have a working `mmap' system call. */
#undef HAVE_MMAP

/* Define if mmap() uses MAP_ANONYMOUS to map anonymous pages, or undefine if
   it uses MAP_ANON */
#undef HAVE_MMAP_ANONYMOUS

/* Define if mmap() can map files into memory */
#undef HAVE_MMAP_FILE

/* Define to 1 if you have the <ndir.h> header file, and it defines `DIR'. */
#cmakedefine HAVE_NDIR_H ${HAVE_NDIR_H}

/* Define to 1 if you have the `nearbyintf' function. */
#cmakedefine HAVE_NEARBYINTF ${HAVE_NEARBYINTF}

/* Define to 1 if you have the `opendir' function. */
#cmakedefine HAVE_OPENDIR ${HAVE_OPENDIR}

/* Define to 1 if you have the `posix_spawn' function. */
#cmakedefine HAVE_POSIX_SPAWN ${HAVE_POSIX_SPAWN}

/* Define to 1 if you have the `powf' function. */
#cmakedefine HAVE_POWF ${HAVE_POWF}

/* Define to 1 if you have the `pread' function. */
#cmakedefine HAVE_PREAD ${HAVE_PREAD}

/* Define if libtool can extract symbol lists from object files. */
#undef HAVE_PRELOADED_SYMBOLS

/* Define to have the %a format string */
#undef HAVE_PRINTF_A

/* Have pthread_getspecific */
#cmakedefine HAVE_PTHREAD_GETSPECIFIC ${HAVE_PTHREAD_GETSPECIFIC}

/* Define to 1 if you have the <pthread.h> header file. */
#cmakedefine HAVE_PTHREAD_H ${HAVE_PTHREAD_H}

/* Have pthread_mutex_lock */
#cmakedefine HAVE_PTHREAD_MUTEX_LOCK ${HAVE_PTHREAD_MUTEX_LOCK}

/* Have pthread_rwlock_init */
#cmakedefine HAVE_PTHREAD_RWLOCK_INIT ${HAVE_PTHREAD_RWLOCK_INIT}

/* Define to 1 if srand48/lrand48/drand48 exist in <stdlib.h> */
#cmakedefine HAVE_RAND48 ${HAVE_RAND48}

/* Define to 1 if you have the `readdir' function. */
#cmakedefine HAVE_READDIR ${HAVE_READDIR}

/* Define to 1 if you have the `realpath' function. */
#cmakedefine HAVE_REALPATH ${HAVE_REALPATH}

/* Define to 1 if you have the `rintf' function. */
#undef HAVE_RINTF

/* Define to 1 if you have the `round' function. */
#cmakedefine HAVE_ROUND ${HAVE_ROUND}

/* Define to 1 if you have the `roundf' function. */
#undef HAVE_ROUNDF

/* Define to 1 if you have the `sbrk' function. */
#cmakedefine HAVE_SBRK ${HAVE_SBRK}

/* Define to 1 if you have the `setenv' function. */
#cmakedefine HAVE_SETENV ${HAVE_SETENV}

/* Define to 1 if you have the `setjmp' function. */
#cmakedefine HAVE_SETJMP ${HAVE_SETJMP}

/* Define to 1 if you have the `setrlimit' function. */
#cmakedefine HAVE_SETRLIMIT ${HAVE_SETRLIMIT}

/* Define if you have the shl_load function. */
#undef HAVE_SHL_LOAD

/* Define to 1 if you have the `siglongjmp' function. */
#cmakedefine HAVE_SIGLONGJMP ${HAVE_SIGLONGJMP}

/* Define to 1 if you have the <signal.h> header file. */
#cmakedefine HAVE_SIGNAL_H ${HAVE_SIGNAL_H}

/* Define to 1 if you have the `sigsetjmp' function. */
#cmakedefine HAVE_SIGSETJMP ${HAVE_SIGSETJMP}

/* Define to 1 if you have the <stdint.h> header file. */
#cmakedefine HAVE_STDINT_H ${HAVE_STDINT_H}

/* Set to 1 if the std::isinf function is found in <cmath> */
#undef HAVE_STD_ISINF_IN_CMATH

/* Set to 1 if the std::isnan function is found in <cmath> */
#undef HAVE_STD_ISNAN_IN_CMATH

/* Define to 1 if you have the `strdup' function. */
#cmakedefine HAVE_STRDUP ${HAVE_STRDUP}

/* Define to 1 if you have the `strerror' function. */
#cmakedefine HAVE_STRERROR ${HAVE_STRERROR}

/* Define to 1 if you have the `strerror_r' function. */
#cmakedefine HAVE_STRERROR_R ${HAVE_STRERROR_R}

/* Define to 1 if you have the `strtof' function. */
#cmakedefine HAVE_STRTOF ${HAVE_STRTOF}

/* Define to 1 if you have the `strtoll' function. */
#cmakedefine HAVE_STRTOLL ${HAVE_STRTOLL}

/* Define to 1 if you have the `strtoq' function. */
#cmakedefine HAVE_STRTOQ ${HAVE_STRTOQ}

/* Define to 1 if you have the `sysconf' function. */
#undef HAVE_SYSCONF

/* Define to 1 if you have the <sys/dir.h> header file, and it defines `DIR'.
   */
#cmakedefine HAVE_SYS_DIR_H ${HAVE_SYS_DIR_H}

/* Define to 1 if you have the <sys/ioctl.h> header file. */
#cmakedefine HAVE_SYS_IOCTL_H ${HAVE_SYS_IOCTL_H}

/* Define to 1 if you have the <sys/mman.h> header file. */
#cmakedefine HAVE_SYS_MMAN_H ${}

/* Define to 1 if you have the <sys/ndir.h> header file, and it defines `DIR'.
   */
#cmakedefine HAVE_SYS_NDIR_H ${HAVE_SYS_NDIR_H}

/* Define to 1 if you have the <sys/param.h> header file. */
#cmakedefine HAVE_SYS_PARAM_H ${HAVE_SYS_PARAM_H}

/* Define to 1 if you have the <sys/resource.h> header file. */
#cmakedefine HAVE_SYS_RESOURCE_H ${HAVE_SYS_RESOURCE_H}

/* Define to 1 if you have the <sys/stat.h> header file. */
#cmakedefine HAVE_SYS_STAT_H ${HAVE_SYS_STAT_H}

/* Define to 1 if you have the <sys/time.h> header file. */
#cmakedefine HAVE_SYS_TIME_H ${HAVE_SYS_TIME_H}

/* Define to 1 if you have the <sys/types.h> header file. */
#cmakedefine HAVE_SYS_TYPES_H ${HAVE_SYS_TYPES_H}

/* Define to 1 if you have the <sys/uio.h> header file. */
#cmakedefine HAVE_SYS_UIO_H ${HAVE_SYS_UIO_H}

/* Define to 1 if you have <sys/wait.h> that is POSIX.1 compatible. */
#cmakedefine HAVE_SYS_WAIT_H ${HAVE_SYS_WAIT_H}

/* Define if the setupterm() function is supported this platform. */
#cmakedefine HAVE_TERMINFO ${HAVE_TERMINFO}

/* Define to 1 if you have the <termios.h> header file. */
#cmakedefine HAVE_TERMIOS_H ${HAVE_TERMIOS_H}

/* Define to 1 if the system has the type `uint64_t'. */
#cmakedefine HAVE_UINT64_T ${HAVE_UINT64_T}

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine HAVE_UNISTD_H ${HAVE_UNISTD_H}

/* Define to 1 if you have the <utime.h> header file. */
#cmakedefine HAVE_UTIME_H ${HAVE_UTIME_H}

/* Define to 1 if the system has the type `u_int64_t'. */
#cmakedefine HAVE_U_INT64_T ${HAVE_U_INT64_T}

/* Define to 1 if you have the <valgrind/valgrind.h> header file. */
#cmakedefine HAVE_VALGRIND_VALGRIND_H ${HAVE_VALGRIND_VALGRIND_H}

/* Define to 1 if you have the `writev' function. */
#cmakedefine HAVE_WRITEV ${HAVE_WRITEV}

/* Define to 1 if you have the <zlib.h> header file. */
#cmakedefine HAVE_ZLIB_H ${HAVE_ZLIB_H}

/* Have host's _alloca */
#cmakedefine HAVE__ALLOCA ${HAVE__ALLOCA}

/* Have host's __alloca */
#cmakedefine HAVE___ALLOCA ${HAVE___ALLOCA}

/* Have host's __ashldi3 */
#cmakedefine HAVE___ASHLDI3 ${HAVE___ASHLDI3}

/* Have host's __ashrdi3 */
#cmakedefine HAVE___ASHRDI3 ${HAVE___ASHRDI3}

/* Have host's __chkstk */
#cmakedefine HAVE___CHKSTK ${HAVE___CHKSTK}

/* Have host's __chkstk_ms */
#cmakedefine HAVE___CHKSTK_MS ${HAVE___CHKSTK_MS}

/* Have host's __cmpdi2 */
#cmakedefine HAVE___CMPDI2 ${HAVE___CMPDI2}

/* Have host's __divdi3 */
#cmakedefine HAVE___DIVDI3 ${HAVE___DIVDI3}

/* Define to 1 if you have the `__dso_handle' function. */
#undef HAVE___DSO_HANDLE

/* Have host's __fixdfdi */
#cmakedefine HAVE___FIXDFDI ${HAVE___FIXDFDI}

/* Have host's __fixsfdi */
#cmakedefine HAVE___FIXSFDI ${HAVE___FIXSFDI}

/* Have host's __floatdidf */
#cmakedefine HAVE___FLOATDIDF ${HAVE___FLOATDIDF}

/* Have host's __lshrdi3 */
#cmakedefine HAVE___LSHRDI3 ${HAVE___LSHRDI3}

/* Have host's __main */
#cmakedefine HAVE___MAIN ${HAVE___MAIN}

/* Have host's __moddi3 */
#cmakedefine HAVE___MODDI3 ${HAVE___MODDI3}

/* Have host's __udivdi3 */
#cmakedefine HAVE___UDIVDI3 ${HAVE___UDIVDI3}

/* Have host's __umoddi3 */
#cmakedefine HAVE___UMODDI3 ${HAVE___UMODDI3}

/* Have host's ___chkstk */
#cmakedefine HAVE____CHKSTK ${HAVE____CHKSTK}

/* Have host's ___chkstk_ms */
#cmakedefine HAVE____CHKSTK_MS ${HAVE____CHKSTK_MS}

/* Define if we link Polly to the tools */
#cmakedefine LINK_POLLY_INTO_TOOLS

/* Define if zlib compression is available */
#cmakedefine01 LLVM_ENABLE_ZLIB

/* Define if the OS needs help to load dependent libraries for dlopen(). */
#cmakedefine LTDL_DLOPEN_DEPLIBS ${LTDL_DLOPEN_DEPLIBS}

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#undef LTDL_OBJDIR

/* Define to the extension used for shared libraries, say, ".so". */
#cmakedefine LTDL_SHLIB_EXT "${LTDL_SHLIB_EXT}"

/* Define to the system default library search path. */
#cmakedefine LTDL_SYSSEARCHPATH "${LTDL_SYSSEARCHPATH}"

/* Define if /dev/zero should be used when mapping RWX memory, or undefine if
   its not necessary */
#undef NEED_DEV_ZERO_FOR_MMAP

/* Define if dlsym() requires a leading underscore in symbol names. */
#undef NEED_USCORE

/* Define to the address where bug reports for this package should be sent. */
#cmakedefine PACKAGE_BUGREPORT "${PACKAGE_BUGREPORT}"

/* Define to the full name of this package. */
#cmakedefine PACKAGE_NAME "${PACKAGE_NAME}"

/* Define to the full name and version of this package. */
#cmakedefine PACKAGE_STRING "${PACKAGE_STRING}"

/* Define to the one symbol short name of this package. */
#undef PACKAGE_TARNAME

/* Define to the version of this package. */
#cmakedefine PACKAGE_VERSION "${PACKAGE_VERSION}"

/* Define as the return type of signal handlers (`int' or `void'). */
#cmakedefine RETSIGTYPE ${RETSIGTYPE}

/* Define to 1 if the `S_IS*' macros in <sys/stat.h> do not work properly. */
#undef STAT_MACROS_BROKEN

/* Define to 1 if you have the ANSI C header files. */
#undef STDC_HEADERS

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#undef TIME_WITH_SYS_TIME

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
#undef TM_IN_SYS_TIME

/* Type of 1st arg on ELM Callback */
#cmakedefine WIN32_ELMCB_PCSTR ${WIN32_ELMCB_PCSTR}

/* Define to `int' if <sys/types.h> does not define. */
#undef pid_t

/* Define to `unsigned int' if <sys/types.h> does not define. */
#undef size_t

/* Define to a function replacing strtoll */
#cmakedefine strtoll ${strtoll}

/* Define to a function implementing strtoull */
#cmakedefine strtoull ${strtoull}

/* Define to a function implementing stricmp */
#cmakedefine stricmp ${stricmp}

/* Define to a function implementing strdup */
#cmakedefine strdup ${strdup}

/* Define to 1 if you have the `_chsize_s' function. */
#cmakedefine HAVE__CHSIZE_S ${HAVE__CHSIZE_S}

#endif
