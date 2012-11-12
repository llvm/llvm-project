/* include/llvm/Config/config.h.cmake corresponding to config.h.in. */

#ifndef CONFIG_H
#define CONFIG_H

/* Bug report URL. */
#define BUG_REPORT_URL "${BUG_REPORT_URL}"

/* Relative directory for resource files */
#define CLANG_RESOURCE_DIR "${CLANG_RESOURCE_DIR}"

/* Directories clang will search for headers */
#define C_INCLUDE_DIRS "${C_INCLUDE_DIRS}"

/* Default <path> to all compiler invocations for --sysroot=<path>. */
#undef DEFAULT_SYSROOT

/* Define if you want backtraces on crash */
#cmakedefine ENABLE_BACKTRACES

/* Define if position independent code is enabled */
#cmakedefine ENABLE_PIC

/* Define if timestamp information (e.g., __DATE__) is allowed */
#cmakedefine ENABLE_TIMESTAMPS ${ENABLE_TIMESTAMPS}

/* Directory where gcc is installed. */
#undef GCC_INSTALL_PREFIX

/* Define to 1 if you have the `arc4random' function. */
#cmakedefine HAVE_ARC4RANDOM

/* Define to 1 if you have the `argz_append' function. */
#cmakedefine HAVE_ARGZ_APPEND ${HAVE_ARGZ_APPEND}

/* Define to 1 if you have the `argz_create_sep' function. */
#cmakedefine HAVE_ARGZ_CREATE_SEP ${HAVE_ARGZ_CREATE_SEP}

/* Define to 1 if you have the <argz.h> header file. */
#cmakedefine HAVE_ARGZ_H ${HAVE_ARGZ_H}

/* Define to 1 if you have the `argz_insert' function. */
#cmakedefine HAVE_ARGZ_INSERT ${HAVE_ARGZ_INSERT}

/* Define to 1 if you have the `argz_next' function. */
#cmakedefine HAVE_ARGZ_NEXT ${HAVE_ARGZ_NEXT}

/* Define to 1 if you have the `argz_stringify' function. */
#cmakedefine HAVE_ARGZ_STRINGIFY ${HAVE_ARGZ_STRINGIFY}

/* Define to 1 if you have the <assert.h> header file. */
#cmakedefine HAVE_ASSERT_H ${HAVE_ASSERT_H}

/* Define to 1 if you have the `backtrace' function. */
#cmakedefine HAVE_BACKTRACE ${HAVE_BACKTRACE}

/* Define to 1 if you have the `bcopy' function. */
#undef HAVE_BCOPY

/* Define to 1 if you have the `ceilf' function. */
#cmakedefine HAVE_CEILF ${HAVE_CEILF}

/* Define if the neat program is available */
#cmakedefine HAVE_CIRCO ${HAVE_CIRCO}

/* Define to 1 if you have the `closedir' function. */
#cmakedefine HAVE_CLOSEDIR ${HAVE_CLOSEDIR}

/* Define to 1 if you have the <CrashReporterClient.h> header file. */
#undef HAVE_CRASHREPORTERCLIENT_H

/* Define if __crashreporter_info__ exists. */
#undef HAVE_CRASHREPORTER_INFO

/* Define to 1 if you have the <ctype.h> header file. */
#cmakedefine HAVE_CTYPE_H ${HAVE_CTYPE_H}

/* Define to 1 if you have the declaration of `strerror_s', and to 0 if you
   don't. */
#cmakedefine01 HAVE_DECL_STRERROR_S

/* Define to 1 if you have the <dirent.h> header file, and it defines `DIR'.
   */
#cmakedefine HAVE_DIRENT_H ${HAVE_DIRENT_H}

/* Define if you have the GNU dld library. */
#undef HAVE_DLD

/* Define to 1 if you have the <dld.h> header file. */
#cmakedefine HAVE_DLD_H ${HAVE_DLD_H}

/* Define to 1 if you have the `dlerror' function. */
#cmakedefine HAVE_DLERROR ${HAVE_DLERROR}

/* Define to 1 if you have the <dlfcn.h> header file. */
#cmakedefine HAVE_DLFCN_H ${HAVE_DLFCN_H}

/* Define if dlopen() is available on this platform. */
#cmakedefine HAVE_DLOPEN ${HAVE_DLOPEN}

/* Define to 1 if you have the <dl.h> header file. */
#cmakedefine HAVE_DL_H ${HAVE_DL_H}

/* Define if the dot program is available */
#cmakedefine HAVE_DOT ${HAVE_DOT}

/* Define if the dotty program is available */
#cmakedefine HAVE_DOTTY ${HAVE_DOTTY}

/* Define if you have the _dyld_func_lookup function. */
#undef HAVE_DYLD

/* Define to 1 if you have the <errno.h> header file. */
#cmakedefine HAVE_ERRNO_H ${HAVE_ERRNO_H}

/* Define to 1 if the system has the type `error_t'. */
#cmakedefine HAVE_ERROR_T ${HAVE_ERROR_T}

/* Define to 1 if you have the <execinfo.h> header file. */
#cmakedefine HAVE_EXECINFO_H ${HAVE_EXECINFO_H}

/* Define to 1 if you have the <fcntl.h> header file. */
#cmakedefine HAVE_FCNTL_H ${HAVE_FCNTL_H}

/* Define if the neat program is available */
#cmakedefine HAVE_FDP ${HAVE_FDP}

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

/* Define to 1 if you have the `fmodf' function. */
#cmakedefine HAVE_FMODF ${HAVE_FMODF}

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

/* Define if the Graphviz program is available */
#cmakedefine HAVE_GRAPHVIZ ${HAVE_GRAPHVIZ}

/* Define if the gv program is available */
#cmakedefine HAVE_GV ${HAVE_GV}

/* Define to 1 if you have the `index' function. */
#cmakedefine HAVE_INDEX ${HAVE_INDEX}

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

/* Define to 1 if you have the `udis86' library (-ludis86). */
#undef HAVE_LIBUDIS86

/* Define to 1 if you have the <limits.h> header file. */
#cmakedefine HAVE_LIMITS_H ${HAVE_LIMITS_H}

/* Define if you can use -Wl,-export-dynamic. */
#define HAVE_LINK_EXPORT_DYNAMIC 1

/* Define to 1 if you have the <link.h> header file. */
#cmakedefine HAVE_LINK_H ${HAVE_LINK_H}

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

/* Define to 1 if you have the `memcpy' function. */
#cmakedefine HAVE_MEMCPY ${HAVE_MEMCPY}

/* Define to 1 if you have the `memmove' function. */
#cmakedefine HAVE_MEMMOVE ${HAVE_MEMMOVE}

/* Define to 1 if you have the <memory.h> header file. */
#cmakedefine HAVE_MEMORY_H ${HAVE_MEMORY_H}

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

/* Define if the neat program is available */
#cmakedefine HAVE_NEATO ${HAVE_NEATO}

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
#undef HAVE_REALPATH

/* Define to 1 if you have the `rindex' function. */
#cmakedefine HAVE_RINDEX ${HAVE_RINDEX}

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

/* Define to 1 if you have the <setjmp.h> header file. */
#cmakedefine HAVE_SETJMP_H ${HAVE_SETJMP_H}

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

/* Define to 1 if you have the <stdio.h> header file. */
#cmakedefine HAVE_STDIO_H ${HAVE_STDIO_H}

/* Define to 1 if you have the <stdlib.h> header file. */
#cmakedefine HAVE_STDLIB_H ${HAVE_STDLIB_H}

/* Set to 1 if the std::isinf function is found in <cmath> */
#undef HAVE_STD_ISINF_IN_CMATH

/* Set to 1 if the std::isnan function is found in <cmath> */
#undef HAVE_STD_ISNAN_IN_CMATH

/* Define to 1 if you have the `strchr' function. */
#cmakedefine HAVE_STRCHR ${HAVE_STRCHR}

/* Define to 1 if you have the `strcmp' function. */
#cmakedefine HAVE_STRCMP ${HAVE_STRCMP}

/* Define to 1 if you have the `strdup' function. */
#cmakedefine HAVE_STRDUP ${HAVE_STRDUP}

/* Define to 1 if you have the `strerror' function. */
#cmakedefine HAVE_STRERROR ${HAVE_STRERROR}

/* Define to 1 if you have the `strerror_r' function. */
#cmakedefine HAVE_STRERROR_R ${HAVE_STRERROR_R}

/* Define to 1 if you have the <strings.h> header file. */
#cmakedefine HAVE_STRINGS_H ${HAVE_STRINGS_H}

/* Define to 1 if you have the <string.h> header file. */
#cmakedefine HAVE_STRING_H ${HAVE_STRING_H}

/* Define to 1 if you have the `strrchr' function. */
#cmakedefine HAVE_STRRCHR ${HAVE_STRRCHR}

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

/* Define to 1 if you have the <sys/dl.h> header file. */
#cmakedefine HAVE_SYS_DL_H ${HAVE_SYS_DL_H}

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

/* Define to 1 if you have the <termios.h> header file. */
#cmakedefine HAVE_TERMIOS_H ${HAVE_TERMIOS_H}

/* Define if the neat program is available */
#cmakedefine HAVE_TWOPI ${HAVE_TWOPI}

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

/* Define to 1 if you have the <windows.h> header file. */
#cmakedefine HAVE_WINDOWS_H ${HAVE_WINDOWS_H}

/* Define to 1 if you have the `writev' function. */
#cmakedefine HAVE_WRITEV ${HAVE_WRITEV}

/* Define if the xdot.py program is available */
#cmakedefine HAVE_XDOT_PY ${HAVE_XDOT_PY}

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

/* Linker version detected at compile time. */
#undef HOST_LINK_VERSION

/* Installation directory for binary executables */
#cmakedefine LLVM_BINDIR "${LLVM_BINDIR}"

/* Time at which LLVM was configured */
#cmakedefine LLVM_CONFIGTIME "${LLVM_CONFIGTIME}"

/* Installation directory for data files */
#cmakedefine LLVM_DATADIR "${LLVM_DATADIR}"

/* Target triple LLVM will generate code for by default */
#cmakedefine LLVM_DEFAULT_TARGET_TRIPLE "${LLVM_DEFAULT_TARGET_TRIPLE}"

/* Installation directory for documentation */
#cmakedefine LLVM_DOCSDIR "${LLVM_DOCSDIR}"

/* Define if threads enabled */
#cmakedefine01 LLVM_ENABLE_THREADS

/* Installation directory for config files */
#cmakedefine LLVM_ETCDIR "${LLVM_ETCDIR}"

/* Has gcc/MSVC atomic intrinsics */
#cmakedefine01 LLVM_HAS_ATOMICS

/* Host triple LLVM will be executed on */
#cmakedefine LLVM_HOSTTRIPLE "${LLVM_HOSTTRIPLE}"

/* Installation directory for include files */
#cmakedefine LLVM_INCLUDEDIR "${LLVM_INCLUDEDIR}"

/* Installation directory for .info files */
#cmakedefine LLVM_INFODIR "${LLVM_INFODIR}"

/* Installation directory for libraries */
#cmakedefine LLVM_LIBDIR "${LLVM_LIBDIR}"

/* Installation directory for man pages */
#cmakedefine LLVM_MANDIR "${LLVM_MANDIR}"

/* LLVM architecture name for the native architecture, if available */
#cmakedefine LLVM_NATIVE_ARCH ${LLVM_NATIVE_ARCH}

/* LLVM name for the native AsmParser init function, if available */
#cmakedefine LLVM_NATIVE_ASMPARSER LLVMInitialize${LLVM_NATIVE_ARCH}AsmParser

/* LLVM name for the native AsmPrinter init function, if available */
#cmakedefine LLVM_NATIVE_ASMPRINTER LLVMInitialize${LLVM_NATIVE_ARCH}AsmPrinter

/* LLVM name for the native Disassembler init function, if available */
#cmakedefine LLVM_NATIVE_DISASSEMBLER LLVMInitialize${LLVM_NATIVE_ARCH}Disassembler

/* LLVM name for the native Target init function, if available */
#cmakedefine LLVM_NATIVE_TARGET LLVMInitialize${LLVM_NATIVE_ARCH}Target

/* LLVM name for the native TargetInfo init function, if available */
#cmakedefine LLVM_NATIVE_TARGETINFO LLVMInitialize${LLVM_NATIVE_ARCH}TargetInfo

/* LLVM name for the native target MC init function, if available */
#cmakedefine LLVM_NATIVE_TARGETMC LLVMInitialize${LLVM_NATIVE_ARCH}TargetMC

/* Define if this is Unixish platform */
#cmakedefine LLVM_ON_UNIX ${LLVM_ON_UNIX}

/* Define if this is Win32ish platform */
#cmakedefine LLVM_ON_WIN32 ${LLVM_ON_WIN32}

/* Define to path to circo program if found or 'echo circo' otherwise */
#cmakedefine LLVM_PATH_CIRCO "${LLVM_PATH_CIRCO}"

/* Define to path to dot program if found or 'echo dot' otherwise */
#cmakedefine LLVM_PATH_DOT "${LLVM_PATH_DOT}"

/* Define to path to dotty program if found or 'echo dotty' otherwise */
#cmakedefine LLVM_PATH_DOTTY "${LLVM_PATH_DOTTY}"

/* Define to path to fdp program if found or 'echo fdp' otherwise */
#cmakedefine LLVM_PATH_FDP "${LLVM_PATH_FDP}"

/* Define to path to Graphviz program if found or 'echo Graphviz' otherwise */
#cmakedefine LLVM_PATH_GRAPHVIZ "${LLVM_PATH_GRAPHVIZ}"

/* Define to path to gv program if found or 'echo gv' otherwise */
#cmakedefine LLVM_PATH_GV "${LLVM_PATH_GV}"

/* Define to path to neato program if found or 'echo neato' otherwise */
#cmakedefine LLVM_PATH_NEATO "${LLVM_PATH_NEATO}"

/* Define to path to twopi program if found or 'echo twopi' otherwise */
#cmakedefine LLVM_PATH_TWOPI "${LLVM_PATH_TWOPI}"

/* Define to path to xdot.py program if found or 'echo xdot.py' otherwise */
#cmakedefine LLVM_PATH_XDOT_PY "${LLVM_PATH_XDOT_PY}"

/* Installation prefix directory */
#cmakedefine LLVM_PREFIX "${LLVM_PREFIX}"

/* Define if we have the Intel JIT API runtime support library */
#cmakedefine LLVM_USE_INTEL_JITEVENTS 1

/* Define if we have the oprofile JIT-support library */
#cmakedefine LLVM_USE_OPROFILE 1

/* Major version of the LLVM API */
#cmakedefine LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR}

/* Minor version of the LLVM API */
#cmakedefine LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR}

/* Define if the OS needs help to load dependent libraries for dlopen(). */
#cmakedefine LTDL_DLOPEN_DEPLIBS ${LTDL_DLOPEN_DEPLIBS}

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#undef LTDL_OBJDIR

/* Define to the name of the environment variable that determines the dynamic
   library search path. */
#cmakedefine LTDL_SHLIBPATH_VAR "${LTDL_SHLIBPATH_VAR}"

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

/* Define if use udis86 library */
#undef USE_UDIS86

/* Type of 1st arg on ELM Callback */
#cmakedefine WIN32_ELMCB_PCSTR ${WIN32_ELMCB_PCSTR}

/* Define to empty if `const' does not conform to ANSI C. */
#undef const

/* Define to a type to use for `error_t' if it is not otherwise available. */
#cmakedefine error_t ${error_t}

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

/* Added by Kevin -- Maximum path length */
#cmakedefine MAXPATHLEN ${MAXPATHLEN}

#endif
