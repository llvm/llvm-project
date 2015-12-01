/*
 * kmp_config.h -- Feature macros
 */
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
#ifndef KMP_CONFIG_H
#define KMP_CONFIG_H

#include "kmp_platform.h"

// cmakedefine01 MACRO will define MACRO as either 0 or 1
// cmakedefine MACRO 1 will define MACRO as 1 or leave undefined
#cmakedefine01 DEBUG_BUILD
#cmakedefine01 RELWITHDEBINFO_BUILD
#cmakedefine01 LIBOMP_USE_ITT_NOTIFY
#define USE_ITT_NOTIFY LIBOMP_USE_ITT_NOTIFY
#if ! LIBOMP_USE_ITT_NOTIFY
# define INTEL_NO_ITTNOTIFY_API
#endif
#cmakedefine01 LIBOMP_USE_VERSION_SYMBOLS
#if LIBOMP_USE_VERSION_SYMBOLS
# define KMP_USE_VERSION_SYMBOLS
#endif
#cmakedefine01 LIBOMP_HAVE_WEAK_ATTRIBUTE
#define KMP_HAVE_WEAK_ATTRIBUTE LIBOMP_HAVE_WEAK_ATTRIBUTE
#cmakedefine01 LIBOMP_HAVE_PSAPI
#define KMP_HAVE_PSAPI LIBOMP_HAVE_PSAPI
#cmakedefine01 LIBOMP_STATS
#define KMP_STATS_ENABLED LIBOMP_STATS
#cmakedefine01 LIBOMP_USE_DEBUGGER
#define USE_DEBUGGER LIBOMP_USE_DEBUGGER
#cmakedefine01 LIBOMP_OMPT_DEBUG
#define OMPT_DEBUG LIBOMP_OMPT_DEBUG
#cmakedefine01 LIBOMP_OMPT_SUPPORT
#define OMPT_SUPPORT LIBOMP_OMPT_SUPPORT
#cmakedefine01 LIBOMP_OMPT_BLAME
#define OMPT_BLAME LIBOMP_OMPT_BLAME
#cmakedefine01 LIBOMP_OMPT_TRACE
#define OMPT_TRACE LIBOMP_OMPT_TRACE
#cmakedefine01 LIBOMP_USE_ADAPTIVE_LOCKS
#define KMP_USE_ADAPTIVE_LOCKS LIBOMP_USE_ADAPTIVE_LOCKS
#define KMP_DEBUG_ADAPTIVE_LOCKS 0
#cmakedefine01 LIBOMP_USE_INTERNODE_ALIGNMENT
#define KMP_USE_INTERNODE_ALIGNMENT LIBOMP_USE_INTERNODE_ALIGNMENT
#cmakedefine01 LIBOMP_ENABLE_ASSERTIONS
#define KMP_USE_ASSERT LIBOMP_ENABLE_ASSERTIONS
#cmakedefine01 STUBS_LIBRARY
#cmakedefine01 LIBOMP_USE_HWLOC
#define KMP_USE_HWLOC LIBOMP_USE_HWLOC
#define KMP_ARCH_STR "@LIBOMP_LEGAL_ARCH@"
#define KMP_LIBRARY_FILE "@LIBOMP_LIB_FILE@"
#define KMP_VERSION_MAJOR @LIBOMP_VERSION_MAJOR@
#define KMP_VERSION_MINOR @LIBOMP_VERSION_MINOR@
#define LIBOMP_OMP_VERSION @LIBOMP_OMP_VERSION@
#define OMP_50_ENABLED (LIBOMP_OMP_VERSION >= 50)
#define OMP_41_ENABLED (LIBOMP_OMP_VERSION >= 41)
#define OMP_40_ENABLED (LIBOMP_OMP_VERSION >= 40)
#define OMP_30_ENABLED (LIBOMP_OMP_VERSION >= 30)

// Configured cache line based on architecture
#if KMP_ARCH_PPC64
# define CACHE_LINE 128
#else
# define CACHE_LINE 64
#endif

#define KMP_DYNAMIC_LIB 1
#define KMP_NESTED_HOT_TEAMS 1
#define KMP_ADJUST_BLOCKTIME 1
#define BUILD_PARALLEL_ORDERED 1
#define KMP_ASM_INTRINS 1
#define USE_ITT_BUILD 1
#define INTEL_ITTNOTIFY_PREFIX __kmp_itt_
#if ! KMP_MIC
# define USE_LOAD_BALANCE 1
#endif
#if ! (KMP_OS_WINDOWS || KMP_OS_DARWIN)
# define KMP_TDATA_GTID 1
#endif
#if STUBS_LIBRARY
# define KMP_STUB 1
#endif
#if DEBUG_BUILD || RELWITHDEBINFO_BUILD
# define KMP_DEBUG 1
#endif

#if KMP_OS_WINDOWS
# define KMP_WIN_CDECL
#else
# define BUILD_TV
# define KMP_GOMP_COMPAT
#endif

#endif // KMP_CONFIG_H
