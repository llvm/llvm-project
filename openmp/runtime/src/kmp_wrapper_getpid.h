/*
 * kmp_wrapper_getpid.h -- getpid() declaration.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_WRAPPER_GETPID_H
#define KMP_WRAPPER_GETPID_H

#if KMP_OS_UNIX

// On Unix-like systems (Linux* OS and OS X*) getpid() is declared in standard
// headers.
#if !defined(KMP_OS_AIX)
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#include <unistd.h>
#if KMP_OS_DARWIN
// OS X
#define __kmp_gettid() pthread_mach_thread_np(pthread_self())
#elif KMP_OS_FREEBSD || KMP_OS_DRAGONFLY
#include <pthread_np.h>
#define __kmp_gettid() pthread_getthreadid_np()
#elif KMP_OS_NETBSD
#include <lwp.h>
#define __kmp_gettid() _lwp_self()
#elif KMP_OS_OPENBSD
#define __kmp_gettid() getthrid()
#elif KMP_OS_AIX
#include <pthread.h>
#define __kmp_gettid() pthread_self()
#elif defined(SYS_gettid)
// Hopefully other Unix systems define SYS_gettid syscall for getting os thread
// id
#define __kmp_gettid() syscall(SYS_gettid)
#else
#warning No gettid found, use getpid instead
#define __kmp_gettid() getpid()
#endif

#elif KMP_OS_WINDOWS

// On Windows* OS _getpid() returns int (not pid_t) and is declared in
// "process.h".
#include <process.h>
// Let us simulate Unix.
#if KMP_MSVC_COMPAT
typedef int pid_t;
#endif
#define getpid _getpid
#define __kmp_gettid() GetCurrentThreadId()

#else

#error Unknown or unsupported OS.

#endif

/* TODO: All the libomp source code uses pid_t type for storing the result of
   getpid(), it is good. But often it printed as "%d", that is not good, because
   it ignores pid_t definition (may pid_t be longer that int?). It seems all pid
   prints should be rewritten as:

   printf( "%" KMP_UINT64_SPEC, (kmp_uint64) pid );

   or (at least) as

   printf( "%" KMP_UINT32_SPEC, (kmp_uint32) pid );

   (kmp_uint32, kmp_uint64, KMP_UINT64_SPEC, and KMP_UNIT32_SPEC are defined in
   "kmp_os.h".)  */

#endif // KMP_WRAPPER_GETPID_H

// end of file //
