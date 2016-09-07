//===-- LibcGlue.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// This files adds functions missing from libc on earlier versions of Android

#include <android/api-level.h>

#include <sys/syscall.h>

#if __ANDROID_API__ < 21

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "lldb/Host/Time.h"

time_t timegm(struct tm *t) { return (time_t)timegm64(t); }

int posix_openpt(int flags) { return open("/dev/ptmx", flags); }

#endif
