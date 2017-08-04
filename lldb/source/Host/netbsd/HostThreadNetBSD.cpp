//===-- HostThreadNetBSD.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// lldb Includes
#include "lldb/Host/netbsd/HostThreadNetBSD.h"
#include "lldb/Host/Host.h"

// C includes
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/user.h>

// C++ includes
#include <string>

using namespace lldb_private;

HostThreadNetBSD::HostThreadNetBSD() {}

HostThreadNetBSD::HostThreadNetBSD(lldb::thread_t thread)
    : HostThreadPosix(thread) {}

void HostThreadNetBSD::SetName(lldb::thread_t thread, llvm::StringRef &name) {
  ::pthread_setname_np(thread, "%s", const_cast<char *>(name.data()));
}

void HostThreadNetBSD::GetName(lldb::thread_t thread,
                               llvm::SmallVectorImpl<char> &name) {
  char buf[PTHREAD_MAX_NAMELEN_NP];
  ::pthread_getname_np(thread, buf, PTHREAD_MAX_NAMELEN_NP);

  name.clear();
  name.append(buf, buf + strlen(buf));
}
