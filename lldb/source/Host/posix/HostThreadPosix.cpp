//===-- HostThreadPosix.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/HostThreadPosix.h"
#include "lldb/Core/Error.h"

#include <errno.h>
#include <pthread.h>

using namespace lldb;
using namespace lldb_private;

HostThreadPosix::HostThreadPosix() {}

HostThreadPosix::HostThreadPosix(lldb::thread_t thread)
    : HostNativeThreadBase(thread) {}

HostThreadPosix::~HostThreadPosix() {}

Error HostThreadPosix::Join(lldb::thread_result_t *result) {
  Error error;
  if (IsJoinable()) {
    int err = ::pthread_join(m_thread, result);
    error.SetError(err, lldb::eErrorTypePOSIX);
  } else {
    if (result)
      *result = NULL;
    error.SetError(EINVAL, eErrorTypePOSIX);
  }

  Reset();
  return error;
}

Error HostThreadPosix::Cancel() {
  Error error;
  if (IsJoinable()) {
#ifndef __ANDROID__
    assert(false && "someone is calling HostThread::Cancel()");
    int err = ::pthread_cancel(m_thread);
    error.SetError(err, eErrorTypePOSIX);
#else
    error.SetErrorString("HostThreadPosix::Cancel() not supported on Android");
#endif
  }
  return error;
}

Error HostThreadPosix::Detach() {
  Error error;
  if (IsJoinable()) {
    int err = ::pthread_detach(m_thread);
    error.SetError(err, eErrorTypePOSIX);
  }
  Reset();
  return error;
}
