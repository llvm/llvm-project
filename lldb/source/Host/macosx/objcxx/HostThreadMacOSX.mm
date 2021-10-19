//===-- HostThreadMacOSX.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/macosx/HostThreadMacOSX.h"
#include "lldb/Host/Host.h"

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>

#include <pthread.h>

using namespace lldb_private;


static pthread_once_t g_thread_create_once = PTHREAD_ONCE_INIT;
static pthread_key_t g_thread_create_key = 0;

namespace {
class MacOSXDarwinThread {
public:
  MacOSXDarwinThread() { m_pool = [[NSAutoreleasePool alloc] init]; }

  ~MacOSXDarwinThread() {
    if (m_pool) {
      [m_pool drain];
      m_pool = nil;
    }
  }

  static void PThreadDestructor(void *v) {
    if (v)
      delete static_cast<MacOSXDarwinThread *>(v);
    ::pthread_setspecific(g_thread_create_key, NULL);
  }

protected:
  NSAutoreleasePool *m_pool = nil;

private:
  MacOSXDarwinThread(const MacOSXDarwinThread &) = delete;
  const MacOSXDarwinThread &operator=(const MacOSXDarwinThread &) = delete;
};
} // namespace

static void InitThreadCreated() {
  ::pthread_key_create(&g_thread_create_key,
                       MacOSXDarwinThread::PThreadDestructor);
}

HostThreadMacOSX::HostThreadMacOSX() : HostThreadPosix() {}

HostThreadMacOSX::HostThreadMacOSX(lldb::thread_t thread)
    : HostThreadPosix(thread) {}

lldb::thread_result_t
HostThreadMacOSX::ThreadCreateTrampoline(lldb::thread_arg_t arg) {
  ::pthread_once(&g_thread_create_once, InitThreadCreated);
  if (g_thread_create_key) {
    ::pthread_setspecific(g_thread_create_key, new MacOSXDarwinThread());
  }

  return HostThreadPosix::ThreadCreateTrampoline(arg);
}
