//===-- HostThreadMacOSX.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/macosx/HostThreadMacOSX.h"
#include "lldb/Host/Host.h"

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>

#include <objc/objc-auto.h>
#include <pthread.h>

using namespace lldb_private;

namespace {

pthread_once_t g_thread_create_once = PTHREAD_ONCE_INIT;
pthread_key_t g_thread_create_key = 0;

class MacOSXDarwinThread {
public:
  MacOSXDarwinThread() : m_pool(nil) {
    // Register our thread with the collector if garbage collection is enabled.
    if (objc_collectingEnabled()) {
#if MAC_OS_X_VERSION_MAX_ALLOWED <= MAC_OS_X_VERSION_10_5
      // On Leopard and earlier there is no way objc_registerThreadWithCollector
      // function, so we do it manually.
      auto_zone_register_thread(auto_zone());
#else
      // On SnowLeopard and later we just call the thread registration function.
      objc_registerThreadWithCollector();
#endif
    } else {
      m_pool = [[NSAutoreleasePool alloc] init];
    }
  }

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
  NSAutoreleasePool *m_pool;

private:
  DISALLOW_COPY_AND_ASSIGN(MacOSXDarwinThread);
};

void InitThreadCreated() {
  ::pthread_key_create(&g_thread_create_key,
                       MacOSXDarwinThread::PThreadDestructor);
}
} // namespace

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
