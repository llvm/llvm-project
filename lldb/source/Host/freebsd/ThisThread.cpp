//===-- ThisThread.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ThisThread.h"
#include "lldb/Host/HostNativeThread.h"

#include "llvm/ADT/SmallVector.h"

#include <pthread.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

using namespace lldb_private;

void ThisThread::SetName(llvm::StringRef name) {
#if defined(__FreeBSD__) // Kfreebsd does not have a simple alternative
  ::pthread_set_name_np(::pthread_self(), name.data());
#endif
}

void ThisThread::GetName(llvm::SmallVectorImpl<char> &name) {
#if defined(__FreeBSD__)
  HostNativeThread::GetName(::pthread_getthreadid_np(), name);
#else
  // Kfreebsd
  HostNativeThread::GetName((unsigned)pthread_self(), name);
#endif
}
