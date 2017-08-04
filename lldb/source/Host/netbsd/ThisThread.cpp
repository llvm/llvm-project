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
#include <string.h>

using namespace lldb_private;

void ThisThread::SetName(llvm::StringRef name) {
  HostNativeThread::SetName(::pthread_self(), name);
}

void ThisThread::GetName(llvm::SmallVectorImpl<char> &name) {
  HostNativeThread::GetName(::pthread_self(), name);
}
