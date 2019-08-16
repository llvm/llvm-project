//= llvm/Support/Unix/RWMutex.inc - Unix Reader/Writer Mutual Exclusion Lock  =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific (non-pthread) RWMutex class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "llvm/Support/Mutex.h"

namespace llvm {

using namespace sys;

// This naive implementation treats readers the same as writers.  This
// will therefore deadlock if a thread tries to acquire a read lock
// multiple times.

RWMutexImpl::RWMutexImpl() : data_(new MutexImpl(false)) { }

RWMutexImpl::~RWMutexImpl() {
  delete static_cast<MutexImpl *>(data_);
}

bool RWMutexImpl::reader_acquire() {
  return static_cast<MutexImpl *>(data_)->acquire();
}

bool RWMutexImpl::reader_release() {
  return static_cast<MutexImpl *>(data_)->release();
}

bool RWMutexImpl::writer_acquire() {
  return static_cast<MutexImpl *>(data_)->acquire();
}

bool RWMutexImpl::writer_release() {
  return static_cast<MutexImpl *>(data_)->release();
}

}
