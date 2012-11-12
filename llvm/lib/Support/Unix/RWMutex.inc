//= llvm/Support/Unix/RWMutex.inc - Unix Reader/Writer Mutual Exclusion Lock  =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace llvm {

using namespace sys;

RWMutexImpl::RWMutexImpl() { }

RWMutexImpl::~RWMutexImpl() { }

bool RWMutexImpl::reader_acquire() {
  return true;
}

bool RWMutexImpl::reader_release() {
  return true;
}

bool RWMutexImpl::writer_acquire() {
  return true;
}

bool RWMutexImpl::writer_release() {
  return true;
}

}
