//===-- HostMemory.cpp - HostMemory implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of HostMemory internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/HostMemory.h"
#include "streamexecutor/Device.h"

namespace streamexecutor {
namespace internal {

void destroyRegisteredHostMemoryInternals(Device *TheDevice, void *Pointer) {
  // TODO(jhen): How to handle errors here?
  if (Pointer) {
    consumeError(TheDevice->unregisterHostMemory(Pointer));
  }
}

} // namespace internal
} // namespace streamexecutor
