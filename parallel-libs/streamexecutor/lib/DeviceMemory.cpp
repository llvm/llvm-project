//===-- DeviceMemory.cpp - DeviceMemory implementation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of DeviceMemory class internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/DeviceMemory.h"

#include "streamexecutor/Device.h"

namespace streamexecutor {

GlobalDeviceMemoryBase::~GlobalDeviceMemoryBase() {
  if (Handle)
    // TODO(jhen): How to handle errors here.
    consumeError(TheDevice->freeDeviceMemory(*this));
}

} // namespace streamexecutor
