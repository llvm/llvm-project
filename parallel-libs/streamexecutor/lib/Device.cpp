//===-- Device.cpp - Device implementation --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of Device class internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/Device.h"

#include <cassert>

#include "streamexecutor/PlatformDevice.h"
#include "streamexecutor/Stream.h"

#include "llvm/ADT/STLExtras.h"

namespace streamexecutor {

Device::Device(PlatformDevice *PDevice) : PDevice(PDevice) {}

Device::~Device() = default;

Expected<Stream> Device::createStream() {
  Expected<const void *> MaybePlatformStream = PDevice->createStream();
  if (!MaybePlatformStream)
    return MaybePlatformStream.takeError();
  return Stream(PDevice, *MaybePlatformStream);
}

} // namespace streamexecutor
