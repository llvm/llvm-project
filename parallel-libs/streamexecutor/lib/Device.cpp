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

#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Stream.h"

#include "llvm/ADT/STLExtras.h"

namespace streamexecutor {

Device::Device(PlatformDevice *PDevice) : PDevice(PDevice) {}

Device::~Device() = default;

Expected<std::unique_ptr<Stream>> Device::createStream() {
  Expected<std::unique_ptr<PlatformStreamHandle>> MaybePlatformStream =
      PDevice->createStream();
  if (!MaybePlatformStream) {
    return MaybePlatformStream.takeError();
  }
  assert((*MaybePlatformStream)->getDevice() == PDevice &&
         "an executor created a stream with a different stored executor");
  return llvm::make_unique<Stream>(std::move(*MaybePlatformStream));
}

} // namespace streamexecutor
