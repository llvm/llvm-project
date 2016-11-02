//===-- HostPlatform.h - Host platform subclass -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the HostPlatform class.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMS_HOST_HOSTPLATFORM_H
#define STREAMEXECUTOR_PLATFORMS_HOST_HOSTPLATFORM_H

#include "HostPlatformDevice.h"
#include "streamexecutor/Device.h"
#include "streamexecutor/Platform.h"

#include "llvm/Support/Mutex.h"

namespace streamexecutor {
namespace host {

/// Platform that performs work on the host rather than offloading to an
/// accelerator.
class HostPlatform : public Platform {
public:
  size_t getDeviceCount() const override { return 1; }

  Expected<Device> getDevice(size_t DeviceIndex) override {
    if (DeviceIndex != 0) {
      return make_error(
          "Requested device index " + llvm::Twine(DeviceIndex) +
          " from host platform which only supports device index 0");
    }
    llvm::sys::ScopedLock Lock(Mutex);
    if (!ThePlatformDevice)
      ThePlatformDevice = llvm::make_unique<HostPlatformDevice>();
    return Device(ThePlatformDevice.get());
  }

private:
  llvm::sys::Mutex Mutex;
  std::unique_ptr<HostPlatformDevice> ThePlatformDevice;
};

} // namespace host
} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMS_HOST_HOSTPLATFORM_H
