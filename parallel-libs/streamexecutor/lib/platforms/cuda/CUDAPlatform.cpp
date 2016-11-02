//===-- CUDAPlatform.cpp - CUDA platform implementation -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of CUDA platform internals.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/platforms/cuda/CUDAPlatform.h"
#include "streamexecutor/Device.h"
#include "streamexecutor/Platform.h"
#include "streamexecutor/platforms/cuda/CUDAPlatformDevice.h"

#include "llvm/Support/Mutex.h"

#include "cuda.h"

#include <map>

namespace streamexecutor {
namespace cuda {

static CUresult ensureCUDAInitialized() {
  static CUresult InitResult = []() { return cuInit(0); }();
  return InitResult;
}

size_t CUDAPlatform::getDeviceCount() const {
  if (ensureCUDAInitialized())
    // TODO(jhen): Log an error.
    return 0;

  int DeviceCount = 0;
  CUresult Result = cuDeviceGetCount(&DeviceCount);
  (void)Result;
  // TODO(jhen): Log an error.

  return DeviceCount;
}

Expected<Device> CUDAPlatform::getDevice(size_t DeviceIndex) {
  if (CUresult InitResult = ensureCUDAInitialized())
    return CUresultToError(InitResult, "cached cuInit return value");

  llvm::sys::ScopedLock Lock(Mutex);
  auto Iterator = PlatformDevices.find(DeviceIndex);
  if (Iterator == PlatformDevices.end()) {
    if (auto MaybePDevice = CUDAPlatformDevice::create(DeviceIndex)) {
      Iterator =
          PlatformDevices.emplace(DeviceIndex, std::move(*MaybePDevice)).first;
    } else {
      return MaybePDevice.takeError();
    }
  }
  return Device(&Iterator->second);
}

} // namespace cuda
} // namespace streamexecutor
