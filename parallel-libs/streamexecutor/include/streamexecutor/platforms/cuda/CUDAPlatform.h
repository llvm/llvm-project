//===-- CUDAPlatform.h - CUDA platform subclass -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the CUDAPlatform class.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMS_CUDA_CUDAPLATFORM_H
#define STREAMEXECUTOR_PLATFORMS_CUDA_CUDAPLATFORM_H

#include "streamexecutor/Platform.h"
#include "streamexecutor/platforms/cuda/CUDAPlatformDevice.h"

#include "llvm/Support/Mutex.h"

#include <map>

namespace streamexecutor {
namespace cuda {

class CUDAPlatform : public Platform {
public:
  size_t getDeviceCount() const override;

  Expected<Device> getDevice(size_t DeviceIndex) override;

private:
  llvm::sys::Mutex Mutex;
  std::map<size_t, CUDAPlatformDevice> PlatformDevices;
};

} // namespace cuda
} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMS_CUDA_CUDAPLATFORM_H
