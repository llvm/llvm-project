//===-- CUDAPlatformDevice.h - CUDAPlatformDevice class ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the CUDAPlatformDevice class.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMS_CUDA_CUDAPLATFORMDEVICE_H
#define STREAMEXECUTOR_PLATFORMS_CUDA_CUDAPLATFORMDEVICE_H

#include "streamexecutor/PlatformDevice.h"

namespace streamexecutor {
namespace cuda {

Error CUresultToError(int CUResult, const llvm::Twine &Message);

class CUDAPlatformDevice : public PlatformDevice {
public:
  static Expected<CUDAPlatformDevice> create(size_t DeviceIndex);

  CUDAPlatformDevice(const CUDAPlatformDevice &) = delete;
  CUDAPlatformDevice &operator=(const CUDAPlatformDevice &) = delete;

  CUDAPlatformDevice(CUDAPlatformDevice &&) noexcept;
  CUDAPlatformDevice &operator=(CUDAPlatformDevice &&) noexcept;

  ~CUDAPlatformDevice() override;

  std::string getName() const override;

  std::string getPlatformName() const override { return "CUDA"; }

  Expected<const void *>
  createKernel(const MultiKernelLoaderSpec &Spec) override;
  Error destroyKernel(const void *Handle) override;

  Expected<const void *> createStream() override;
  Error destroyStream(const void *Handle) override;

  Error launch(const void *PlatformStreamHandle, BlockDimensions BlockSize,
               GridDimensions GridSize, const void *PKernelHandle,
               const PackedKernelArgumentArrayBase &ArgumentArray) override;

  Error copyD2H(const void *PlatformStreamHandle, const void *DeviceSrcHandle,
                size_t SrcByteOffset, void *HostDst, size_t DstByteOffset,
                size_t ByteCount) override;

  Error copyH2D(const void *PlatformStreamHandle, const void *HostSrc,
                size_t SrcByteOffset, const void *DeviceDstHandle,
                size_t DstByteOffset, size_t ByteCount) override;

  Error copyD2D(const void *PlatformStreamHandle, const void *DeviceSrcHandle,
                size_t SrcByteOffset, const void *DeviceDstHandle,
                size_t DstByteOffset, size_t ByteCount) override;

  Error blockHostUntilDone(const void *PlatformStreamHandle) override;

  Expected<void *> allocateDeviceMemory(size_t ByteCount) override;
  Error freeDeviceMemory(const void *Handle) override;

  Error registerHostMemory(void *Memory, size_t ByteCount) override;
  Error unregisterHostMemory(const void *Memory) override;

  Error synchronousCopyD2H(const void *DeviceSrcHandle, size_t SrcByteOffset,
                           void *HostDst, size_t DstByteOffset,
                           size_t ByteCount) override;

  Error synchronousCopyH2D(const void *HostSrc, size_t SrcByteOffset,
                           const void *DeviceDstHandle, size_t DstByteOffset,
                           size_t ByteCount) override;

  Error synchronousCopyD2D(const void *DeviceDstHandle, size_t DstByteOffset,
                           const void *DeviceSrcHandle, size_t SrcByteOffset,
                           size_t ByteCount) override;

private:
  CUDAPlatformDevice(size_t DeviceIndex) : DeviceIndex(DeviceIndex) {}

  int DeviceIndex;
};

} // namespace cuda
} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMS_CUDA_CUDAPLATFORMDEVICE_H
