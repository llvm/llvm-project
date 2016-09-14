//===-- HostPlatformDevice.h - HostPlatformDevice class ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the HostPlatformDevice class.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMS_HOST_HOSTPLATFORMDEVICE_H
#define STREAMEXECUTOR_PLATFORMS_HOST_HOSTPLATFORMDEVICE_H

#include <cstdlib>
#include <cstring>

#include "streamexecutor/PlatformDevice.h"

namespace streamexecutor {
namespace host {

/// A concrete PlatformDevice subclass that performs its work on the host rather
/// than offloading to an accelerator.
class HostPlatformDevice : public PlatformDevice {
public:
  std::string getName() const override { return "host"; }

  Expected<const void *>
  createKernel(const MultiKernelLoaderSpec &Spec) override {
    if (!Spec.hasHostFunction()) {
      return make_error("no host implementation available for kernel " +
                        Spec.getKernelName());
    }
    return static_cast<const void *>(&Spec.getHostFunction());
  }

  Error destroyKernel(const void *Handle) override { return Error::success(); }

  Expected<const void *> createStream() override {
    // TODO(jhen): Do something with threads to allow multiple streams.
    return this;
  }

  Error destroyStream(const void *Handle) override { return Error::success(); }

  Error launch(const void *PlatformStreamHandle, BlockDimensions BlockSize,
               GridDimensions GridSize, const void *PKernelHandle,
               const PackedKernelArgumentArrayBase &ArgumentArray) override {
    // TODO(jhen): Can we do something with BlockSize and GridSize?
    if (!(BlockSize.X == 1 && BlockSize.Y == 1 && BlockSize.Z == 1)) {
      return make_error(
          "Block dimensions were (" + llvm::Twine(BlockSize.X) + "," +
          llvm::Twine(BlockSize.Y) + "," + llvm::Twine(BlockSize.Z) +
          "), but only size (1,1,1) is permitted for this platform");
    }
    if (!(GridSize.X == 1 && GridSize.Y == 1 && GridSize.Z == 1)) {
      return make_error(
          "Grid dimensions were (" + llvm::Twine(GridSize.X) + "," +
          llvm::Twine(GridSize.Y) + "," + llvm::Twine(GridSize.Z) +
          "), but only size (1,1,1) is permitted for this platform");
    }

    (*static_cast<const std::function<void(const void *const *)> *>(
        PKernelHandle))(ArgumentArray.getAddresses());
    return Error::success();
  }

  Error copyD2H(const void *PlatformStreamHandle, const void *DeviceSrcHandle,
                size_t SrcByteOffset, void *HostDst, size_t DstByteOffset,
                size_t ByteCount) override {
    std::memcpy(offset(HostDst, DstByteOffset),
                offset(DeviceSrcHandle, SrcByteOffset), ByteCount);
    return Error::success();
  }

  Error copyH2D(const void *PlatformStreamHandle, const void *HostSrc,
                size_t SrcByteOffset, const void *DeviceDstHandle,
                size_t DstByteOffset, size_t ByteCount) override {
    std::memcpy(offset(DeviceDstHandle, DstByteOffset),
                offset(HostSrc, SrcByteOffset), ByteCount);
    return Error::success();
  }

  Error copyD2D(const void *PlatformStreamHandle, const void *DeviceSrcHandle,
                size_t SrcByteOffset, const void *DeviceDstHandle,
                size_t DstByteOffset, size_t ByteCount) override {
    std::memcpy(offset(DeviceDstHandle, DstByteOffset),
                offset(DeviceSrcHandle, SrcByteOffset), ByteCount);
    return Error::success();
  }

  Error blockHostUntilDone(const void *PlatformStreamHandle) override {
    // All host operations are synchronous anyway.
    return Error::success();
  }

  Expected<void *> allocateDeviceMemory(size_t ByteCount) override {
    return std::malloc(ByteCount);
  }

  Error freeDeviceMemory(const void *Handle) override {
    std::free(const_cast<void *>(Handle));
    return Error::success();
  }

  Error registerHostMemory(void *Memory, size_t ByteCount) override {
    return Error::success();
  }

  Error unregisterHostMemory(const void *Memory) override {
    return Error::success();
  }

  Error synchronousCopyD2H(const void *DeviceSrcHandle, size_t SrcByteOffset,
                           void *HostDst, size_t DstByteOffset,
                           size_t ByteCount) override {
    std::memcpy(offset(HostDst, DstByteOffset),
                offset(DeviceSrcHandle, SrcByteOffset), ByteCount);
    return Error::success();
  }

  Error synchronousCopyH2D(const void *HostSrc, size_t SrcByteOffset,
                           const void *DeviceDstHandle, size_t DstByteOffset,
                           size_t ByteCount) override {
    std::memcpy(offset(DeviceDstHandle, DstByteOffset),
                offset(HostSrc, SrcByteOffset), ByteCount);
    return Error::success();
  }

  Error synchronousCopyD2D(const void *DeviceSrcHandle, size_t SrcByteOffset,
                           const void *DeviceDstHandle, size_t DstByteOffset,
                           size_t ByteCount) override {
    std::memcpy(offset(DeviceDstHandle, DstByteOffset),
                offset(DeviceSrcHandle, SrcByteOffset), ByteCount);
    return Error::success();
  }

  /// Gets the value at the given index from a GlobalDeviceMemory<T> instance
  /// created by this class.
  template <typename T>
  static T getDeviceValue(const streamexecutor::GlobalDeviceMemory<T> &Memory,
                          size_t Index) {
    return static_cast<const T *>(Memory.getHandle())[Index];
  }

private:
  static void *offset(const void *Base, size_t Offset) {
    return const_cast<char *>(static_cast<const char *>(Base) + Offset);
  }
};

} // namespace host
} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMS_HOST_HOSTPLATFORMDEVICE_H
