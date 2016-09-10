//===-- PlatformDevice.h - PlatformDevice class -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declaration of the PlatformDevice class.
///
/// Each specific platform such as CUDA or OpenCL must subclass PlatformDevice
/// and override streamexecutor::Platform::getDevice to return an instance of
/// their PlatformDevice subclass.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMDEVICE_H
#define STREAMEXECUTOR_PLATFORMDEVICE_H

#include "streamexecutor/DeviceMemory.h"
#include "streamexecutor/Error.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/LaunchDimensions.h"
#include "streamexecutor/PackedKernelArgumentArray.h"

namespace streamexecutor {

/// Raw executor methods that must be implemented by each platform.
///
/// The public Device and Stream classes have the type-safe versions of the
/// functions in this interface.
class PlatformDevice {
public:
  virtual ~PlatformDevice();

  virtual std::string getName() const = 0;

  /// Creates a platform-specific kernel.
  virtual Expected<const void *>
  createKernel(const MultiKernelLoaderSpec &Spec) {
    return make_error("createKernel not implemented for platform " + getName());
  }

  virtual Error destroyKernel(const void *Handle) {
    return make_error("destroyKernel not implemented for platform " +
                      getName());
  }

  /// Creates a platform-specific stream.
  virtual Expected<const void *> createStream() {
    return make_error("createStream not implemented for platform " + getName());
  }

  virtual Error destroyStream(const void *Handle) {
    return make_error("destroyStream not implemented for platform " +
                      getName());
  }

  /// Launches a kernel on the given stream.
  virtual Error launch(const void *PlatformStreamHandle,
                       BlockDimensions BlockSize, GridDimensions GridSize,
                       const void *PKernelHandle,
                       const PackedKernelArgumentArrayBase &ArgumentArray) {
    return make_error("launch not implemented for platform " + getName());
  }

  /// Copies data from the device to the host.
  ///
  /// HostDst should have been allocated by allocateHostMemory or registered
  /// with registerHostMemory.
  virtual Error copyD2H(const void *PlatformStreamHandle,
                        const void *DeviceSrcHandle, size_t SrcByteOffset,
                        void *HostDst, size_t DstByteOffset, size_t ByteCount) {
    return make_error("copyD2H not implemented for platform " + getName());
  }

  /// Copies data from the host to the device.
  ///
  /// HostSrc should have been allocated by allocateHostMemory or registered
  /// with registerHostMemory.
  virtual Error copyH2D(const void *PlatformStreamHandle, const void *HostSrc,
                        size_t SrcByteOffset, const void *DeviceDstHandle,
                        size_t DstByteOffset, size_t ByteCount) {
    return make_error("copyH2D not implemented for platform " + getName());
  }

  /// Copies data from one device location to another.
  virtual Error copyD2D(const void *PlatformStreamHandle,
                        const void *DeviceSrcHandle, size_t SrcByteOffset,
                        const void *DeviceDstHandle, size_t DstByteOffset,
                        size_t ByteCount) {
    return make_error("copyD2D not implemented for platform " + getName());
  }

  /// Blocks the host until the given stream completes all the work enqueued up
  /// to the point this function is called.
  virtual Error blockHostUntilDone(const void *PlatformStreamHandle) {
    return make_error("blockHostUntilDone not implemented for platform " +
                      getName());
  }

  /// Allocates untyped device memory of a given size in bytes.
  virtual Expected<void *> allocateDeviceMemory(size_t ByteCount) {
    return make_error("allocateDeviceMemory not implemented for platform " +
                      getName());
  }

  /// Frees device memory previously allocated by allocateDeviceMemory.
  virtual Error freeDeviceMemory(const void *Handle) {
    return make_error("freeDeviceMemory not implemented for platform " +
                      getName());
  }

  /// Allocates untyped host memory of a given size in bytes.
  ///
  /// Host memory allocated via this method is suitable for use with copyH2D and
  /// copyD2H.
  virtual Expected<void *> allocateHostMemory(size_t ByteCount) {
    return make_error("allocateHostMemory not implemented for platform " +
                      getName());
  }

  /// Frees host memory allocated by allocateHostMemory.
  virtual Error freeHostMemory(void *Memory) {
    return make_error("freeHostMemory not implemented for platform " +
                      getName());
  }

  /// Registers previously allocated host memory so it can be used with copyH2D
  /// and copyD2H.
  virtual Error registerHostMemory(void *Memory, size_t ByteCount) {
    return make_error("registerHostMemory not implemented for platform " +
                      getName());
  }

  /// Unregisters host memory previously registered with registerHostMemory.
  virtual Error unregisterHostMemory(void *Memory) {
    return make_error("unregisterHostMemory not implemented for platform " +
                      getName());
  }

  /// Copies the given number of bytes from device memory to host memory.
  ///
  /// Blocks the calling host thread until the copy is completed. Can operate on
  /// any host memory, not just registered host memory or host memory allocated
  /// by allocateHostMemory. Does not block any ongoing device calls.
  virtual Error synchronousCopyD2H(const void *DeviceSrcHandle,
                                   size_t SrcByteOffset, void *HostDst,
                                   size_t DstByteOffset, size_t ByteCount) {
    return make_error("synchronousCopyD2H not implemented for platform " +
                      getName());
  }

  /// Similar to synchronousCopyD2H(const void *, size_t, void
  /// *, size_t, size_t), but copies memory from host to device rather than
  /// device to host.
  virtual Error synchronousCopyH2D(const void *HostSrc, size_t SrcByteOffset,
                                   const void *DeviceDstHandle,
                                   size_t DstByteOffset, size_t ByteCount) {
    return make_error("synchronousCopyH2D not implemented for platform " +
                      getName());
  }

  /// Similar to synchronousCopyD2H(const void *, size_t, void
  /// *, size_t, size_t), but copies memory from one location in device memory
  /// to another rather than from device to host.
  virtual Error synchronousCopyD2D(const void *DeviceDstHandle,
                                   size_t DstByteOffset,
                                   const void *DeviceSrcHandle,
                                   size_t SrcByteOffset, size_t ByteCount) {
    return make_error("synchronousCopyD2D not implemented for platform " +
                      getName());
  }
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMDEVICE_H
