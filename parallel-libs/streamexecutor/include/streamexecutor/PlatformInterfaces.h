//===-- PlatformInterfaces.h - Interfaces to platform impls -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Interfaces to platform-specific implementations.
///
/// The general pattern is that the functions in these interfaces take raw
/// handle types as parameters. This means that these types and functions are
/// not intended for public use. Instead, corresponding methods in public types
/// like Stream, StreamExecutor, and Kernel use C++ templates to create
/// type-safe public interfaces. Those public functions do the type-unsafe work
/// of extracting raw handles from their arguments and forwarding those handles
/// to the methods defined in this file in the proper format.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PLATFORMINTERFACES_H
#define STREAMEXECUTOR_PLATFORMINTERFACES_H

#include "streamexecutor/DeviceMemory.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/LaunchDimensions.h"
#include "streamexecutor/PackedKernelArgumentArray.h"
#include "streamexecutor/Utils/Error.h"

namespace streamexecutor {

class PlatformDevice;

/// Methods supported by device kernel function objects on all platforms.
class KernelInterface {
  // TODO(jhen): Add methods.
};

/// Platform-specific stream handle.
class PlatformStreamHandle {
public:
  explicit PlatformStreamHandle(PlatformDevice *PDevice) : PDevice(PDevice) {}

  virtual ~PlatformStreamHandle();

  PlatformDevice *getDevice() { return PDevice; }

private:
  PlatformDevice *PDevice;
};

/// Raw executor methods that must be implemented by each platform.
///
/// This class defines the platform interface that supports executing work on a
/// device.
///
/// The public Device and Stream classes have the type-safe versions of the
/// functions in this interface.
class PlatformDevice {
public:
  virtual ~PlatformDevice();

  virtual std::string getName() const = 0;

  /// Creates a platform-specific stream.
  virtual Expected<std::unique_ptr<PlatformStreamHandle>> createStream() = 0;

  /// Launches a kernel on the given stream.
  virtual Error launch(PlatformStreamHandle *S, BlockDimensions BlockSize,
                       GridDimensions GridSize, const KernelBase &Kernel,
                       const PackedKernelArgumentArrayBase &ArgumentArray) {
    return make_error("launch not implemented for platform " + getName());
  }

  /// Copies data from the device to the host.
  ///
  /// HostDst should have been allocated by allocateHostMemory or registered
  /// with registerHostMemory.
  virtual Error copyD2H(PlatformStreamHandle *S,
                        const GlobalDeviceMemoryBase &DeviceSrc,
                        size_t SrcByteOffset, void *HostDst,
                        size_t DstByteOffset, size_t ByteCount) {
    return make_error("copyD2H not implemented for platform " + getName());
  }

  /// Copies data from the host to the device.
  ///
  /// HostSrc should have been allocated by allocateHostMemory or registered
  /// with registerHostMemory.
  virtual Error copyH2D(PlatformStreamHandle *S, const void *HostSrc,
                        size_t SrcByteOffset, GlobalDeviceMemoryBase DeviceDst,
                        size_t DstByteOffset, size_t ByteCount) {
    return make_error("copyH2D not implemented for platform " + getName());
  }

  /// Copies data from one device location to another.
  virtual Error copyD2D(PlatformStreamHandle *S,
                        const GlobalDeviceMemoryBase &DeviceSrc,
                        size_t SrcByteOffset, GlobalDeviceMemoryBase DeviceDst,
                        size_t DstByteOffset, size_t ByteCount) {
    return make_error("copyD2D not implemented for platform " + getName());
  }

  /// Blocks the host until the given stream completes all the work enqueued up
  /// to the point this function is called.
  virtual Error blockHostUntilDone(PlatformStreamHandle *S) {
    return make_error("blockHostUntilDone not implemented for platform " +
                      getName());
  }

  /// Allocates untyped device memory of a given size in bytes.
  virtual Expected<GlobalDeviceMemoryBase>
  allocateDeviceMemory(size_t ByteCount) {
    return make_error("allocateDeviceMemory not implemented for platform " +
                      getName());
  }

  /// Frees device memory previously allocated by allocateDeviceMemory.
  virtual Error freeDeviceMemory(GlobalDeviceMemoryBase Memory) {
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
  virtual Error synchronousCopyD2H(const GlobalDeviceMemoryBase &DeviceSrc,
                                   size_t SrcByteOffset, void *HostDst,
                                   size_t DstByteOffset, size_t ByteCount) {
    return make_error("synchronousCopyD2H not implemented for platform " +
                      getName());
  }

  /// Similar to synchronousCopyD2H(const GlobalDeviceMemoryBase &, size_t, void
  /// *, size_t, size_t), but copies memory from host to device rather than
  /// device to host.
  virtual Error synchronousCopyH2D(const void *HostSrc, size_t SrcByteOffset,
                                   GlobalDeviceMemoryBase DeviceDst,
                                   size_t DstByteOffset, size_t ByteCount) {
    return make_error("synchronousCopyH2D not implemented for platform " +
                      getName());
  }

  /// Similar to synchronousCopyD2H(const GlobalDeviceMemoryBase &, size_t, void
  /// *, size_t, size_t), but copies memory from one location in device memory
  /// to another rather than from device to host.
  virtual Error synchronousCopyD2D(GlobalDeviceMemoryBase DeviceDst,
                                   size_t DstByteOffset,
                                   const GlobalDeviceMemoryBase &DeviceSrc,
                                   size_t SrcByteOffset, size_t ByteCount) {
    return make_error("synchronousCopyD2D not implemented for platform " +
                      getName());
  }
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMINTERFACES_H
