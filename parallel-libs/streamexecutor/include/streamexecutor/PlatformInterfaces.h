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

class PlatformExecutor;

/// Methods supported by device kernel function objects on all platforms.
class KernelInterface {
  // TODO(jhen): Add methods.
};

/// Platform-specific stream handle.
class PlatformStreamHandle {
public:
  explicit PlatformStreamHandle(PlatformExecutor *PExecutor)
      : PExecutor(PExecutor) {}

  virtual ~PlatformStreamHandle();

  PlatformExecutor *getExecutor() { return PExecutor; }

private:
  PlatformExecutor *PExecutor;
};

/// Raw executor methods that must be implemented by each platform.
///
/// This class defines the platform interface that supports executing work on a
/// device.
///
/// The public Executor and Stream classes have the type-safe versions of the
/// functions in this interface.
class PlatformExecutor {
public:
  virtual ~PlatformExecutor();

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
  virtual Error memcpyD2H(PlatformStreamHandle *S,
                          const GlobalDeviceMemoryBase &DeviceSrc,
                          void *HostDst, size_t ByteCount) {
    return make_error("memcpyD2H not implemented for platform " + getName());
  }

  /// Copies data from the host to the device.
  virtual Error memcpyH2D(PlatformStreamHandle *S, const void *HostSrc,
                          GlobalDeviceMemoryBase *DeviceDst, size_t ByteCount) {
    return make_error("memcpyH2D not implemented for platform " + getName());
  }

  /// Copies data from one device location to another.
  virtual Error memcpyD2D(PlatformStreamHandle *S,
                          const GlobalDeviceMemoryBase &DeviceSrc,
                          GlobalDeviceMemoryBase *DeviceDst, size_t ByteCount) {
    return make_error("memcpyD2D not implemented for platform " + getName());
  }

  /// Blocks the host until the given stream completes all the work enqueued up
  /// to the point this function is called.
  virtual Error blockHostUntilDone(PlatformStreamHandle *S) {
    return make_error("blockHostUntilDone not implemented for platform " +
                      getName());
  }
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_PLATFORMINTERFACES_H
