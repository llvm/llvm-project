//===- RPC.h - Interface for remote procedure calls from the GPU ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the interface to support remote procedure calls (RPC) from
// the GPU. This is required to implement host services like printf or malloc.
// The interface to the RPC server is provided by the 'libc' project in LLVM.
// For more information visit https://libc.llvm.org/gpu/.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_RPC_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_RPC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace llvm::omp::target {
namespace plugin {
struct GenericPluginTy;
struct GenericDeviceTy;
class GenericGlobalHandlerTy;
class DeviceImageTy;
} // namespace plugin

/// A generic class implementing the interface between the RPC server provided
/// by the 'libc' project and 'libomptarget'. If the RPC server is not availible
/// these routines will perform no action.
struct RPCServerTy {
public:
  /// Initializes the handles to the number of devices we may need to service.
  RPCServerTy(plugin::GenericPluginTy &Plugin);

  /// Check if this device image is using an RPC server. This checks for the
  /// precense of an externally visible symbol in the device image that will
  /// be present whenever RPC code is called.
  llvm::Expected<bool> isDeviceUsingRPC(plugin::GenericDeviceTy &Device,
                                        plugin::GenericGlobalHandlerTy &Handler,
                                        plugin::DeviceImageTy &Image);

  /// Initialize the RPC server for the given device. This will allocate host
  /// memory for the internal server and copy the data to the client on the
  /// device. The device must be loaded before this is valid.
  llvm::Error initDevice(plugin::GenericDeviceTy &Device,
                         plugin::GenericGlobalHandlerTy &Handler,
                         plugin::DeviceImageTy &Image);

  /// Runs the RPC server associated with the \p Device until the pending work
  /// is cleared.
  llvm::Error runServer(plugin::GenericDeviceTy &Device);

  /// Deinitialize the RPC server for the given device. This will free the
  /// memory associated with the k
  llvm::Error deinitDevice(plugin::GenericDeviceTy &Device);

private:
  /// Array from this device's identifier to its attached devices.
  llvm::SmallVector<uintptr_t> Handles;
};

} // namespace llvm::omp::target

#endif
