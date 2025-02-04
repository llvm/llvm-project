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

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace llvm::omp::target {
namespace plugin {
struct GenericPluginTy;
struct GenericDeviceTy;
class GenericGlobalHandlerTy;
class DeviceImageTy;
} // namespace plugin

/// A generic class implementing the interface between the RPC server provided
/// by the 'libc' project and 'libomptarget'. If the RPC server is not available
/// these routines will perform no action.
struct RPCServerTy {
public:
  /// Initializes the handles to the number of devices we may need to service.
  RPCServerTy(plugin::GenericPluginTy &Plugin);

  /// Deinitialize the associated memory and resources.
  llvm::Error shutDown();

  /// Initialize the worker thread.
  llvm::Error startThread();

  /// Check if this device image is using an RPC server. This checks for the
  /// presence of an externally visible symbol in the device image that will
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

  /// Deinitialize the RPC server for the given device. This will free the
  /// memory associated with the k
  llvm::Error deinitDevice(plugin::GenericDeviceTy &Device);

private:
  /// Array from this device's identifier to its attached devices.
  std::unique_ptr<void *[]> Buffers;

  /// Array of associated devices. These must be alive as long as the server is.
  std::unique_ptr<plugin::GenericDeviceTy *[]> Devices;

  /// A helper class for running the user thread that handles the RPC interface.
  /// Because we only need to check the RPC server while any kernels are
  /// working, we track submission / completion events to allow the thread to
  /// sleep when it is not needed.
  struct ServerThread {
    std::thread Worker;

    /// A boolean indicating whether or not the worker thread should continue.
    std::atomic<bool> Running;

    /// The number of currently executing kernels across all devices that need
    /// the server thread to be running.
    std::atomic<uint32_t> NumUsers;

    /// The condition variable used to suspend the thread if no work is needed.
    std::condition_variable CV;
    std::mutex Mutex;

    /// A reference to all the RPC interfaces that the server is handling.
    llvm::ArrayRef<void *> Buffers;

    /// A reference to the associated generic device for the buffer.
    llvm::ArrayRef<plugin::GenericDeviceTy *> Devices;

    /// Initialize the worker thread to run in the background.
    ServerThread(void *Buffers[], plugin::GenericDeviceTy *Devices[],
                 size_t Length)
        : Running(false), NumUsers(0), CV(), Mutex(), Buffers(Buffers, Length),
          Devices(Devices, Length) {}

    ~ServerThread() { assert(!Running && "Thread not shut down explicitly\n"); }

    /// Notify the worker thread that there is a user that needs it.
    void notify() {
      std::lock_guard<decltype(Mutex)> Lock(Mutex);
      NumUsers.fetch_add(1, std::memory_order_relaxed);
      CV.notify_all();
    }

    /// Indicate that one of the dependent users has finished.
    void finish() {
      [[maybe_unused]] uint32_t Old =
          NumUsers.fetch_sub(1, std::memory_order_relaxed);
      assert(Old > 0 && "Attempt to signal finish with no pending work");
    }

    /// Destroy the worker thread and wait.
    void shutDown();

    /// Initialize the worker thread.
    void startThread();

    /// Run the server thread to continuously check the RPC interface for work
    /// to be done for the device.
    void run();
  };

public:
  /// Pointer to the server thread instance.
  std::unique_ptr<ServerThread> Thread;
};

} // namespace llvm::omp::target

#endif
