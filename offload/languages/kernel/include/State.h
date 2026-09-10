//===-- State.h - Kernel language persistent state ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STATE_H
#define LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STATE_H

#include "OffloadAPI.h"
#include "Stream.h"
#include "Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <mutex>

#define CHECK_FATAL(ResultExpr, ...)                                           \
  do {                                                                         \
    ol_result_t CheckFatalResult = (ResultExpr);                               \
    if (CheckFatalResult && CheckFatalResult->Code) {                          \
      llvm::errs() << __VA_ARGS__;                                             \
      if (CheckFatalResult->Details)                                           \
        llvm::errs() << ": " << CheckFatalResult->Details;                     \
      llvm::errs() << '\n';                                                    \
      abort();                                                                 \
    }                                                                          \
  } while (false)

#define FATAL_UNIMPLEMENTED(...)                                               \
  do {                                                                         \
    llvm::errs() << __VA_ARGS__ << '\n';                                       \
    abort();                                                                   \
  } while (false)

namespace llvm {
namespace offload {

static constexpr unsigned AssumedDeviceCount = 8;
static constexpr unsigned AssumedStreamCount = 8;

/// Opaque host-side key used to identify a registered kernel.
///
/// This is the address emitted in the offload entry table for the kernel
using KernelIDTy = const void *;

/// Per-thread state used by the language runtime entry points.
///
/// Tracks the current thread's default device, optional per-thread queue,
/// last-error code, and pending kernel launch configuration.
struct ThreadStateTy {
  ~ThreadStateTy();

  /// Return the thread-local state for the current host thread.
  static ThreadStateTy &get();

  /// Return the default queue for the current host thread.
  ol_queue_handle_t getDefaultQueue();

  /// Return the default stream for the current host thread and device.
  StreamTy *getDefaultStream();

  /// Return the thread-local default device, or the first discovered device.
  ol_device_handle_t getDefaultDevice();

  /// Return the thread-local default device and write its number to \p
  /// DeviceNo.
  ol_device_handle_t getDevice(int *DeviceNo);

  /// Set the thread-local default device by device number.
  ///
  /// \returns the selected device, or nullptr if \p DeviceNo is invalid.
  ol_device_handle_t setDefaultDevice(int DeviceNo);

  /// Return the last language-runtime error code for this thread.
  uint32_t getLastError();

  /// Set the last language-runtime error code for this thread.
  uint32_t setLastError(uint32_t Error);

  /// Return the pending kernel launch configuration for this thread.
  CallConfigurationTy &getCallConfiguration();

private:
  StreamTy *getOrCreateDefaultStream(ol_device_handle_t Device);
  void destroyDefaultStreams();

  int DefaultDevice = 0;
  uint32_t LastError = 0;
  DenseMap<ol_device_handle_t, StreamTy *> PerThreadDeviceDefaultStreamMap;

  CallConfigurationTy CC = {};

  ThreadStateTy();
};

/// Process-wide state shared by CUDA and HIP language entry points.
///
/// Owns the discovered devices, host device, process default queue, and maps
/// from registered binaries and kernels to liboffload handles.
struct StateTy {
  ~StateTy();

  friend struct ThreadStateTy;

  /// Return the process-wide state singleton.
  static StateTy &get();

  /// Return the process-wide state singleton if it has been initialized.
  static StateTy *tryGet();

  /// Return the host device discovered during runtime initialization.
  ol_device_handle_t getHostDevice();

  /// Return the shared context that owns the discovered non-host devices.
  ol_context_handle_t getContext();

  /// Return the number of non-host devices available to kernel languages.
  int getDeviceCount();

  /// Register \p Kernel for the host-side kernel identifier \p ID.
  ///
  /// \p ID is the opaque kernel key emitted by Clang in the offload entry
  /// table.  It is later passed to the launch entry point to recover the
  /// corresponding liboffload symbol handle.
  void registerKernel(const void *ID, ol_symbol_handle_t Kernel);

  /// Remove any registered kernel handle for the host-side kernel key \p ID.
  void unregisterKernel(const void *ID);

  /// Return the registered kernel handle for the host-side kernel key \p ID.
  ol_symbol_handle_t getKernel(const void *ID);

  /// Register \p Program for the binary image identifier \p ID.
  ///
  /// \p ID is the device image start address from the offload binary
  /// descriptor.  It keys the loaded program so later function registration
  /// can look up the program that owns each kernel symbol.
  void registerProgram(const void *ID, ol_program_handle_t Program);

  /// Remove and return the loaded program handle for binary image key \p ID.
  ol_program_handle_t unregisterProgram(const void *ID);

  /// Return the loaded program handle for binary image key \p ID.
  ol_program_handle_t getProgram(const void *ID);

  /// Return all streams currently known for \p Device.
  SmallPtrSet<StreamTy *, 8> getDeviceStreams(ol_device_handle_t Device);

  /// Return all explicitly created blocking streams for \p Device.
  SmallPtrSet<StreamTy *, 8> getBlockingStreams(ol_device_handle_t Device);

  /// Return true if \p Device has an existing legacy default stream.
  bool hasLegacyDefaultStream(ol_device_handle_t Device);

  /// Create a stream for \p Device and register it with the process state.
  ol_result_t createStream(ol_device_handle_t Device, QueueKind Kind,
                           StreamTy **Stream);

  /// Destroy \p Stream after removing it from the process state.
  ol_result_t destroyStream(StreamTy *Stream);

  /// Return true if \p Stream is currently registered with the process state.
  bool isStreamRegistered(StreamTy *Stream);

private:
  static bool addDevices(ol_device_handle_t Device, void *Payload);

  ArrayRef<ol_device_handle_t> getDevices() const;

  void addDevice(ol_device_handle_t Device);
  void setHostDevice(ol_device_handle_t Device);

  StreamTy *getOrCreateDefaultStream(ol_device_handle_t Device);
  void destroyDefaultStreams();

  /// Inserts the Stream into the DeviceStreamsMap and the
  /// DeviceBlockingStreamsMap
  void addStream(StreamTy *Stream);
  /// Removes the Stream from DeviceStreamsMap and DeviceBlockingStreamsMap
  void removeStream(StreamTy *Stream);

  void destroyRegisteredStreams();
  void destroyRegisteredPrograms();

  DenseMap<const void *, ol_program_handle_t> BinaryRegisterMap;
  DenseMap<KernelIDTy, ol_symbol_handle_t> KernelMap;
  SmallVector<ol_device_handle_t, AssumedDeviceCount> Devices;
  DenseMap<ol_device_handle_t, StreamTy *> DeviceDefaultStreamsMap;
  DenseMap<ol_device_handle_t, SmallPtrSet<StreamTy *, AssumedStreamCount>>
      DeviceStreamsMap;
  DenseMap<ol_device_handle_t, SmallPtrSet<StreamTy *, AssumedStreamCount>>
      DeviceBlockingStreamsMap;

  ol_context_handle_t Context = nullptr;
  ol_device_handle_t HostDevice = nullptr;

  std::mutex DeviceDefaultStreamsMapLock;
  std::mutex DeviceStreamsMapLock;
  std::mutex DeviceBlockingStreamsMapLock;

  StateTy();
};

} // namespace offload
} // namespace llvm

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STATE_H
