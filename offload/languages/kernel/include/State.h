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
#include "Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

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

  /// Return the default queue for the current host thread
  static ol_queue_handle_t getDefaultQueue();

  /// Return the thread-local default device, or the first discovered device.
  static ol_device_handle_t getDefaultDevice();

  /// Return the thread-local default device and write its number to \p
  /// DeviceNo.
  static ol_device_handle_t getDevice(int *DeviceNo);

  /// Set the thread-local default device by device number.
  ///
  /// \returns the selected device, or nullptr if \p DeviceNo is invalid.
  static ol_device_handle_t setDefaultDevice(int DeviceNo);

  /// Return the last language-runtime error code for this thread.
  static uint32_t getLastError();

  /// Set the last language-runtime error code for this thread.
  static uint32_t setLastError(uint32_t Error);

  /// Return the pending kernel launch configuration for this thread.
  static CallConfigurationTy &getCallConfiguration();

  /// Set the thread-local default device to \p Device and recreate its queue.
  static void setDefaultDevice(ol_device_handle_t Device);

private:
  static ThreadStateTy &get();

  void createDefaultQueue(ol_device_handle_t Device);

  int DefaultDevice = 0;
  uint32_t LastError = 0;
  ol_queue_handle_t DefaultQueue = nullptr;

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

  /// Return the host device discovered during runtime initialization.
  static ol_device_handle_t getHostDevice();

  /// Return the shared context that owns the discovered non-host devices.
  static ol_context_handle_t getContext();

  /// Return the number of non-host devices available to kernel languages.
  static int getDeviceCount();

  /// Register \p Kernel for the host-side kernel identifier \p ID.
  ///
  /// \p ID is the opaque kernel key emitted by Clang in the offload entry
  /// table.  It is later passed to the launch entry point to recover the
  /// corresponding liboffload symbol handle.
  static void registerKernel(const void *ID, ol_symbol_handle_t Kernel);

  /// Remove any registered kernel handle for the host-side kernel key \p ID.
  static void unregisterKernel(const void *ID);

  /// Return the registered kernel handle for the host-side kernel key \p ID.
  static ol_symbol_handle_t getKernel(const void *ID);

  /// Register \p Program for the binary image identifier \p ID.
  ///
  /// \p ID is the device image start address from the offload binary
  /// descriptor.  It keys the loaded program so later function registration
  /// can look up the program that owns each kernel symbol.
  static void registerProgram(const void *ID, ol_program_handle_t Program);

  /// Remove and return the loaded program handle for binary image key \p ID.
  static ol_program_handle_t unregisterProgram(const void *ID);

  /// Return the loaded program handle for binary image key \p ID.
  static ol_program_handle_t getProgram(const void *ID);

private:
  static StateTy &get();
  static StateTy *tryGet();
  static bool addDevices(ol_device_handle_t Device, void *Payload);

  llvm::ArrayRef<ol_device_handle_t> getDevices() const;

  void addDevice(ol_device_handle_t Device);
  void setHostDevice(ol_device_handle_t Device);

  void addKernel(KernelIDTy KernelID, ol_symbol_handle_t Kernel);
  void removeKernel(KernelIDTy KernelID);
  ol_symbol_handle_t lookupKernel(KernelIDTy KernelID);

  void addProgram(const void *Binary, ol_program_handle_t Program);
  ol_program_handle_t removeProgram(const void *Binary);
  ol_program_handle_t lookupProgram(const void *Binary);

  void destroyRegisteredPrograms();

  llvm::DenseMap<const void *, ol_program_handle_t> BinaryRegisterMap;
  llvm::DenseMap<KernelIDTy, ol_symbol_handle_t> KernelMap;
  llvm::SmallVector<ol_device_handle_t, 8> Devices;

  ol_context_handle_t Context = nullptr;
  ol_queue_handle_t DefaultQueue = nullptr;
  ol_device_handle_t HostDevice = nullptr;

  StateTy();
};

} // namespace offload
} // namespace llvm

#endif // LLVM_OFFLOAD_LANGUAGES_KERNEL_INCLUDE_STATE_H
