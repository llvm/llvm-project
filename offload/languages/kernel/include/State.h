//===------- State.h - Kernel Language (CUDA/HIP) persistent state --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "OffloadAPI.h"
#include "Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
namespace offload {

using KernelIDTy = const void *;

struct ThreadStateTy {
  ~ThreadStateTy();

  static ThreadStateTy &get();

  static ol_queue_handle_t getDefaultQueue();
  static ol_device_handle_t getDefaultDevice();
  static CallConfigurationTy &getCallConfiguration();
  void setDefaultDevice(ol_device_handle_t Device);

private:
  void createDefaultQueue(ol_device_handle_t Device);

  ol_device_handle_t DefaultDevice = nullptr;
  ol_queue_handle_t DefaultQueue = nullptr;

  CallConfigurationTy CC = {};

  ThreadStateTy();
};

struct StateTy {
  ~StateTy();

  friend struct ThreadStateTy;

  static StateTy &get();
  static StateTy *tryGet();

  static ol_device_handle_t getHostDevice() { return get().HostDevice; }

  ArrayRef<ol_device_handle_t> getDevices() const { return Devices; }

  void addDevice(ol_device_handle_t Device) { Devices.push_back(Device); }
  void setHostDevice(ol_device_handle_t Device) {
    if (!HostDevice)
      HostDevice = Device;
  }

  void addKernel(KernelIDTy KernelID, ol_symbol_handle_t Kernel) {
    KernelMap[KernelID] = Kernel;
  }

  void removeKernel(KernelIDTy KernelID) { KernelMap.erase(KernelID); }

  ol_symbol_handle_t getKernel(KernelIDTy KernelID) {
    return KernelMap[KernelID];
  }

  void addProgram(const void *Binary, ol_program_handle_t Program) {
    BinaryRegisterMap[Binary] = Program;
  }

  ol_program_handle_t removeProgram(const void *Binary) {
    auto It = BinaryRegisterMap.find(Binary);
    if (It == BinaryRegisterMap.end())
      return nullptr;
    ol_program_handle_t Program = It->second;
    BinaryRegisterMap.erase(It);
    return Program;
  }

  ol_program_handle_t getProgram(const void *Binary) {
    assert(BinaryRegisterMap.count(Binary));
    return BinaryRegisterMap[Binary];
  }

  void destroyRegisteredPrograms();

private:
  DenseMap<const void *, ol_program_handle_t> BinaryRegisterMap;
  DenseMap<KernelIDTy, ol_symbol_handle_t> KernelMap;
  SmallVector<ol_device_handle_t, 8> Devices;

  ol_queue_handle_t DefaultQueue = nullptr;
  ol_device_handle_t HostDevice = nullptr;

  StateTy();
};

} // namespace offload
} // namespace llvm
