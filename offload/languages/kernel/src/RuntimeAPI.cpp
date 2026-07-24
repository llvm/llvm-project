//===------ RuntimeAPI.cpp - Kernel language runtime internals ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "RuntimeAPI.h"

#include "State.h"
#include "Types.h"

#include "OffloadAPI.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdio>
#include <stdint.h>

namespace llvm {
namespace offload {
namespace kernel {

ol_device_handle_t getDefaultDevice() {
  ol_device_handle_t DefaultDevice = ThreadStateTy::getDefaultDevice();
  return DefaultDevice;
}

ol_device_handle_t getHostDevice() {
  ol_device_handle_t HostDevice = StateTy::getHostDevice();
  return HostDevice;
}

int getDeviceCount() {
  int DeviceCount = StateTy::get().getDevices().size();
  return DeviceCount;
}

ol_device_handle_t getDevice(int *DeviceNo) {
  ol_device_handle_t DefaultDevice = ThreadStateTy::getDefaultDevice();
  int DeviceCount = StateTy::get().getDevices().size();
  ArrayRef<ol_device_handle_t> Devices = StateTy::get().getDevices();
  for (int i = 0; i < DeviceCount; i++) {
    if (Devices[i] == DefaultDevice) {
      *DeviceNo = i;
      return Devices[i];
    }
  }
  return nullptr;
}

ol_device_handle_t setDefaultDevice(int DeviceNo) {
  ArrayRef<ol_device_handle_t> Devices = StateTy::get().getDevices();
  if (DeviceNo < 0 || DeviceNo >= static_cast<int>(Devices.size()))
    return nullptr;
  ol_device_handle_t Device = Devices[DeviceNo];
  ThreadStateTy::get().setDefaultDevice(Device);
  return Device;
}

ol_queue_handle_t getDefaultQueue() {
  ol_queue_handle_t DefaultQueue = ThreadStateTy::getDefaultQueue();
  return DefaultQueue;
}

CallConfigurationTy *getCallConfiguration() {
  return &ThreadStateTy::getCallConfiguration();
}

void registerKernel(const void *ID, ol_symbol_handle_t Kernel) {
  StateTy::get().addKernel(ID, Kernel);
}

void unregisterKernel(const void *ID) {
  if (StateTy *State = StateTy::tryGet())
    State->removeKernel(ID);
}

ol_symbol_handle_t getKernel(const void *ID) {
  return StateTy::get().getKernel(ID);
}

void registerProgram(const void *ID, ol_program_handle_t Program) {
  StateTy::get().addProgram(ID, Program);
}

ol_program_handle_t unregisterProgram(const void *ID) {
  if (StateTy *State = StateTy::tryGet())
    return State->removeProgram(ID);
  return nullptr;
}

ol_program_handle_t getProgram(const void *ID) {
  return StateTy::get().getProgram(ID);
}

} // namespace kernel
} // namespace offload
} // namespace llvm
