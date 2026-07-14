//===------ ExportedAPI.cpp - Kernel Language runtime - exported api ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "ExportedAPI.h"

#include "State.h"
#include "Types.h"

#include "OffloadAPI.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdio>
#include <stdint.h>

using namespace llvm;
using namespace offload;

/// Runtime API
///{
ol_device_handle_t olKGetDefaultDevice() {
  ol_device_handle_t DefaultDevice = ThreadStateTy::getDefaultDevice();
  return DefaultDevice;
}

ol_device_handle_t olKGetHostDevice() {
  ol_device_handle_t HostDevice = StateTy::getHostDevice();
  return HostDevice;
}

int olKGetDeviceCount() {
  int DeviceCount = StateTy::get().getDevices().size();
  return DeviceCount;
}

ol_device_handle_t olKGetDevice(int *DeviceNo) {
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

ol_device_handle_t olKSetDefaultDevice(int DeviceNo) {
  ArrayRef<ol_device_handle_t> Devices = StateTy::get().getDevices();
  if (DeviceNo < 0 || DeviceNo >= static_cast<int>(Devices.size()))
    return nullptr;
  ol_device_handle_t Device = Devices[DeviceNo];
  ThreadStateTy::get().setDefaultDevice(Device);
  return Device;
}

ol_queue_handle_t olKGetDefaultQueue() {
  ol_queue_handle_t DefaultQueue = ThreadStateTy::getDefaultQueue();
  return DefaultQueue;
}

CallConfigurationTy *olKGetCallConfiguration() {
  return &ThreadStateTy::getCallConfiguration();
}

void olKRegisterKernel(const void *ID, ol_symbol_handle_t Kernel) {
  StateTy::get().addKernel(ID, Kernel);
}

void olKUnregisterKernel(const void *ID) {
  if (StateTy *State = StateTy::tryGet())
    State->removeKernel(ID);
}

ol_symbol_handle_t olKGetKernel(const void *ID) {
  return StateTy::get().getKernel(ID);
}

void olKRegisterProgram(const void *ID, ol_program_handle_t Program) {
  StateTy::get().addProgram(ID, Program);
}

ol_program_handle_t olKUnregisterProgram(const void *ID) {
  if (StateTy *State = StateTy::tryGet())
    return State->removeProgram(ID);
  return nullptr;
}

ol_program_handle_t olKGetProgram(const void *ID) {
  return StateTy::get().getProgram(ID);
}
///}
