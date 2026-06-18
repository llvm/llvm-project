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

ol_queue_handle_t olKGetDefaultQueue() {
  ol_queue_handle_t DefaultQueue = ThreadStateTy::getDefaultQueue();
  return DefaultQueue;
}

CallConfigurationTy* olKGetCallConfiguration() {
  return &ThreadStateTy::getCallConfiguration();
}

void olKRegisterKernel(const void *ID, ol_symbol_handle_t Kernel) {
  StateTy::get().addKernel(ID, Kernel);
}

ol_symbol_handle_t olKGetKernel(const void *ID) {
  return StateTy::get().getKernel(ID);
}

void olKRegisterProgram(const void *ID, ol_program_handle_t Program) {
  StateTy::get().addProgram(ID, Program);
}

ol_program_handle_t olKGetProgram(const void *ID) {
  return StateTy::get().getProgram(ID);
}
///}
