/*===---- RuntimeAPI.h - Kernel language runtime internals ----------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#pragma once

#include "OffloadAPI.h"
#include "Types.h"

namespace llvm {
namespace offload {
namespace kernel {

ol_device_handle_t getDefaultDevice();

ol_device_handle_t getHostDevice();

int getDeviceCount();

ol_device_handle_t getDevice(int *DeviceNo);

ol_device_handle_t setDefaultDevice(int DeviceNo);

ol_queue_handle_t getDefaultQueue();

CallConfigurationTy *getCallConfiguration();

void registerKernel(const void *ID, ol_symbol_handle_t Kernel);

void unregisterKernel(const void *ID);

ol_symbol_handle_t getKernel(const void *ID);

void registerProgram(const void *ID, ol_program_handle_t Program);

ol_program_handle_t unregisterProgram(const void *ID);

ol_program_handle_t getProgram(const void *ID);

} // namespace kernel
} // namespace offload
} // namespace llvm
