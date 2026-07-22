/*===---- ExportedAPI.h - Kernel language runtime - exported api  ----------===
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

extern "C" {
ol_device_handle_t olKGetDefaultDevice();

ol_device_handle_t olKGetHostDevice();

int olKGetDeviceCount();

ol_device_handle_t olKGetDevice(int *DeviceNo);

ol_device_handle_t olKSetDefaultDevice(int DeviceNo);

ol_queue_handle_t olKGetDefaultQueue();

CallConfigurationTy *olKGetCallConfiguration();

void olKRegisterKernel(const void *ID, ol_symbol_handle_t Kernel);

void olKUnregisterKernel(const void *ID);

ol_symbol_handle_t olKGetKernel(const void *ID);

void olKRegisterProgram(const void *ID, ol_program_handle_t Program);

ol_program_handle_t olKUnregisterProgram(const void *ID);

ol_program_handle_t olKGetProgram(const void *ID);
}
