//===------- Offload API tests - gtest environment ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/MemoryBuffer.h"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

namespace TestEnvironment {

struct Device {
  ol_device_handle_t Handle;
  std::string Name;
};

const std::vector<Device> &getDevices();
ol_device_handle_t getHostDevice();
bool loadDeviceBinary(const std::string &BinaryName, ol_device_handle_t Device,
                      std::unique_ptr<llvm::MemoryBuffer> &BinaryOut);
} // namespace TestEnvironment
