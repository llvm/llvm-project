//===------- Offload API tests - gtest environment ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <gtest/gtest.h>
#include <offload_api.h>

namespace TestEnvironment {
const std::vector<offload_platform_handle_t> &getPlatforms();
offload_platform_handle_t getPlatform();
} // namespace TestEnvironment
