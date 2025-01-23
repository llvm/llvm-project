//===------- Offload API tests - gtest environment ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <OffloadAPI.h>
#include <gtest/gtest.h>

namespace TestEnvironment {
const std::vector<ol_platform_handle_t> &getPlatforms();
ol_platform_handle_t getPlatform();
} // namespace TestEnvironment
