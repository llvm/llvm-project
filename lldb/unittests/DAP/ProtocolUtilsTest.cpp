//===-- ProtocolUtilsTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProtocolUtils.h"
#include "JSONUtils.h"
#include "lldb/API/LLDB.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb;
using namespace lldb_dap;

TEST(ProtocolUtilsTest, CreateModule) {
  SBTarget target;
  SBModule module;

  std::optional<protocol::Module> module_opt = CreateModule(target, module);
  EXPECT_EQ(module_opt, std::nullopt);
}
