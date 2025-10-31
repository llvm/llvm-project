//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClientLauncher.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"
#include <optional>

using namespace lldb_dap;
using namespace llvm;

TEST(ClientLauncherTest, GetClientFromVSCode) {
  std::optional<ClientLauncher::Client> result =
      ClientLauncher::GetClientFrom("vscode");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(ClientLauncher::VSCode, result.value());
}

TEST(ClientLauncherTest, GetClientFromVSCodeUpperCase) {
  std::optional<ClientLauncher::Client> result =
      ClientLauncher::GetClientFrom("VSCODE");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(ClientLauncher::VSCode, result.value());
}

TEST(ClientLauncherTest, GetClientFromVSCodeMixedCase) {
  std::optional<ClientLauncher::Client> result =
      ClientLauncher::GetClientFrom("VSCode");
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(ClientLauncher::VSCode, result.value());
}

TEST(ClientLauncherTest, GetClientFromInvalidString) {
  std::optional<ClientLauncher::Client> result =
      ClientLauncher::GetClientFrom("invalid");
  EXPECT_FALSE(result.has_value());
}

TEST(ClientLauncherTest, GetClientFromEmptyString) {
  std::optional<ClientLauncher::Client> result =
      ClientLauncher::GetClientFrom("");
  EXPECT_FALSE(result.has_value());
}
