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

TEST(ClientLauncherTest, URLEncode) {
  EXPECT_EQ("", VSCodeLauncher::URLEncode(""));
  EXPECT_EQ(
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.~",
      VSCodeLauncher::URLEncode("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST"
                                "UVWXYZ0123456789-_.~"));
  EXPECT_EQ("hello%20world", VSCodeLauncher::URLEncode("hello world"));
  EXPECT_EQ("hello%21%40%23%24", VSCodeLauncher::URLEncode("hello!@#$"));
  EXPECT_EQ("%2Fpath%2Fto%2Ffile", VSCodeLauncher::URLEncode("/path/to/file"));
  EXPECT_EQ("key%3Dvalue%26key2%3Dvalue2",
            VSCodeLauncher::URLEncode("key=value&key2=value2"));
  EXPECT_EQ("100%25complete", VSCodeLauncher::URLEncode("100%complete"));
  EXPECT_EQ("file_name%20with%20spaces%20%26%20special%21.txt",
            VSCodeLauncher::URLEncode("file_name with spaces & special!.txt"));
  EXPECT_EQ("%00%01%02",
            VSCodeLauncher::URLEncode(llvm::StringRef("\x00\x01\x02", 3)));
  EXPECT_EQ("test-file_name.txt~",
            VSCodeLauncher::URLEncode("test-file_name.txt~"));

  // UTF-8 encoded characters should be percent-encoded byte by byte.
  EXPECT_EQ("%C3%A9", VSCodeLauncher::URLEncode("é"));
}
