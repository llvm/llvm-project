//===-- PlatformGDBStubLauncherTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/GDBStubLauncher/PlatformGDBStubLauncher.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using ConnectMode = PlatformGDBStubLauncher::ConnectMode;
using LaunchConfig = PlatformGDBStubLauncher::LaunchConfig;

TEST(PlatformGDBStubLauncherTest, ExpandCommandSubstitutesPort) {
  EXPECT_EQ(
      PlatformGDBStubLauncher::ExpandCommand(
          "emulator --gdb-port {port} --again {port}", 4242, "/path/to/a.out"),
      "emulator --gdb-port 4242 --again 4242");
}

TEST(PlatformGDBStubLauncherTest, ExpandCommandQuotesExecutable) {
  EXPECT_EQ(PlatformGDBStubLauncher::ExpandCommand(
                "emulator --image {executable}", 1, "/path with space/a.out"),
            "emulator --image '/path with space/a.out'");
  EXPECT_EQ(PlatformGDBStubLauncher::ExpandCommand("emulator {executable}", 1,
                                                   "/a$b/`c`|d*/it's"),
            "emulator '/a$b/`c`|d*/it'\\''s'");
  EXPECT_EQ(
      PlatformGDBStubLauncher::ExpandCommand("emulator {executable}", 1, ""),
      "emulator ''");
}

TEST(PlatformGDBStubLauncherTest, ExpandCommandLeavesOtherBraces) {
  EXPECT_EQ(PlatformGDBStubLauncher::ExpandCommand("echo {} {x} ${HOME}", 1,
                                                   "/path/to/a.out"),
            "echo {} {x} ${HOME}");
}

TEST(PlatformGDBStubLauncherTest, CheckConfigNeedsACommand) {
  EXPECT_THAT_ERROR(
      PlatformGDBStubLauncher::CheckConfig(LaunchConfig()),
      llvm::FailedWithMessage(testing::HasSubstr("missing settings: "
                                                 "launch-command (")));

  LaunchConfig config;
  config.command = "emulator --gdb-port {port}";
  EXPECT_THAT_ERROR(PlatformGDBStubLauncher::CheckConfig(config),
                    llvm::Succeeded());
}

TEST(PlatformGDBStubLauncherTest, CheckConfigRemotePlatformNeedsURLAndTarget) {
  LaunchConfig config;
  config.command = "vm";
  config.connect = ConnectMode::RemotePlatform;
  EXPECT_THAT_ERROR(
      PlatformGDBStubLauncher::CheckConfig(config),
      llvm::FailedWithMessage(testing::HasSubstr("connect-url, attach")));

  config.connect_url = "connect://127.0.0.1:4000";
  config.attach = "inferior";
  EXPECT_THAT_ERROR(PlatformGDBStubLauncher::CheckConfig(config),
                    llvm::Succeeded());
}

TEST(PlatformGDBStubLauncherTest, CheckConfigReadyLogAndPatternGoTogether) {
  LaunchConfig config;
  config.command = "vm";
  config.ready_log = "boot.log";
  EXPECT_THAT_ERROR(
      PlatformGDBStubLauncher::CheckConfig(config),
      llvm::FailedWithMessage(testing::HasSubstr("ready-pattern")));

  config.ready_log.clear();
  config.ready_pattern = "ready";
  EXPECT_THAT_ERROR(PlatformGDBStubLauncher::CheckConfig(config),
                    llvm::FailedWithMessage(testing::HasSubstr("ready-log")));

  config.ready_log = "boot.log";
  EXPECT_THAT_ERROR(PlatformGDBStubLauncher::CheckConfig(config),
                    llvm::Succeeded());
}
