//===- CGPassBuilderOptionTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/CGPassBuilderOption.h"
#include "llvm/Support/CommandLine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Round-trip "-global-isel=<v>" through the cl::opt parser into
// CGPassBuilderOption and check the optional<bool> reflects the user's choice.
// This guards against the enum->bool conversion that previously collapsed
// BOU_FALSE (=2) to `true`.

static void parseArgs(std::initializer_list<const char *> Args) {
  cl::ResetAllOptionOccurrences();
  SmallVector<const char *, 4> Argv;
  Argv.push_back("CGPassBuilderOptionTest");
  for (const char *A : Args)
    Argv.push_back(A);
  cl::ParseCommandLineOptions(Argv.size(), Argv.data());
}

TEST(CGPassBuilderOption, GlobalISelNotSpecified) {
  parseArgs({});
  auto Opt = getCGPassBuilderOption();
  EXPECT_FALSE(Opt.EnableGlobalISelOption.has_value());
}

TEST(CGPassBuilderOption, GlobalISelEnabled) {
  parseArgs({"-global-isel=1"});
  auto Opt = getCGPassBuilderOption();
  ASSERT_TRUE(Opt.EnableGlobalISelOption.has_value());
  EXPECT_TRUE(*Opt.EnableGlobalISelOption);
}

TEST(CGPassBuilderOption, GlobalISelDisabled) {
  parseArgs({"-global-isel=0"});
  auto Opt = getCGPassBuilderOption();
  ASSERT_TRUE(Opt.EnableGlobalISelOption.has_value());
  EXPECT_FALSE(*Opt.EnableGlobalISelOption);
}

} // namespace
