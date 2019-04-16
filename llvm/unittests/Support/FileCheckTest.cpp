//===- llvm/unittest/Support/FileCheckTest.cpp - FileCheck tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileCheck.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace {

class FileCheckTest : public ::testing::Test {};

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt;
  std::vector<std::string> GlobalDefines;

  // Define local and global variables from command-line.
  GlobalDefines.emplace_back(std::string("LocalVar=FOO"));
  Cxt.defineCmdlineVariables(GlobalDefines);

  // Check defined variables are present and undefined is absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef UnknownVarStr = "UnknownVar";
  llvm::Optional<StringRef> LocalVar = Cxt.getVarValue(LocalVarStr);
  llvm::Optional<StringRef> UnknownVar = Cxt.getVarValue(UnknownVarStr);
  EXPECT_TRUE(LocalVar);
  EXPECT_EQ(*LocalVar, "FOO");
  EXPECT_FALSE(UnknownVar);

  // Clear local variables and check they become absent.
  Cxt.clearLocalVars();
  LocalVar = Cxt.getVarValue(LocalVarStr);
  EXPECT_FALSE(LocalVar);

  // Redefine global variables and check variables are defined again.
  GlobalDefines.emplace_back(std::string("$GlobalVar=BAR"));
  Cxt.defineCmdlineVariables(GlobalDefines);
  StringRef GlobalVarStr = "$GlobalVar";
  llvm::Optional<StringRef> GlobalVar = Cxt.getVarValue(GlobalVarStr);
  EXPECT_TRUE(GlobalVar);
  EXPECT_EQ(*GlobalVar, "BAR");

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  GlobalVar = Cxt.getVarValue(GlobalVarStr);
  EXPECT_TRUE(GlobalVar);
}
} // namespace
