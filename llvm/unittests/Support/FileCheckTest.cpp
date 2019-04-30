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

TEST_F(FileCheckTest, ValidVarNameStart) {
  EXPECT_TRUE(FileCheckPattern::isValidVarNameStart('a'));
  EXPECT_TRUE(FileCheckPattern::isValidVarNameStart('G'));
  EXPECT_TRUE(FileCheckPattern::isValidVarNameStart('_'));
  EXPECT_FALSE(FileCheckPattern::isValidVarNameStart('2'));
  EXPECT_FALSE(FileCheckPattern::isValidVarNameStart('$'));
  EXPECT_FALSE(FileCheckPattern::isValidVarNameStart('@'));
  EXPECT_FALSE(FileCheckPattern::isValidVarNameStart('+'));
  EXPECT_FALSE(FileCheckPattern::isValidVarNameStart('-'));
  EXPECT_FALSE(FileCheckPattern::isValidVarNameStart(':'));
}

TEST_F(FileCheckTest, ParseVar) {
  StringRef VarName = "GoodVar42";
  bool IsPseudo = true;
  unsigned TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, VarName.size());

  VarName = "$GoodGlobalVar";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, VarName.size());

  VarName = "@GoodPseudoVar";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_TRUE(IsPseudo);
  EXPECT_EQ(TrailIdx, VarName.size());

  VarName = "42BadVar";
  EXPECT_TRUE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));

  VarName = "$@";
  EXPECT_TRUE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));

  VarName = "B@dVar";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, 1U);

  VarName = "B$dVar";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, 1U);

  VarName = "BadVar+";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, VarName.size() - 1);

  VarName = "BadVar-";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, VarName.size() - 1);

  VarName = "BadVar:";
  IsPseudo = true;
  TrailIdx = 0;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, IsPseudo, TrailIdx));
  EXPECT_FALSE(IsPseudo);
  EXPECT_EQ(TrailIdx, VarName.size() - 1);
}

class ExprTester {
private:
  SourceMgr SM;
  FileCheckPatternContext Context;
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Context);

public:
  bool parseExpect(std::string &VarName, std::string &Trailer) {
    std::string NameTrailer = VarName + Trailer;
    std::unique_ptr<MemoryBuffer> Buffer =
        MemoryBuffer::getMemBufferCopy(NameTrailer, "TestBuffer");
    StringRef NameTrailerRef = Buffer->getBuffer();
    SM.AddNewSourceBuffer(std::move(Buffer), SMLoc());
    StringRef VarNameRef = NameTrailerRef.substr(0, VarName.size());
    StringRef TrailerRef = NameTrailerRef.substr(VarName.size());
    return P.parseExpression(VarNameRef, TrailerRef, SM);
  }
};

TEST_F(FileCheckTest, ParseExpr) {
  ExprTester Tester;

  // @LINE with offset.
  std::string VarName = "@LINE";
  std::string Trailer = "+3";
  EXPECT_FALSE(Tester.parseExpect(VarName, Trailer));

  // @LINE only.
  Trailer = "";
  EXPECT_FALSE(Tester.parseExpect(VarName, Trailer));

  // Wrong Pseudovar.
  VarName = "@FOO";
  EXPECT_TRUE(Tester.parseExpect(VarName, Trailer));

  // Unsupported operator.
  VarName = "@LINE";
  Trailer = "/2";
  EXPECT_TRUE(Tester.parseExpect(VarName, Trailer));

  // Missing offset operand.
  VarName = "@LINE";
  Trailer = "+";
  EXPECT_TRUE(Tester.parseExpect(VarName, Trailer));

  // Cannot parse offset operand.
  VarName = "@LINE";
  Trailer = "+x";
  EXPECT_TRUE(Tester.parseExpect(VarName, Trailer));

  // Unexpected string at end of numeric expression.
  VarName = "@LINE";
  Trailer = "+5x";
  EXPECT_TRUE(Tester.parseExpect(VarName, Trailer));
}

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt = FileCheckPatternContext();
  std::vector<std::string> GlobalDefines;
  SourceMgr SM;

  // Missing equal sign
  GlobalDefines.emplace_back(std::string("LocalVar"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Empty variable
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("=18"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Invalid variable
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("18LocalVar=18"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Define local variables from command-line.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("LocalVar=FOO"));
  GlobalDefines.emplace_back(std::string("EmptyVar="));
  bool GotError = Cxt.defineCmdlineVariables(GlobalDefines, SM);
  EXPECT_FALSE(GotError);

  // Check defined variables are present and undefined is absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef EmptyVarStr = "EmptyVar";
  StringRef UnknownVarStr = "UnknownVar";
  llvm::Optional<StringRef> LocalVar = Cxt.getVarValue(LocalVarStr);
  llvm::Optional<StringRef> EmptyVar = Cxt.getVarValue(EmptyVarStr);
  llvm::Optional<StringRef> UnknownVar = Cxt.getVarValue(UnknownVarStr);
  EXPECT_TRUE(LocalVar);
  EXPECT_EQ(*LocalVar, "FOO");
  EXPECT_TRUE(EmptyVar);
  EXPECT_EQ(*EmptyVar, "");
  EXPECT_FALSE(UnknownVar);

  // Clear local variables and check they become absent.
  Cxt.clearLocalVars();
  LocalVar = Cxt.getVarValue(LocalVarStr);
  EXPECT_FALSE(LocalVar);
  EmptyVar = Cxt.getVarValue(EmptyVarStr);
  EXPECT_FALSE(EmptyVar);

  // Redefine global variables and check variables are defined again.
  GlobalDefines.emplace_back(std::string("$GlobalVar=BAR"));
  GotError = Cxt.defineCmdlineVariables(GlobalDefines, SM);
  EXPECT_FALSE(GotError);
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
