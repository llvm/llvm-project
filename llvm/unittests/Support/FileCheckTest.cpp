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

TEST_F(FileCheckTest, NumericVariable) {
  FileCheckNumericVariable FooVar = FileCheckNumericVariable("FOO", 42);
  EXPECT_EQ("FOO", FooVar.getName());

  // Defined variable: getValue returns a value, setValue fails and value
  // remains unchanged.
  llvm::Optional<uint64_t> Value = FooVar.getValue();
  EXPECT_TRUE(Value);
  EXPECT_EQ(42U, *Value);
  EXPECT_TRUE(FooVar.setValue(43));
  Value = FooVar.getValue();
  EXPECT_TRUE(Value);
  EXPECT_EQ(42U, *Value);

  // Clearing variable: getValue fails, clearValue again fails.
  EXPECT_FALSE(FooVar.clearValue());
  Value = FooVar.getValue();
  EXPECT_FALSE(Value);
  EXPECT_TRUE(FooVar.clearValue());

  // Undefined variable: setValue works, getValue returns value set.
  EXPECT_FALSE(FooVar.setValue(43));
  Value = FooVar.getValue();
  EXPECT_TRUE(Value);
  EXPECT_EQ(43U, *Value);
}

uint64_t doAdd(uint64_t OpL, uint64_t OpR) { return OpL + OpR; }

TEST_F(FileCheckTest, NumExpr) {
  FileCheckNumericVariable FooVar = FileCheckNumericVariable("FOO", 42);
  FileCheckNumExpr NumExpr = FileCheckNumExpr(doAdd, &FooVar, 18);

  // Defined variable: eval returns right value, no undefined variable
  // returned.
  llvm::Optional<uint64_t> Value = NumExpr.eval();
  EXPECT_TRUE(Value);
  EXPECT_EQ(60U, *Value);
  StringRef UndefVar = NumExpr.getUndefVarName();
  EXPECT_EQ("", UndefVar);

  // Undefined variable: eval fails, undefined variable returned. We call
  // getUndefVarName first to check that it can be called without calling
  // eval() first.
  FooVar.clearValue();
  UndefVar = NumExpr.getUndefVarName();
  EXPECT_EQ("FOO", UndefVar);
  Value = NumExpr.eval();
  EXPECT_FALSE(Value);
}

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

static StringRef bufferize(SourceMgr &SM, StringRef Str) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(Str, "TestBuffer");
  StringRef StrBufferRef = Buffer->getBuffer();
  SM.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  return StrBufferRef;
}

class ExprTester {
private:
  SourceMgr SM;
  FileCheckRequest Req;
  FileCheckPatternContext Context;
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Context);

public:
  ExprTester() {
    std::vector<std::string> GlobalDefines;
    GlobalDefines.emplace_back(std::string("#FOO=42"));
    Context.defineCmdlineVariables(GlobalDefines, SM);
    // Call ParsePattern to have @LINE defined.
    P.ParsePattern("N/A", "CHECK", SM, 1, Req);
  }

  bool parseExpect(std::string &VarName, std::string &Trailer) {
    bool IsPseudo = VarName[0] == '@';
    std::string NameTrailer = VarName + Trailer;
    StringRef NameTrailerRef = bufferize(SM, NameTrailer);
    StringRef VarNameRef = NameTrailerRef.substr(0, VarName.size());
    StringRef TrailerRef = NameTrailerRef.substr(VarName.size());
    return P.parseNumericExpression(VarNameRef, IsPseudo, TrailerRef, SM) ==
           nullptr;
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

  // Defined variable.
  VarName = "FOO";
  EXPECT_FALSE(Tester.parseExpect(VarName, Trailer));

  // Undefined variable.
  VarName = "UNDEF";
  EXPECT_TRUE(Tester.parseExpect(VarName, Trailer));

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

TEST_F(FileCheckTest, Substitution) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  Context.defineCmdlineVariables(GlobalDefines, SM);

  // Substitution of undefined pattern variable fails.
  FileCheckPatternSubstitution PatternSubstitution =
      FileCheckPatternSubstitution(&Context, "VAR404", 42);
  EXPECT_FALSE(PatternSubstitution.getResult());

  // Substitutions of defined pseudo and non-pseudo numeric variables return
  // the right value.
  FileCheckNumericVariable LineVar = FileCheckNumericVariable("@LINE", 42);
  FileCheckNumericVariable NVar = FileCheckNumericVariable("@N", 10);
  FileCheckNumExpr NumExprLine = FileCheckNumExpr(doAdd, &LineVar, 0);
  FileCheckNumExpr NumExprN = FileCheckNumExpr(doAdd, &NVar, 3);
  FileCheckPatternSubstitution SubstitutionLine =
      FileCheckPatternSubstitution(&Context, "@LINE", &NumExprLine, 12);
  FileCheckPatternSubstitution SubstitutionN =
      FileCheckPatternSubstitution(&Context, "N", &NumExprN, 30);
  llvm::Optional<std::string> Value = SubstitutionLine.getResult();
  EXPECT_TRUE(Value);
  EXPECT_EQ("42", *Value);
  Value = SubstitutionN.getResult();
  EXPECT_TRUE(Value);
  EXPECT_EQ("13", *Value);

  // Substitution of undefined numeric variable fails.
  LineVar.clearValue();
  EXPECT_FALSE(SubstitutionLine.getResult());
  NVar.clearValue();
  EXPECT_FALSE(SubstitutionN.getResult());

  // Substitution of defined pattern variable returns the right value.
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Context);
  PatternSubstitution = FileCheckPatternSubstitution(&Context, "FOO", 42);
  Value = PatternSubstitution.getResult();
  EXPECT_TRUE(Value);
  EXPECT_EQ("BAR", *Value);
}

TEST_F(FileCheckTest, UndefVars) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  Context.defineCmdlineVariables(GlobalDefines, SM);

  // getUndefVarName() on a pattern variable substitution with an undefined
  // variable returns that variable.
  FileCheckPatternSubstitution Substitution =
      FileCheckPatternSubstitution(&Context, "VAR404", 42);
  StringRef UndefVar = Substitution.getUndefVarName();
  EXPECT_EQ("VAR404", UndefVar);

  // getUndefVarName() on a pattern variable substitution with a defined
  // variable returns an empty string.
  Substitution = FileCheckPatternSubstitution(&Context, "FOO", 42);
  UndefVar = Substitution.getUndefVarName();
  EXPECT_EQ("", UndefVar);

  // getUndefVarName() on a numeric expression substitution with a defined
  // variable returns an empty string.
  FileCheckNumericVariable LineVar = FileCheckNumericVariable("@LINE", 42);
  FileCheckNumExpr NumExpr = FileCheckNumExpr(doAdd, &LineVar, 0);
  Substitution = FileCheckPatternSubstitution(&Context, "@LINE", &NumExpr, 12);
  UndefVar = Substitution.getUndefVarName();
  EXPECT_EQ("", UndefVar);

  // getUndefVarName() on a numeric expression substitution with an undefined
  // variable returns that variable.
  LineVar.clearValue();
  UndefVar = Substitution.getUndefVarName();
  EXPECT_EQ("@LINE", UndefVar);
}

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt = FileCheckPatternContext();
  std::vector<std::string> GlobalDefines;
  SourceMgr SM;

  // Missing equal sign.
  GlobalDefines.emplace_back(std::string("LocalVar"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#LocalNumVar"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Empty variable name.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("=18"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#=18"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Invalid variable name.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("18LocalVar=18"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#18LocalNumVar=18"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Name conflict between pattern and numeric variable.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("LocalVar=18"));
  GlobalDefines.emplace_back(std::string("#LocalVar=36"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));
  Cxt = FileCheckPatternContext();
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#LocalNumVar=18"));
  GlobalDefines.emplace_back(std::string("LocalNumVar=36"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));
  Cxt = FileCheckPatternContext();

  // Invalid numeric value for numeric variable.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#LocalNumVar=x"));
  EXPECT_TRUE(Cxt.defineCmdlineVariables(GlobalDefines, SM));

  // Define local variables from command-line.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("LocalVar=FOO"));
  GlobalDefines.emplace_back(std::string("EmptyVar="));
  GlobalDefines.emplace_back(std::string("#LocalNumVar=18"));
  bool GotError = Cxt.defineCmdlineVariables(GlobalDefines, SM);
  EXPECT_FALSE(GotError);

  // Check defined variables are present and undefined is absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef LocalNumVarRef = bufferize(SM, "LocalNumVar");
  StringRef EmptyVarStr = "EmptyVar";
  StringRef UnknownVarStr = "UnknownVar";
  llvm::Optional<StringRef> LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Cxt);
  FileCheckNumExpr *NumExpr =
      P.parseNumericExpression(LocalNumVarRef, false /*IsPseudo*/, "", SM);
  llvm::Optional<StringRef> EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  llvm::Optional<StringRef> UnknownVar = Cxt.getPatternVarValue(UnknownVarStr);
  EXPECT_TRUE(LocalVar);
  EXPECT_EQ(*LocalVar, "FOO");
  EXPECT_TRUE(NumExpr);
  llvm::Optional<uint64_t> NumExprVal = NumExpr->eval();
  EXPECT_TRUE(NumExprVal);
  EXPECT_EQ(*NumExprVal, 18U);
  EXPECT_TRUE(EmptyVar);
  EXPECT_EQ(*EmptyVar, "");
  EXPECT_FALSE(UnknownVar);

  // Clear local variables and check they become absent.
  Cxt.clearLocalVars();
  LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  EXPECT_FALSE(LocalVar);
  // Check a numeric expression's evaluation fails if called after clearing of
  // local variables, if it was created before. This is important because local
  // variable clearing due to --enable-var-scope happens after numeric
  // expressions are linked to the numeric variables they use.
  EXPECT_FALSE(NumExpr->eval());
  P = FileCheckPattern(Check::CheckPlain, &Cxt);
  NumExpr =
      P.parseNumericExpression(LocalNumVarRef, false /*IsPseudo*/, "", SM);
  EXPECT_FALSE(NumExpr);
  EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  EXPECT_FALSE(EmptyVar);

  // Redefine global variables and check variables are defined again.
  GlobalDefines.emplace_back(std::string("$GlobalVar=BAR"));
  GlobalDefines.emplace_back(std::string("#$GlobalNumVar=36"));
  GotError = Cxt.defineCmdlineVariables(GlobalDefines, SM);
  EXPECT_FALSE(GotError);
  StringRef GlobalVarStr = "$GlobalVar";
  StringRef GlobalNumVarRef = bufferize(SM, "$GlobalNumVar");
  llvm::Optional<StringRef> GlobalVar = Cxt.getPatternVarValue(GlobalVarStr);
  EXPECT_TRUE(GlobalVar);
  EXPECT_EQ(*GlobalVar, "BAR");
  P = FileCheckPattern(Check::CheckPlain, &Cxt);
  NumExpr =
      P.parseNumericExpression(GlobalNumVarRef, false /*IsPseudo*/, "", SM);
  EXPECT_TRUE(NumExpr);
  NumExprVal = NumExpr->eval();
  EXPECT_TRUE(NumExprVal);
  EXPECT_EQ(*NumExprVal, 36U);

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  GlobalVar = Cxt.getPatternVarValue(GlobalVarStr);
  EXPECT_TRUE(GlobalVar);
  P = FileCheckPattern(Check::CheckPlain, &Cxt);
  NumExpr =
      P.parseNumericExpression(GlobalNumVarRef, false /*IsPseudo*/, "", SM);
  EXPECT_TRUE(NumExpr);
  NumExprVal = NumExpr->eval();
  EXPECT_TRUE(NumExprVal);
  EXPECT_EQ(*NumExprVal, 36U);
}
} // namespace
