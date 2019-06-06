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
  // Undefined variable: getValue and clearValue fails, setValue works.
  FileCheckNumericVariable FooVar = FileCheckNumericVariable(1, "FOO");
  EXPECT_EQ("FOO", FooVar.getName());
  llvm::Optional<uint64_t> Value = FooVar.getValue();
  EXPECT_FALSE(Value);
  EXPECT_TRUE(FooVar.clearValue());
  EXPECT_FALSE(FooVar.setValue(42));

  // Defined variable: getValue returns value set, setValue fails.
  Value = FooVar.getValue();
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
  StringRef OrigVarName = "GoodVar42";
  StringRef VarName = OrigVarName;
  StringRef ParsedName;
  bool IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(ParsedName, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(IsPseudo);

  VarName = OrigVarName = "$GoodGlobalVar";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(ParsedName, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(IsPseudo);

  VarName = OrigVarName = "@GoodPseudoVar";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(ParsedName, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_TRUE(IsPseudo);

  VarName = "42BadVar";
  EXPECT_TRUE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));

  VarName = "$@";
  EXPECT_TRUE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));

  VarName = OrigVarName = "B@dVar";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedName, "B");
  EXPECT_FALSE(IsPseudo);

  VarName = OrigVarName = "B$dVar";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedName, "B");
  EXPECT_FALSE(IsPseudo);

  VarName = "BadVar+";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(VarName, "+");
  EXPECT_EQ(ParsedName, "BadVar");
  EXPECT_FALSE(IsPseudo);

  VarName = "BadVar-";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(VarName, "-");
  EXPECT_EQ(ParsedName, "BadVar");
  EXPECT_FALSE(IsPseudo);

  VarName = "BadVar:";
  IsPseudo = true;
  EXPECT_FALSE(FileCheckPattern::parseVariable(VarName, ParsedName, IsPseudo));
  EXPECT_EQ(VarName, ":");
  EXPECT_EQ(ParsedName, "BadVar");
  EXPECT_FALSE(IsPseudo);
}

static StringRef bufferize(SourceMgr &SM, StringRef Str) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(Str, "TestBuffer");
  StringRef StrBufferRef = Buffer->getBuffer();
  SM.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  return StrBufferRef;
}

class PatternTester {
private:
  size_t LineNumber = 1;
  SourceMgr SM;
  FileCheckRequest Req;
  FileCheckPatternContext Context;
  FileCheckPattern P =
      FileCheckPattern(Check::CheckPlain, &Context, LineNumber++);

public:
  PatternTester() {
    std::vector<std::string> GlobalDefines;
    GlobalDefines.emplace_back(std::string("#FOO=42"));
    GlobalDefines.emplace_back(std::string("BAR=BAZ"));
    Context.defineCmdlineVariables(GlobalDefines, SM);
    // Call parsePattern to have @LINE defined.
    P.parsePattern("N/A", "CHECK", SM, Req);
    // parsePattern does not expect to be called twice for the same line and
    // will set FixedStr and RegExStr incorrectly if it is. Therefore prepare
    // a pattern for a different line.
    initNextPattern();
  }

  void initNextPattern() {
    P = FileCheckPattern(Check::CheckPlain, &Context, LineNumber++);
  }

  bool parseNumVarDefExpect(StringRef Expr) {
    StringRef ExprBufferRef = bufferize(SM, Expr);
    StringRef Name;
    return FileCheckPattern::parseNumericVariableDefinition(ExprBufferRef, Name,
                                                            &Context, SM);
  }

  bool parseSubstExpect(StringRef Expr) {
    StringRef ExprBufferRef = bufferize(SM, Expr);
    FileCheckNumericVariable *DefinedNumericVariable;
    return P.parseNumericSubstitutionBlock(
               ExprBufferRef, DefinedNumericVariable, SM) == nullptr;
  }

  bool parsePatternExpect(StringRef Pattern) {
    StringRef PatBufferRef = bufferize(SM, Pattern);
    return P.parsePattern(PatBufferRef, "CHECK", SM, Req);
  }

  bool matchExpect(StringRef Buffer) {
    StringRef BufferRef = bufferize(SM, Buffer);
    size_t MatchLen;
    return P.match(BufferRef, MatchLen, SM);
  }
};

TEST_F(FileCheckTest, ParseNumericVariableDefinition) {
  PatternTester Tester;

  // Invalid definition of pseudo.
  EXPECT_TRUE(Tester.parseNumVarDefExpect("@LINE"));

  // Conflict with pattern variable.
  EXPECT_TRUE(Tester.parseNumVarDefExpect("BAR"));

  // Defined variable.
  EXPECT_FALSE(Tester.parseNumVarDefExpect("FOO"));
}

TEST_F(FileCheckTest, ParseExpr) {
  PatternTester Tester;

  // Variable definition.

  // Definition of invalid variable.
  EXPECT_TRUE(Tester.parseSubstExpect("10VAR:"));
  EXPECT_TRUE(Tester.parseSubstExpect("@FOO:"));
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE:"));

  // Garbage after name of variable being defined.
  EXPECT_TRUE(Tester.parseSubstExpect("VAR GARBAGE:"));

  // Variable defined to numeric expression.
  EXPECT_TRUE(Tester.parseSubstExpect("VAR1: FOO"));

  // Acceptable variable definition.
  EXPECT_FALSE(Tester.parseSubstExpect("VAR1:"));
  EXPECT_FALSE(Tester.parseSubstExpect("  VAR2:"));
  EXPECT_FALSE(Tester.parseSubstExpect("VAR3  :"));
  EXPECT_FALSE(Tester.parseSubstExpect("VAR3:  "));

  // Numeric expression.

  // Unacceptable variable.
  EXPECT_TRUE(Tester.parseSubstExpect("10VAR"));
  EXPECT_TRUE(Tester.parseSubstExpect("@FOO"));
  EXPECT_TRUE(Tester.parseSubstExpect("UNDEF"));

  // Only valid variable.
  EXPECT_FALSE(Tester.parseSubstExpect("@LINE"));
  EXPECT_FALSE(Tester.parseSubstExpect("FOO"));

  // Use variable defined on same line.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#LINE1VAR:]]"));
  EXPECT_TRUE(Tester.parseSubstExpect("LINE1VAR"));

  // Unsupported operator.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE/2"));

  // Missing offset operand.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+"));

  // Cannot parse offset operand.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+x"));

  // Unexpected string at end of numeric expression.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+5x"));

  // Valid expression.
  EXPECT_FALSE(Tester.parseSubstExpect("@LINE+5"));
  EXPECT_FALSE(Tester.parseSubstExpect("FOO+4"));
}

TEST_F(FileCheckTest, ParsePattern) {
  PatternTester Tester;

  // Space in pattern variable expression.
  EXPECT_TRUE(Tester.parsePatternExpect("[[ BAR]]"));

  // Invalid variable name.
  EXPECT_TRUE(Tester.parsePatternExpect("[[42INVALID]]"));

  // Invalid pattern variable definition.
  EXPECT_TRUE(Tester.parsePatternExpect("[[@PAT:]]"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[PAT+2:]]"));

  // Collision with numeric variable.
  EXPECT_TRUE(Tester.parsePatternExpect("[[FOO:]]"));

  // Valid use of pattern variable.
  EXPECT_FALSE(Tester.parsePatternExpect("[[BAR]]"));

  // Valid pattern variable definition.
  EXPECT_FALSE(Tester.parsePatternExpect("[[PAT:[0-9]+]]"));

  // Invalid numeric expressions.
  EXPECT_TRUE(Tester.parsePatternExpect("[[#42INVALID]]"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[#@FOO]]"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[#@LINE/2]]"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[#2+@LINE]]"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[#YUP:@LINE]]"));

  // Valid numeric expressions and numeric variable definition.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#FOO]]"));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE+2]]"));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#NUMVAR:]]"));
}

TEST_F(FileCheckTest, Match) {
  PatternTester Tester;

  // Check matching a definition only matches a number.
  Tester.parsePatternExpect("[[#NUMVAR:]]");
  EXPECT_TRUE(Tester.matchExpect("FAIL"));
  EXPECT_FALSE(Tester.matchExpect("18"));

  // Check matching the variable defined matches the correct number only
  Tester.initNextPattern();
  Tester.parsePatternExpect("[[#NUMVAR]] [[#NUMVAR+2]]");
  EXPECT_TRUE(Tester.matchExpect("19 21"));
  EXPECT_TRUE(Tester.matchExpect("18 21"));
  EXPECT_FALSE(Tester.matchExpect("18 20"));
}

TEST_F(FileCheckTest, Substitution) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  Context.defineCmdlineVariables(GlobalDefines, SM);

  // Substitution of an undefined string variable fails.
  FileCheckStringSubstitution StringSubstitution =
      FileCheckStringSubstitution(&Context, "VAR404", 42);
  EXPECT_FALSE(StringSubstitution.getResult());

  // Substitutions of defined pseudo and non-pseudo numeric variables return
  // the right value.
  FileCheckNumericVariable LineVar = FileCheckNumericVariable("@LINE", 42);
  FileCheckNumericVariable NVar = FileCheckNumericVariable("N", 10);
  FileCheckNumExpr NumExprLine = FileCheckNumExpr(doAdd, &LineVar, 0);
  FileCheckNumExpr NumExprN = FileCheckNumExpr(doAdd, &NVar, 3);
  FileCheckNumericSubstitution SubstitutionLine =
      FileCheckNumericSubstitution(&Context, "@LINE", &NumExprLine, 12);
  FileCheckNumericSubstitution SubstitutionN =
      FileCheckNumericSubstitution(&Context, "N", &NumExprN, 30);
  llvm::Optional<std::string> Value = SubstitutionLine.getResult();
  EXPECT_TRUE(Value);
  EXPECT_EQ("42", *Value);
  Value = SubstitutionN.getResult();
  EXPECT_TRUE(Value);
  EXPECT_EQ("13", *Value);

  // Substitution of an undefined numeric variable fails.
  LineVar.clearValue();
  EXPECT_FALSE(SubstitutionLine.getResult());
  NVar.clearValue();
  EXPECT_FALSE(SubstitutionN.getResult());

  // Substitution of a defined string variable returns the right value.
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Context, 1);
  StringSubstitution = FileCheckStringSubstitution(&Context, "FOO", 42);
  Value = StringSubstitution.getResult();
  EXPECT_TRUE(Value);
  EXPECT_EQ("BAR", *Value);
}

TEST_F(FileCheckTest, UndefVars) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  Context.defineCmdlineVariables(GlobalDefines, SM);

  // getUndefVarName() on a string substitution with an undefined variable
  // returns that variable.
  FileCheckStringSubstitution StringSubstitution =
      FileCheckStringSubstitution(&Context, "VAR404", 42);
  StringRef UndefVar = StringSubstitution.getUndefVarName();
  EXPECT_EQ("VAR404", UndefVar);

  // getUndefVarName() on a string substitution with a defined variable returns
  // an empty string.
  StringSubstitution = FileCheckStringSubstitution(&Context, "FOO", 42);
  UndefVar = StringSubstitution.getUndefVarName();
  EXPECT_EQ("", UndefVar);

  // getUndefVarName() on a numeric substitution with a defined variable
  // returns an empty string.
  FileCheckNumericVariable LineVar = FileCheckNumericVariable("@LINE", 42);
  FileCheckNumExpr NumExpr = FileCheckNumExpr(doAdd, &LineVar, 0);
  FileCheckNumericSubstitution NumericSubstitution =
      FileCheckNumericSubstitution(&Context, "@LINE", &NumExpr, 12);
  UndefVar = NumericSubstitution.getUndefVarName();
  EXPECT_EQ("", UndefVar);

  // getUndefVarName() on a numeric substitution with an undefined variable
  // returns that variable.
  LineVar.clearValue();
  UndefVar = NumericSubstitution.getUndefVarName();
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
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Cxt, 1);
  FileCheckNumericVariable *DefinedNumericVariable;
  FileCheckNumExpr *NumExpr = P.parseNumericSubstitutionBlock(
      LocalNumVarRef, DefinedNumericVariable, SM);
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
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 2);
  NumExpr = P.parseNumericSubstitutionBlock(LocalNumVarRef,
                                            DefinedNumericVariable, SM);
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
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 3);
  NumExpr = P.parseNumericSubstitutionBlock(GlobalNumVarRef,
                                            DefinedNumericVariable, SM);
  EXPECT_TRUE(NumExpr);
  NumExprVal = NumExpr->eval();
  EXPECT_TRUE(NumExprVal);
  EXPECT_EQ(*NumExprVal, 36U);

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  GlobalVar = Cxt.getPatternVarValue(GlobalVarStr);
  EXPECT_TRUE(GlobalVar);
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 4);
  NumExpr = P.parseNumericSubstitutionBlock(GlobalNumVarRef,
                                            DefinedNumericVariable, SM);
  EXPECT_TRUE(NumExpr);
  NumExprVal = NumExpr->eval();
  EXPECT_TRUE(NumExprVal);
  EXPECT_EQ(*NumExprVal, 36U);
}
} // namespace
