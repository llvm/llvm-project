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
  // Undefined variable: getValue fails, setValue does not trigger assert.
  FileCheckNumericVariable FooVar = FileCheckNumericVariable(1, "FOO");
  EXPECT_EQ("FOO", FooVar.getName());
  llvm::Optional<uint64_t> Value = FooVar.getValue();
  EXPECT_FALSE(Value);
  FooVar.clearValue();
  FooVar.setValue(42);

  // Defined variable: getValue returns value set.
  Value = FooVar.getValue();
  EXPECT_TRUE(Value);
  EXPECT_EQ(42U, *Value);

  // Clearing variable: getValue fails.
  FooVar.clearValue();
  Value = FooVar.getValue();
  EXPECT_FALSE(Value);
}

uint64_t doAdd(uint64_t OpL, uint64_t OpR) { return OpL + OpR; }

static void expectUndefError(const Twine &ExpectedStr, Error Err) {
  handleAllErrors(std::move(Err), [&](const FileCheckUndefVarError &E) {
    EXPECT_EQ(ExpectedStr.str(), E.getVarName());
  });
}

TEST_F(FileCheckTest, Expression) {
  FileCheckNumericVariable FooVar = FileCheckNumericVariable("FOO", 42);
  FileCheckExpression Expression = FileCheckExpression(doAdd, &FooVar, 18);

  // Defined variable: eval returns right value.
  Expected<uint64_t> Value = Expression.eval();
  EXPECT_TRUE(bool(Value));
  EXPECT_EQ(60U, *Value);

  // Undefined variable: eval fails, undefined variable returned. We call
  // getUndefVarName first to check that it can be called without calling
  // eval() first.
  FooVar.clearValue();
  Error EvalError = Expression.eval().takeError();
  EXPECT_TRUE(errorToBool(std::move(EvalError)));
  expectUndefError("FOO", std::move(EvalError));
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

static StringRef bufferize(SourceMgr &SM, StringRef Str) {
  std::unique_ptr<MemoryBuffer> Buffer =
      MemoryBuffer::getMemBufferCopy(Str, "TestBuffer");
  StringRef StrBufferRef = Buffer->getBuffer();
  SM.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  return StrBufferRef;
}

TEST_F(FileCheckTest, ParseVar) {
  SourceMgr SM;
  StringRef OrigVarName = bufferize(SM, "GoodVar42");
  StringRef VarName = OrigVarName;
  bool IsPseudo = true;
  Expected<StringRef> ParsedName =
      FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(*ParsedName, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(IsPseudo);

  VarName = OrigVarName = bufferize(SM, "$GoodGlobalVar");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(*ParsedName, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(IsPseudo);

  VarName = OrigVarName = bufferize(SM, "@GoodPseudoVar");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(*ParsedName, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_TRUE(IsPseudo);

  VarName = bufferize(SM, "42BadVar");
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(errorToBool(ParsedName.takeError()));

  VarName = bufferize(SM, "$@");
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(errorToBool(ParsedName.takeError()));

  VarName = OrigVarName = bufferize(SM, "B@dVar");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(*ParsedName, "B");
  EXPECT_FALSE(IsPseudo);

  VarName = OrigVarName = bufferize(SM, "B$dVar");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(*ParsedName, "B");
  EXPECT_FALSE(IsPseudo);

  VarName = bufferize(SM, "BadVar+");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(VarName, "+");
  EXPECT_EQ(*ParsedName, "BadVar");
  EXPECT_FALSE(IsPseudo);

  VarName = bufferize(SM, "BadVar-");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(VarName, "-");
  EXPECT_EQ(*ParsedName, "BadVar");
  EXPECT_FALSE(IsPseudo);

  VarName = bufferize(SM, "BadVar:");
  IsPseudo = true;
  ParsedName = FileCheckPattern::parseVariable(VarName, IsPseudo, SM);
  EXPECT_TRUE(bool(ParsedName));
  EXPECT_EQ(VarName, ":");
  EXPECT_EQ(*ParsedName, "BadVar");
  EXPECT_FALSE(IsPseudo);
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
    EXPECT_FALSE(
        errorToBool(Context.defineCmdlineVariables(GlobalDefines, SM)));
    Context.createLineVariable();
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
    return errorToBool(FileCheckPattern::parseNumericVariableDefinition(
                           ExprBufferRef, &Context, LineNumber, SM)
                           .takeError());
  }

  bool parseSubstExpect(StringRef Expr) {
    StringRef ExprBufferRef = bufferize(SM, Expr);
    Optional<FileCheckNumericVariable *> DefinedNumericVariable;
    return errorToBool(P.parseNumericSubstitutionBlock(
                            ExprBufferRef, DefinedNumericVariable, SM)
                           .takeError());
  }

  bool parsePatternExpect(StringRef Pattern) {
    StringRef PatBufferRef = bufferize(SM, Pattern);
    return P.parsePattern(PatBufferRef, "CHECK", SM, Req);
  }

  bool matchExpect(StringRef Buffer) {
    StringRef BufferRef = bufferize(SM, Buffer);
    size_t MatchLen;
    return errorToBool(P.match(BufferRef, MatchLen, SM).takeError());
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

  // Only valid variable.
  EXPECT_FALSE(Tester.parseSubstExpect("@LINE"));
  EXPECT_FALSE(Tester.parseSubstExpect("FOO"));
  EXPECT_FALSE(Tester.parseSubstExpect("UNDEF"));

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

  // Check matching a numeric expression using @LINE after match failure uses
  // the correct value for @LINE.
  Tester.initNextPattern();
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE]]"));
  // Ok, @LINE is 4 now.
  EXPECT_FALSE(Tester.matchExpect("4"));
  Tester.initNextPattern();
  // @LINE is now 5, match with substitution failure.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#UNKNOWN]]"));
  EXPECT_TRUE(Tester.matchExpect("FOO"));
  Tester.initNextPattern();
  // Check that @LINE is 6 as expected.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE]]"));
  EXPECT_FALSE(Tester.matchExpect("6"));
}

TEST_F(FileCheckTest, Substitution) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  EXPECT_FALSE(errorToBool(Context.defineCmdlineVariables(GlobalDefines, SM)));

  // Substitution of an undefined string variable fails and error holds that
  // variable's name.
  FileCheckStringSubstitution StringSubstitution =
      FileCheckStringSubstitution(&Context, "VAR404", 42);
  Expected<std::string> SubstValue = StringSubstitution.getResult();
  EXPECT_FALSE(bool(SubstValue));
  expectUndefError("VAR404", SubstValue.takeError());

  // Substitutions of defined pseudo and non-pseudo numeric variables return
  // the right value.
  FileCheckNumericVariable LineVar = FileCheckNumericVariable("@LINE", 42);
  FileCheckNumericVariable NVar = FileCheckNumericVariable("N", 10);
  FileCheckExpression LineExpression = FileCheckExpression(doAdd, &LineVar, 0);
  FileCheckExpression NExpression = FileCheckExpression(doAdd, &NVar, 3);
  FileCheckNumericSubstitution SubstitutionLine =
      FileCheckNumericSubstitution(&Context, "@LINE", &LineExpression, 12);
  FileCheckNumericSubstitution SubstitutionN =
      FileCheckNumericSubstitution(&Context, "N", &NExpression, 30);
  Expected<std::string> Value = SubstitutionLine.getResult();
  EXPECT_TRUE(bool(Value));
  EXPECT_EQ("42", *Value);
  Value = SubstitutionN.getResult();
  EXPECT_TRUE(bool(Value));
  EXPECT_EQ("13", *Value);

  // Substitution of an undefined numeric variable fails.
  LineVar.clearValue();
  SubstValue = SubstitutionLine.getResult().takeError();
  EXPECT_FALSE(bool(SubstValue));
  expectUndefError("@LINE", SubstValue.takeError());
  NVar.clearValue();
  SubstValue = SubstitutionN.getResult().takeError();
  EXPECT_FALSE(bool(SubstValue));
  expectUndefError("N", SubstValue.takeError());

  // Substitution of a defined string variable returns the right value.
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Context, 1);
  StringSubstitution = FileCheckStringSubstitution(&Context, "FOO", 42);
  Value = StringSubstitution.getResult();
  EXPECT_TRUE(bool(Value));
  EXPECT_EQ("BAR", *Value);
}

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt = FileCheckPatternContext();
  std::vector<std::string> GlobalDefines;
  SourceMgr SM;

  // Missing equal sign.
  GlobalDefines.emplace_back(std::string("LocalVar"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#LocalNumVar"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

  // Empty variable name.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("=18"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#=18"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

  // Invalid variable name.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("18LocalVar=18"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#18LocalNumVar=18"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

  // Name conflict between pattern and numeric variable.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("LocalVar=18"));
  GlobalDefines.emplace_back(std::string("#LocalVar=36"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  Cxt = FileCheckPatternContext();
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#LocalNumVar=18"));
  GlobalDefines.emplace_back(std::string("LocalNumVar=36"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  Cxt = FileCheckPatternContext();

  // Invalid numeric value for numeric variable.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("#LocalNumVar=x"));
  EXPECT_TRUE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

  // Define local variables from command-line.
  GlobalDefines.clear();
  GlobalDefines.emplace_back(std::string("LocalVar=FOO"));
  GlobalDefines.emplace_back(std::string("EmptyVar="));
  GlobalDefines.emplace_back(std::string("#LocalNumVar=18"));
  EXPECT_FALSE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

  // Check defined variables are present and undefined is absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef LocalNumVarRef = bufferize(SM, "LocalNumVar");
  StringRef EmptyVarStr = "EmptyVar";
  StringRef UnknownVarStr = "UnknownVar";
  Expected<StringRef> LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  FileCheckPattern P = FileCheckPattern(Check::CheckPlain, &Cxt, 1);
  Optional<FileCheckNumericVariable *> DefinedNumericVariable;
  Expected<FileCheckExpression *> Expression = P.parseNumericSubstitutionBlock(
      LocalNumVarRef, DefinedNumericVariable, SM);
  Expected<StringRef> EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  Expected<StringRef> UnknownVar = Cxt.getPatternVarValue(UnknownVarStr);
  EXPECT_TRUE(bool(LocalVar));
  EXPECT_EQ(*LocalVar, "FOO");
  EXPECT_TRUE(bool(Expression));
  Expected<uint64_t> ExpressionVal = (*Expression)->eval();
  EXPECT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 18U);
  EXPECT_TRUE(bool(EmptyVar));
  EXPECT_EQ(*EmptyVar, "");
  EXPECT_TRUE(errorToBool(UnknownVar.takeError()));

  // Clear local variables and check they become absent.
  Cxt.clearLocalVars();
  LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  EXPECT_TRUE(errorToBool(LocalVar.takeError()));
  // Check a numeric expression's evaluation fails if called after clearing of
  // local variables, if it was created before. This is important because local
  // variable clearing due to --enable-var-scope happens after numeric
  // expressions are linked to the numeric variables they use.
  EXPECT_TRUE(errorToBool((*Expression)->eval().takeError()));
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 2);
  Expression = P.parseNumericSubstitutionBlock(LocalNumVarRef,
                                               DefinedNumericVariable, SM);
  EXPECT_TRUE(bool(Expression));
  ExpressionVal = (*Expression)->eval();
  EXPECT_TRUE(errorToBool(ExpressionVal.takeError()));
  EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  EXPECT_TRUE(errorToBool(EmptyVar.takeError()));
  // Clear again because parseNumericSubstitutionBlock would have created a
  // dummy variable and stored it in GlobalNumericVariableTable.
  Cxt.clearLocalVars();

  // Redefine global variables and check variables are defined again.
  GlobalDefines.emplace_back(std::string("$GlobalVar=BAR"));
  GlobalDefines.emplace_back(std::string("#$GlobalNumVar=36"));
  EXPECT_FALSE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  StringRef GlobalVarStr = "$GlobalVar";
  StringRef GlobalNumVarRef = bufferize(SM, "$GlobalNumVar");
  Expected<StringRef> GlobalVar = Cxt.getPatternVarValue(GlobalVarStr);
  EXPECT_TRUE(bool(GlobalVar));
  EXPECT_EQ(*GlobalVar, "BAR");
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 3);
  Expression = P.parseNumericSubstitutionBlock(GlobalNumVarRef,
                                               DefinedNumericVariable, SM);
  EXPECT_TRUE(bool(Expression));
  ExpressionVal = (*Expression)->eval();
  EXPECT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 36U);

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  EXPECT_FALSE(errorToBool(Cxt.getPatternVarValue(GlobalVarStr).takeError()));
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 4);
  Expression = P.parseNumericSubstitutionBlock(GlobalNumVarRef,
                                               DefinedNumericVariable, SM);
  EXPECT_TRUE(bool(Expression));
  ExpressionVal = (*Expression)->eval();
  EXPECT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 36U);
}
} // namespace
