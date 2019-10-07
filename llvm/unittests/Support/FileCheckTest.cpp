//===- llvm/unittest/Support/FileCheckTest.cpp - FileCheck tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileCheck.h"
#include "../lib/Support/FileCheckImpl.h"
#include "gtest/gtest.h"
#include <unordered_set>

using namespace llvm;

namespace {

class FileCheckTest : public ::testing::Test {};

TEST_F(FileCheckTest, Literal) {
  // Eval returns the literal's value.
  FileCheckExpressionLiteral Ten(10);
  Expected<uint64_t> Value = Ten.eval();
  ASSERT_TRUE(bool(Value));
  EXPECT_EQ(10U, *Value);

  // Max value can be correctly represented.
  FileCheckExpressionLiteral Max(std::numeric_limits<uint64_t>::max());
  Value = Max.eval();
  ASSERT_TRUE(bool(Value));
  EXPECT_EQ(std::numeric_limits<uint64_t>::max(), *Value);
}

static std::string toString(const std::unordered_set<std::string> &Set) {
  bool First = true;
  std::string Str;
  for (StringRef S : Set) {
    Str += Twine(First ? "{" + S : ", " + S).str();
    First = false;
  }
  Str += '}';
  return Str;
}

static void
expectUndefErrors(std::unordered_set<std::string> ExpectedUndefVarNames,
                  Error Err) {
  handleAllErrors(std::move(Err), [&](const FileCheckUndefVarError &E) {
    ExpectedUndefVarNames.erase(E.getVarName());
  });
  EXPECT_TRUE(ExpectedUndefVarNames.empty()) << toString(ExpectedUndefVarNames);
}

// Return whether Err contains any FileCheckUndefVarError whose associated name
// is not ExpectedUndefVarName.
static void expectUndefError(const Twine &ExpectedUndefVarName, Error Err) {
  expectUndefErrors({ExpectedUndefVarName.str()}, std::move(Err));
}

uint64_t doAdd(uint64_t OpL, uint64_t OpR) { return OpL + OpR; }

TEST_F(FileCheckTest, NumericVariable) {
  // Undefined variable: getValue and eval fail, error returned by eval holds
  // the name of the undefined variable.
  FileCheckNumericVariable FooVar("FOO", 1);
  EXPECT_EQ("FOO", FooVar.getName());
  FileCheckNumericVariableUse FooVarUse("FOO", &FooVar);
  EXPECT_FALSE(FooVar.getValue());
  Expected<uint64_t> EvalResult = FooVarUse.eval();
  ASSERT_FALSE(EvalResult);
  expectUndefError("FOO", EvalResult.takeError());

  FooVar.setValue(42);

  // Defined variable: getValue and eval return value set.
  Optional<uint64_t> Value = FooVar.getValue();
  ASSERT_TRUE(bool(Value));
  EXPECT_EQ(42U, *Value);
  EvalResult = FooVarUse.eval();
  ASSERT_TRUE(bool(EvalResult));
  EXPECT_EQ(42U, *EvalResult);

  // Clearing variable: getValue and eval fail. Error returned by eval holds
  // the name of the cleared variable.
  FooVar.clearValue();
  EXPECT_FALSE(FooVar.getValue());
  EvalResult = FooVarUse.eval();
  ASSERT_FALSE(EvalResult);
  expectUndefError("FOO", EvalResult.takeError());
}

TEST_F(FileCheckTest, Binop) {
  FileCheckNumericVariable FooVar("FOO", 1);
  FooVar.setValue(42);
  std::unique_ptr<FileCheckNumericVariableUse> FooVarUse =
      std::make_unique<FileCheckNumericVariableUse>("FOO", &FooVar);
  FileCheckNumericVariable BarVar("BAR", 2);
  BarVar.setValue(18);
  std::unique_ptr<FileCheckNumericVariableUse> BarVarUse =
      std::make_unique<FileCheckNumericVariableUse>("BAR", &BarVar);
  FileCheckASTBinop Binop(doAdd, std::move(FooVarUse), std::move(BarVarUse));

  // Defined variable: eval returns right value.
  Expected<uint64_t> Value = Binop.eval();
  ASSERT_TRUE(bool(Value));
  EXPECT_EQ(60U, *Value);

  // 1 undefined variable: eval fails, error contains name of undefined
  // variable.
  FooVar.clearValue();
  Value = Binop.eval();
  ASSERT_FALSE(Value);
  expectUndefError("FOO", Value.takeError());

  // 2 undefined variables: eval fails, error contains names of all undefined
  // variables.
  BarVar.clearValue();
  Value = Binop.eval();
  ASSERT_FALSE(Value);
  expectUndefErrors({"FOO", "BAR"}, Value.takeError());
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
  Expected<FileCheckPattern::VariableProperties> ParsedVarResult =
      FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "$GoodGlobalVar");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "@GoodPseudoVar");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_TRUE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "42BadVar");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  EXPECT_TRUE(errorToBool(ParsedVarResult.takeError()));

  VarName = bufferize(SM, "$@");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  EXPECT_TRUE(errorToBool(ParsedVarResult.takeError()));

  VarName = OrigVarName = bufferize(SM, "B@dVar");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedVarResult->Name, "B");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "B$dVar");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedVarResult->Name, "B");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar+");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, "+");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar-");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, "-");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar:");
  ParsedVarResult = FileCheckPattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, ":");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);
}

class PatternTester {
private:
  size_t LineNumber = 1;
  SourceMgr SM;
  FileCheckRequest Req;
  FileCheckPatternContext Context;
  FileCheckPattern P{Check::CheckPlain, &Context, LineNumber++};

public:
  PatternTester() {
    std::vector<std::string> GlobalDefines;
    GlobalDefines.emplace_back(std::string("#FOO=42"));
    GlobalDefines.emplace_back(std::string("BAR=BAZ"));
    // An ASSERT_FALSE would make more sense but cannot be used in a
    // constructor.
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

  bool parseSubstExpect(StringRef Expr) {
    StringRef ExprBufferRef = bufferize(SM, Expr);
    Optional<FileCheckNumericVariable *> DefinedNumericVariable;
    return errorToBool(
        P.parseNumericSubstitutionBlock(ExprBufferRef, DefinedNumericVariable,
                                        false, LineNumber - 1, &Context, SM)
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

TEST_F(FileCheckTest, ParseExpr) {
  PatternTester Tester;

  // Variable definition.

  // Definition of invalid variable.
  EXPECT_TRUE(Tester.parseSubstExpect("10VAR:"));
  EXPECT_TRUE(Tester.parseSubstExpect("@FOO:"));
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE:"));

  // Conflict with pattern variable.
  EXPECT_TRUE(Tester.parseSubstExpect("BAR:"));

  // Garbage after name of variable being defined.
  EXPECT_TRUE(Tester.parseSubstExpect("VAR GARBAGE:"));

  // Acceptable variable definition.
  EXPECT_FALSE(Tester.parseSubstExpect("VAR1:"));
  EXPECT_FALSE(Tester.parseSubstExpect("  VAR2:"));
  EXPECT_FALSE(Tester.parseSubstExpect("VAR3  :"));
  EXPECT_FALSE(Tester.parseSubstExpect("VAR3:  "));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#FOOBAR: FOO+1]]"));

  // Numeric expression.

  // Unacceptable variable.
  EXPECT_TRUE(Tester.parseSubstExpect("10VAR"));
  EXPECT_TRUE(Tester.parseSubstExpect("@FOO"));

  // Only valid variable.
  EXPECT_FALSE(Tester.parseSubstExpect("@LINE"));
  EXPECT_FALSE(Tester.parseSubstExpect("FOO"));
  EXPECT_FALSE(Tester.parseSubstExpect("UNDEF"));

  // Valid empty expression.
  EXPECT_FALSE(Tester.parseSubstExpect(""));

  // Invalid use of variable defined on the same line from expression. Note
  // that the same pattern object is used for the parsePatternExpect and
  // parseSubstExpect since no initNextPattern is called, thus appearing as
  // being on the same line from the pattern's point of view.
  ASSERT_FALSE(Tester.parsePatternExpect("[[#LINE1VAR:FOO+1]]"));
  EXPECT_TRUE(Tester.parseSubstExpect("LINE1VAR"));

  // Invalid use of variable defined on same line from input. As above, the
  // absence of a call to initNextPattern makes it appear to be on the same
  // line from the pattern's point of view.
  ASSERT_FALSE(Tester.parsePatternExpect("[[#LINE2VAR:]]"));
  EXPECT_TRUE(Tester.parseSubstExpect("LINE2VAR"));

  // Unsupported operator.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE/2"));

  // Missing offset operand.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+"));

  // Valid expression.
  EXPECT_FALSE(Tester.parseSubstExpect("@LINE+5"));
  EXPECT_FALSE(Tester.parseSubstExpect("FOO+4"));
  Tester.initNextPattern();
  EXPECT_FALSE(Tester.parseSubstExpect("FOOBAR"));
  EXPECT_FALSE(Tester.parseSubstExpect("LINE1VAR"));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#FOO+FOO]]"));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#FOO+3-FOO]]"));
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

  // Valid numeric expressions and numeric variable definition.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#FOO]]"));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE+2]]"));
  EXPECT_FALSE(Tester.parsePatternExpect("[[#NUMVAR:]]"));
}

TEST_F(FileCheckTest, Match) {
  PatternTester Tester;

  // Check matching an empty expression only matches a number.
  Tester.parsePatternExpect("[[#]]");
  EXPECT_TRUE(Tester.matchExpect("FAIL"));
  EXPECT_FALSE(Tester.matchExpect("18"));

  // Check matching a definition only matches a number.
  Tester.initNextPattern();
  Tester.parsePatternExpect("[[#NUMVAR:]]");
  EXPECT_TRUE(Tester.matchExpect("FAIL"));
  EXPECT_TRUE(Tester.matchExpect(""));
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
  // Ok, @LINE is 5 now.
  EXPECT_FALSE(Tester.matchExpect("5"));
  Tester.initNextPattern();
  // @LINE is now 6, match with substitution failure.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#UNKNOWN]]"));
  EXPECT_TRUE(Tester.matchExpect("FOO"));
  Tester.initNextPattern();
  // Check that @LINE is 7 as expected.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE]]"));
  EXPECT_FALSE(Tester.matchExpect("7"));
}

TEST_F(FileCheckTest, Substitution) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  EXPECT_FALSE(errorToBool(Context.defineCmdlineVariables(GlobalDefines, SM)));

  // Substitution of an undefined string variable fails and error holds that
  // variable's name.
  FileCheckStringSubstitution StringSubstitution(&Context, "VAR404", 42);
  Expected<std::string> SubstValue = StringSubstitution.getResult();
  ASSERT_FALSE(bool(SubstValue));
  expectUndefError("VAR404", SubstValue.takeError());

  // Substitutions of defined pseudo and non-pseudo numeric variables return
  // the right value.
  FileCheckNumericVariable LineVar("@LINE", 1);
  FileCheckNumericVariable NVar("N", 1);
  LineVar.setValue(42);
  NVar.setValue(10);
  auto LineVarUse =
      std::make_unique<FileCheckNumericVariableUse>("@LINE", &LineVar);
  auto NVarUse = std::make_unique<FileCheckNumericVariableUse>("N", &NVar);
  FileCheckNumericSubstitution SubstitutionLine(&Context, "@LINE",
                                                std::move(LineVarUse), 12);
  FileCheckNumericSubstitution SubstitutionN(&Context, "N", std::move(NVarUse),
                                             30);
  SubstValue = SubstitutionLine.getResult();
  ASSERT_TRUE(bool(SubstValue));
  EXPECT_EQ("42", *SubstValue);
  SubstValue = SubstitutionN.getResult();
  ASSERT_TRUE(bool(SubstValue));
  EXPECT_EQ("10", *SubstValue);

  // Substitution of an undefined numeric variable fails, error holds name of
  // undefined variable.
  LineVar.clearValue();
  SubstValue = SubstitutionLine.getResult();
  ASSERT_FALSE(bool(SubstValue));
  expectUndefError("@LINE", SubstValue.takeError());
  NVar.clearValue();
  SubstValue = SubstitutionN.getResult();
  ASSERT_FALSE(bool(SubstValue));
  expectUndefError("N", SubstValue.takeError());

  // Substitution of a defined string variable returns the right value.
  FileCheckPattern P(Check::CheckPlain, &Context, 1);
  StringSubstitution = FileCheckStringSubstitution(&Context, "FOO", 42);
  SubstValue = StringSubstitution.getResult();
  ASSERT_TRUE(bool(SubstValue));
  EXPECT_EQ("BAR", *SubstValue);
}

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt;
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
  // Clear local variables to remove dummy numeric variable x that
  // parseNumericSubstitutionBlock would have created and stored in
  // GlobalNumericVariableTable.
  Cxt.clearLocalVars();
  GlobalDefines.emplace_back(std::string("LocalVar=FOO"));
  GlobalDefines.emplace_back(std::string("EmptyVar="));
  GlobalDefines.emplace_back(std::string("#LocalNumVar1=18"));
  GlobalDefines.emplace_back(std::string("#LocalNumVar2=LocalNumVar1+2"));
  ASSERT_FALSE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

  // Check defined variables are present and undefined is absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef LocalNumVar1Ref = bufferize(SM, "LocalNumVar1");
  StringRef LocalNumVar2Ref = bufferize(SM, "LocalNumVar2");
  StringRef EmptyVarStr = "EmptyVar";
  StringRef UnknownVarStr = "UnknownVar";
  Expected<StringRef> LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  FileCheckPattern P(Check::CheckPlain, &Cxt, 1);
  Optional<FileCheckNumericVariable *> DefinedNumericVariable;
  Expected<std::unique_ptr<FileCheckExpressionAST>> ExpressionAST =
      P.parseNumericSubstitutionBlock(LocalNumVar1Ref, DefinedNumericVariable,
                                      /*IsLegacyLineExpr=*/false,
                                      /*LineNumber=*/1, &Cxt, SM);
  ASSERT_TRUE(bool(LocalVar));
  EXPECT_EQ(*LocalVar, "FOO");
  Expected<StringRef> EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  Expected<StringRef> UnknownVar = Cxt.getPatternVarValue(UnknownVarStr);
  ASSERT_TRUE(bool(ExpressionAST));
  Expected<uint64_t> ExpressionVal = (*ExpressionAST)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 18U);
  ExpressionAST =
      P.parseNumericSubstitutionBlock(LocalNumVar2Ref, DefinedNumericVariable,
                                      /*IsLegacyLineExpr=*/false,
                                      /*LineNumber=*/1, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionAST));
  ExpressionVal = (*ExpressionAST)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 20U);
  ASSERT_TRUE(bool(EmptyVar));
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
  EXPECT_TRUE(errorToBool((*ExpressionAST)->eval().takeError()));
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 2);
  ExpressionAST = P.parseNumericSubstitutionBlock(
      LocalNumVar1Ref, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      /*LineNumber=*/2, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionAST));
  ExpressionVal = (*ExpressionAST)->eval();
  EXPECT_TRUE(errorToBool(ExpressionVal.takeError()));
  ExpressionAST = P.parseNumericSubstitutionBlock(
      LocalNumVar2Ref, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      /*LineNumber=*/2, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionAST));
  ExpressionVal = (*ExpressionAST)->eval();
  EXPECT_TRUE(errorToBool(ExpressionVal.takeError()));
  EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  EXPECT_TRUE(errorToBool(EmptyVar.takeError()));
  // Clear again because parseNumericSubstitutionBlock would have created a
  // dummy variable and stored it in GlobalNumericVariableTable.
  Cxt.clearLocalVars();

  // Redefine global variables and check variables are defined again.
  GlobalDefines.emplace_back(std::string("$GlobalVar=BAR"));
  GlobalDefines.emplace_back(std::string("#$GlobalNumVar=36"));
  ASSERT_FALSE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));
  StringRef GlobalVarStr = "$GlobalVar";
  StringRef GlobalNumVarRef = bufferize(SM, "$GlobalNumVar");
  Expected<StringRef> GlobalVar = Cxt.getPatternVarValue(GlobalVarStr);
  ASSERT_TRUE(bool(GlobalVar));
  EXPECT_EQ(*GlobalVar, "BAR");
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 3);
  ExpressionAST = P.parseNumericSubstitutionBlock(
      GlobalNumVarRef, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      /*LineNumber=*/3, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionAST));
  ExpressionVal = (*ExpressionAST)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 36U);

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  EXPECT_FALSE(errorToBool(Cxt.getPatternVarValue(GlobalVarStr).takeError()));
  P = FileCheckPattern(Check::CheckPlain, &Cxt, 4);
  ExpressionAST = P.parseNumericSubstitutionBlock(
      GlobalNumVarRef, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      /*LineNumber=*/4, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionAST));
  ExpressionVal = (*ExpressionAST)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 36U);
}
} // namespace
