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
  ExpressionLiteral Ten(10);
  Expected<uint64_t> Value = Ten.eval();
  ASSERT_TRUE(bool(Value));
  EXPECT_EQ(10U, *Value);

  // Max value can be correctly represented.
  ExpressionLiteral Max(std::numeric_limits<uint64_t>::max());
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
  handleAllErrors(std::move(Err), [&](const UndefVarError &E) {
    ExpectedUndefVarNames.erase(E.getVarName());
  });
  EXPECT_TRUE(ExpectedUndefVarNames.empty()) << toString(ExpectedUndefVarNames);
}

// Return whether Err contains any UndefVarError whose associated name is not
// ExpectedUndefVarName.
static void expectUndefError(const Twine &ExpectedUndefVarName, Error Err) {
  expectUndefErrors({ExpectedUndefVarName.str()}, std::move(Err));
}

uint64_t doAdd(uint64_t OpL, uint64_t OpR) { return OpL + OpR; }

TEST_F(FileCheckTest, NumericVariable) {
  // Undefined variable: getValue and eval fail, error returned by eval holds
  // the name of the undefined variable.
  NumericVariable FooVar("FOO", 1);
  EXPECT_EQ("FOO", FooVar.getName());
  NumericVariableUse FooVarUse("FOO", &FooVar);
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
  NumericVariable FooVar("FOO", 1);
  FooVar.setValue(42);
  std::unique_ptr<NumericVariableUse> FooVarUse =
      std::make_unique<NumericVariableUse>("FOO", &FooVar);
  NumericVariable BarVar("BAR", 2);
  BarVar.setValue(18);
  std::unique_ptr<NumericVariableUse> BarVarUse =
      std::make_unique<NumericVariableUse>("BAR", &BarVar);
  BinaryOperation Binop(doAdd, std::move(FooVarUse), std::move(BarVarUse));

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
  EXPECT_TRUE(Pattern::isValidVarNameStart('a'));
  EXPECT_TRUE(Pattern::isValidVarNameStart('G'));
  EXPECT_TRUE(Pattern::isValidVarNameStart('_'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('2'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('$'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('@'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('+'));
  EXPECT_FALSE(Pattern::isValidVarNameStart('-'));
  EXPECT_FALSE(Pattern::isValidVarNameStart(':'));
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
  Expected<Pattern::VariableProperties> ParsedVarResult =
      Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "$GoodGlobalVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "@GoodPseudoVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(ParsedVarResult->Name, OrigVarName);
  EXPECT_TRUE(VarName.empty());
  EXPECT_TRUE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "42BadVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  EXPECT_TRUE(errorToBool(ParsedVarResult.takeError()));

  VarName = bufferize(SM, "$@");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  EXPECT_TRUE(errorToBool(ParsedVarResult.takeError()));

  VarName = OrigVarName = bufferize(SM, "B@dVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedVarResult->Name, "B");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = OrigVarName = bufferize(SM, "B$dVar");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, OrigVarName.substr(1));
  EXPECT_EQ(ParsedVarResult->Name, "B");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar+");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, "+");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar-");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
  ASSERT_TRUE(bool(ParsedVarResult));
  EXPECT_EQ(VarName, "-");
  EXPECT_EQ(ParsedVarResult->Name, "BadVar");
  EXPECT_FALSE(ParsedVarResult->IsPseudo);

  VarName = bufferize(SM, "BadVar:");
  ParsedVarResult = Pattern::parseVariable(VarName, SM);
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
  Pattern P{Check::CheckPlain, &Context, LineNumber};

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
    P = Pattern(Check::CheckPlain, &Context, ++LineNumber);
  }

  size_t getLineNumber() const { return LineNumber; }

  bool parseSubstExpect(StringRef Expr, bool IsLegacyLineExpr = false) {
    StringRef ExprBufferRef = bufferize(SM, Expr);
    Optional<NumericVariable *> DefinedNumericVariable;
    return errorToBool(P.parseNumericSubstitutionBlock(
                            ExprBufferRef, DefinedNumericVariable,
                            IsLegacyLineExpr, LineNumber, &Context, SM)
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

TEST_F(FileCheckTest, ParseNumericSubstitutionBlock) {
  PatternTester Tester;

  // Variable definition.

  // Invalid variable name.
  EXPECT_TRUE(Tester.parseSubstExpect("%VAR:"));

  // Invalid definition of pseudo variable.
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
  EXPECT_FALSE(Tester.parseSubstExpect("FOOBAR: FOO+1"));

  // Numeric expression.

  // Invalid variable name.
  EXPECT_TRUE(Tester.parseSubstExpect("%VAR"));

  // Invalid pseudo variable.
  EXPECT_TRUE(Tester.parseSubstExpect("@FOO"));

  // Invalid use of variable defined on the same line. Use parsePatternExpect
  // for the variable to be recorded in GlobalNumericVariableTable and thus
  // appear defined to parseNumericVariableUse. Note that the same pattern
  // object is used for the parsePatternExpect and parseSubstExpect since no
  // initNextPattern is called, thus appearing as being on the same line from
  // the pattern's point of view.
  ASSERT_FALSE(Tester.parsePatternExpect("[[#SAME_LINE_VAR:]]"));
  EXPECT_TRUE(Tester.parseSubstExpect("SAME_LINE_VAR"));

  // Invalid use of variable defined on the same line from an expression not
  // using any variable defined on the same line.
  ASSERT_FALSE(Tester.parsePatternExpect("[[#SAME_LINE_EXPR_VAR:@LINE+1]]"));
  EXPECT_TRUE(Tester.parseSubstExpect("SAME_LINE_EXPR_VAR"));

  // Valid use of undefined variable which creates the variable and record it
  // in GlobalNumericVariableTable.
  ASSERT_FALSE(Tester.parseSubstExpect("UNDEF"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[UNDEF:.*]]"));

  // Invalid literal.
  EXPECT_TRUE(Tester.parseSubstExpect("42U"));

  // Valid empty expression.
  EXPECT_FALSE(Tester.parseSubstExpect(""));

  // Valid single operand expression.
  EXPECT_FALSE(Tester.parseSubstExpect("FOO"));

  // Valid expression with 2 or more operands.
  EXPECT_FALSE(Tester.parseSubstExpect("FOO+3"));
  EXPECT_FALSE(Tester.parseSubstExpect("FOO-3+FOO"));

  // Unsupported operator.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE/2"));

  // Missing RHS operand.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+"));

  // Errors in RHS operand are bubbled up by parseBinop() to
  // parseNumericSubstitutionBlock.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+%VAR"));

  // Invalid legacy @LINE expression with non literal rhs.
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+@LINE", /*IsLegacyNumExpr=*/true));

  // Invalid legacy @LINE expression made of a single literal.
  EXPECT_TRUE(Tester.parseSubstExpect("2", /*IsLegacyNumExpr=*/true));

  // Valid legacy @LINE expression.
  EXPECT_FALSE(Tester.parseSubstExpect("@LINE+2", /*IsLegacyNumExpr=*/true));

  // Invalid legacy @LINE expression with more than 2 operands.
  EXPECT_TRUE(
      Tester.parseSubstExpect("@LINE+2+@LINE", /*IsLegacyNumExpr=*/true));
  EXPECT_TRUE(Tester.parseSubstExpect("@LINE+2+2", /*IsLegacyNumExpr=*/true));
}

TEST_F(FileCheckTest, ParsePattern) {
  PatternTester Tester;

  // Invalid space in string substitution.
  EXPECT_TRUE(Tester.parsePatternExpect("[[ BAR]]"));

  // Invalid variable name in string substitution.
  EXPECT_TRUE(Tester.parsePatternExpect("[[42INVALID]]"));

  // Invalid string variable definition.
  EXPECT_TRUE(Tester.parsePatternExpect("[[@PAT:]]"));
  EXPECT_TRUE(Tester.parsePatternExpect("[[PAT+2:]]"));

  // Collision with numeric variable.
  EXPECT_TRUE(Tester.parsePatternExpect("[[FOO:]]"));

  // Valid use of string variable.
  EXPECT_FALSE(Tester.parsePatternExpect("[[BAR]]"));

  // Valid string variable definition.
  EXPECT_FALSE(Tester.parsePatternExpect("[[PAT:[0-9]+]]"));

  // Invalid numeric substitution.
  EXPECT_TRUE(Tester.parsePatternExpect("[[#42INVALID]]"));

  // Valid numeric substitution.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#FOO]]"));
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

  // Check matching an undefined variable returns a NotFound error.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePatternExpect("100"));
  EXPECT_TRUE(Tester.matchExpect("101"));

  // Check matching the defined variable matches the correct number only.
  Tester.initNextPattern();
  ASSERT_FALSE(Tester.parsePatternExpect("[[#NUMVAR]]"));
  EXPECT_FALSE(Tester.matchExpect("18"));

  // Check matching several substitutions does not match them independently.
  Tester.initNextPattern();
  Tester.parsePatternExpect("[[#NUMVAR]] [[#NUMVAR+2]]");
  EXPECT_TRUE(Tester.matchExpect("19 21"));
  EXPECT_TRUE(Tester.matchExpect("18 21"));
  EXPECT_FALSE(Tester.matchExpect("18 20"));

  // Check matching a numeric expression using @LINE after match failure uses
  // the correct value for @LINE.
  Tester.initNextPattern();
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE]]"));
  // Ok, @LINE matches the current line number.
  EXPECT_FALSE(Tester.matchExpect(std::to_string(Tester.getLineNumber())));
  Tester.initNextPattern();
  // Match with substitution failure.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#UNKNOWN]]"));
  EXPECT_TRUE(Tester.matchExpect("FOO"));
  Tester.initNextPattern();
  // Check that @LINE matches the later (given the calls to initNextPattern())
  // line number.
  EXPECT_FALSE(Tester.parsePatternExpect("[[#@LINE]]"));
  EXPECT_FALSE(Tester.matchExpect(std::to_string(Tester.getLineNumber())));
}

TEST_F(FileCheckTest, Substitution) {
  SourceMgr SM;
  FileCheckPatternContext Context;
  std::vector<std::string> GlobalDefines;
  GlobalDefines.emplace_back(std::string("FOO=BAR"));
  EXPECT_FALSE(errorToBool(Context.defineCmdlineVariables(GlobalDefines, SM)));

  // Substitution of an undefined string variable fails and error holds that
  // variable's name.
  StringSubstitution StringSubstitution(&Context, "VAR404", 42);
  Expected<std::string> SubstValue = StringSubstitution.getResult();
  ASSERT_FALSE(bool(SubstValue));
  expectUndefError("VAR404", SubstValue.takeError());

  // Numeric substitution blocks constituted of defined numeric variables are
  // substituted for the variable's value.
  NumericVariable NVar("N", 1);
  NVar.setValue(10);
  auto NVarUse = std::make_unique<NumericVariableUse>("N", &NVar);
  NumericSubstitution SubstitutionN(&Context, "N", std::move(NVarUse),
                                    /*InsertIdx=*/30);
  SubstValue = SubstitutionN.getResult();
  ASSERT_TRUE(bool(SubstValue));
  EXPECT_EQ("10", *SubstValue);

  // Substitution of an undefined numeric variable fails, error holds name of
  // undefined variable.
  NVar.clearValue();
  SubstValue = SubstitutionN.getResult();
  ASSERT_FALSE(bool(SubstValue));
  expectUndefError("N", SubstValue.takeError());

  // Substitution of a defined string variable returns the right value.
  Pattern P(Check::CheckPlain, &Context, 1);
  StringSubstitution = llvm::StringSubstitution(&Context, "FOO", 42);
  SubstValue = StringSubstitution.getResult();
  ASSERT_TRUE(bool(SubstValue));
  EXPECT_EQ("BAR", *SubstValue);
}

TEST_F(FileCheckTest, FileCheckContext) {
  FileCheckPatternContext Cxt;
  std::vector<std::string> GlobalDefines;
  SourceMgr SM;

  // No definition.
  EXPECT_FALSE(errorToBool(Cxt.defineCmdlineVariables(GlobalDefines, SM)));

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

  // Create @LINE pseudo numeric variable and check it is present by matching
  // it.
  size_t LineNumber = 1;
  Pattern P(Check::CheckPlain, &Cxt, LineNumber);
  FileCheckRequest Req;
  Cxt.createLineVariable();
  ASSERT_FALSE(P.parsePattern("[[@LINE]]", "CHECK", SM, Req));
  size_t MatchLen;
  ASSERT_FALSE(errorToBool(P.match("1", MatchLen, SM).takeError()));

#ifndef NDEBUG
  // Recreating @LINE pseudo numeric variable fails.
  EXPECT_DEATH(Cxt.createLineVariable(),
               "@LINE pseudo numeric variable already created");
#endif

  // Check defined variables are present and undefined ones are absent.
  StringRef LocalVarStr = "LocalVar";
  StringRef LocalNumVar1Ref = bufferize(SM, "LocalNumVar1");
  StringRef LocalNumVar2Ref = bufferize(SM, "LocalNumVar2");
  StringRef EmptyVarStr = "EmptyVar";
  StringRef UnknownVarStr = "UnknownVar";
  Expected<StringRef> LocalVar = Cxt.getPatternVarValue(LocalVarStr);
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  Optional<NumericVariable *> DefinedNumericVariable;
  Expected<std::unique_ptr<ExpressionAST>> ExpressionASTPointer =
      P.parseNumericSubstitutionBlock(LocalNumVar1Ref, DefinedNumericVariable,
                                      /*IsLegacyLineExpr=*/false, LineNumber,
                                      &Cxt, SM);
  ASSERT_TRUE(bool(LocalVar));
  EXPECT_EQ(*LocalVar, "FOO");
  Expected<StringRef> EmptyVar = Cxt.getPatternVarValue(EmptyVarStr);
  Expected<StringRef> UnknownVar = Cxt.getPatternVarValue(UnknownVarStr);
  ASSERT_TRUE(bool(ExpressionASTPointer));
  Expected<uint64_t> ExpressionVal = (*ExpressionASTPointer)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 18U);
  ExpressionASTPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar2Ref, DefinedNumericVariable,
      /*IsLegacyLineExpr=*/false, LineNumber, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionASTPointer));
  ExpressionVal = (*ExpressionASTPointer)->eval();
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
  EXPECT_TRUE(errorToBool((*ExpressionASTPointer)->eval().takeError()));
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  ExpressionASTPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar1Ref, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionASTPointer));
  ExpressionVal = (*ExpressionASTPointer)->eval();
  EXPECT_TRUE(errorToBool(ExpressionVal.takeError()));
  ExpressionASTPointer = P.parseNumericSubstitutionBlock(
      LocalNumVar2Ref, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionASTPointer));
  ExpressionVal = (*ExpressionASTPointer)->eval();
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
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  ExpressionASTPointer = P.parseNumericSubstitutionBlock(
      GlobalNumVarRef, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionASTPointer));
  ExpressionVal = (*ExpressionASTPointer)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 36U);

  // Clear local variables and check global variables remain defined.
  Cxt.clearLocalVars();
  EXPECT_FALSE(errorToBool(Cxt.getPatternVarValue(GlobalVarStr).takeError()));
  P = Pattern(Check::CheckPlain, &Cxt, ++LineNumber);
  ExpressionASTPointer = P.parseNumericSubstitutionBlock(
      GlobalNumVarRef, DefinedNumericVariable, /*IsLegacyLineExpr=*/false,
      LineNumber, &Cxt, SM);
  ASSERT_TRUE(bool(ExpressionASTPointer));
  ExpressionVal = (*ExpressionASTPointer)->eval();
  ASSERT_TRUE(bool(ExpressionVal));
  EXPECT_EQ(*ExpressionVal, 36U);
}
} // namespace
