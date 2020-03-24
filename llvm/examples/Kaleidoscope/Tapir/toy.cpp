#include "../include/KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Instrumentation/CilkSanitizer.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/TapirToTarget.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

// The lexer returns tokens [0-255] if it is an unknown character, otherwise one
// of these for known things.
enum Token {
  tok_eof = -1,

  // commands
  tok_def = -2,
  tok_extern = -3,

  // primary
  tok_identifier = -4,
  tok_number = -5,
  tok_integer = -6,

  // control
  tok_if = -7,
  tok_then = -8,
  tok_else = -9,
  tok_for = -10,
  tok_in = -11,

  // operators
  tok_binary = -12,
  tok_unary = -13,

  // var definition
  tok_var = -14,

  // parallel control
  tok_spawn = -15,
  tok_sync = -16,
  tok_parfor = -17
};

static std::string IdentifierStr; // Filled in if tok_identifier
static int64_t IntVal;             // Filled in if tok_integer
static double NumVal;             // Filled in if tok_number

/// gettok - Return the next token from standard input.
static int gettok() {
  static int LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();

  if (isalpha(LastChar)) { // identifier: [a-zA-Z][a-zA-Z0-9]*
    IdentifierStr = LastChar;
    while (isalnum((LastChar = getchar())))
      IdentifierStr += LastChar;

    if (IdentifierStr == "def")
      return tok_def;
    if (IdentifierStr == "extern")
      return tok_extern;
    if (IdentifierStr == "if")
      return tok_if;
    if (IdentifierStr == "then")
      return tok_then;
    if (IdentifierStr == "else")
      return tok_else;
    if (IdentifierStr == "for")
      return tok_for;
    if (IdentifierStr == "in")
      return tok_in;
    if (IdentifierStr == "binary")
      return tok_binary;
    if (IdentifierStr == "unary")
      return tok_unary;
    if (IdentifierStr == "var")
      return tok_var;
    if (IdentifierStr == "spawn")
      return tok_spawn;
    if (IdentifierStr == "sync")
      return tok_sync;
    if (IdentifierStr == "parfor")
      return tok_parfor;
    return tok_identifier;
  }

  {
    std::string NumStr;
    if (isdigit(LastChar)) { // Integer: [0-9]+
      do {
        NumStr += LastChar;
        LastChar = getchar();
      } while (isdigit(LastChar));
      if (LastChar != '.') {
        IntVal = strtol(NumStr.c_str(), nullptr, 10);
        return tok_integer;
      }
    }
    if (isdigit(LastChar) || LastChar == '.') { // Number: [0-9.]+
      // std::string NumStr;
      do {
        NumStr += LastChar;
        LastChar = getchar();
      } while (isdigit(LastChar) || LastChar == '.');

      NumVal = strtod(NumStr.c_str(), nullptr);
      return tok_number;
    }
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return gettok();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}

//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

namespace {

/// ExprAST - Base class for all expression nodes.
class ExprAST {
public:
  virtual ~ExprAST() = default;

  virtual Value *codegen() = 0;
  virtual void setIntegerRes(bool v = true) {}
};

/// IntegerExprAST - Expression class for integer literals like "1".
class IntegerExprAST : public ExprAST {
  int64_t Val;

public:
  IntegerExprAST(int64_t Val) : Val(Val) {}

  Value *codegen() override;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double Val;

public:
  NumberExprAST(double Val) : Val(Val) {}

  Value *codegen() override;
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string Name;

public:
  VariableExprAST(const std::string &Name) : Name(Name) {}

  Value *codegen() override;
  const std::string &getName() const { return Name; }
};

/// UnaryExprAST - Expression class for a unary operator.
class UnaryExprAST : public ExprAST {
  char Opcode;
  std::unique_ptr<ExprAST> Operand;

public:
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
      : Opcode(Opcode), Operand(std::move(Operand)) {}

  Value *codegen() override;
};

/// BinaryExprAST - Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char Op;
  bool IntegerRes = false;
  std::unique_ptr<ExprAST> LHS, RHS;

public:
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
      : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *codegen() override;
  void setIntegerRes(bool v = true) override { IntegerRes = v; }
};

/// CallExprAST - Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string Callee;
  std::vector<std::unique_ptr<ExprAST>> Args;

public:
  CallExprAST(const std::string &Callee,
              std::vector<std::unique_ptr<ExprAST>> Args)
      : Callee(Callee), Args(std::move(Args)) {}

  Value *codegen() override;
};

/// IfExprAST - Expression class for if/then/else.
class IfExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Cond, Then, Else;

public:
  IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
            std::unique_ptr<ExprAST> Else)
      : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}

  Value *codegen() override;
};

/// ForExprAST - Expression class for for/in.
class ForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
  ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
             std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
             std::unique_ptr<ExprAST> Body)
      : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
        Step(std::move(Step)), Body(std::move(Body)) {}

  Value *codegen() override;
};

/// VarExprAST - Expression class for var/in
class VarExprAST : public ExprAST {
  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
  std::unique_ptr<ExprAST> Body;

public:
  VarExprAST(
      std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
      std::unique_ptr<ExprAST> Body)
      : VarNames(std::move(VarNames)), Body(std::move(Body)) {}

  Value *codegen() override;
};

/// SpawnExprAST - Expression class for spawn.
class SpawnExprAST : public ExprAST {
  std::unique_ptr<ExprAST> Spawned;

public:
  SpawnExprAST(std::unique_ptr<ExprAST> Spawned)
      : Spawned(std::move(Spawned)) {}

  Value *codegen() override;
};

/// SyncExprAST - Expression class for spawn.
class SyncExprAST : public ExprAST {
public:
  SyncExprAST() {}

  Value *codegen() override;
};

/// ParForExprAST - Expression class for parfor/in.
class ParForExprAST : public ExprAST {
  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
  ParForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
                std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
                std::unique_ptr<ExprAST> Body)
      : VarName(VarName), Start(std::move(Start)), End(std::move(End)),
        Step(std::move(Step)), Body(std::move(Body)) {}

  Value *codegen() override;
};

/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes), as well as if it is an operator.
class PrototypeAST {
  std::string Name;
  std::vector<std::string> Args;
  bool IsOperator;
  unsigned Precedence; // Precedence if a binary op.

public:
  PrototypeAST(const std::string &Name, std::vector<std::string> Args,
               bool IsOperator = false, unsigned Prec = 0)
      : Name(Name), Args(std::move(Args)), IsOperator(IsOperator),
        Precedence(Prec) {}

  Function *codegen();
  const std::string &getName() const { return Name; }

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size() - 1];
  }

  unsigned getBinaryPrecedence() const { return Precedence; }
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
      : Proto(std::move(Proto)), Body(std::move(Body)) {}

  Function *codegen();
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0)
    return -1;
  return TokPrec;
}

/// LogError* - These are little helper functions for error handling.
std::unique_ptr<ExprAST> LogError(const char *Str) {
  fprintf(stderr, "Error: %s\n", Str);
  return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
  LogError(Str);
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// integerexpr ::= integer
static std::unique_ptr<ExprAST> ParseIntegerExpr() {
  auto Result = llvm::make_unique<IntegerExprAST>(IntVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

/// numberexpr ::= number
static std::unique_ptr<ExprAST> ParseNumberExpr() {
  auto Result = llvm::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return std::move(Result);
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken(); // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return LogError("expected ')'");
  getNextToken(); // eat ).
  return V;
}

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  getNextToken(); // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return llvm::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken(); // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (true) {
      if (auto Arg = ParseExpression())
        Args.push_back(std::move(Arg));
      else
        return nullptr;

      if (CurTok == ')')
        break;

      if (CurTok != ',')
        return LogError("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return llvm::make_unique<CallExprAST>(IdName, std::move(Args));
}

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
  getNextToken(); // eat the if.

  // condition.
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return LogError("expected then");
  getNextToken(); // eat the then

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return LogError("expected else");

  getNextToken();

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return llvm::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
}

/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseForExpr() {
  getNextToken(); // eat the for.

  if (CurTok != tok_identifier)
    return LogError("expected identifier after for");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  if (CurTok != '=')
    return LogError("expected '=' after for");
  getNextToken(); // eat '='.

  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("expected ',' after for start value");
  getNextToken();

  auto End = ParseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' after for");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return llvm::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}

/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<ExprAST> ParseVarExpr() {
  getNextToken(); // eat the var.

  std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return LogError("expected identifier after var");

  while (true) {
    std::string Name = IdentifierStr;
    getNextToken(); // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init = nullptr;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseExpression();
      if (!Init)
        return nullptr;
    }

    VarNames.push_back(std::make_pair(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',')
      break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return LogError("expected identifier list after var");
  }

  // At this point, we have to have 'in'.
  if (CurTok != tok_in)
    return LogError("expected 'in' keyword after 'var'");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return llvm::make_unique<VarExprAST>(std::move(VarNames), std::move(Body));
}

/// spawnexpr ::= 'spawn' expression
static std::unique_ptr<ExprAST> ParseSpawnExpr() {
  getNextToken(); // eat the spawn.
  auto Spawned = ParseExpression();
  if (!Spawned)
    return nullptr;
  return llvm::make_unique<SpawnExprAST>(std::move(Spawned));
}

/// syncexpr ::= 'sync'
static std::unique_ptr<ExprAST> ParseSyncExpr() {
  getNextToken(); // eat the sync.
  return llvm::make_unique<SyncExprAST>();
}

/// parforexpr ::= 'parfor' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ExprAST> ParseParForExpr() {
  getNextToken(); // eat the parfor.

  if (CurTok != tok_identifier)
    return LogError("expected identifier after parfor");

  std::string IdName = IdentifierStr;
  getNextToken(); // eat identifier.

  if (CurTok != '=')
    return LogError("expected '=' after for");
  getNextToken(); // eat '='.

  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return LogError("expected ',' after for start value");
  getNextToken();

  auto End = ParseExpression();
  if (!End)
    return nullptr;

  // The step value is optional.
  std::unique_ptr<ExprAST> Step;
  if (CurTok == ',') {
    getNextToken();
    Step = ParseExpression();
    if (!Step)
      return nullptr;
  }

  if (CurTok != tok_in)
    return LogError("expected 'in' after for");
  getNextToken(); // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return llvm::make_unique<ParForExprAST>(IdName, std::move(Start), std::move(End),
                                          std::move(Step), std::move(Body));
}

/// primary
///   ::= identifierexpr
///   ::= integerexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
///   ::= spawnexpr
///   ::= syncexpr
///   ::= parforexpr
static std::unique_ptr<ExprAST> ParsePrimary(bool Integer = false) {
  switch (CurTok) {
  default:
    return LogError("unknown token when expecting an expression");
  case tok_identifier:
    return ParseIdentifierExpr();
  case tok_integer:
    return ParseIntegerExpr();
  case tok_number:
    return ParseNumberExpr();
  case '(':
    return ParseParenExpr();
  case tok_if:
    return ParseIfExpr();
  case tok_for:
    return ParseForExpr();
  case tok_var:
    return ParseVarExpr();
  case tok_spawn:
    return ParseSpawnExpr();
  case tok_sync:
    return ParseSyncExpr();
  case tok_parfor:
    return ParseParForExpr();
  }
}

/// unary
///   ::= primary
///   ::= '!' unary
static std::unique_ptr<ExprAST> ParseUnary() {
  // If the current token is not an operator, it must be a primary expr.
  if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
    return ParsePrimary();

  // If this is a unary operator, read it.
  int Opc = CurTok;
  getNextToken();
  if (auto Operand = ParseUnary())
    return llvm::make_unique<UnaryExprAST>(Opc, std::move(Operand));
  return nullptr;
}

/// binoprhs
///   ::= ('+' unary)*
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec,
                                              std::unique_ptr<ExprAST> LHS,
                                              bool Integer = false) {
  // If this is a binop, find its precedence.
  while (true) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    getNextToken(); // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = ParseUnary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
    LHS =
        llvm::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
  }
}

/// expression
///   ::= unary binoprhs
///
static std::unique_ptr<ExprAST> ParseExpression() {
  auto LHS = ParseUnary();
  if (!LHS)
    return nullptr;

  return ParseBinOpRHS(0, std::move(LHS));
}

/// prototype
///   ::= id '(' id* ')'
///   ::= binary LETTER number? (id, id)
///   ::= unary LETTER (id)
static std::unique_ptr<PrototypeAST> ParsePrototype() {
  std::string FnName;

  unsigned Kind = 0; // 0 = identifier, 1 = unary, 2 = binary.
  unsigned BinaryPrecedence = 30;

  switch (CurTok) {
  default:
    return LogErrorP("Expected function name in prototype");
  case tok_identifier:
    FnName = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected unary operator");
    FnName = "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return LogErrorP("Expected binary operator");
    FnName = "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_integer) {
      if (IntVal < 1 || IntVal > 100)
        return LogErrorP("Invalid precedence: must be 1..100");
      BinaryPrecedence = (unsigned)IntVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return LogErrorP("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return LogErrorP("Expected ')' in prototype");

  // success.
  getNextToken(); // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return LogErrorP("Invalid number of operands for operator");

  return llvm::make_unique<PrototypeAST>(FnName, ArgNames, Kind != 0,
                                         BinaryPrecedence);
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken(); // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto E = ParseExpression())
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto = llvm::make_unique<PrototypeAST>("__anon_expr",
                                                 std::vector<std::string>());
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken(); // eat extern.
  return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static LLVMContext TheContext;
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;
// static std::map<std::string, AllocaInst *> NamedValues;
static std::map<std::string, Value *> NamedValues;
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<legacy::PassManager> TheMPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static TapirTargetID TheTapirTarget;
static bool Optimize = true;
static bool RunCilksan = false;
// Variables for codegen for the current task scope.
static BasicBlock *TaskScopeEntry = nullptr;
static Value *TaskScopeSyncRegion = nullptr;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

Function *getFunction(std::string Name) {
  // First, see if the function has already been added to the current module.
  if (auto *F = TheModule->getFunction(Name))
    return F;

  // If not, check whether we can codegen the declaration from some existing
  // prototype.
  auto FI = FunctionProtos.find(Name);
  if (FI != FunctionProtos.end())
    return FI->second->codegen();

  // If no existing prototype exists, return null.
  return nullptr;
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          const std::string &VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                   TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getDoubleTy(TheContext), nullptr, VarName);
}

/// CreateTaskEntryBlockAlloca - Create an alloca instruction in the entry block
/// of the current task.  This is used for mutable variables etc.
///
/// Requires the CFG of the function to be constructed up to BB.
static AllocaInst *CreateTaskEntryBlockAlloca(const std::string &VarName,
                                              Type *AllocaTy =
                                              Type::getDoubleTy(TheContext)) {
  // BasicBlock *TaskEntry = GetDetachedCtx(BB);
  BasicBlock *TaskEntry = TaskScopeEntry;
  if (!TaskEntry) {
    LogError("No local task scope.");
    return nullptr;
  }
  IRBuilder<> TmpB(TaskEntry, TaskEntry->begin());
  return TmpB.CreateAlloca(AllocaTy, nullptr, VarName);
}

Value *IntegerExprAST::codegen() {
  return ConstantInt::get(TheContext, APSInt::get(Val));
}

Value *NumberExprAST::codegen() {
  return ConstantFP::get(TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");

  if (!isa<AllocaInst>(V))
    return V;

  // Load the value.
  return Builder.CreateLoad(V, Name.c_str());
}

Value *UnaryExprAST::codegen() {
  Value *OperandV = Operand->codegen();
  if (!OperandV)
    return nullptr;

  Function *F = getFunction(std::string("unary") + Opcode);
  if (!F)
    return LogErrorV("Unknown unary operator");

  return Builder.CreateCall(F, OperandV, "unop");
}

Value *BinaryExprAST::codegen() {
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    // This assume we're building without RTTI because LLVM builds that way by
    // default.  If you build LLVM with RTTI this can be changed to a
    // dynamic_cast for automatic error checking.
    VariableExprAST *LHSE = static_cast<VariableExprAST *>(LHS.get());
    if (!LHSE)
      return LogErrorV("destination of '=' must be a variable");
    // Codegen the RHS.
    Value *Val = RHS->codegen();
    if (!Val)
      return nullptr;

    // Look up the name.
    Value *Variable = NamedValues[LHSE->getName()];
    if (!Variable)
      return LogErrorV("Unknown variable name");

    Builder.CreateStore(Val, Variable);
    return Val;
  }

  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R)
    return nullptr;
  Type *LTy = L->getType();
  Type *RTy = R->getType();
  bool IntegerOp = IntegerRes ||
    (LTy->isIntegerTy() && RTy->isIntegerTy());
  // Cast the operand types if necessary
  if (!IntegerOp) {
    if (LTy->isIntegerTy())
      L = Builder.CreateSIToFP(L, Type::getDoubleTy(TheContext));
    if (RTy->isIntegerTy())
      R = Builder.CreateSIToFP(R, Type::getDoubleTy(TheContext));
  } else if (IntegerRes) {
    if (!LTy->isIntegerTy())
      L = Builder.CreateFPToSI(L, Type::getInt64Ty(TheContext));
    if (!RTy->isIntegerTy())
      R = Builder.CreateFPToSI(R, Type::getInt64Ty(TheContext));
  }
  // Create the appropriate operation
  switch (Op) {
  case '+':
    if (IntegerOp)
      return Builder.CreateAdd(L, R, "addtmp");
    return Builder.CreateFAdd(L, R, "addtmp");
  case '-':
    if (IntegerOp)
      return Builder.CreateSub(L, R, "subtmp");
    return Builder.CreateFSub(L, R, "subtmp");
  case '*':
    if (IntegerOp)
      return Builder.CreateMul(L, R, "multmp");
    return Builder.CreateFMul(L, R, "multmp");
  case '<':
    if (IntegerOp) {
      L = Builder.CreateICmpSLT(L, R, "cmptmp");
      return Builder.CreateZExt(L, Type::getInt64Ty(TheContext), "booltmp");
    }
    L = Builder.CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return Builder.CreateUIToFP(L, Type::getDoubleTy(TheContext), "booltmp");
  default:
    break;
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "binary operator not found!");

  Value *Ops[] = {L, R};
  return Builder.CreateCall(F, Ops, "binop");
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *CalleeF = getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    Value *ArgVal = Args[i]->codegen();
    if (ArgVal->getType()->isIntegerTy())
      ArgVal = Builder.CreateSIToFP(ArgVal, Type::getDoubleTy(TheContext));
    ArgsV.push_back(ArgVal);
    if (!ArgsV.back())
      return nullptr;
  }

  return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
}

Value *IfExprAST::codegen() {
  Value *CondV = Cond->codegen();
  if (!CondV)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  CondV = Builder.CreateFCmpONE(
      CondV, ConstantFP::get(TheContext, APFloat(0.0)), "ifcond");

  Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(TheContext, "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(TheContext, "else");
  BasicBlock *MergeBB = BasicBlock::Create(TheContext, "ifcont");

  Builder.CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  Builder.SetInsertPoint(ThenBB);

  Value *ThenV = Then->codegen();
  if (!ThenV)
    return nullptr;

  Builder.CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = Builder.GetInsertBlock();

  // Emit else block.
  TheFunction->getBasicBlockList().push_back(ElseBB);
  Builder.SetInsertPoint(ElseBB);

  Value *ElseV = Else->codegen();
  if (!ElseV)
    return nullptr;

  Builder.CreateBr(MergeBB);
  // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
  ElseBB = Builder.GetInsertBlock();

  // Emit merge block.
  TheFunction->getBasicBlockList().push_back(MergeBB);
  Builder.SetInsertPoint(MergeBB);
  bool IntegerType = (ThenV->getType()->isIntegerTy() &&
                      ElseV->getType()->isIntegerTy());
  Type *PNTy = IntegerType ? Type::getInt64Ty(TheContext) :
    Type::getDoubleTy(TheContext);
  PHINode *PN = Builder.CreatePHI(PNTy, 2, "iftmp");
  if (!IntegerType) {
    if (ThenV->getType()->isIntegerTy())
      ThenV = Builder.CreateSIToFP(ThenV, Type::getDoubleTy(TheContext));
    if (ElseV->getType()->isIntegerTy())
      ElseV = Builder.CreateSIToFP(ElseV, Type::getDoubleTy(TheContext));
  }
  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  return PN;
}

// Output for-loop as:
//   var = alloca double
//   ...
//   start = startexpr
//   store start -> var
//   br cond
// cond:
//   endcond = endexpr
//   br endcond, loop, afterloop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br cond
// afterloop:
Value *ForExprAST::codegen() {
  Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateTaskEntryBlockAlloca(VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;
  if (StartVal->getType()->isIntegerTy())
    StartVal = Builder.CreateSIToFP(StartVal, Type::getDoubleTy(TheContext));

  // Store the value into the alloca.
  Builder.CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *CondBB = BasicBlock::Create(TheContext, "cond", TheFunction);
  BasicBlock *LoopBB = BasicBlock::Create(TheContext, "loop", TheFunction);
  BasicBlock *AfterBB = BasicBlock::Create(TheContext, "afterloop");

  // Insert an explicit fall through from the current block to the CondBB.
  Builder.CreateBr(CondBB);

  // Start insertion in CondBB.
  Builder.SetInsertPoint(CondBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  Value *OldVal = NamedValues[VarName];
  NamedValues[VarName] = Alloca;

  // Compute the end condition.
  Value *EndCond = End->codegen();
  if (!EndCond)
    return nullptr;

  // Convert condition to a bool by comparing non-equal to 0.0.
  EndCond = Builder.CreateFCmpONE(
      EndCond, ConstantFP::get(TheContext, APFloat(0.0)), "loopcond");

  // Insert the conditional branch into the end of LoopEndBB.
  Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder.SetInsertPoint(LoopBB);

  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  if (!Body->codegen())
    return nullptr;

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen();
    if (!StepVal)
      return nullptr;
  } else {
    // If not specified, use 1.0.
    StepVal = ConstantFP::get(TheContext, APFloat(1.0));
  }

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = Builder.CreateLoad(Alloca, VarName.c_str());
  Value *NextVar = Builder.CreateFAdd(CurVar, StepVal, "nextvar");
  Builder.CreateStore(NextVar, Alloca);

  // Insert a back edge to CondBB.
  Builder.CreateBr(CondBB);

  // Emit the "after loop" block.
  TheFunction->getBasicBlockList().push_back(AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder.SetInsertPoint(AfterBB);

  // Restore the unshadowed variable.
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

Value *VarExprAST::codegen() {
  // std::vector<AllocaInst *> OldBindings;
  std::vector<Value *> OldBindings;

  // Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
    const std::string &VarName = VarNames[i].first;
    ExprAST *Init = VarNames[i].second.get();

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    //    var a = a in ...   # refers to outer 'a'.
    Value *InitVal;
    if (Init) {
      InitVal = Init->codegen();
      if (!InitVal)
        return nullptr;
    } else { // If not specified, use 0.0.
      InitVal = ConstantFP::get(TheContext, APFloat(0.0));
    }

    AllocaInst *Alloca = CreateTaskEntryBlockAlloca(VarName);
    Builder.CreateStore(InitVal, Alloca);

    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    OldBindings.push_back(NamedValues[VarName]);

    // Remember this binding.
    NamedValues[VarName] = Alloca;
  }

  // Codegen the body, now that all vars are in scope.
  Value *BodyVal = Body->codegen();
  if (!BodyVal)
    return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
    NamedValues[VarNames[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
}

// RAII class to manage the entry block and sync region in each nested task
// scope.
class TaskScopeRAII {
  BasicBlock *OldTaskScopeEntry;
  Value *OldSyncRegion = nullptr;
public:
  explicit TaskScopeRAII(BasicBlock *NewTaskScopeEntry) :
      OldTaskScopeEntry(TaskScopeEntry), OldSyncRegion(TaskScopeSyncRegion) {
    TaskScopeEntry = NewTaskScopeEntry;
    TaskScopeSyncRegion = nullptr;
  }
  ~TaskScopeRAII() {
    TaskScopeEntry = OldTaskScopeEntry;
    TaskScopeSyncRegion = OldSyncRegion;
  }
};

// Helper method for creating sync regions.
static Value *CreateSyncRegion(Module &M) {
  // BasicBlock *TaskEntry = GetDetachedCtx(BB);
  BasicBlock *TaskEntry = TaskScopeEntry;
  if (!TaskEntry)
    return LogErrorV("No local task scope.");
  IRBuilder<> TmpB(TaskEntry, TaskEntry->begin());
  return TmpB.CreateCall(
      Intrinsic::getDeclaration(&M, Intrinsic::syncregion_start), {});
}

// Output spawn spawned_expr as:
//   sync_region = call token @llvm.syncregion.start()
//   ...
//   detach within sync_region, label detachbb, label continbb
// detachbb:
//   ...
//   spawned_expr
//   ...
//   reattach within sync_region, continbb
// continbb:
Value *SpawnExprAST::codegen() {
  // Create a sync region for the local function or task scope, if necessary.
  if (!TaskScopeSyncRegion)
    TaskScopeSyncRegion = CreateSyncRegion(*TheModule);
  Value *SyncRegion = TaskScopeSyncRegion;
  Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Create the detach and continue blocks.  Insert the continue block at the
  // end of the function.
  BasicBlock *DetachBB = BasicBlock::Create(TheContext, "detachbb",
                                            TheFunction);
  BasicBlock *ContinBB = BasicBlock::Create(TheContext, "continbb");

  // Create the detach and prepare to emit the spawned expression starting in
  // the detach block.
  Builder.CreateDetach(DetachBB, ContinBB, SyncRegion);
  Builder.SetInsertPoint(DetachBB);

  // Emit the spawned computation.
  {
    TaskScopeRAII TaskScope(DetachBB);
    // Emit the spawned expr.  This, like any other expr, can change the current
    // BB.
    if (!Spawned->codegen())
      return nullptr;

    // Emit a reattach to the continue block.
    Builder.CreateReattach(ContinBB, SyncRegion);
  }

  TheFunction->getBasicBlockList().push_back(ContinBB);
  Builder.SetInsertPoint(ContinBB);

  // Return a default value of 0.0.
  return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

Value *SyncExprAST::codegen() {
  // Create a sync region for the local function or task scope, if necessary.
  if (!TaskScopeSyncRegion)
    TaskScopeSyncRegion = CreateSyncRegion(*TheModule);
  Value *SyncRegion = TaskScopeSyncRegion;
  Function *TheFunction = Builder.GetInsertBlock()->getParent();

  // Create a continuation block for the sync.
  BasicBlock *SyncContinueBB = BasicBlock::Create(TheContext, "sync.continue",
                                                  TheFunction);

  // Create the sync, and set the insert point to the continue block.
  Builder.CreateSync(SyncContinueBB, SyncRegion);
  Builder.SetInsertPoint(SyncContinueBB);

  // Return a default value of 0.0.
  return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

static std::vector<Metadata *> GetTapirLoopMetadata() {
  std::string TapirLoopSpawningStrategy = "tapir.loop.spawn.strategy";
  const int32_t DACLoopSpawning = 1;
  std::vector<Metadata *> Result;

  // Add the DAC loop-spawning strategy for Tapir loops.
  Result.push_back(MDNode::get(TheContext,
                               { MDString::get(TheContext,
                                               TapirLoopSpawningStrategy),
                                 ConstantAsMetadata::get(
                                     Builder.getInt32(DACLoopSpawning)) }));

  return Result;
}

// Output parfor-loop as:
//   sr = call token @llvm.syncregion.start
//   ...
//   start = startexpr
//   br pcond
// pcond:
//   variable = phi [start, loopheader], [nextvar, loopend]
//   endcond = endexpr
//   br endcond, ploop, afterloop
// ploop:
//   detach within sr, ploop.bodyentry, ploop.continue
// ploop.bodyentry:
//   var = alloca double
//   store variable -> var
//   ...
//   bodyexpr
//   ...
//   reattach within sr, ploop.continue
// ploop.continue:
//   step = stepexpr
//   nextvar = variable + step
//   br cond
// afterloop:
//   sync within sr, aftersync
// aftersync:
Value *ParForExprAST::codegen() {
#define BUG true
#if BUG
  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca =
    CreateTaskEntryBlockAlloca(VarName, Type::getInt64Ty(TheContext));
#endif // BUG

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->codegen();
  if (!StartVal)
    return nullptr;

#if BUG
  // Store the value into the alloca.
  Builder.CreateStore(StartVal, Alloca);
#endif // BUG

  // Make the new basic block for the loop header, inserting after current
  // block.
  Function *TheFunction = Builder.GetInsertBlock()->getParent();
  BasicBlock *PreheaderBB = Builder.GetInsertBlock();
  BasicBlock *CondBB = BasicBlock::Create(TheContext, "pcond", TheFunction);
  BasicBlock *LoopBB = BasicBlock::Create(TheContext, "ploop", TheFunction);
  BasicBlock *AfterBB = BasicBlock::Create(TheContext, "afterloop");

  // Create a sync region just for the loop, so we can sync the loop iterations
  // separately from other computation.
  Value *SyncRegion = CreateSyncRegion(*TheFunction->getParent());

  // Insert an explicit fall through from the current block to the CondBB.
  Builder.CreateBr(CondBB);

  // Start insertion in CondBB.
  Builder.SetInsertPoint(CondBB);

#if !BUG
  // Start the PHI node with an entry for Start.
  PHINode *Variable =
      Builder.CreatePHI(Type::getInt64Ty(TheContext), 2, VarName);
  Variable->addIncoming(StartVal, PreheaderBB);
#endif // !BUG

  // Within the parallel loop, we use new different copies of the variable.
  // Save any existing variables that are shadowed.
  Value *OldVal = NamedValues[VarName];
#if BUG
  NamedValues[VarName] = Alloca;
#else
  // For the end condition, use the PHI node as the variable VarName.
  NamedValues[VarName] = Variable;
#endif

  // If the end is a binary expression, force it to produce an integer result.
  End->setIntegerRes();
  // Compute the end condition.
  Value *EndCond = End->codegen();
  if (!EndCond)
    return nullptr;
  if (!EndCond->getType()->isIntegerTy())
    EndCond = Builder.CreateFPToSI(EndCond, Type::getInt64Ty(TheContext));

  // Convert condition to a bool by comparing non-equal to 0.
  EndCond = Builder.CreateICmpNE(
      EndCond, ConstantInt::get(TheContext, APSInt::get(0)), "loopcond");

  // Insert the conditional branch to either LoopBB or AfterBB.
  Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

  // Start insertion in LoopBB.
  Builder.SetInsertPoint(LoopBB);

  // Create a block for detaching the loop body and a block for the continuation
  // of the detach.
  BasicBlock *DetachBB =
    BasicBlock::Create(TheContext, "ploop.bodyentry", TheFunction);
  BasicBlock *ContinueBB =
    BasicBlock::Create(TheContext, "ploop.continue");

  // Insert a detach to spawn the loop body.
  Builder.CreateDetach(DetachBB, ContinueBB, SyncRegion);
  Builder.SetInsertPoint(DetachBB);

  // Emit the spawned loop body.
  {
    // Create a nested task scope corresponding to the loop body.
    TaskScopeRAII TaskScope(DetachBB);

#if !BUG
    // To avoid races, within the parallel loop's body, the variable is stored
    // in a task-local allocation. Create an alloca in the task's entry block
    // for this version of the variable.
    AllocaInst *VarAlloca =
      CreateTaskEntryBlockAlloca(VarName, Type::getInt64Ty(TheContext));
    // Store the value into the alloca.
    Builder.CreateStore(Variable, VarAlloca);
    NamedValues[VarName] = VarAlloca;
#endif // !BUG

    // Emit the body of the loop.  This, like any other expr, can change the
    // current BB.  Note that we ignore the value computed by the body, but
    // don't allow an error.
    if (!Body->codegen())
      return nullptr;

    // Emit the reattach to terminate the task containing the body of the
    // parallel loop.
    Builder.CreateReattach(ContinueBB, SyncRegion);
  }

  // Emit the continue block of the detach.
  TheFunction->getBasicBlockList().push_back(ContinueBB);

  // Set the insertion point to the continue block of the detach.
  Builder.SetInsertPoint(ContinueBB);

  // Emit the step value.
  Value *StepVal = nullptr;
  if (Step) {
    StepVal = Step->codegen();
    if (!StepVal)
      return nullptr;
  } else {
    // If not specified, use 1.
    StepVal = ConstantInt::get(TheContext, APSInt::get(1));
  }
#if BUG
  Value *CurVar = Builder.CreateLoad(Alloca, VarName.c_str());
  Value *NextVar = Builder.CreateAdd(CurVar, StepVal, "nextvar");
  Builder.CreateStore(NextVar, Alloca);
#else
  Value *NextVar = Builder.CreateAdd(Variable, StepVal, "nextvar");
#endif // BUG

  // Insert a back edge to CondBB
  BranchInst *BackEdge = Builder.CreateBr(CondBB);

  // Emit loop metadata
  std::vector<Metadata *> LoopMetadata = GetTapirLoopMetadata();
  if (!LoopMetadata.empty()) {
    auto TempNode = MDNode::getTemporary(TheContext, None);
    LoopMetadata.insert(LoopMetadata.begin(), TempNode.get());
    auto LoopID = MDNode::get(TheContext, LoopMetadata);
    LoopID->replaceOperandWith(0, LoopID);
    BackEdge->setMetadata(LLVMContext::MD_loop, LoopID);
  }

#if !BUG
  // Add a new entry to the PHI node for the backedge.
  Variable->addIncoming(NextVar, ContinueBB);
#endif // !BUG

  // Emit the "after loop" block.
  TheFunction->getBasicBlockList().push_back(AfterBB);

  // Any new code will be inserted in AfterBB.
  Builder.SetInsertPoint(AfterBB);

  // Create the "after loop" block and insert it.
  BasicBlock *AfterSync =
      BasicBlock::Create(TheContext, "aftersync", TheFunction);

  // Insert a sync for the loop.
  Builder.CreateSync(AfterSync, SyncRegion);
  Builder.SetInsertPoint(AfterSync);

  // Restore the unshadowed variable.
  if (OldVal)
    NamedValues[VarName] = OldVal;
  else
    NamedValues.erase(VarName);

  // parfor expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(TheContext));
}

Function *PrototypeAST::codegen() {
  // Make the function type:  double(double,double) etc.
  std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(TheContext));
  FunctionType *FT =
      FunctionType::get(Type::getDoubleTy(TheContext), Doubles, false);

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);

  return F;
}

Function *FunctionAST::codegen() {
  // Transfer ownership of the prototype to the FunctionProtos map, but keep a
  // reference to it for use below.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = getFunction(P.getName());
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (P.isBinaryOp())
    BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(TheContext, "entry", TheFunction);
  Builder.SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  for (auto &Arg : TheFunction->args()) {
    // Create an alloca for this variable.
    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, Arg.getName());

    // Store the initial value into the alloca.
    Builder.CreateStore(&Arg, Alloca);

    // Add arguments to variable symbol table.
    NamedValues[Arg.getName()] = Alloca;
  }

  TaskScopeRAII TaskScope(BB);
  if (Value *RetVal = Body->codegen()) {
    // Finish off the function.
    if (RetVal->getType()->isIntegerTy())
      RetVal = Builder.CreateSIToFP(RetVal, Type::getDoubleTy(TheContext));
    Builder.CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    // Mark the function for race-detection
    if (RunCilksan)
      TheFunction->addFnAttr(Attribute::SanitizeCilk);

    // Run the optimizer on the function.
    TheFPM->run(*TheFunction);
    TheMPM->run(*TheModule.get());

    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  if (P.isBinaryOp())
    BinopPrecedence.erase(P.getOperatorName());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static void AddTapirLoweringPasses() {
  // First, handle Tapir loops.  Loops are handled by first canonicalizing their
  // representation and then performing LoopSpawning to ensure that iterations
  // are spawned efficiently in parallel.

  if (Optimize) {
    // Start by simplifying the loops.
    TheMPM->add(createLoopInstSimplifyPass());
    TheMPM->add(createLoopSimplifyCFGPass());
    // Hoist loop invariants
    TheMPM->add(createLICMPass());
    // Cleanup the CFG and instructions
    TheMPM->add(createCFGSimplificationPass());
    TheMPM->add(createInstructionCombiningPass());
    // Re-rotate loops in all our loop nests.
    TheMPM->add(createLoopRotatePass(-1));
    // Simplify the loop induction variables.
    TheMPM->add(createIndVarSimplifyPass());

    // Transform Tapir loops to ensure that iterations are spawned efficiently
    // in parallel.
    if (TheTapirTarget != TapirTargetID::None)
      TheMPM->add(createLoopSpawningTIPass());
    // The LoopSpawning pass may leave cruft around.  Clean it up.
    TheMPM->add(createCFGSimplificationPass());
  }

  // Second, lower Tapir constructs in general to some parallel runtime system,
  // as specified in TargetLibraryInfo.

  // Add pass to lower Tapir to the target runtime.
  if (TheTapirTarget != TapirTargetID::None)
    TheMPM->add(createLowerTapirToTargetPass());
  // Perform some cleanup after the lowering pass.
  TheMPM->add(createCFGSimplificationPass());
}

static void InitializeModuleAndPassManager() {
  // Open a new module.
  TheModule = llvm::make_unique<Module>("my cool jit", TheContext);

  // Set the target triple to match the system.
  auto SysTargetTriple = sys::getDefaultTargetTriple();
  TheModule->setTargetTriple(SysTargetTriple);
  // Set an appropriate data layout
  TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

  // Create a new pass manager attached to it.
  TheMPM = llvm::make_unique<legacy::PassManager>();
  TheFPM = llvm::make_unique<legacy::FunctionPassManager>(TheModule.get());

  // Create TargetLibraryInfo for setting the target of Tapir lowering.
  Triple TargetTriple(TheModule->getTargetTriple());
  TargetLibraryInfoImpl TLII(TargetTriple);

  // Set the target for Tapir lowering to the Cilk runtime system.
  TLII.setTapirTarget(TheTapirTarget);

  // Add the TargetLibraryInfo to the pass manager.
  TheMPM->add(new TargetLibraryInfoWrapperPass(TLII));

  if (Optimize) {
    // Promote allocas to registers.
    TheFPM->add(createPromoteMemoryToRegisterPass());
    // Do simple "peephole" optimizations and bit-twiddling optzns.
    TheFPM->add(createInstructionCombiningPass());
    // Reassociate expressions.
    TheFPM->add(createReassociatePass());
    // Eliminate Common SubExpressions.
    TheFPM->add(createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    TheFPM->add(createCFGSimplificationPass());

    TheFPM->doInitialization();
  }

  // If requested, run the CilkSanitizer pass.
  if (RunCilksan)
    TheMPM->add(createCilkSanitizerLegacyPass(/*JitMode*/true,
                                              /*CallsMayThrow*/false));

  // Add Tapir lowering passes.
  AddTapirLoweringPasses();
}

static void HandleDefinition() {
  if (auto FnAST = ParseDefinition()) {
    if (auto *FnIR = FnAST->codegen()) {
      fprintf(stderr, "Read function definition:");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      auto K = TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();

      if (RunCilksan) {
        // Run the CSI constructor
        auto ExprSymbol = TheJIT->findSymbolInModule(K, "csirt.unit_ctor");
        assert(ExprSymbol && "Function not found");
        void (*CSICtor)() =
          (void (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
        CSICtor();
      }
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern() {
  if (auto ProtoAST = ParseExtern()) {
    if (auto *FnIR = ProtoAST->codegen()) {
      fprintf(stderr, "Read extern: ");
      FnIR->print(errs());
      fprintf(stderr, "\n");
      FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (auto FnAST = ParseTopLevelExpr()) {
    if (FnAST->codegen()) {
      // JIT the module containing the anonymous expression, keeping a handle so
      // we can free it later.
      auto H = TheJIT->addModule(std::move(TheModule));
      InitializeModuleAndPassManager();

      // Search the JIT for the __anon_expr symbol.
      auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
      assert(ExprSymbol && "Function not found");

      if (RunCilksan) {
        // Run the CSI constructor
        auto ExprSymbol = TheJIT->findSymbolInModule(H, "csirt.unit_ctor");
        assert(ExprSymbol && "Function not found");
        void (*CSICtor)() =
          (void (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
        CSICtor();
      }

      std::unique_ptr<Timer> T =
        llvm::make_unique<Timer>("__anon_expr", "Top-level expression");
      // Get the symbol's address and cast it to the right type (takes no
      // arguments, returns a double) so we can call it as a native function.
      double (*FP)() = (double (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
      T->startTimer();
      double Result = FP();
      T->stopTimer();
      fprintf(stderr, "Evaluated to %f\n", Result);

      // Delete the anonymous expression module from the JIT.
      TheJIT->removeModule(H);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    switch (CurTok) {
    case tok_eof:
      return;
    case ';': // ignore top-level semicolons.
      getNextToken();
      break;
    case tok_def:
      HandleDefinition();
      break;
    case tok_extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
    fprintf(stderr, "ready> ");
  }
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

static int usage(char *argv[]) {
  errs() << "Usage: " << argv[0]
         << " --lower-tapir-to {cilk|none}"
         << " [--run-cilksan]"
         << "\n";
  return 1;
}

int main(int argc, char *argv[]) {
  // Set the default Tapir target to be Cilk.
  TheTapirTarget = TapirTargetID::Cilk;

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--lower-tapir-to") {
      std::string targetStr = std::string(argv[++i]);
      if (targetStr == "cilk") {
        TheTapirTarget = TapirTargetID::Cilk;
      } else if (targetStr == "none") {
        TheTapirTarget = TapirTargetID::None;
      } else {
        return usage(argv);
      }
    } else if (std::string(argv[i]) == "--run-cilksan") {
      RunCilksan = true;
    } else if (std::string(argv[i]) == "-O0") {
      Optimize = false;
    } else if ((std::string(argv[i]) == "-O1") ||
               (std::string(argv[i]) == "-O2") ||
               (std::string(argv[i]) == "-O3")) {
      Optimize = true;
    } else {
      return usage(argv);
    }
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['*'] = 40; // highest.

  // Prime the first token.
  fprintf(stderr, "ready> ");
  getNextToken();

  TheJIT = llvm::make_unique<KaleidoscopeJIT>();

  InitializeModuleAndPassManager();

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
