#include "llvm/Analysis/Passes.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include <cctype>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
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
  tok_def = -2, tok_extern = -3,

  // primary
  tok_identifier = -4, tok_number = -5,

  // control
  tok_if = -6, tok_then = -7, tok_else = -8,
  tok_for = -9, tok_in = -10,

  // operators
  tok_binary = -11, tok_unary = -12,

  // var definition
  tok_var = -13
};

static std::string IdentifierStr;  // Filled in if tok_identifier
static double NumVal;              // Filled in if tok_number

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

    if (IdentifierStr == "def") return tok_def;
    if (IdentifierStr == "extern") return tok_extern;
    if (IdentifierStr == "if") return tok_if;
    if (IdentifierStr == "then") return tok_then;
    if (IdentifierStr == "else") return tok_else;
    if (IdentifierStr == "for") return tok_for;
    if (IdentifierStr == "in") return tok_in;
    if (IdentifierStr == "binary") return tok_binary;
    if (IdentifierStr == "unary") return tok_unary;
    if (IdentifierStr == "var") return tok_var;
    return tok_identifier;
  }

  if (isdigit(LastChar) || LastChar == '.') {   // Number: [0-9.]+
    std::string NumStr;
    do {
      NumStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    NumVal = strtod(NumStr.c_str(), nullptr);
    return tok_number;
  }

  if (LastChar == '#') {
    // Comment until end of line.
    do LastChar = getchar();
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

class IRGenContext;

/// ExprAST - Base class for all expression nodes.
struct ExprAST {
  virtual ~ExprAST() {}
  virtual Value *IRGen(IRGenContext &C) const = 0;
};

/// NumberExprAST - Expression class for numeric literals like "1.0".
struct NumberExprAST : public ExprAST {
  NumberExprAST(double Val) : Val(Val) {}
  Value *IRGen(IRGenContext &C) const override;

  double Val;
};

/// VariableExprAST - Expression class for referencing a variable, like "a".
struct VariableExprAST : public ExprAST {
  VariableExprAST(std::string Name) : Name(std::move(Name)) {}
  Value *IRGen(IRGenContext &C) const override;

  std::string Name;
};

/// UnaryExprAST - Expression class for a unary operator.
struct UnaryExprAST : public ExprAST {
  UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
    : Opcode(std::move(Opcode)), Operand(std::move(Operand)) {}

  Value *IRGen(IRGenContext &C) const override;

  char Opcode;
  std::unique_ptr<ExprAST> Operand;
};

/// BinaryExprAST - Expression class for a binary operator.
struct BinaryExprAST : public ExprAST {
  BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
                std::unique_ptr<ExprAST> RHS)
    : Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

  Value *IRGen(IRGenContext &C) const override;

  char Op;
  std::unique_ptr<ExprAST> LHS, RHS;
};

/// CallExprAST - Expression class for function calls.
struct CallExprAST : public ExprAST {
  CallExprAST(std::string CalleeName,
              std::vector<std::unique_ptr<ExprAST>> Args)
    : CalleeName(std::move(CalleeName)), Args(std::move(Args)) {}

  Value *IRGen(IRGenContext &C) const override;

  std::string CalleeName;
  std::vector<std::unique_ptr<ExprAST>> Args;
};

/// IfExprAST - Expression class for if/then/else.
struct IfExprAST : public ExprAST {
  IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
            std::unique_ptr<ExprAST> Else)
    : Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else)) {}
  Value *IRGen(IRGenContext &C) const override;

  std::unique_ptr<ExprAST> Cond, Then, Else;
};

/// ForExprAST - Expression class for for/in.
struct ForExprAST : public ExprAST {
  ForExprAST(std::string VarName, std::unique_ptr<ExprAST> Start,
             std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
             std::unique_ptr<ExprAST> Body)
    : VarName(std::move(VarName)), Start(std::move(Start)), End(std::move(End)),
      Step(std::move(Step)), Body(std::move(Body)) {}

  Value *IRGen(IRGenContext &C) const override;

  std::string VarName;
  std::unique_ptr<ExprAST> Start, End, Step, Body;
};

/// VarExprAST - Expression class for var/in
struct VarExprAST : public ExprAST {
  typedef std::pair<std::string, std::unique_ptr<ExprAST>> Binding;
  typedef std::vector<Binding> BindingList;

  VarExprAST(BindingList VarBindings, std::unique_ptr<ExprAST> Body)
    : VarBindings(std::move(VarBindings)), Body(std::move(Body)) {}

  Value *IRGen(IRGenContext &C) const override;

  BindingList VarBindings;
  std::unique_ptr<ExprAST> Body;
};

/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its argument names as well as if it is an operator.
struct PrototypeAST {
  PrototypeAST(std::string Name, std::vector<std::string> Args,
               bool IsOperator = false, unsigned Precedence = 0)
    : Name(std::move(Name)), Args(std::move(Args)), IsOperator(IsOperator),
      Precedence(Precedence) {}

  Function *IRGen(IRGenContext &C) const;
  void CreateArgumentAllocas(Function *F, IRGenContext &C);

  bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
  bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

  char getOperatorName() const {
    assert(isUnaryOp() || isBinaryOp());
    return Name[Name.size()-1];
  }

  std::string Name;
  std::vector<std::string> Args;
  bool IsOperator;
  unsigned Precedence;  // Precedence if a binary op.
};

/// FunctionAST - This class represents a function definition itself.
struct FunctionAST {
  FunctionAST(std::unique_ptr<PrototypeAST> Proto,
              std::unique_ptr<ExprAST> Body)
    : Proto(std::move(Proto)), Body(std::move(Body)) {}

  Function *IRGen(IRGenContext &C) const;

  std::unique_ptr<PrototypeAST> Proto;
  std::unique_ptr<ExprAST> Body;
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
/// token the parser is looking at.  getNextToken reads another token from the
/// lexer and updates CurTok with its results.
static int CurTok;
static int getNextToken() {
  return CurTok = gettok();
}

/// BinopPrecedence - This holds the precedence for each binary operator that is
/// defined.
static std::map<char, int> BinopPrecedence;

/// GetTokPrecedence - Get the precedence of the pending binary operator token.
static int GetTokPrecedence() {
  if (!isascii(CurTok))
    return -1;

  // Make sure it's a declared binop.
  int TokPrec = BinopPrecedence[CurTok];
  if (TokPrec <= 0) return -1;
  return TokPrec;
}

template <typename T>
std::unique_ptr<T> ErrorU(const std::string &Str) {
  std::cerr << "Error: " << Str << "\n";
  return nullptr;
}

template <typename T>
T* ErrorP(const std::string &Str) {
  std::cerr << "Error: " << Str << "\n";
  return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

/// identifierexpr
///   ::= identifier
///   ::= identifier '(' expression* ')'
static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
  std::string IdName = IdentifierStr;

  getNextToken();  // eat identifier.

  if (CurTok != '(') // Simple variable ref.
    return llvm::make_unique<VariableExprAST>(IdName);

  // Call.
  getNextToken();  // eat (
  std::vector<std::unique_ptr<ExprAST>> Args;
  if (CurTok != ')') {
    while (1) {
      auto Arg = ParseExpression();
      if (!Arg) return nullptr;
      Args.push_back(std::move(Arg));

      if (CurTok == ')') break;

      if (CurTok != ',')
        return ErrorU<CallExprAST>("Expected ')' or ',' in argument list");
      getNextToken();
    }
  }

  // Eat the ')'.
  getNextToken();

  return llvm::make_unique<CallExprAST>(IdName, std::move(Args));
}

/// numberexpr ::= number
static std::unique_ptr<NumberExprAST> ParseNumberExpr() {
  auto Result = llvm::make_unique<NumberExprAST>(NumVal);
  getNextToken(); // consume the number
  return Result;
}

/// parenexpr ::= '(' expression ')'
static std::unique_ptr<ExprAST> ParseParenExpr() {
  getNextToken();  // eat (.
  auto V = ParseExpression();
  if (!V)
    return nullptr;

  if (CurTok != ')')
    return ErrorU<ExprAST>("expected ')'");
  getNextToken();  // eat ).
  return V;
}

/// ifexpr ::= 'if' expression 'then' expression 'else' expression
static std::unique_ptr<ExprAST> ParseIfExpr() {
  getNextToken();  // eat the if.

  // condition.
  auto Cond = ParseExpression();
  if (!Cond)
    return nullptr;

  if (CurTok != tok_then)
    return ErrorU<ExprAST>("expected then");
  getNextToken();  // eat the then

  auto Then = ParseExpression();
  if (!Then)
    return nullptr;

  if (CurTok != tok_else)
    return ErrorU<ExprAST>("expected else");

  getNextToken();

  auto Else = ParseExpression();
  if (!Else)
    return nullptr;

  return llvm::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
                                      std::move(Else));
}

/// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
static std::unique_ptr<ForExprAST> ParseForExpr() {
  getNextToken();  // eat the for.

  if (CurTok != tok_identifier)
    return ErrorU<ForExprAST>("expected identifier after for");

  std::string IdName = IdentifierStr;
  getNextToken();  // eat identifier.

  if (CurTok != '=')
    return ErrorU<ForExprAST>("expected '=' after for");
  getNextToken();  // eat '='.

  auto Start = ParseExpression();
  if (!Start)
    return nullptr;
  if (CurTok != ',')
    return ErrorU<ForExprAST>("expected ',' after for start value");
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
    return ErrorU<ForExprAST>("expected 'in' after for");
  getNextToken();  // eat 'in'.

  auto Body = ParseExpression();
  if (Body)
    return nullptr;

  return llvm::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
                                       std::move(Step), std::move(Body));
}

/// varexpr ::= 'var' identifier ('=' expression)?
//                    (',' identifier ('=' expression)?)* 'in' expression
static std::unique_ptr<VarExprAST> ParseVarExpr() {
  getNextToken();  // eat the var.

  VarExprAST::BindingList VarBindings;

  // At least one variable name is required.
  if (CurTok != tok_identifier)
    return ErrorU<VarExprAST>("expected identifier after var");

  while (1) {
    std::string Name = IdentifierStr;
    getNextToken();  // eat identifier.

    // Read the optional initializer.
    std::unique_ptr<ExprAST> Init;
    if (CurTok == '=') {
      getNextToken(); // eat the '='.

      Init = ParseExpression();
      if (!Init)
        return nullptr;
    }

    VarBindings.push_back(VarExprAST::Binding(Name, std::move(Init)));

    // End of var list, exit loop.
    if (CurTok != ',') break;
    getNextToken(); // eat the ','.

    if (CurTok != tok_identifier)
      return ErrorU<VarExprAST>("expected identifier list after var");
  }

  // At this point, we have to have 'in'.
  if (CurTok != tok_in)
    return ErrorU<VarExprAST>("expected 'in' keyword after 'var'");
  getNextToken();  // eat 'in'.

  auto Body = ParseExpression();
  if (!Body)
    return nullptr;

  return llvm::make_unique<VarExprAST>(std::move(VarBindings), std::move(Body));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
///   ::= ifexpr
///   ::= forexpr
///   ::= varexpr
static std::unique_ptr<ExprAST> ParsePrimary() {
  switch (CurTok) {
  default: return ErrorU<ExprAST>("unknown token when expecting an expression");
  case tok_identifier: return ParseIdentifierExpr();
  case tok_number:     return ParseNumberExpr();
  case '(':            return ParseParenExpr();
  case tok_if:         return ParseIfExpr();
  case tok_for:        return ParseForExpr();
  case tok_var:        return ParseVarExpr();
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
                                              std::unique_ptr<ExprAST> LHS) {
  // If this is a binop, find its precedence.
  while (1) {
    int TokPrec = GetTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (TokPrec < ExprPrec)
      return LHS;

    // Okay, we know this is a binop.
    int BinOp = CurTok;
    getNextToken();  // eat binop

    // Parse the unary expression after the binary operator.
    auto RHS = ParseUnary();
    if (!RHS)
      return nullptr;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    int NextPrec = GetTokPrecedence();
    if (TokPrec < NextPrec) {
      RHS = ParseBinOpRHS(TokPrec+1, std::move(RHS));
      if (!RHS)
        return nullptr;
    }

    // Merge LHS/RHS.
    LHS = llvm::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
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
    return ErrorU<PrototypeAST>("Expected function name in prototype");
  case tok_identifier:
    FnName = IdentifierStr;
    Kind = 0;
    getNextToken();
    break;
  case tok_unary:
    getNextToken();
    if (!isascii(CurTok))
      return ErrorU<PrototypeAST>("Expected unary operator");
    FnName = "unary";
    FnName += (char)CurTok;
    Kind = 1;
    getNextToken();
    break;
  case tok_binary:
    getNextToken();
    if (!isascii(CurTok))
      return ErrorU<PrototypeAST>("Expected binary operator");
    FnName = "binary";
    FnName += (char)CurTok;
    Kind = 2;
    getNextToken();

    // Read the precedence if present.
    if (CurTok == tok_number) {
      if (NumVal < 1 || NumVal > 100)
        return ErrorU<PrototypeAST>("Invalid precedecnce: must be 1..100");
      BinaryPrecedence = (unsigned)NumVal;
      getNextToken();
    }
    break;
  }

  if (CurTok != '(')
    return ErrorU<PrototypeAST>("Expected '(' in prototype");

  std::vector<std::string> ArgNames;
  while (getNextToken() == tok_identifier)
    ArgNames.push_back(IdentifierStr);
  if (CurTok != ')')
    return ErrorU<PrototypeAST>("Expected ')' in prototype");

  // success.
  getNextToken();  // eat ')'.

  // Verify right number of names for operator.
  if (Kind && ArgNames.size() != Kind)
    return ErrorU<PrototypeAST>("Invalid number of operands for operator");

  return llvm::make_unique<PrototypeAST>(FnName, std::move(ArgNames), Kind != 0,
                                         BinaryPrecedence);
}

/// definition ::= 'def' prototype expression
static std::unique_ptr<FunctionAST> ParseDefinition() {
  getNextToken();  // eat def.
  auto Proto = ParsePrototype();
  if (!Proto)
    return nullptr;

  if (auto Body = ParseExpression())
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(Body));
  return nullptr;
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
  if (auto E = ParseExpression()) {
    // Make an anonymous proto.
    auto Proto =
      llvm::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
    return llvm::make_unique<FunctionAST>(std::move(Proto), std::move(E));
  }
  return nullptr;
}

/// external ::= 'extern' prototype
static std::unique_ptr<PrototypeAST> ParseExtern() {
  getNextToken();  // eat extern.
  return ParsePrototype();
}

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

// FIXME: Obviously we can do better than this
std::string GenerateUniqueName(const std::string &Root) {
  static int i = 0;
  std::ostringstream NameStream;
  NameStream << Root << ++i;
  return NameStream.str();
}

std::string MakeLegalFunctionName(std::string Name)
{
  std::string NewName;
  assert(!Name.empty() && "Base name must not be empty");

  // Start with what we have
  NewName = Name;

  // Look for a numberic first character
  if (NewName.find_first_of("0123456789") == 0) {
    NewName.insert(0, 1, 'n');
  }

  // Replace illegal characters with their ASCII equivalent
  std::string legal_elements = "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  size_t pos;
  while ((pos = NewName.find_first_not_of(legal_elements)) != std::string::npos) {
    std::ostringstream NumStream;
    NumStream << (int)NewName.at(pos);
    NewName = NewName.replace(pos, 1, NumStream.str());
  }

  return NewName;
}

class SessionContext {
public:
  SessionContext(LLVMContext &C)
    : Context(C), TM(EngineBuilder().selectTarget()) {}
  LLVMContext& getLLVMContext() const { return Context; }
  TargetMachine& getTarget() { return *TM; }
  void addPrototypeAST(std::unique_ptr<PrototypeAST> P);
  PrototypeAST* getPrototypeAST(const std::string &Name);
private:
  typedef std::map<std::string, std::unique_ptr<PrototypeAST>> PrototypeMap;

  LLVMContext &Context;
  std::unique_ptr<TargetMachine> TM;

  PrototypeMap Prototypes;
};

void SessionContext::addPrototypeAST(std::unique_ptr<PrototypeAST> P) {
  Prototypes[P->Name] = std::move(P);
}

PrototypeAST* SessionContext::getPrototypeAST(const std::string &Name) {
  PrototypeMap::iterator I = Prototypes.find(Name);
  if (I != Prototypes.end())
    return I->second.get();
  return nullptr;
}

class IRGenContext {
public:

  IRGenContext(SessionContext &S)
    : Session(S),
      M(new Module(GenerateUniqueName("jit_module_"),
                   Session.getLLVMContext())),
      Builder(Session.getLLVMContext()) {
    M->setDataLayout(Session.getTarget().createDataLayout());
  }

  SessionContext& getSession() { return Session; }
  Module& getM() const { return *M; }
  std::unique_ptr<Module> takeM() { return std::move(M); }
  IRBuilder<>& getBuilder() { return Builder; }
  LLVMContext& getLLVMContext() { return Session.getLLVMContext(); }
  Function* getPrototype(const std::string &Name);

  std::map<std::string, AllocaInst*> NamedValues;
private:
  SessionContext &Session;
  std::unique_ptr<Module> M;
  IRBuilder<> Builder;
};

Function* IRGenContext::getPrototype(const std::string &Name) {
  if (Function *ExistingProto = M->getFunction(Name))
    return ExistingProto;
  if (PrototypeAST *ProtoAST = Session.getPrototypeAST(Name))
    return ProtoAST->IRGen(*this);
  return nullptr;
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static AllocaInst *CreateEntryBlockAlloca(Function *TheFunction,
                                          const std::string &VarName) {
  IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                 TheFunction->getEntryBlock().begin());
  return TmpB.CreateAlloca(Type::getDoubleTy(getGlobalContext()), nullptr,
                           VarName.c_str());
}

Value *NumberExprAST::IRGen(IRGenContext &C) const {
  return ConstantFP::get(C.getLLVMContext(), APFloat(Val));
}

Value *VariableExprAST::IRGen(IRGenContext &C) const {
  // Look this variable up in the function.
  Value *V = C.NamedValues[Name];

  if (!V)
    return ErrorP<Value>("Unknown variable name '" + Name + "'");

  // Load the value.
  return C.getBuilder().CreateLoad(V, Name.c_str());
}

Value *UnaryExprAST::IRGen(IRGenContext &C) const {
  if (Value *OperandV = Operand->IRGen(C)) {
    std::string FnName = MakeLegalFunctionName(std::string("unary")+Opcode);
    if (Function *F = C.getPrototype(FnName))
      return C.getBuilder().CreateCall(F, OperandV, "unop");
    return ErrorP<Value>("Unknown unary operator");
  }

  // Could not codegen operand - return null.
  return nullptr;
}

Value *BinaryExprAST::IRGen(IRGenContext &C) const {
  // Special case '=' because we don't want to emit the LHS as an expression.
  if (Op == '=') {
    // Assignment requires the LHS to be an identifier.
    auto &LHSVar = static_cast<VariableExprAST &>(*LHS);
    // Codegen the RHS.
    Value *Val = RHS->IRGen(C);
    if (!Val) return nullptr;

    // Look up the name.
    if (auto Variable = C.NamedValues[LHSVar.Name]) {
      C.getBuilder().CreateStore(Val, Variable);
      return Val;
    }
    return ErrorP<Value>("Unknown variable name");
  }

  Value *L = LHS->IRGen(C);
  Value *R = RHS->IRGen(C);
  if (!L || !R) return nullptr;

  switch (Op) {
  case '+': return C.getBuilder().CreateFAdd(L, R, "addtmp");
  case '-': return C.getBuilder().CreateFSub(L, R, "subtmp");
  case '*': return C.getBuilder().CreateFMul(L, R, "multmp");
  case '/': return C.getBuilder().CreateFDiv(L, R, "divtmp");
  case '<':
    L = C.getBuilder().CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return C.getBuilder().CreateUIToFP(L, Type::getDoubleTy(getGlobalContext()),
                                "booltmp");
  default: break;
  }

  // If it wasn't a builtin binary operator, it must be a user defined one. Emit
  // a call to it.
  std::string FnName = MakeLegalFunctionName(std::string("binary")+Op);
  if (Function *F = C.getPrototype(FnName)) {
    Value *Ops[] = { L, R };
    return C.getBuilder().CreateCall(F, Ops, "binop");
  }

  return ErrorP<Value>("Unknown binary operator");
}

Value *CallExprAST::IRGen(IRGenContext &C) const {
  // Look up the name in the global module table.
  if (auto CalleeF = C.getPrototype(CalleeName)) {
    // If argument mismatch error.
    if (CalleeF->arg_size() != Args.size())
      return ErrorP<Value>("Incorrect # arguments passed");

    std::vector<Value*> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      ArgsV.push_back(Args[i]->IRGen(C));
      if (!ArgsV.back()) return nullptr;
    }

    return C.getBuilder().CreateCall(CalleeF, ArgsV, "calltmp");
  }

  return ErrorP<Value>("Unknown function referenced");
}

Value *IfExprAST::IRGen(IRGenContext &C) const {
  Value *CondV = Cond->IRGen(C);
  if (!CondV) return nullptr;

  // Convert condition to a bool by comparing equal to 0.0.
  ConstantFP *FPZero =
    ConstantFP::get(C.getLLVMContext(), APFloat(0.0));
  CondV = C.getBuilder().CreateFCmpONE(CondV, FPZero, "ifcond");

  Function *TheFunction = C.getBuilder().GetInsertBlock()->getParent();

  // Create blocks for the then and else cases.  Insert the 'then' block at the
  // end of the function.
  BasicBlock *ThenBB = BasicBlock::Create(C.getLLVMContext(), "then", TheFunction);
  BasicBlock *ElseBB = BasicBlock::Create(C.getLLVMContext(), "else");
  BasicBlock *MergeBB = BasicBlock::Create(C.getLLVMContext(), "ifcont");

  C.getBuilder().CreateCondBr(CondV, ThenBB, ElseBB);

  // Emit then value.
  C.getBuilder().SetInsertPoint(ThenBB);

  Value *ThenV = Then->IRGen(C);
  if (!ThenV) return nullptr;

  C.getBuilder().CreateBr(MergeBB);
  // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
  ThenBB = C.getBuilder().GetInsertBlock();

  // Emit else block.
  TheFunction->getBasicBlockList().push_back(ElseBB);
  C.getBuilder().SetInsertPoint(ElseBB);

  Value *ElseV = Else->IRGen(C);
  if (!ElseV) return nullptr;

  C.getBuilder().CreateBr(MergeBB);
  // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
  ElseBB = C.getBuilder().GetInsertBlock();

  // Emit merge block.
  TheFunction->getBasicBlockList().push_back(MergeBB);
  C.getBuilder().SetInsertPoint(MergeBB);
  PHINode *PN = C.getBuilder().CreatePHI(Type::getDoubleTy(getGlobalContext()), 2,
                                  "iftmp");

  PN->addIncoming(ThenV, ThenBB);
  PN->addIncoming(ElseV, ElseBB);
  return PN;
}

Value *ForExprAST::IRGen(IRGenContext &C) const {
  // Output this as:
  //   var = alloca double
  //   ...
  //   start = startexpr
  //   store start -> var
  //   goto loop
  // loop:
  //   ...
  //   bodyexpr
  //   ...
  // loopend:
  //   step = stepexpr
  //   endcond = endexpr
  //
  //   curvar = load var
  //   nextvar = curvar + step
  //   store nextvar -> var
  //   br endcond, loop, endloop
  // outloop:

  Function *TheFunction = C.getBuilder().GetInsertBlock()->getParent();

  // Create an alloca for the variable in the entry block.
  AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

  // Emit the start code first, without 'variable' in scope.
  Value *StartVal = Start->IRGen(C);
  if (!StartVal) return nullptr;

  // Store the value into the alloca.
  C.getBuilder().CreateStore(StartVal, Alloca);

  // Make the new basic block for the loop header, inserting after current
  // block.
  BasicBlock *LoopBB = BasicBlock::Create(getGlobalContext(), "loop", TheFunction);

  // Insert an explicit fall through from the current block to the LoopBB.
  C.getBuilder().CreateBr(LoopBB);

  // Start insertion in LoopBB.
  C.getBuilder().SetInsertPoint(LoopBB);

  // Within the loop, the variable is defined equal to the PHI node.  If it
  // shadows an existing variable, we have to restore it, so save it now.
  AllocaInst *OldVal = C.NamedValues[VarName];
  C.NamedValues[VarName] = Alloca;

  // Emit the body of the loop.  This, like any other expr, can change the
  // current BB.  Note that we ignore the value computed by the body, but don't
  // allow an error.
  if (!Body->IRGen(C))
    return nullptr;

  // Emit the step value.
  Value *StepVal;
  if (Step) {
    StepVal = Step->IRGen(C);
    if (!StepVal) return nullptr;
  } else {
    // If not specified, use 1.0.
    StepVal = ConstantFP::get(getGlobalContext(), APFloat(1.0));
  }

  // Compute the end condition.
  Value *EndCond = End->IRGen(C);
  if (!EndCond) return nullptr;

  // Reload, increment, and restore the alloca.  This handles the case where
  // the body of the loop mutates the variable.
  Value *CurVar = C.getBuilder().CreateLoad(Alloca, VarName.c_str());
  Value *NextVar = C.getBuilder().CreateFAdd(CurVar, StepVal, "nextvar");
  C.getBuilder().CreateStore(NextVar, Alloca);

  // Convert condition to a bool by comparing equal to 0.0.
  EndCond = C.getBuilder().CreateFCmpONE(EndCond,
                              ConstantFP::get(getGlobalContext(), APFloat(0.0)),
                                  "loopcond");

  // Create the "after loop" block and insert it.
  BasicBlock *AfterBB = BasicBlock::Create(getGlobalContext(), "afterloop", TheFunction);

  // Insert the conditional branch into the end of LoopEndBB.
  C.getBuilder().CreateCondBr(EndCond, LoopBB, AfterBB);

  // Any new code will be inserted in AfterBB.
  C.getBuilder().SetInsertPoint(AfterBB);

  // Restore the unshadowed variable.
  if (OldVal)
    C.NamedValues[VarName] = OldVal;
  else
    C.NamedValues.erase(VarName);

  // for expr always returns 0.0.
  return Constant::getNullValue(Type::getDoubleTy(getGlobalContext()));
}

Value *VarExprAST::IRGen(IRGenContext &C) const {
  std::vector<AllocaInst *> OldBindings;

  Function *TheFunction = C.getBuilder().GetInsertBlock()->getParent();

  // Register all variables and emit their initializer.
  for (unsigned i = 0, e = VarBindings.size(); i != e; ++i) {
    auto &VarName = VarBindings[i].first;
    auto &Init = VarBindings[i].second;

    // Emit the initializer before adding the variable to scope, this prevents
    // the initializer from referencing the variable itself, and permits stuff
    // like this:
    //  var a = 1 in
    //    var a = a in ...   # refers to outer 'a'.
    Value *InitVal;
    if (Init) {
      InitVal = Init->IRGen(C);
      if (!InitVal) return nullptr;
    } else // If not specified, use 0.0.
      InitVal = ConstantFP::get(getGlobalContext(), APFloat(0.0));

    AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
    C.getBuilder().CreateStore(InitVal, Alloca);

    // Remember the old variable binding so that we can restore the binding when
    // we unrecurse.
    OldBindings.push_back(C.NamedValues[VarName]);

    // Remember this binding.
    C.NamedValues[VarName] = Alloca;
  }

  // Codegen the body, now that all vars are in scope.
  Value *BodyVal = Body->IRGen(C);
  if (!BodyVal) return nullptr;

  // Pop all our variables from scope.
  for (unsigned i = 0, e = VarBindings.size(); i != e; ++i)
    C.NamedValues[VarBindings[i].first] = OldBindings[i];

  // Return the body computation.
  return BodyVal;
}

Function *PrototypeAST::IRGen(IRGenContext &C) const {
  std::string FnName = MakeLegalFunctionName(Name);

  // Make the function type:  double(double,double) etc.
  std::vector<Type*> Doubles(Args.size(),
                             Type::getDoubleTy(getGlobalContext()));
  FunctionType *FT = FunctionType::get(Type::getDoubleTy(getGlobalContext()),
                                       Doubles, false);
  Function *F = Function::Create(FT, Function::ExternalLinkage, FnName,
                                 &C.getM());

  // If F conflicted, there was already something named 'FnName'.  If it has a
  // body, don't allow redefinition or reextern.
  if (F->getName() != FnName) {
    // Delete the one we just made and get the existing one.
    F->eraseFromParent();
    F = C.getM().getFunction(Name);

    // If F already has a body, reject this.
    if (!F->empty()) {
      ErrorP<Function>("redefinition of function");
      return nullptr;
    }

    // If F took a different number of args, reject.
    if (F->arg_size() != Args.size()) {
      ErrorP<Function>("redefinition of function with different # args");
      return nullptr;
    }
  }

  // Set names for all arguments.
  unsigned Idx = 0;
  for (Function::arg_iterator AI = F->arg_begin(); Idx != Args.size();
       ++AI, ++Idx)
    AI->setName(Args[Idx]);

  return F;
}

/// CreateArgumentAllocas - Create an alloca for each argument and register the
/// argument in the symbol table so that references to it will succeed.
void PrototypeAST::CreateArgumentAllocas(Function *F, IRGenContext &C) {
  Function::arg_iterator AI = F->arg_begin();
  for (unsigned Idx = 0, e = Args.size(); Idx != e; ++Idx, ++AI) {
    // Create an alloca for this variable.
    AllocaInst *Alloca = CreateEntryBlockAlloca(F, Args[Idx]);

    // Store the initial value into the alloca.
    C.getBuilder().CreateStore(&*AI, Alloca);

    // Add arguments to variable symbol table.
    C.NamedValues[Args[Idx]] = Alloca;
  }
}

Function *FunctionAST::IRGen(IRGenContext &C) const {
  C.NamedValues.clear();

  Function *TheFunction = Proto->IRGen(C);
  if (!TheFunction)
    return nullptr;

  // If this is an operator, install it.
  if (Proto->isBinaryOp())
    BinopPrecedence[Proto->getOperatorName()] = Proto->Precedence;

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(getGlobalContext(), "entry", TheFunction);
  C.getBuilder().SetInsertPoint(BB);

  // Add all arguments to the symbol table and create their allocas.
  Proto->CreateArgumentAllocas(TheFunction, C);

  if (Value *RetVal = Body->IRGen(C)) {
    // Finish off the function.
    C.getBuilder().CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  if (Proto->isBinaryOp())
    BinopPrecedence.erase(Proto->getOperatorName());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

static std::unique_ptr<llvm::Module> IRGen(SessionContext &S,
                                           const FunctionAST &F) {
  IRGenContext C(S);
  auto LF = F.IRGen(C);
  if (!LF)
    return nullptr;
#ifndef MINIMAL_STDERR_OUTPUT
  fprintf(stderr, "Read function definition:");
  LF->dump();
#endif
  return C.takeM();
}

template <typename T>
static std::vector<T> singletonSet(T t) {
  std::vector<T> Vec;
  Vec.push_back(std::move(t));
  return Vec;
}

class KaleidoscopeJIT {
public:
  typedef ObjectLinkingLayer<> ObjLayerT;
  typedef IRCompileLayer<ObjLayerT> CompileLayerT;
  typedef CompileLayerT::ModuleSetHandleT ModuleHandleT;

  KaleidoscopeJIT(SessionContext &Session)
      : DL(Session.getTarget().createDataLayout()),
        CompileLayer(ObjectLayer, SimpleCompiler(Session.getTarget())) {}

  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  ModuleHandleT addModule(std::unique_ptr<Module> M) {
    // We need a memory manager to allocate memory and resolve symbols for this
    // new module. Create one that resolves symbols by looking back into the
    // JIT.
    auto Resolver = createLambdaResolver(
                      [&](const std::string &Name) {
                        if (auto Sym = findSymbol(Name))
                          return RuntimeDyld::SymbolInfo(Sym.getAddress(),
                                                         Sym.getFlags());
                        return RuntimeDyld::SymbolInfo(nullptr);
                      },
                      [](const std::string &S) { return nullptr; }
                    );
    return CompileLayer.addModuleSet(singletonSet(std::move(M)),
                                     make_unique<SectionMemoryManager>(),
                                     std::move(Resolver));
  }

  void removeModule(ModuleHandleT H) { CompileLayer.removeModuleSet(H); }

  JITSymbol findSymbol(const std::string &Name) {
    return CompileLayer.findSymbol(Name, true);
  }

  JITSymbol findUnmangledSymbol(const std::string Name) {
    return findSymbol(mangle(Name));
  }

private:
  const DataLayout DL;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
};

static void HandleDefinition(SessionContext &S, KaleidoscopeJIT &J) {
  if (auto F = ParseDefinition()) {
    if (auto M = IRGen(S, *F)) {
      S.addPrototypeAST(llvm::make_unique<PrototypeAST>(*F->Proto));
      J.addModule(std::move(M));
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleExtern(SessionContext &S) {
  if (auto P = ParseExtern())
    S.addPrototypeAST(std::move(P));
  else {
    // Skip token for error recovery.
    getNextToken();
  }
}

static void HandleTopLevelExpression(SessionContext &S, KaleidoscopeJIT &J) {
  // Evaluate a top-level expression into an anonymous function.
  if (auto F = ParseTopLevelExpr()) {
    IRGenContext C(S);
    if (auto ExprFunc = F->IRGen(C)) {
#ifndef MINIMAL_STDERR_OUTPUT
      std::cerr << "Expression function:\n";
      ExprFunc->dump();
#endif
      // Add the CodeGen'd module to the JIT. Keep a handle to it: We can remove
      // this module as soon as we've executed Function ExprFunc.
      auto H = J.addModule(C.takeM());

      // Get the address of the JIT'd function in memory.
      auto ExprSymbol = J.findUnmangledSymbol("__anon_expr");

      // Cast it to the right type (takes no arguments, returns a double) so we
      // can call it as a native function.
      double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();
#ifdef MINIMAL_STDERR_OUTPUT
      FP();
#else
      std::cerr << "Evaluated to " << FP() << "\n";
#endif

      // Remove the function.
      J.removeModule(H);
    }
  } else {
    // Skip token for error recovery.
    getNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  SessionContext S(getGlobalContext());
  KaleidoscopeJIT J(S);

  while (1) {
    switch (CurTok) {
    case tok_eof:    return;
    case ';':        getNextToken(); continue;  // ignore top-level semicolons.
    case tok_def:    HandleDefinition(S, J); break;
    case tok_extern: HandleExtern(S); break;
    default:         HandleTopLevelExpression(S, J); break;
    }
#ifndef MINIMAL_STDERR_OUTPUT
    std::cerr << "ready> ";
#endif
  }
}

//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

/// putchard - putchar that takes a double and returns 0.
extern "C"
double putchard(double X) {
  putchar((char)X);
  return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C"
double printd(double X) {
  printf("%f", X);
  return 0;
}

extern "C"
double printlf() {
  printf("\n");
  return 0;
}

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//

int main() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();

  // Install standard binary operators.
  // 1 is lowest precedence.
  BinopPrecedence['='] = 2;
  BinopPrecedence['<'] = 10;
  BinopPrecedence['+'] = 20;
  BinopPrecedence['-'] = 20;
  BinopPrecedence['/'] = 40;
  BinopPrecedence['*'] = 40;  // highest.

  // Prime the first token.
#ifndef MINIMAL_STDERR_OUTPUT
  std::cerr << "ready> ";
#endif
  getNextToken();

  std::cerr << std::fixed;

  // Run the main "interpreter loop" now.
  MainLoop();

  return 0;
}
