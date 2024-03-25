#ifndef OBS_AST_H
#define OBS_AST_H

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <utility>
#include <vector>
#include <optional>

namespace obs {

struct VarType {
    std::vector<int64_t> shape ;
};

class ExprAST {
public:
    enum ExprASTKind {
        Expr_VarDecl,
        Expr_Return,
        Expr_Num,
        Expr_Literal,
        Expr_Var,
        Expr_BinOp,
        Expr_Call,
        Expr_Print,
    };
    ExprAST(ExprASTKind kind, Location location): kind(kind), location(std::move(location)) {}

    ExprASTKind getKind() const { return kind; }

    const Location &loc() {
        return location;
    }

    ~ExprAST() = default; 

private:
    const ExprASTKind kind;
    Location location;
};

using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class NumberExprAST: public ExprAST {
    double val;
public:
    NumberExprAST(Location loc, double val): ExprAST(Expr_Num, loc), val(val) { }
    double getValue() {
        return val;
    }
    static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

class LiteralExprAST : public ExprAST {
    std::vector<std::unique_ptr<ExprAST>> values;
    std::vector<int64_t> dims;
public:
    LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values, std::vector<int64_t> dims ):
    ExprAST(Expr_Literal, loc), values(std::move(values)), dims(std::move(dims)) { }

    llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() {
        return values;
    }

    llvm::ArrayRef<int64_t> getDims() {
        return dims;
    }

    static bool classof(const ExprAST * c) {
        return c->getKind() == Expr_Literal;
    }

};

class VariableExprAST : public ExprAST {
    std::string name;

public:
    VariableExprAST(Location loc, llvm::StringRef name): ExprAST(Expr_Var, std::move(loc)), name(name) { }
    llvm::StringRef getName() {
        return name;
    }

    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Var;
    }
};

class VarDeclExprAST : public ExprAST {
    std::string name;
    VarType type;
    std::unique_ptr<ExprAST> initVal;

public:
    VarDeclExprAST(Location loc, llvm::StringRef name, VarType type, std::unique_ptr<ExprAST>initVal ):
    ExprAST(Expr_VarDecl, loc), name(name), type(type), initVal(std::move(initVal)) {}

    llvm::StringRef getName() { return name; }
    VarType getType() { return type; }
    ExprAST * getInitVal() { return initVal.get(); }

    static bool classof(const ExprAST * c) {
        return c -> getKind() == Expr_VarDecl;
    }
};

class ReturnExprAST : public ExprAST {
    std::optional<std::unique_ptr<ExprAST>> expr;
public:
    ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr): 
        ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

    std::optional<ExprAST *> getExpr() {
        if (expr.has_value()) {
            return expr->get();
        }
        return std::nullopt;
    }

    static bool classof(const ExprAST * c) {
        return c->getKind() == Expr_Return;
    }
};

class BinaryExprAST : public ExprAST {
    char op;
    std::unique_ptr<ExprAST> lhs, rhs;
public:
    char getOp() { return op; }
    ExprAST * getLHS() { return lhs.get(); }
    ExprAST * getRHS() { return rhs.get(); }
    BinaryExprAST(Location location, char op, std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs):
    ExprAST(Expr_BinOp, location), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

    static bool classof(const ExprAST * c) {
        return c->getKind() == Expr_BinOp;
    }
};

class CallExprAST : public ExprAST {
    std::string callee;
    std::vector<std::unique_ptr<ExprAST>> args;

public:
    CallExprAST(Location loc, std::string callee, std::vector<std::unique_ptr<ExprAST>> args):
    ExprAST(Expr_Call, loc), callee(callee), args(std::move(args)) {}

    llvm::StringRef getCallee() {
        return callee;
    }
    llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() {
        return args;
    }

    static bool classof(const ExprAST *c) {
        return c->getKind() == Expr_Call;
    }

};

/// Expression class for builtin print calls.                                                                                                                         
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(Expr_Print, std::move(loc)), arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM style RTTI                                                                                                                                                 
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};

class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(std::move(location)), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
};

/// This class represents a function definition itself.                                                                                                               
class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}
  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }
};

/// This class represents a list of functions to be processed together                                                                                                
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }
};

void dump(ModuleAST &);


}


#endif //OBS_AST_H


