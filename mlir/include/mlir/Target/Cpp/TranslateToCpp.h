//===-- TranslateToCpp.h - Define CppEmitter struct -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CppEmitter meta struct used for Cpp translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_TRANSLATETOCPP_H
#define MLIR_TARGET_CPP_TRANSLATETOCPP_H

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Target/Cpp/CppTranslationInterface.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

namespace mlir {

/// Emitter that uses dialect specific emitters to emit C++ code.
struct CppEmitter {
  friend LogicalResult translateToCpp(Operation *op, raw_ostream &os,
                                      bool declareVariablesAtTop);

  explicit CppEmitter(raw_ostream &os, Operation *module,
                      bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  /// Emits a declaration of a variable with the given type and name.
  LogicalResult emitVariableDeclaration(Location loc, Type type,
                                        StringRef name);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Emits value as an operands of an operation
  LogicalResult emitOperand(Value value);

  /// Emit an expression as a C expression.
  LogicalResult emitExpression(emitc::ExpressionOp expressionOp);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(CppEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    CppEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

  /// Get expression currently being emitted.
  emitc::ExpressionOp getEmittedExpression() { return emittedExpression; }

  /// Determine whether given value is part of the expression potentially being
  /// emitted.
  bool isPartOfCurrentExpression(Value value) {
    if (!emittedExpression)
      return false;
    Operation *def = value.getDefiningOp();
    if (!def)
      return false;
    auto operandExpression = dyn_cast<emitc::ExpressionOp>(def->getParentOp());
    return operandExpression == emittedExpression;
  };

protected:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Module op to get current context
  Operation *module;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// A dialect interface collection used for dispatching the translation to
  /// specific dialects.
  CppTranslationInterface iface;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;

  /// State of the current expression being emitted.
  emitc::ExpressionOp emittedExpression;
  SmallVector<int> emittedExpressionPrecedence;

  void pushExpressionPrecedence(int precedence) {
    emittedExpressionPrecedence.push_back(precedence);
  }
  void popExpressionPrecedence() { emittedExpressionPrecedence.pop_back(); }
  static int lowestPrecedence() { return 0; }
  int getExpressionPrecedence() {
    if (emittedExpressionPrecedence.empty())
      return lowestPrecedence();
    return emittedExpressionPrecedence.back();
  }
};

} // namespace mlir

#endif // MLIR_TARGET_CPP_TRANSLATETOCPP_H
