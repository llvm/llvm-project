//===- CodeComplete.h - PDLL Frontend CodeComplete Context ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_PDLL_PARSER_CODECOMPLETE_H_
#define MLIR_TOOLS_PDLL_PARSER_CODECOMPLETE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/Support/SourceMgr.h"
#include <optional>

namespace mlir {
namespace pdll {
namespace ast {
class CallableDecl;
class DeclScope;
class Expr;
class OperationType;
class TupleType;
class Type;
class VariableDecl;
} // namespace ast

/// This class provides an abstract interface into the parser for hooking in
/// code completion events.
class CodeCompleteContext {
public:
  virtual ~CodeCompleteContext();

  /// Return the location used to provide code completion.
  SMLoc getCodeCompleteLoc() const { return codeCompleteLoc; }

  //===--------------------------------------------------------------------===//
  // Completion Hooks
  //===--------------------------------------------------------------------===//

  /// Signal code completion for a member access into the given tuple type.
  virtual void codeCompleteTupleMemberAccess(ast::TupleType tupleType);

  /// Signal code completion for a member access into the given operation type.
  virtual void codeCompleteOperationMemberAccess(ast::OperationType opType);

  /// Signal code completion for a member access into the given operation type.
  virtual void codeCompleteOperationAttributeName(StringRef opName) {}

  /// Signal code completion for a constraint name with an optional decl scope.
  /// `currentType` is the current type of the variable that will use the
  /// constraint, or nullptr if a type is unknown. `allowInlineTypeConstraints`
  /// enables inline type constraints for Attr/Value/ValueRange.
  virtual void codeCompleteConstraintName(ast::Type currentType,
                                          bool allowInlineTypeConstraints,
                                          const ast::DeclScope *scope);

  /// Signal code completion for a dialect name.
  virtual void codeCompleteDialectName() {}

  /// Signal code completion for an operation name in the given dialect.
  virtual void codeCompleteOperationName(StringRef dialectName) {}

  /// Signal code completion for Pattern metadata.
  virtual void codeCompletePatternMetadata() {}

  /// Signal code completion for an include filename.
  virtual void codeCompleteIncludeFilename(StringRef curPath) {}

  //===--------------------------------------------------------------------===//
  // Signature Hooks
  //===--------------------------------------------------------------------===//

  /// Signal code completion for the signature of a callable.
  virtual void codeCompleteCallSignature(const ast::CallableDecl *callable,
                                         unsigned currentNumArgs) {}

  /// Signal code completion for the signature of an operation's operands.
  virtual void
  codeCompleteOperationOperandsSignature(std::optional<StringRef> opName,
                                         unsigned currentNumOperands) {}

  /// Signal code completion for the signature of an operation's results.
  virtual void
  codeCompleteOperationResultsSignature(std::optional<StringRef> opName,
                                        unsigned currentNumResults) {}

protected:
  /// Create a new code completion context with the given code complete
  /// location.
  explicit CodeCompleteContext(SMLoc codeCompleteLoc)
      : codeCompleteLoc(codeCompleteLoc) {}

private:
  /// The location used to code complete.
  SMLoc codeCompleteLoc;
};
} // namespace pdll
} // namespace mlir

#endif // MLIR_TOOLS_PDLL_PARSER_CODECOMPLETE_H_
