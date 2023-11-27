//===- CSE.h - Common Subexpression Elimination -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for eliminating common subexpressions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CSE_H_
#define MLIR_TRANSFORMS_CSE_H_

#include "mlir/IR/DialectInterface.h"

namespace mlir {

class DominanceInfo;
class Operation;
class RewriterBase;
struct LogicalResult;

/// Eliminate common subexpressions within the given operation. This transform
/// looks for and deduplicates equivalent operations.
///
/// `changed` indicates whether the IR was modified or not.
void eliminateCommonSubExpressions(RewriterBase &rewriter,
                                   DominanceInfo &domInfo, Operation *op,
                                   bool *changed = nullptr);

//===----------------------------------------------------------------------===//
// CSEInterface
//===----------------------------------------------------------------------===//

/// This is the interface that allows users to customize CSE.
class DialectCSEInterface : public DialectInterface::Base<DialectCSEInterface> {
public:
  DialectCSEInterface(Dialect *dialect) : Base(dialect) {}

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// These two hooks are called in DenseMapInfo used by CSE.

  /// Returns a hash for the operation.
  /// CSE will use default implementation if `std::nullopt` is returned.
  virtual std::optional<unsigned> getHashValue(Operation *op) const = 0;

  /// Returns true if two operations are considered to be equivalent.
  /// CSE will use default implementation if `std::nullopt` is returned.
  virtual std::optional<bool> isEqual(Operation *lhs, Operation *rhs) const = 0;

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// This hook is called when 'op' is considered to be a common subexpression
  /// of 'existingOp' and is going to be eliminated. This hook allows users to
  /// propagate information of 'op' to 'existingOp'. Note that the hash value of
  /// 'existingOp' must not be changed due the mutation of 'existingOp'.
  virtual void mergeOperations(Operation *existingOp, Operation *op) const {}
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_CSE_H_
