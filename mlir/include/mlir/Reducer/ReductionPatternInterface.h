//===- ReducePatternInterface.h - Collecting Reduce Patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONPATTERNINTERFACE_H
#define MLIR_REDUCER_REDUCTIONPATTERNINTERFACE_H

#include "mlir/IR/DialectInterface.h"

namespace mlir {

class RewritePatternSet;

/// This is used to report the reduction patterns for a Dialect. While using
/// mlir-reduce to reduce a module, we may want to transform certain cases into
/// simpler forms by applying certain rewrite patterns. Implement the
/// `populateReductionPatterns` to report those patterns by adding them to the
/// RewritePatternSet.
///
/// Example:
///   MyDialectReductionPattern::populateReductionPatterns(
///       RewritePatternSet &patterns) {
///       patterns.add<TensorOpReduction>(patterns.getContext());
///   }
///
/// For DRR, mlir-tblgen will generate a helper function
/// `populateWithGenerated` which has the same signature therefore you can
/// delegate to the helper function as well.
///
/// Example:
///   MyDialectReductionPattern::populateReductionPatterns(
///       RewritePatternSet &patterns) {
///       // Include the autogen file somewhere above.
///       populateWithGenerated(patterns);
///   }
class DialectReductionPatternInterface
    : public DialectInterface::Base<DialectReductionPatternInterface> {
public:
  /// Patterns provided here are intended to transform operations from a complex
  /// form to a simpler form, without breaking the semantics of the program
  /// being reduced. For example, you may want to replace the
  /// tensor<?xindex> with a known rank and type, e.g. tensor<1xi32>, or
  /// replacing an operation with a constant.
  virtual void populateReductionPatterns(RewritePatternSet &patterns) const = 0;

protected:
  DialectReductionPatternInterface(Dialect *dialect) : Base(dialect) {}
};

} // end namespace mlir

#endif // MLIR_REDUCER_REDUCTIONPATTERNINTERFACE_H
