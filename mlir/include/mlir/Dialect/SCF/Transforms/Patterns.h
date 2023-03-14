//===- Patterns.h - SCF dialect rewrite patterns ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_PATTERNS_H
#define MLIR_DIALECT_SCF_TRANSFORMS_PATTERNS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace scf {
/// Generate a pipelined version of the scf.for loop based on the schedule given
/// as option. This applies the mechanical transformation of changing the loop
/// and generating the prologue/epilogue for the pipelining and doesn't make any
/// decision regarding the schedule.
/// Based on the options the loop is split into several stages.
/// The transformation assumes that the scheduling given by user is valid.
/// For example if we break a loop into 3 stages named S0, S1, S2 we would
/// generate the following code with the number in parenthesis as the iteration
/// index:
/// S0(0)                        // Prologue
/// S0(1) S1(0)                  // Prologue
/// scf.for %I = %C0 to %N - 2 {
///  S0(I+2) S1(I+1) S2(I)       // Pipelined kernel
/// }
/// S1(N) S2(N-1)                // Epilogue
/// S2(N)                        // Epilogue
FailureOr<ForOp> pipelineForLoop(RewriterBase &rewriter, ForOp forOp,
                                 const PipeliningOption &options);

// TODO: such patterns should be auto-generated.
class ForLoopPipeliningPattern : public OpRewritePattern<ForOp> {
public:
  ForLoopPipeliningPattern(const PipeliningOption &options,
                           MLIRContext *context)
      : OpRewritePattern<ForOp>(context), options(options) {}
  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(forOp, rewriter);
  }

  FailureOr<ForOp> returningMatchAndRewrite(ForOp forOp,
                                            PatternRewriter &rewriter) const {
    return pipelineForLoop(rewriter, forOp, options);
  }

protected:
  PipeliningOption options;
};

} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_PATTERNS_H
