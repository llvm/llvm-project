//===- LoopUtils.h - Helpers related to loop operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for loop operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"

namespace mlir {

// This structure is to pass and return sets of loop parameters without
// confusing the order.
struct LoopParams {
  Value lowerBound;
  Value upperBound;
  Value step;
};

/// Calculate the normalized loop upper bounds with lower bound equal to zero
/// and step equal to one.
LoopParams emitNormalizedLoopBounds(RewriterBase &rewriter, Location loc,
                                    Value lb, Value ub, Value step);

} // namespace mlir
