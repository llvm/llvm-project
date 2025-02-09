//===- DIExpressionRewriter.cpp - Rewriter for DIExpression operators -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/DIExpressionRewriter.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace LLVM;

#define DEBUG_TYPE "llvm-di-expression-simplifier"

//===----------------------------------------------------------------------===//
// DIExpressionRewriter
//===----------------------------------------------------------------------===//

void DIExpressionRewriter::addPattern(
    std::unique_ptr<ExprRewritePattern> pattern) {
  patterns.emplace_back(std::move(pattern));
}

DIExpressionAttr
DIExpressionRewriter::simplify(DIExpressionAttr expr,
                               std::optional<uint64_t> maxNumRewrites) const {
  ArrayRef<OperatorT> operators = expr.getOperations();

  // `inputs` contains the unprocessed postfix of operators.
  // `result` contains the already finalized prefix of operators.
  // Invariant: concat(result, inputs) is equivalent to `operators` after some
  // application of the rewrite patterns.
  // Using a deque for inputs so that we have efficient front insertion and
  // removal. Random access is not necessary for patterns.
  std::deque<OperatorT> inputs(operators.begin(), operators.end());
  SmallVector<OperatorT> result;

  uint64_t numRewrites = 0;
  while (!inputs.empty() &&
         (!maxNumRewrites || numRewrites < *maxNumRewrites)) {
    bool foundMatch = false;
    for (const std::unique_ptr<ExprRewritePattern> &pattern : patterns) {
      ExprRewritePattern::OpIterT matchEnd = pattern->match(inputs);
      if (matchEnd == inputs.begin())
        continue;

      foundMatch = true;
      SmallVector<OperatorT> replacement =
          pattern->replace(llvm::make_range(inputs.cbegin(), matchEnd));
      inputs.erase(inputs.begin(), matchEnd);
      inputs.insert(inputs.begin(), replacement.begin(), replacement.end());
      ++numRewrites;
      break;
    }

    if (!foundMatch) {
      // If no match, pass along the current operator.
      result.push_back(inputs.front());
      inputs.pop_front();
    }
  }

  if (maxNumRewrites && numRewrites >= *maxNumRewrites) {
    LLVM_DEBUG(llvm::dbgs()
               << "LLVMDIExpressionSimplifier exceeded max num rewrites ("
               << maxNumRewrites << ")\n");
    // Skip rewriting the rest.
    result.append(inputs.begin(), inputs.end());
  }

  return LLVM::DIExpressionAttr::get(expr.getContext(), result);
}
