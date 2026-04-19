//===- ReplaceOperands.h - Replacing Operands Reduction Pattern -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REPLACEOPERANDS_H
#define MLIR_REDUCER_REPLACEOPERANDS_H

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Reducer/Tester.h"

namespace mlir {

struct OperandReductionNode {
  using Range = std::pair<size_t, size_t>;

  OperandReductionNode(Operation *reductionOp, ArrayRef<Range> ranges);

  ArrayRef<OpOperand *> getOperandForType(Type type) {
    return operandMap[type];
  }

  const DenseSet<Value> &getUsedValues() const { return usedValues; }

  auto getNeededTypes() const { return operandMap.keys(); }

  ModuleOp getModule() const { return module.get(); }
  Operation *getOperation() const { return op; }

  Value getMappedValue(Value value) { return mapping.lookup(value); }

private:
  Operation *op;
  OwningOpRef<ModuleOp> module;
  IRMapping mapping;
  DenseSet<Value> usedValues;
  SmallVector<Range, 0> startRanges;
  SmallVector<Range, 0> discardRanges;
  llvm::MapVector<Type, SmallVector<OpOperand *, 2>> operandMap;
};

struct ReplaceOperandsPattern : public mlir::RewritePattern {
private:
  Tester &tester;

public:
  ReplaceOperandsPattern(mlir::MLIRContext *context, Tester &tester)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        tester(tester) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};
} // namespace mlir

#endif
