//===- SymbolDCE.cpp - Pass to delete dead symbols ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an algorithm for eliminating symbol operations that are
// known to be dead.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Transforms/SymbolDceUtils.h"

namespace mlir {
#define GEN_PASS_DEF_SYMBOLDCE
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct SymbolDCE : public impl::SymbolDCEBase<SymbolDCE> {
  void runOnOperation() override;
};
} // namespace

void SymbolDCE::runOnOperation() {
  Operation *symbolTableOp = getOperation();
  if (failed(symbolDce(symbolTableOp)))
    return signalPassFailure();
}

std::unique_ptr<Pass> mlir::createSymbolDCEPass() {
  return std::make_unique<SymbolDCE>();
}
