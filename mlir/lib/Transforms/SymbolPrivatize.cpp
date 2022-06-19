//===- SymbolPrivatize.cpp - Pass to mark symbols private -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an pass that marks all symbols as private unless
// excluded.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct SymbolPrivatize : public SymbolPrivatizeBase<SymbolPrivatize> {
  explicit SymbolPrivatize(ArrayRef<std::string> excludeSymbols);
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;

  /// Symbols whose visibility won't be changed.
  DenseSet<StringAttr> excludedSymbols;
};
} // namespace

SymbolPrivatize::SymbolPrivatize(llvm::ArrayRef<std::string> excludeSymbols) {
  exclude = excludeSymbols;
}

LogicalResult SymbolPrivatize::initialize(MLIRContext *context) {
  for (const std::string &symbol : exclude)
    excludedSymbols.insert(StringAttr::get(context, symbol));
  return success();
}

void SymbolPrivatize::runOnOperation() {
  for (Region &region : getOperation()->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        auto symbol = dyn_cast<SymbolOpInterface>(op);
        if (!symbol)
          continue;
        if (!excludedSymbols.contains(symbol.getNameAttr()))
          symbol.setVisibility(SymbolTable::Visibility::Private);
      }
    }
  }
}

std::unique_ptr<Pass>
mlir::createSymbolPrivatizePass(ArrayRef<std::string> exclude) {
  return std::make_unique<SymbolPrivatize>(exclude);
}
