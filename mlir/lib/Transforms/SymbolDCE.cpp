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

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct SymbolDCE : public OperationPass<SymbolDCE> {
  void runOnOperation() override;

  /// Compute the liveness of the symbols within the given symbol table.
  /// `symbolTableIsHidden` is true if this symbol table is known to be
  /// unaccessible from operations in its parent regions.
  LogicalResult computeLiveness(Operation *symbolTableOp,
                                bool symbolTableIsHidden,
                                DenseSet<Operation *> &liveSymbols);
};
} // end anonymous namespace

void SymbolDCE::runOnOperation() {
  Operation *symbolTableOp = getOperation();

  // SymbolDCE should only be run on operations that define a symbol table.
  if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>()) {
    symbolTableOp->emitOpError()
        << " was scheduled to run under SymbolDCE, but does not define a "
           "symbol table";
    return signalPassFailure();
  }

  // A flag that signals if the top level symbol table is hidden, i.e. not
  // accessible from parent scopes.
  bool symbolTableIsHidden = true;
  if (symbolTableOp->getParentOp() && SymbolTable::isSymbol(symbolTableOp)) {
    symbolTableIsHidden = SymbolTable::getSymbolVisibility(symbolTableOp) ==
                          SymbolTable::Visibility::Private;
  }

  // Compute the set of live symbols within the symbol table.
  DenseSet<Operation *> liveSymbols;
  if (failed(computeLiveness(symbolTableOp, symbolTableIsHidden, liveSymbols)))
    return signalPassFailure();

  // After computing the liveness, delete all of the symbols that were found to
  // be dead.
  symbolTableOp->walk([&](Operation *nestedSymbolTable) {
    if (!nestedSymbolTable->hasTrait<OpTrait::SymbolTable>())
      return;
    for (auto &block : nestedSymbolTable->getRegion(0)) {
      for (Operation &op :
           llvm::make_early_inc_range(block.without_terminator())) {
        if (SymbolTable::isSymbol(&op) && !liveSymbols.count(&op))
          op.erase();
      }
    }
  });
}

/// Compute the liveness of the symbols within the given symbol table.
/// `symbolTableIsHidden` is true if this symbol table is known to be
/// unaccessible from operations in its parent regions.
LogicalResult SymbolDCE::computeLiveness(Operation *symbolTableOp,
                                         bool symbolTableIsHidden,
                                         DenseSet<Operation *> &liveSymbols) {
  // A worklist of live operations to propagate uses from.
  SmallVector<Operation *, 16> worklist;

  // Walk the symbols within the current symbol table, marking the symbols that
  // are known to be live.
  for (auto &block : symbolTableOp->getRegion(0)) {
    for (Operation &op : block.without_terminator()) {
      // Always add non symbol operations to the worklist.
      if (!SymbolTable::isSymbol(&op)) {
        worklist.push_back(&op);
        continue;
      }

      // Check the visibility to see if this symbol may be referenced
      // externally.
      SymbolTable::Visibility visibility =
          SymbolTable::getSymbolVisibility(&op);

      // Private symbols are always initially considered dead.
      if (visibility == mlir::SymbolTable::Visibility::Private)
        continue;
      // We only include nested visibility here if the symbol table isn't
      // hidden.
      if (symbolTableIsHidden && visibility == SymbolTable::Visibility::Nested)
        continue;

      // TODO(riverriddle) Add hooks here to allow symbols to provide additional
      // information, e.g. linkage can be used to drop some symbols that may
      // otherwise be considered "live".
      if (liveSymbols.insert(&op).second)
        worklist.push_back(&op);
    }
  }

  // Process the set of symbols that were known to be live, adding new symbols
  // that are referenced within.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    // If this is a symbol table, recursively compute its liveness.
    if (op->hasTrait<OpTrait::SymbolTable>()) {
      // The internal symbol table is hidden if the parent is, if its not a
      // symbol, or if it is a private symbol.
      bool symbolIsHidden = symbolTableIsHidden || !SymbolTable::isSymbol(op) ||
                            SymbolTable::getSymbolVisibility(op) ==
                                SymbolTable::Visibility::Private;
      if (failed(computeLiveness(op, symbolIsHidden, liveSymbols)))
        return failure();
    }

    // Collect the uses held by this operation.
    Optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(op);
    if (!uses) {
      return op->emitError()
             << "operation contains potentially unknown symbol table, "
                "meaning that we can't reliable compute symbol uses";
    }

    SmallVector<Operation *, 4> resolvedSymbols;
    for (const SymbolTable::SymbolUse &use : *uses) {
      // Lookup the symbols referenced by this use.
      resolvedSymbols.clear();
      if (failed(SymbolTable::lookupSymbolIn(
              op->getParentOp(), use.getSymbolRef(), resolvedSymbols))) {
        return use.getUser()->emitError()
               << "unable to resolve reference to symbol "
               << use.getSymbolRef();
      }

      // Mark each of the resolved symbols as live.
      for (Operation *resolvedSymbol : resolvedSymbols)
        if (liveSymbols.insert(resolvedSymbol).second)
          worklist.push_back(resolvedSymbol);
    }
  }

  return success();
}

std::unique_ptr<Pass> mlir::createSymbolDCEPass() {
  return std::make_unique<SymbolDCE>();
}

static PassRegistration<SymbolDCE> pass("symbol-dce", "Eliminate dead symbols");
