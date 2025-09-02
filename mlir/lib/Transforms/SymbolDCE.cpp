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
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

namespace mlir {
#define GEN_PASS_DEF_SYMBOLDCE
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "symbol-dce"

namespace {
struct SymbolDCE : public impl::SymbolDCEBase<SymbolDCE> {
  void runOnOperation() override;

  /// Compute the liveness of the symbols within the given symbol table.
  /// `symbolTableIsHidden` is true if this symbol table is known to be
  /// unaccessible from operations in its parent regions.
  LogicalResult computeLiveness(Operation *symbolTableOp,
                                SymbolTableCollection &symbolTable,
                                bool symbolTableIsHidden,
                                DenseSet<Operation *> &liveSymbols);
};
} // namespace

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
  SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(symbolTableOp);
  if (symbolTableOp->getParentOp() && symbol)
    symbolTableIsHidden = symbol.isPrivate();

  // Compute the set of live symbols within the symbol table.
  DenseSet<Operation *> liveSymbols;
  SymbolTableCollection symbolTable;
  if (failed(computeLiveness(symbolTableOp, symbolTable, symbolTableIsHidden,
                             liveSymbols)))
    return signalPassFailure();

  // After computing the liveness, delete all of the symbols that were found to
  // be dead.
  symbolTableOp->walk([&](Operation *nestedSymbolTable) {
    if (!nestedSymbolTable->hasTrait<OpTrait::SymbolTable>())
      return;
    for (auto &block : nestedSymbolTable->getRegion(0)) {
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (isa<SymbolOpInterface>(&op) && !liveSymbols.count(&op)) {
          op.erase();
          ++numDCE;
        }
      }
    }
  });
}

/// Compute the liveness of the symbols within the given symbol table.
/// `symbolTableIsHidden` is true if this symbol table is known to be
/// unaccessible from operations in its parent regions.
LogicalResult SymbolDCE::computeLiveness(Operation *symbolTableOp,
                                         SymbolTableCollection &symbolTable,
                                         bool symbolTableIsHidden,
                                         DenseSet<Operation *> &liveSymbols) {
  LDBG() << "computeLiveness: "
         << OpWithFlags(symbolTableOp, OpPrintingFlags().skipRegions());
  // A worklist of live operations to propagate uses from.
  SmallVector<Operation *, 16> worklist;

  // Walk the symbols within the current symbol table, marking the symbols that
  // are known to be live.
  for (auto &block : symbolTableOp->getRegion(0)) {
    // Add all non-symbols or symbols that can't be discarded.
    for (Operation &op : block) {
      SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(&op);
      if (!symbol) {
        worklist.push_back(&op);
        continue;
      }
      bool isDiscardable = (symbolTableIsHidden || symbol.isPrivate()) &&
                           symbol.canDiscardOnUseEmpty();
      if (!isDiscardable && liveSymbols.insert(&op).second)
        worklist.push_back(&op);
    }
  }

  // Process the set of symbols that were known to be live, adding new symbols
  // that are referenced within. For operations that are not symbol tables, it
  // considers the liveness with respect to the op itself rather than scope of
  // nested symbol tables by enqueuing all the top level operations for
  // consideration.
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    LDBG() << "processing: "
           << OpWithFlags(op, OpPrintingFlags().skipRegions());

    // If this is a symbol table, recursively compute its liveness.
    if (op->hasTrait<OpTrait::SymbolTable>()) {
      // The internal symbol table is hidden if the parent is, if its not a
      // symbol, or if it is a private symbol.
      SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(op);
      bool symIsHidden = symbolTableIsHidden || !symbol || symbol.isPrivate();
      LDBG() << "\tsymbol table: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions())
             << " is hidden: " << symIsHidden;
      if (failed(computeLiveness(op, symbolTable, symIsHidden, liveSymbols)))
        return failure();
    } else {
      LDBG() << "\tnon-symbol table: "
             << OpWithFlags(op, OpPrintingFlags().skipRegions());
      // If the op is not a symbol table, then, unless op itself is dead which
      // would be handled by DCE, we need to check all the regions and blocks
      // within the op to find the uses (e.g., consider visibility within op as
      // if top level rather than relying on pure symbol table visibility). This
      // is more conservative than SymbolTable::walkSymbolTables in the case
      // where there is again SymbolTable information to take advantage of.
      for (auto &region : op->getRegions())
        for (auto &block : region.getBlocks())
          for (Operation &op : block)
            if (op.getNumRegions())
              worklist.push_back(&op);
    }

    // Get the first parent symbol table op. Note: due to enqueueing of
    // top-level ops, we may not have a symbol table parent here, but if we do
    // not, then we also don't have a symbol.
    Operation *parentOp = op->getParentOp();
    if (!parentOp->hasTrait<OpTrait::SymbolTable>())
      continue;

    // Collect the uses held by this operation.
    std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(op);
    if (!uses) {
      return op->emitError()
             << "operation contains potentially unknown symbol table, meaning "
             << "that we can't reliable compute symbol uses";
    }

    SmallVector<Operation *, 4> resolvedSymbols;
    LDBG() << "uses of " << OpWithFlags(op, OpPrintingFlags().skipRegions());
    for (const SymbolTable::SymbolUse &use : *uses) {
      LDBG() << "\tuse: " << use.getUser();
      // Lookup the symbols referenced by this use.
      resolvedSymbols.clear();
      if (failed(symbolTable.lookupSymbolIn(parentOp, use.getSymbolRef(),
                                            resolvedSymbols)))
        // Ignore references to unknown symbols.
        continue;
      LDBG() << "\t\tresolved symbols: "
             << llvm::interleaved(resolvedSymbols, ", ");

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
