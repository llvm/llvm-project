//===- PipelineGlobalOpsPass.cpp - Pipeline Global Ops Pass ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/Transforms/Passes.h"

#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MLProgram/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace ml_program {
#define GEN_PASS_DEF_MLPROGRAMPIPELINEGLOBALS
#include "mlir/Dialect/MLProgram/Transforms/Passes.h.inc"

namespace {

class MLProgramPipelineGlobals
    : public impl::MLProgramPipelineGlobalsBase<MLProgramPipelineGlobals> {
public:
  void runOnOperation() override;

private:
  LogicalResult buildGlobalMap(ModuleOp op);

  void processBlock(Block &block, llvm::DenseSet<SymbolRefAttr> &symbolLoad,
                    llvm::DenseSet<SymbolRefAttr> &symbolStore);

  llvm::DenseMap<SymbolRefAttr, llvm::DenseSet<SymbolRefAttr>> loadSymbolsMap;
  llvm::DenseMap<SymbolRefAttr, llvm::DenseSet<SymbolRefAttr>> storeSymbolsMap;
};

// Traverses upwards searchign for the operation mapped by the symbol.
static Operation *getFromSymbol(Operation *baseOp, SymbolRefAttr symbol) {
  for (auto *op = baseOp; op; op = op->getParentOp()) {
    auto *lookup = SymbolTable::lookupNearestSymbolFrom(op, symbol);
    if (lookup)
      return lookup;
  }
  return nullptr;
}

// Builds map from a symbol to MLProgram global symbols loaded or stored
// during processing.
LogicalResult MLProgramPipelineGlobals::buildGlobalMap(ModuleOp module) {
  llvm::DenseMap<SymbolRefAttr, Operation *> callableMap;
  auto res = module->walk([&](Operation *op) {
    if (auto caller = mlir::dyn_cast<CallOpInterface>(op)) {
      auto callable = caller.getCallableForCallee();
      // For now we do not know how to handle Value based tracing, so fail.
      if (mlir::isa<Value>(callable)) {
        return WalkResult::interrupt();
      }

      auto symbol = mlir::dyn_cast<SymbolRefAttr>(callable);
      auto *func = getFromSymbol(op, symbol);
      callableMap[symbol] = func;
    }
    return WalkResult::advance();
  });

  if (res.wasInterrupted()) {
    return failure();
  }

  // First grab all symbols loaded or stored by each function. This
  // will not handle calls initially.
  llvm::DenseMap<SymbolRefAttr, llvm::DenseSet<SymbolRefAttr>> opLoadSymbols;
  llvm::DenseMap<SymbolRefAttr, llvm::DenseSet<SymbolRefAttr>> opStoreSymbols;
  for (auto callable : callableMap) {
    llvm::DenseSet<SymbolRefAttr> loadSymbols;
    llvm::DenseSet<SymbolRefAttr> storeSymbols;

    callable.getSecond()->walk(
        [&](GlobalLoadOp op) { loadSymbols.insert(op.getGlobal()); });

    callable.getSecond()->walk(
        [&](GlobalStoreOp op) { storeSymbols.insert(op.getGlobal()); });

    opLoadSymbols[callable.getFirst()] = std::move(loadSymbols);
    opStoreSymbols[callable.getFirst()] = std::move(storeSymbols);
  }

  // For each callable function we find each global loaded/stored within the
  // function or a nested called function. This includes recursion checking to
  // avoid infinitely recursing.
  for (auto callable : callableMap) {
    SymbolRefAttr thisSymbol = llvm::dyn_cast<SymbolRefAttr>(callable.first);
    llvm::SmallVector<SymbolRefAttr> work = {thisSymbol};
    llvm::DenseSet<SymbolRefAttr> visited = {thisSymbol};
    llvm::DenseSet<SymbolRefAttr> loadSymbols;
    llvm::DenseSet<SymbolRefAttr> storeSymbols;

    for (size_t i = 0; i < work.size(); ++i) {
      callableMap[work[i]]->walk([&](CallOpInterface call) {
        auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
        if (visited.insert(symbol).second)
          work.push_back(symbol);
      });

      for (auto load : opLoadSymbols[work[i]])
        loadSymbols.insert(load);

      for (auto store : opStoreSymbols[work[i]])
        storeSymbols.insert(store);
    }

    loadSymbolsMap[thisSymbol] = std::move(loadSymbols);
    storeSymbolsMap[thisSymbol] = std::move(storeSymbols);
  }

  return success();
}

// Process each operation in the block deleting unneeded loads / stores,
// recursing on subblocks and checking function calls.
void MLProgramPipelineGlobals::processBlock(
    Block &block, llvm::DenseSet<SymbolRefAttr> &symbolLoad,
    llvm::DenseSet<SymbolRefAttr> &symbolStore) {

  llvm::DenseMap<SymbolRefAttr, Value> previousLoads;
  llvm::DenseMap<SymbolRefAttr, Operation *> previousStores;
  llvm::SmallVector<Operation *> toDelete;
  for (auto &op : block) {
    // If this is a global load, remap to a previous value if known
    // and delete this load. Remember that this value is the currently
    // known load.
    if (auto load = mlir::dyn_cast<GlobalLoadOp>(op)) {
      auto ref = load.getGlobal();
      symbolLoad.insert(ref);
      if (previousLoads.contains(ref)) {
        toDelete.push_back(&op);
        load.getResult().replaceAllUsesWith(previousLoads[ref]);
      } else {
        previousLoads[ref] = load.getResult();
      }
      continue;
    }

    // Delete a previous store if it exists and is not needed, update
    // the most recent known value for this global ref.
    if (auto store = mlir::dyn_cast<GlobalStoreOp>(op)) {
      auto ref = store.getGlobal();
      symbolStore.insert(ref);
      auto it = previousStores.find(ref);
      if (it != previousStores.end()) {
        toDelete.push_back(it->getSecond());
      }

      previousLoads[ref] = store.getValue();
      previousStores[ref] = &op;
      continue;
    }

    // If a function is called, clear known values for loads/stores used by
    // the function or its sub-functions.
    if (auto call = mlir::dyn_cast<CallOpInterface>(op)) {
      auto loadSymbols =
          loadSymbolsMap[dyn_cast<SymbolRefAttr>(call.getCallableForCallee())];
      auto storeSymbols =
          storeSymbolsMap[dyn_cast<SymbolRefAttr>(call.getCallableForCallee())];

      for (auto sym : loadSymbols) {
        previousStores.erase(sym);
      }

      for (auto sym : storeSymbols) {
        previousLoads.erase(sym);
        previousStores.erase(sym);
      }
      continue;
    }

    // If the op has sub-regions, recurse inside. We make no guarantees whether
    // the recursion occurs.
    llvm::DenseSet<SymbolRefAttr> opSymbolLoad;
    llvm::DenseSet<SymbolRefAttr> opSymbolStore;
    for (auto &region : op.getRegions()) {
      for (auto &block : region) {
        processBlock(block, opSymbolLoad, opSymbolStore);
      }
    }

    // Update current state from the subblock.
    for (auto change : opSymbolLoad) {
      symbolLoad.insert(change);
      previousStores.erase(change);
    }

    for (auto change : opSymbolStore) {
      symbolStore.insert(change);
      previousLoads.erase(change);
      previousStores.erase(change);
    }
  }

  for (auto *op : toDelete) {
    op->erase();
  }
}

void MLProgramPipelineGlobals::runOnOperation() {
  auto targetOp = getOperation();
  if (failed(buildGlobalMap(targetOp))) {
    return;
  }

  for (auto &funcOp : *targetOp.getBody()) {
    for (auto &region : funcOp.getRegions()) {
      for (auto &block : region.getBlocks()) {
        llvm::DenseSet<SymbolRefAttr> symbolsLoaded;
        llvm::DenseSet<SymbolRefAttr> symbolsStored;
        processBlock(block, symbolsLoaded, symbolsStored);
      }
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMLProgramPipelineGlobalsPass() {
  return std::make_unique<MLProgramPipelineGlobals>();
}

} // namespace ml_program
} // namespace mlir
