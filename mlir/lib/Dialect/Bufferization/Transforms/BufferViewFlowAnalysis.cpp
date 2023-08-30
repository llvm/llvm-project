//======- BufferViewFlowAnalysis.cpp - Buffer alias analysis -*- C++ -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

/// Constructs a new alias analysis using the op provided.
BufferViewFlowAnalysis::BufferViewFlowAnalysis(Operation *op) { build(op); }

/// Find all immediate and indirect dependent buffers this value could
/// potentially have. Note that the resulting set will also contain the value
/// provided as it is a dependent alias of itself.
BufferViewFlowAnalysis::ValueSetT
BufferViewFlowAnalysis::resolve(Value rootValue) const {
  ValueSetT result;
  SmallVector<Value, 8> queue;
  queue.push_back(rootValue);
  while (!queue.empty()) {
    Value currentValue = queue.pop_back_val();
    if (result.insert(currentValue).second) {
      auto it = dependencies.find(currentValue);
      if (it != dependencies.end()) {
        for (Value aliasValue : it->second)
          queue.push_back(aliasValue);
      }
    }
  }
  return result;
}

/// Removes the given values from all alias sets.
void BufferViewFlowAnalysis::remove(const SetVector<Value> &aliasValues) {
  for (auto &entry : dependencies)
    llvm::set_subtract(entry.second, aliasValues);
}

void BufferViewFlowAnalysis::rename(Value from, Value to) {
  dependencies[to] = dependencies[from];
  dependencies.erase(from);

  for (auto &[key, value] : dependencies) {
    if (value.contains(from)) {
      value.insert(to);
      value.erase(from);
    }
  }
}

/// This function constructs a mapping from values to its immediate
/// dependencies. It iterates over all blocks, gets their predecessors,
/// determines the values that will be passed to the corresponding block
/// arguments and inserts them into the underlying map. Furthermore, it wires
/// successor regions and branch-like return operations from nested regions.
void BufferViewFlowAnalysis::build(Operation *op) {
  // Registers all dependencies of the given values.
  auto registerDependencies = [&](ValueRange values, ValueRange dependencies) {
    for (auto [value, dep] : llvm::zip(values, dependencies))
      this->dependencies[value].insert(dep);
  };

  op->walk([&](Operation *op) {
    // TODO: We should have an op interface instead of a hard-coded list of
    // interfaces/ops.

    // Add additional dependencies created by view changes to the alias list.
    if (auto viewInterface = dyn_cast<ViewLikeOpInterface>(op)) {
      dependencies[viewInterface.getViewSource()].insert(
          viewInterface->getResult(0));
      return WalkResult::advance();
    }

    if (auto branchInterface = dyn_cast<BranchOpInterface>(op)) {
      // Query all branch interfaces to link block argument dependencies.
      Block *parentBlock = branchInterface->getBlock();
      for (auto it = parentBlock->succ_begin(), e = parentBlock->succ_end();
           it != e; ++it) {
        // Query the branch op interface to get the successor operands.
        auto successorOperands =
            branchInterface.getSuccessorOperands(it.getIndex());
        // Build the actual mapping of values to their immediate dependencies.
        registerDependencies(successorOperands.getForwardedOperands(),
                             (*it)->getArguments().drop_front(
                                 successorOperands.getProducedOperandCount()));
      }
      return WalkResult::advance();
    }

    if (auto regionInterface = dyn_cast<RegionBranchOpInterface>(op)) {
      // Query the RegionBranchOpInterface to find potential successor regions.
      // Extract all entry regions and wire all initial entry successor inputs.
      SmallVector<RegionSuccessor, 2> entrySuccessors;
      regionInterface.getSuccessorRegions(/*index=*/std::nullopt,
                                          entrySuccessors);
      for (RegionSuccessor &entrySuccessor : entrySuccessors) {
        // Wire the entry region's successor arguments with the initial
        // successor inputs.
        registerDependencies(
            regionInterface.getEntrySuccessorOperands(
                entrySuccessor.isParent()
                    ? std::optional<unsigned>()
                    : entrySuccessor.getSuccessor()->getRegionNumber()),
            entrySuccessor.getSuccessorInputs());
      }

      // Wire flow between regions and from region exits.
      for (Region &region : regionInterface->getRegions()) {
        // Iterate over all successor region entries that are reachable from the
        // current region.
        SmallVector<RegionSuccessor, 2> successorRegions;
        regionInterface.getSuccessorRegions(region.getRegionNumber(),
                                            successorRegions);
        for (RegionSuccessor &successorRegion : successorRegions) {
          // Determine the current region index (if any).
          std::optional<unsigned> regionIndex;
          Region *regionSuccessor = successorRegion.getSuccessor();
          if (regionSuccessor)
            regionIndex = regionSuccessor->getRegionNumber();
          // Iterate over all immediate terminator operations and wire the
          // successor inputs with the successor operands of each terminator.
          for (Block &block : region)
            if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
                    block.getTerminator()))
              registerDependencies(terminator.getSuccessorOperands(regionIndex),
                                   successorRegion.getSuccessorInputs());
        }
      }

      return WalkResult::advance();
    }

    // Unknown op: Assume that all operands alias with all results.
    for (Value operand : op->getOperands()) {
      if (!isa<BaseMemRefType>(operand.getType()))
        continue;
      for (Value result : op->getResults()) {
        if (!isa<BaseMemRefType>(result.getType()))
          continue;
        registerDependencies({operand}, {result});
      }
    }
    return WalkResult::advance();
  });
}
