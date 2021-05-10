//======- BufferViewFlowAnalysis.cpp - Buffer alias analysis -*- C++ -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/BufferViewFlowAnalysis.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetOperations.h"

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
void BufferViewFlowAnalysis::remove(const SmallPtrSetImpl<Value> &aliasValues) {
  for (auto &entry : dependencies)
    llvm::set_subtract(entry.second, aliasValues);
}

/// This function constructs a mapping from values to its immediate
/// dependencies. It iterates over all blocks, gets their predecessors,
/// determines the values that will be passed to the corresponding block
/// arguments and inserts them into the underlying map. Furthermore, it wires
/// successor regions and branch-like return operations from nested regions.
void BufferViewFlowAnalysis::build(Operation *op) {
  // Registers all dependencies of the given values.
  auto registerDependencies = [&](auto values, auto dependencies) {
    for (auto entry : llvm::zip(values, dependencies))
      this->dependencies[std::get<0>(entry)].insert(std::get<1>(entry));
  };

  // Add additional dependencies created by view changes to the alias list.
  op->walk([&](ViewLikeOpInterface viewInterface) {
    dependencies[viewInterface.getViewSource()].insert(
        viewInterface->getResult(0));
  });

  // Query all branch interfaces to link block argument dependencies.
  op->walk([&](BranchOpInterface branchInterface) {
    Block *parentBlock = branchInterface->getBlock();
    for (auto it = parentBlock->succ_begin(), e = parentBlock->succ_end();
         it != e; ++it) {
      // Query the branch op interface to get the successor operands.
      auto successorOperands =
          branchInterface.getSuccessorOperands(it.getIndex());
      if (!successorOperands.hasValue())
        continue;
      // Build the actual mapping of values to their immediate dependencies.
      registerDependencies(successorOperands.getValue(), (*it)->getArguments());
    }
  });

  // Query the RegionBranchOpInterface to find potential successor regions.
  op->walk([&](RegionBranchOpInterface regionInterface) {
    // Extract all entry regions and wire all initial entry successor inputs.
    SmallVector<RegionSuccessor, 2> entrySuccessors;
    regionInterface.getSuccessorRegions(/*index=*/llvm::None, entrySuccessors);
    for (RegionSuccessor &entrySuccessor : entrySuccessors) {
      // Wire the entry region's successor arguments with the initial
      // successor inputs.
      assert(entrySuccessor.getSuccessor() &&
             "Invalid entry region without an attached successor region");
      registerDependencies(
          regionInterface.getSuccessorEntryOperands(
              entrySuccessor.getSuccessor()->getRegionNumber()),
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
        // Iterate over all immediate terminator operations and wire the
        // successor inputs with the operands of each terminator.
        for (Block &block : region) {
          for (Operation &operation : block) {
            if (operation.hasTrait<OpTrait::ReturnLike>())
              registerDependencies(operation.getOperands(),
                                   successorRegion.getSuccessorInputs());
          }
        }
      }
    }
  });
}
