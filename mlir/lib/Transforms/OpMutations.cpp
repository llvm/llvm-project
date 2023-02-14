//===- OpMutations.cpp - Location Snapshot Utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/OpMutations.h"
#include "PassDetail.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/LocationSnapshot.h"

using namespace mlir;

void mlir::getOpMutations(Operation *op_before, Operation *op_after,
                          const IRMapping &ir_map) {
  llvm::outs()
      << "******************************************************************\n";
  if (op_before == nullptr)
    return;

  // Create a map of locations to/from Ops from the BEFORE version of the
  // module. The locations in this module are guarrenteed to be unique as they
  // are re-numbered just befor this function call.
  DenseMap<Location, Operation *> loc_to_op_map_before;
  DenseMap<Operation *, Location> op_to_loc_map_before;

  op_before->walk([&](Operation *opIt) {
    loc_to_op_map_before.insert({opIt->getLoc(), opIt});
    op_to_loc_map_before.insert({opIt, opIt->getLoc()});
  });

  // Create a map of locations to/from Ops from the AFTER version of the
  // module. The locations in this module are need not be unique as there could
  // have beed some module transformations
  DenseMap<Location, std::vector<Operation *>> loc_to_op_map_after;

  op_after->walk([&](Operation *opIt) {
    if (loc_to_op_map_after.find(opIt->getLoc()) == loc_to_op_map_after.end()) {
      loc_to_op_map_after.insert({opIt->getLoc(), {}});
    }

    loc_to_op_map_after[opIt->getLoc()].push_back(opIt);
  });

  // We have access to the versions of the module before and after the
  // transform. Following are the possible mutations that can occur on the
  // module.
  op_after->walk([&](Operation *opIt) {
    // If an op exists BEFORE and AFTER, it can still have mutaions, like
    // arguments and resulsts
    if (Operation *op = ir_map.lookupOrNull(opIt)) {
      bool is_op_mutated = false;
      // The information related to 1.1 and 1.2 is not available directly from
      // the Source-Location information. However, we should be able to evaluate
      // these mutations as we have access to versions of Op/Module before and
      // after the transform and we should be able to apply equivalance
      // comparisions like this easily-
      // mlir/lib/IR/OperationSupport.cpp;l=704-783
      // OperationEquivalence::isEquivalentTo

      // 1.1. Check if arguments are mutated

      // 1.2. Check if Op result is mutated

      // 1.3. Check if Locations match
      if (!is_op_mutated && op->getLoc() == opIt->getLoc())
        llvm::outs() << "Op remained unmutated in the transform- "
                     << op->getName() << "\n";

      // 1.4. Check if Locations do not match. If the locations do not match, it
      // means the op has been moved
      if (!is_op_mutated && op->getLoc() != opIt->getLoc())
        llvm::outs() << "Op remained unmutated in the transform but is moved- "
                     << op->getName() << "\n";

      // If the Op is present BEFORE and AFTER, delete it from the
      // loc_to_op_map_before and op_to_loc_map_before maps, as these maps will
      // be used to track the deleted Ops and other kinds of mutations
      op_to_loc_map_before.erase(op);
    } else {
      // If the Op didn't exist BEFORE and exists AFTER the transform, it is
      // probably newly introduce but still may have been derived from an Op in
      // BEFORE.
      llvm::outs() << "New Op- " << opIt->getName()
                   << " is inserted after the transform as a result of the ";

      std::vector<Operation *> mutated_ops_before;
      if (loc_to_op_map_after[opIt->getLoc()].size() > 1 &&
          loc_to_op_map_before.find(opIt->getLoc()) !=
              loc_to_op_map_before.end()) {
        // 2.1. If more than one Op in AFTER have the same location as an Op
        // in BEFORE, its probably an outcome of an unroll action.
        llvm::outs() << "unroll of- "
                     << loc_to_op_map_before[opIt->getLoc()]->getName() << "\n";
        mutated_ops_before.push_back(loc_to_op_map_before[opIt->getLoc()]);
      } else if (loc_to_op_map_before.find(opIt->getLoc()) !=
                 loc_to_op_map_before.end()) {
        // 2.2. Check if the Op in AFTER has the same location as an Op in
        // BEFORE. Its probably a result of 1->1 conversion pattern. Ex. TF ->
        // TFL Delete the converted Op from the loc_to_op_map_before and
        // op_to_loc_map_before maps
        llvm::outs() << "convertion of- "
                     << loc_to_op_map_before[opIt->getLoc()]->getName() << "\n";
        mutated_ops_before.push_back(loc_to_op_map_before[opIt->getLoc()]);
      } else if (auto fused_loc = opIt->getLoc().dyn_cast<FusedLoc>()) {
        // 2.3. Check if the Op in AFTER is a result of fusion. Get the list of
        // fused locations in BEFORE. Delete the fused Ops from the
        // loc_to_op_map_before and op_to_loc_map_before maps
        llvm::outs() << "Fusion of-";
        for (size_t loc_idx = 0; loc_idx < fused_loc.getLocations().size();
             ++loc_idx) {
          llvm::outs()
              << " "
              << loc_to_op_map_before[fused_loc.getLocations()[loc_idx]]
                     ->getName();
          mutated_ops_before.push_back(
              loc_to_op_map_before[fused_loc.getLocations()[loc_idx]]);
        }
        llvm::outs() << "\n";
      }

      for (Operation *m_op : mutated_ops_before) {
        op_to_loc_map_before.erase(m_op);
      }
    }
  });

  if (!op_to_loc_map_before.empty()) {
    // There are some Ops BEFORE that have not been accounted for in AFTER,
    // consider them deleted?
    for (auto deleted_op_loc_pair : op_to_loc_map_before) {
      Operation *deleted_op = deleted_op_loc_pair.first;
      llvm::outs() << "Op- " << deleted_op->getName()
                   << " is deleted in the transform\n";
    }
  }
}

namespace {
struct OpMutationsPass : public OpMutationsBase<OpMutationsPass> {
  OpMutationsPass() = default;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Print Op Mutations
    getOpMutations(module_op_prev, op, ir_map_prev);

    // Create snapshots of the locations
    OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(5).enableDebugInfo();
    if (failed(mlir::generateLocationsFromIR(StringRef(), op, flags))) {
      llvm::outs() << "Failed to create Location Snapshots\n";
    }

    // Store the state of the Module for a later pass
    ir_map_prev = mlir::IRMapping{};
    module_op_prev = op->clone(ir_map_prev);
  }

  static mlir::Operation *module_op_prev;
  static mlir::IRMapping ir_map_prev;
};

mlir::Operation *LocationSnapshotPass::module_op_prev = nullptr;
mlir::IRMapping LocationSnapshotPass::ir_map_prev = mlir::IRMapping{};
} // namespace

std::unique_ptr<Pass> mlir::createOpMutationsPass() {
  return std::make_unique<OpMutationsPass>();
}
