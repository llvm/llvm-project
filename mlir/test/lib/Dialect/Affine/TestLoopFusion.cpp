//===- TestLoopFusion.cpp - Test loop fusion ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test various loop fusion utility functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "test-loop-fusion"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

static llvm::cl::opt<bool> clTestDependenceCheck(
    "test-loop-fusion-dependence-check",
    llvm::cl::desc("Enable testing of loop fusion dependence check"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clTestSliceComputation(
    "test-loop-fusion-slice-computation",
    llvm::cl::desc("Enable testing of loop fusion slice computation"),
    llvm::cl::cat(clOptionsCategory));

static llvm::cl::opt<bool> clTestLoopFusionTransformation(
    "test-loop-fusion-transformation",
    llvm::cl::desc("Enable testing of loop fusion transformation"),
    llvm::cl::cat(clOptionsCategory));

namespace {

struct TestLoopFusion
    : public PassWrapper<TestLoopFusion, OperationPass<FuncOp>> {
  StringRef getArgument() const final { return "test-loop-fusion"; }
  StringRef getDescription() const final {
    return "Tests loop fusion utility functions.";
  }
  void runOnOperation() override;
};

} // namespace

// Run fusion dependence check on 'loops[i]' and 'loops[j]' at loop depths
// in range ['loopDepth' + 1, 'maxLoopDepth'].
// Emits a remark on 'loops[i]' if a fusion-preventing dependence exists.
// Returns false as IR is not transformed.
static bool testDependenceCheck(AffineForOp srcForOp, AffineForOp dstForOp,
                                unsigned i, unsigned j, unsigned loopDepth,
                                unsigned maxLoopDepth) {
  mlir::ComputationSliceState sliceUnion;
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    FusionResult result =
        mlir::canFuseLoops(srcForOp, dstForOp, d, &sliceUnion);
    if (result.value == FusionResult::FailBlockDependence) {
      srcForOp->emitRemark("block-level dependence preventing"
                           " fusion of loop nest ")
          << i << " into loop nest " << j << " at depth " << loopDepth;
    }
  }
  return false;
}

// Returns the index of 'op' in its block.
static unsigned getBlockIndex(Operation &op) {
  unsigned index = 0;
  for (auto &opX : *op.getBlock()) {
    if (&op == &opX)
      break;
    ++index;
  }
  return index;
}

// Returns a string representation of 'sliceUnion'.
static std::string getSliceStr(const mlir::ComputationSliceState &sliceUnion) {
  std::string result;
  llvm::raw_string_ostream os(result);
  // Slice insertion point format [loop-depth, operation-block-index]
  unsigned ipd = getNestingDepth(&*sliceUnion.insertPoint);
  unsigned ipb = getBlockIndex(*sliceUnion.insertPoint);
  os << "insert point: (" << std::to_string(ipd) << ", " << std::to_string(ipb)
     << ")";
  assert(sliceUnion.lbs.size() == sliceUnion.ubs.size());
  os << " loop bounds: ";
  for (unsigned k = 0, e = sliceUnion.lbs.size(); k < e; ++k) {
    os << '[';
    sliceUnion.lbs[k].print(os);
    os << ", ";
    sliceUnion.ubs[k].print(os);
    os << "] ";
  }
  return os.str();
}

/// Computes fusion slice union on 'loops[i]' and 'loops[j]' at loop depths
/// in range ['loopDepth' + 1, 'maxLoopDepth'].
/// Emits a string representation of the slice union as a remark on 'loops[j]'
/// and marks this as incorrect slice if the slice is invalid. Returns false as
/// IR is not transformed.
static bool testSliceComputation(AffineForOp forOpA, AffineForOp forOpB,
                                 unsigned i, unsigned j, unsigned loopDepth,
                                 unsigned maxLoopDepth) {
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    mlir::ComputationSliceState sliceUnion;
    FusionResult result = mlir::canFuseLoops(forOpA, forOpB, d, &sliceUnion);
    if (result.value == FusionResult::Success) {
      forOpB->emitRemark("slice (")
          << " src loop: " << i << ", dst loop: " << j << ", depth: " << d
          << " : " << getSliceStr(sliceUnion) << ")";
    } else if (result.value == FusionResult::FailIncorrectSlice) {
      forOpB->emitRemark("Incorrect slice (")
          << " src loop: " << i << ", dst loop: " << j << ", depth: " << d
          << " : " << getSliceStr(sliceUnion) << ")";
    }
  }
  return false;
}

// Attempts to fuse 'forOpA' into 'forOpB' at loop depths in range
// ['loopDepth' + 1, 'maxLoopDepth'].
// Returns true if loops were successfully fused, false otherwise.
static bool testLoopFusionTransformation(AffineForOp forOpA, AffineForOp forOpB,
                                         unsigned i, unsigned j,
                                         unsigned loopDepth,
                                         unsigned maxLoopDepth) {
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    mlir::ComputationSliceState sliceUnion;
    FusionResult result = mlir::canFuseLoops(forOpA, forOpB, d, &sliceUnion);
    if (result.value == FusionResult::Success) {
      mlir::fuseLoops(forOpA, forOpB, sliceUnion);
      // Note: 'forOpA' is removed to simplify test output. A proper loop
      // fusion pass should check the data dependence graph and run memref
      // region analysis to ensure removing 'forOpA' is safe.
      forOpA.erase();
      return true;
    }
  }
  return false;
}

using LoopFunc = function_ref<bool(AffineForOp, AffineForOp, unsigned, unsigned,
                                   unsigned, unsigned)>;

// Run tests on all combinations of src/dst loop nests in 'depthToLoops'.
// If 'return_on_change' is true, returns on first invocation of 'fn' which
// returns true.
static bool iterateLoops(ArrayRef<SmallVector<AffineForOp, 2>> depthToLoops,
                         LoopFunc fn, bool returnOnChange = false) {
  bool changed = false;
  for (unsigned loopDepth = 0, end = depthToLoops.size(); loopDepth < end;
       ++loopDepth) {
    auto &loops = depthToLoops[loopDepth];
    unsigned numLoops = loops.size();
    for (unsigned j = 0; j < numLoops; ++j) {
      for (unsigned k = 0; k < numLoops; ++k) {
        if (j != k)
          changed |=
              fn(loops[j], loops[k], j, k, loopDepth, depthToLoops.size());
        if (changed && returnOnChange)
          return true;
      }
    }
  }
  return changed;
}

void TestLoopFusion::runOnOperation() {
  std::vector<SmallVector<AffineForOp, 2>> depthToLoops;
  if (clTestLoopFusionTransformation) {
    // Run loop fusion until a fixed point is reached.
    do {
      depthToLoops.clear();
      // Gather all AffineForOps by loop depth.
      gatherLoops(getOperation(), depthToLoops);

      // Try to fuse all combinations of src/dst loop nests in 'depthToLoops'.
    } while (iterateLoops(depthToLoops, testLoopFusionTransformation,
                          /*returnOnChange=*/true));
    return;
  }

  // Gather all AffineForOps by loop depth.
  gatherLoops(getOperation(), depthToLoops);

  // Run tests on all combinations of src/dst loop nests in 'depthToLoops'.
  if (clTestDependenceCheck)
    iterateLoops(depthToLoops, testDependenceCheck);
  if (clTestSliceComputation)
    iterateLoops(depthToLoops, testSliceComputation);
}

namespace mlir {
namespace test {
void registerTestLoopFusion() { PassRegistration<TestLoopFusion>(); }
} // namespace test
} // namespace mlir
