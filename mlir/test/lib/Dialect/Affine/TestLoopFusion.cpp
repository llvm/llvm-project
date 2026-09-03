//===- TestLoopFusion.cpp - Test loop fusion ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to test various loop fusion utilities. It is not
// meant to be a pass to perform valid fusion.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-loop-fusion"

using namespace mlir;
using namespace mlir::affine;

namespace {

struct TestLoopFusion
    : public PassWrapper<TestLoopFusion, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoopFusion)

  StringRef getArgument() const final { return "test-loop-fusion"; }
  StringRef getDescription() const final {
    return "Tests loop fusion utility functions.";
  }
  void runOnOperation() override;

  TestLoopFusion() = default;
  TestLoopFusion(const TestLoopFusion &pass) : PassWrapper(pass){};

  Option<bool> clTestDependenceCheck{
      *this, "test-loop-fusion-dependence-check",
      llvm::cl::desc("Enable testing of loop fusion dependence check"),
      llvm::cl::init(false)};

  Option<bool> clTestSliceComputation{
      *this, "test-loop-fusion-slice-computation",
      llvm::cl::desc("Enable testing of loop fusion slice computation"),
      llvm::cl::init(false)};

  Option<bool> clTestLoopFusionUtilities{
      *this, "test-loop-fusion-utilities",
      llvm::cl::desc("Enable testing of loop fusion transformation utilities"),
      llvm::cl::init(false)};

  Option<bool> clTestEdgeKinds{
      *this, "test-loop-fusion-edge-kinds",
      llvm::cl::desc("Enable testing of dependence graph edge kinds"),
      llvm::cl::init(false)};

  Option<bool> clTestFusionFailure{
      *this, "test-loop-fusion-failure",
      llvm::cl::desc("Enable testing of failed producer-consumer fusion"),
      llvm::cl::init(false)};
};

} // namespace

// Run fusion dependence check on 'loops[i]' and 'loops[j]' at loop depths
// in range ['loopDepth' + 1, 'maxLoopDepth'].
// Emits a remark on 'loops[i]' if a fusion-preventing dependence exists.
// Returns false as IR is not transformed.
static bool testDependenceCheck(AffineForOp srcForOp, AffineForOp dstForOp,
                                unsigned i, unsigned j, unsigned loopDepth,
                                unsigned maxLoopDepth) {
  ComputationSliceState sliceUnion;
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    FusionResult result = canFuseLoops(srcForOp, dstForOp, d, &sliceUnion);
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
static std::string getSliceStr(const ComputationSliceState &sliceUnion) {
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
    ComputationSliceState sliceUnion;
    FusionResult result = canFuseLoops(forOpA, forOpB, d, &sliceUnion);
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
// Returns true if loops were successfully fused, false otherwise. This tests
// `fuseLoops` and `canFuseLoops` utilities.
static bool testLoopFusionUtilities(AffineForOp forOpA, AffineForOp forOpB,
                                    unsigned i, unsigned j, unsigned loopDepth,
                                    unsigned maxLoopDepth) {
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    ComputationSliceState sliceUnion;
    // This check isn't a sufficient one, but necessary.
    FusionResult result = canFuseLoops(forOpA, forOpB, d, &sliceUnion);
    if (result.value != FusionResult::Success)
      continue;
    fuseLoops(forOpA, forOpB, sliceUnion);
    // Note: 'forOpA' is removed to simplify test output. A proper loop
    // fusion pass should perform additional checks to check safe removal.
    if (forOpA.use_empty())
      forOpA.erase();
    return true;
  }
  return false;
}

// Verify that an unanalyzable dependence is reported by the
// producer-consumer legality check before slice computation is attempted.
// This distinguishes a failed dependence proof from a later generic slice
// failure, which would otherwise leave the same IR unchanged for both cases.
static bool testFusionFailure(AffineForOp forOpA, AffineForOp forOpB, unsigned,
                              unsigned, unsigned loopDepth,
                              unsigned maxLoopDepth) {
  if (forOpA->getBlock() != forOpB->getBlock() ||
      !forOpA->isBeforeInBlock(forOpB))
    return false;

  FusionStrategy strategy(FusionStrategy::ProducerConsumer);
  for (unsigned d = loopDepth + 1; d <= maxLoopDepth; ++d) {
    ComputationSliceState sliceUnion;
    FusionResult result =
        canFuseLoops(forOpA, forOpB, d, &sliceUnion, strategy);
    if (result.value == FusionResult::FailFusionDependence) {
      forOpA->emitRemark("fusion dependence prevents fusion");
      return false;
    }
  }
  return false;
}

// Exercises the distinction between storage and exact-SSA graph edges. Keep
// both edge kinds on the same value so that a query or count that accidentally
// assumes one kind cannot pass by observing a different value.
static bool testEdgeKinds(func::FuncOp funcOp) {
  Operation *srcOp = nullptr;
  Operation *dstOp = nullptr;
  Value memref;
  for (Operation &op : funcOp.getBody().front()) {
    if (op.getNumResults() == 0 ||
        !isa<BaseMemRefType>(op.getResult(0).getType()))
      continue;
    if (!srcOp) {
      srcOp = &op;
      memref = op.getResult(0);
    } else {
      dstOp = &op;
      break;
    }
  }

  if (!srcOp || !dstOp) {
    funcOp.emitError("edge-kind test requires two memref-producing operations");
    return false;
  }

  MemRefDependenceGraph graph(funcOp.getBody().front());
  unsigned srcId = graph.addNode(srcOp);
  unsigned dstId = graph.addNode(dstOp);

  auto checkState = [&](StringRef label, bool any, bool memory, bool ssa,
                        unsigned edgeCount) {
    bool anyEdge = graph.hasEdge(srcId, dstId, memref);
    bool memoryEdge = graph.hasEdge(srcId, dstId, memref,
                                    MemRefDependenceGraph::Edge::Kind::Memory);
    bool ssaEdge = graph.hasEdge(srcId, dstId, memref,
                                 MemRefDependenceGraph::Edge::Kind::SSA);
    bool result = anyEdge == any && memoryEdge == memory && ssaEdge == ssa &&
                  graph.outEdges.lookup(srcId).size() == edgeCount &&
                  graph.inEdges.lookup(dstId).size() == edgeCount &&
                  graph.memrefEdgeCount.lookup(memref) == edgeCount;
    llvm::outs() << "edge-kinds " << label << " any=" << (anyEdge ? 1 : 0)
                 << " memory=" << (memoryEdge ? 1 : 0)
                 << " ssa=" << (ssaEdge ? 1 : 0)
                 << " out=" << graph.outEdges.lookup(srcId).size()
                 << " in=" << graph.inEdges.lookup(dstId).size()
                 << " count=" << graph.memrefEdgeCount.lookup(memref) << "\n";
    return result;
  };

  graph.addEdge(srcId, dstId, memref, MemRefDependenceGraph::Edge::Kind::SSA);
  if (!checkState("ssa-only", true, false, true, 1))
    return false;

  graph.addEdge(srcId, dstId, memref,
                MemRefDependenceGraph::Edge::Kind::Memory);
  if (!checkState("both", true, true, true, 2))
    return false;

  graph.removeEdge(srcId, dstId, memref,
                   MemRefDependenceGraph::Edge::Kind::Memory);
  if (!checkState("ssa-after-memory-removal", true, false, true, 1))
    return false;

  graph.removeEdge(srcId, dstId, memref,
                   MemRefDependenceGraph::Edge::Kind::SSA);
  return checkState("empty", false, false, false, 0);
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
  if (clTestEdgeKinds) {
    if (!testEdgeKinds(getOperation()))
      signalPassFailure();
    return;
  }
  if (clTestFusionFailure) {
    gatherLoops(getOperation(), depthToLoops);
    iterateLoops(depthToLoops, testFusionFailure);
    return;
  }
  if (clTestLoopFusionUtilities) {
    // Run loop fusion until a fixed point is reached.
    do {
      depthToLoops.clear();
      // Gather all AffineForOps by loop depth.
      gatherLoops(getOperation(), depthToLoops);

      // Try to fuse all combinations of src/dst loop nests in 'depthToLoops'.
    } while (iterateLoops(depthToLoops, testLoopFusionUtilities,
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
