//===- StaticMemoryPlannerAnalysis.cpp - Static memory planning -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms memref.alloc/memref.dealloc pairs into a single arena allocation
// with memref.view. Delegates offset computation to planning algorithms in
// StaticMemoryPlanning.h.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/StaticMemoryPlanning.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "static-memory-planner"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_STATICMEMORYPLANNERANALYSISPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;

namespace {

/// A candidate allocation with its matching deallocation(s) and assigned
/// offset. An alloc may be freed indirectly through arith.select chains,
/// yielding multiple potential deallocs — all must be in the same block.
struct AllocationCandidate {
  memref::AllocOp alloc;
  SmallVector<memref::DeallocOp> deallocs;
  int64_t offset = 0; // Offset in bytes from arena start (assigned by planner)
  int64_t sizeInBytes = 0; // Size in bytes
  int64_t alignment = 1;   // Required alignment in bytes
};

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Collect all dealloc ops that might free the given value, following ops
/// that implement BufferViewFlowOpInterface (e.g. arith.select). For example:
///   %0 = memref.alloc()
///   %2 = arith.select %c, %0, %1
///   memref.dealloc %2    <- this covers %0 conditionally
/// `visited` prevents cycles in the use-def graph.
///
/// TODO: This relies on BufferViewFlowOpInterface external models being
/// registered for the ops in the IR (e.g. via
/// arith::registerBufferViewFlowOpInterfaceExternalModels).
static void findPotentialDeallocs(Value value,
                                  SmallVectorImpl<memref::DeallocOp> &deallocs,
                                  SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  for (Operation *user : value.getUsers()) {
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocs.push_back(dealloc);
    } else if (dyn_cast<bufferization::BufferViewFlowOpInterface>(user)) {
      // Follow any op that propagates buffer values to its results.
      for (Value result : user->getResults())
        if (isa<MemRefType>(result.getType()))
          findPotentialDeallocs(result, deallocs, visited);
    }
  }
}

/// Compute the size in bytes for a memref type.
static int64_t computeSizeInBytes(MemRefType memrefType) {
  int64_t numElements = memrefType.getNumElements();
  unsigned elementSizeInBits = memrefType.getElementTypeBitWidth();
  return (numElements * elementSizeInBits + 7) / 8; // Round up to bytes
}

/// Build lifetime-annotated allocation descriptors from candidates.
/// Returns the arena alignment (LCM of all individual alignments).
/// Uses a single block scan (O(n+m)) instead of one scan per candidate.
static int64_t buildAllocInfos(
    MutableArrayRef<AllocationCandidate> candidates,
    SmallVectorImpl<bufferization::MemoryPlannerAlloc> &allocInfos) {
  // Build an op-index map with a single pass over the block.
  DenseMap<Operation *, int64_t> opIndex;
  if (!candidates.empty()) {
    Block *block = candidates.front().alloc->getBlock();
    int64_t idx = 0;
    for (Operation &op : *block)
      opIndex[&op] = idx++;
  }

  int64_t arenaAlignment = 1;
  for (auto &candidate : candidates) {
    bufferization::MemoryPlannerAlloc info;
    info.sizeInBytes = candidate.sizeInBytes;
    info.alignment = candidate.alignment;
    info.timeStart = opIndex.lookup(candidate.alloc.getOperation());
    // Conservative: timeEnd = latest dealloc index among all potential
    // deallocs.
    int64_t timeEnd = 0;
    for (memref::DeallocOp d : candidate.deallocs)
      timeEnd = std::max(timeEnd, opIndex.lookup(d.getOperation()));
    info.timeEnd = timeEnd;
    allocInfos.push_back(info);
    arenaAlignment = std::lcm(arenaAlignment, candidate.alignment);
  }
  return arenaAlignment;
}

/// Collect alloc/dealloc pairs eligible for arena placement.
/// An allocation is eligible if it has a static shape and its deallocs
/// (including those reached via BufferViewFlowOpInterface chains)
/// are in the same block. Allocations with dynamic shapes are skipped.
/// Missing deallocs or cross-block deallocs are reported as errors.
static LogicalResult
collectCandidates(FunctionOpInterface funcOp, llvm::Statistic &numSkipDynamic,
                  llvm::Statistic &numEligible,
                  SmallVector<AllocationCandidate> &candidates) {
  bool walkFailed = false;
  funcOp->walk([&](memref::AllocOp allocOp) -> WalkResult {
    MemRefType memrefType = allocOp.getType();
    if (!memrefType.hasStaticShape()) {
      ++numSkipDynamic;
      return WalkResult::advance();
    }

    SmallVector<memref::DeallocOp> deallocs;
    SmallPtrSet<Value, 8> visited;
    findPotentialDeallocs(allocOp.getResult(), deallocs, visited);

    if (deallocs.empty()) {
      allocOp.emitError("no dealloc found; run the deallocation pipeline "
                        "before this pass");
      walkFailed = true;
      return WalkResult::interrupt();
    }

    for (memref::DeallocOp d : deallocs) {
      if (d->getBlock() != allocOp->getBlock()) {
        allocOp.emitError("dealloc is in a different block than the alloc; "
                          "run the deallocation pipeline before this pass");
        walkFailed = true;
        return WalkResult::interrupt();
      }
    }

    ++numEligible;
    AllocationCandidate candidate;
    candidate.alloc = allocOp;
    candidate.deallocs = deallocs;
    candidate.sizeInBytes = computeSizeInBytes(memrefType);
    candidate.alignment = allocOp.getAlignment().value_or(1);
    candidates.push_back(candidate);
    return WalkResult::advance();
  });

  return failure(walkFailed);
}

/// Create or obtain the arena buffer based on the arena mode.
/// Returns failure if the mode is invalid or preconditions aren't met.
static FailureOr<Value> createArena(OpBuilder &builder,
                                    FunctionOpInterface funcOp,
                                    StringRef arenaMode, int64_t totalSize,
                                    int64_t arenaAlignment) {
  Location loc = funcOp->getLoc();

  if (arenaMode == "allocate") {
    auto arenaType = MemRefType::get({totalSize}, builder.getI8Type());
    auto arenaAlloc =
        memref::AllocOp::create(builder, loc, arenaType, ValueRange{},
                                builder.getI64IntegerAttr(arenaAlignment));
    LLVM_DEBUG(llvm::dbgs()
               << "[static-memory-planner] created arena via AllocOp: size="
               << totalSize << " bytes, alignment=" << arenaAlignment
               << " bytes\n");
    return arenaAlloc.getResult();
  }

  if (arenaMode == "arg") {
    if (funcOp.getNumArguments() == 0)
      return funcOp->emitError(
          "arena-mode=arg requires at least one function argument");

    Value arenaValue = funcOp.getArgument(0);
    auto arenaType = dyn_cast<MemRefType>(arenaValue.getType());
    if (!arenaType || !arenaType.getElementType().isInteger(8) ||
        arenaType.getRank() != 1)
      return funcOp->emitError(
          "arena-mode=arg requires first argument to be memref<...xi8>");

    LLVM_DEBUG(llvm::dbgs()
               << "[static-memory-planner] using arena from function arg 0\n");
    return arenaValue;
  }

  return funcOp->emitError("invalid arena-mode: '" + arenaMode +
                           "' (must be 'allocate' or 'arg')");
}

/// Replace each alloc/dealloc pair with a memref.view into the arena.
static void rewriteAllocations(MutableArrayRef<AllocationCandidate> candidates,
                               Value arenaValue) {
  SmallPtrSet<Operation *, 8> deallocsToErase;
  SmallVector<Operation *> allocsToErase;

  // Replace all alloc results with views (rewires selects too).
  for (auto &candidate : candidates) {
    OpBuilder builder(candidate.alloc);
    Location loc = candidate.alloc.getLoc();
    MemRefType originalType = candidate.alloc.getType();

    Value offsetIndex =
        arith::ConstantIndexOp::create(builder, loc, candidate.offset);
    auto view = memref::ViewOp::create(builder, loc, originalType, arenaValue,
                                       offsetIndex, SmallVector<Value>{});
    candidate.alloc.getResult().replaceAllUsesWith(view.getResult());
    allocsToErase.push_back(candidate.alloc.getOperation());

    for (memref::DeallocOp d : candidate.deallocs)
      deallocsToErase.insert(d.getOperation());
  }

  // Erase deallocs first (they may reference alloc results via selects).
  for (Operation *d : deallocsToErase)
    d->erase();

  // Erase allocs last (no users remain after replaceAllUsesWith).
  for (Operation *allocOp : allocsToErase)
    allocOp->erase();
}

//===----------------------------------------------------------------------===//
// StaticMemoryPlannerAnalysisPass
//===----------------------------------------------------------------------===//

struct StaticMemoryPlannerAnalysisPass
    : public bufferization::impl::StaticMemoryPlannerAnalysisPassBase<
          StaticMemoryPlannerAnalysisPass> {
public:
  using Base = bufferization::impl::StaticMemoryPlannerAnalysisPassBase<
      StaticMemoryPlannerAnalysisPass>;
  using Base::Base;

  void runOnOperation() override;
};

void StaticMemoryPlannerAnalysisPass::runOnOperation() {
  auto funcOp = llvm::cast<FunctionOpInterface>(getOperation());

  // Step 0: Check for memref return types (not supported)
  for (Type resultType : funcOp.getResultTypes()) {
    if (isa<BaseMemRefType>(resultType)) {
      funcOp->emitError("static-memory-planner does not support functions "
                        "with memref return types");
      return signalPassFailure();
    }
  }

  // Step 1: Collect eligible allocation candidates.
  SmallVector<AllocationCandidate> candidates;
  if (failed(
          collectCandidates(funcOp, numSkipDynamic, numEligible, candidates)))
    return signalPassFailure();

  if (candidates.empty())
    return;

  // Step 2: Build allocation descriptors with lifetime info.
  SmallVector<bufferization::MemoryPlannerAlloc> allocInfos;
  int64_t arenaAlignment = buildAllocInfos(candidates, allocInfos);

  // Step 3: Run the planning algorithm.
  SmallVector<int64_t> offsets;
  switch (algorithm) {
  case bufferization::MemoryPlannerAlgorithm::Trivial:
    offsets = bufferization::trivialMemoryPlanner(arenaAlignment, allocInfos);
    break;
  case bufferization::MemoryPlannerAlgorithm::BestFit:
    offsets = bufferization::bestFitMemoryPlanner(arenaAlignment, allocInfos);
    break;
  }

  // Step 4: Compute total arena size and assign offsets.
  int64_t totalSize = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    candidates[i].offset = offsets[i];
    totalSize = std::max(totalSize, offsets[i] + candidates[i].sizeInBytes);
    LLVM_DEBUG(llvm::dbgs()
               << "[static-memory-planner] offset=" << candidates[i].offset
               << " size=" << candidates[i].sizeInBytes
               << " alignment=" << candidates[i].alignment << "\n");
  }

  // Step 5: Obtain arena based on arena mode.
  Operation *firstAlloc = candidates.front().alloc;
  OpBuilder builder(firstAlloc);
  FailureOr<Value> arenaValue =
      createArena(builder, funcOp, arenaMode, totalSize, arenaAlignment);
  if (failed(arenaValue))
    return signalPassFailure();

  // Step 6: Replace each alloc with memref.view into the arena.
  rewriteAllocations(candidates, *arenaValue);
}

} // end anonymous namespace
