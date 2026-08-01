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
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/StaticMemoryPlanning.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

/// A candidate allocation with its matching deallocation and assigned offset.
struct AllocationCandidate {
  memref::AllocOp alloc;
  memref::DeallocOp dealloc;
  int64_t offset = 0; // Offset in bytes from arena start (assigned by planner)
  int64_t sizeInBytes = 0; // Size in bytes
  int64_t alignment = 1;   // Required alignment in bytes
};

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Finds the unique dealloc operation for a given alloc value.
/// Returns nullptr if there are zero or multiple deallocs.
static memref::DeallocOp findUniqueDealloc(Value allocValue) {
  memref::DeallocOp deallocOp = nullptr;
  for (Operation *user : allocValue.getUsers()) {
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      if (deallocOp)
        return nullptr; // Multiple deallocs found
      deallocOp = dealloc;
    }
  }
  return deallocOp;
}

/// Compute the size in bytes for a memref type.
static int64_t computeSizeInBytes(MemRefType memrefType) {
  int64_t numElements = memrefType.getNumElements();
  unsigned elementSizeInBits = memrefType.getElementTypeBitWidth();
  return (numElements * elementSizeInBits + 7) / 8; // Round up to bytes
}

/// Build lifetime-annotated allocation descriptors from candidates.
/// Returns the arena alignment (LCM of all individual alignments).
static int64_t buildAllocInfos(
    MutableArrayRef<AllocationCandidate> candidates,
    SmallVectorImpl<bufferization::MemoryPlannerAlloc> &allocInfos) {
  int64_t arenaAlignment = 1;
  for (auto &candidate : candidates) {
    bufferization::MemoryPlannerAlloc info;
    info.sizeInBytes = candidate.sizeInBytes;
    info.alignment = candidate.alignment;

    Block *block = candidate.alloc->getBlock();
    int64_t opIdx = 0;
    for (Operation &op : *block) {
      if (&op == candidate.alloc.getOperation())
        info.timeStart = opIdx;
      if (&op == candidate.dealloc.getOperation())
        info.timeEnd = opIdx;
      ++opIdx;
    }

    allocInfos.push_back(info);
    arenaAlignment = std::lcm(arenaAlignment, candidate.alignment);
  }
  return arenaAlignment;
}

/// Collect alloc/dealloc pairs eligible for arena placement.
/// An allocation is eligible if it has a static shape and a unique dealloc
/// in the same block.
static SmallVector<AllocationCandidate>
collectCandidates(FunctionOpInterface funcOp, llvm::Statistic &numSkipDynamic,
                  llvm::Statistic &numSkipNoDealloc,
                  llvm::Statistic &numEligible) {
  SmallVector<AllocationCandidate> candidates;

  funcOp->walk([&](memref::AllocOp allocOp) {
    MemRefType memrefType = allocOp.getType();

    // Skip dynamic shapes
    if (!memrefType.hasStaticShape()) {
      ++numSkipDynamic;
      return;
    }

    // Find unique dealloc in the same block
    memref::DeallocOp deallocOp = findUniqueDealloc(allocOp.getResult());
    if (!deallocOp) {
      ++numSkipNoDealloc;
      return;
    }

    if (deallocOp->getBlock() != allocOp->getBlock()) {
      ++numSkipNoDealloc;
      return;
    }

    // This allocation is eligible
    ++numEligible;
    AllocationCandidate candidate;
    candidate.alloc = allocOp;
    candidate.dealloc = deallocOp;
    candidate.sizeInBytes = computeSizeInBytes(memrefType);
    candidate.alignment = allocOp.getAlignment().value_or(1);
    candidates.push_back(candidate);
  });

  return candidates;
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
  for (auto &candidate : candidates) {
    OpBuilder builder(candidate.alloc);
    Location loc = candidate.alloc.getLoc();
    MemRefType originalType = candidate.alloc.getType();

    Value offsetIndex =
        arith::ConstantIndexOp::create(builder, loc, candidate.offset);
    auto view = memref::ViewOp::create(builder, loc, originalType, arenaValue,
                                       offsetIndex, SmallVector<Value>{});

    candidate.alloc.getResult().replaceAllUsesWith(view.getResult());
    candidate.alloc.erase();
    candidate.dealloc.erase();
  }
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
  SmallVector<AllocationCandidate> candidates =
      collectCandidates(funcOp, numSkipDynamic, numSkipNoDealloc, numEligible);

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
