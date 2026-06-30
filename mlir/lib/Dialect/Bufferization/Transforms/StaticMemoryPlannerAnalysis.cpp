//===- StaticMemoryPlannerAnalysis.cpp - Static memory planning -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms memref.alloc/memref.dealloc pairs into a single arena allocation
// with memref.view. Uses simple sequential offset assignment where each
// allocation gets its own space without overlap (baseline algorithm).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
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

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

/// Allocation info for memory planning (independent of MLIR).
/// This can be used with pure planning algorithms.
struct Alloc {
  int64_t sizeInBytes = 0; // Size in bytes
  int64_t alignment = 1;   // Required alignment in bytes
  int64_t timeStart = 0;   // Operation index when allocation starts
  int64_t timeEnd = 0;     // Operation index when allocation ends (dealloc)
};

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

/// Align an offset to the specified alignment.
/// Returns the smallest value >= offset that is a multiple of alignment.
static int64_t alignOffset(int64_t offset, int64_t alignment) {
  return llvm::alignTo(offset, alignment);
}

//===----------------------------------------------------------------------===//
// Memory Planning Algorithms
//===----------------------------------------------------------------------===//

/// Simple sequential memory planner (baseline algorithm).
/// arenaAlignment must be a multiple (LCM) of all alloc.alignment values.
/// Allocates each buffer one after another with proper alignment padding.
/// Returns offsets in bytes for each allocation.
static SmallVector<int64_t> trivialMemoryPlanner(int64_t arenaAlignment,
                                                 ArrayRef<Alloc> allocs) {
  SmallVector<int64_t> offsets;
  int64_t currentOffset = 0;

  for (const auto &alloc : allocs) {
    currentOffset = alignOffset(currentOffset, alloc.alignment);
#ifndef NDEBUG
    assert((arenaAlignment + currentOffset) % alloc.alignment == 0 &&
           "invalid alignment");
#endif
    offsets.push_back(currentOffset);
    currentOffset += alloc.sizeInBytes;
  }

  return offsets;
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

  void runOnOperation() override {
    auto funcOp = llvm::cast<FunctionOpInterface>(getOperation());

    // Step 0: Check for memref return types (not supported)
    for (Type resultType : funcOp.getResultTypes()) {
      if (isa<BaseMemRefType>(resultType)) {
        funcOp->emitError("static-memory-planner does not support functions "
                          "with memref return types");
        return signalPassFailure();
      }
    }

    // Step 1: Collect eligible allocation candidates
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

    if (candidates.empty())
      return;

    // Step 2: Prepare allocation info for planner
    SmallVector<Alloc> allocInfos;
    int64_t arenaAlignment = 1;
    for (const auto &candidate : candidates) {
      Alloc allocInfo;
      allocInfo.sizeInBytes = candidate.sizeInBytes;
      allocInfo.alignment = candidate.alignment;
      allocInfos.push_back(allocInfo);
      arenaAlignment = std::lcm(arenaAlignment, candidate.alignment);
    }

    // Step 3: Run the planning algorithm
    SmallVector<int64_t> offsets =
        trivialMemoryPlanner(arenaAlignment, allocInfos);

    // Assign computed offsets back to candidates
    int64_t totalSize = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      candidates[i].offset = offsets[i];
      totalSize = std::max(totalSize, offsets[i] + candidates[i].sizeInBytes);
      LLVM_DEBUG(llvm::dbgs()
                 << "[static-memory-planner] offset=" << candidates[i].offset
                 << " size=" << candidates[i].sizeInBytes
                 << " alignment=" << candidates[i].alignment << "\n");
    }

    // Step 4: Obtain arena based on arena mode
    Operation *firstAlloc = candidates.front().alloc;
    OpBuilder builder(firstAlloc);
    Value arenaValue;

    if (arenaMode == "allocate") {
      // Arena is i8 byte buffer to support multiple data types (f32, i64, etc.)
      auto arenaType = MemRefType::get({totalSize}, builder.getI8Type());
      auto arenaAlloc = memref::AllocOp::create(
          builder, firstAlloc->getLoc(), arenaType, ValueRange{},
          builder.getI64IntegerAttr(arenaAlignment));
      arenaValue = arenaAlloc.getResult();

      LLVM_DEBUG(llvm::dbgs()
                 << "[static-memory-planner] created arena via AllocOp: size="
                 << totalSize << " bytes, alignment=" << arenaAlignment
                 << " bytes\n");
    } else if (arenaMode == "arg") {
      if (funcOp.getNumArguments() == 0) {
        funcOp->emitError(
            "arena-mode=arg requires at least one function argument");
        return signalPassFailure();
      }

      arenaValue = funcOp.getArgument(0);
      auto arenaType = dyn_cast<MemRefType>(arenaValue.getType());
      if (!arenaType || !arenaType.getElementType().isInteger(8) ||
          arenaType.getRank() != 1) {
        funcOp->emitError(
            "arena-mode=arg requires first argument to be memref<...xi8>");
        return signalPassFailure();
      }

      LLVM_DEBUG(
          llvm::dbgs()
          << "[static-memory-planner] using arena from function arg 0\n");
    } else {
      funcOp->emitError("invalid arena-mode: '" + arenaMode +
                        "' (must be 'allocate' or 'arg')");
      return signalPassFailure();
    }

    // Step 5: Replace each alloc with memref.view directly on arena
    for (auto &candidate : candidates) {
      builder.setInsertionPoint(candidate.alloc);
      Location loc = candidate.alloc.getLoc();

      MemRefType originalType = candidate.alloc.getType();

      // Create a constant for the byte offset into the arena
      Value offsetIndex =
          arith::ConstantIndexOp::create(builder, loc, candidate.offset);

      // Use memref.view to create a typed view into the i8 arena
      auto view = memref::ViewOp::create(builder, loc, originalType, arenaValue,
                                         offsetIndex, SmallVector<Value>{});

      candidate.alloc.getResult().replaceAllUsesWith(view.getResult());
      candidate.alloc.erase();
      candidate.dealloc.erase();
    }
  }
};

} // end anonymous namespace
