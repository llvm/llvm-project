//===- StaticMemoryPlannerAnalysis.cpp - Static memory planning -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms memref.alloc/memref.dealloc pairs into a single arena allocation
// with subviews. Uses simple sequential offset assignment where each allocation
// gets its own space without overlap (baseline algorithm for e2e testing).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "static-memory-planner"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_STATICMEMORYPLANNERANALYSISPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

/// Allocation info for memory planning (independent of MLIR).
/// This can be used with pure planning algorithms.
struct Alloc {
  int64_t sizeInBytes = 0;    // Size in bytes
  int64_t alignment = 1;      // Required alignment in bytes
  // Note: time_start and time_end will be added later for lifetime-aware planning
};

/// A candidate allocation with its matching deallocation and assigned offset.
struct AllocationCandidate {
  mlir::memref::AllocOp alloc;
  mlir::memref::DeallocOp dealloc;
  int64_t offset = 0;         // Offset in bytes from arena start (assigned by planner)
  int64_t sizeInBytes = 0;    // Size in bytes
  int64_t alignment = 1;      // Required alignment in bytes
};

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

/// Finds the unique dealloc operation for a given alloc value.
/// Returns nullptr if there are zero or multiple deallocs.
static mlir::memref::DeallocOp findUniqueDealloc(mlir::Value allocValue) {
  mlir::memref::DeallocOp deallocOp = nullptr;
  for (mlir::Operation *user : allocValue.getUsers()) {
    if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
      if (deallocOp)
        return nullptr; // Multiple deallocs found
      deallocOp = dealloc;
    }
  }
  return deallocOp;
}

/// Compute the number of elements in a static-shape memref.
static int64_t computeSizeInElements(mlir::MemRefType memrefType) {
  int64_t size = 1;
  for (int64_t dim : memrefType.getShape())
    size *= dim;
  return size;
}

/// Compute the size in bytes for a memref type.
static int64_t computeSizeInBytes(mlir::MemRefType memrefType) {
  int64_t numElements = computeSizeInElements(memrefType);
  unsigned elementSizeInBits = memrefType.getElementTypeBitWidth();
  return (numElements * elementSizeInBits + 7) / 8; // Round up to bytes
}

/// Align an offset to the specified alignment.
/// Returns the smallest value >= offset that is a multiple of alignment.
static int64_t alignOffset(int64_t offset, int64_t alignment) {
  if (alignment <= 1)
    return offset;
  return (offset + alignment - 1) / alignment * alignment;
}

//===----------------------------------------------------------------------===//
// Memory Planning Algorithms
//===----------------------------------------------------------------------===//

/// Simple sequential memory planner (baseline algorithm).
/// Allocates each buffer one after another with proper alignment padding.
/// Returns offsets in bytes for each allocation.
static llvm::SmallVector<int64_t>
trivialMemoryPlanner(int64_t arenaAlignment,
                     llvm::ArrayRef<Alloc> allocs) {
  llvm::SmallVector<int64_t> offsets;
  int64_t currentOffset = 0;
  
  for (const auto &alloc : allocs) {
    // Ensure offset respects both arena alignment and this allocation's alignment
    // The comment from mentor: (arenaAlignment + offset) % alloc.alignment == 0
    // This means: if arena starts at an arenaAlignment boundary,
    // then offset must make the final address properly aligned for alloc.alignment
    currentOffset = alignOffset(currentOffset, alloc.alignment);
    offsets.push_back(currentOffset);
    currentOffset += alloc.sizeInBytes;
  }
  
  return offsets;
}

//===----------------------------------------------------------------------===//
// StaticMemoryPlannerAnalysisPass
//===----------------------------------------------------------------------===//

struct StaticMemoryPlannerAnalysisPass
    : public mlir::bufferization::impl::StaticMemoryPlannerAnalysisPassBase<
          StaticMemoryPlannerAnalysisPass> {
public:
  StaticMemoryPlannerAnalysisPass() = default;

  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    // Step 1: Collect eligible allocation candidates
    llvm::SmallVector<AllocationCandidate> candidates;

    op->walk([&](mlir::memref::AllocOp allocOp) {
      mlir::MemRefType memrefType = allocOp.getType();

      // Skip dynamic shapes
      if (!memrefType.hasStaticShape()) {
        ++numSkipDynamic;
        return;
      }

      // Find unique dealloc in the same block
      mlir::memref::DeallocOp deallocOp = findUniqueDealloc(allocOp.getResult());
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
      // Extract alignment requirement (default to 1 if not specified)
      candidate.alignment = allocOp.getAlignment().value_or(1);
      candidates.push_back(candidate);
    });

    if (candidates.empty())
      return;

    // Step 2: Prepare allocation info for planner
    llvm::SmallVector<Alloc> allocInfos;
    int64_t maxAlignment = 1;
    for (const auto &candidate : candidates) {
      Alloc allocInfo;
      allocInfo.sizeInBytes = candidate.sizeInBytes;
      allocInfo.alignment = candidate.alignment;
      allocInfos.push_back(allocInfo);
      maxAlignment = std::max(maxAlignment, candidate.alignment);
    }

    // Step 3: Run the planning algorithm
    llvm::SmallVector<int64_t> offsets = 
        trivialMemoryPlanner(maxAlignment, allocInfos);
    
    // Assign computed offsets back to candidates
    int64_t totalSize = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      candidates[i].offset = offsets[i];
      totalSize = std::max(totalSize, offsets[i] + candidates[i].sizeInBytes);
      LLVM_DEBUG(llvm::dbgs() << "[static-memory-planner] offset="
                              << candidates[i].offset
                              << " size=" << candidates[i].sizeInBytes
                              << " alignment=" << candidates[i].alignment << "\n");
    }

    // Step 4: Find the first allocation's location to place arena
    mlir::Operation *firstAlloc = candidates.front().alloc;
    mlir::OpBuilder builder(firstAlloc);
    
    // Step 5: Create arena allocation as i8 byte buffer
    // This allows the same arena to hold multiple data types (f32, i64, etc.)
    auto i8Type = builder.getI8Type();
    auto arenaType = mlir::MemRefType::get({totalSize}, i8Type);
    auto arenaAlloc = mlir::memref::AllocOp::create(builder, firstAlloc->getLoc(), arenaType);
    arenaAlloc.setAlignmentAttr(builder.getI64IntegerAttr(maxAlignment));

    LLVM_DEBUG(llvm::dbgs() << "[static-memory-planner] created arena: size="
                            << totalSize << " bytes, alignment="
                            << maxAlignment << " bytes\n");

    // Step 6: Replace each alloc with memref.view directly on arena
    for (auto &candidate : candidates) {
      mlir::OpBuilder viewBuilder(candidate.alloc);
      mlir::Location loc = candidate.alloc.getLoc();
      
      // Get the original memref type that we need to recreate
      mlir::MemRefType originalType = candidate.alloc.getType();
      
      // Create a constant for the byte offset into the arena
      mlir::Value offsetIndex = mlir::arith::ConstantIndexOp::create(
          viewBuilder, loc, candidate.offset);
      
      // Use memref.view to create a typed view directly on the i8 arena
      // memref.view: memref<...xi8>, offset -> memref<shape x type>
      llvm::SmallVector<mlir::Value> dynamicSizes; // Empty for static shapes
      
      auto view = mlir::memref::ViewOp::create(
          viewBuilder, loc, originalType, arenaAlloc.getResult(), 
          offsetIndex, dynamicSizes);

      // Replace all uses of the original alloc with the viewed memref
      candidate.alloc.getResult().replaceAllUsesWith(view.getResult());
      
      // Remove the original alloc and dealloc
      candidate.alloc.erase();
      candidate.dealloc.erase();
    }
  }
};

} // end anonymous namespace
