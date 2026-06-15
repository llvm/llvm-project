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

/// A candidate allocation with its matching deallocation and assigned offset.
struct AllocationCandidate {
  mlir::memref::AllocOp alloc;
  mlir::memref::DeallocOp dealloc;
  int64_t offset = 0;         // Offset in bytes from arena start
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

    // Step 2: Compute simple sequential offsets with alignment padding
    int64_t totalSize = 0;
    int64_t maxAlignment = 1;
    for (auto &candidate : candidates) {
      // Align current offset to this allocation's requirement
      totalSize = alignOffset(totalSize, candidate.alignment);
      candidate.offset = totalSize;
      totalSize += candidate.sizeInBytes;
      maxAlignment = std::max(maxAlignment, candidate.alignment);
      LLVM_DEBUG(llvm::dbgs() << "[static-memory-planner] offset="
                              << candidate.offset
                              << " size=" << candidate.sizeInBytes
                              << " alignment=" << candidate.alignment << "\n");
    }

    // Step 3: Find the first allocation's location to place arena
    mlir::Operation *firstAlloc = candidates.front().alloc;
    mlir::OpBuilder builder(firstAlloc);

    // Get element type from first allocation (assume all same type for now)
    mlir::Type elementType = candidates.front().alloc.getType().getElementType();
    unsigned elementSizeInBits = elementType.getIntOrFloatBitWidth();
    unsigned elementSizeInBytes = (elementSizeInBits + 7) / 8;
    
    // Step 4: Create arena allocation with maximum alignment
    // Convert total size from bytes to elements
    int64_t arenaSizeInElements = (totalSize + elementSizeInBytes - 1) / elementSizeInBytes;
    auto arenaType = mlir::MemRefType::get({arenaSizeInElements}, elementType);
    auto arenaAlloc = mlir::memref::AllocOp::create(builder, firstAlloc->getLoc(), arenaType);
    arenaAlloc.setAlignmentAttr(builder.getI64IntegerAttr(maxAlignment));

    LLVM_DEBUG(llvm::dbgs() << "[static-memory-planner] created arena: size="
                            << arenaSizeInElements << " elements (" << totalSize 
                            << " bytes), alignment=" << maxAlignment << " bytes\n");

    // Step 5: Replace each alloc with a subview and remove deallocs
    for (auto &candidate : candidates) {
      mlir::OpBuilder subviewBuilder(candidate.alloc);
      
      // Create subview into arena
      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      
      // Convert byte offsets/sizes back to element indices for the arena
      int64_t offsetInElements = candidate.offset / elementSizeInBytes;
      int64_t sizeInElements = candidate.sizeInBytes / elementSizeInBytes;
      
      offsets.push_back(subviewBuilder.getIndexAttr(offsetInElements));
      sizes.push_back(subviewBuilder.getIndexAttr(sizeInElements));
      strides.push_back(subviewBuilder.getIndexAttr(1));
      
      auto subview = mlir::memref::SubViewOp::create(
          subviewBuilder, candidate.alloc.getLoc(), arenaAlloc.getResult(),
          offsets, sizes, strides);

      // Replace all uses of the original alloc
      candidate.alloc.getResult().replaceAllUsesWith(subview.getResult());
      
      // Remove the original alloc and dealloc
      candidate.alloc.erase();
      candidate.dealloc.erase();
    }
  }
};

} // end anonymous namespace
