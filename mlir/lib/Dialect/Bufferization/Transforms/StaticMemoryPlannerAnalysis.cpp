//===- StaticMemoryPlannerAnalysis.cpp - Analysis for static memory -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Discovers same-block memref.alloc/memref.dealloc pairs and analyzes their
// lifetime relationships to identify opportunities for memory reuse.
//
// This pass performs basic structural analysis without computing actual memory
// layouts. It reports which allocations are eligible for static planning and
// whether pairs of allocations have non-overlapping lifetimes (can reuse).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "static-memory-planner-analysis"

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

/// A candidate allocation with its matching deallocation.
struct AllocationCandidate {
  memref::AllocOp alloc;
  memref::DeallocOp dealloc;
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

/// Checks if two allocation candidates have non-overlapping lifetimes.
/// Returns true if the first's dealloc is strictly before the second's alloc,
/// or vice versa.
static bool canReuseMemory(const AllocationCandidate &first,
                           const AllocationCandidate &second) {
  Operation *firstDealloc = first.dealloc;
  Operation *firstAlloc = first.alloc;
  Operation *secondDealloc = second.dealloc;
  Operation *secondAlloc = second.alloc;

  // Check if both are in the same block
  if (firstAlloc->getBlock() != secondAlloc->getBlock())
    return false;

  // Check if first ends before second starts
  if (firstDealloc->isBeforeInBlock(secondAlloc))
    return true;

  // Check if second ends before first starts
  if (secondDealloc->isBeforeInBlock(firstAlloc))
    return true;

  return false;
}

//===----------------------------------------------------------------------===//
// StaticMemoryPlannerAnalysisPass
//===----------------------------------------------------------------------===//

struct StaticMemoryPlannerAnalysisPass
    : public bufferization::impl::StaticMemoryPlannerAnalysisPassBase<
          StaticMemoryPlannerAnalysisPass> {
public:
  StaticMemoryPlannerAnalysisPass() = default;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Collect eligible allocation candidates
    SmallVector<AllocationCandidate> candidates;

    op->walk([&](memref::AllocOp allocOp) {
      MemRefType memrefType = allocOp.getType();

      // Skip dynamic shapes
      if (!memrefType.hasStaticShape()) {
        ++numSkipDynamic;
        allocOp.emitRemark("static-memory-planner: skip: dynamic shape");
        return;
      }

      // Find unique dealloc in the same block
      memref::DeallocOp deallocOp = findUniqueDealloc(allocOp.getResult());
      if (!deallocOp) {
        ++numSkipNoDealloc;
        allocOp.emitRemark(
            "static-memory-planner: skip: no unique dealloc");
        return;
      }

      if (deallocOp->getBlock() != allocOp->getBlock()) {
        ++numSkipNoDealloc;
        allocOp.emitRemark(
            "static-memory-planner: skip: dealloc in different block");
        return;
      }

      // This allocation is eligible
      ++numEligible;
      allocOp.emitRemark("static-memory-planner: eligible");
      candidates.push_back({allocOp, deallocOp});
    });

    // Analyze reuse opportunities between pairs of candidates
    unsigned numReusable = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      for (size_t j = i + 1; j < candidates.size(); ++j) {
        if (canReuseMemory(candidates[i], candidates[j])) {
          ++numReusable;
          LLVM_DEBUG(llvm::dbgs()
                     << "[static-memory-planner] reuse opportunity: alloc "
                     << i << " and alloc " << j << "\n");
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[static-memory-planner] summary: eligible="
                            << (unsigned)numEligible << " skip-dynamic="
                            << (unsigned)numSkipDynamic << " skip-no-dealloc="
                            << (unsigned)numSkipNoDealloc << " reusable-pairs="
                            << numReusable << "\n");
  }
};

} // end anonymous namespace
