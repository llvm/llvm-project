//===- BufferLifetimeStats.cpp - Buffer lifetime statistics pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_PRINTBUFFERLIFETIMESTATSPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Assign a sequential index to each operation in the block.
static DenseMap<Operation *, unsigned> buildOperationIndex(Block &block) {
  DenseMap<Operation *, unsigned> opIndex;
  unsigned idx = 0;
  for (Operation &op : block)
    opIndex[&op] = idx++;
  return opIndex;
}

/// Find the unique dealloc for `allocResult` in `block`, or nullptr.
static Operation *findDeallocInSameBlock(Value allocResult, Block *block) {
  Operation *deallocOp = nullptr;
  for (Operation *user : allocResult.getUsers()) {
    auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(user);
    if (!memEffectOp)
      continue;
    SmallVector<MemoryEffects::EffectInstance, 2> effects;
    memEffectOp.getEffects(effects);
    for (const auto &effect : effects) {
      if (isa<MemoryEffects::Free>(effect.getEffect()) &&
          user->getBlock() == block) {
        if (deallocOp)
          return nullptr;
        deallocOp = user;
      }
    }
  }
  return deallocOp;
}

/// Compute the size in bytes for a statically-shaped memref type.
static int64_t getMemRefSizeInBytes(MemRefType type) {
  if (!type.hasStaticShape())
    return 0;
  int64_t numElements = type.getNumElements();
  unsigned bitsPerElement = type.getElementTypeBitWidth();
  return (numElements * bitsPerElement + 7) / 8;
}

/// A buffer lifetime interval: [allocIndex, deallocIndex).
struct LifetimeInterval {
  Value allocResult;
  unsigned allocIndex;
  unsigned deallocIndex;
  int64_t sizeInBytes;
};

/// Check whether two lifetime intervals are non-overlapping.
static bool areNonOverlapping(const LifetimeInterval &a,
                              const LifetimeInterval &b) {
  return a.deallocIndex <= b.allocIndex || b.deallocIndex <= a.allocIndex;
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct PrintBufferLifetimeStats
    : public bufferization::impl::PrintBufferLifetimeStatsPassBase<
          PrintBufferLifetimeStats> {
public:
  PrintBufferLifetimeStats() = default;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (func.isExternal())
      return;

    // We only handle single-block functions for now.
    if (!func.getBody().hasOneBlock())
      return;

    Block &entryBlock = func.getBody().front();
    DenseMap<Operation *, unsigned> opIndex = buildOperationIndex(entryBlock);
    SmallVector<LifetimeInterval> intervals;

    entryBlock.walk([&](MemoryEffectOpInterface memEffectOp) {
      SmallVector<MemoryEffects::EffectInstance, 2> effects;
      memEffectOp.getEffects(effects);

      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (!isa<MemoryEffects::Allocate>(effect.getEffect()))
          continue;

        Value val = effect.getValue();
        if (!val || val.getDefiningOp() != memEffectOp.getOperation())
          continue;

        auto memrefType = dyn_cast<MemRefType>(val.getType());
        if (!memrefType)
          continue;

        Operation *deallocOp =
            findDeallocInSameBlock(val, memEffectOp->getBlock());
        if (!deallocOp)
          continue;

        auto allocIt = opIndex.find(memEffectOp.getOperation());
        auto deallocIt = opIndex.find(deallocOp);
        if (allocIt == opIndex.end() || deallocIt == opIndex.end())
          continue;

        int64_t sizeBytes = getMemRefSizeInBytes(memrefType);
        intervals.push_back(
            {val, allocIt->second, deallocIt->second, sizeBytes});
      }
    });

    // Compute statistics.
    int64_t totalBytes = 0;
    for (const auto &interval : intervals)
      totalBytes += interval.sizeInBytes;

    // Compute peak live bytes by sweeping through all time points.
    int64_t peakLiveBytes = 0;
    if (!intervals.empty()) {
      // Collect all unique time points.
      SmallVector<unsigned> timePoints;
      for (const auto &interval : intervals) {
        timePoints.push_back(interval.allocIndex);
        timePoints.push_back(interval.deallocIndex);
      }
      llvm::sort(timePoints);
      timePoints.erase(llvm::unique(timePoints), timePoints.end());

      for (unsigned t : timePoints) {
        int64_t liveBytes = 0;
        for (const auto &interval : intervals) {
          if (interval.allocIndex <= t && t < interval.deallocIndex)
            liveBytes += interval.sizeInBytes;
        }
        peakLiveBytes = std::max(peakLiveBytes, liveBytes);
      }
    }

    // Count non-overlapping pairs (reuse opportunities).
    unsigned nonOverlappingPairs = 0;
    for (unsigned i = 0; i < intervals.size(); ++i)
      for (unsigned j = i + 1; j < intervals.size(); ++j)
        if (areNonOverlapping(intervals[i], intervals[j]))
          ++nonOverlappingPairs;

    llvm::outs() << "--- Buffer Lifetime Statistics for '" << func.getSymName()
                 << "' ---\n";
    llvm::outs() << "  Tracked allocations     : " << intervals.size() << "\n";
    llvm::outs() << "  Total allocated bytes    : " << totalBytes << "\n";
    llvm::outs() << "  Peak live bytes          : " << peakLiveBytes << "\n";
    llvm::outs() << "  Non-overlapping pairs    : " << nonOverlappingPairs
                 << "\n";

    for (const auto &interval : intervals) {
      llvm::outs() << "  Buffer: " << interval.allocResult.getType()
                   << " | size=" << interval.sizeInBytes << " | lifetime=["
                   << interval.allocIndex << ", " << interval.deallocIndex
                   << ")\n";
    }
    llvm::outs() << "---\n";
  }
};

} // namespace
