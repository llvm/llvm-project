//===- StaticMemoryPlanning.cpp - Memory planning algorithms --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/StaticMemoryPlanning.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir::bufferization;

/// Align an offset to the specified alignment.
static int64_t alignOffset(int64_t offset, int64_t alignment) {
  return llvm::alignTo(offset, alignment);
}

/// Trivial sequential packing: places each allocation immediately after the
/// previous one with alignment padding. Does not consider lifetimes, so no
/// memory is reused. This gives a simple upper bound on arena size.
/// Complexity: O(n) where n is the number of allocations.
llvm::SmallVector<int64_t> mlir::bufferization::trivialMemoryPlanner(
    int64_t arenaAlignment, llvm::ArrayRef<MemoryPlannerAlloc> allocs) {
  llvm::SmallVector<int64_t> offsets;
  int64_t currentOffset = 0;

  for (const auto &alloc : allocs) {
    currentOffset = alignOffset(currentOffset, alloc.alignment);
    assert((arenaAlignment + currentOffset) % alloc.alignment == 0 &&
           "invalid alignment");
    offsets.push_back(currentOffset);
    currentOffset += alloc.sizeInBytes;
  }

  return offsets;
}

/// Best-fit lifetime-aware packing: processes allocations in start-time order
/// and places each one in the smallest gap left by allocations whose lifetimes
/// have ended. If no existing gap is large enough, the arena is extended.
/// This minimizes peak memory usage when allocations have non-overlapping
/// lifetimes.
/// Complexity: O(n^2) where n is the number of allocations.
llvm::SmallVector<int64_t> mlir::bufferization::bestFitMemoryPlanner(
    int64_t arenaAlignment, llvm::ArrayRef<MemoryPlannerAlloc> allocs) {
  // Tracks where each allocation was placed. We only need timeEnd because
  // allocations are processed in timeStart order — by the time we place a new
  // allocation, all earlier placements already started, so we only need to
  // check which ones are still live.
  struct Placement {
    int64_t offset;
    int64_t size;
    int64_t timeEnd;
  };

  // Process allocations in order of start time.
  llvm::SmallVector<unsigned> order(allocs.size());
  std::iota(order.begin(), order.end(), 0);
  llvm::sort(order, [&](unsigned a, unsigned b) {
    return allocs[a].timeStart < allocs[b].timeStart;
  });

  llvm::SmallVector<Placement> placements;
  llvm::SmallVector<int64_t> offsets(allocs.size(), 0);
  int64_t arenaEnd = 0;

  // Loop over all required allocations and fit them into best gaps.
  for (unsigned idx : order) {
    const MemoryPlannerAlloc &alloc = allocs[idx];

    // Collect allocations that are still live at this alloc's start time.
    // occupied is pairs of <offset_start, offset_end>.
    llvm::SmallVector<std::pair<int64_t, int64_t>> occupied;
    for (const auto &p : placements) {
      if (p.timeEnd > alloc.timeStart)
        occupied.push_back({p.offset, p.offset + p.size});
    }
    llvm::sort(occupied);

    // Find the best (smallest) gap that fits this allocation.
    int64_t bestOffset = -1;
    int64_t bestGapSize = INT64_MAX;

    int64_t gapStart = 0;
    for (const auto &[occStart, occEnd] : occupied) {
      int64_t alignedStart = alignOffset(gapStart, alloc.alignment);
      if (alignedStart >= occStart) {
        gapStart = std::max(gapStart, occEnd);
        continue;
      }
      int64_t gapEnd = occStart;
      int64_t gapSize = gapEnd - alignedStart;
      // Gap is large enough to fit this allocation (after alignment).
      if (gapSize >= alloc.sizeInBytes) {
        // Track the smallest sufficient gap (best-fit strategy).
        if (gapSize < bestGapSize) {
          bestGapSize = gapSize;
          bestOffset = alignedStart;
        }
      }
      gapStart = std::max(gapStart, occEnd);
    }

    // Check the trailing gap (between last occupied region and arena end).
    // This is only a reuse candidate if the allocation fits within the current
    // arena bounds. If it doesn't fit, placing here would extend the arena —
    // that case is handled by the fallback below (bestOffset < 0).
    int64_t alignedTrailing = alignOffset(gapStart, alloc.alignment);
    if (alignedTrailing + alloc.sizeInBytes <= arenaEnd) {
      int64_t trailingSize = arenaEnd - alignedTrailing;
      if (trailingSize < bestGapSize) {
        bestGapSize = trailingSize;
        bestOffset = alignedTrailing;
      }
    }

    // If no existing gap worked, append at the end.
    if (bestOffset < 0)
      bestOffset = alignedTrailing;

    assert((arenaAlignment + bestOffset) % alloc.alignment == 0 &&
           "invalid alignment");
    offsets[idx] = bestOffset;
    placements.push_back({bestOffset, alloc.sizeInBytes, alloc.timeEnd});
    arenaEnd = std::max(arenaEnd, bestOffset + alloc.sizeInBytes);
  }

  return offsets;
}
