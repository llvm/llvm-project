//===- AffineDataCopyGeneration.cpp - Explicit memref copying pass ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to automatically promote accessed memref regions
// to buffers in a faster memory space that is explicitly managed, with the
// necessary data movement operations performed through either regular
// point-wise load/store's or DMAs. Such explicit copying (also referred to as
// array packing/unpacking in the literature), when done on arrays that exhibit
// reuse, results in near elimination of conflict misses, TLB misses, reduced
// use of hardware prefetch streams, and reduced false sharing. It is also
// necessary for hardware that explicitly managed levels in the memory
// hierarchy, and where DMAs may have to be used. This optimization is often
// performed on already tiled code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

namespace mlir {
#define GEN_PASS_DEF_AFFINEDATACOPYGENERATION
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "affine-data-copy-generate"

using namespace mlir;

namespace {

/// Replaces all loads and stores on memref's living in 'slowMemorySpace' by
/// introducing copy operations to transfer data into `fastMemorySpace` and
/// rewriting the original load's/store's to instead load/store from the
/// allocated fast memory buffers. Additional options specify the identifier
/// corresponding to the fast memory space and the amount of fast memory space
/// available. The pass traverses through the nesting structure, recursing to
/// inner levels if necessary to determine at what depth copies need to be
/// placed so that the allocated buffers fit within the memory capacity
/// provided.
// TODO: We currently can't generate copies correctly when stores
// are strided. Check for strided stores.
struct AffineDataCopyGeneration
    : public impl::AffineDataCopyGenerationBase<AffineDataCopyGeneration> {
  AffineDataCopyGeneration() = default;
  explicit AffineDataCopyGeneration(unsigned slowMemorySpace,
                                    unsigned fastMemorySpace,
                                    unsigned tagMemorySpace,
                                    int minDmaTransferSize,
                                    uint64_t fastMemCapacityBytes) {
    this->slowMemorySpace = slowMemorySpace;
    this->fastMemorySpace = fastMemorySpace;
    this->tagMemorySpace = tagMemorySpace;
    this->minDmaTransferSize = minDmaTransferSize;
    this->fastMemoryCapacity = fastMemCapacityBytes / 1024;
  }

  void runOnOperation() override;
  void runOnBlock(Block *block, DenseSet<Operation *> &copyNests);

  // Constant zero index to avoid too many duplicates.
  Value zeroIndex = nullptr;
};

} // namespace

/// Generates copies for memref's living in 'slowMemorySpace' into newly created
/// buffers in 'fastMemorySpace', and replaces memory operations to the former
/// by the latter. Only load op's handled for now.
/// TODO: extend this to store op's.
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineDataCopyGenerationPass(unsigned slowMemorySpace,
                                         unsigned fastMemorySpace,
                                         unsigned tagMemorySpace,
                                         int minDmaTransferSize,
                                         uint64_t fastMemCapacityBytes) {
  return std::make_unique<AffineDataCopyGeneration>(
      slowMemorySpace, fastMemorySpace, tagMemorySpace, minDmaTransferSize,
      fastMemCapacityBytes);
}
std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createAffineDataCopyGenerationPass() {
  return std::make_unique<AffineDataCopyGeneration>();
}

/// Generate copies for this block. The block is partitioned into separate
/// ranges: each range is either a sequence of one or more operations starting
/// and ending with an affine load or store op, or just an affine.forop (which
/// could have other affine for op's nested within).
void AffineDataCopyGeneration::runOnBlock(Block *block,
                                          DenseSet<Operation *> &copyNests) {
  if (block->empty())
    return;

  uint64_t fastMemCapacityBytes =
      fastMemoryCapacity != std::numeric_limits<uint64_t>::max()
          ? fastMemoryCapacity * 1024
          : fastMemoryCapacity;
  AffineCopyOptions copyOptions = {generateDma, slowMemorySpace,
                                   fastMemorySpace, tagMemorySpace,
                                   fastMemCapacityBytes};

  // Every affine.for op in the block starts and ends a block range for copying;
  // in addition, a contiguous sequence of operations starting with a
  // load/store op but not including any copy nests themselves is also
  // identified as a copy block range. Straightline code (a contiguous chunk of
  // operations excluding AffineForOp's) are always assumed to not exhaust
  // memory. As a result, this approach is conservative in some cases at the
  // moment; we do a check later and report an error with location info.
  // TODO: An 'affine.if' operation is being treated similar to an
  // operation. 'affine.if''s could have 'affine.for's in them;
  // treat them separately.

  // Get to the first load, store, or for op (that is not a copy nest itself).
  auto curBegin =
      std::find_if(block->begin(), block->end(), [&](Operation &op) {
        return isa<AffineLoadOp, AffineStoreOp, AffineForOp>(op) &&
               copyNests.count(&op) == 0;
      });

  // Create [begin, end) ranges.
  auto it = curBegin;
  while (it != block->end()) {
    AffineForOp forOp;
    // If you hit a non-copy for loop, we will split there.
    if ((forOp = dyn_cast<AffineForOp>(&*it)) && copyNests.count(forOp) == 0) {
      // Perform the copying up unti this 'for' op first.
      (void)affineDataCopyGenerate(/*begin=*/curBegin, /*end=*/it, copyOptions,
                                   /*filterMemRef=*/llvm::None, copyNests);

      // Returns true if the footprint is known to exceed capacity.
      auto exceedsCapacity = [&](AffineForOp forOp) {
        Optional<int64_t> footprint =
            getMemoryFootprintBytes(forOp,
                                    /*memorySpace=*/0);
        return (footprint.has_value() &&
                static_cast<uint64_t>(footprint.value()) >
                    fastMemCapacityBytes);
      };

      // If the memory footprint of the 'affine.for' loop is higher than fast
      // memory capacity (when provided), we recurse to copy at an inner level
      // until we find a depth at which footprint fits in fast mem capacity. If
      // the footprint can't be calculated, we assume for now it fits. Recurse
      // inside if footprint for 'forOp' exceeds capacity, or when
      // skipNonUnitStrideLoops is set and the step size is not one.
      bool recurseInner = skipNonUnitStrideLoops ? forOp.getStep() != 1
                                                 : exceedsCapacity(forOp);
      if (recurseInner) {
        // We'll recurse and do the copies at an inner level for 'forInst'.
        // Recurse onto the body of this loop.
        runOnBlock(forOp.getBody(), copyNests);
      } else {
        // We have enough capacity, i.e., copies will be computed for the
        // portion of the block until 'it', and for 'it', which is 'forOp'. Note
        // that for the latter, the copies are placed just before this loop (for
        // incoming copies) and right after (for outgoing ones).

        // Inner loop copies have their own scope - we don't thus update
        // consumed capacity. The footprint check above guarantees this inner
        // loop's footprint fits.
        (void)affineDataCopyGenerate(/*begin=*/it, /*end=*/std::next(it),
                                     copyOptions,
                                     /*filterMemRef=*/llvm::None, copyNests);
      }
      // Get to the next load or store op after 'forOp'.
      curBegin = std::find_if(std::next(it), block->end(), [&](Operation &op) {
        return isa<AffineLoadOp, AffineStoreOp, AffineForOp>(op) &&
               copyNests.count(&op) == 0;
      });
      it = curBegin;
    } else {
      assert(copyNests.count(&*it) == 0 &&
             "all copy nests generated should have been skipped above");
      // We simply include this op in the current range and continue for more.
      ++it;
    }
  }

  // Generate the copy for the final block range.
  if (curBegin != block->end()) {
    // Can't be a terminator because it would have been skipped above.
    assert(!curBegin->hasTrait<OpTrait::IsTerminator>() &&
           "can't be a terminator");
    // Exclude the affine.yield - hence, the std::prev.
    (void)affineDataCopyGenerate(/*begin=*/curBegin,
                                 /*end=*/std::prev(block->end()), copyOptions,
                                 /*filterMemRef=*/llvm::None, copyNests);
  }
}

void AffineDataCopyGeneration::runOnOperation() {
  func::FuncOp f = getOperation();
  OpBuilder topBuilder(f.getBody());
  zeroIndex = topBuilder.create<arith::ConstantIndexOp>(f.getLoc(), 0);

  // Nests that are copy-in's or copy-out's; the root AffineForOps of those
  // nests are stored herein.
  DenseSet<Operation *> copyNests;

  // Clear recorded copy nests.
  copyNests.clear();

  for (auto &block : f)
    runOnBlock(&block, copyNests);

  // Promote any single iteration loops in the copy nests and collect
  // load/stores to simplify.
  SmallVector<Operation *, 4> copyOps;
  for (Operation *nest : copyNests)
    // With a post order walk, the erasure of loops does not affect
    // continuation of the walk or the collection of load/store ops.
    nest->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<AffineForOp>(op))
        (void)promoteIfSingleIteration(forOp);
      else if (isa<AffineLoadOp, AffineStoreOp>(op))
        copyOps.push_back(op);
    });

  // Promoting single iteration loops could lead to simplification of
  // contained load's/store's, and the latter could anyway also be
  // canonicalized.
  RewritePatternSet patterns(&getContext());
  AffineLoadOp::getCanonicalizationPatterns(patterns, &getContext());
  AffineStoreOp::getCanonicalizationPatterns(patterns, &getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyOpPatternsAndFold(copyOps, frozenPatterns, /*strict=*/true);
}
