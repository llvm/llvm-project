//===- VectorShuffleTreeBuilder.cpp ----- Vector shuffle tree builder -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pattern rewrites to lower sequences of
// `vector.to_elements` and `vector.from_elements` operations into a tree of
// `vector.shuffle` operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace vector {

#define GEN_PASS_DEF_LOWERVECTORTOFROMELEMENTSTOSHUFFLETREE
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"

} // namespace vector
} // namespace mlir

#define DEBUG_TYPE "lower-vector-to-from-elements-to-shuffle-tree"

using namespace mlir;
using namespace mlir::vector;

namespace {

// Indentation unit for debug output formatting.
[[maybe_unused]] constexpr unsigned kIndScale = 2;

/// Represents a closed interval of elements (e.g., [0, 7] = 8 elements).
using Interval = std::pair<unsigned, unsigned>;
// Sentinel value for uninitialized intervals.
constexpr unsigned kMaxUnsigned = std::numeric_limits<unsigned>::max();

/// The VectorShuffleTreeBuilder builds a balanced binary tree of
/// `vector.shuffle` operations from one or more `vector.to_elements`
/// operations feeding a single `vector.from_elements` operation.
///
/// The implementation generates hardware-agnostic `vector.shuffle` operations
/// that minimize both the number of shuffle operations and the length of
/// intermediate vectors (to the extent possible). The tree has the
/// following properties:
///
///   1. Vectors are shuffled in pairs by order of appearance in
///      the `vector.from_elements` operand list.
///   2. Each vector at each level is used only once.
///   3. The number of levels in the tree is:
///        1 (input vectors) + ceil(max(1,log2(# `vector.to_elements` ops))).
///   4. Vectors at each level of the tree have the same vector length.
///   5. Vector positions that do not need to be shuffled are represented with
///      poison in the shuffle mask.
///
/// Examples #1: Concatenation of 3x vector<4xf32> to vector<12xf32>:
///
///   %0:4 = vector.to_elements %a : vector<4xf32>
///   %1:4 = vector.to_elements %b : vector<4xf32>
///   %2:4 = vector.to_elements %c : vector<4xf32>
///   %3 = vector.from_elements %0#0, %0#1, %0#2, %0#3, %1#0, %1#1,
///                             %1#2, %1#3, %2#0, %2#1, %2#2, %2#3
///                               : vector<12xf32>
///   =>
///
///   %shuffle0 = vector.shuffle %a, %b [0, 1, 2, 3, 4, 5, 6, 7]
///     : vector<4xf32>, vector<4xf32>
///   %shuffle1 = vector.shuffle %c, %c [0, 1, 2, 3, -1, -1, -1, -1]
///     : vector<4xf32>, vector<4xf32>
///   %result = vector.shuffle %shuffle0, %shuffle1 [0, 1, 2, 3, 4, 5,
///                                                  6, 7, 8, 9, 10, 11]
///     : vector<8xf32>, vector<8xf32>
///
///   Comments:
///     * The shuffle tree has three levels:
///         - Level 0 = (%a, %b, %c, %c)
///         - Level 1 = (%shuffle0, %shuffle1)
///         - Level 2 = (%result)
///     * `%a` and `%b` are shuffled first because they appear first in the
///       `vector.from_elements` operand list (`%0#0` and `%1#0`).
///     * `%c` is shuffled with itself because the number of
///       `vector.from_elements` operands is odd.
///     * The vector length for level 1 and level 2 are 8 and 16, respectively.
///     * `%shuffle1` uses poison values to match the vector length of its
///       tree level (8).
///
///
/// Example #2: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   %0:5 = vector.to_elements %a : vector<5xf32>
///   %1:5 = vector.to_elements %b : vector<5xf32>
///   %2:5 = vector.to_elements %c : vector<5xf32>
///   %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2,
///                             %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
///   =>
///
///   %shuffle0 = vector.shuffle %[[C]], %[[B]] [2, 6, -1, -1, 7, 2, 0, 6]
///     : vector<5xf32>, vector<5xf32>
///   %shuffle1 = vector.shuffle %[[A]], %[[A]] [1, 1, -1, -1, -1, -1, 4, -1]
///     : vector<5xf32>, vector<5xf32>
///   %result = vector.shuffle %shuffle0, %shuffle1 [0, 1, 8, 9, 4, 5, 6, 7, 14]
///     : vector<8xf32>, vector<8xf32>
///
///   Comments:
///     * `%c` and `%b` are shuffled first because they appear first in the
///       `vector.from_elements` operand list (`%2#2` and `%1#1`).
///     * `%a` is shuffled with itself because the number of
///       `vector.from_elements` operands is odd.
///     * The vector length for level 1 and level 2 are 8 and 9, respectively.
///     * `%shuffle0` uses poison values to mark unused vector positions and
///       match the vector length of its tree level (8).
///
/// TODO: Implement mask compression to reduce the number of intermediate poison
/// values.
class VectorShuffleTreeBuilder {
public:
  VectorShuffleTreeBuilder() = delete;
  VectorShuffleTreeBuilder(FromElementsOp fromElemOp,
                           ArrayRef<ToElementsOp> toElemDefs);

  /// Analyze the input `vector.to_elements` + `vector.from_elements` sequence
  /// and compute the shuffle tree configuration. This method does not generate
  /// any IR.
  LogicalResult computeShuffleTree();

  /// Materialize the shuffle tree configuration computed by
  /// `computeShuffleTree` in the IR.
  Value generateShuffleTree(PatternRewriter &rewriter);

private:
  // IR input information.
  FromElementsOp fromElemsOp;
  SmallVector<ToElementsOp> toElemsDefs;

  // Shuffle tree configuration.
  unsigned numLevels;
  SmallVector<unsigned> vectorSizePerLevel;
  /// Holds the range of positions each vector in the tree contributes to in the
  /// final output vector.
  SmallVector<SmallVector<Interval>> intervalsPerLevel;

  // Utility methods to compute the shuffle tree configuration.
  void computeShuffleTreeIntervals();
  void computeShuffleTreeVectorSizes();

  /// Dump the shuffle tree configuration.
  void dump();
};

VectorShuffleTreeBuilder::VectorShuffleTreeBuilder(
    FromElementsOp fromElemOp, ArrayRef<ToElementsOp> toElemDefs)
    : fromElemsOp(fromElemOp), toElemsDefs(toElemDefs) {
  assert(fromElemsOp && "from_elements op is required");
  assert(!toElemsDefs.empty() && "At least one to_elements op is required");
}

/// Duplicate the last operation, value or interval if the total number of them
/// is odd. This is useful to simplify the shuffle tree algorithm given that
/// vectors are shuffled in pairs and duplication would lead to the last shuffle
/// to have a single (duplicated) input vector.
template <typename T>
static void duplicateLastIfOdd(SmallVectorImpl<T> &values) {
  if (values.size() % 2 != 0)
    values.push_back(values.back());
}

// ===---------------------------------------------------------------------===//
// Shuffle Tree Analysis Utilities.
// ===---------------------------------------------------------------------===//

/// Compute the intervals for all the vectors in the shuffle tree. The interval
/// of a vector is the range of positions that the vector contributes to in the
/// final output vector.
///
/// Example: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   %0:5 = vector.to_elements %a : vector<5xf32>
///   %1:5 = vector.to_elements %b : vector<5xf32>
///   %2:5 = vector.to_elements %c : vector<5xf32>
///   %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2,
///                             %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
///
/// The shuffle tree has 3 levels. Level 0 has 4 vectors (%2, %1, %0, %0, the
/// last one is duplicated to make the number of inputs even) so we compute the
/// interval for each vector:
///
///    * intervalsPerLevel[0][0] = interval(%2) = [0,6]
///    * intervalsPerLevel[0][1] = interval(%1) = [1,7]
///    * intervalsPerLevel[0][2] = interval(%0) = [2,8]
///    * intervalsPerLevel[0][3] = interval(%0) = [2,8]
///
/// Level 1 has 2 vectors, resulting from the shuffling of %2 + %1 and %0 + %0
/// so we compute the intervals for each vector at level 1 as:
///    * intervalsPerLevel[1][0] = intervalsPerLevel[0][0] U
///                                intervalsPerLevel[0][1] = [0,7]
///    * intervalsPerLevel[1][1] = intervalsPerLevel[0][2] U
///                                intervalsPerLevel[0][3] = [2,8]
///
/// Level 2 is the last level and only contains the output vector so the
/// interval should be the whole output vector:
///    * intervalsPerLevel[2][0] = intervalsPerLevel[1][0] U
///                                intervalsPerLevel[1][1] = [0,8]
///
void VectorShuffleTreeBuilder::computeShuffleTreeIntervals() {
  // Map `vector.to_elements` ops to their ordinal position in the
  // `vector.from_elements` operand list. Make sure duplicated
  // `vector.to_elements` ops are mapped to the its first occurrence.
  DenseMap<ToElementsOp, unsigned> toElemsToInputOrdinal;
  for (const auto &[idx, toElemsOp] : llvm::enumerate(toElemsDefs))
    toElemsToInputOrdinal.insert({toElemsOp, idx});

  // Compute intervals for each vector in the shuffle tree. The first
  // level computation is special-cased to keep the implementation simpler.

  SmallVector<Interval> firstLevelIntervals(toElemsDefs.size(),
                                            {kMaxUnsigned, kMaxUnsigned});

  for (const auto &[idx, element] :
       llvm::enumerate(fromElemsOp.getElements())) {
    auto toElemsOp = cast<ToElementsOp>(element.getDefiningOp());
    unsigned inputIdx = toElemsToInputOrdinal[toElemsOp];
    Interval &currentInterval = firstLevelIntervals[inputIdx];

    // Set lower bound to the first occurrence of the `vector.to_elements`.
    if (currentInterval.first == kMaxUnsigned)
      currentInterval.first = idx;

    // Set upper bound to the last occurrence of the `vector.to_elements`.
    currentInterval.second = idx;
  }

  duplicateLastIfOdd(toElemsDefs);
  duplicateLastIfOdd(firstLevelIntervals);
  intervalsPerLevel.push_back(std::move(firstLevelIntervals));

  // Compute intervals for the remaining levels.
  for (unsigned level = 1; level < numLevels; ++level) {
    bool isLastLevel = level == numLevels - 1;
    const auto &prevLevelIntervals = intervalsPerLevel[level - 1];
    SmallVector<Interval> currentLevelIntervals(
        llvm::divideCeil(prevLevelIntervals.size(), 2),
        {kMaxUnsigned, kMaxUnsigned});

    size_t currentNumLevels = currentLevelIntervals.size();
    for (size_t inputIdx = 0; inputIdx < currentNumLevels; ++inputIdx) {
      auto &interval = currentLevelIntervals[inputIdx];
      const auto &prevLhsInterval = prevLevelIntervals[inputIdx * 2];
      const auto &prevRhsInterval = prevLevelIntervals[inputIdx * 2 + 1];

      // The interval of a vector at the current level is the union of the
      // intervals of the two vectors from the previous level being shuffled at
      // this level.
      interval.first = prevLhsInterval.first;
      interval.second =
          std::max(prevLhsInterval.second, prevRhsInterval.second);
    }

    // Duplicate the last interval if the number of intervals is odd, except for
    // the last level as it only contains the output vector, which doesn't have
    // to be shuffled.
    if (!isLastLevel)
      duplicateLastIfOdd(currentLevelIntervals);

    intervalsPerLevel.push_back(std::move(currentLevelIntervals));
  }
}

/// Compute the uniform vector size for each level of the shuffle tree, given
/// the intervals of the vectors at each level. The vector size of a level is
/// the size of the widest interval at that level.
///
/// Example: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   Intervals:
///     * Level 0: [0,6], [1,7], [2,8], [2,8]
///     * Level 1: [0,7], [2,8]
///     * Level 2: [0,8]
///
///   Vector sizes:
///     * Level 0: Arbitrary sizes from input vectors.
///     * Level 1: max(size_of([0,7]) = 8, size_of([2,8]) = 7) = 8
///     * Level 2: max(size_of([0,8]) = 9) = 9
///
void VectorShuffleTreeBuilder::computeShuffleTreeVectorSizes() {
  // Compute vector size for each level. There are two direct cases:
  //   * First level: the vector size depends on the actual size of the input
  //     vectors and it's allowed to be non-uniform. We set it to 0.
  //   * Last level: the vector size is the output vector size so it doesn't
  //     have to be computed using intervals.
  vectorSizePerLevel.front() = 0;
  vectorSizePerLevel.back() =
      cast<VectorType>(fromElemsOp.getResult().getType()).getNumElements();

  for (unsigned level = 1; level < numLevels - 1; ++level) {
    const auto &currentLevelIntervals = intervalsPerLevel[level];
    unsigned currentVectorSize = 1;
    size_t numIntervals = currentLevelIntervals.size();
    for (size_t i = 0; i < numIntervals; ++i) {
      const auto &interval = currentLevelIntervals[i];
      unsigned intervalSize = interval.second - interval.first + 1;
      currentVectorSize = std::max(currentVectorSize, intervalSize);
    }
    assert(currentVectorSize > 0 && "vector size must be positive");
    vectorSizePerLevel[level] = currentVectorSize;
  }
}

void VectorShuffleTreeBuilder::dump() {
  LLVM_DEBUG({
    unsigned indLv = 0;

    llvm::dbgs() << "VectorShuffleTreeBuilder Configuration:\n";
    ++indLv;
    llvm::dbgs() << llvm::indent(indLv, kIndScale) << "* Inputs:\n";
    ++indLv;
    for (const auto &toElemsOp : toElemsDefs)
      llvm::dbgs() << llvm::indent(indLv, kIndScale) << toElemsOp << "\n";
    llvm::dbgs() << llvm::indent(indLv, kIndScale) << fromElemsOp << "\n\n";
    --indLv;

    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* Total levels: " << numLevels << "\n";
    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* Vector sizes per level: ";
    llvm::interleaveComma(vectorSizePerLevel, llvm::dbgs());
    llvm::dbgs() << "\n";
    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* Input intervals per level:\n";
    ++indLv;
    for (const auto &[level, intervals] : llvm::enumerate(intervalsPerLevel)) {
      llvm::dbgs() << llvm::indent(indLv, kIndScale) << "* Level " << level
                   << ": ";
      llvm::interleaveComma(intervals, llvm::dbgs(),
                            [](const Interval &interval) {
                              llvm::dbgs() << "[" << interval.first << ","
                                           << interval.second << "]";
                            });
      llvm::dbgs() << "\n";
    }
  });
}

/// Compute the shuffle tree configuration for the given `vector.to_elements` +
/// `vector.from_elements` input sequence. This method builds a balanced binary
/// shuffle tree that combines pairs of vectors at each level.
///
/// Example: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   %0:5 = vector.to_elements %a : vector<5xf32>
///   %1:5 = vector.to_elements %b : vector<5xf32>
///   %2:5 = vector.to_elements %c : vector<5xf32>
///   %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2,
///                             %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
///
///   build a tree that looks like:
///
///          %2    %1                   %0    %0
///            \  /                       \  /
///  %2_1 = vector.shuffle     %0_0 = vector.shuffle
///              \                    /
///             %2_1_0_0 =vector.shuffle
///
/// The actual representation of the shuffle tree configuration is based on
/// intervals of each vector at each level of the shuffle tree (i.e., %2, %1,
/// %0, %0, %2_1, %0_0 and %2_1_0_0) and the vector size for each level. For
/// further details on intervals and vector size computation, please, take a
/// look at the corresponding utility functions.
LogicalResult VectorShuffleTreeBuilder::computeShuffleTree() {
  // Initialize shuffle tree information based on its size. For the number of
  // levels, we add one to account for the input `vector.to_elements` as one
  // tree level. We need the std::max(1) to account for a single element input.
  numLevels = 1u + std::max(1u, llvm::Log2_64_Ceil(toElemsDefs.size()));
  vectorSizePerLevel.resize(numLevels, 0);
  intervalsPerLevel.reserve(numLevels);

  computeShuffleTreeIntervals();
  computeShuffleTreeVectorSizes();
  dump();

  return success();
}

// ===---------------------------------------------------------------------===//
// Shuffle Tree Code Generation Utilities.
// ===---------------------------------------------------------------------===//

/// Compute the permutation mask for shuffling two input `vector.to_elements`
/// ops. The permutation mask is the mapping of the vector elements to their
/// final position in the output vector, relative to the intermediate output
/// vector of the `vector.shuffle` operation combining the two inputs.
///
/// Example: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   %0:5 = vector.to_elements %a : vector<5xf32>
///   %1:5 = vector.to_elements %b : vector<5xf32>
///   %2:5 = vector.to_elements %c : vector<5xf32>
///   %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2,
///                             %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
///
///   =>
///
///   // Level 1, vector length = 8
///   %2_1 = PermutationShuffleMask(%2, %1) = [2, 6, -1, -1, 7, 2, 0, 6]
///   %0_0 = PermutationShuffleMask(%0, %0) = [1, 1, -1, -1, -1, -1, 4, -1]
///
/// TODO: Implement mask compression to reduce the number of intermediate poison
/// values.
static SmallVector<int64_t> computePermutationShuffleMask(
    ToElementsOp toElementOp0, const Interval &interval0,
    ToElementsOp toElementOp1, const Interval &interval1,
    FromElementsOp fromElemsOp, unsigned outputVectorSize) {
  SmallVector<int64_t> mask(outputVectorSize, ShuffleOp::kPoisonIndex);
  unsigned inputVectorSize =
      toElementOp0.getSource().getType().getNumElements();

  for (const auto &[inputIdx, element] :
       llvm::enumerate(fromElemsOp.getElements())) {
    auto currentToElemOp = cast<ToElementsOp>(element.getDefiningOp());
    // Match `vector.from_elements` operands to the two input ops.
    if (currentToElemOp != toElementOp0 && currentToElemOp != toElementOp1)
      continue;

    // The permutation value for a particular operand is the ordinal position of
    // the operand in the `vector.to_elements` list of results.
    unsigned permVal = cast<OpResult>(element).getResultNumber();
    unsigned maskIdx = inputIdx;

    // The mask index is the ordinal position of the operand in
    // `vector.from_elements` operand list. We make this position relative to
    // the output interval resulting from combining the two input intervals.
    if (currentToElemOp == toElementOp0) {
      maskIdx -= interval0.first;
    } else {
      // currentToElemOp == toElementOp1
      unsigned intervalOffset = interval1.first - interval0.first;
      maskIdx += intervalOffset - interval1.first;
      permVal += inputVectorSize;
    }

    mask[maskIdx] = permVal;
  }

  LLVM_DEBUG({
    unsigned indLv = 1;
    llvm::dbgs() << llvm::indent(indLv, kIndScale) << "* Permutation mask: [";
    llvm::interleaveComma(mask, llvm::dbgs());
    llvm::dbgs() << "]\n";
    ++indLv;
    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* Combining: " << toElementOp0 << " and " << toElementOp1
                 << "\n";
  });

  return mask;
}

/// Compute the propagation shuffle mask for combining two intermediate shuffle
/// operations of the tree. The propagation shuffle mask is the mapping of the
/// intermediate vector elements, which have already been shuffled to their
/// relative output position using the mask generated by
/// `computePermutationShuffleMask`, to their next position in the tree.
///
/// Example: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   %0:5 = vector.to_elements %a : vector<5xf32>
///   %1:5 = vector.to_elements %b : vector<5xf32>
///   %2:5 = vector.to_elements %c : vector<5xf32>
///   %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2,
///                             %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
///
///   // Level 1, vector length = 8
///   %2_1 = PermutationShuffleMask(%2, %1) = [2, 6, -1, -1, 7, 2, 0, 6]
///   %0_0 = PermutationShuffleMask(%0, %0) = [1, 1, -1, -1, -1, -1, 4, -1]
///
///   =>
///
///   // Level 2, vector length = 9
///   PropagationShuffleMask(%2_1, %0_0) = [0, 1, 8, 9, 4, 5, 6, 7, 14]
///
/// TODO: Implement mask compression to reduce the number of intermediate poison
/// values.
static SmallVector<int64_t> computePropagationShuffleMask(
    ShuffleOp lhsShuffleOp, const Interval &lhsInterval, ShuffleOp rhsShuffleOp,
    const Interval &rhsInterval, unsigned outputVectorSize) {
  ArrayRef<int64_t> lhsShuffleMask = lhsShuffleOp.getMask();
  ArrayRef<int64_t> rhsShuffleMask = rhsShuffleOp.getMask();
  unsigned inputVectorSize = lhsShuffleMask.size();
  assert(inputVectorSize == rhsShuffleMask.size() &&
         "Expected both shuffle masks to have the same size");

  bool hasSameInput = lhsShuffleOp == rhsShuffleOp;
  unsigned lhsRhsOffset = rhsInterval.first - lhsInterval.first;
  SmallVector<int64_t> mask(outputVectorSize, ShuffleOp::kPoisonIndex);

  // Propagate any element from the input mask that is not poison. For the RHS
  // vector, offset mask index by the distance between the intervals.
  for (unsigned i = 0; i < inputVectorSize; ++i) {
    if (lhsShuffleMask[i] != ShuffleOp::kPoisonIndex)
      mask[i] = i;

    if (hasSameInput)
      continue;

    unsigned rhsIdx = i + lhsRhsOffset;
    if (rhsShuffleMask[i] != ShuffleOp::kPoisonIndex) {
      assert(rhsIdx < outputVectorSize && "RHS index out of bounds");
      assert(mask[rhsIdx] == ShuffleOp::kPoisonIndex && "mask already set");
      mask[rhsIdx] = i + inputVectorSize;
    }
  }

  LLVM_DEBUG({
    unsigned indLv = 1;
    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* Propagation shuffle mask computation:\n";
    ++indLv;
    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* LHS shuffle op: " << lhsShuffleOp << "\n";
    llvm::dbgs() << llvm::indent(indLv, kIndScale)
                 << "* RHS shuffle op: " << rhsShuffleOp << "\n";
    llvm::dbgs() << llvm::indent(indLv, kIndScale) << "* Result mask: [";
    llvm::interleaveComma(mask, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  return mask;
}

/// Materialize the pre-computed shuffle tree configuration in the IR by
/// generating the corresponding `vector.shuffle` ops.
///
/// Example: Arbitrary shuffling of 3x vector<5xf32> to vector<9xf32>:
///
///   %0:5 = vector.to_elements %a : vector<5xf32>
///   %1:5 = vector.to_elements %b : vector<5xf32>
///   %2:5 = vector.to_elements %c : vector<5xf32>
///   %3 = vector.from_elements %2#2, %1#1, %0#1, %0#1, %1#2,
///                             %2#2, %2#0, %1#1, %0#4 : vector<9xf32>
///
///   with the pre-computed shuffle tree configuration:
///
///     * Vector sizes per level: 0, 8, 9
///     * Input intervals per level:
///       * Level 0: [0,6], [1,7], [2,8], [2,8]
///       * Level 1: [0,7], [2,8]
///       * Level 2: [0,8]
///
///   =>
///
///    %0 = vector.shuffle %arg2, %arg1 [2, 6, -1, -1, 7, 2, 0, 6]
///        : vector<5xf32>, vector<5xf32>
///    %1 = vector.shuffle %arg0, %arg0 [1, 1, -1, -1, -1, -1, 4, -1]
///        : vector<5xf32>, vector<5xf32>
///    %2 = vector.shuffle %0, %1 [0, 1, 8, 9, 4, 5, 6, 7, 14]
///        : vector<8xf32>, vector<8xf32>
///
/// The code generation consists of combining pairs of vectors at each level of
/// the tree, using the pre-computed tree intervals and vector sizes. The
/// algorithm generates two kinds of shuffle masks:
///   * Permutation masks: computed for the first level of the tree and permute
///     the input vector elements to their relative position in the final
///     output.
///   * Propagation masks: computed for subsequent levels and propagate the
///     elements to the next level without permutation.
///
/// For further details on the shuffle mask computation, please, take a look at
/// the corresponding `computePermutationShuffleMask` and
/// `computePropagationShuffleMask` functions.
///
Value VectorShuffleTreeBuilder::generateShuffleTree(PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "VectorShuffleTreeBuilder Code Generation:\n");

  // Initialize work list with the `vector.to_elements` sources.
  SmallVector<Value> levelInputs;
  llvm::transform(toElemsDefs, std::back_inserter(levelInputs),
                  [](ToElementsOp toElemsOp) { return toElemsOp.getSource(); });

  // Build shuffle tree by combining pairs of vectors (represented by their
  // corresponding intervals) in one level and producing a new vector with the
  // next level's vector length. Skip the interval from the last tree level
  // (actual shuffle tree output) as it doesn't have to be combined with
  // anything else.
  Location loc = fromElemsOp.getLoc();
  unsigned currentLevel = 0;
  for (const auto &[nextLevelVectorSize, intervals] :
       llvm::zip_equal(ArrayRef(vectorSizePerLevel).drop_front(),
                       ArrayRef(intervalsPerLevel).drop_back())) {

    duplicateLastIfOdd(levelInputs);

    LLVM_DEBUG(llvm::dbgs() << llvm::indent(1, kIndScale)
                            << "* Processing level " << currentLevel
                            << " (output vector size: " << nextLevelVectorSize
                            << ", # inputs: " << levelInputs.size() << ")\n");

    // Process level input vectors in pairs.
    SmallVector<Value> levelOutputs;
    for (size_t i = 0, numLevelInputs = levelInputs.size(); i < numLevelInputs;
         i += 2) {
      Value lhsVector = levelInputs[i];
      Value rhsVector = levelInputs[i + 1];
      const Interval &lhsInterval = intervals[i];
      const Interval &rhsInterval = intervals[i + 1];

      // For the first level of the tree, permute the vector elements to their
      // relative position in the final output. For subsequent levels, we
      // propagate the elements to the next level without permutation.
      SmallVector<int64_t> shuffleMask;
      if (currentLevel == 0) {
        shuffleMask = computePermutationShuffleMask(
            toElemsDefs[i], lhsInterval, toElemsDefs[i + 1], rhsInterval,
            fromElemsOp, nextLevelVectorSize);
      } else {
        auto lhsShuffleOp = cast<ShuffleOp>(lhsVector.getDefiningOp());
        auto rhsShuffleOp = cast<ShuffleOp>(rhsVector.getDefiningOp());
        shuffleMask = computePropagationShuffleMask(lhsShuffleOp, lhsInterval,
                                                    rhsShuffleOp, rhsInterval,
                                                    nextLevelVectorSize);
      }

      Value shuffleVal = vector::ShuffleOp::create(rewriter, loc, lhsVector,
                                                   rhsVector, shuffleMask);
      levelOutputs.push_back(shuffleVal);
    }

    levelInputs = std::move(levelOutputs);
    ++currentLevel;
  }

  assert(levelInputs.size() == 1 && "Should have exactly one result");
  return levelInputs.front();
}

/// Gather and unique all the `vector.to_elements` operations that feed the
/// `vector.from_elements` operation. The `vector.to_elements` operations are
/// returned in order of appearance in the `vector.from_elements`'s operand
/// list.
static LogicalResult
getToElementsDefiningOps(FromElementsOp fromElemsOp,
                         SmallVectorImpl<ToElementsOp> &toElemsDefs) {
  SetVector<ToElementsOp> toElemsDefsSet;
  for (Value element : fromElemsOp.getElements()) {
    auto toElemsOp = element.getDefiningOp<ToElementsOp>();
    if (!toElemsOp)
      return failure();
    toElemsDefsSet.insert(toElemsOp);
  }

  toElemsDefs.assign(toElemsDefsSet.begin(), toElemsDefsSet.end());
  return success();
}

/// Pass to rewrite `vector.to_elements` + `vector.from_elements` sequences into
/// a tree of `vector.shuffle` operations. Only 1-D input vectors are supported
/// for now.
struct ToFromElementsToShuffleTreeRewrite final
    : OpRewritePattern<vector::FromElementsOp> {

  using Base::Base;

  LogicalResult matchAndRewrite(vector::FromElementsOp fromElemsOp,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = fromElemsOp.getType();
    if (resultType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          fromElemsOp,
          "multi-dimensional output vectors are not supported yet");
    if (resultType.isScalable())
      return rewriter.notifyMatchFailure(
          fromElemsOp,
          "'vector.from_elements' does not support scalable vectors");

    // Gather all the `vector.to_elements` operations that feed the
    // `vector.from_elements` operation. Other op definitions are not supported.
    SmallVector<ToElementsOp> toElemsDefs;
    if (failed(getToElementsDefiningOps(fromElemsOp, toElemsDefs)))
      return rewriter.notifyMatchFailure(fromElemsOp, "unsupported sources");

    if (llvm::any_of(toElemsDefs, [](ToElementsOp toElemsOp) {
          return toElemsOp.getSource().getType().getRank() != 1;
        })) {
      return rewriter.notifyMatchFailure(
          fromElemsOp, "multi-dimensional input vectors are not supported yet");
    }

    if (llvm::any_of(toElemsDefs, [](ToElementsOp toElemsOp) {
          return !toElemsOp.getSource().getType().hasRank();
        })) {
      return rewriter.notifyMatchFailure(fromElemsOp,
                                         "0-D vectors are not supported");
    }

    // Avoid generating a shuffle tree for trivial `vector.to_elements` ->
    // `vector.from_elements` forwarding cases that do not require shuffling.
    if (toElemsDefs.size() == 1) {
      ToElementsOp toElemsOp0 = toElemsDefs.front();
      if (llvm::equal(fromElemsOp.getElements(), toElemsOp0.getResults())) {
        return rewriter.notifyMatchFailure(
            fromElemsOp, "trivial forwarding case does not require shuffling");
      }
    }

    VectorShuffleTreeBuilder shuffleTreeBuilder(fromElemsOp, toElemsDefs);
    if (failed(shuffleTreeBuilder.computeShuffleTree()))
      return rewriter.notifyMatchFailure(fromElemsOp,
                                         "failed to compute shuffle tree");

    Value finalShuffle = shuffleTreeBuilder.generateShuffleTree(rewriter);
    rewriter.replaceOp(fromElemsOp, finalShuffle);
    return success();
  }
};

struct LowerVectorToFromElementsToShuffleTreePass
    : public vector::impl::LowerVectorToFromElementsToShuffleTreeBase<
          LowerVectorToFromElementsToShuffleTreePass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateVectorToFromElementsToShuffleTreePatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void mlir::vector::populateVectorToFromElementsToShuffleTreePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ToFromElementsToShuffleTreeRewrite>(patterns.getContext(),
                                                   benefit);
}
