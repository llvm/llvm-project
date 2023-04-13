//===- LoopEmitter.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORLOOPEMITTER_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORLOOPEMITTER_H_

#include <vector>

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace sparse_tensor {

//===----------------------------------------------------------------------===//
/// The position of a loop in the loop-stack, or the position of a
/// `LoopId` in a topologically-sorted list of `LoopId`s.
///
/// Although this type may have the same cardinality as `LoopId`, it must
/// not be confused with that type.  The `LoopId` type is used by the `Merger`
/// as a unique identifier for loop-variables, regardless of the ordering
/// of those loops.  Whereas the `LoopOrd` type is used by the `LoopEmitter`
/// (and `CodegenEnv`) to refer to the actual order in which loops are
/// generated.
///
/// TODO: further explicate the correspondences between these various
/// types.  In particular, since the `$dim` argument to `linalg::IndexOp`
/// is a De Bruijn index, it seems like that should correspond to `LoopOrd`,
/// and yet the `Merger` has that correspond with `LoopId` instead.
/// In addition `LoopEmitter::genAffine` has `AffineDimExpr::position`
/// correspond to `LoopId`, however it is unclear what the providence
/// of those `AffineDimExpr` is.
//
// TODO: use a struct/class rather than a typedef, so that we can actually
// typecheck this to avoid mixups in the code.
using LoopOrd = unsigned;

//===----------------------------------------------------------------------===//
// SparseTensorLoopEmiter class, manages sparse tensors and helps to
// generate loop structure to (co)-iterate sparse tensors.
//
// An example usage:
// To generate the following loops over T1<?x?> and T2<?x?>
//
// for i in TENSOR_1_0 {
//   for j : TENSOR_2_0 {
//     for k : TENSOR_1_1 {}
//     for k : TENSOR_2_1 {}
//   }
// }
//
// One can use
//
// LoopEmiter loopEmiter({T1, T1});
// loopEmiter.initializeLoopEmit();
// loopEmiter.enterLoopOverTensorAtLvl(T1, 0);
// loopEmiter.enterLoopOverTensorAtLvl(T2, 0);
// loopEmiter.enterLoopOverTensorAtLvl(T1, 1);
// loopEmiter.exitCurrentLoop();
// loopEmiter.enterLoopOverTensorAtLvl(T2, 1);
// loopEmiter.exitCurrentLoop(); // exit k
// loopEmiter.exitCurrentLoop(); // exit j
// loopEmiter.exitCurrentLoop(); // exit i
//===----------------------------------------------------------------------===//

class LoopEmitter {
public:
  /// Optional callback function to setup dense output tensors when
  /// initializing the loop emitter (e.g., to fill a dense output with zeros).
  using OutputUpdater = function_ref<Value(OpBuilder &builder, Location loc,
                                           Value memref, Value tensor)>;
  // Map from [tid, dim] to a list of dependent [tid, dim] for affine expression
  // index on sparse tensors.
  // E.g., for affine index (d0 + d1), it depends on two [tid, dim] that defines
  // d0 and d1 (for affine expression reduction).
  // If the list is empty, it means that there is no affine expression on the
  // input [tid, dim].
  // NOTE: The caller is responsible to ensure that the order of the returned
  // list to be consistent with the topological order of the iteration graph,
  // otherwise the loop emitter might reduce a wrong dependent index variable
  // when generating slice-driven loops.
  using DependentLvlGetter =
      function_ref<std::vector<std::pair<TensorId, Level>>(TensorId, Level)>;

  LoopEmitter() = default;

  /// Takes an array of input tensors, which the generated loops will
  /// iterate over.  Each tensor is given a `TensorId` (numerically equal
  /// to the position of that tensor `Value` in the array).  Setting
  /// `isSparseOut` indicates that the sparse output tensor is empty,
  /// so the loop emitter will generate loops over it according to the
  /// level-sizes.  The `topSort` array specifies the actual order in
  /// which loops are generated, thus providing a mapping from `LoopOrd`
  /// to `LoopId`.
  void initialize(ValueRange tensors, StringAttr loopTag = nullptr,
                  bool hasOutput = false, bool isSparseOut = false,
                  ArrayRef<LoopId> topSort = {},
                  DependentLvlGetter getter = nullptr);

  explicit LoopEmitter(ValueRange tensors, StringAttr loopTag = nullptr,
                       bool hasOutput = false, bool isSparseOut = false,
                       ArrayRef<LoopId> topSort = {},
                       DependentLvlGetter getter = nullptr);

  /// Starts a loop emitting session by generating all the buffers needed
  /// for iterating over the tensors.
  void initializeLoopEmit(OpBuilder &builder, Location loc,
                          OutputUpdater updater = nullptr);

  /// Generates code to compute an affine expression whose variables are
  /// `LoopId`s (i.e., `a.cast<AffineDimExpr>().getPosition()` is a valid
  /// `LoopId`).
  Value genAffine(OpBuilder &builder, Location loc, AffineExpr a);

  /// Enters a new loop sequence, the loops within the same sequence starts
  /// from the break points of previous loop instead of starting over from 0.
  /// e.g.,
  /// {
  ///   // loop sequence start.
  ///   p0 = while(xxx)
  ///     ...
  ///     break p0
  ///
  ///   // Starts loop from p0
  ///   for (i = p0; i < end; i++)
  ///     ...
  ///   // loop sequence end.
  /// }
  void enterNewLoopSeq(OpBuilder &builder, Location loc,
                       ArrayRef<TensorId> tids, ArrayRef<Level> lvls);

  /// Exits the current loop sequence, this will reset universal index to 0.
  void exitCurrentLoopSeq(OpBuilder &builder, Location loc);

  // TODO: Get rid of `lvls` in the argument list? Track the level we
  // are currently at internally. Then it would be enterNextLvlForTensor.
  // Still need a way to specify the lvl for non-annotated tensors though,
  // as those can be accessed out of order.
  //
  /// Emits loop over tensor_tid_lvl, it assumes that loops between
  /// tensor_tid_[0, lvl - 1] have already been generated.
  /// The function will also perform in-place update on the `reduc` vector to
  /// return the reduction variable used inside the generated loop.
  Operation *enterLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                      ArrayRef<TensorId> tids,
                                      ArrayRef<Level> lvls,
                                      MutableArrayRef<Value> reduc = {},
                                      bool isParallel = false);

  Operation *enterFilterLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                            TensorId tid, Level lvl,
                                            AffineExpr affine,
                                            MutableArrayRef<Value> reduc = {});

  void genDenseAffineAddress(OpBuilder &builder, Location loc, TensorId tid,
                             Level lvl, AffineExpr lvlExpr);

  /// Emits a co-iteration loop over a set of tensors.
  Operation *enterCoIterationOverTensorsAtLvls(
      OpBuilder &builder, Location loc, ArrayRef<TensorId> tids,
      ArrayRef<Level> lvls, bool needsUniv, MutableArrayRef<Value> reduc = {});

  void exitCurrentLoop(RewriterBase &rewriter, Location loc,
                       MutableArrayRef<Value> reduc = {});

  /// Fills the out-parameter with the loop induction variables for all
  /// loops in the current loop-stack.  The variables are given in the
  /// same order as the loop-stack, hence `ivs` should be indexed into
  /// by `LoopOrd` (not `LoopId`).
  void getLoopIVs(SmallVectorImpl<Value> &ivs) const {
    ivs.clear();
    ivs.reserve(getCurrentDepth());
    for (auto &l : loopStack)
      ivs.push_back(l.iv);
  }

  /// Gets the current depth of the loop-stack.  The result is given
  /// the type `LoopOrd` for the same reason as one-past-the-end iterators.
  LoopOrd getCurrentDepth() const { return loopStack.size(); }

  /// Gets loop induction variable for the given `LoopOrd`.
  Value getLoopIV(LoopOrd n) const {
    return n < getCurrentDepth() ? loopStack[n].iv : Value();
  }

  ///
  /// Getters.
  ///
  const std::vector<std::vector<Value>> &getPosits() const { return posits; };
  const std::vector<std::vector<Value>> &getCoords() const { return coords; };
  const std::vector<std::vector<Value>> &getHighs() const { return highs; };
  const std::vector<std::vector<Value>> &getPositionBuffers() const {
    return positionsBuffers;
  };
  const std::vector<std::vector<Value>> &getCoordinateBuffers() const {
    return coordinatesBuffers;
  };
  const std::vector<Value> &getValBuffer() const { return valBuffer; };

  constexpr static llvm::StringLiteral getLoopEmitterLoopAttrName() {
    return llvm::StringLiteral("Emitted from");
  }

private:
  // LoopInfo stores information of a loop generated by LoopEmitter. E.g.,
  // the set of tensors levels that the loop is iterating over.
  struct LoopInfo final {
    LoopInfo(ArrayRef<TensorId> tids, ArrayRef<Level> lvls,
             ArrayRef<TensorId> slicedTids, ArrayRef<Level> slicedLvls,
             ArrayRef<bool> sliceReduced, Operation *loop, Block *userBlock,
             Value iv, StringAttr loopTag)
        : tids(tids), lvls(lvls), slicedTids(slicedTids),
          slicedLvls(slicedLvls), sliceReduced(sliceReduced), loop(loop),
          userCodeBlock(userBlock), iv(iv) {
      // Attached a special tag to loop emitter generated loop.
      if (loopTag)
        loop->setAttr(LoopEmitter::getLoopEmitterLoopAttrName(), loopTag);
    }
    // TODO: maybe use a vector<pair> for tid and lvl?
    //       (Or compress them together with a `TensorLoopId`.)
    // The set of tensors that the loop is operating on
    const llvm::SmallVector<TensorId> tids;
    // The corresponding levels for the tensors
    const llvm::SmallVector<Level> lvls;
    // The set of tensors for slice-driven loop conditions.
    const llvm::SmallVector<TensorId> slicedTids;
    // The corresponding level for slice-driven tensors.
    const llvm::SmallVector<Level> slicedLvls;
    // Whether the tensor is fully reduced (e.g., i + j => j).
    const llvm::SmallVector<bool> sliceReduced;
    const Operation *loop;      // the loop operation
    Block *const userCodeBlock; // the block holding users' generated code.
    const Value iv;             // the induction variable for the loop
  };

  // SliceInfo stores information of an extracted slice for slice-driven loop.
  // E.g., the in-scope SSA values for the minimum coordinates and offset for
  // the slice, etc.
  struct SliceInfo final {
    // Note that we do not need to create a actual sparse tensor slice but
    // instead only need to maintain the metadata of the slice.
    SliceInfo(Value minCrd, Value offset, Value isNonEmpty,
              std::optional<Level> slicedOnLvl, unsigned depth)
        : minCrd(minCrd), offset(offset), isNonEmpty(isNonEmpty),
          slicedOnLvl(slicedOnLvl), depth(depth) {
      // TODO: use std::optional<pair<Level, minCrd>>
      assert(!slicedOnLvl || minCrd);
    }

    // Whether this is the tensor that has not yet been sliced.
    bool isInitialTensor() const { return !slicedOnLvl.has_value(); }

    Value minCrd;                     // the minimum coordinate of the slice.
    Value offset;                     // the offset of the current slice.
    Value isNonEmpty;                 // whether the slice is empty.
    std::optional<Level> slicedOnLvl; // the level on which the slice is done
    unsigned depth; // the depth (relative to dependentDimMap[tid][lvl]).
  };

  /// Linearizes address for dense dimension (i.e., p = (i * d0) + j).
  Value genAddress(OpBuilder &builder, Location loc, TensorId tid, Level lvl,
                   Value iv);

  /// Generates the segment high for a non-unique level (to fast forward
  /// duplicated coordinates).  That is, it generates the code:
  ///
  ///   crd = coordinates_tid_lvl[pos]
  ///   while (pos < pHi && coordinates_tid_lvl[pos] == crd)
  ///      pos++;
  ///   <return pos>;
  Value genSegmentHigh(OpBuilder &builder, Location loc, TensorId tid,
                       Level lvl, Value pos, Value pHi);

  /// Generates instructions to compute the coordinate of tensors[tid][lvl]
  /// under the current loop context.  The final argument is the
  /// collapsed-output level, whereas this function handles converting
  /// that to the uncollapsed-input level
  Value genSparseCrd(OpBuilder &builder, Location loc, TensorId tid,
                     Level dstLvl);

  /// Generates a predicate to determine whether the tranformed coordinates are
  /// in the given slice.
  /// Returns std::pair<Transformed coordinates, Predicate>
  std::pair<Value, Value> genSliceLegitPredicate(OpBuilder &builder,
                                                 Location loc, Value crd,
                                                 TensorId tid, Level lvl);

  unsigned getNumTensors() const { return tensors.size(); }

  bool isOutputTensor(TensorId tid) const {
    return hasOutput && tid == getNumTensors() - 1;
  }

  bool isSparseOutput(TensorId tid) const {
    return isOutputTensor(tid) && isSparseOut;
  }

  bool isValidLevel(TensorId tid, Level lvl) const {
    return tid < lvlTypes.size() && lvl < lvlTypes[tid].size();
  }

  /// Prepares loop for iterating over `tensor[lvl]`, under the assumption
  /// that `tensor[0...lvl-1]` loops have already been set up.
  void prepareLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                  TensorId tid, Level lvl);

  /// Emits extra locals, since the locals might not be in simplified lattices
  /// point used to generate the loops, but are still required to generate
  /// expressions.
  void emitExtraLocalsForTensorsAtDenseLvls(OpBuilder &builder, Location loc,
                                            ArrayRef<TensorId> tids,
                                            ArrayRef<Level> lvls);

  /// Emits a for loop to iterate over a dense level, or a sparse level that has
  /// not been sliced.
  Operation *emitForLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                        TensorId tid, Level lvl,
                                        MutableArrayRef<Value> reduc,
                                        bool isParallel);

  /// Emits a while loop to iterate over a sparse level that has been sliced.
  /// Inserts break statement when the coordinate exceeds the sliceSize;
  /// The method sets the insertion point inside the generated while loop body
  /// after the break statement before return (so that callers need to handle
  /// only in-bound coordinates).
  Operation *emitWhileLoopOverSliceAtSparseLvl(OpBuilder &builder, Location loc,
                                               Value pLo, Value pHi,
                                               Value offset, Value sliceSize,
                                               TensorId tid, Level lvl,
                                               MutableArrayRef<Value> reduc);

  /// Exits a for loop, returns the reduction results, e.g.,
  /// For sequential for loops:
  /// %ret = for () {
  ///   ...
  ///   %val = addi %args, %c
  ///   yield %val
  /// }
  /// For parallel loops, the following generated code by users:
  /// %ret = parallel () init(%args) {
  ///   ...
  ///   %val = op %args, %c
  /// }
  /// will be transformed into
  /// %ret = parallel () init(%args) {
  ///   ...
  ///   scf.reduce(%c) bb0(%0, %1){
  ///     %val = op %0, %1
  ///     scf.reduce.return %val
  ///   }
  /// }
  /// NOTE: only one instruction will be moved into reduce block,
  /// transformation will fail if multiple instructions are used to compute
  /// the reduction value. Return %ret to user, while %val is provided by
  /// users (`reduc`).
  void exitForLoop(RewriterBase &rewriter, Location loc,
                   MutableArrayRef<Value> reduc);

  /// Exits a while loop, returns the reduction results.
  void exitWhileLoop(OpBuilder &builder, Location loc,
                     MutableArrayRef<Value> reduc);

  //
  // View-based-reshape methods.
  //

  /// Get the collapse reassociation for `tensors[tid][dstLvl]`.
  /// For unreshaped operands, the reassociation is simply an identity
  /// transformation.
  ///
  /// NOTE: the result uses `Level` rather than the `int64_t` of
  /// `ReassociationIndices`, since the former gives clarity to what
  /// the values actually mean.
  ///
  /// TODO: why not do this computation when we first store the reassoc,
  /// instead of doing it every time we look it up?
  SmallVector<Level, 2> getCollapseReassociation(TensorId tid, Level dstLvl) {
    assert(tid < getNumTensors() && "Invalid TensorId");
    assert(collapseReassoc.size() == getNumTensors());
    if (const auto reassoc = collapseReassoc[tid]) {
      // TODO: store the dstLvlRank in the LoopEmitter so that we can
      // check `dstLvl < dstLvlRank` at the top; and only here need to
      // assert that `reassoc.size() == dstLvlRank`.
      assert(dstLvl < reassoc.size() && "Level is out-of-bounds");
      const auto srcLvls = reassoc[dstLvl].cast<ArrayAttr>();
      return llvm::to_vector<2>(
          llvm::map_range(srcLvls, [&](Attribute srcLvl) -> Level {
            // TODO: replace this with the converter for `LevelAttr`.
            return srcLvl.cast<IntegerAttr>().getValue().getZExtValue();
          }));
    }
    return {dstLvl};
  }

  //
  // Slice-driven loop related methods.
  //

  /// Retrieves the most recent slice on lvl. To reduce affine expression like
  /// d0 + d1 + d2, we need two slices (one of size d1 + d2, and the other of
  /// size d2). This methods returns the latter slice (of size d2), which is
  /// also the final slice on the level.
  const SliceInfo &getFinalSliceOnLvl(TensorId tid, Level lvl);

  /// Get the remaining number of constraints needed to fully *resolve*
  /// dependent levels on tensor[tid].
  unsigned remDepOnLevel(TensorId tid, Level lvl) const;

  /// Whether the tid, lvl is fully *reduced*, i.e., the non-trivial index
  /// expression has been reduced to a trivial one.
  /// E.g., A[i + j] => A[i + 2] (j is reduced)
  bool depFullyReduced(TensorId tid, Level lvl) const {
    return remDepOnLevel(tid, lvl) == 1;
  }

  /// Whether the tid, lvl is fully resolved, i.e., we entered the level already
  /// (the index on that level is determined).
  /// E.g., A[i + j] => A[2 + 3] (both i and j become invariants for inner
  /// loops).
  bool lvlFullyResolved(TensorId tid, Level lvl) const {
    return remDepOnLevel(tid, lvl) == 0;
  }

  /// Generates a whileOp to iterate over a subset of coordinates on tid on lvl
  /// using the pHi and pLo provided, the loop break on the first coordinate
  /// that exceeds the slice boundary (i.e., coord >= slice.offset +
  /// slice.size).
  std::pair<Operation *, ValueRange>
  genSliceLvlTraverseLoop(OpBuilder &builder, Location loc, Value pLo,
                          Value pHi, Value offset, Value size, TensorId tid,
                          Level lvl, ValueRange userReduc, bool genYield,
                          /*bodyBuilder=*/
                          llvm::function_ref<void(OpBuilder &, Location, Value,
                                                  MutableArrayRef<Value>)>);

  /// Generates a nested loop that iterates over tid on all the coordinates on
  /// lvl.
  ValueRange genUnResolvedSliceTreeTraverse(
      OpBuilder &builder, Location loc, Value offset, TensorId tid, Level lvl,
      size_t depth, ValueRange userReduc,
      /*bodyBody=*/
      llvm::function_ref<void(OpBuilder &, Location, Value,
                              MutableArrayRef<Value>)>);

  /// Generates code to get the first non-empty slice of tid on lvl, when all
  /// the previous level before `lvl` are resolved (or lvl is the first level).
  ///
  /// This is the simple case because the previous level are resolved into a
  /// single node in the storage tree.
  void genResolvedSliceBegin(OpBuilder &builder, Location loc, TensorId tid,
                             Level lvl);

  /// Generates code to get the first non-empty slice of tid on lvl, when
  /// the previous levels before `lvl` are unresolved
  ///
  /// This is the complex case because the previous levels corresponding to a
  /// range of nodes in the storage tree.
  void genUnResolvedSliceBegin(OpBuilder &builder, Location loc, TensorId tid,
                               Level lvl);

  /// Generates code to get the first non-empty slice of tid on lvl.
  /// return true if has already been resolved.
  bool genSliceBegin(OpBuilder &builder, Location loc, TensorId tid, Level lvl);

  /// Generates code to get the next non-empty slices of tid on lvl.
  void genSliceNextInduction(OpBuilder &builder, Location loc,
                             const Operation *whileOp, TensorId tid, Level lvl,
                             SmallVectorImpl<Value> &operands,
                             unsigned &retIdx);

  /// Generates a slice-driven while loop as follows.
  ///
  /// curSlice = getFirstNonEmptySlice(tensor).
  ///
  /// while(isNonEmpty) {
  ///   ..user code..
  ///   isNonEmpty, curSlice = getNextNonEmptySlice(curSlice)
  /// }
  Operation *emitSliceDrivenLoopOverTensorAtLvl(OpBuilder &builder,
                                                Location loc, TensorId tid,
                                                Level lvl,
                                                MutableArrayRef<Value> reduc);

  /// A optional string attribute that should be attached to the loop
  /// generated by loop emitter, it might help following passes to identify
  /// loops that operates on sparse tensors more easily.
  StringAttr loopTag;
  /// Whether the loop emitter needs to treat the last tensor as the output
  /// tensor.
  bool hasOutput;
  bool isSparseOut;

  /// The insertion point to allocate top level local variables.
  Operation *localInsertPos;

  //
  // Fields which have `numTensor` many entries.
  //
  // TODO: switch to an AOS style to avoid any possible mismatches.
  //

  /// Input and (optional) output tensors.
  std::vector<Value> tensors;
  /// Level-types for each `(TensorId, Level)` pair.
  std::vector<std::vector<DimLevelType>> lvlTypes;
  // Sparse iteration information for each `(TensorId, Level)` pair.
  // These arrays are updated to remain current within the current loop.
  // TODO: Clarify which of these are indexed by dstLvl vs srcLvl.
  //
  /// The collection of positions for a given element (one such collection
  /// for each tensor).  This is the position analogue of the "coords"
  /// naming convention.
  ///
  /// FIXME: [CLARIFY_POSITS_LVL] It's unclear which levels are used
  /// to index the `posits` array.  On the one hand `genSparseCrd`
  /// uses dstLvl; on the other hand `enterLoopOverTensorAtLvl`,
  /// `prepareLoopOverTensorAtLvl`, and `enterCoIterationOverTensorsAtLvls`
  /// uses srcLvl.  So which is it?
  std::vector<std::vector<Value>> posits;
  /// The collection of coordinates for a given element (one such
  /// collection for each tensor).
  std::vector<std::vector<Value>> coords;
  // The segment upper bound for non-uniques level after de-duplication.
  std::vector<std::vector<Value>> segHi;
  std::vector<std::vector<Value>> highs;
  std::vector<std::vector<Value>> lvlSizes;
  std::vector<std::vector<Value>> positionsBuffers;   // to_positions
  std::vector<std::vector<Value>> coordinatesBuffers; // to_coordinates
  std::vector<Value> valBuffer;                       // to_value

  //
  // Slice-driven loops related fields.
  //

  /// Whether the sparse input is a slice.
  std::vector<bool> isSparseSlices;
  /// Values related to slices.
  std::vector<std::vector<Value>> sliceOffsets;
  std::vector<std::vector<Value>> sliceStrides;

  // Map from [tid, level] to a list of dependent [tid, level].
  // See comments for `DependentDimGetter`.
  std::vector<std::vector<std::vector<std::pair<TensorId, Level>>>>
      dependentLvlMap;

  // The cached position buffer for the slices, they serve the same purpose as
  // ptrBuffer for compressed dimensions.
  // But they always starts with the first pidx pointing to coord > slice.offset
  // to avoid iteration from the beginning.
  std::vector<std::vector<std::vector<Value>>> slicePosBuffer;

  // The cached size for each slices.
  std::vector<std::vector<std::vector<Value>>> sliceSizes;

  // The number of reduced dependencies on a tensor level so far.
  std::vector<std::vector<unsigned>> levelReducedDep;

  // sliceStack[tid] holds the generated slice stack on tid.
  std::vector<std::vector<SliceInfo>> sliceStack;

  //
  // View based reshape related-fields and methods
  //

  /// Collapse Reassociations related to a specific tensor
  // TODO: support expand.
  std::vector<ArrayAttr> collapseReassoc;

  /// TODO: not yet used, it should track the current level for each tensor
  /// to help eliminate `lvls` paramters from above APIs.
  /// std::vector<Level> curLvl;

  //
  // Fields which have at most `numLoops` many entries.
  //

  /// Loop Stack, stores the information of all the nested loops that are
  /// alive.
  std::vector<LoopInfo> loopStack;

  // Loop Sequence Stack, stores the unversial index for the current loop
  // sequence. and a list of tids which was taken sliced.
  // TODO: maybe we should have a LoopSeqInfo
  std::vector<std::pair<Value, std::vector<std::tuple<TensorId, Level, bool>>>>
      loopSeqStack;

  /// Maps `LoopId` (used by `AffineDimExpr`) to `LoopOrd` (in the `loopStack`).
  /// TODO: We should probably use a callback function here to make it more
  /// general.
  std::vector<LoopOrd> loopIdToOrd;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORLOOPEMITTER_H_
