//===- LoopEmitter.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_LOOPEMITTER_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_LOOPEMITTER_H_

#include <vector>

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace sparse_tensor {

// A compressed <tensor id, level> pair.
using TensorLevel = unsigned;

//
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
//
class LoopEmitter {
public:
  /// Optional callback function to setup dense output tensors when
  /// initializing the loop emitter (e.g., to fill a dense output with zeros).
  using OutputUpdater = function_ref<Value(OpBuilder &builder, Location loc,
                                           Value memref, Value tensor)>;

  /// Optional callback function to set the bound for the synthetic tensor,
  /// which essentially is the dense loop bound.
  using SynTensorBoundSetter =
      function_ref<Value(OpBuilder &builder, Location loc, Level lvl)>;

  // Map from [tid, lvl] to a list of dependent [tidlvl, coeffecient] for
  // subscript expressions on sparse tensors.
  //
  // E.g., for affine index (2 * d0 + d1), it depends on two tidlvls that
  // defines d0 and d1 (for affine expression reduction) and uses 2 and 1 for
  // cofficients on d0, d1 respectively.
  // If the list is empty, it means that there is no affine expression on the
  // input [tid, lvl].
  //
  // NOTE: The caller is responsible to ensure that the order of the returned
  // list to be consistent with the topological order of the iteration graph,
  // otherwise the loop emitter might reduce a wrong dependent index variable
  // when generating slice-driven loops.
  using DependentLvlGetter =
      function_ref<std::vector<std::pair<TensorLevel, unsigned>>(TensorId,
                                                                 Level)>;

  LoopEmitter() = default;

  /// Takes an array of input tensors, which the generated loops will
  /// iterate over.  Each tensor is given a `TensorId` (numerically equal
  /// to the position of that tensor `Value` in the array).  Setting
  /// `isSparseOut` indicates that the sparse output tensor is empty,
  /// so the loop emitter will generate loops over it according to the
  /// level-sizes.
  void initialize(ValueRange tensors, StringAttr loopTag = nullptr,
                  bool hasOutput = false, bool isSparseOut = false,
                  unsigned numLoops = 0, DependentLvlGetter getter = nullptr);

  explicit LoopEmitter(ValueRange tensors, StringAttr loopTag = nullptr,
                       bool hasOutput = false, bool isSparseOut = false,
                       unsigned numLoops = 0,
                       DependentLvlGetter getter = nullptr);

  /// Starts a loop emitting session by generating all the buffers needed
  /// for iterating over the tensors.
  void initializeLoopEmit(OpBuilder &builder, Location loc,
                          OutputUpdater updater = nullptr,
                          SynTensorBoundSetter synSetter = nullptr);

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
                       ArrayRef<TensorLevel> tidLvls);

  /// Exits the current loop sequence, this will reset universal index to 0.
  void exitCurrentLoopSeq(OpBuilder &builder, Location loc);

  /// Enters a loop that tries to locate a coordinates in a sparse level based
  /// on the value evaluated by the provided affine expression.
  /// DEPRECATED: affine index expression should be handled by index reduction
  /// loop, filter loop-based solution is slow.
  Operation *enterFilterLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                            TensorId tid, Level lvl,
                                            AffineExpr affine,
                                            MutableArrayRef<Value> reduc = {});

  /// Emits the address for a dense level based on the value evaluated by the
  /// provided affine expression.
  /// DEPRECATED: affine index expression should be handled by index reduction
  /// loop, filter loop-based solution is slow.
  void genDenseAffineAddress(OpBuilder &builder, Location loc,
                             TensorLevel tidLvl, AffineExpr lvlExpr);

  // TODO: Get rid of `lvls` in the argument list? Track the level we
  // are currently at internally. Then it would be enterNextLvlForTensor.
  // Still need a way to specify the lvl for non-annotated tensors though,
  // as those can be accessed out of order.
  //
  /// Emits a co-iteration loop over a set of tensors.
  /// Emits loop over tensor_tid_lvl, it assumes that loops between
  /// tensor_tid_[0, lvl - 1] have already been generated.
  /// The function will also perform in-place update on the `reduc` vector to
  /// return the reduction variable used inside the generated loop.
  Operation *enterCoIterationOverTensorsAtLvls(
      OpBuilder &builder, Location loc, ArrayRef<TensorLevel> tidLvls,
      MutableArrayRef<Value> reduc = {}, bool isParallel = false,
      bool genDedup = false, bool needsUniv = false);

  /// Generates code to exit the current loop (e.g., generates yields, forwards
  /// loop induction variables, etc).
  void exitCurrentLoop(RewriterBase &rewriter, Location loc,
                       MutableArrayRef<Value> reduc = {});

  /// Get the range of values for all induction variables.
  auto getLoopIVsRange() const {
    return llvm::map_range(loopStack, [](const LoopInfo &li) { return li.iv; });
  }

  /// Fills the out-parameter with the loop induction variables for all
  /// loops in the current loop-stack.
  SmallVector<Value> getLoopIVs() const {
    return llvm::to_vector(getLoopIVsRange());
  }

  /// Gets the current depth of the loop-stack.
  LoopId getCurrentDepth() const { return llvm::range_size(getLoopIVsRange()); }

  /// Gets loop induction variable for the given loop
  Value getLoopIV(LoopId n) const {
    if (n >= getCurrentDepth())
      return Value();
    auto it = getLoopIVsRange().begin();
    std::advance(it, n);
    return *it;
  }

  /// Gets the total number of manifest tensors (excluding the synthetic
  /// tensor).
  unsigned getNumManifestTensors() const { return tensors.size(); }

  /// Gets the total number of tensors that loopEmitter is operating on.
  unsigned getNumTensors() const {
    // Manifest tensors with one synthetic tensor at the end.
    return getNumManifestTensors() + 1;
  }

  /// Gets the TensorId for synthetic tensor.
  TensorId getSynTensorId() const { return tensors.size(); }

  /// Gets the TensorId for output tensor.
  TensorId getOutTensorId() const {
    assert(hasOutput);
    return getNumManifestTensors() - 1;
  }

  /// Compresses a TensorId and Level into a TensorLevel.
  TensorLevel makeTensorLevel(TensorId t, Level l) const {
    return l * getNumTensors() + t;
  }

  /// De-compresses a TensorLevel back to a pair of TensorId and Level.
  std::pair<TensorId, Level> unpackTensorLevel(TensorLevel tidLvl) const {
    unsigned nt = getNumTensors();
    return std::make_pair(tidLvl % nt, tidLvl / nt);
  }

  /// Converts a range of TensorLevel to a range of std::pair<TensorId, Level>
  template <class ContainerTy>
  auto unpackTensorLevelRange(ContainerTy &&c) const {
    using EltTy = decltype(*c.begin());
    static_assert(std::is_same_v<llvm::remove_cvref_t<EltTy>, TensorLevel>,
                  "Must be unpacking a TensorLevel range");
    return llvm::map_range(std::forward<ContainerTy>(c), [this](EltTy tl) {
      return this->unpackTensorLevel(tl);
    });
  }

  template <class ContainerTy>
  auto unpackTensorLevelFromCondRange(ContainerTy &&c) const {
    using EltTy = decltype(*c.begin());
    static_assert(std::is_same_v<llvm::remove_cvref_t<EltTy>, TensorLvlCond>,
                  "Must be unpacking a TensorLvlCond range");
    return unpackTensorLevelRange(
        llvm::make_first_range(std::forward<ContainerTy>(c)));
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
  ///
  /// Structure definitions that hold different kinds of loops information.
  ///

  // A tuple that stored the slice-driven loop information.
  struct SliceLoopInfo final {
    SliceLoopInfo(TensorId tid, Level lvl, bool reduced)
        : tid(tid), lvl(lvl), reduced(reduced) {}
    TensorId tid;
    Level lvl;
    bool reduced;
  };
  // LoopInfo stores information of a loop generated by LoopEmitter. E.g.,
  // the set of tensors levels that the loop is iterating over.
  struct LoopInfo final {
    LoopInfo(ArrayRef<TensorLevel> trivialTidLvls,
             ArrayRef<SliceLoopInfo> sliceDrivenInfo, Operation *loop,
             Block *userBlock, Value iv, StringAttr loopTag)
        : trivialTidLvls(trivialTidLvls), sliceDrivenInfo(sliceDrivenInfo),
          loop(loop), userCodeBlock(userBlock), iv(iv) {
      // Attached a special tag to loop emitter generated loop.
      if (loopTag)
        loop->setAttr(LoopEmitter::getLoopEmitterLoopAttrName(), loopTag);
    }
    // The set of <tensor, lvl>, with *only* trivial index expressions, that are
    // used as the condition for the generated loop. Extra information is
    // required for levels with non-tivial index expressions, which is
    // maintained by the sliceDrivenInfo array below.
    const llvm::SmallVector<TensorLevel> trivialTidLvls;
    // The set of <tensor, lvl>, with *only* non-trivial index expressions, that
    // are used as the condition for the generated loop.
    const llvm::SmallVector<SliceLoopInfo> sliceDrivenInfo;
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
    SliceInfo(Value minCrd, Value offset, Value isNonEmpty, Value posTupleNum,
              std::optional<Level> slicedOnLvl, unsigned depth)
        : minCrd(minCrd), offset(offset), isNonEmpty(isNonEmpty),
          posTupleNum(posTupleNum), slicedOnLvl(slicedOnLvl), depth(depth) {
      // TODO: use std::optional<pair<Level, minCrd>>
      assert(!slicedOnLvl || minCrd);
    }

    // Whether this is the tensor that has not yet been sliced.
    bool isInitialTensor() const { return !slicedOnLvl.has_value(); }

    Value minCrd;      // the minimum coordinate of the slice.
    Value offset;      // the *absolute* offset of the current slice.
    Value isNonEmpty;  // whether the slice is empty.
    Value posTupleNum; // The number of position tuples used in the slice.
    std::optional<Level> slicedOnLvl; // the level on which the slice is done
    unsigned depth; // the depth (relative to dependentDimMap[tid][lvl]).
  };

  ///
  /// Enums for different kinds of loop conditions.
  ///

  // The bit indicating whether the loop conditions is sparse.
  static constexpr uint8_t kSparseCond = 1 << 3;
  // The bit indicating whether the loop iterates over sparse tensor slices
  // (i.e., with non-empty SliceDimAttr).
  static constexpr uint8_t kSliceCond = 1 << 2;
  // The bit indicating whether the loop iterates over tensor levels with
  // non-trivial affine index reduction.
  static constexpr uint8_t kAffineIdxCond = 1 << 1;
  // The bit indicating whether the loop iterates over tensor levels with
  // non-trivial affine index reduction, and it is not fully reduced.
  static constexpr uint8_t kAffineIdxCondUnRed = 1 << 0;

  enum class LoopCondKind : uint8_t {
    // Dense conditions.
    DenseCond = 0,
    DenseSliceCond = kSliceCond,
    DenseAffineCond = kAffineIdxCond,
    DenseAffineUnRedCond = kAffineIdxCond | kAffineIdxCondUnRed,
    // Sparse Conditions.
    SparseCond = kSparseCond,
    SparseSliceCond = kSparseCond | kSliceCond,
    SparseAffineCond = kSparseCond | kAffineIdxCond,
    SparseAffineUnRedCond = kSparseCond | kAffineIdxCond | kAffineIdxCondUnRed,
  };
  using TensorLvlCond = std::pair<TensorLevel, LoopCondKind>;

  /// Sparse or dense loop condition.
  static bool isSparseCond(LoopCondKind k) {
    return static_cast<uint8_t>(k) & kSparseCond;
  }
  static bool isDenseCond(LoopCondKind k) { return !isSparseCond(k); }

  /// Whether loops over sparse tensor slices or sparse tensors.
  static bool isSliceCond(LoopCondKind k) {
    return static_cast<uint8_t>(k) & kSliceCond;
  }

  /// Affine or trivial index expression loop condition.
  static bool isAffineIdxCond(LoopCondKind k) {
    return static_cast<uint8_t>(k) & kAffineIdxCond;
  }
  static bool isTrivalIdxCond(LoopCondKind k) { return !isAffineIdxCond(k); }

  /// Whether the affine index expression is fully reduced.
  static bool isAffineIdxUnRedCond(LoopCondKind k) {
    return isAffineIdxCond(k) && static_cast<uint8_t>(k) & kAffineIdxCondUnRed;
  }
  static bool isAffineIdxRedCond(LoopCondKind k) {
    return isAffineIdxCond(k) && !isAffineIdxUnRedCond(k);
  }

  // Whether the loop condition kind requires extra check inside the loop body.
  // E.g., to iterate over sparse tensor slice, we need to check whether the
  // current cooridnate is on the slice (e.g., due to stride) or not.
  static bool isCondWithExtraCheck(LoopCondKind k) {
    return isSparseCond(k) && (isSliceCond(k) || isAffineIdxUnRedCond(k));
  }

  static LoopCondKind makeLoopCondKind(bool isSparse, bool isSlice,
                                       bool isAffine, bool isUnRedu) {
    assert(!isUnRedu || isAffine);
    uint8_t bits = 0;
    bits = isSparse ? bits | kSparseCond : bits;
    bits = isSlice ? bits | kSliceCond : bits;
    bits = isAffine ? bits | kAffineIdxCond : bits;
    bits = isUnRedu ? bits | kAffineIdxCondUnRed : bits;
    LoopCondKind kind = static_cast<LoopCondKind>(bits);

    // Sanity checks.
    assert(isSparse == isSparseCond(kind));
    assert(isSlice == isSliceCond(kind));
    assert(isAffine == isAffineIdxCond(kind));
    assert(isUnRedu == isAffineIdxUnRedCond(kind));
    return kind;
  }

  void categorizeLoopCondition(ArrayRef<TensorLevel> tidLvls,
                               SmallVectorImpl<TensorLvlCond> &dnConds,
                               SmallVectorImpl<TensorLvlCond> &spConds);

  ///
  /// LoopEmitter internal helper functions.
  ///

  using LoopBodyBuilder = llvm::function_ref<void(OpBuilder &, Location, Value,
                                                  MutableArrayRef<Value>)>;

  /// Whether the list of the sparse condition should be iterated by for loop.
  bool shouldIteratedByForLoop(ArrayRef<TensorLvlCond> spConds, bool genDedup);

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

  bool isSynTensor(TensorId tid) const { return tid == getSynTensorId(); }

  bool isOutputTensor(TensorId tid) const {
    return hasOutput && tid == getOutTensorId();
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

  /// Enter dense tensor levels. Since the dense tensor condition could be
  /// optimized from the loop condition, we need to compute the
  /// positions/coordinates inside the loop body.
  void enterTensorsAtDenseLvls(OpBuilder &builder, Location loc,
                               ArrayRef<TensorLvlCond> dnConds, Value iv,
                               SmallVectorImpl<SliceLoopInfo> &sliceInfo);

  /// Emits a for loop to iterate over a tensor level with the provided
  /// lower bound `lo` and upper bound `hi`. Apart from iterating just
  /// single tensor level, for loops can be used for slice-driven loop on
  /// dense level too.
  /// Returns a pair: the loop generated and the value for the induction
  /// variable.
  std::pair<Operation *, Value>
  emitForLoopOverTensorAtLvl(OpBuilder &builder, Location loc, TensorId tid,
                             Level lvl, Value lo, Value hi,
                             MutableArrayRef<Value> reduc, bool isParallel);

  /// Emits a while loop to co-iterate over a list of sparse condition, or
  /// (complex) single sparse condition that can not be handled by for loop
  /// (e.g., index reduction loop).
  /// Returns a pair: the loop generated and the value for the induction
  /// variable (which is the minimum coordinate of all the tensor that being
  /// iterated).
  std::pair<Operation *, Value>
  emitWhileLoopOverTensorsAtLvls(OpBuilder &builder, Location loc,
                                 ArrayRef<TensorLvlCond> spConds,
                                 MutableArrayRef<Value> reduc, bool needsUniv);

  /// Generates the while loop condition for the given tensor level condition.
  Value genWhileLoopConditions(OpBuilder &builder, Location loc, ValueRange ivs,
                               TensorLvlCond cond);

  /// Generates the while loop body for the given tensor level condition.
  std::optional<Value> genWhileLoopBody(OpBuilder &builder, Location loc,
                                        ValueRange ivs, TensorLvlCond cond);

  /// Generates the values (to forward the loop) if the extra check failes.
  /// E.g., to iterate over a sparse tensor slice, we need:
  ///
  /// pos = onSlice(curCrd) ? pos : pos + 1
  ///
  /// to skip invalid coordinate that is included in the slice.
  ValueRange genCheckedValue(OpBuilder &builder, Location loc, Value pred,
                             ValueRange curArg, TensorLvlCond cond);

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
  // Slice-driven loop related methods.
  //

  /// Retrieves the most recent slice on lvl. To reduce affine expression like
  /// d0 + d1 + d2, we need two slices (one of size d1 + d2, and the other of
  /// size d2). This methods returns the latter slice (of size d2).
  const SliceInfo &getMostRecentSliceOnLvl(TensorId tid, Level lvl);

  /// Similar to getMostRecentSliceOnLvl, but yields error when the most recent
  /// slice is not the final slice needed to fully reduced the dependencies.
  const SliceInfo &getFinalSliceOnLvl(TensorId tid, Level lvl) {
    const SliceInfo &info = getMostRecentSliceOnLvl(tid, lvl);
    assert(info.depth == dependentLvlMap[tid][lvl].size() - 1);
    return info;
  }

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
                          Level lvl, ValueRange userReduc,
                          LoopBodyBuilder bodyBuilder);

  /// Generates a nested loop that iterates over tid on all the coordinates on
  /// lvl.
  ValueRange genUnResolvedSliceTreeTraverse(
      OpBuilder &builder, Location loc, TensorId tid,
      ArrayRef<const SliceInfo *> unResLvls,
      std::optional<std::pair<TensorId, Level>> firstResLvl,
      ValueRange userReduc, LoopBodyBuilder bodyBuilder);

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
  /// Returns a tuple of values for <NonEmpty, MinCrd, AbsOffset> (see
  /// SliceInfo) respectively.
  std::tuple<Value, Value, Value> genSliceNextInduction(OpBuilder &builder,
                                                        Location loc,
                                                        TensorId tid,
                                                        Level lvl);

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
  std::vector<std::vector<LevelType>> lvlTypes;
  // Sparse iteration information for each `(TensorId, Level)` pair.
  // These arrays are updated to remain current within the current loop.
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

  // Map from [tid, level] to a list of dependent [tidlevel, coefficient].
  // See comments for `DependentLvlGetter`.
  std::vector<std::vector<std::vector<std::pair<TensorLevel, unsigned>>>>
      dependentLvlMap;

  // The cached position buffer for the slices, they serve the same purpose as
  // ptrBuffer for compressed dimensions.
  // But they always starts with the first pidx pointing to coord > slice.offset
  // to avoid iteration from the beginning.
  std::vector<std::vector<std::vector<Value>>> slicePosBuffer;
  std::vector<std::vector<Value>> sliceTupleNxStartIdx;
  std::vector<std::vector<Value>> sliceTupleFwdCnt;
  std::vector<std::vector<bool>> trivialSlice;

  // The (size, stride) for each conceptual slice used for index reduction
  // loops.
  std::vector<std::vector<std::vector<std::pair<Value, unsigned>>>> sliceMeta;

  // The number of reduced dependencies on a tensor level so far.
  std::vector<std::vector<unsigned>> levelReducedDep;

  // sliceStack[tid] holds the generated slice stack on tid.
  std::vector<std::vector<SliceInfo>> sliceStack;

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
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_LOOPEMITTER_H_
