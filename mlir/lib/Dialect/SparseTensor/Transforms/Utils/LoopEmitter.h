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

#include "SparseTensorLevel.h"

#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
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

  // Map from [tid, lvl] to a list of dependent [LoopId, coeffecient] for
  // subscript expressions on sparse tensors.
  //
  // E.g., for affine index (2 * d0 + d1), it depends on loop d0 and d1 (for
  // affine expression reduction) and uses 2 and 1 for coefficients on d0, d1
  // respectively. If the list is empty, it means that there is no affine
  // expression on the input [tid, lvl].
  //
  // NOTE: LoopEmitter assumes that the loop id is consistent with the loop
  // order, i.e., loop `d0` will be generated before loop `d1`.
  using DependentLvlGetter =
      function_ref<std::vector<std::pair<LoopId, unsigned>>(TensorId, Level)>;

  LoopEmitter() = default;

  /// Takes an array of input tensors, which the generated loops will
  /// iterate over.  Each tensor is given a `TensorId` (numerically equal
  /// to the position of that tensor `Value` in the array).  Setting
  /// `isSparseOut` indicates that the sparse output tensor is empty,
  /// so the loop emitter will generate loops over it according to the
  /// level-sizes.
  void
  initialize(ValueRange tensors, StringAttr loopTag = nullptr,
             bool hasOutput = false, bool isSparseOut = false,
             unsigned numLoops = 0, DependentLvlGetter getter = nullptr,
             SparseEmitStrategy emitStrategy = SparseEmitStrategy::kFunctional);

  explicit LoopEmitter(
      ValueRange tensors, StringAttr loopTag = nullptr, bool hasOutput = false,
      bool isSparseOut = false, unsigned numLoops = 0,
      DependentLvlGetter getter = nullptr,
      SparseEmitStrategy emitStrategy = SparseEmitStrategy::kFunctional);

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

  /// Emits the address for a dense level based on the value evaluated by the
  /// provided affine expression.
  void locateLvlAtAffineAddress(OpBuilder &builder, Location loc,
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
      bool needsUniv = false);

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

  ///
  /// Getters.
  ///
  SmallVector<Value> getValPosits(TensorId tid) const {
    SmallVector<Value> batchCrds = iters[tid].back().back()->getBatchCrds();
    Value lastLvlPos = iters[tid].back().back()->getCurPosition().first;
    batchCrds.push_back(lastLvlPos);
    return batchCrds;
  };
  Value getCoord(TensorId tid, Level lvl) const {
    return getCurIterator(tid, lvl).getCrd();
  };
  const std::vector<Value> &getValBuffer() const { return valBuffer; };

  constexpr static llvm::StringLiteral getLoopEmitterLoopAttrName() {
    return llvm::StringLiteral("Emitted from");
  }

private:
  ///
  /// Structure definitions that hold different kinds of loops information.
  ///

  // LoopInfo stores information of a loop generated by LoopEmitter. E.g.,
  // the set of tensors levels that the loop is iterating over.
  struct LoopInfo final {
    LoopInfo(ArrayRef<TensorLevel> tidLvls, Operation *loop, Block *userBlock,
             Value iv, StringAttr loopTag)
        : tidLvls(tidLvls), loop(loop), userCodeBlock(userBlock), iv(iv) {
      // Attached a special tag to loop emitter generated loop.
      if (loopTag)
        loop->setAttr(LoopEmitter::getLoopEmitterLoopAttrName(), loopTag);
    }
    // The set of <tensor, lvl>, with *only* trivial index expressions, that are
    // used as the condition for the generated loop. Extra information is
    // required for levels with non-tivial index expressions, which is
    // maintained by the sliceDrivenInfo array below.
    const llvm::SmallVector<TensorLevel> tidLvls;
    const Operation *loop;      // the loop operation
    Block *const userCodeBlock; // the block holding users' generated code.
    const Value iv;             // the induction variable for the loop
  };

  void categorizeIterators(ArrayRef<TensorLevel> tidLvls,
                           SmallVectorImpl<SparseIterator *> &raIters,
                           SmallVectorImpl<SparseIterator *> &spIters);
  ///
  /// LoopEmitter internal helper functions.
  ///

  using LoopBodyBuilder = llvm::function_ref<void(OpBuilder &, Location, Value,
                                                  MutableArrayRef<Value>)>;

  /// Whether the list of the sparse condition should be iterated by for loop.
  bool shouldIteratedByForLoop(ArrayRef<SparseIterator *> spIters);

  /// Generates instructions to compute the coordinate of tensors[tid][lvl]
  /// under the current loop context.  The final argument is the
  /// collapsed-output level, whereas this function handles converting
  /// that to the uncollapsed-input level
  Value genSparseCrd(OpBuilder &builder, Location loc, TensorId tid,
                     Level dstLvl);

  bool isSynTensor(TensorId tid) const { return tid == getSynTensorId(); }

  bool isOutputTensor(TensorId tid) const {
    return hasOutput && tid == getOutTensorId();
  }

  bool isSparseOutput(TensorId tid) const {
    return isOutputTensor(tid) && isSparseOut;
  }

  bool isValidLevel(TensorId tid, Level lvl) const {
    return tid < lvls.size() && lvl < lvls[tid].size();
  }

  /// Prepares loop for iterating over `tensor[lvl]`, under the assumption
  /// that `tensor[0...lvl-1]` loops have already been set up.
  void prepareLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                                  TensorId tid, Level lvl);

  /// Emits a for loop to iterate over a tensor level with the provided
  /// lower bound `lo` and upper bound `hi`. Apart from iterating just
  /// single tensor level, for loops can be used for slice-driven loop on
  /// dense level too.
  /// Returns a pair: the loop generated and the value for the induction
  /// variable.
  std::pair<Operation *, Value>
  emitForLoopOverTensorAtLvl(OpBuilder &builder, Location loc,
                             SparseIterator &iter, MutableArrayRef<Value> reduc,
                             bool isParallel);

  /// Emits a while loop to co-iterate over a list of sparse condition, or
  /// (complex) single sparse condition that can not be handled by for loop
  /// (e.g., index reduction loop).
  /// Returns a pair: the loop generated and the value for the induction
  /// variable (which is the minimum coordinate of all the tensor that being
  /// iterated).
  std::pair<Operation *, Value>
  emitWhileLoopOverTensorsAtLvls(OpBuilder &builder, Location loc,
                                 ArrayRef<SparseIterator *> iters,
                                 MutableArrayRef<Value> reduc, bool needsUniv);

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

  void initSubSectIterator(OpBuilder &builder, Location loc);

  /// Get the reduced number of contraints on tensor[tid][lvl].
  unsigned redDepOnLevel(TensorId tid, Level lvl) const {
    return levelReducedDep[tid][lvl];
  };

  SparseIterator &getCurIterator(TensorId tid, Level lvl) const {
    if (dependentLvlMap[tid][lvl].empty())
      return *iters[tid][lvl].back();

    assert(redDepOnLevel(tid, lvl) >= 1);
    return *iters[tid][lvl][redDepOnLevel(tid, lvl) - 1];
  }

  std::unique_ptr<SparseIterator>
  makeLevelIterator(OpBuilder &builder, Location loc, TensorId tid, Level l);

  /// A optional string attribute that should be attached to the loop
  /// generated by loop emitter, it might help following passes to identify
  /// loops that operates on sparse tensors more easily.
  StringAttr loopTag;
  /// Whether the loop emitter needs to treat the last tensor as the output
  /// tensor.
  bool hasOutput;
  bool isSparseOut;
  SparseEmitStrategy emitStrategy;

  //
  // Fields which have `numTensor` many entries.
  //

  /// Input and (optional) output tensors.
  std::vector<Value> tensors;
  std::vector<Value> loopHighs;
  std::vector<std::vector<std::unique_ptr<SparseTensorLevel>>> lvls;
  std::vector<std::vector<std::vector<std::unique_ptr<SparseIterator>>>> iters;
  std::vector<Value> valBuffer; // to_value

  // Map from [tid, level] to a list of dependent [tidlevel, coefficient].
  // See comments for `DependentLvlGetter`.
  std::vector<std::vector<std::vector<std::pair<LoopId, unsigned>>>>
      dependentLvlMap;

  // The (size, stride) for each conceptual slice used for index reduction
  // loops.
  std::vector<std::vector<std::vector<std::pair<Value, unsigned>>>> sliceMeta;

  // The number of reduced dependencies on a tensor level so far.
  std::vector<std::vector<unsigned>> levelReducedDep;

  //
  // Fields which have at most `numLoops` many entries.
  //

  /// Loop Stack, stores the information of all the nested loops that are
  /// alive.
  std::vector<LoopInfo> loopStack;

  // Loop Sequence Stack, stores the universal index for the current loop
  // sequence. and a list of tid level that the loop sequence traverse.
  std::vector<std::pair<Value, std::vector<TensorLevel>>> loopSeqStack;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_LOOPEMITTER_H_
