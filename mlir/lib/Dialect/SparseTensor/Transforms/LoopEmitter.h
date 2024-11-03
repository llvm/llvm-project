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
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace sparse_tensor {

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
// SparseTensorLoopEmiter loopEmiter({T1, T1});
// loopEmiter.initializeLoopEmit();
// loopEmiter.enterLoopOverTensorAtDim(T1, 0);
// loopEmiter.enterLoopOverTensorAtDim(T2, 0);
// loopEmiter.enterLoopOverTensorAtDim(T1, 1);
// loopEmiter.exitCurrentLoop();
// loopEmiter.enterLoopOverTensorAtDim(T2, 1);
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

  LoopEmitter() = default;

  /// Takes an array of tensors inputs, on which the generated loops will
  /// iterate on. The index of the tensor in the array is also the tensor id
  /// (tid) used in related functions. If isSparseOut is set, loop emitter
  /// assume that the sparse output tensor is empty, and will always generate
  /// loops on it based on the dim sizes. An optional array could be provided
  /// (by sparsification) to indicate the loop id sequence that will be
  /// generated. It is used to establish the mapping between affineDimExpr to
  /// the corresponding loop index in the loop stack that are maintained by the
  /// loop emitter.
  void initialize(ValueRange tensors, StringAttr loopTag = nullptr,
                  bool hasOutput = false, bool isSparseOut = false,
                  ArrayRef<unsigned> topSort = {});

  explicit LoopEmitter(ValueRange tensors, StringAttr loopTag = nullptr,
                       bool hasOutput = false, bool isSparseOut = false,
                       ArrayRef<unsigned> topSort = {});

  /// Starts a loop emitting session by generating all the buffers needed to
  /// iterate tensors.
  void initializeLoopEmit(OpBuilder &builder, Location loc,
                          OutputUpdater updater = nullptr);

  /// Generates a list of operations to compute the affine expression.
  Value genAffine(OpBuilder &builder, AffineExpr a, Location loc);

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
  void enterNewLoopSeq(OpBuilder &builder, Location loc, ArrayRef<size_t> tids,
                       ArrayRef<size_t> dims);

  // exit the current loop sequence, this will reset universal index to 0.
  void exitCurrentLoopSeq() {
    assert(loopSeqStack.size() == loopStack.size() + 1);
    loopSeqStack.pop_back();
  }

  // TODO: Gets rid of `dim` in the argument list? Track the dimension we
  // are currently at internally. Then it would be enterNextDimForTensor.
  // Still need a way to specify the dim for non annoated dense tensor though,
  // as it can be accessed out of order.
  /// Emits loop over tensor_tid_dim, it assumes that loops between
  /// tensor_tid_[0, dim - 1] have already been generated.
  /// The function will also perform in-place update on the `reduc` vector to
  /// return the reduction variable used inside the generated loop.
  Operation *enterLoopOverTensorAtDim(OpBuilder &builder, Location loc,
                                      ArrayRef<size_t> tids,
                                      ArrayRef<size_t> dims,
                                      MutableArrayRef<Value> reduc = {},
                                      bool isParallel = false);

  Operation *enterFilterLoopOverTensorAtDim(OpBuilder &builder, Location loc,
                                            size_t tid, size_t dim,
                                            AffineExpr affine,
                                            MutableArrayRef<Value> reduc = {});

  void genDenseAffineAddressAtCurLevel(OpBuilder &builder, Location loc,
                                       size_t tid, size_t dim,
                                       AffineExpr affine);

  /// Emits a co-iteration loop over a set of tensors.
  Operation *enterCoIterationOverTensorsAtDims(
      OpBuilder &builder, Location loc, ArrayRef<size_t> tids,
      ArrayRef<size_t> dims, bool needsUniv, MutableArrayRef<Value> reduc = {});

  void exitCurrentLoop(RewriterBase &rewriter, Location loc,
                       MutableArrayRef<Value> reduc = {});

  /// Returns the array of coordinate for all the loop generated till now.
  void getCoordinateArray(SmallVectorImpl<Value> &coords) const {
    for (auto &l : loopStack)
      coords.push_back(l.iv);
  }

  /// Gets loop induction variable at the given level.
  unsigned getCurrentDepth() const { return loopStack.size(); }

  /// Gets loop induction variable at the given level.
  Value getLoopIV(size_t level) const {
    if (level < loopStack.size())
      return loopStack[level].iv;
    return nullptr;
  }

  ///
  /// Getters.
  ///
  const std::vector<std::vector<Value>> &getPidxs() const { return pidxs; };
  const std::vector<std::vector<Value>> &getCoord() const { return coord; };
  const std::vector<std::vector<Value>> &getHighs() const { return highs; };
  const std::vector<std::vector<Value>> &getPtrBuffer() const {
    return ptrBuffer;
  };
  const std::vector<std::vector<Value>> &getIdxBuffer() const {
    return idxBuffer;
  };
  const std::vector<Value> &getValBuffer() const { return valBuffer; };

  constexpr static llvm::StringLiteral getLoopEmitterLoopAttrName() {
    return llvm::StringLiteral("Emitted from");
  }

private:
  struct LoopLevelInfo {
    LoopLevelInfo(ArrayRef<size_t> tids, ArrayRef<size_t> dims, Operation *loop,
                  Block *userBlock, Value iv, StringAttr loopTag)
        : tids(tids), dims(dims), loop(loop), userCodeBlock(userBlock), iv(iv) {
      // Attached a special tag to loop emitter generated loop.
      if (loopTag)
        loop->setAttr(LoopEmitter::getLoopEmitterLoopAttrName(), loopTag);
    }
    // TODO: maybe use a vector<pair> for tid and dim?
    // The set of tensors that the loop is operating on
    const llvm::SmallVector<size_t> tids;
    // The corresponding dims for the tensors
    const llvm::SmallVector<size_t> dims;
    const Operation *loop;      // the loop operation
    Block *const userCodeBlock; // the block holding users' generated code.
    const Value iv;             // the induction variable for the loop
  };

  /// Linearizes address for dense dimension (i.e., p = (i * d0) + j).
  Value genAddress(OpBuilder &builder, Location loc, size_t tid, size_t dim,
                   Value iv);

  bool isOutputTensor(size_t tid) {
    return hasOutput && tid == tensors.size() - 1;
  }

  bool isSparseOutput(size_t tid) { return isOutputTensor(tid) && isSparseOut; }

  /// Setups [lo, hi] for iterating tensor[dim], it assumes that tensor[0
  /// ...dims-1] has already been setup.
  void prepareLoopOverTensorAtDim(OpBuilder &builder, Location loc, size_t tid,
                                  size_t dim);

  /// Emits extra locals, since the locals might not be in simplified lattices
  /// point used to generate the loops, but are still required to generates
  /// expressions.
  void emitExtraLocalsForTensorsAtDenseDims(OpBuilder &builder, Location loc,
                                            ArrayRef<size_t> tids,
                                            ArrayRef<size_t> dims);

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
  void exitCoIterationLoop(OpBuilder &builder, Location loc,
                           MutableArrayRef<Value> reduc);

  /// A optional string attribute that should be attached to the loop
  /// generated by loop emitter, it might help following passes to identify
  /// loops that operates on sparse tensors more easily.
  StringAttr loopTag;
  /// Whether the loop emitter needs to treat the last tensor as the output
  /// tensor.
  bool hasOutput;
  bool isSparseOut;
  /// Input and (optional) output tensors.
  std::vector<Value> tensors;
  /// The dim type array for each tensor.
  std::vector<std::vector<DimLevelType>> dimTypes;
  /// Sparse iteration information (by tensor and dim). These arrays
  /// are updated to remain current within the current loop.
  std::vector<std::vector<Value>> pidxs;
  std::vector<std::vector<Value>> coord;
  std::vector<std::vector<Value>> highs;
  std::vector<std::vector<Value>> ptrBuffer; // to_pointers
  std::vector<std::vector<Value>> idxBuffer; // to_indices
  std::vector<Value> valBuffer;              // to_value

  /// Whether the sparse input is a slice.
  std::vector<bool> isSparseSlices;

  /// Loop Stack, stores the information of all the nested loops that are
  /// alive.
  std::vector<LoopLevelInfo> loopStack;

  /// Loop Sequence Stack, stores the unversial index for the current loop
  /// sequence.
  std::vector<Value> loopSeqStack;

  /// Maps AffineDimExpr to the index of the loop in loopStack.
  /// TODO: We should probably use a callback function here to make it more
  /// general.
  std::vector<unsigned> sparsiferLoopLvlMap;

  /// TODO: not yet used, it should track the current level for each tensor
  /// to help eliminate `dim` paramters from above APIs.
  /// std::vector<size_t> curLv;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_SPARSETENSORLOOPEMITTER_H_
