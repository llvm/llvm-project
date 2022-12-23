//===- CodegenUtils.h - Utilities for generating MLIR -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for generating MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"

namespace mlir {

class Location;
class Type;
class Value;

namespace sparse_tensor {

/// Shorthand aliases for the `emitCInterface` argument to `getFunc()`,
/// `createFuncCall()`, and `replaceOpWithFuncCall()`.
enum class EmitCInterface : bool { Off = false, On = true };

//===----------------------------------------------------------------------===//
// ExecutionEngine/SparseTensorUtils helper functions.
//===----------------------------------------------------------------------===//

/// Converts an overhead storage bitwidth to its internal type-encoding.
OverheadType overheadTypeEncoding(unsigned width);

/// Converts an overhead storage type to its internal type-encoding.
OverheadType overheadTypeEncoding(Type tp);

/// Converts the internal type-encoding for overhead storage to an mlir::Type.
Type getOverheadType(Builder &builder, OverheadType ot);

/// Returns the OverheadType for pointer overhead storage.
OverheadType pointerOverheadTypeEncoding(SparseTensorEncodingAttr enc);

/// Returns the OverheadType for index overhead storage.
OverheadType indexOverheadTypeEncoding(SparseTensorEncodingAttr enc);

/// Returns the mlir::Type for pointer overhead storage.
Type getPointerOverheadType(Builder &builder, SparseTensorEncodingAttr enc);

/// Returns the mlir::Type for index overhead storage.
Type getIndexOverheadType(Builder &builder, SparseTensorEncodingAttr enc);

/// Convert OverheadType to its function-name suffix.
StringRef overheadTypeFunctionSuffix(OverheadType ot);

/// Converts an overhead storage type to its function-name suffix.
StringRef overheadTypeFunctionSuffix(Type overheadTp);

/// Converts a primary storage type to its internal type-encoding.
PrimaryType primaryTypeEncoding(Type elemTp);

/// Convert PrimaryType to its function-name suffix.
StringRef primaryTypeFunctionSuffix(PrimaryType pt);

/// Converts a primary storage type to its function-name suffix.
StringRef primaryTypeFunctionSuffix(Type elemTp);

//===----------------------------------------------------------------------===//
// Misc code generators and utilities.
//===----------------------------------------------------------------------===//

/// Generates a 1-valued attribute of the given type.  This supports
/// all the same types as `getZeroAttr`; however, unlike `getZeroAttr`,
/// for unsupported types we raise `llvm_unreachable` rather than
/// returning a null attribute.
Attribute getOneAttr(Builder &builder, Type tp);

/// Generates the comparison `v != 0` where `v` is of numeric type.
/// For floating types, we use the "unordered" comparator (i.e., returns
/// true if `v` is NaN).
Value genIsNonzero(OpBuilder &builder, Location loc, Value v);

/// Computes the shape of destination tensor of a reshape operator. This is only
/// used when operands have dynamic shape. The shape of the destination is
/// stored into dstShape.
void genReshapeDstShape(Location loc, PatternRewriter &rewriter,
                        SmallVectorImpl<Value> &dstShape,
                        ArrayRef<Value> srcShape,
                        ArrayRef<int64_t> staticDstShape,
                        ArrayRef<ReassociationIndices> reassociation);

/// Translate indices during a reshaping operation.
void translateIndicesArray(OpBuilder &builder, Location loc,
                           ArrayRef<ReassociationIndices> reassociation,
                           ValueRange srcIndices, ArrayRef<Value> srcShape,
                           ArrayRef<Value> dstShape,
                           SmallVectorImpl<Value> &dstIndices);

/// Returns a function reference (first hit also inserts into module). Sets
/// the "_emit_c_interface" on the function declaration when requested,
/// so that LLVM lowering generates a wrapper function that takes care
/// of ABI complications with passing in and returning MemRefs to C functions.
FlatSymbolRefAttr getFunc(ModuleOp module, StringRef name, TypeRange resultType,
                          ValueRange operands, EmitCInterface emitCInterface);

/// Creates a `CallOp` to the function reference returned by `getFunc()` in
/// the builder's module.
func::CallOp createFuncCall(OpBuilder &builder, Location loc, StringRef name,
                            TypeRange resultType, ValueRange operands,
                            EmitCInterface emitCInterface);

/// Returns the equivalent of `void*` for opaque arguments to the
/// execution engine.
Type getOpaquePointerType(OpBuilder &builder);

/// Generates an uninitialized temporary buffer of the given size and
/// type, but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`).
Value genAlloca(OpBuilder &builder, Location loc, Value sz, Type tp);

/// Generates an uninitialized temporary buffer of the given size and
/// type, but returns it as type `memref<? x $tp>` (rather than as type
/// `memref<$sz x $tp>`).
Value genAlloca(OpBuilder &builder, Location loc, unsigned sz, Type tp);

/// Generates an uninitialized temporary buffer with room for one value
/// of the given type, and returns the `memref<$tp>`.
Value genAllocaScalar(OpBuilder &builder, Location loc, Type tp);

/// Generates a temporary buffer, initializes it with the given contents,
/// and returns it as type `memref<? x $tp>` (rather than specifying the
/// size of the buffer).
Value allocaBuffer(OpBuilder &builder, Location loc, ValueRange values);

/// Generates code to allocate a buffer of the given type, and zero
/// initialize it.  If the buffer type has any dynamic sizes, then the
/// `sizes` parameter should be as filled by sizesFromPtr(); that way
/// we can reuse the genDimSizeCall() results generated by sizesFromPtr().
Value allocDenseTensor(OpBuilder &builder, Location loc,
                       RankedTensorType tensorTp, ValueRange sizes);

/// Generates code to deallocate a dense buffer.
void deallocDenseTensor(OpBuilder &builder, Location loc, Value buffer);

/// Generates the code to read the value from tensor[ivs]. The generated code
/// looks like the following and the insertion point after this routine is
/// inside the if-then branch behind the assignment to ind.
///    if (tensor[ivs] != 0)
///      insert_point
Value genValueForDense(OpBuilder &builder, Location loc, Value tensor,
                       ValueRange ivs);

/// Generates the loop structure to iterate over a dense tensor or a sparse
/// tensor constant to support the lowering of dense-to-sparse convert operator.
//
// The loop to iterate a dense tensor:
//   for i1 in dim1
//    ..
//     for ik in dimk
//       val = a[i1,..,ik]
//       if val != 0
//         loop-body
//
// The loop to iterate a sparse tensor constant:
//   for i in range(NNZ)
//     val = values[i]
//     [i1,..,ik] = indices[i]
//     loop-body
void genDenseTensorOrSparseConstantIterLoop(
    OpBuilder &builder, Location loc, Value src, unsigned rank,
    function_ref<void(OpBuilder &, Location, Value, ValueRange)> bodyBuilder);

/// Populates given sizes array from dense tensor or sparse tensor constant.
void sizesFromSrc(OpBuilder &builder, SmallVectorImpl<Value> &sizes,
                  Location loc, Value src);

/// Scans to top of generated loop.
Operation *getTop(Operation *op);

/// Iterate over a sparse constant, generates constantOp for value and indices.
/// E.g.,
/// sparse<[ [0], [28], [31] ],
///          [ (-5.13, 2.0), (3.0, 4.0), (5.0, 6.0) ] >
/// =>
/// %c1 = arith.constant 0
/// %v1 = complex.constant (5.13, 2.0)
/// callback({%c1}, %v1)
///
/// %c2 = arith.constant 28
/// %v2 = complex.constant (3.0, 4.0)
/// callback({%c2}, %v2)
///
/// %c3 = arith.constant 31
/// %v3 = complex.constant (5.0, 6.0)
/// callback({%c3}, %v3)
void foreachInSparseConstant(
    Location loc, RewriterBase &rewriter, SparseElementsAttr attr,
    function_ref<void(ArrayRef<Value>, Value)> callback);

//===----------------------------------------------------------------------===//
// Inlined constant generators.
//
// All these functions are just wrappers to improve code legibility;
// therefore, we mark them as `inline` to avoid introducing any additional
// overhead due to the legibility.
//
// TODO: Ideally these should move upstream, so that we don't
// develop a design island.  However, doing so will involve
// substantial design work.  For related prior discussion, see
// <https://llvm.discourse.group/t/evolving-builder-apis-based-on-lessons-learned-from-edsc/879>
//===----------------------------------------------------------------------===//

/// Generates a 0-valued constant of the given type.  In addition to
/// the scalar types (`ComplexType`, ``FloatType`, `IndexType`,
/// `IntegerType`), this also works for `RankedTensorType` and `VectorType`
/// (for which it generates a constant `DenseElementsAttr` of zeros).
inline Value constantZero(OpBuilder &builder, Location loc, Type tp) {
  if (auto ctp = tp.dyn_cast<ComplexType>()) {
    auto zeroe = builder.getZeroAttr(ctp.getElementType());
    auto zeroa = builder.getArrayAttr({zeroe, zeroe});
    return builder.create<complex::ConstantOp>(loc, tp, zeroa);
  }
  return builder.create<arith::ConstantOp>(loc, tp, builder.getZeroAttr(tp));
}

/// Generates a 1-valued constant of the given type.  This supports all
/// the same types as `constantZero`.
inline Value constantOne(OpBuilder &builder, Location loc, Type tp) {
  if (auto ctp = tp.dyn_cast<ComplexType>()) {
    auto zeroe = builder.getZeroAttr(ctp.getElementType());
    auto onee = getOneAttr(builder, ctp.getElementType());
    auto zeroa = builder.getArrayAttr({onee, zeroe});
    return builder.create<complex::ConstantOp>(loc, tp, zeroa);
  }
  return builder.create<arith::ConstantOp>(loc, tp, getOneAttr(builder, tp));
}

/// Generates a constant of `index` type.
inline Value constantIndex(OpBuilder &builder, Location loc, int64_t i) {
  return builder.create<arith::ConstantIndexOp>(loc, i);
}

/// Generates a constant of `i32` type.
inline Value constantI32(OpBuilder &builder, Location loc, int32_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 32);
}

/// Generates a constant of `i16` type.
inline Value constantI16(OpBuilder &builder, Location loc, int16_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 16);
}

/// Generates a constant of `i8` type.
inline Value constantI8(OpBuilder &builder, Location loc, int8_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 8);
}

/// Generates a constant of `i1` type.
inline Value constantI1(OpBuilder &builder, Location loc, bool b) {
  return builder.create<arith::ConstantIntOp>(loc, b, 1);
}

/// Generates a constant of the given `Action`.
inline Value constantAction(OpBuilder &builder, Location loc, Action action) {
  return constantI32(builder, loc, static_cast<uint32_t>(action));
}

/// Generates a constant of the internal type-encoding for overhead storage.
inline Value constantOverheadTypeEncoding(OpBuilder &builder, Location loc,
                                          unsigned width) {
  return constantI32(builder, loc,
                     static_cast<uint32_t>(overheadTypeEncoding(width)));
}

/// Generates a constant of the internal type-encoding for pointer
/// overhead storage.
inline Value constantPointerTypeEncoding(OpBuilder &builder, Location loc,
                                         SparseTensorEncodingAttr enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getPointerBitWidth());
}

/// Generates a constant of the internal type-encoding for index overhead
/// storage.
inline Value constantIndexTypeEncoding(OpBuilder &builder, Location loc,
                                       SparseTensorEncodingAttr enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getIndexBitWidth());
}

/// Generates a constant of the internal type-encoding for primary storage.
inline Value constantPrimaryTypeEncoding(OpBuilder &builder, Location loc,
                                         Type elemTp) {
  return constantI32(builder, loc,
                     static_cast<uint32_t>(primaryTypeEncoding(elemTp)));
}

/// Generates a constant of the internal dimension level type encoding.
inline Value constantDimLevelTypeEncoding(OpBuilder &builder, Location loc,
                                          DimLevelType dlt) {
  return constantI8(builder, loc, static_cast<uint8_t>(dlt));
}

inline bool isZeroRankedTensorOrScalar(Type type) {
  auto rtp = type.dyn_cast<RankedTensorType>();
  return !rtp || rtp.getRank() == 0;
}

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

class SparseTensorLoopEmitter {
public:
  /// Optional callback function to setup dense output tensors when
  /// initializing the loop emitter (e.g., to fill a dense output with zeros).
  using OutputUpdater = function_ref<Value(OpBuilder &builder, Location loc,
                                           Value memref, Value tensor)>;

  SparseTensorLoopEmitter() = default;

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

  explicit SparseTensorLoopEmitter(ValueRange tensors,
                                   StringAttr loopTag = nullptr,
                                   bool hasOutput = false,
                                   bool isSparseOut = false,
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
                  Value iv, StringAttr loopTag)
        : tids(tids), dims(dims), loop(loop), iv(iv) {
      // Attached a special tag to loop emitter generated loop.
      if (loopTag)
        loop->setAttr(SparseTensorLoopEmitter::getLoopEmitterLoopAttrName(),
                      loopTag);
    }
    // TODO: maybe use a vector<pair> for tid and dim?
    // The set of tensors that the loop is operating on
    const llvm::SmallVector<size_t> tids;
    // The corresponding dims for the tensors
    const llvm::SmallVector<size_t> dims;
    const Operation *loop; // the loop operation
    const Value iv;        // the induction variable for the loop
  };

  /// Linearizes address for dense dimension (i.e., p = (i * d0) + j).
  Value genAddress(OpBuilder &builder, Location loc, size_t tid, size_t dim,
                   Value iv) {
    Value p = dim == 0 ? constantIndex(builder, loc, 0) : pidxs[tid][dim - 1];
    Value mul = builder.create<arith::MulIOp>(loc, highs[tid][dim], p);
    Value add = builder.create<arith::AddIOp>(loc, mul, iv);
    return add;
  }

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

  // Loop Stack, stores the information of all the nested loops that are
  // alive.
  std::vector<LoopLevelInfo> loopStack;

  // Loop Sequence Stack, stores the unversial index for the current loop
  // sequence.
  std::vector<Value> loopSeqStack;

  // Maps AffineDimExpr to the index of the loop in loopStack.
  // TODO: We should probably use a callback function here to make it more
  // general.
  std::vector<unsigned> sparsiferLoopLvlMap;

  // TODO: not yet used, it should track the current level for each tensor
  // to help eliminate `dim` paramters from above APIs.
  // std::vector<size_t> curLv;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_
