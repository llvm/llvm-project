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
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/ExecutionEngine/SparseTensor/Enums.h"
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
// SparseTensorLoopEmiter class, manages sparse tensors and helps to generate
// loop structure to (co-iterate) sparse tensors.
//
// An example usage:
// To generate following loops over T1<?x?> and T2<?x?>
//
// for i in T1[0] {
//   for j : T2[0] {
//     for k : T1[1] {}
//     for k : T2[1] {}
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
// for 0 -> 3:
//    loopEmiter.exitCurrentLoop();
//===----------------------------------------------------------------------===//

// TODO: Sparsification should also rely on this class to generate loops.
class SparseTensorLoopEmitter {
public:
  /// Constructor: take an array of tensors inputs, on which the generated loops
  /// will iterate on. The index of the tensor in the array is also the
  /// tensor id (tid) used in related functions.
  explicit SparseTensorLoopEmitter(ValueRange tensors,
                                   bool isLastOutput = false);

  ///
  /// Core functions.
  ///

  /// Starts a loop emitting session:
  /// 1. Generates all the buffers needed to iterate tensors.
  /// 2. Generates the lo/hi bounds to iterate tensors[0].
  void initializeLoopEmit(OpBuilder &builder, Location loc);

  // TODO: Gets rid of `dim` in the argument list? Track the dimension we
  // are currently at internally. Then it would be enterNextDimForTensor.

  /// Emits loop over tensor[dim], it assumes that loops between
  /// tensor[0...dim - 1] have already been generated.
  /// It also prepares to enter tensor[dim + 1].
  Operation *enterLoopOverTensorAtDim(OpBuilder &builder, Location loc,
                                      size_t tid, size_t dim,
                                      ArrayRef<Value> reduc = {});

  /// Emits a coiteration loop over a set of tensors.
  // TODO: not yet implemented
  void enterCoiterationOverTensorsAtDims(OpBuilder &builder, Location loc,
                                         ArrayRef<size_t> ts,
                                         ArrayRef<size_t> ds);

  /// Emits extra locals, since the locals might not be in simplified lattices
  /// point used to generate the loops, but are still required to generates
  /// expressions.
  Value emitExtraLocalsForTensorsAtDims(OpBuilder &builder, Location loc,
                                        size_t tid, size_t dim);

  void exitCurrentLoop();

  /// Return the array of coordinate for all the loop generated till now.
  void getCoordinateArray(SmallVectorImpl<Value> &coords) {
    for (auto &l : loopStack)
      coords.push_back(l.idx);
  }

  ///
  /// Getters.
  ///

  Value getTensorValueBuffer(size_t tid) { return valBuffer[tid]; }
  Value getLastLevelTensorPointerIndex(size_t tid) {
    return pidxs[tid].back();
  };

private:
  struct LoopLevelInfo {
    LoopLevelInfo(ArrayRef<size_t> ts, ArrayRef<size_t> ds, Value idx)
        : tensors(ts), dims(ds), idx(idx) {}
    llvm::SmallVector<size_t, 4> tensors;
    llvm::SmallVector<size_t, 4> dims;
    Value idx;
  };

  /// Return false if tid[dim] is a dense dimension that does not need to be
  /// prepared (to be used by sparsification for needUniv).
  bool prepareLoopOverTensorAtDim(OpBuilder &builder, Location loc, size_t tid,
                                  size_t dim);

  /// Input (TODO: and output) tensors.
  std::vector<Value> tensors;
  /// The dim type array for each tensor.
  std::vector<std::vector<SparseTensorEncodingAttr::DimLevelType>> dims;
  /// Sparse iteration information (by tensor and dim). These arrays
  /// are updated to remain current within the current loop.
  std::vector<std::vector<Value>> pidxs;
  std::vector<std::vector<Value>> coord;
  std::vector<std::vector<Value>> highs;
  /// Universal dense indices and upper bounds (by index). The sizes array is
  /// set once with the inferred dimension sizes.
  std::vector<std::vector<Value>> sizes;
  std::vector<std::vector<Value>> ptrBuffer; // to_pointers
  std::vector<std::vector<Value>> idxBuffer; // to_indices
  std::vector<Value> valBuffer;              // to_value

  bool isLastOutput; // Is the last tensor output tensor
  std::vector<LoopLevelInfo> loopStack;
  // TODO: not yet used, it should track the current level for each tensor
  // to help eliminate `dim` paramters from above APIs.
  std::vector<size_t> curLv;
};

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
OverheadType pointerOverheadTypeEncoding(const SparseTensorEncodingAttr &enc);

/// Returns the OverheadType for index overhead storage.
OverheadType indexOverheadTypeEncoding(const SparseTensorEncodingAttr &enc);

/// Returns the mlir::Type for pointer overhead storage.
Type getPointerOverheadType(Builder &builder,
                            const SparseTensorEncodingAttr &enc);

/// Returns the mlir::Type for index overhead storage.
Type getIndexOverheadType(Builder &builder,
                          const SparseTensorEncodingAttr &enc);

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

/// Converts the IR's dimension level type to its internal type-encoding.
DimLevelType dimLevelTypeEncoding(SparseTensorEncodingAttr::DimLevelType dlt);

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
                        SmallVector<Value, 4> &dstShape,
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
/// the scalar types (`ComplexType`, ``FloatType`, `IndexType`, `IntegerType`),
/// this also works for `RankedTensorType` and `VectorType` (for which it
/// generates a constant `DenseElementsAttr` of zeros).
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
                                         const SparseTensorEncodingAttr &enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getPointerBitWidth());
}

/// Generates a constant of the internal type-encoding for index overhead
/// storage.
inline Value constantIndexTypeEncoding(OpBuilder &builder, Location loc,
                                       const SparseTensorEncodingAttr &enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getIndexBitWidth());
}

/// Generates a constant of the internal type-encoding for primary storage.
inline Value constantPrimaryTypeEncoding(OpBuilder &builder, Location loc,
                                         Type elemTp) {
  return constantI32(builder, loc,
                     static_cast<uint32_t>(primaryTypeEncoding(elemTp)));
}

/// Generates a constant of the internal dimension level type encoding.
inline Value
constantDimLevelTypeEncoding(OpBuilder &builder, Location loc,
                             SparseTensorEncodingAttr::DimLevelType dlt) {
  return constantI8(builder, loc,
                    static_cast<uint8_t>(dimLevelTypeEncoding(dlt)));
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_
