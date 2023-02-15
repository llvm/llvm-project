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
/// type, and returns it as type `memref<? x $tp>` (staticShape=false) or
/// `memref<$sz x $tp>` (staticShape=true).
Value genAlloca(OpBuilder &builder, Location loc, unsigned sz, Type tp,
                bool staticShape = false);

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

/// Generates a 1D MemRefType with a dynamic size. When withLayout is set, the
/// returned memref has a layout has unknown strides and offsets. Otherwise,
/// a memref with a standard unit stride zero offset layout is returned.
inline MemRefType get1DMemRefType(Type etp, bool withLayout) {
  auto layout = withLayout ? StridedLayoutAttr::StridedLayoutAttr::get(
                                 etp.getContext(), ShapedType::kDynamic,
                                 {ShapedType::kDynamic})
                           : StridedLayoutAttr();
  return MemRefType::get(ShapedType::kDynamic, etp, layout);
}

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

/// Converts the vector indices and store it into the memory pointed by
/// `ind`, apply (optional) `offset` on `offsetDim`.
void storeIndices(OpBuilder &builder, Location loc, unsigned rank, Value ind,
                  ValueRange ivs, unsigned offsetDim = 0,
                  Value offset = Value());

/// Reshapes the linear values buffer for an annotated all dense sparse tensor
/// to match the shape of the corresponding dense tensor to support direct
/// access of the buffer through indices.
Value reshapeValuesToLevels(OpBuilder &builder, Location loc,
                            SparseTensorEncodingAttr enc, ValueRange dimSizes,
                            Value valuesBuffer, Value idxBuffer);

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

/// Generates a constant of `i64` type.
inline Value constantI64(OpBuilder &builder, Location loc, int64_t i) {
  return builder.create<arith::ConstantIntOp>(loc, i, 64);
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

/// Infers the result type and generates ToPointersOp.
Value genToPointers(OpBuilder &builder, Location loc, Value tensor, Level lvl);

/// Infers the result type and generates ToIndicesOp. If the lvl is within a COO
/// region, the result type is a memref with unknown stride and offset.
/// Otherwise, the result type is a memref without any specified layout.
Value genToIndices(OpBuilder &builder, Location loc, Value tensor, Level lvl,
                   Level cooStart);

/// Infers the result type and generates ToValuesOp.
Value genToValues(OpBuilder &builder, Location loc, Value tensor);

/// Generates code to retrieve the values size for the sparse tensor.
Value genValMemSize(OpBuilder &builder, Location loc, Value tensor);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENUTILS_H_
