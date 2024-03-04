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

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_CODEGENUTILS_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_CODEGENUTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
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

/// Returns the OverheadType for position overhead storage.
OverheadType posTypeEncoding(SparseTensorEncodingAttr enc);

/// Returns the OverheadType for coordinate overhead storage.
OverheadType crdTypeEncoding(SparseTensorEncodingAttr enc);

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

/// A helper class to simplify lowering operations with/without function calls.
template <class SubClass>
class FuncCallOrInlineGenerator {
public:
  FuncCallOrInlineGenerator(TypeRange retTypes, ValueRange params, bool genCall)
      : retTypes(retTypes), params(params), genCall(genCall) {}

  // The main API invoked by clients, which abstracts away the details of
  // creating function calls from clients.
  SmallVector<Value> genCallOrInline(OpBuilder &builder, Location loc) {
    if (!genCall)
      return genImplementation(retTypes, params, builder, loc);

    // Looks up the function.
    std::string funcName = getMangledFuncName();
    ModuleOp module = getParentOpOf<ModuleOp>(builder);
    MLIRContext *context = module.getContext();
    auto result = SymbolRefAttr::get(context, funcName);
    auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());

    if (!func) {
      // Create the function if not already exist.
      OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPoint(getParentOpOf<func::FuncOp>(builder));
      func = builder.create<func::FuncOp>(
          loc, funcName,
          FunctionType::get(context, params.getTypes(), retTypes));
      func.setPrivate();
      // Set the insertion point to the body of the function.
      Block *entryBB = func.addEntryBlock();
      builder.setInsertionPointToStart(entryBB);
      ValueRange args = entryBB->getArguments();
      // Delegates to user to generate the actually implementation.
      SmallVector<Value> result =
          genImplementation(retTypes, args, builder, loc);
      builder.create<func::ReturnOp>(loc, result);
    }
    // Returns the CallOp result.
    func::CallOp call = builder.create<func::CallOp>(loc, func, params);
    return call.getResults();
  }

private:
  template <class OpTp>
  OpTp getParentOpOf(OpBuilder &builder) {
    return builder.getInsertionBlock()->getParent()->getParentOfType<OpTp>();
  }

  // CRTP: get the mangled function name (only called when genCall=true).
  std::string getMangledFuncName() {
    return static_cast<SubClass *>(this)->getMangledFuncName();
  }

  // CRTP: Client implementation.
  SmallVector<Value> genImplementation(TypeRange retTypes, ValueRange params,
                                       OpBuilder &builder, Location loc) {
    return static_cast<SubClass *>(this)->genImplementation(retTypes, params,
                                                            builder, loc);
  }

private:
  TypeRange retTypes; // The types of all returned results
  ValueRange params;  // The values of all input parameters
  bool genCall;       // Should the implemetantion be wrapped in a function
};

/// Add type casting between arith and index types when needed.
Value genCast(OpBuilder &builder, Location loc, Value value, Type dstTy);

/// Add conversion from scalar to given type (possibly a 0-rank tensor).
Value genScalarToTensor(OpBuilder &builder, Location loc, Value elem,
                        Type dstTp);

/// Generates a pointer/index load from the sparse storage scheme. Narrower
/// data types need to be zero extended before casting the value into the
/// index type used for looping and indexing.
Value genIndexLoad(OpBuilder &builder, Location loc, Value mem, ValueRange s);

/// Generates a 1-valued attribute of the given type.  This supports
/// all the same types as `getZeroAttr`; however, unlike `getZeroAttr`,
/// for unsupported types we raise `llvm_unreachable` rather than
/// returning a null attribute.
TypedAttr getOneAttr(Builder &builder, Type tp);

/// Generates the comparison `v != 0` where `v` is of numeric type.
/// For floating types, we use the "unordered" comparator (i.e., returns
/// true if `v` is NaN).
Value genIsNonzero(OpBuilder &builder, Location loc, Value v);

/// Computes the shape of destination tensor of a reshape operator. This is only
/// used when operands have dynamic shape. The shape of the destination is
/// stored into dstShape.
void genReshapeDstShape(OpBuilder &builder, Location loc,
                        SmallVectorImpl<Value> &dstShape,
                        ArrayRef<Value> srcShape, ArrayRef<Size> staticDstShape,
                        ArrayRef<ReassociationIndices> reassociation);

/// Reshape coordinates during a reshaping operation.
void reshapeCvs(OpBuilder &builder, Location loc,
                ArrayRef<ReassociationIndices> reassociation,
                ValueRange srcSizes, ValueRange srcCvs, // NOLINT
                ValueRange dstSizes, SmallVectorImpl<Value> &dstCvs);

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
Type getOpaquePointerType(MLIRContext *ctx);
Type getOpaquePointerType(Builder &builder);

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

/// Populates given sizes array from dense tensor or sparse tensor constant.
void sizesFromSrc(OpBuilder &builder, SmallVectorImpl<Value> &sizes,
                  Location loc, Value src);

/// Scans to top of generated loop.
Operation *getTop(Operation *op);

/// Iterate over a sparse constant, generates constantOp for value
/// and coordinates.  E.g.,
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
    OpBuilder &builder, Location loc, SparseElementsAttr attr, AffineMap order,
    function_ref<void(ArrayRef<Value>, Value)> callback);

/// Loads `size`-many values from the memref, which must have rank-1 and
/// size greater-or-equal to `size`.  If the optional `(offsetIdx,offsetVal)`
/// arguments are provided, then the `offsetVal` will be added to the
/// `offsetIdx`-th value after loading.
SmallVector<Value> loadAll(OpBuilder &builder, Location loc, size_t size,
                           Value mem, size_t offsetIdx = 0,
                           Value offsetVal = Value());

/// Stores all the values of `vs` into the memref `mem`, which must have
/// rank-1 and size greater-or-equal to `vs.size()`.  If the optional
/// `(offsetIdx,offsetVal)` arguments are provided, then the `offsetVal`
/// will be added to the `offsetIdx`-th value before storing.
void storeAll(OpBuilder &builder, Location loc, Value mem, ValueRange vs,
              size_t offsetIdx = 0, Value offsetVal = Value());

// Generates code to cast a tensor to a memref.
TypedValue<BaseMemRefType> genToMemref(OpBuilder &builder, Location loc,
                                       Value tensor);

/// Generates code to retrieve the values size for the sparse tensor.
Value genValMemSize(OpBuilder &builder, Location loc, Value tensor);

/// Generates code to retrieve the slice offset for the sparse tensor slice,
/// return a constant if the offset is statically known.
Value createOrFoldSliceOffsetOp(OpBuilder &builder, Location loc, Value tensor,
                                Dimension dim);

/// Generates code to retrieve the slice slice for the sparse tensor slice,
/// return a constant if the offset is statically known.
Value createOrFoldSliceStrideOp(OpBuilder &builder, Location loc, Value tensor,
                                Dimension dim);

/// Generates code that opens a reader and sets the dimension sizes.
Value genReader(OpBuilder &builder, Location loc, SparseTensorType stt,
                Value tensor,
                /*out*/ SmallVectorImpl<Value> &dimSizesValues,
                /*out*/ Value &dimSizesBuffer);

/// Generates code to set up the buffer parameters for a map.
Value genMapBuffers(OpBuilder &builder, Location loc, SparseTensorType stt,
                    ArrayRef<Value> dimSizesValues, Value dimSizesBuffer,
                    /*out*/ SmallVectorImpl<Value> &lvlSizesValues,
                    /*out*/ Value &dim2lvlBuffer,
                    /*out*/ Value &lvl2dimBuffer);

//===----------------------------------------------------------------------===//
// Inlined constant generators.
//
// All these functions are just wrappers to improve code legibility;
// therefore, we mark them as `inline` to avoid introducing any additional
// overhead due to the legibility. Ideally these should move upstream.
//
//===----------------------------------------------------------------------===//

/// Generates a 0-valued constant of the given type.  In addition to
/// the scalar types (`ComplexType`, `FloatType`, `IndexType`,
/// `IntegerType`), this also works for `RankedTensorType` and `VectorType`
/// (for which it generates a constant `DenseElementsAttr` of zeros).
inline Value constantZero(OpBuilder &builder, Location loc, Type tp) {
  if (auto ctp = dyn_cast<ComplexType>(tp)) {
    auto zeroe = builder.getZeroAttr(ctp.getElementType());
    auto zeroa = builder.getArrayAttr({zeroe, zeroe});
    return builder.create<complex::ConstantOp>(loc, tp, zeroa);
  }
  return builder.create<arith::ConstantOp>(loc, tp, builder.getZeroAttr(tp));
}

/// Generates a 1-valued constant of the given type.  This supports all
/// the same types as `constantZero`.
inline Value constantOne(OpBuilder &builder, Location loc, Type tp) {
  if (auto ctp = dyn_cast<ComplexType>(tp)) {
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

/// Generates a constant of the internal type-encoding for position
/// overhead storage.
inline Value constantPosTypeEncoding(OpBuilder &builder, Location loc,
                                     SparseTensorEncodingAttr enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getPosWidth());
}

/// Generates a constant of the internal type-encoding for coordinate
/// overhead storage.
inline Value constantCrdTypeEncoding(OpBuilder &builder, Location loc,
                                     SparseTensorEncodingAttr enc) {
  return constantOverheadTypeEncoding(builder, loc, enc.getCrdWidth());
}

/// Generates a constant of the internal type-encoding for primary storage.
inline Value constantPrimaryTypeEncoding(OpBuilder &builder, Location loc,
                                         Type elemTp) {
  return constantI32(builder, loc,
                     static_cast<uint32_t>(primaryTypeEncoding(elemTp)));
}

/// Generates a constant of the internal dimension level type encoding.
inline Value constantLevelTypeEncoding(OpBuilder &builder, Location loc,
                                       LevelType lt) {
  return constantI64(builder, loc, static_cast<uint64_t>(lt));
}

inline bool isZeroRankedTensorOrScalar(Type type) {
  auto rtp = dyn_cast<RankedTensorType>(type);
  return !rtp || rtp.getRank() == 0;
}

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_CODEGENUTILS_H_
