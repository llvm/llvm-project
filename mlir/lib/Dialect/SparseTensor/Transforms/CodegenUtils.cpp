//===- CodegenUtils.cpp - Utilities for generating MLIR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

//===----------------------------------------------------------------------===//
// ExecutionEngine/SparseTensorUtils helper functions.
//===----------------------------------------------------------------------===//

OverheadType mlir::sparse_tensor::overheadTypeEncoding(unsigned width) {
  switch (width) {
  case 64:
    return OverheadType::kU64;
  case 32:
    return OverheadType::kU32;
  case 16:
    return OverheadType::kU16;
  case 8:
    return OverheadType::kU8;
  case 0:
    return OverheadType::kIndex;
  }
  llvm_unreachable("Unsupported overhead bitwidth");
}

OverheadType mlir::sparse_tensor::overheadTypeEncoding(Type tp) {
  if (tp.isIndex())
    return OverheadType::kIndex;
  if (auto intTp = tp.dyn_cast<IntegerType>())
    return overheadTypeEncoding(intTp.getWidth());
  llvm_unreachable("Unknown overhead type");
}

Type mlir::sparse_tensor::getOverheadType(Builder &builder, OverheadType ot) {
  switch (ot) {
  case OverheadType::kIndex:
    return builder.getIndexType();
  case OverheadType::kU64:
    return builder.getIntegerType(64);
  case OverheadType::kU32:
    return builder.getIntegerType(32);
  case OverheadType::kU16:
    return builder.getIntegerType(16);
  case OverheadType::kU8:
    return builder.getIntegerType(8);
  }
  llvm_unreachable("Unknown OverheadType");
}

OverheadType mlir::sparse_tensor::pointerOverheadTypeEncoding(
    const SparseTensorEncodingAttr &enc) {
  return overheadTypeEncoding(enc.getPointerBitWidth());
}

OverheadType mlir::sparse_tensor::indexOverheadTypeEncoding(
    const SparseTensorEncodingAttr &enc) {
  return overheadTypeEncoding(enc.getIndexBitWidth());
}

Type mlir::sparse_tensor::getPointerOverheadType(
    Builder &builder, const SparseTensorEncodingAttr &enc) {
  return getOverheadType(builder, pointerOverheadTypeEncoding(enc));
}

Type mlir::sparse_tensor::getIndexOverheadType(
    Builder &builder, const SparseTensorEncodingAttr &enc) {
  return getOverheadType(builder, indexOverheadTypeEncoding(enc));
}

// TODO: Adjust the naming convention for the constructors of
// `OverheadType` so we can use the `MLIR_SPARSETENSOR_FOREVERY_O` x-macro
// here instead of `MLIR_SPARSETENSOR_FOREVERY_FIXED_O`; to further reduce
// the possibility of typo bugs or things getting out of sync.
StringRef mlir::sparse_tensor::overheadTypeFunctionSuffix(OverheadType ot) {
  switch (ot) {
  case OverheadType::kIndex:
    return "0";
#define CASE(ONAME, O)                                                         \
  case OverheadType::kU##ONAME:                                                \
    return #ONAME;
    MLIR_SPARSETENSOR_FOREVERY_FIXED_O(CASE)
#undef CASE
  }
  llvm_unreachable("Unknown OverheadType");
}

StringRef mlir::sparse_tensor::overheadTypeFunctionSuffix(Type tp) {
  return overheadTypeFunctionSuffix(overheadTypeEncoding(tp));
}

PrimaryType mlir::sparse_tensor::primaryTypeEncoding(Type elemTp) {
  if (elemTp.isF64())
    return PrimaryType::kF64;
  if (elemTp.isF32())
    return PrimaryType::kF32;
  if (elemTp.isF16())
    return PrimaryType::kF16;
  if (elemTp.isBF16())
    return PrimaryType::kBF16;
  if (elemTp.isInteger(64))
    return PrimaryType::kI64;
  if (elemTp.isInteger(32))
    return PrimaryType::kI32;
  if (elemTp.isInteger(16))
    return PrimaryType::kI16;
  if (elemTp.isInteger(8))
    return PrimaryType::kI8;
  if (auto complexTp = elemTp.dyn_cast<ComplexType>()) {
    auto complexEltTp = complexTp.getElementType();
    if (complexEltTp.isF64())
      return PrimaryType::kC64;
    if (complexEltTp.isF32())
      return PrimaryType::kC32;
  }
  llvm_unreachable("Unknown primary type");
}

StringRef mlir::sparse_tensor::primaryTypeFunctionSuffix(PrimaryType pt) {
  switch (pt) {
#define CASE(VNAME, V)                                                         \
  case PrimaryType::k##VNAME:                                                  \
    return #VNAME;
    MLIR_SPARSETENSOR_FOREVERY_V(CASE)
#undef CASE
  }
  llvm_unreachable("Unknown PrimaryType");
}

StringRef mlir::sparse_tensor::primaryTypeFunctionSuffix(Type elemTp) {
  return primaryTypeFunctionSuffix(primaryTypeEncoding(elemTp));
}

DimLevelType mlir::sparse_tensor::dimLevelTypeEncoding(
    SparseTensorEncodingAttr::DimLevelType dlt) {
  switch (dlt) {
  case SparseTensorEncodingAttr::DimLevelType::Dense:
    return DimLevelType::kDense;
  case SparseTensorEncodingAttr::DimLevelType::Compressed:
    return DimLevelType::kCompressed;
  case SparseTensorEncodingAttr::DimLevelType::CompressedNu:
    return DimLevelType::kCompressedNu;
  case SparseTensorEncodingAttr::DimLevelType::CompressedNo:
    return DimLevelType::kCompressedNo;
  case SparseTensorEncodingAttr::DimLevelType::CompressedNuNo:
    return DimLevelType::kCompressedNuNo;
  case SparseTensorEncodingAttr::DimLevelType::Singleton:
    return DimLevelType::kSingleton;
  case SparseTensorEncodingAttr::DimLevelType::SingletonNu:
    return DimLevelType::kSingletonNu;
  case SparseTensorEncodingAttr::DimLevelType::SingletonNo:
    return DimLevelType::kSingletonNo;
  case SparseTensorEncodingAttr::DimLevelType::SingletonNuNo:
    return DimLevelType::kSingletonNuNo;
  }
  llvm_unreachable("Unknown SparseTensorEncodingAttr::DimLevelType");
}

//===----------------------------------------------------------------------===//
// Misc code generators.
//===----------------------------------------------------------------------===//

mlir::Attribute mlir::sparse_tensor::getOneAttr(Builder &builder, Type tp) {
  if (tp.isa<FloatType>())
    return builder.getFloatAttr(tp, 1.0);
  if (tp.isa<IndexType>())
    return builder.getIndexAttr(1);
  if (auto intTp = tp.dyn_cast<IntegerType>())
    return builder.getIntegerAttr(tp, APInt(intTp.getWidth(), 1));
  if (tp.isa<RankedTensorType, VectorType>()) {
    auto shapedTp = tp.cast<ShapedType>();
    if (auto one = getOneAttr(builder, shapedTp.getElementType()))
      return DenseElementsAttr::get(shapedTp, one);
  }
  llvm_unreachable("Unsupported attribute type");
}

Value mlir::sparse_tensor::genIsNonzero(OpBuilder &builder, mlir::Location loc,
                                        Value v) {
  Type tp = v.getType();
  Value zero = constantZero(builder, loc, tp);
  if (tp.isa<FloatType>())
    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, v,
                                         zero);
  if (tp.isIntOrIndex())
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, v,
                                         zero);
  if (tp.dyn_cast<ComplexType>())
    return builder.create<complex::NotEqualOp>(loc, v, zero);
  llvm_unreachable("Non-numeric type");
}

void mlir::sparse_tensor::translateIndicesArray(
    OpBuilder &builder, Location loc,
    ArrayRef<ReassociationIndices> reassociation, ValueRange srcIndices,
    ArrayRef<Value> srcShape, ArrayRef<Value> dstShape,
    SmallVectorImpl<Value> &dstIndices) {
  unsigned i = 0;
  unsigned start = 0;
  unsigned dstRank = dstShape.size();
  unsigned srcRank = srcShape.size();
  assert(srcRank == srcIndices.size());
  bool isCollapse = srcRank > dstRank;
  ArrayRef<Value> shape = isCollapse ? srcShape : dstShape;
  // Iterate over reassociation map.
  for (const auto &map : llvm::enumerate(reassociation)) {
    // Prepare strides information in dimension slice.
    Value linear = constantIndex(builder, loc, 1);
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::MulIOp>(loc, linear, shape[j]);
    }
    // Start expansion.
    Value val;
    if (!isCollapse)
      val = srcIndices[i];
    // Iterate over dimension slice.
    for (unsigned j = start, end = start + map.value().size(); j < end; j++) {
      linear = builder.create<arith::DivUIOp>(loc, linear, shape[j]);
      if (isCollapse) {
        Value old = srcIndices[j];
        Value mul = builder.create<arith::MulIOp>(loc, old, linear);
        val = val ? builder.create<arith::AddIOp>(loc, val, mul) : mul;
      } else {
        Value old = val;
        val = builder.create<arith::DivUIOp>(loc, val, linear);
        assert(dstIndices.size() == j);
        dstIndices.push_back(val);
        val = builder.create<arith::RemUIOp>(loc, old, linear);
      }
    }
    // Finalize collapse.
    if (isCollapse) {
      assert(dstIndices.size() == i);
      dstIndices.push_back(val);
    }
    start += map.value().size();
    i++;
  }
  assert(dstIndices.size() == dstRank);
}
