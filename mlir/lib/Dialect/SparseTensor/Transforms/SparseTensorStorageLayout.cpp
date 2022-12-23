//===- SparseTensorStorageLayout.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SparseTensorStorageLayout.h"
#include "CodegenUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace sparse_tensor;

static Value createIndexCast(OpBuilder &builder, Location loc, Value value,
                             Type to) {
  if (value.getType() != to)
    return builder.create<arith::IndexCastOp>(loc, to, value);
  return value;
}

static IntegerAttr fromOptionalInt(MLIRContext *ctx, Optional<unsigned> dim) {
  if (!dim)
    return nullptr;
  return IntegerAttr::get(IndexType::get(ctx), dim.value());
}

unsigned
builder::StorageLayout::getMemRefFieldIndex(SparseTensorFieldKind kind,
                                            Optional<unsigned> dim) const {
  unsigned fieldIdx = -1u;
  foreachFieldInSparseTensor(
      enc,
      [dim, kind, &fieldIdx](unsigned fIdx, SparseTensorFieldKind fKind,
                             unsigned fDim, DimLevelType dlt) -> bool {
        if ((dim && fDim == dim.value() && kind == fKind) ||
            (kind == fKind && fKind == SparseTensorFieldKind::ValMemRef)) {
          fieldIdx = fIdx;
          // Returns false to break the iteration.
          return false;
        }
        return true;
      });
  assert(fieldIdx != -1u);
  return fieldIdx;
}

unsigned
builder::StorageLayout::getMemRefFieldIndex(StorageSpecifierKind kind,
                                            Optional<unsigned> dim) const {
  return getMemRefFieldIndex(toFieldKind(kind), dim);
}

Value builder::SparseTensorSpecifier::getInitValue(OpBuilder &builder,
                                                   Location loc,
                                                   RankedTensorType rtp) {
  return builder.create<StorageSpecifierInitOp>(
      loc, StorageSpecifierType::get(getSparseTensorEncoding(rtp)));
}

Value builder::SparseTensorSpecifier::getSpecifierField(
    OpBuilder &builder, Location loc, StorageSpecifierKind kind,
    Optional<unsigned> dim) {
  return createIndexCast(builder, loc,
                         builder.create<GetStorageSpecifierOp>(
                             loc, getFieldType(kind, dim), specifier, kind,
                             fromOptionalInt(specifier.getContext(), dim)),
                         builder.getIndexType());
}

void builder::SparseTensorSpecifier::setSpecifierField(
    OpBuilder &builder, Location loc, Value v, StorageSpecifierKind kind,
    Optional<unsigned> dim) {
  specifier = builder.create<SetStorageSpecifierOp>(
      loc, specifier, kind, fromOptionalInt(specifier.getContext(), dim),
      createIndexCast(builder, loc, v, getFieldType(kind, dim)));
}

constexpr uint64_t kDataFieldStartingIdx = 0;

void sparse_tensor::builder::foreachFieldInSparseTensor(
    const SparseTensorEncodingAttr enc,
    llvm::function_ref<bool(unsigned, SparseTensorFieldKind, unsigned,
                            DimLevelType)>
        callback) {
  assert(enc);

#define RETURN_ON_FALSE(idx, kind, dim, dlt)                                   \
  if (!(callback(idx, kind, dim, dlt)))                                        \
    return;

  static_assert(kDataFieldStartingIdx == 0);
  unsigned fieldIdx = kDataFieldStartingIdx;
  // Per-dimension storage.
  for (unsigned r = 0, rank = enc.getDimLevelType().size(); r < rank; r++) {
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order.
    auto dlt = getDimLevelType(enc, r);
    if (isCompressedDLT(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::PtrMemRef, r, dlt);
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::IdxMemRef, r, dlt);
    } else if (isSingletonDLT(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::IdxMemRef, r, dlt);
    } else {
      assert(isDenseDLT(dlt)); // no fields
    }
  }
  // The values array.
  RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::ValMemRef, -1u,
                  DimLevelType::Undef);

  // Put metadata at the end.
  RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::StorageSpec, -1u,
                  DimLevelType::Undef);

#undef RETURN_ON_FALSE
}

void sparse_tensor::builder::foreachFieldAndTypeInSparseTensor(
    RankedTensorType rType,
    llvm::function_ref<bool(Type, unsigned, SparseTensorFieldKind, unsigned,
                            DimLevelType)>
        callback) {
  auto enc = getSparseTensorEncoding(rType);
  assert(enc);
  // Construct the basic types.
  Type idxType = enc.getIndexType();
  Type ptrType = enc.getPointerType();
  Type eltType = rType.getElementType();

  Type metaDataType = StorageSpecifierType::get(enc);
  // memref<? x ptr>  pointers
  Type ptrMemType = MemRefType::get({ShapedType::kDynamic}, ptrType);
  // memref<? x idx>  indices
  Type idxMemType = MemRefType::get({ShapedType::kDynamic}, idxType);
  // memref<? x eltType> values
  Type valMemType = MemRefType::get({ShapedType::kDynamic}, eltType);

  foreachFieldInSparseTensor(
      enc,
      [metaDataType, ptrMemType, idxMemType, valMemType,
       callback](unsigned fieldIdx, SparseTensorFieldKind fieldKind,
                 unsigned dim, DimLevelType dlt) -> bool {
        switch (fieldKind) {
        case SparseTensorFieldKind::StorageSpec:
          return callback(metaDataType, fieldIdx, fieldKind, dim, dlt);
        case SparseTensorFieldKind::PtrMemRef:
          return callback(ptrMemType, fieldIdx, fieldKind, dim, dlt);
        case SparseTensorFieldKind::IdxMemRef:
          return callback(idxMemType, fieldIdx, fieldKind, dim, dlt);
        case SparseTensorFieldKind::ValMemRef:
          return callback(valMemType, fieldIdx, fieldKind, dim, dlt);
        };
        llvm_unreachable("unrecognized field kind");
      });
}

unsigned
sparse_tensor::builder::getNumFieldsFromEncoding(SparseTensorEncodingAttr enc) {
  unsigned numFields = 0;
  foreachFieldInSparseTensor(enc,
                             [&numFields](unsigned, SparseTensorFieldKind,
                                          unsigned, DimLevelType) -> bool {
                               numFields++;
                               return true;
                             });
  return numFields;
}

unsigned sparse_tensor::builder::getNumDataFieldsFromEncoding(
    SparseTensorEncodingAttr enc) {
  unsigned numFields = 0; // one value memref
  foreachFieldInSparseTensor(enc,
                             [&numFields](unsigned fidx, SparseTensorFieldKind,
                                          unsigned, DimLevelType) -> bool {
                               if (fidx >= kDataFieldStartingIdx)
                                 numFields++;
                               return true;
                             });
  numFields -= 1; // the last field is MetaData field
  assert(numFields ==
         builder::getNumFieldsFromEncoding(enc) - kDataFieldStartingIdx - 1);
  return numFields;
}
