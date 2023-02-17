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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace sparse_tensor;

//===----------------------------------------------------------------------===//
// Private helper methods.
//===----------------------------------------------------------------------===//

static Value createIndexCast(OpBuilder &builder, Location loc, Value value,
                             Type to) {
  if (value.getType() != to)
    return builder.create<arith::IndexCastOp>(loc, to, value);
  return value;
}

static IntegerAttr fromOptionalInt(MLIRContext *ctx,
                                   std::optional<unsigned> dim) {
  if (!dim)
    return nullptr;
  return IntegerAttr::get(IndexType::get(ctx), dim.value());
}

// This is only ever called from `SparseTensorTypeToBufferConverter`,
// which is why the first argument is `RankedTensorType` rather than
// `SparseTensorType`.
static std::optional<LogicalResult>
convertSparseTensorType(RankedTensorType rtp, SmallVectorImpl<Type> &fields) {
  const SparseTensorType stt(rtp);
  if (!stt.hasEncoding())
    return std::nullopt;

  foreachFieldAndTypeInSparseTensor(
      stt,
      [&fields](Type fieldType, FieldIndex fieldIdx,
                SparseTensorFieldKind /*fieldKind*/, Level /*lvl*/,
                DimLevelType /*dlt*/) -> bool {
        assert(fieldIdx == fields.size());
        fields.push_back(fieldType);
        return true;
      });
  return success();
}

//===----------------------------------------------------------------------===//
// The sparse tensor type converter (defined in Passes.h).
//===----------------------------------------------------------------------===//

SparseTensorTypeToBufferConverter::SparseTensorTypeToBufferConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertSparseTensorType);

  // Required by scf.for 1:N type conversion.
  addSourceMaterialization([](OpBuilder &builder, RankedTensorType tp,
                              ValueRange inputs,
                              Location loc) -> std::optional<Value> {
    if (!getSparseTensorEncoding(tp))
      // Not a sparse tensor.
      return std::nullopt;
    // Sparse compiler knows how to cancel out these casts.
    return genTuple(builder, loc, tp, inputs);
  });
}

//===----------------------------------------------------------------------===//
// StorageTensorSpecifier methods.
//===----------------------------------------------------------------------===//

Value SparseTensorSpecifier::getInitValue(OpBuilder &builder, Location loc,
                                          SparseTensorType stt) {
  return builder.create<StorageSpecifierInitOp>(
      loc, StorageSpecifierType::get(stt.getEncoding()));
}

Value SparseTensorSpecifier::getSpecifierField(OpBuilder &builder, Location loc,
                                               StorageSpecifierKind kind,
                                               std::optional<unsigned> dim) {
  return createIndexCast(builder, loc,
                         builder.create<GetStorageSpecifierOp>(
                             loc, getFieldType(kind, dim), specifier, kind,
                             fromOptionalInt(specifier.getContext(), dim)),
                         builder.getIndexType());
}

void SparseTensorSpecifier::setSpecifierField(OpBuilder &builder, Location loc,
                                              Value v,
                                              StorageSpecifierKind kind,
                                              std::optional<unsigned> dim) {
  specifier = builder.create<SetStorageSpecifierOp>(
      loc, specifier, kind, fromOptionalInt(specifier.getContext(), dim),
      createIndexCast(builder, loc, v, getFieldType(kind, dim)));
}

//===----------------------------------------------------------------------===//
// SparseTensorDescriptor methods.
//===----------------------------------------------------------------------===//

Value sparse_tensor::SparseTensorDescriptor::getIdxMemRefOrView(
    OpBuilder &builder, Location loc, Level idxLvl) const {
  const Level cooStart = getCOOStart(rType.getEncoding());
  if (idxLvl < cooStart)
    return getMemRefField(SparseTensorFieldKind::IdxMemRef, idxLvl);

  Value stride = constantIndex(builder, loc, rType.getLvlRank() - cooStart);
  Value size = getIdxMemSize(builder, loc, cooStart);
  size = builder.create<arith::DivUIOp>(loc, size, stride);
  return builder.create<memref::SubViewOp>(
      loc, getMemRefField(SparseTensorFieldKind::IdxMemRef, cooStart),
      /*offset=*/ValueRange{constantIndex(builder, loc, idxLvl - cooStart)},
      /*size=*/ValueRange{size},
      /*step=*/ValueRange{stride});
}

//===----------------------------------------------------------------------===//
// Public methods.
//===----------------------------------------------------------------------===//

constexpr FieldIndex kDataFieldStartingIdx = 0;

void sparse_tensor::foreachFieldInSparseTensor(
    const SparseTensorEncodingAttr enc,
    llvm::function_ref<bool(FieldIndex, SparseTensorFieldKind, Level,
                            DimLevelType)>
        callback) {
  assert(enc);

#define RETURN_ON_FALSE(idx, kind, dim, dlt)                                   \
  if (!(callback(idx, kind, dim, dlt)))                                        \
    return;

  const auto lvlTypes = enc.getDimLevelType();
  const Level lvlRank = enc.getLvlRank();
  const Level cooStart = getCOOStart(enc);
  const Level end = cooStart == lvlRank ? cooStart : cooStart + 1;
  FieldIndex fieldIdx = kDataFieldStartingIdx;
  // Per-dimension storage.
  for (Level l = 0; l < end; l++) {
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order.
    const auto dlt = lvlTypes[l];
    if (isCompressedDLT(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::PtrMemRef, l, dlt);
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::IdxMemRef, l, dlt);
    } else if (isSingletonDLT(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::IdxMemRef, l, dlt);
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

void sparse_tensor::foreachFieldAndTypeInSparseTensor(
    SparseTensorType stt,
    llvm::function_ref<bool(Type, FieldIndex, SparseTensorFieldKind, Level,
                            DimLevelType)>
        callback) {
  const auto enc = stt.getEncoding();
  assert(enc);
  // Construct the basic types.
  Type idxType = enc.getIndexType();
  Type ptrType = enc.getPointerType();
  Type eltType = stt.getElementType();

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
       callback](FieldIndex fieldIdx, SparseTensorFieldKind fieldKind,
                 Level lvl, DimLevelType dlt) -> bool {
        switch (fieldKind) {
        case SparseTensorFieldKind::StorageSpec:
          return callback(metaDataType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::PtrMemRef:
          return callback(ptrMemType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::IdxMemRef:
          return callback(idxMemType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::ValMemRef:
          return callback(valMemType, fieldIdx, fieldKind, lvl, dlt);
        };
        llvm_unreachable("unrecognized field kind");
      });
}

unsigned sparse_tensor::getNumFieldsFromEncoding(SparseTensorEncodingAttr enc) {
  unsigned numFields = 0;
  foreachFieldInSparseTensor(enc,
                             [&numFields](FieldIndex, SparseTensorFieldKind,
                                          Level, DimLevelType) -> bool {
                               numFields++;
                               return true;
                             });
  return numFields;
}

unsigned
sparse_tensor::getNumDataFieldsFromEncoding(SparseTensorEncodingAttr enc) {
  unsigned numFields = 0; // one value memref
  foreachFieldInSparseTensor(enc,
                             [&numFields](FieldIndex fidx,
                                          SparseTensorFieldKind, Level,
                                          DimLevelType) -> bool {
                               if (fidx >= kDataFieldStartingIdx)
                                 numFields++;
                               return true;
                             });
  numFields -= 1; // the last field is MetaData field
  assert(numFields ==
         getNumFieldsFromEncoding(enc) - kDataFieldStartingIdx - 1);
  return numFields;
}
