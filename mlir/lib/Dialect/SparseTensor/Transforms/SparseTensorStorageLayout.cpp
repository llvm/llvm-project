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

/// Constructs a nullable `LevelAttr` from the `std::optional<Level>`.
static IntegerAttr optionalLevelAttr(MLIRContext *ctx,
                                     std::optional<Level> lvl) {
  return lvl ? IntegerAttr::get(IndexType::get(ctx), lvl.value())
             : IntegerAttr();
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
                                               std::optional<Level> lvl) {
  return builder.create<GetStorageSpecifierOp>(
      loc, specifier, kind, optionalLevelAttr(specifier.getContext(), lvl));
}

void SparseTensorSpecifier::setSpecifierField(OpBuilder &builder, Location loc,
                                              Value v,
                                              StorageSpecifierKind kind,
                                              std::optional<Level> lvl) {
  // TODO: make `v` have type `TypedValue<IndexType>` instead.
  assert(v.getType().isIndex());
  specifier = builder.create<SetStorageSpecifierOp>(
      loc, specifier, kind, optionalLevelAttr(specifier.getContext(), lvl), v);
}

//===----------------------------------------------------------------------===//
// SparseTensorDescriptor methods.
//===----------------------------------------------------------------------===//

Value sparse_tensor::SparseTensorDescriptor::getCrdMemRefOrView(
    OpBuilder &builder, Location loc, Level lvl) const {
  const Level cooStart = getCOOStart(rType.getEncoding());
  if (lvl < cooStart)
    return getMemRefField(SparseTensorFieldKind::CrdMemRef, lvl);

  Value stride = constantIndex(builder, loc, rType.getLvlRank() - cooStart);
  Value size = getCrdMemSize(builder, loc, cooStart);
  size = builder.create<arith::DivUIOp>(loc, size, stride);
  return builder.create<memref::SubViewOp>(
      loc, getMemRefField(SparseTensorFieldKind::CrdMemRef, cooStart),
      /*offset=*/ValueRange{constantIndex(builder, loc, lvl - cooStart)},
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

#define RETURN_ON_FALSE(fidx, kind, dim, dlt)                                  \
  if (!(callback(fidx, kind, dim, dlt)))                                       \
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
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::PosMemRef, l, dlt);
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::CrdMemRef, l, dlt);
    } else if (isSingletonDLT(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::CrdMemRef, l, dlt);
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
  assert(stt.hasEncoding());
  // Construct the basic types.
  const Type crdType = stt.getCrdType();
  const Type posType = stt.getPosType();
  const Type eltType = stt.getElementType();

  const Type metaDataType = StorageSpecifierType::get(stt.getEncoding());
  // memref<? x pos>  positions
  const Type posMemType = MemRefType::get({ShapedType::kDynamic}, posType);
  // memref<? x crd>  coordinates
  const Type crdMemType = MemRefType::get({ShapedType::kDynamic}, crdType);
  // memref<? x eltType> values
  const Type valMemType = MemRefType::get({ShapedType::kDynamic}, eltType);

  foreachFieldInSparseTensor(
      stt.getEncoding(),
      [metaDataType, posMemType, crdMemType, valMemType,
       callback](FieldIndex fieldIdx, SparseTensorFieldKind fieldKind,
                 Level lvl, DimLevelType dlt) -> bool {
        switch (fieldKind) {
        case SparseTensorFieldKind::StorageSpec:
          return callback(metaDataType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::PosMemRef:
          return callback(posMemType, fieldIdx, fieldKind, lvl, dlt);
        case SparseTensorFieldKind::CrdMemRef:
          return callback(crdMemType, fieldIdx, fieldKind, lvl, dlt);
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
