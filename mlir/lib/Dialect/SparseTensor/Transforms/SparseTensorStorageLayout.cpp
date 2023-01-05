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

static std::optional<LogicalResult>
convertSparseTensorType(RankedTensorType rtp, SmallVectorImpl<Type> &fields) {
  auto enc = getSparseTensorEncoding(rtp);
  if (!enc)
    return std::nullopt;

  foreachFieldAndTypeInSparseTensor(
      rtp,
      [&fields](Type fieldType, unsigned fieldIdx,
                SparseTensorFieldKind /*fieldKind*/, unsigned /*dim*/,
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
  addConversion([&](RankedTensorType rtp, SmallVectorImpl<Type> &fields) {
    return convertSparseTensorType(rtp, fields);
  });

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
                                          RankedTensorType rtp) {
  return builder.create<StorageSpecifierInitOp>(
      loc, StorageSpecifierType::get(getSparseTensorEncoding(rtp)));
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

sparse_tensor::SparseTensorDescriptor::SparseTensorDescriptor(
    OpBuilder &builder, Location loc, Type tp, ValueArrayRef buffers)
    : SparseTensorDescriptorImpl<false>(tp), expandedFields() {
  SparseTensorEncodingAttr enc = getSparseTensorEncoding(tp);
  unsigned rank = enc.getDimLevelType().size();
  unsigned cooStart = getCOOStart(enc);
  if (cooStart < rank) {
    ValueRange beforeFields = buffers.drop_back(3);
    expandedFields.append(beforeFields.begin(), beforeFields.end());
    Value buffer = buffers[buffers.size() - 3];

    Value stride = constantIndex(builder, loc, rank - cooStart);
    SmallVector<Value> buffersArray(buffers.begin(), buffers.end());
    MutSparseTensorDescriptor mutDesc(tp, buffersArray);
    // Calculate subbuffer size as memSizes[idx] / (stride).
    Value subBufferSize = mutDesc.getIdxMemSize(builder, loc, cooStart);
    subBufferSize = builder.create<arith::DivUIOp>(loc, subBufferSize, stride);

    // Create views of the linear idx buffer for the COO indices.
    for (unsigned i = cooStart; i < rank; i++) {
      Value subBuffer = builder.create<memref::SubViewOp>(
          loc, buffer,
          /*offset=*/ValueRange{constantIndex(builder, loc, i - cooStart)},
          /*size=*/ValueRange{subBufferSize},
          /*step=*/ValueRange{stride});
      expandedFields.push_back(subBuffer);
    }
    expandedFields.push_back(buffers[buffers.size() - 2]); // The Values memref.
    expandedFields.push_back(buffers.back());              // The specifier.
    fields = expandedFields;
  } else {
    fields = buffers;
  }

  sanityCheck();
}

//===----------------------------------------------------------------------===//
// Public methods.
//===----------------------------------------------------------------------===//

constexpr uint64_t kDataFieldStartingIdx = 0;

void sparse_tensor::foreachFieldInSparseTensor(
    const SparseTensorEncodingAttr enc,
    llvm::function_ref<bool(unsigned, SparseTensorFieldKind, unsigned,
                            DimLevelType)>
        callback,
    bool isBuffer) {
  assert(enc);

#define RETURN_ON_FALSE(idx, kind, dim, dlt)                                   \
  if (!(callback(idx, kind, dim, dlt)))                                        \
    return;

  unsigned rank = enc.getDimLevelType().size();
  unsigned cooStart = isBuffer ? getCOOStart(enc) : rank;
  static_assert(kDataFieldStartingIdx == 0);
  unsigned fieldIdx = kDataFieldStartingIdx;
  // Per-dimension storage.
  for (unsigned r = 0; r < rank; r++) {
    // Dimension level types apply in order to the reordered dimension.
    // As a result, the compound type can be constructed directly in the given
    // order.
    auto dlt = getDimLevelType(enc, r);
    if (isCompressedDLT(dlt)) {
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::PtrMemRef, r, dlt);
      RETURN_ON_FALSE(fieldIdx++, SparseTensorFieldKind::IdxMemRef, r, dlt);
    } else if (isSingletonDLT(dlt)) {
      if (r < cooStart)
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

void sparse_tensor::foreachFieldAndTypeInSparseTensor(
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
      },
      /*isBuffer=*/true);
}

unsigned sparse_tensor::getNumFieldsFromEncoding(SparseTensorEncodingAttr enc,
                                                 bool isBuffer) {
  unsigned numFields = 0;
  foreachFieldInSparseTensor(
      enc,
      [&numFields](unsigned, SparseTensorFieldKind, unsigned,
                   DimLevelType) -> bool {
        numFields++;
        return true;
      },
      isBuffer);
  return numFields;
}

unsigned
sparse_tensor::getNumDataFieldsFromEncoding(SparseTensorEncodingAttr enc) {
  unsigned numFields = 0; // one value memref
  foreachFieldInSparseTensor(
      enc,
      [&numFields](unsigned fidx, SparseTensorFieldKind, unsigned,
                   DimLevelType) -> bool {
        if (fidx >= kDataFieldStartingIdx)
          numFields++;
        return true;
      },
      /*isBuffer=*/true);
  numFields -= 1; // the last field is MetaData field
  assert(numFields == getNumFieldsFromEncoding(enc, /*isBuffer=*/true) -
                          kDataFieldStartingIdx - 1);
  return numFields;
}
