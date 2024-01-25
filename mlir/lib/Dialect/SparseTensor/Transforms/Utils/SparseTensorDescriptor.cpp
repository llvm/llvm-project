//===- SparseTensorDescriptor.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparseTensorDescriptor.h"
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
                LevelType /*lt*/) -> bool {
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
    // Sparsifier knows how to cancel out these casts.
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
  const Level cooStart = rType.getCOOStart();
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
