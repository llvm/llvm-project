//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/Transforms/RuntimeOpVerification.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

using namespace mlir;

namespace mlir {
namespace tensor {
namespace {
/// Generate a runtime check for lb <= value < ub.
Value generateInBoundsCheck(OpBuilder &builder, Location loc, Value value,
                            Value lb, Value ub) {
  Value inBounds1 = builder.createOrFold<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, value, lb);
  Value inBounds2 = builder.createOrFold<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, value, ub);
  Value inBounds =
      builder.createOrFold<arith::AndIOp>(loc, inBounds1, inBounds2);
  return inBounds;
}

struct CastOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<CastOpInterface,
                                                         CastOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto castOp = cast<CastOp>(op);
    auto srcType = cast<TensorType>(castOp.getSource().getType());

    // Nothing to check if the result is an unranked tensor.
    auto resultType = dyn_cast<RankedTensorType>(castOp.getType());
    if (!resultType)
      return;

    if (isa<UnrankedTensorType>(srcType)) {
      // Check rank.
      Value srcRank = builder.create<RankOp>(loc, castOp.getSource());
      Value resultRank =
          builder.create<arith::ConstantIndexOp>(loc, resultType.getRank());
      Value isSameRank = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, srcRank, resultRank);
      builder.create<cf::AssertOp>(
          loc, isSameRank,
          RuntimeVerifiableOpInterface::generateErrorMessage(op,
                                                             "rank mismatch"));
    }

    // Check dimension sizes.
    for (const auto &it : llvm::enumerate(resultType.getShape())) {
      // Static dim size -> static/dynamic dim size does not need verification.
      if (auto rankedSrcType = dyn_cast<RankedTensorType>(srcType))
        if (!rankedSrcType.isDynamicDim(it.index()))
          continue;

      // Static/dynamic dim size -> dynamic dim size does not need verification.
      if (resultType.isDynamicDim(it.index()))
        continue;

      Value srcDimSz =
          builder.create<DimOp>(loc, castOp.getSource(), it.index());
      Value resultDimSz =
          builder.create<arith::ConstantIndexOp>(loc, it.value());
      Value isSameSz = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, srcDimSz, resultDimSz);
      builder.create<cf::AssertOp>(
          loc, isSameSz,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "size mismatch of dim " + std::to_string(it.index())));
    }
  }
};

struct DimOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<DimOpInterface,
                                                         DimOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto dimOp = cast<DimOp>(op);
    Value rank = builder.create<RankOp>(loc, dimOp.getSource());
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    builder.create<cf::AssertOp>(
        loc, generateInBoundsCheck(builder, loc, dimOp.getIndex(), zero, rank),
        RuntimeVerifiableOpInterface::generateErrorMessage(
            op, "index is out of bounds"));
  }
};

/// Verifies that the indices on extract/insert ops are in-bounds of the
/// tensor's index space: 0 <= index#i < dim#i
template <typename OpTy>
struct ExtractInsertOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          ExtractInsertOpInterface<OpTy>, OpTy> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto extractInsertOp = cast<OpTy>(op);

    Value tensor;
    if constexpr (std::is_same_v<OpTy, ExtractOp>) {
      tensor = extractInsertOp.getTensor();
    } else if constexpr (std::is_same_v<OpTy, InsertOp>) {
      tensor = extractInsertOp.getDest();
    } else {
      llvm_unreachable("invalid op");
    }
    auto tensorType = cast<RankedTensorType>(tensor.getType());
    auto rank = tensorType.getRank();
    if (rank == 0) {
      // Nothing to check for 0-d tensors.
      return;
    }

    auto indices = extractInsertOp.getIndices();
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value assertCond;
    for (auto i : llvm::seq<int64_t>(0, rank)) {
      Value dimOp = builder.createOrFold<tensor::DimOp>(loc, tensor, i);
      Value inBounds =
          generateInBoundsCheck(builder, loc, indices[i], zero, dimOp);
      assertCond =
          i > 0 ? builder.createOrFold<arith::AndIOp>(loc, assertCond, inBounds)
                : inBounds;
    }
    builder.create<cf::AssertOp>(
        loc, assertCond,
        RuntimeVerifiableOpInterface::generateErrorMessage(
            op, "out-of-bounds access"));
  }
};

struct ExtractSliceOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          ExtractSliceOpInterface, ExtractSliceOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto extractSliceOp = cast<ExtractSliceOp>(op);
    RankedTensorType sourceType = extractSliceOp.getSource().getType();

    // For each dimension, assert that:
    // 0 <= offset < dim_size
    // 0 <= offset + (size - 1) * stride < dim_size
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (int64_t i = 0, e = sourceType.getRank(); i < e; ++i) {
      Value offset = getValueOrCreateConstantIndexOp(
          builder, loc, extractSliceOp.getMixedOffsets()[i]);
      Value size = getValueOrCreateConstantIndexOp(
          builder, loc, extractSliceOp.getMixedSizes()[i]);
      Value stride = getValueOrCreateConstantIndexOp(
          builder, loc, extractSliceOp.getMixedStrides()[i]);

      // Verify that offset is in-bounds.
      Value dimSize = builder.createOrFold<tensor::DimOp>(
          loc, extractSliceOp.getSource(), i);
      Value offsetInBounds =
          generateInBoundsCheck(builder, loc, offset, zero, dimSize);
      builder.create<cf::AssertOp>(
          loc, offsetInBounds,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "offset " + std::to_string(i) + " is out-of-bounds"));

      // Verify that slice does not run out-of-bounds.
      Value sizeMinusOne = builder.create<arith::SubIOp>(loc, size, one);
      Value sizeMinusOneTimesStride =
          builder.create<arith::MulIOp>(loc, sizeMinusOne, stride);
      Value lastPos =
          builder.create<arith::AddIOp>(loc, offset, sizeMinusOneTimesStride);
      Value lastPosInBounds =
          generateInBoundsCheck(builder, loc, lastPos, zero, dimSize);
      builder.create<cf::AssertOp>(
          loc, lastPosInBounds,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "extract_slice runs out-of-bounds along dimension " +
                      std::to_string(i)));
    }
  }
};
} // namespace
} // namespace tensor
} // namespace mlir

void mlir::tensor::registerRuntimeVerifiableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    CastOp::attachInterface<CastOpInterface>(*ctx);
    DimOp::attachInterface<DimOpInterface>(*ctx);
    ExtractOp::attachInterface<ExtractInsertOpInterface<ExtractOp>>(*ctx);
    ExtractSliceOp::attachInterface<ExtractSliceOpInterface>(*ctx);
    InsertOp::attachInterface<ExtractInsertOpInterface<InsertOp>>(*ctx);

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<arith::ArithDialect, cf::ControlFlowDialect>();
  });
}
