//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

using namespace mlir;

namespace mlir {
namespace memref {
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

struct AssumeAlignmentOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          AssumeAlignmentOpInterface, AssumeAlignmentOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto assumeOp = cast<AssumeAlignmentOp>(op);
    Value ptr = builder.create<ExtractAlignedPointerAsIndexOp>(
        loc, assumeOp.getMemref());
    Value rest = builder.create<arith::RemUIOp>(
        loc, ptr,
        builder.create<arith::ConstantIndexOp>(loc, assumeOp.getAlignment()));
    Value isAligned = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rest,
        builder.create<arith::ConstantIndexOp>(loc, 0));
    builder.create<cf::AssertOp>(
        loc, isAligned,
        RuntimeVerifiableOpInterface::generateErrorMessage(
            op, "memref is not aligned to " +
                    std::to_string(assumeOp.getAlignment())));
  }
};

struct CastOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<CastOpInterface,
                                                         CastOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto castOp = cast<CastOp>(op);
    auto srcType = cast<BaseMemRefType>(castOp.getSource().getType());

    // Nothing to check if the result is an unranked memref.
    auto resultType = dyn_cast<MemRefType>(castOp.getType());
    if (!resultType)
      return;

    if (isa<UnrankedMemRefType>(srcType)) {
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

    // Get source offset and strides. We do not have an op to get offsets and
    // strides from unranked memrefs, so cast the source to a type with fully
    // dynamic layout, from which we can then extract the offset and strides.
    // (Rank was already verified.)
    int64_t dynamicOffset = ShapedType::kDynamic;
    SmallVector<int64_t> dynamicShape(resultType.getRank(),
                                      ShapedType::kDynamic);
    auto stridedLayout = StridedLayoutAttr::get(builder.getContext(),
                                                dynamicOffset, dynamicShape);
    auto dynStridesType =
        MemRefType::get(dynamicShape, resultType.getElementType(),
                        stridedLayout, resultType.getMemorySpace());
    Value helperCast =
        builder.create<CastOp>(loc, dynStridesType, castOp.getSource());
    auto metadataOp = builder.create<ExtractStridedMetadataOp>(loc, helperCast);

    // Check dimension sizes.
    for (const auto &it : llvm::enumerate(resultType.getShape())) {
      // Static dim size -> static/dynamic dim size does not need verification.
      if (auto rankedSrcType = dyn_cast<MemRefType>(srcType))
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

    // Get result offset and strides.
    int64_t resultOffset;
    SmallVector<int64_t> resultStrides;
    if (failed(resultType.getStridesAndOffset(resultStrides, resultOffset)))
      return;

    // Check offset.
    if (resultOffset != ShapedType::kDynamic) {
      // Static/dynamic offset -> dynamic offset does not need verification.
      Value srcOffset = metadataOp.getResult(1);
      Value resultOffsetVal =
          builder.create<arith::ConstantIndexOp>(loc, resultOffset);
      Value isSameOffset = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, srcOffset, resultOffsetVal);
      builder.create<cf::AssertOp>(
          loc, isSameOffset,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "offset mismatch"));
    }

    // Check strides.
    for (const auto &it : llvm::enumerate(resultStrides)) {
      // Static/dynamic stride -> dynamic stride does not need verification.
      if (it.value() == ShapedType::kDynamic)
        continue;

      Value srcStride =
          metadataOp.getResult(2 + resultType.getRank() + it.index());
      Value resultStrideVal =
          builder.create<arith::ConstantIndexOp>(loc, it.value());
      Value isSameStride = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, srcStride, resultStrideVal);
      builder.create<cf::AssertOp>(
          loc, isSameStride,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "stride mismatch of dim " + std::to_string(it.index())));
    }
  }
};

struct CopyOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<CopyOpInterface,
                                                         CopyOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto copyOp = cast<CopyOp>(op);
    BaseMemRefType sourceType = copyOp.getSource().getType();
    BaseMemRefType targetType = copyOp.getTarget().getType();
    auto rankedSourceType = dyn_cast<MemRefType>(sourceType);
    auto rankedTargetType = dyn_cast<MemRefType>(targetType);

    // TODO: Verification for unranked memrefs is not supported yet.
    if (!rankedSourceType || !rankedTargetType)
      return;

    assert(sourceType.getRank() == targetType.getRank() && "rank mismatch");
    for (int64_t i = 0, e = sourceType.getRank(); i < e; ++i) {
      // Fully static dimensions in both source and target operand are already
      // verified by the op verifier.
      if (!rankedSourceType.isDynamicDim(i) &&
          !rankedTargetType.isDynamicDim(i))
        continue;
      auto getDimSize = [&](Value memRef, MemRefType type,
                            int64_t dim) -> Value {
        return type.isDynamicDim(dim)
                   ? builder.create<DimOp>(loc, memRef, dim).getResult()
                   : builder
                         .create<arith::ConstantIndexOp>(loc,
                                                         type.getDimSize(dim))
                         .getResult();
      };
      Value sourceDim = getDimSize(copyOp.getSource(), rankedSourceType, i);
      Value targetDim = getDimSize(copyOp.getTarget(), rankedTargetType, i);
      Value sameDimSize = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, sourceDim, targetDim);
      builder.create<cf::AssertOp>(
          loc, sameDimSize,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "size of " + std::to_string(i) +
                      "-th source/target dim does not match"));
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

/// Verifies that the indices on load/store ops are in-bounds of the memref's
/// index space: 0 <= index#i < dim#i
template <typename LoadStoreOp>
struct LoadStoreOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          LoadStoreOpInterface<LoadStoreOp>, LoadStoreOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto loadStoreOp = cast<LoadStoreOp>(op);

    auto memref = loadStoreOp.getMemref();
    auto rank = memref.getType().getRank();
    if (rank == 0) {
      return;
    }
    auto indices = loadStoreOp.getIndices();

    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value assertCond;
    for (auto i : llvm::seq<int64_t>(0, rank)) {
      Value dimOp = builder.createOrFold<memref::DimOp>(loc, memref, i);
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

struct SubViewOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<SubViewOpInterface,
                                                         SubViewOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto subView = cast<SubViewOp>(op);
    MemRefType sourceType = subView.getSource().getType();

    // For each dimension, assert that:
    // 0 <= offset < dim_size
    // 0 <= offset + (size - 1) * stride < dim_size
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    auto metadataOp =
        builder.create<ExtractStridedMetadataOp>(loc, subView.getSource());
    for (int64_t i = 0, e = sourceType.getRank(); i < e; ++i) {
      Value offset = getValueOrCreateConstantIndexOp(
          builder, loc, subView.getMixedOffsets()[i]);
      Value size = getValueOrCreateConstantIndexOp(builder, loc,
                                                   subView.getMixedSizes()[i]);
      Value stride = getValueOrCreateConstantIndexOp(
          builder, loc, subView.getMixedStrides()[i]);

      // Verify that offset is in-bounds.
      Value dimSize = metadataOp.getSizes()[i];
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
              op, "subview runs out-of-bounds along dimension " +
                      std::to_string(i)));
    }
  }
};

struct ExpandShapeOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<ExpandShapeOpInterface,
                                                         ExpandShapeOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto expandShapeOp = cast<ExpandShapeOp>(op);

    // Verify that the expanded dim sizes are a product of the collapsed dim
    // size.
    for (const auto &it :
         llvm::enumerate(expandShapeOp.getReassociationIndices())) {
      Value srcDimSz =
          builder.create<DimOp>(loc, expandShapeOp.getSrc(), it.index());
      int64_t groupSz = 1;
      bool foundDynamicDim = false;
      for (int64_t resultDim : it.value()) {
        if (expandShapeOp.getResultType().isDynamicDim(resultDim)) {
          // Keep this assert here in case the op is extended in the future.
          assert(!foundDynamicDim &&
                 "more than one dynamic dim found in reassoc group");
          (void)foundDynamicDim;
          foundDynamicDim = true;
          continue;
        }
        groupSz *= expandShapeOp.getResultType().getDimSize(resultDim);
      }
      Value staticResultDimSz =
          builder.create<arith::ConstantIndexOp>(loc, groupSz);
      // staticResultDimSz must divide srcDimSz evenly.
      Value mod =
          builder.create<arith::RemSIOp>(loc, srcDimSz, staticResultDimSz);
      Value isModZero = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, mod,
          builder.create<arith::ConstantIndexOp>(loc, 0));
      builder.create<cf::AssertOp>(
          loc, isModZero,
          RuntimeVerifiableOpInterface::generateErrorMessage(
              op, "static result dims in reassoc group do not "
                  "divide src dim evenly"));
    }
  }
};
} // namespace
} // namespace memref
} // namespace mlir

void mlir::memref::registerRuntimeVerifiableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    AssumeAlignmentOp::attachInterface<AssumeAlignmentOpInterface>(*ctx);
    AtomicRMWOp::attachInterface<LoadStoreOpInterface<AtomicRMWOp>>(*ctx);
    CastOp::attachInterface<CastOpInterface>(*ctx);
    CopyOp::attachInterface<CopyOpInterface>(*ctx);
    DimOp::attachInterface<DimOpInterface>(*ctx);
    ExpandShapeOp::attachInterface<ExpandShapeOpInterface>(*ctx);
    GenericAtomicRMWOp::attachInterface<
        LoadStoreOpInterface<GenericAtomicRMWOp>>(*ctx);
    LoadOp::attachInterface<LoadStoreOpInterface<LoadOp>>(*ctx);
    StoreOp::attachInterface<LoadStoreOpInterface<StoreOp>>(*ctx);
    SubViewOp::attachInterface<SubViewOpInterface>(*ctx);
    // Note: There is nothing to verify for ReinterpretCastOp.

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<affine::AffineDialect, arith::ArithDialect,
                     cf::ControlFlowDialect>();
  });
}
