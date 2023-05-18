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

/// Generate an error message string for the given op and the specified error.
static std::string generateErrorMessage(Operation *op, const std::string &msg) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  OpPrintingFlags flags;
  // We may generate a lot of error messages and so we need to ensure the
  // printing is fast.
  flags.elideLargeElementsAttrs();
  flags.printGenericOpForm();
  flags.skipRegions();
  flags.useLocalScope();
  stream << "ERROR: Runtime op verification failed\n";
  op->print(stream, flags);
  stream << "\n^ " << msg;
  stream << "\nLocation: ";
  op->getLoc().print(stream);
  return stream.str();
}

namespace mlir {
namespace memref {
namespace {
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
      builder.create<cf::AssertOp>(loc, isSameRank,
                                   generateErrorMessage(op, "rank mismatch"));
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
          generateErrorMessage(op, "size mismatch of dim " +
                                       std::to_string(it.index())));
    }

    // Get result offset and strides.
    int64_t resultOffset;
    SmallVector<int64_t> resultStrides;
    if (failed(getStridesAndOffset(resultType, resultStrides, resultOffset)))
      return;

    // Check offset.
    if (resultOffset != ShapedType::kDynamic) {
      // Static/dynamic offset -> dynamic offset does not need verification.
      Value srcOffset = metadataOp.getResult(1);
      Value resultOffsetVal =
          builder.create<arith::ConstantIndexOp>(loc, resultOffset);
      Value isSameOffset = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, srcOffset, resultOffsetVal);
      builder.create<cf::AssertOp>(loc, isSameOffset,
                                   generateErrorMessage(op, "offset mismatch"));
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
          generateErrorMessage(op, "stride mismatch of dim " +
                                       std::to_string(it.index())));
    }
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
      auto index = indices[i];

      auto dimOp = builder.createOrFold<memref::DimOp>(loc, memref, i);

      auto geLow = builder.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, index, zero);
      auto ltHigh = builder.createOrFold<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, index, dimOp);
      auto andOp = builder.createOrFold<arith::AndIOp>(loc, geLow, ltHigh);

      assertCond =
          i > 0 ? builder.createOrFold<arith::AndIOp>(loc, assertCond, andOp)
                : andOp;
    }
    builder.create<cf::AssertOp>(
        loc, assertCond, generateErrorMessage(op, "out-of-bounds access"));
  }
};

/// Compute the linear index for the provided strided layout and indices.
Value computeLinearIndex(OpBuilder &builder, Location loc, OpFoldResult offset,
                         ArrayRef<OpFoldResult> strides,
                         ArrayRef<OpFoldResult> indices) {
  auto [expr, values] = computeLinearIndex(offset, strides, indices);
  auto index =
      affine::makeComposedFoldedAffineApply(builder, loc, expr, values);
  return getValueOrCreateConstantIndexOp(builder, loc, index);
}

/// Returns two Values representing the bounds of the provided strided layout
/// metadata. The bounds are returned as a half open interval -- [low, high).
std::pair<Value, Value> computeLinearBounds(OpBuilder &builder, Location loc,
                                            OpFoldResult offset,
                                            ArrayRef<OpFoldResult> strides,
                                            ArrayRef<OpFoldResult> sizes) {
  auto zeros = SmallVector<int64_t>(sizes.size(), 0);
  auto indices = getAsIndexOpFoldResult(builder.getContext(), zeros);
  auto lowerBound = computeLinearIndex(builder, loc, offset, strides, indices);
  auto upperBound = computeLinearIndex(builder, loc, offset, strides, sizes);
  return {lowerBound, upperBound};
}

/// Returns two Values representing the bounds of the memref. The bounds are
/// returned as a half open interval -- [low, high).
std::pair<Value, Value> computeLinearBounds(OpBuilder &builder, Location loc,
                                            TypedValue<BaseMemRefType> memref) {
  auto runtimeMetadata = builder.create<ExtractStridedMetadataOp>(loc, memref);
  auto offset = runtimeMetadata.getConstifiedMixedOffset();
  auto strides = runtimeMetadata.getConstifiedMixedStrides();
  auto sizes = runtimeMetadata.getConstifiedMixedSizes();
  return computeLinearBounds(builder, loc, offset, strides, sizes);
}

/// Verifies that the linear bounds of a reinterpret_cast op are within the
/// linear bounds of the base memref: low >= baseLow && high <= baseHigh
struct ReinterpretCastOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<
          ReinterpretCastOpInterface, ReinterpretCastOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto reinterpretCast = cast<ReinterpretCastOp>(op);
    auto baseMemref = reinterpretCast.getSource();
    auto resultMemref =
        cast<TypedValue<BaseMemRefType>>(reinterpretCast.getResult());

    builder.setInsertionPointAfter(op);

    // Compute the linear bounds of the base memref
    auto [baseLow, baseHigh] = computeLinearBounds(builder, loc, baseMemref);

    // Compute the linear bounds of the resulting memref
    auto [low, high] = computeLinearBounds(builder, loc, resultMemref);

    // Check low >= baseLow
    auto geLow = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, low, baseLow);

    // Check high <= baseHigh
    auto leHigh = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, high, baseHigh);

    auto assertCond = builder.createOrFold<arith::AndIOp>(loc, geLow, leHigh);

    builder.create<cf::AssertOp>(
        loc, assertCond,
        generateErrorMessage(
            op,
            "result of reinterpret_cast is out-of-bounds of the base memref"));
  }
};

/// Verifies that the linear bounds of a subview op are within the linear bounds
/// of the base memref: low >= baseLow && high <= baseHigh
/// TODO: This is not yet a full runtime verification of subview. For example,
/// consider:
///   %m = memref.alloc(%c10, %c10) : memref<10x10xf32>
///   memref.subview %m[%c0, %c0][%c20, %c2][%c1, %c1]
///      : memref<?x?xf32> to memref<?x?xf32>
/// The subview is in-bounds of the entire base memref but the first dimension
/// is out-of-bounds. Future work would verify the bounds on a per-dimension
/// basis.
struct SubViewOpInterface
    : public RuntimeVerifiableOpInterface::ExternalModel<SubViewOpInterface,
                                                         SubViewOp> {
  void generateRuntimeVerification(Operation *op, OpBuilder &builder,
                                   Location loc) const {
    auto subView = cast<SubViewOp>(op);
    auto baseMemref = cast<TypedValue<BaseMemRefType>>(subView.getSource());
    auto resultMemref = cast<TypedValue<BaseMemRefType>>(subView.getResult());

    builder.setInsertionPointAfter(op);

    // Compute the linear bounds of the base memref
    auto [baseLow, baseHigh] = computeLinearBounds(builder, loc, baseMemref);

    // Compute the linear bounds of the resulting memref
    auto [low, high] = computeLinearBounds(builder, loc, resultMemref);

    // Check low >= baseLow
    auto geLow = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, low, baseLow);

    // Check high <= baseHigh
    auto leHigh = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, high, baseHigh);

    auto assertCond = builder.createOrFold<arith::AndIOp>(loc, geLow, leHigh);

    builder.create<cf::AssertOp>(
        loc, assertCond,
        generateErrorMessage(op,
                             "subview is out-of-bounds of the base memref"));
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
          generateErrorMessage(op, "static result dims in reassoc group do not "
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
    CastOp::attachInterface<CastOpInterface>(*ctx);
    ExpandShapeOp::attachInterface<ExpandShapeOpInterface>(*ctx);
    LoadOp::attachInterface<LoadStoreOpInterface<LoadOp>>(*ctx);
    ReinterpretCastOp::attachInterface<ReinterpretCastOpInterface>(*ctx);
    StoreOp::attachInterface<LoadStoreOpInterface<StoreOp>>(*ctx);
    SubViewOp::attachInterface<SubViewOpInterface>(*ctx);

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<affine::AffineDialect, arith::ArithDialect,
                     cf::ControlFlowDialect>();
  });
}
