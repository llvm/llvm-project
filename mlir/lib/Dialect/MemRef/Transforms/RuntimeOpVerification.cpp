//===- RuntimeOpVerification.cpp - Op Verification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/RuntimeOpVerification.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/RuntimeVerifiableOpInterface.h"

using namespace mlir;

/// Generate an error message string for the given op and the specified error.
static std::string generateErrorMessage(Operation *op, const std::string &msg) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  OpPrintingFlags flags;
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

    // Load additional dialects of which ops may get created.
    ctx->loadDialect<arith::ArithDialect, cf::ControlFlowDialect>();
  });
}
