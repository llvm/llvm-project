//===- VectorReductionToDot.cpp - Lower vector reductions to dot ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Transforms/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace mlir::amdgpu {
#define GEN_PASS_DEF_AMDGPUVECTORREDUCTIONTODOTPASS
#include "mlir/Dialect/AMDGPU/Transforms/Passes.h.inc"
} // namespace mlir::amdgpu

using namespace mlir;
using namespace mlir::amdgpu;

namespace {

enum class ExtensionKind { None, Float, Signed, Unsigned };

struct DotOperandInfo {
  Value source;
  VectorType vectorType;
  SmallVector<OpFoldResult> extractPosition;
  ExtensionKind extensionKind = ExtensionKind::None;
};

struct DotChainInfo {
  int64_t chunkSize = 0;
  bool unsignedA = false;
  bool unsignedB = false;
};

static bool isSupportedFp8DotType(Type type) {
  return isa<Float8E4M3FNType, Float8E5M2Type>(type);
}

static bool isDotSupportedOnChipset(Type lhsElementType, Type rhsElementType,
                                    Type accumulatorType, bool unsignedA,
                                    bool unsignedB, Chipset chipset) {
  if (lhsElementType.isF16() && rhsElementType.isF16()) {
    if (accumulatorType.isF32())
      return hasDot10Insts(chipset);
    if (accumulatorType.isF16())
      return hasDot9Insts(chipset);
    return false;
  }

  if (lhsElementType.isBF16() && rhsElementType.isBF16()) {
    if (accumulatorType.isF32())
      return hasDot12Insts(chipset);
    if (accumulatorType.isBF16())
      return hasDot9Insts(chipset);
    return false;
  }

  if (isSupportedFp8DotType(lhsElementType) &&
      isSupportedFp8DotType(rhsElementType))
    return accumulatorType.isF32() && hasDot11Insts(chipset);

  auto integerType = dyn_cast<IntegerType>(lhsElementType);
  auto accumulatorIntegerType = dyn_cast<IntegerType>(accumulatorType);
  if (!integerType || lhsElementType != rhsElementType ||
      !accumulatorIntegerType || !accumulatorIntegerType.isSignless() ||
      accumulatorIntegerType.getWidth() != 32)
    return false;

  bool mixedSign = unsignedA != unsignedB;
  if (mixedSign) {
    if (!hasDot8Insts(chipset))
      return false;
    return integerType.getWidth() == 8 || integerType.getWidth() == 4;
  }

  switch (integerType.getWidth()) {
  case 16:
    return hasDot2Insts(chipset);
  case 8:
  case 4:
    return unsignedA ? hasDot7Insts(chipset)
                     : hasDot1Insts(chipset) || hasDot8Insts(chipset);
  default:
    return false;
  }
}

static std::optional<int64_t> getDotInputChunkSize(Type sourceElementType) {
  if (sourceElementType.isF16() || sourceElementType.isBF16())
    return 2;
  if (isSupportedFp8DotType(sourceElementType))
    return 4;

  auto integerType = dyn_cast<IntegerType>(sourceElementType);
  if (!integerType)
    return std::nullopt;

  switch (integerType.getWidth()) {
  case 16:
    return 2;
  case 8:
    return 4;
  case 4:
    return 8;
  default:
    return std::nullopt;
  }
}

template <typename ExtOp>
static std::optional<DotOperandInfo>
matchExtendedVector(Value value, ExtensionKind extensionKind) {
  if (auto extOp = value.getDefiningOp<ExtOp>()) {
    auto sourceType = dyn_cast<VectorType>(extOp.getIn().getType());
    if (!sourceType)
      return std::nullopt;
    return DotOperandInfo{extOp.getIn(), sourceType, {}, extensionKind};
  }

  auto extractOp = value.getDefiningOp<vector::ExtractOp>();
  if (!extractOp)
    return std::nullopt;

  auto extOp = extractOp.getSource().getDefiningOp<ExtOp>();
  if (!extOp)
    return std::nullopt;

  auto extractedType = dyn_cast<VectorType>(extractOp.getType());
  if (!extractedType)
    return std::nullopt;

  auto extSourceType = dyn_cast<VectorType>(extOp.getIn().getType());
  if (!extSourceType)
    return std::nullopt;

  VectorType narrowedType = extractedType.clone(extSourceType.getElementType());
  return DotOperandInfo{extOp.getIn(), narrowedType,
                        extractOp.getMixedPosition(), extensionKind};
}

static std::optional<DotOperandInfo> matchDotOperand(Value value,
                                                     Type accumulatorType) {
  auto vectorType = dyn_cast<VectorType>(value.getType());
  if (!vectorType)
    return std::nullopt;

  Type elementType = vectorType.getElementType();
  if ((elementType.isF16() || elementType.isBF16()) &&
      elementType == accumulatorType)
    return DotOperandInfo{value, vectorType, {}, ExtensionKind::None};

  if (accumulatorType.isF32())
    return matchExtendedVector<arith::ExtFOp>(value, ExtensionKind::Float);

  if (!accumulatorType.isInteger(32))
    return std::nullopt;

  if (std::optional<DotOperandInfo> signedOperand =
          matchExtendedVector<arith::ExtSIOp>(value, ExtensionKind::Signed))
    return signedOperand;

  return matchExtendedVector<arith::ExtUIOp>(value, ExtensionKind::Unsigned);
}

static bool sameVectorShape(VectorType lhsType, VectorType rhsType) {
  return llvm::equal(lhsType.getShape(), rhsType.getShape()) &&
         llvm::equal(lhsType.getScalableDims(), rhsType.getScalableDims());
}

static FailureOr<DotChainInfo> getDotChainInfo(vector::ReductionOp reductionOp,
                                               const DotOperandInfo &lhs,
                                               const DotOperandInfo &rhs,
                                               Chipset chipset,
                                               PatternRewriter &rewriter) {
  VectorType lhsType = lhs.vectorType;
  VectorType rhsType = rhs.vectorType;

  if (lhsType.getRank() != 1 || rhsType.getRank() != 1)
    return rewriter.notifyMatchFailure(reductionOp,
                                       "dot operands are not rank-1 vectors");

  if (lhsType.isScalable() || rhsType.isScalable())
    return rewriter.notifyMatchFailure(reductionOp,
                                       "scalable vectors are not supported");

  if (!sameVectorShape(lhsType, rhsType))
    return rewriter.notifyMatchFailure(reductionOp,
                                       "dot operands have different shapes");

  Type lhsElementType = lhsType.getElementType();
  Type rhsElementType = rhsType.getElementType();

  Type accumulatorType = reductionOp.getType();
  std::optional<int64_t> lhsChunkSize = getDotInputChunkSize(lhsElementType);
  std::optional<int64_t> rhsChunkSize = getDotInputChunkSize(rhsElementType);
  if (!lhsChunkSize || !rhsChunkSize || *lhsChunkSize != *rhsChunkSize)
    return rewriter.notifyMatchFailure(reductionOp,
                                       "unsupported dot input element type");

  int64_t vectorSize = lhsType.getDimSize(0);
  if (vectorSize % *lhsChunkSize != 0)
    return rewriter.notifyMatchFailure(reductionOp,
                                       "vector length is not a multiple of the "
                                       "native dot input width");

  bool unsignedA = lhs.extensionKind == ExtensionKind::Unsigned;
  bool unsignedB = rhs.extensionKind == ExtensionKind::Unsigned;

  VectorType chunkedLhsType = VectorType::get({*lhsChunkSize}, lhsElementType);
  VectorType chunkedRhsType = VectorType::get({*rhsChunkSize}, rhsElementType);
  if (!amdgpu::DotOp::isCompatibleTypeCombination(
          chunkedLhsType, chunkedRhsType, accumulatorType, unsignedA, unsignedB,
          /*clamp=*/false))
    return rewriter.notifyMatchFailure(
        reductionOp, "dot operand types are not compatible with amdgpu.dot");

  if (!isDotSupportedOnChipset(lhsElementType, rhsElementType, accumulatorType,
                               unsignedA, unsignedB, chipset))
    return rewriter.notifyMatchFailure(
        reductionOp, "dot operation is not supported on the selected chipset");

  return DotChainInfo{*lhsChunkSize, unsignedA, unsignedB};
}

static bool hasRequiredFastMath(vector::ReductionOp reductionOp,
                                arith::MulFOp mulOp) {
  arith::FastMathFlags requiredReductionFlags =
      arith::FastMathFlags::contract | arith::FastMathFlags::reassoc;
  if (!reductionOp.getAcc())
    requiredReductionFlags = requiredReductionFlags | arith::FastMathFlags::nsz;
  return arith::bitEnumContainsAll(reductionOp.getFastmath(),
                                   requiredReductionFlags) &&
         arith::bitEnumContainsAll(mulOp.getFastmath(),
                                   arith::FastMathFlags::contract);
}

static Value materializeNarrowedVector(PatternRewriter &rewriter, Location loc,
                                       const DotOperandInfo &operand) {
  if (operand.extractPosition.empty())
    return operand.source;
  return vector::ExtractOp::create(rewriter, loc, operand.source,
                                   operand.extractPosition);
}

static Value extractDotChunk(PatternRewriter &rewriter, Location loc,
                             Value source, int64_t offset, int64_t chunkSize) {
  return vector::ExtractStridedSliceOp::create(
      rewriter, loc, source, ArrayRef<int64_t>{offset},
      ArrayRef<int64_t>{chunkSize}, ArrayRef<int64_t>{1});
}

struct VectorReductionToDotChain final : OpRewritePattern<vector::ReductionOp> {
  VectorReductionToDotChain(MLIRContext *context, Chipset chipset,
                            PatternBenefit benefit = 1)
      : OpRewritePattern<vector::ReductionOp>(context, benefit),
        chipset(chipset) {}

  LogicalResult matchAndRewrite(vector::ReductionOp reductionOp,
                                PatternRewriter &rewriter) const override {
    if (reductionOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(reductionOp,
                                         "combining kind is not 'add'");

    auto maskableOp =
        cast<vector::MaskableOpInterface>(reductionOp.getOperation());
    if (maskableOp.isMasked())
      return rewriter.notifyMatchFailure(reductionOp,
                                         "masked reductions are not supported");

    Value reductionVector = reductionOp.getVector();
    auto mulFOp = reductionVector.getDefiningOp<arith::MulFOp>();
    auto mulIOp = reductionVector.getDefiningOp<arith::MulIOp>();
    if (!mulFOp && !mulIOp)
      return rewriter.notifyMatchFailure(
          reductionOp, "reduction operand is not an arithmetic multiply");

    if (mulFOp && !hasRequiredFastMath(reductionOp, mulFOp))
      return rewriter.notifyMatchFailure(
          reductionOp,
          "floating-point dot requires contract and reassoc flags");
    if (mulIOp &&
        mulIOp.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return rewriter.notifyMatchFailure(
          reductionOp, "integer multiply overflow flags are not supported");

    Value lhs = mulFOp ? mulFOp.getLhs() : mulIOp.getLhs();
    Value rhs = mulFOp ? mulFOp.getRhs() : mulIOp.getRhs();

    std::optional<DotOperandInfo> lhsInfo =
        matchDotOperand(lhs, reductionOp.getType());
    if (!lhsInfo)
      return rewriter.notifyMatchFailure(reductionOp,
                                         "unsupported LHS dot operand");

    std::optional<DotOperandInfo> rhsInfo =
        matchDotOperand(rhs, reductionOp.getType());
    if (!rhsInfo)
      return rewriter.notifyMatchFailure(reductionOp,
                                         "unsupported RHS dot operand");

    FailureOr<DotChainInfo> dotChainInfo =
        getDotChainInfo(reductionOp, *lhsInfo, *rhsInfo, chipset, rewriter);
    if (failed(dotChainInfo))
      return failure();

    Location loc = reductionOp.getLoc();
    Value lhsVector = materializeNarrowedVector(rewriter, loc, *lhsInfo);
    Value rhsVector = materializeNarrowedVector(rewriter, loc, *rhsInfo);

    Value accumulator = reductionOp.getAcc();
    if (!accumulator) {
      accumulator = arith::ConstantOp::create(
          rewriter, loc, reductionOp.getType(),
          rewriter.getZeroAttr(reductionOp.getType()));
    }

    int64_t vectorSize = lhsInfo->vectorType.getDimSize(0);
    for (int64_t offset = 0; offset < vectorSize;
         offset += dotChainInfo->chunkSize) {
      Value lhsChunk = extractDotChunk(rewriter, loc, lhsVector, offset,
                                       dotChainInfo->chunkSize);
      Value rhsChunk = extractDotChunk(rewriter, loc, rhsVector, offset,
                                       dotChainInfo->chunkSize);
      UnitAttr unsignedA =
          dotChainInfo->unsignedA ? rewriter.getUnitAttr() : UnitAttr();
      UnitAttr unsignedB =
          dotChainInfo->unsignedB ? rewriter.getUnitAttr() : UnitAttr();
      auto dotOp = amdgpu::DotOp::create(
          rewriter, loc, reductionOp.getType(), lhsChunk, rhsChunk, accumulator,
          unsignedA, unsignedB, /*clamp=*/UnitAttr());
      accumulator = dotOp.getDestD();
    }

    rewriter.replaceOp(reductionOp, accumulator);
    return success();
  }

private:
  Chipset chipset;
};

struct AmdgpuVectorReductionToDotPass final
    : amdgpu::impl::AmdgpuVectorReductionToDotPassBase<
          AmdgpuVectorReductionToDotPass> {
  using Base::Base;

  void runOnOperation() override {
    FailureOr<Chipset> maybeChipset = Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    populateAmdgpuVectorReductionToDotPatterns(patterns, *maybeChipset);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::amdgpu::populateAmdgpuVectorReductionToDotPatterns(
    RewritePatternSet &patterns, Chipset chipset, PatternBenefit benefit) {
  patterns.add<VectorReductionToDotChain>(patterns.getContext(), chipset,
                                          benefit);
}
