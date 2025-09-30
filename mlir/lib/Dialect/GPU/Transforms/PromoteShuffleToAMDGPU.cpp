//===- PromoteShuffleToAMDGPU.cpp - Promote shuffle to AMDGPU -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains patterns to try to promote `gpu.shuffle`s to specialized
// AMDGPU intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>

using namespace mlir;

namespace {

constexpr amdgpu::Chipset kGfx950 = amdgpu::Chipset(9, 5, 0);

/// Try to promote `gpu.shuffle` to `amdgpu.swizzle_bitmode`, width must be 64
/// and offset must be a constant integer in the range [0, 31].
struct PromoteShuffleToSwizzlePattern
    : public OpRewritePattern<gpu::ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getMode() != gpu::ShuffleMode::XOR)
      return rewriter.notifyMatchFailure(op,
                                         "only xor shuffle mode is supported");

    if (!isConstantIntValue(op.getWidth(), 64))
      return rewriter.notifyMatchFailure(op,
                                         "only 64 width shuffle is supported");

    std::optional<int64_t> offset = getConstantIntValue(op.getOffset());
    if (!offset)
      return rewriter.notifyMatchFailure(op,
                                         "offset must be a constant integer");

    int64_t offsetValue = *offset;
    if (offsetValue < 0 || offsetValue >= 32)
      return rewriter.notifyMatchFailure(op,
                                         "offset must be in the range [0, 31]");

    Location loc = op.getLoc();
    Value res = amdgpu::SwizzleBitModeOp::create(
        rewriter, loc, op.getResult(0).getType(), op.getValue(), /*andMask=*/31,
        /*orMask=*/0, /*xorMask=*/offsetValue);
    Value valid = arith::ConstantIntOp::create(rewriter, loc, 1, /*width*/ 1);
    rewriter.replaceOp(op, {res, valid});
    return success();
  }
};

/// Try to promote `gpu.shuffle` to `amdgpu.permlane_swap`, width must be 64
/// and offset must be a constant integer in the set {16, 32}.
struct PromoteShuffleToPermlanePattern
    : public OpRewritePattern<gpu::ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getMode() != gpu::ShuffleMode::XOR)
      return rewriter.notifyMatchFailure(op,
                                         "only xor shuffle mode is supported");

    if (!isConstantIntValue(op.getWidth(), 64))
      return rewriter.notifyMatchFailure(op,
                                         "only 64 width shuffle is supported");

    std::optional<int64_t> offset = getConstantIntValue(op.getOffset());
    if (!offset)
      return rewriter.notifyMatchFailure(op,
                                         "offset must be a constant integer");

    int64_t offsetValue = *offset;
    if (offsetValue != 16 && offsetValue != 32)
      return rewriter.notifyMatchFailure(op, "offset must be either 16 or 32");

    Location loc = op.getLoc();
    Value res = amdgpu::PermlaneSwapOp::create(
        rewriter, loc, op.getResult(0).getType(), op.getValue(), offsetValue);
    Value valid = arith::ConstantIntOp::create(rewriter, loc, 1, /*width*/ 1);
    rewriter.replaceOp(op, {res, valid});
    return success();
  }
};

static Value getLaneId(RewriterBase &rewriter, Location loc) {
  auto int32Type = IntegerType::get(rewriter.getContext(), 32);
  Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  Value minus1 = arith::ConstantIntOp::create(rewriter, loc, -1, 32);
  NamedAttribute noundef = {LLVM::LLVMDialect::getNoUndefAttrName(),
                            rewriter.getUnitAttr()};
  NamedAttribute lowRange = {LLVM::LLVMDialect::getRangeAttrName(),
                             LLVM::ConstantRangeAttr::get(rewriter.getContext(),
                                                          APInt::getZero(32),
                                                          APInt(32, 32))};
  NamedAttribute highRange = {
      LLVM::LLVMDialect::getRangeAttrName(),
      LLVM::ConstantRangeAttr::get(rewriter.getContext(), APInt::getZero(32),
                                   APInt(32, 64))};
  Value mbcntLo = ROCDL::MbcntLoOp::create(
      rewriter, loc, int32Type, minus1, zero, /*arg_attrs=*/{},
      /*res_attrs=*/
      rewriter.getArrayAttr(rewriter.getDictionaryAttr({noundef, lowRange})));
  Value laneId = ROCDL::MbcntHiOp::create(
      rewriter, loc, int32Type, minus1, mbcntLo, /*arg_attrs=*/{},
      rewriter.getArrayAttr(rewriter.getDictionaryAttr({noundef, highRange})));
  return laneId;
}

/// Try to promote `gpu.shuffle` to `amdgpu.dpp`, width must be 64
/// and offset must be a constant integer in the set {16, 32}.
struct PromoteShuffleToDPPPattern : public OpRewritePattern<gpu::ShuffleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::ShuffleOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<int64_t> width = getConstantIntValue(op.getWidth());
    if (!width)
      return rewriter.notifyMatchFailure(op,
                                         "width must be a constant integer");
    int64_t widthValue = *width;
    if (!llvm::is_contained({4, 8, 12, 16, 32, 48, 64}, widthValue))
      return rewriter.notifyMatchFailure(
          op, "width must be 4, 8, 12, 16, 32, 48 or 64");

    std::optional<int64_t> offset = getConstantIntValue(op.getOffset());
    if (!offset)
      return rewriter.notifyMatchFailure(op,
                                         "offset must be a constant integer");

    int64_t offsetValue = *offset;
    Location loc = op.getLoc();
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);

    amdgpu::DPPPerm kind;
    Attribute permAttr = rewriter.getUnitAttr();
    Value srcLane;
    Value dstLane;
    switch (op.getMode()) {
    case gpu::ShuffleMode::XOR: {
      if (offsetValue != 1 && offsetValue != 2)
        return rewriter.notifyMatchFailure(
            op, "xor shuffle mode is only supported for offsets of 1 or 2");
      kind = amdgpu::DPPPerm::quad_perm;
      srcLane = getLaneId(rewriter, loc);
      dstLane = LLVM::XOrOp::create(rewriter, loc, int32Type, srcLane,
                                    op.getOffset());

      if (offsetValue == 1)
        permAttr = rewriter.getI32ArrayAttr({1, 0, 3, 2});
      else if (offsetValue == 2)
        permAttr = rewriter.getI32ArrayAttr({2, 3, 0, 1});
      break;
    }
    case gpu::ShuffleMode::UP: {
      if (offsetValue != 1)
        return rewriter.notifyMatchFailure(
            op, "up shuffle mode is only supported for offset 1");
      kind = amdgpu::DPPPerm::wave_shr;
      srcLane = getLaneId(rewriter, loc);
      dstLane = LLVM::SubOp::create(rewriter, loc, int32Type, srcLane,
                                    op.getOffset());
      break;
    }
    case gpu::ShuffleMode::DOWN: {
      if (offsetValue != 1)
        return rewriter.notifyMatchFailure(
            op, "down shuffle mode is only supported for offset 1");
      kind = amdgpu::DPPPerm::wave_shl;
      srcLane = getLaneId(rewriter, loc);
      dstLane = LLVM::AddOp::create(rewriter, loc, int32Type, srcLane,
                                    op.getOffset());
      break;
    }
    case gpu::ShuffleMode::IDX:
      return rewriter.notifyMatchFailure(op,
                                         "idx shuffle mode is not supported");
    }

    unsigned bankMask = 0xF;
    if (widthValue == 4)
      bankMask = 0x1;
    else if (widthValue == 8)
      bankMask = 0x3;
    else if (widthValue == 12)
      bankMask = 0x7;

    unsigned rowMask = 0xF;
    if (widthValue == 16)
      rowMask = 0x1;
    else if (widthValue == 32)
      rowMask = 0x3;
    else if (widthValue == 48)
      rowMask = 0x7;

    constexpr bool boundCtrl = false;

    Value negwidth =
        arith::ConstantIntOp::create(rewriter, loc, int32Type, -widthValue);
    Value add =
        arith::AddIOp::create(rewriter, loc, int32Type, srcLane, op.getWidth());
    Value widthOrZeroIfOutside =
        arith::AndIOp::create(rewriter, loc, int32Type, add, negwidth);
    Value isActiveSrcLane =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt, dstLane,
                              widthOrZeroIfOutside);

    Value dpp = amdgpu::DPPOp::create(rewriter, loc, op.getResult(0).getType(),
                                      op.getValue(), op.getValue(), kind,
                                      permAttr, rowMask, bankMask, boundCtrl);
    Value poison =
        LLVM::PoisonOp::create(rewriter, loc, op.getResult(0).getType());

    Value selectResult =
        arith::SelectOp::create(rewriter, loc, isActiveSrcLane, dpp, poison);

    rewriter.replaceOp(op, {selectResult, isActiveSrcLane});
    return success();
  }
};

} // namespace

void mlir::populateGpuPromoteShuffleToAMDGPUPatterns(
    RewritePatternSet &patterns, std::optional<amdgpu::Chipset> maybeChipset) {
  patterns.add<PromoteShuffleToSwizzlePattern>(patterns.getContext(),
                                               /*benefit*/ 1);
  patterns.add<PromoteShuffleToDPPPattern>(patterns.getContext(),
                                           /*benefit*/ 2);
  if (maybeChipset && *maybeChipset >= kGfx950)
    patterns.add<PromoteShuffleToPermlanePattern>(patterns.getContext(),
                                                  /*benefit*/ 3);
}
