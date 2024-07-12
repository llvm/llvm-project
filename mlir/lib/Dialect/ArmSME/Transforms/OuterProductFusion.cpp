//===- OuterProductFusion.cpp - Fuse 'arm_sme.outerproduct' ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrites that fuse 'arm_sme.outerproduct' operations
// into the 2-way or 4-way widening outerproduct operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "arm-sme-outerproduct-fusion"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_OUTERPRODUCTFUSION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

// Common match failure reasons.
static constexpr StringLiteral
    kMatchFailureNoAccumulator("no accumulator operand");
static constexpr StringLiteral kMatchFailureExpectedOuterProductDefOp(
    "defining op of accumulator must be 'arm_sme.outerproduct'");
static constexpr StringLiteral kMatchFailureInconsistentCombiningKind(
    "combining kind (add or sub) of outer products must match");
static constexpr StringLiteral kMatchFailureInconsistentMasking(
    "unsupported masking, either both outerproducts are masked "
    "or neither");
static constexpr StringLiteral kMatchFailureOuterProductNotSingleUse(
    "outer product(s) not single use and cannot be removed, no benefit to "
    "fusing");

// An outer product is compatible if all of the following are true:
// - the result type matches `resultType`.
// - the defining operation of LHS is of the type `LhsExtOp`.
// - the defining operation of RHS is of the type `RhsExtOp`.
// - the input types of the defining operations are identical and match
//   `inputType`.
template <typename LhsExtOp, typename RhsExtOp = LhsExtOp>
static LogicalResult isCompatible(PatternRewriter &rewriter,
                                  arm_sme::OuterProductOp op,
                                  VectorType resultType, VectorType inputType) {
  if (op.getResultType() != resultType)
    return rewriter.notifyMatchFailure(op.getLoc(), [&](Diagnostic &diag) {
      diag << "unsupported result type, expected " << resultType;
    });

  auto lhsDefOp = op.getLhs().getDefiningOp<LhsExtOp>();
  auto rhsDefOp = op.getRhs().getDefiningOp<RhsExtOp>();

  if (!lhsDefOp || !rhsDefOp)
    return rewriter.notifyMatchFailure(
        op, "defining op of outerproduct operands must be one of: "
            "'arith.extf' or 'arith.extsi' or 'arith.extui'");

  auto lhsInType = cast<VectorType>(lhsDefOp.getIn().getType());
  auto rhsInType = cast<VectorType>(rhsDefOp.getIn().getType());

  if (lhsInType != inputType || rhsInType != inputType)
    return rewriter.notifyMatchFailure(op.getLoc(), [&](Diagnostic &diag) {
      diag << "unsupported input type, expected " << inputType;
    });

  return success();
}

// Create 'llvm.experimental.vector.interleave2' intrinsic from `lhs` and `rhs`.
static Value createInterleave2Intrinsic(RewriterBase &rewriter, Location loc,
                                        Value lhs, Value rhs) {
  auto inputType = cast<VectorType>(lhs.getType());
  VectorType inputTypeX2 =
      VectorType::Builder(inputType).setDim(0, inputType.getShape()[0] * 2);
  return rewriter.create<LLVM::vector_interleave2>(loc, inputTypeX2, lhs, rhs);
}

// Fuse two 'arm_sme.outerproduct' operations that are chained via the
// accumulator into 2-way outer product operation.
//
// For example:
//
//  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
//  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
//  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>,
//                                               vector<[4]xf32>
//
//  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
//  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>
//  %1 = arm_sme.outerproduct %a1_ext, %b1_ext, %0 : vector<[4]xf32>,
//                                                   vector<[4]xf32>
//
// Becomes:
//
//  %a_packed = "llvm.intr.experimental.vector.interleave2"(%a0, %a1)
//    : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
//  %b_packed = "llvm.intr.experimental.vector.interleave2"(%b0, %b1)
//    : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
//  %0 = arm_sme.fmopa_2way %a_packed, %b_packed
//    : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
class OuterProductFusion2Way
    : public OpRewritePattern<arm_sme::OuterProductOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    Value acc = op.getAcc();
    if (!acc)
      return rewriter.notifyMatchFailure(op, kMatchFailureNoAccumulator);

    arm_sme::OuterProductOp op1 = acc.getDefiningOp<arm_sme::OuterProductOp>();
    arm_sme::OuterProductOp op2 = op;
    if (!op1)
      return rewriter.notifyMatchFailure(
          op, kMatchFailureExpectedOuterProductDefOp);

    if (op1.getKind() != op2.getKind())
      return rewriter.notifyMatchFailure(
          op, kMatchFailureInconsistentCombiningKind);

    if (!op1->hasOneUse()) {
      // If the first outer product has uses other than as the input to another
      // outer product, it can't be erased after fusion. This is a problem when
      // it also has an accumulator as this will be used as the root for tile
      // allocation and since the widening outer product uses the same
      // accumulator it will get assigned the same tile ID, resulting in 3
      // outer products accumulating to the same tile and incorrect results.
      //
      // Example:
      //
      //  %acc = arith.constant dense<0.0> ; root for tile allocation
      //  %0 = arm_sme.outerproduct %a0, %b0 acc(%acc)
      //  vector.print %0                  ; intermediary use, can't erase %0
      //  %1 = arm_sme.outerproduct %a1, %b1 acc(%0)
      //
      // After fusion and tile allocation
      //
      //  %0 = arm_sme.zero {tile_id = 0 : i32}
      //  %1 = arm_sme.outerproduct %a0, %b0 acc(%0) {tile_id = 0 : i32}
      //  vector.print %1
      //  %2 = arm_sme.fmopa_2way %a, %b acc(%0) {tile_id = 0 : i32}
      //
      // No accumulator would be ok, but it's simpler to prevent this
      // altogether, since it has no benefit.
      return rewriter.notifyMatchFailure(op,
                                         kMatchFailureOuterProductNotSingleUse);
    }

    if (bool(op1.getLhsMask()) != bool(op2.getLhsMask()))
      return rewriter.notifyMatchFailure(op, kMatchFailureInconsistentMasking);

    if (failed(canFuseOuterProducts(rewriter, op1, op2)))
      return failure();

    auto loc = op.getLoc();
    auto packInputs = [&](Value lhs, Value rhs) {
      return createInterleave2Intrinsic(rewriter, loc, lhs, rhs);
    };

    auto lhs = packInputs(op1.getLhs().getDefiningOp()->getOperand(0),
                          op2.getLhs().getDefiningOp()->getOperand(0));
    auto rhs = packInputs(op1.getRhs().getDefiningOp()->getOperand(0),
                          op2.getRhs().getDefiningOp()->getOperand(0));

    Value lhsMask, rhsMask;
    if (op1.getLhsMask() || op2.getLhsMask()) {
      lhsMask = packInputs(op1.getLhsMask(), op2.getLhsMask());
      rhsMask = packInputs(op1.getRhsMask(), op2.getRhsMask());
    }

    auto extOp = op.getLhs().getDefiningOp();

    arm_sme::CombiningKind kind = op.getKind();
    if (kind == arm_sme::CombiningKind::Add) {
      TypeSwitch<Operation *>(extOp)
          .Case<arith::ExtFOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::FMopa2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtSIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::SMopa2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtUIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::UMopa2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Default([&](auto) { llvm_unreachable("unexpected extend op!"); });
    } else if (kind == arm_sme::CombiningKind::Sub) {
      TypeSwitch<Operation *>(extOp)
          .Case<arith::ExtFOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::FMops2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtSIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::SMops2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtUIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::UMops2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Default([&](auto) { llvm_unreachable("unexpected extend op!"); });
    } else {
      llvm_unreachable("unexpected arm_sme::CombiningKind!");
    }

    rewriter.eraseOp(op1);

    return success();
  }

private:
  // A pair of outer product can be fused if all of the following are true:
  // - input and result types match.
  // - the defining operations of the inputs are identical extensions,
  //   specifically either:
  //     - a signed or unsigned extension for integer types.
  //     - a floating-point extension for floating-point types.
  // - the types and extension are supported, i.e. there's a 2-way operation
  //   they can be fused into.
  LogicalResult canFuseOuterProducts(PatternRewriter &rewriter,
                                     arm_sme::OuterProductOp op1,
                                     arm_sme::OuterProductOp op2) const {
    // Supported result types.
    auto nxnxv4i32 =
        VectorType::get({4, 4}, rewriter.getI32Type(), {true, true});
    auto nxnxv4f32 =
        VectorType::get({4, 4}, rewriter.getF32Type(), {true, true});
    // Supported input types.
    // Note: this is before packing so these have half the number of elements
    // of the input vector types of the 2-way operations.
    auto nxv4i16 = VectorType::get({4}, rewriter.getI16Type(), true);
    auto nxv4f16 = VectorType::get({4}, rewriter.getF16Type(), true);
    auto nxv4bf16 = VectorType::get({4}, rewriter.getBF16Type(), true);
    if ((failed(
             isCompatible<arith::ExtFOp>(rewriter, op1, nxnxv4f32, nxv4f16)) ||
         failed(
             isCompatible<arith::ExtFOp>(rewriter, op2, nxnxv4f32, nxv4f16))) &&
        (failed(
             isCompatible<arith::ExtFOp>(rewriter, op1, nxnxv4f32, nxv4bf16)) ||
         failed(isCompatible<arith::ExtFOp>(rewriter, op2, nxnxv4f32,
                                            nxv4bf16))) &&
        (failed(
             isCompatible<arith::ExtSIOp>(rewriter, op1, nxnxv4i32, nxv4i16)) ||
         failed(isCompatible<arith::ExtSIOp>(rewriter, op2, nxnxv4i32,
                                             nxv4i16))) &&
        (failed(
             isCompatible<arith::ExtUIOp>(rewriter, op1, nxnxv4i32, nxv4i16)) ||
         failed(
             isCompatible<arith::ExtUIOp>(rewriter, op2, nxnxv4i32, nxv4i16))))
      return failure();

    return success();
  }
};

// Fuse four 'arm_sme.outerproduct' operations that are chained via the
// accumulator into 4-way outer product operation.
class OuterProductFusion4Way
    : public OpRewritePattern<arm_sme::OuterProductOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<arm_sme::OuterProductOp, 4> outerProductChain;
    outerProductChain.push_back(op);

    for (int i = 0; i < 3; ++i) {
      auto currentOp = outerProductChain.back();
      auto acc = currentOp.getAcc();
      if (!acc)
        return rewriter.notifyMatchFailure(op, kMatchFailureNoAccumulator);
      auto previousOp = acc.getDefiningOp<arm_sme::OuterProductOp>();
      if (!previousOp)
        return rewriter.notifyMatchFailure(
            op, kMatchFailureExpectedOuterProductDefOp);
      if (!previousOp->hasOneUse())
        return rewriter.notifyMatchFailure(
            op, kMatchFailureOuterProductNotSingleUse);
      if (previousOp.getKind() != currentOp.getKind())
        return rewriter.notifyMatchFailure(
            op, kMatchFailureInconsistentCombiningKind);
      if (bool(previousOp.getLhsMask()) != bool(currentOp.getLhsMask()))
        return rewriter.notifyMatchFailure(
            op, kMatchFailureInconsistentCombiningKind);
      outerProductChain.push_back(previousOp);
    }

    if (failed(canFuseOuterProducts(rewriter, outerProductChain)))
      return failure();

    arm_sme::OuterProductOp op1 = outerProductChain[3];
    arm_sme::OuterProductOp op2 = outerProductChain[2];
    arm_sme::OuterProductOp op3 = outerProductChain[1];
    arm_sme::OuterProductOp op4 = outerProductChain[0];

    auto loc = op.getLoc();
    auto packInputs = [&](Value lhs, Value rhs) {
      return createInterleave2Intrinsic(rewriter, loc, lhs, rhs);
    };

    auto lhs0 = packInputs(op1.getLhs().getDefiningOp()->getOperand(0),
                           op3.getLhs().getDefiningOp()->getOperand(0));
    auto lhs1 = packInputs(op2.getLhs().getDefiningOp()->getOperand(0),
                           op4.getLhs().getDefiningOp()->getOperand(0));
    auto lhs = packInputs(lhs0, lhs1);

    auto rhs0 = packInputs(op1.getRhs().getDefiningOp()->getOperand(0),
                           op3.getRhs().getDefiningOp()->getOperand(0));
    auto rhs1 = packInputs(op2.getRhs().getDefiningOp()->getOperand(0),
                           op4.getRhs().getDefiningOp()->getOperand(0));
    auto rhs = packInputs(rhs0, rhs1);

    Value lhsMask, rhsMask;
    if (op1.getLhsMask() || op2.getLhsMask() || op3.getLhsMask() ||
        op4.getLhsMask()) {
      auto lhs0Mask = packInputs(op1.getLhsMask(), op3.getLhsMask());
      auto lhs1Mask = packInputs(op2.getLhsMask(), op4.getLhsMask());
      lhsMask = packInputs(lhs0Mask, lhs1Mask);

      auto rhs0Mask = packInputs(op1.getRhsMask(), op3.getRhsMask());
      auto rhs1Mask = packInputs(op2.getRhsMask(), op4.getRhsMask());
      rhsMask = packInputs(rhs0Mask, rhs1Mask);
    }

    auto lhsExtOp = op.getLhs().getDefiningOp();
    auto rhsExtOp = op.getRhs().getDefiningOp();

    arm_sme::CombiningKind kind = op.getKind();
    if (kind == arm_sme::CombiningKind::Add) {
      if (isa<arith::ExtSIOp>(lhsExtOp) && isa<arith::ExtSIOp>(rhsExtOp)) {
        // signed
        rewriter.replaceOpWithNewOp<arm_sme::SMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else if (isa<arith::ExtUIOp>(lhsExtOp) &&
                 isa<arith::ExtUIOp>(rhsExtOp)) {
        // unsigned
        rewriter.replaceOpWithNewOp<arm_sme::UMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else if (isa<arith::ExtSIOp>(lhsExtOp) &&
                 isa<arith::ExtUIOp>(rhsExtOp)) {
        // signed by unsigned
        rewriter.replaceOpWithNewOp<arm_sme::SuMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else if (isa<arith::ExtUIOp>(lhsExtOp) &&
                 isa<arith::ExtSIOp>(rhsExtOp)) {
        // unsigned by signed
        rewriter.replaceOpWithNewOp<arm_sme::UsMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else {
        llvm_unreachable("unexpected extend op!");
      }
    } else if (kind == arm_sme::CombiningKind::Sub) {
      if (isa<arith::ExtSIOp>(lhsExtOp) && isa<arith::ExtSIOp>(rhsExtOp)) {
        // signed
        rewriter.replaceOpWithNewOp<arm_sme::SMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else if (isa<arith::ExtUIOp>(lhsExtOp) &&
                 isa<arith::ExtUIOp>(rhsExtOp)) {
        // unsigned
        rewriter.replaceOpWithNewOp<arm_sme::UMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else if (isa<arith::ExtSIOp>(lhsExtOp) &&
                 isa<arith::ExtUIOp>(rhsExtOp)) {
        // signed by unsigned
        rewriter.replaceOpWithNewOp<arm_sme::SuMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else if (isa<arith::ExtUIOp>(lhsExtOp) &&
                 isa<arith::ExtSIOp>(rhsExtOp)) {
        // unsigned by signed
        rewriter.replaceOpWithNewOp<arm_sme::UsMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      } else {
        llvm_unreachable("unexpected extend op!");
      }
    } else {
      llvm_unreachable("unexpected arm_sme::CombiningKind!");
    }

    rewriter.eraseOp(op3);
    rewriter.eraseOp(op2);
    rewriter.eraseOp(op1);

    return success();
  }

private:
  // Four outer products can be fused if all of the following are true:
  // - input and result types match.
  // - the defining operations of the inputs are identical extensions,
  //   specifically either:
  //     - a signed or unsigned extension for integer types.
  //     - a floating-point extension for floating-point types.
  // - the types and extension are supported, i.e. there's a 4-way operation
  //   they can be fused into.
  LogicalResult
  canFuseOuterProducts(PatternRewriter &rewriter,
                       ArrayRef<arm_sme::OuterProductOp> ops) const {
    // Supported result types.
    auto nxnxv4i32 =
        VectorType::get({4, 4}, rewriter.getI32Type(), {true, true});
    auto nxnxv2i64 =
        VectorType::get({2, 2}, rewriter.getI64Type(), {true, true});

    // Supported input types.
    // Note: this is before packing so these have 1/4 the number of elements
    // of the input vector types of the 4-way operations.
    auto nxv4i8 = VectorType::get({4}, rewriter.getI8Type(), true);
    auto nxv2i16 = VectorType::get({2}, rewriter.getI16Type(), true);

    auto failedToMatch = [&](VectorType resultType, VectorType inputType,
                             auto lhsExtendOp, auto rhsExtendOp) {
      using LhsExtendOpTy = decltype(lhsExtendOp);
      using RhsExtendOpTy = decltype(rhsExtendOp);
      for (auto op : ops) {
        if (failed(isCompatible<LhsExtendOpTy, RhsExtendOpTy>(
                rewriter, op, resultType, inputType)))
          return true;
      }
      return false;
    };

    if (failedToMatch(nxnxv4i32, nxv4i8, arith::ExtSIOp{}, arith::ExtSIOp{}) &&
        failedToMatch(nxnxv4i32, nxv4i8, arith::ExtUIOp{}, arith::ExtUIOp{}) &&
        failedToMatch(nxnxv4i32, nxv4i8, arith::ExtSIOp{}, arith::ExtUIOp{}) &&
        failedToMatch(nxnxv4i32, nxv4i8, arith::ExtUIOp{}, arith::ExtSIOp{}) &&
        failedToMatch(nxnxv2i64, nxv2i16, arith::ExtSIOp{}, arith::ExtSIOp{}) &&
        failedToMatch(nxnxv2i64, nxv2i16, arith::ExtUIOp{}, arith::ExtUIOp{}) &&
        failedToMatch(nxnxv2i64, nxv2i16, arith::ExtSIOp{}, arith::ExtUIOp{}) &&
        failedToMatch(nxnxv2i64, nxv2i16, arith::ExtUIOp{}, arith::ExtSIOp{}))
      return failure();

    return success();
  }
};

// Rewrites: vector.extract(arith.extend) -> arith.extend(vector.extract).
//
// This transforms IR like:
//   %0 = arith.extsi %src : vector<4x[8]xi8> to vector<4x[8]xi32>
//   %1 = vector.extract %0[0] : vector<[8]xi32> from vector<4x[8]xi32>
// Into:
//   %0 = vector.extract %src[0] : vector<[8]xi8> from vector<4x[8]xi8>
//   %1 = arith.extsi %0 : vector<[8]xi8> to vector<[8]xi32>
//
// This enables outer product fusion in the `-arm-sme-outer-product-fusion`
// pass when the result is the input to an outer product.
struct SwapVectorExtractOfArithExtend
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    VectorType resultType = llvm::dyn_cast<VectorType>(extractOp.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(extractOp,
                                         "extracted type is not a vector type");

    auto numScalableDims = llvm::count(resultType.getScalableDims(), true);
    if (numScalableDims != 1)
      return rewriter.notifyMatchFailure(
          extractOp, "extracted type is not a 1-D scalable vector type");

    auto *extendOp = extractOp.getVector().getDefiningOp();
    if (!isa_and_present<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>(
            extendOp))
      return rewriter.notifyMatchFailure(extractOp,
                                         "extract not from extend op");

    auto loc = extractOp.getLoc();
    StringAttr extendOpName = extendOp->getName().getIdentifier();
    Value extendSource = extendOp->getOperand(0);

    // Create new extract from source of extend.
    Value newExtract = rewriter.create<vector::ExtractOp>(
        loc, extendSource, extractOp.getMixedPosition());

    // Extend new extract to original result type.
    Operation *newExtend =
        rewriter.create(loc, extendOpName, Value(newExtract), resultType);

    rewriter.replaceOp(extractOp, newExtend);

    return success();
  }
};

// Same as above, but for vector.scalable.extract.
//
// This transforms IR like:
//   %0 = arith.extsi %src : vector<[8]xi8> to vector<[8]xi32>
//   %1 = vector.scalable.extract %0[0] : vector<[4]xi32> from vector<[8]xi32>
// Into:
//   %0 = vector.scalable.extract %src[0] : vector<[4]xi8> from vector<[8]xi8>
//   %1 = arith.extsi %0 : vector<[4]xi8> to vector<[4]xi32>
//
// This enables outer product fusion in the `-arm-sme-outer-product-fusion`
// pass when the result is the input to an outer product.
struct SwapVectorScalableExtractOfArithExtend
    : public OpRewritePattern<vector::ScalableExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ScalableExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto *extendOp = extractOp.getSource().getDefiningOp();
    if (!isa_and_present<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>(
            extendOp))
      return rewriter.notifyMatchFailure(extractOp,
                                         "extract not from extend op");

    auto loc = extractOp.getLoc();
    VectorType resultType = extractOp.getResultVectorType();

    Value extendSource = extendOp->getOperand(0);
    StringAttr extendOpName = extendOp->getName().getIdentifier();
    VectorType extendSourceVectorType =
        cast<VectorType>(extendSource.getType());

    // Create new extract from source of extend.
    VectorType extractResultVectorType =
        resultType.clone(extendSourceVectorType.getElementType());
    Value newExtract = rewriter.create<vector::ScalableExtractOp>(
        loc, extractResultVectorType, extendSource, extractOp.getPos());

    // Extend new extract to original result type.
    Operation *newExtend =
        rewriter.create(loc, extendOpName, Value(newExtract), resultType);

    rewriter.replaceOp(extractOp, newExtend);

    return success();
  }
};

struct OuterProductFusionPass
    : public arm_sme::impl::OuterProductFusionBase<OuterProductFusionPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOuterProductFusionPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::arm_sme::populateOuterProductFusionPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  // Note: High benefit to ensure extract(extend) are swapped first.
  patterns.add<SwapVectorExtractOfArithExtend,
               SwapVectorScalableExtractOfArithExtend>(context, 1024);
  patterns.add<OuterProductFusion2Way, OuterProductFusion4Way>(context);
}

std::unique_ptr<Pass> mlir::arm_sme::createOuterProductFusionPass() {
  return std::make_unique<OuterProductFusionPass>();
}
