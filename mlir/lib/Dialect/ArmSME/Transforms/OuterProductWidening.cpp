//===- OuterProductWidening.cpp - Widen 'arm_sme.outerproduct' ops --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrites that fold 'arm_sme.outerproduct' operations
// into the 2-way or 4-way widening outerproduct operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arm-sme-outerproduct-widening"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_OUTERPRODUCTWIDENING
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {
// Fold two 'arm_sme.outerproduct' operations that are chained via the
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
//  %a_packed = arm_sve.zip %a0, %a1 : vector<[8]xf16> to vector<[8]xf16>
//  %b_packed = arm_sve.zip %b0, %b1 : vector<[8]xf16> to vector<[8]xf16>
//  %0 = arm_sme.fmopa_wide_2way %a_packed, %b_packed : vector<[8]xf16>,
//                                                      vector<[4]xf32>
class OuterProduct2WayWidening
    : public OpRewritePattern<arm_sme::OuterProductOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    Value acc = op.getAcc();
    if (!acc)
      return rewriter.notifyMatchFailure(op, "no accumulator operand");

    arm_sme::OuterProductOp op1 = acc.getDefiningOp<arm_sme::OuterProductOp>();
    arm_sme::OuterProductOp op2 = op;
    if (!op1)
      return rewriter.notifyMatchFailure(op,
                                         "defining op of accumulator operand "
                                         "must be an 'arm_sme.outerproduct'");

    if (op1.getKind() != op2.getKind())
      return rewriter.notifyMatchFailure(
          op, "combining kind (add or sub) of outer products must match");

    if (!llvm::hasSingleElement(op1->getUses())) {
      // We could still widen, but if the first outer product has an
      // accumulator it will be used as the root for tile allocation and since
      // the widening outer product uses the same accumulator it will get
      // assigned the same tile ID, resulting in 3 outer products and incorrect
      // results. No accumulator would be ok, but it's simpler to prevent this
      // altogether, since it has no benefit.
      return rewriter.notifyMatchFailure(
          op, "first outer product is not single use and cannot be removed, "
              "no benefit to widening");
    }

    auto nxnxv4i32 =
        VectorType::get({4, 4}, rewriter.getI32Type(), {true, true});
    auto nxnxv4f32 =
        VectorType::get({4, 4}, rewriter.getF32Type(), {true, true});
    auto nxv4i16 = VectorType::get({4}, rewriter.getI16Type(), true);
    auto nxv4f16 = VectorType::get({4}, rewriter.getF16Type(), true);
    auto nxv4bf16 = VectorType::get({4}, rewriter.getBF16Type(), true);
    if ((failed(
             isWidenable<arith::ExtFOp>(rewriter, op1, nxnxv4f32, nxv4f16)) ||
         failed(
             isWidenable<arith::ExtFOp>(rewriter, op2, nxnxv4f32, nxv4f16))) &&
        (failed(
             isWidenable<arith::ExtFOp>(rewriter, op1, nxnxv4f32, nxv4bf16)) ||
         failed(
             isWidenable<arith::ExtFOp>(rewriter, op2, nxnxv4f32, nxv4bf16))) &&
        (failed(
             isWidenable<arith::ExtSIOp>(rewriter, op1, nxnxv4i32, nxv4i16)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op2, nxnxv4i32, nxv4i16))) &&
        (failed(
             isWidenable<arith::ExtUIOp>(rewriter, op1, nxnxv4i32, nxv4i16)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op2, nxnxv4i32, nxv4i16))))
      return failure();

    auto loc = op.getLoc();

    // zip(lhs, rhs)
    auto packInputs = [&](VectorType type, Value lhs, Value rhs) {
      auto undef = rewriter.create<LLVM::UndefOp>(loc, type);
      auto insertLHS =
          rewriter.create<vector::ScalableInsertOp>(loc, lhs, undef, 0);
      auto insertRHS =
          rewriter.create<vector::ScalableInsertOp>(loc, rhs, undef, 0);
      return rewriter.create<arm_sve::Zip1IntrOp>(loc, type, insertLHS,
                                                  insertRHS);
    };

    auto extOp = op.getLhs().getDefiningOp();
    VectorType extSourceVectorType =
        cast<VectorType>(extOp->getOperand(0).getType());
    VectorType widenedVectorType =
        VectorType::Builder(extSourceVectorType)
            .setDim(0, extSourceVectorType.getShape()[0] * 2);
    auto lhs = packInputs(widenedVectorType,
                          op1.getLhs().getDefiningOp()->getOperand(0),
                          op2.getLhs().getDefiningOp()->getOperand(0));
    auto rhs = packInputs(widenedVectorType,
                          op1.getRhs().getDefiningOp()->getOperand(0),
                          op2.getRhs().getDefiningOp()->getOperand(0));

    Value lhsMask, rhsMask;
    if (op1.getLhsMask() || op2.getLhsMask()) {
      if (!(op1.getLhsMask() && op2.getLhsMask()))
        return rewriter.notifyMatchFailure(
            op, "unsupported masking, either both outerproducts are masked "
                "or neither");

      VectorType maskType = VectorType::Builder(widenedVectorType)
                                .setElementType(rewriter.getI1Type());
      lhsMask = packInputs(maskType, op1.getLhsMask(), op2.getLhsMask());
      rhsMask = packInputs(maskType, op1.getRhsMask(), op2.getRhsMask());
    }

    arm_sme::CombiningKind kind = op.getKind();
    assert((kind == arm_sme::CombiningKind::Add ||
            kind == arm_sme::CombiningKind::Sub) &&
           "unhandled arm_sme::CombiningKind!");

    if (isa<arith::ExtFOp>(extOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::FMopaWide2WayOp>(
            op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::FMopsWide2WayOp>(
            op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else if (isa<arith::ExtSIOp>(extOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::SMopaWide2WayOp>(
            op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::SMopsWide2WayOp>(
            op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else if (isa<arith::ExtUIOp>(extOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::UMopaWide2WayOp>(
            op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::UMopsWide2WayOp>(
            op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else
      llvm_unreachable("unexpected extend op!");

    op1.erase();

    return success();
  }

private:
  template <typename ExtOp>
  LogicalResult isWidenable(PatternRewriter &rewriter,
                            arm_sme::OuterProductOp op, VectorType resultType,
                            VectorType inputType) const {
    if (op.getResultType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "unsupported result type, expected 'vector<[4]x[4]xi32>' or "
              "'vector<[4]x[4]xf32>'");

    auto lhsDefOp = op.getLhs().getDefiningOp<ExtOp>();
    auto rhsDefOp = op.getRhs().getDefiningOp<ExtOp>();

    if (!lhsDefOp || !rhsDefOp)
      return rewriter.notifyMatchFailure(
          op, "defining op of outerproduct operands must be 'arith.extf' or "
              "'arith.extsi' or 'arith.extui'");

    auto lhsInType = cast<VectorType>(lhsDefOp->getOperand(0).getType());
    auto rhsInType = cast<VectorType>(rhsDefOp->getOperand(0).getType());

    if (lhsInType != inputType || rhsInType != inputType)
      return rewriter.notifyMatchFailure(
          op, "unsupported input types, expected 'vector<[4]xi16>' or "
              "'vector<[4]xf16>' or 'vector<[4]xbf16>'");
    return success();
  }
};

struct OuterProductWideningPass
    : public arm_sme::impl::OuterProductWideningBase<OuterProductWideningPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOuterProductWideningPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::arm_sme::populateOuterProductWideningPatterns(
    RewritePatternSet &patterns) {
  patterns.add<OuterProduct2WayWidening>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::arm_sme::createOuterProductWideningPass() {
  return std::make_unique<OuterProductWideningPass>();
}
