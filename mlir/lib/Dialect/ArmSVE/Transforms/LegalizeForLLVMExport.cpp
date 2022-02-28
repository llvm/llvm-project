//===- LegalizeForLLVMExport.cpp - Prepare ArmSVE for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arm_sve;

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

using SdotOpLowering = OneToOneConvertToLLVMPattern<SdotOp, SdotIntrOp>;
using SmmlaOpLowering = OneToOneConvertToLLVMPattern<SmmlaOp, SmmlaIntrOp>;
using UdotOpLowering = OneToOneConvertToLLVMPattern<UdotOp, UdotIntrOp>;
using UmmlaOpLowering = OneToOneConvertToLLVMPattern<UmmlaOp, UmmlaIntrOp>;
using ScalableMaskedAddIOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedAddIOp,
                                 ScalableMaskedAddIIntrOp>;
using ScalableMaskedAddFOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedAddFOp,
                                 ScalableMaskedAddFIntrOp>;
using ScalableMaskedSubIOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedSubIOp,
                                 ScalableMaskedSubIIntrOp>;
using ScalableMaskedSubFOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedSubFOp,
                                 ScalableMaskedSubFIntrOp>;
using ScalableMaskedMulIOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedMulIOp,
                                 ScalableMaskedMulIIntrOp>;
using ScalableMaskedMulFOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedMulFOp,
                                 ScalableMaskedMulFIntrOp>;
using ScalableMaskedSDivIOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedSDivIOp,
                                 ScalableMaskedSDivIIntrOp>;
using ScalableMaskedUDivIOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedUDivIOp,
                                 ScalableMaskedUDivIIntrOp>;
using ScalableMaskedDivFOpLowering =
    OneToOneConvertToLLVMPattern<ScalableMaskedDivFOp,
                                 ScalableMaskedDivFIntrOp>;

/// Populate the given list with patterns that convert from ArmSVE to LLVM.
void mlir::populateArmSVELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Populate conversion patterns

  // clang-format off
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>>(converter,
                                          &converter.getContext());
  patterns.add<SdotOpLowering,
               SmmlaOpLowering,
               UdotOpLowering,
               UmmlaOpLowering,
               ScalableMaskedAddIOpLowering,
               ScalableMaskedAddFOpLowering,
               ScalableMaskedSubIOpLowering,
               ScalableMaskedSubFOpLowering,
               ScalableMaskedMulIOpLowering,
               ScalableMaskedMulFOpLowering,
               ScalableMaskedSDivIOpLowering,
               ScalableMaskedUDivIOpLowering,
               ScalableMaskedDivFOpLowering>(converter);
  // clang-format on
}

void mlir::configureArmSVELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<SdotIntrOp,
                    SmmlaIntrOp,
                    UdotIntrOp,
                    UmmlaIntrOp,
                    ScalableMaskedAddIIntrOp,
                    ScalableMaskedAddFIntrOp,
                    ScalableMaskedSubIIntrOp,
                    ScalableMaskedSubFIntrOp,
                    ScalableMaskedMulIIntrOp,
                    ScalableMaskedMulFIntrOp,
                    ScalableMaskedSDivIIntrOp,
                    ScalableMaskedUDivIIntrOp,
                    ScalableMaskedDivFIntrOp>();
  target.addIllegalOp<SdotOp,
                      SmmlaOp,
                      UdotOp,
                      UmmlaOp,
                      ScalableMaskedAddIOp,
                      ScalableMaskedAddFOp,
                      ScalableMaskedSubIOp,
                      ScalableMaskedSubFOp,
                      ScalableMaskedMulIOp,
                      ScalableMaskedMulFOp,
                      ScalableMaskedSDivIOp,
                      ScalableMaskedUDivIOp,
                      ScalableMaskedDivFOp>();
  // clang-format on
}
