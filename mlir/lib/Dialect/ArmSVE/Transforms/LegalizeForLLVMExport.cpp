//===- LegalizeForLLVMExport.cpp - Prepare ArmSVE for LLVM translation ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ArmSVE/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

namespace {

template <typename Op, typename IntrOp>
struct SvboolConversionOpLowering : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op convertOp, typename Op::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = convertOp.getLoc();

    auto source = convertOp.getSource();
    VectorType sourceType = source.getType();
    VectorType resultType = convertOp.getResult().getType();

    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultType, rewriter.getZeroAttr(resultType));

    SmallVector<int64_t> tileShape(sourceType.getRank(), 1);
    tileShape.back() = sourceType.getShape().back();

    for (SmallVector<int64_t> index :
         StaticTileOffsetRange(sourceType.getShape(), tileShape)) {
      auto extractOrInsertPosition = ArrayRef(index).drop_back();
      auto sourceVector = rewriter.create<vector::ExtractOp>(
          loc, source, extractOrInsertPosition);
      auto convertedType =
          VectorType::Builder(llvm::cast<VectorType>(sourceVector.getType()))
              .setDim(0, resultType.getShape().back());
      auto convertedVector =
          rewriter.create<IntrOp>(loc, TypeRange{convertedType}, sourceVector);
      result = rewriter.create<vector::InsertOp>(loc, convertedVector, result,
                                                 extractOrInsertPosition);
    }

    rewriter.replaceOp(convertOp, result);
    return success();
  }
};

using ConvertToSvboolOpLowering =
    SvboolConversionOpLowering<ConvertToSvboolOp, ConvertToSvboolIntrOp>;

using ConvertFromSvboolOpLowering =
    SvboolConversionOpLowering<ConvertFromSvboolOp, ConvertFromSvboolIntrOp>;

} // namespace

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
               ScalableMaskedDivFOpLowering,
               ConvertToSvboolOpLowering,
               ConvertFromSvboolOpLowering>(converter);
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
                    ScalableMaskedDivFIntrOp,
                    ConvertToSvboolIntrOp,
                    ConvertFromSvboolIntrOp>();
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
                      ScalableMaskedDivFOp,
                      ConvertToSvboolOp,
                      ConvertFromSvboolOp>();
  // clang-format on
}
