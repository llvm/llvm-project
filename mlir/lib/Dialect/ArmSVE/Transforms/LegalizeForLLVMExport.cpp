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

    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
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

/// Unrolls a conversion to/from equivalent vector types, to allow using a
/// conversion intrinsic that only supports 1-D vector types.
///
/// Example:
/// ```
/// %result = arm_sve.convert_to_svbool %source : vector<2x[4]xi1>
/// ```
/// is rewritten into:
/// ```
/// %cst = arith.constant dense<false> : vector<2x[16]xi1>
/// %1 = vector.extract %source[0] : vector<[4]xi1> from vector<2x[4]xi1>
/// %2 = "arm_sve.intr.convert.to.svbool"(%1)
///                : (vector<[4]xi1>) -> vector<[16]xi1>
/// %3 = vector.insert %2, %cst[0] : vector<[16]xi1> into vector<2x[16]xi1>
/// %4 = vector.extract %source[1] : vector<[4]xi1> from vector<2x[4]xi1>
/// %5 = "arm_sve.intr.convert.to.svbool"(%4)
///                : (vector<[4]xi1>) -> vector<[16]xi1>
/// %result = vector.insert %5, %3[1] : vector<[16]xi1> into vector<2x[16]xi1>
/// ```
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

    // We want to iterate over the input vector in steps of the trailing
    // dimension. So this creates tile shape where all leading dimensions are 1,
    // and the trailing dimension step is the size of the dimension.
    SmallVector<int64_t> tileShape(sourceType.getRank(), 1);
    tileShape.back() = sourceType.getShape().back();

    // Iterate over all scalable mask/predicate slices of the source vector.
    for (SmallVector<int64_t> index :
         StaticTileOffsetRange(sourceType.getShape(), tileShape)) {
      auto extractOrInsertPosition = ArrayRef(index).drop_back();
      auto sourceVector = rewriter.create<vector::ExtractOp>(
          loc, source, extractOrInsertPosition);
      VectorType convertedType =
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

using ZipX2OpLowering = OneToOneConvertToLLVMPattern<ZipX2Op, ZipX2IntrOp>;
using ZipX4OpLowering = OneToOneConvertToLLVMPattern<ZipX4Op, ZipX4IntrOp>;

/// Lower `arm_sve.psel` to LLVM intrinsics. This is almost a 1-to-1 conversion
/// but first input (P1) and result predicates need conversion to/from svbool.
struct PselOpLowering : public ConvertOpToLLVMPattern<PselOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PselOp pselOp, PselOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto svboolType = VectorType::get(16, rewriter.getI1Type(), true);
    auto loc = pselOp.getLoc();
    auto svboolP1 = rewriter.create<ConvertToSvboolIntrOp>(loc, svboolType,
                                                           adaptor.getP1());
    auto indexI32 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI32Type(), pselOp.getIndex());
    auto pselIntr = rewriter.create<PselIntrOp>(loc, svboolType, svboolP1,
                                                pselOp.getP2(), indexI32);
    rewriter.replaceOpWithNewOp<ConvertFromSvboolIntrOp>(
        pselOp, adaptor.getP1().getType(), pselIntr);
    return success();
  }
};

/// Converts `vector.create_mask` ops that match the size of an SVE predicate
/// to the `whilelt` intrinsic. This produces more canonical codegen than the
/// generic LLVM lowering, see https://github.com/llvm/llvm-project/issues/81840
/// for more details. Note that we can't use (the more general) active.lane.mask
/// as its semantics don't neatly map on to `vector.create_mask`, as it does an
/// unsigned comparison (whereas `create_mask` is signed), and is UB/posion if
/// `n` is zero (whereas `create_mask` just returns an all-false mask).
struct CreateMaskOpLowering
    : public ConvertOpToLLVMPattern<vector::CreateMaskOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(vector::CreateMaskOp createMaskOp,
                  vector::CreateMaskOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto maskType = createMaskOp.getVectorType();
    if (maskType.getRank() != 1 || !maskType.isScalable())
      return rewriter.notifyMatchFailure(createMaskOp, "not 1-D and scalable");

    // TODO: Support masks which are multiples of SVE predicates.
    auto maskBaseSize = maskType.getDimSize(0);
    if (maskBaseSize < 2 || maskBaseSize > 16 ||
        !llvm::isPowerOf2_32(uint32_t(maskBaseSize)))
      return rewriter.notifyMatchFailure(createMaskOp,
                                         "not SVE predicate-sized");

    auto loc = createMaskOp.getLoc();
    auto zero = rewriter.create<LLVM::ZeroOp>(loc, rewriter.getI64Type());
    rewriter.replaceOpWithNewOp<WhileLTIntrOp>(createMaskOp, maskType, zero,
                                               adaptor.getOperands()[0]);
    return success();
  }
};

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
               ConvertFromSvboolOpLowering,
               ZipX2OpLowering,
               ZipX4OpLowering,
               PselOpLowering>(converter);
  // Add vector.create_mask conversion with a high benefit as it produces much
  // nicer code than the generic lowering.
  patterns.add<CreateMaskOpLowering>(converter, /*benefit=*/4096);
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
                    ConvertFromSvboolIntrOp,
                    ZipX2IntrOp,
                    ZipX4IntrOp,
                    PselIntrOp,
                    WhileLTIntrOp>();
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
                      ConvertFromSvboolOp,
                      ZipX2Op,
                      ZipX4Op>();
  // clang-format on
}
