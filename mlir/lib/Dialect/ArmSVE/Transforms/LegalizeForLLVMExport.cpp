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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arm_sve;

// Extract an LLVM IR type from the LLVM IR dialect type.
static Type unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  if (!LLVM::isCompatibleType(type))
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return type;
}

static Optional<Type>
convertScalableVectorTypeToLLVM(ScalableVectorType svType,
                                LLVMTypeConverter &converter) {
  auto elementType = unwrap(converter.convertType(svType.getElementType()));
  if (!elementType)
    return {};

  auto sVectorType =
      LLVM::LLVMScalableVectorType::get(elementType, svType.getShape().back());
  return sVectorType;
}

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

class ReturnOpTypeConversion : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

static Optional<Value> addUnrealizedCast(OpBuilder &builder,
                                         ScalableVectorType svType,
                                         ValueRange inputs, Location loc) {
  if (inputs.size() != 1 ||
      !inputs[0].getType().isa<LLVM::LLVMScalableVectorType>())
    return Value();
  return builder.create<UnrealizedConversionCastOp>(loc, svType, inputs)
      .getResult(0);
}

using SdotOpLowering = OneToOneConvertToLLVMPattern<SdotOp, SdotIntrOp>;
using SmmlaOpLowering = OneToOneConvertToLLVMPattern<SmmlaOp, SmmlaIntrOp>;
using UdotOpLowering = OneToOneConvertToLLVMPattern<UdotOp, UdotIntrOp>;
using UmmlaOpLowering = OneToOneConvertToLLVMPattern<UmmlaOp, UmmlaIntrOp>;
using VectorScaleOpLowering =
    OneToOneConvertToLLVMPattern<VectorScaleOp, VectorScaleIntrOp>;
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

// Load operation is lowered to code that obtains a pointer to the indexed
// element and loads from it.
struct ScalableLoadOpLowering : public ConvertOpToLLVMPattern<ScalableLoadOp> {
  using ConvertOpToLLVMPattern<ScalableLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ScalableLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = loadOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(loadOp.getContext());

    auto resultType = loadOp.result().getType();
    LLVM::LLVMPointerType llvmDataTypePtr;
    if (resultType.isa<VectorType>()) {
      llvmDataTypePtr =
          LLVM::LLVMPointerType::get(resultType.cast<VectorType>());
    } else if (resultType.isa<ScalableVectorType>()) {
      llvmDataTypePtr = LLVM::LLVMPointerType::get(
          convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
                                          converter)
              .getValue());
    }
    Value dataPtr = getStridedElementPtr(loadOp.getLoc(), type, adaptor.base(),
                                         adaptor.index(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        loadOp.getLoc(), llvmDataTypePtr, dataPtr);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, bitCastedPtr);
    return success();
  }
};

// Store operation is lowered to code that obtains a pointer to the indexed
// element, and stores the given value to it.
struct ScalableStoreOpLowering
    : public ConvertOpToLLVMPattern<ScalableStoreOp> {
  using ConvertOpToLLVMPattern<ScalableStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ScalableStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = storeOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(storeOp.getContext());

    auto resultType = storeOp.value().getType();
    LLVM::LLVMPointerType llvmDataTypePtr;
    if (resultType.isa<VectorType>()) {
      llvmDataTypePtr =
          LLVM::LLVMPointerType::get(resultType.cast<VectorType>());
    } else if (resultType.isa<ScalableVectorType>()) {
      llvmDataTypePtr = LLVM::LLVMPointerType::get(
          convertScalableVectorTypeToLLVM(resultType.cast<ScalableVectorType>(),
                                          converter)
              .getValue());
    }
    Value dataPtr = getStridedElementPtr(storeOp.getLoc(), type, adaptor.base(),
                                         adaptor.index(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        storeOp.getLoc(), llvmDataTypePtr, dataPtr);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, adaptor.value(),
                                               bitCastedPtr);
    return success();
  }
};

static void
populateBasicSVEArithmeticExportPatterns(LLVMTypeConverter &converter,
                                         OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.add<OneToOneConvertToLLVMPattern<ScalableAddIOp, LLVM::AddOp>,
               OneToOneConvertToLLVMPattern<ScalableAddFOp, LLVM::FAddOp>,
               OneToOneConvertToLLVMPattern<ScalableSubIOp, LLVM::SubOp>,
               OneToOneConvertToLLVMPattern<ScalableSubFOp, LLVM::FSubOp>,
               OneToOneConvertToLLVMPattern<ScalableMulIOp, LLVM::MulOp>,
               OneToOneConvertToLLVMPattern<ScalableMulFOp, LLVM::FMulOp>,
               OneToOneConvertToLLVMPattern<ScalableSDivIOp, LLVM::SDivOp>,
               OneToOneConvertToLLVMPattern<ScalableUDivIOp, LLVM::UDivOp>,
               OneToOneConvertToLLVMPattern<ScalableDivFOp, LLVM::FDivOp>
              >(converter);
  // clang-format on
}

static void
configureBasicSVEArithmeticLegalizations(LLVMConversionTarget &target) {
  // clang-format off
  target.addIllegalOp<ScalableAddIOp,
                      ScalableAddFOp,
                      ScalableSubIOp,
                      ScalableSubFOp,
                      ScalableMulIOp,
                      ScalableMulFOp,
                      ScalableSDivIOp,
                      ScalableUDivIOp,
                      ScalableDivFOp>();
  // clang-format on
}

static void
populateSVEMaskGenerationExportPatterns(LLVMTypeConverter &converter,
                                        OwningRewritePatternList &patterns) {
  // clang-format off
  patterns.add<OneToOneConvertToLLVMPattern<ScalableCmpFOp, LLVM::FCmpOp>,
               OneToOneConvertToLLVMPattern<ScalableCmpIOp, LLVM::ICmpOp>
              >(converter);
  // clang-format on
}

static void
configureSVEMaskGenerationLegalizations(LLVMConversionTarget &target) {
  // clang-format off
  target.addIllegalOp<ScalableCmpFOp,
                      ScalableCmpIOp>();
  // clang-format on
}

/// Populate the given list with patterns that convert from ArmSVE to LLVM.
void mlir::populateArmSVELegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // Populate conversion patterns
  // Remove any ArmSVE-specific types from function signatures and results.
  populateFuncOpTypeConversionPattern(patterns, converter);
  converter.addConversion([&converter](ScalableVectorType svType) {
    return convertScalableVectorTypeToLLVM(svType, converter);
  });
  converter.addSourceMaterialization(addUnrealizedCast);

  // clang-format off
  patterns.add<ForwardOperands<CallOp>,
               ForwardOperands<CallIndirectOp>,
               ForwardOperands<ReturnOp>>(converter,
                                          &converter.getContext());
  patterns.add<SdotOpLowering,
               SmmlaOpLowering,
               UdotOpLowering,
               UmmlaOpLowering,
               VectorScaleOpLowering,
               ScalableMaskedAddIOpLowering,
               ScalableMaskedAddFOpLowering,
               ScalableMaskedSubIOpLowering,
               ScalableMaskedSubFOpLowering,
               ScalableMaskedMulIOpLowering,
               ScalableMaskedMulFOpLowering,
               ScalableMaskedSDivIOpLowering,
               ScalableMaskedUDivIOpLowering,
               ScalableMaskedDivFOpLowering>(converter);
  patterns.add<ScalableLoadOpLowering,
               ScalableStoreOpLowering>(converter);
  // clang-format on
  populateBasicSVEArithmeticExportPatterns(converter, patterns);
  populateSVEMaskGenerationExportPatterns(converter, patterns);
}

void mlir::configureArmSVELegalizeForExportTarget(
    LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<SdotIntrOp,
                    SmmlaIntrOp,
                    UdotIntrOp,
                    UmmlaIntrOp,
                    VectorScaleIntrOp,
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
                      VectorScaleOp,
                      ScalableMaskedAddIOp,
                      ScalableMaskedAddFOp,
                      ScalableMaskedSubIOp,
                      ScalableMaskedSubFOp,
                      ScalableMaskedMulIOp,
                      ScalableMaskedMulFOp,
                      ScalableMaskedSDivIOp,
                      ScalableMaskedUDivIOp,
                      ScalableMaskedDivFOp,
                      ScalableLoadOp,
                      ScalableStoreOp>();
  // clang-format on
  auto hasScalableVectorType = [](TypeRange types) {
    for (Type type : types)
      if (type.isa<arm_sve::ScalableVectorType>())
        return true;
    return false;
  };
  target.addDynamicallyLegalOp<FuncOp>([hasScalableVectorType](FuncOp op) {
    return !hasScalableVectorType(op.getType().getInputs()) &&
           !hasScalableVectorType(op.getType().getResults());
  });
  target.addDynamicallyLegalOp<CallOp, CallIndirectOp, ReturnOp>(
      [hasScalableVectorType](Operation *op) {
        return !hasScalableVectorType(op->getOperandTypes()) &&
               !hasScalableVectorType(op->getResultTypes());
      });
  configureBasicSVEArithmeticLegalizations(target);
  configureSVEMaskGenerationLegalizations(target);
}
