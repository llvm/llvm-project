//===- ArmSVEToLLVM.cpp - Convert ArmSVE to the LLVM dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArmSVEToLLVM/ArmSVEToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/ArmSVE/ArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::arm_sve;
using namespace mlir::vector;

using SdotOpLowering =
    OneToOneConvertToLLVMPattern<SdotOp, LLVM::aarch64_arm_sve_sdot>;

using SmmlaOpLowering =
    OneToOneConvertToLLVMPattern<SmmlaOp, LLVM::aarch64_arm_sve_smmla>;

using UdotOpLowering =
    OneToOneConvertToLLVMPattern<UdotOp, LLVM::aarch64_arm_sve_udot>;

using UmmlaOpLowering =
    OneToOneConvertToLLVMPattern<UmmlaOp, LLVM::aarch64_arm_sve_ummla>;

using VectorScaleOpLowering =
    OneToOneConvertToLLVMPattern<VectorScaleOp, LLVM::vector_scale>;

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
  matchAndRewrite(OpTy op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (ValueRange(operands).getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.updateRootInPlace(op, [&]() { op->setOperands(operands); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [&]() { op->setOperands(operands); });
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

/// Populate the given list with patterns that convert from ArmSVE to LLVM.
void mlir::populateArmSVEToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  converter.addConversion([&converter](ScalableVectorType svType) {
    return convertScalableVectorTypeToLLVM(svType, converter);
  });
  converter.addSourceMaterialization(addUnrealizedCast);

  // clang-format off
  patterns.insert<ForwardOperands<CallOp>,
                  ForwardOperands<CallIndirectOp>,
                  ForwardOperands<ReturnOp>>(converter,
                                             &converter.getContext());
  patterns.insert<SdotOpLowering,
                  SmmlaOpLowering,
                  UdotOpLowering,
                  UmmlaOpLowering,
                  VectorScaleOpLowering>(converter);
  // clang-format on
}
