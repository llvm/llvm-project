//===- ConvertStandardToSPIRV.cpp - Standard to SPIR-V dialect conversion--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert standard ops to SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is a boolean scalar or vector type.
static bool isBoolScalarOrVector(Type type) {
  if (type.isInteger(1))
    return true;
  if (auto vecType = type.dyn_cast<VectorType>())
    return vecType.getElementType().isInteger(1);
  return false;
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

// Note that DRR cannot be used for the patterns in this file: we may need to
// convert type along the way, which requires ConversionPattern. DRR generates
// normal RewritePattern.

namespace {

/// Converts binary standard operations to SPIR-V operations.
template <typename StdOp, typename SPIRVOp>
class BinaryOpPattern final : public SPIRVOpLowering<StdOp> {
public:
  using SPIRVOpLowering<StdOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 2);
    auto dstType = this->typeConverter.convertType(operation.getType());
    if (!dstType)
      return failure();
    rewriter.template replaceOpWithNewOp<SPIRVOp>(operation, dstType, operands,
                                                  ArrayRef<NamedAttribute>());
    return success();
  }
};

/// Converts bitwise standard operations to SPIR-V operations. This is a special
/// pattern other than the BinaryOpPatternPattern because if the operands are
/// boolean values, SPIR-V uses different operations (`SPIRVLogicalOp`). For
/// non-boolean operands, SPIR-V should use `SPIRVBitwiseOp`.
template <typename StdOp, typename SPIRVLogicalOp, typename SPIRVBitwiseOp>
class BitwiseOpPattern final : public SPIRVOpLowering<StdOp> {
public:
  using SPIRVOpLowering<StdOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 2);
    auto dstType =
        this->typeConverter.convertType(operation.getResult().getType());
    if (!dstType)
      return failure();
    if (isBoolScalarOrVector(operands.front().getType())) {
      rewriter.template replaceOpWithNewOp<SPIRVLogicalOp>(
          operation, dstType, operands, ArrayRef<NamedAttribute>());
    } else {
      rewriter.template replaceOpWithNewOp<SPIRVBitwiseOp>(
          operation, dstType, operands, ArrayRef<NamedAttribute>());
    }
    return success();
  }
};

/// Converts composite std.constant operation to spv.constant.
class ConstantCompositeOpPattern final : public SPIRVOpLowering<ConstantOp> {
public:
  using SPIRVOpLowering<ConstantOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(ConstantOp constCompositeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts scalar std.constant operation to spv.constant.
class ConstantScalarOpPattern final : public SPIRVOpLowering<ConstantOp> {
public:
  using SPIRVOpLowering<ConstantOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(ConstantOp constIndexOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts floating-point comparison operations to SPIR-V ops.
class CmpFOpPattern final : public SPIRVOpLowering<CmpFOp> {
public:
  using SPIRVOpLowering<CmpFOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts integer compare operation to SPIR-V ops.
class CmpIOpPattern final : public SPIRVOpLowering<CmpIOp> {
public:
  using SPIRVOpLowering<CmpIOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.load to spv.Load.
class LoadOpPattern final : public SPIRVOpLowering<LoadOp> {
public:
  using SPIRVOpLowering<LoadOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.return to spv.Return.
class ReturnOpPattern final : public SPIRVOpLowering<ReturnOp> {
public:
  using SPIRVOpLowering<ReturnOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.select to spv.Select.
class SelectOpPattern final : public SPIRVOpLowering<SelectOp> {
public:
  using SPIRVOpLowering<SelectOp>::SPIRVOpLowering;
  LogicalResult
  matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts std.store to spv.Store.
class StoreOpPattern final : public SPIRVOpLowering<StoreOp> {
public:
  using SPIRVOpLowering<StoreOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Converts type-casting standard operations to SPIR-V operations.
template <typename StdOp, typename SPIRVOp>
class TypeCastingOpPattern final : public SPIRVOpLowering<StdOp> {
public:
  using SPIRVOpLowering<StdOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    auto dstType =
        this->typeConverter.convertType(operation.getResult().getType());
    if (dstType == operands.front().getType()) {
      // Due to type conversion, we are seeing the same source and target type.
      // Then we can just erase this operation by forwarding its operand.
      rewriter.replaceOp(operation, operands.front());
    } else {
      rewriter.template replaceOpWithNewOp<SPIRVOp>(
          operation, dstType, operands, ArrayRef<NamedAttribute>());
    }
    return success();
  }
};

/// Converts std.xor to SPIR-V operations.
class XOrOpPattern final : public SPIRVOpLowering<XOrOp> {
public:
  using SPIRVOpLowering<XOrOp>::SPIRVOpLowering;

  LogicalResult
  matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOp with composite type.
//===----------------------------------------------------------------------===//

LogicalResult ConstantCompositeOpPattern::matchAndRewrite(
    ConstantOp constCompositeOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto compositeType =
      constCompositeOp.getResult().getType().dyn_cast<RankedTensorType>();
  if (!compositeType)
    return failure();

  auto spirvCompositeType = typeConverter.convertType(compositeType);
  if (!spirvCompositeType)
    return failure();

  auto linearizedElements =
      constCompositeOp.value().dyn_cast<DenseElementsAttr>();
  if (!linearizedElements)
    return failure();

  // If composite type has rank greater than one, then perform linearization.
  if (compositeType.getRank() > 1) {
    auto linearizedType = RankedTensorType::get(compositeType.getNumElements(),
                                                compositeType.getElementType());
    linearizedElements = linearizedElements.reshape(linearizedType);
  }

  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
      constCompositeOp, spirvCompositeType, linearizedElements);
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp with scalar type.
//===----------------------------------------------------------------------===//

LogicalResult ConstantScalarOpPattern::matchAndRewrite(
    ConstantOp constIndexOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!constIndexOp.getResult().getType().isa<IndexType>()) {
    return failure();
  }
  // The attribute has index type which is not directly supported in
  // SPIR-V. Get the integer value and create a new IntegerAttr.
  auto constAttr = constIndexOp.value().dyn_cast<IntegerAttr>();
  if (!constAttr) {
    return failure();
  }

  // Use the bitwidth set in the value attribute to decide the result type
  // of the SPIR-V constant operation since SPIR-V does not support index
  // types.
  auto constVal = constAttr.getValue();
  auto constValType = constAttr.getType().dyn_cast<IndexType>();
  if (!constValType) {
    return failure();
  }
  auto spirvConstType =
      typeConverter.convertType(constIndexOp.getResult().getType());
  auto spirvConstVal =
      rewriter.getIntegerAttr(spirvConstType, constAttr.getInt());
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constIndexOp, spirvConstType,
                                                 spirvConstVal);
  return success();
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpFOpPattern::matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  CmpFOpOperandAdaptor cmpFOpOperands(operands);

  switch (cmpFOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpFOp, cmpFOp.getResult().getType(), \
                                         cmpFOpOperands.lhs(),                 \
                                         cmpFOpOperands.rhs());                \
    return success();

    // Ordered.
    DISPATCH(CmpFPredicate::OEQ, spirv::FOrdEqualOp);
    DISPATCH(CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
    DISPATCH(CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::OLT, spirv::FOrdLessThanOp);
    DISPATCH(CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
    DISPATCH(CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
    // Unordered.
    DISPATCH(CmpFPredicate::UEQ, spirv::FUnordEqualOp);
    DISPATCH(CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
    DISPATCH(CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::ULT, spirv::FUnordLessThanOp);
    DISPATCH(CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
    DISPATCH(CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

  default:
    break;
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

LogicalResult
CmpIOpPattern::matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  CmpIOpOperandAdaptor cmpIOpOperands(operands);

  switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpIOp, cmpIOp.getResult().getType(), \
                                         cmpIOpOperands.lhs(),                 \
                                         cmpIOpOperands.rhs());                \
    return success();

    DISPATCH(CmpIPredicate::eq, spirv::IEqualOp);
    DISPATCH(CmpIPredicate::ne, spirv::INotEqualOp);
    DISPATCH(CmpIPredicate::slt, spirv::SLessThanOp);
    DISPATCH(CmpIPredicate::sle, spirv::SLessThanEqualOp);
    DISPATCH(CmpIPredicate::sgt, spirv::SGreaterThanOp);
    DISPATCH(CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
    DISPATCH(CmpIPredicate::ult, spirv::ULessThanOp);
    DISPATCH(CmpIPredicate::ule, spirv::ULessThanEqualOp);
    DISPATCH(CmpIPredicate::ugt, spirv::UGreaterThanOp);
    DISPATCH(CmpIPredicate::uge, spirv::UGreaterThanEqualOp);

#undef DISPATCH
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

LogicalResult
LoadOpPattern::matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  LoadOpOperandAdaptor loadOperands(operands);
  auto loadPtr = spirv::getElementPtr(
      typeConverter, loadOp.memref().getType().cast<MemRefType>(),
      loadOperands.memref(), loadOperands.indices(), loadOp.getLoc(), rewriter);
  rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, loadPtr);
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult
ReturnOpPattern::matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  if (returnOp.getNumOperands()) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult
SelectOpPattern::matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  SelectOpOperandAdaptor selectOperands(operands);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, selectOperands.condition(),
                                               selectOperands.true_value(),
                                               selectOperands.false_value());
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult
StoreOpPattern::matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
  StoreOpOperandAdaptor storeOperands(operands);
  auto storePtr = spirv::getElementPtr(
      typeConverter, storeOp.memref().getType().cast<MemRefType>(),
      storeOperands.memref(), storeOperands.indices(), storeOp.getLoc(),
      rewriter);
  rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, storePtr,
                                              storeOperands.value());
  return success();
}

//===----------------------------------------------------------------------===//
// XorOp
//===----------------------------------------------------------------------===//

LogicalResult
XOrOpPattern::matchAndRewrite(XOrOp xorOp, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) const {
  assert(operands.size() == 2);

  if (isBoolScalarOrVector(operands.front().getType()))
    return failure();

  auto dstType = typeConverter.convertType(xorOp.getType());
  if (!dstType)
    return failure();
  rewriter.replaceOpWithNewOp<spirv::BitwiseXorOp>(xorOp, dstType, operands,
                                                   ArrayRef<NamedAttribute>());

  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

namespace {
/// Import the Standard Ops to SPIR-V Patterns.
#include "StandardToSPIRV.cpp.inc"
} // namespace

namespace mlir {
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     SPIRVTypeConverter &typeConverter,
                                     OwningRewritePatternList &patterns) {
  // Add patterns that lower operations into SPIR-V dialect.
  populateWithGenerated(context, &patterns);
  patterns.insert<
      BinaryOpPattern<AddFOp, spirv::FAddOp>,
      BinaryOpPattern<AddIOp, spirv::IAddOp>,
      BinaryOpPattern<DivFOp, spirv::FDivOp>,
      BinaryOpPattern<MulFOp, spirv::FMulOp>,
      BinaryOpPattern<MulIOp, spirv::IMulOp>,
      BinaryOpPattern<RemFOp, spirv::FRemOp>,
      BinaryOpPattern<ShiftLeftOp, spirv::ShiftLeftLogicalOp>,
      BinaryOpPattern<SignedShiftRightOp, spirv::ShiftRightArithmeticOp>,
      BinaryOpPattern<SignedDivIOp, spirv::SDivOp>,
      BinaryOpPattern<SignedRemIOp, spirv::SRemOp>,
      BinaryOpPattern<SubFOp, spirv::FSubOp>,
      BinaryOpPattern<SubIOp, spirv::ISubOp>,
      BinaryOpPattern<UnsignedDivIOp, spirv::UDivOp>,
      BinaryOpPattern<UnsignedRemIOp, spirv::UModOp>,
      BinaryOpPattern<UnsignedShiftRightOp, spirv::ShiftRightLogicalOp>,
      BitwiseOpPattern<AndOp, spirv::LogicalAndOp, spirv::BitwiseAndOp>,
      BitwiseOpPattern<OrOp, spirv::LogicalOrOp, spirv::BitwiseOrOp>,
      ConstantCompositeOpPattern, ConstantScalarOpPattern, CmpFOpPattern,
      CmpIOpPattern, LoadOpPattern, ReturnOpPattern, SelectOpPattern,
      StoreOpPattern, TypeCastingOpPattern<SIToFPOp, spirv::ConvertSToFOp>,
      TypeCastingOpPattern<FPExtOp, spirv::FConvertOp>,
      TypeCastingOpPattern<FPTruncOp, spirv::FConvertOp>, XOrOpPattern>(
      context, typeConverter);
}
} // namespace mlir
