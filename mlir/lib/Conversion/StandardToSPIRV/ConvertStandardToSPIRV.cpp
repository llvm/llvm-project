//===- ConvertStandardToSPIRV.cpp - Standard to SPIR-V dialect conversion--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Standard Ops to the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Convert composite constant operation to SPIR-V dialect.
// TODO(denis0x0D) : move to DRR.
class ConstantCompositeOpConversion final : public SPIRVOpLowering<ConstantOp> {
public:
  using SPIRVOpLowering<ConstantOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ConstantOp constCompositeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert constant operation with IndexType return to SPIR-V constant
/// operation. Since IndexType is not used within SPIR-V dialect, this needs
/// special handling to make sure the result type and the type of the value
/// attribute are consistent.
// TODO(ravishankarm) : This should be moved into DRR.
class ConstantIndexOpConversion final : public SPIRVOpLowering<ConstantOp> {
public:
  using SPIRVOpLowering<ConstantOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ConstantOp constIndexOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert floating-point comparison operations to SPIR-V dialect.
class CmpFOpConversion final : public SPIRVOpLowering<CmpFOp> {
public:
  using SPIRVOpLowering<CmpFOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert compare operation to SPIR-V dialect.
class CmpIOpConversion final : public SPIRVOpLowering<CmpIOp> {
public:
  using SPIRVOpLowering<CmpIOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert integer binary operations to SPIR-V operations. Cannot use
/// tablegen for this. If the integer operation is on variables of IndexType,
/// the type of the return value of the replacement operation differs from
/// that of the replaced operation. This is not handled in tablegen-based
/// pattern specification.
// TODO(ravishankarm) : This should be moved into DRR.
template <typename StdOp, typename SPIRVOp>
class IntegerOpConversion final : public SPIRVOpLowering<StdOp> {
public:
  using SPIRVOpLowering<StdOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(StdOp operation, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        this->typeConverter.convertType(operation.getResult().getType());
    rewriter.template replaceOpWithNewOp<SPIRVOp>(
        operation, resultType, operands, ArrayRef<NamedAttribute>());
    return this->matchSuccess();
  }
};

/// Convert load -> spv.LoadOp. The operands of the replaced operation are of
/// IndexType while that of the replacement operation are of type i32. This is
/// not supported in tablegen based pattern specification.
// TODO(ravishankarm) : This should be moved into DRR.
class LoadOpConversion final : public SPIRVOpLowering<LoadOp> {
public:
  using SPIRVOpLowering<LoadOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert return -> spv.Return.
// TODO(ravishankarm) : This should be moved into DRR.
class ReturnOpConversion final : public SPIRVOpLowering<ReturnOp> {
public:
  using SPIRVOpLowering<ReturnOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert select -> spv.Select
// TODO(ravishankarm) : This should be moved into DRR.
class SelectOpConversion final : public SPIRVOpLowering<SelectOp> {
public:
  using SPIRVOpLowering<SelectOp>::SPIRVOpLowering;
  PatternMatchResult
  matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Convert store -> spv.StoreOp. The operands of the replaced operation are
/// of IndexType while that of the replacement operation are of type i32. This
/// is not supported in tablegen based pattern specification.
// TODO(ravishankarm) : This should be moved into DRR.
class StoreOpConversion final : public SPIRVOpLowering<StoreOp> {
public:
  using SPIRVOpLowering<StoreOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// ConstantOp with composite type.
//===----------------------------------------------------------------------===//

PatternMatchResult ConstantCompositeOpConversion::matchAndRewrite(
    ConstantOp constCompositeOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto compositeType =
      constCompositeOp.getResult().getType().dyn_cast<RankedTensorType>();
  if (!compositeType)
    return matchFailure();

  auto spirvCompositeType = typeConverter.convertType(compositeType);
  if (!spirvCompositeType)
    return matchFailure();

  auto linearizedElements =
      constCompositeOp.value().dyn_cast<DenseElementsAttr>();
  if (!linearizedElements)
    return matchFailure();

  // If composite type has rank greater than one, then perform linearization.
  if (compositeType.getRank() > 1) {
    auto linearizedType = RankedTensorType::get(compositeType.getNumElements(),
                                                compositeType.getElementType());
    linearizedElements = linearizedElements.reshape(linearizedType);
  }

  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(
      constCompositeOp, spirvCompositeType, linearizedElements);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ConstantOp with index type.
//===----------------------------------------------------------------------===//

PatternMatchResult ConstantIndexOpConversion::matchAndRewrite(
    ConstantOp constIndexOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!constIndexOp.getResult().getType().isa<IndexType>()) {
    return matchFailure();
  }
  // The attribute has index type which is not directly supported in
  // SPIR-V. Get the integer value and create a new IntegerAttr.
  auto constAttr = constIndexOp.value().dyn_cast<IntegerAttr>();
  if (!constAttr) {
    return matchFailure();
  }

  // Use the bitwidth set in the value attribute to decide the result type
  // of the SPIR-V constant operation since SPIR-V does not support index
  // types.
  auto constVal = constAttr.getValue();
  auto constValType = constAttr.getType().dyn_cast<IndexType>();
  if (!constValType) {
    return matchFailure();
  }
  auto spirvConstType =
      typeConverter.convertType(constIndexOp.getResult().getType());
  auto spirvConstVal =
      rewriter.getIntegerAttr(spirvConstType, constAttr.getInt());
  rewriter.replaceOpWithNewOp<spirv::ConstantOp>(constIndexOp, spirvConstType,
                                                 spirvConstVal);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//

PatternMatchResult
CmpFOpConversion::matchAndRewrite(CmpFOp cmpFOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  CmpFOpOperandAdaptor cmpFOpOperands(operands);

  switch (cmpFOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpFOp, cmpFOp.getResult().getType(), \
                                         cmpFOpOperands.lhs(),                 \
                                         cmpFOpOperands.rhs());                \
    return matchSuccess();

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
  return matchFailure();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

PatternMatchResult
CmpIOpConversion::matchAndRewrite(CmpIOp cmpIOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  CmpIOpOperandAdaptor cmpIOpOperands(operands);

  switch (cmpIOp.getPredicate()) {
#define DISPATCH(cmpPredicate, spirvOp)                                        \
  case cmpPredicate:                                                           \
    rewriter.replaceOpWithNewOp<spirvOp>(cmpIOp, cmpIOp.getResult().getType(), \
                                         cmpIOpOperands.lhs(),                 \
                                         cmpIOpOperands.rhs());                \
    return matchSuccess();

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
  return matchFailure();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

PatternMatchResult
LoadOpConversion::matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  LoadOpOperandAdaptor loadOperands(operands);
  auto loadPtr = spirv::getElementPtr(
      typeConverter, loadOp.memref().getType().cast<MemRefType>(),
      loadOperands.memref(), loadOperands.indices(), loadOp.getLoc(), rewriter);
  rewriter.replaceOpWithNewOp<spirv::LoadOp>(loadOp, loadPtr);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

PatternMatchResult
ReturnOpConversion::matchAndRewrite(ReturnOp returnOp, ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  if (returnOp.getNumOperands()) {
    return matchFailure();
  }
  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

PatternMatchResult
SelectOpConversion::matchAndRewrite(SelectOp op, ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
  SelectOpOperandAdaptor selectOperands(operands);
  rewriter.replaceOpWithNewOp<spirv::SelectOp>(op, selectOperands.condition(),
                                               selectOperands.true_value(),
                                               selectOperands.false_value());
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

PatternMatchResult
StoreOpConversion::matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                                   ConversionPatternRewriter &rewriter) const {
  StoreOpOperandAdaptor storeOperands(operands);
  auto storePtr = spirv::getElementPtr(
      typeConverter, storeOp.memref().getType().cast<MemRefType>(),
      storeOperands.memref(), storeOperands.indices(), storeOp.getLoc(),
      rewriter);
  rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, storePtr,
                                              storeOperands.value());
  return matchSuccess();
}

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
  patterns.insert<ConstantCompositeOpConversion, ConstantIndexOpConversion,
                  CmpFOpConversion, CmpIOpConversion,
                  IntegerOpConversion<AddIOp, spirv::IAddOp>,
                  IntegerOpConversion<MulIOp, spirv::IMulOp>,
                  IntegerOpConversion<SignedDivIOp, spirv::SDivOp>,
                  IntegerOpConversion<SignedRemIOp, spirv::SModOp>,
                  IntegerOpConversion<SubIOp, spirv::ISubOp>, LoadOpConversion,
                  ReturnOpConversion, SelectOpConversion, StoreOpConversion>(
      context, typeConverter);
}
} // namespace mlir
