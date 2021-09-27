//===- LinalgToSPIRV.cpp - Linalg to SPIR-V Patterns ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToSPIRV/LinalgToSPIRV.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Returns a `Value` containing the `dim`-th dimension's size of SPIR-V
/// location invocation ID. This function will create necessary operations with
/// `builder` at the proper region containing `op`.
static Value getLocalInvocationDimSize(Operation *op, int dim, Type integerType,
                                       Location loc, OpBuilder *builder) {
  assert(dim >= 0 && dim < 3 && "local invocation only has three dimensions");
  Value invocation = spirv::getBuiltinVariableValue(
      op, spirv::BuiltIn::LocalInvocationId, integerType, *builder);
  Type xType = invocation.getType().cast<ShapedType>().getElementType();
  return builder->create<spirv::CompositeExtractOp>(
      loc, xType, invocation, builder->getI32ArrayAttr({dim}));
}

//===----------------------------------------------------------------------===//
// Reduction (single workgroup)
//===----------------------------------------------------------------------===//

namespace {

/// A pattern to convert a linalg.generic op to SPIR-V ops under the condition
/// that the linalg.generic op is performing reduction with a workload size that
/// can fit in one workgroup.
struct SingleWorkgroupReduction final
    : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern::OpConversionPattern;

  /// Matches the given linalg.generic op as performing reduction and returns
  /// the binary op kind if successful.
  static Optional<linalg::RegionMatcher::BinaryOpKind>
  matchAsPerformingReduction(linalg::GenericOp genericOp);

  LogicalResult
  matchAndRewrite(linalg::GenericOp genericOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

Optional<linalg::RegionMatcher::BinaryOpKind>
SingleWorkgroupReduction::matchAsPerformingReduction(
    linalg::GenericOp genericOp) {
  Operation *op = genericOp.getOperation();

  // Make sure the linalg.generic is working on memrefs.
  if (!genericOp.hasBufferSemantics())
    return llvm::None;

  // Make sure this is reduction with one input and one output.
  if (genericOp.getNumInputs() != 1 || genericOp.getNumOutputs() != 1)
    return llvm::None;

  auto originalInputType = op->getOperand(0).getType().cast<MemRefType>();
  auto originalOutputType = op->getOperand(1).getType().cast<MemRefType>();

  // Make sure the original input has one dimension.
  if (!originalInputType.hasStaticShape() || originalInputType.getRank() != 1)
    return llvm::None;
  // Make sure the original output has one element.
  if (!originalOutputType.hasStaticShape() ||
      originalOutputType.getNumElements() != 1)
    return llvm::None;

  if (!genericOp.hasSingleReductionLoop())
    return llvm::None;

  if (genericOp.indexing_maps().getValue().size() != 2)
    return llvm::None;

  // TODO: create utility functions for these checks in Linalg
  // and use them.
  auto inputMap = genericOp.indexing_maps().getValue()[0].cast<AffineMapAttr>();
  auto outputMap =
      genericOp.indexing_maps().getValue()[1].cast<AffineMapAttr>();
  // The indexing map for the input should be `(i) -> (i)`.
  if (inputMap.getValue() !=
      AffineMap::get(1, 0, getAffineDimExpr(0, op->getContext())))
    return llvm::None;
  // The indexing map for the input should be `(i) -> (0)`.
  if (outputMap.getValue() !=
      AffineMap::get(1, 0, getAffineConstantExpr(0, op->getContext())))
    return llvm::None;

  return linalg::RegionMatcher::matchAsScalarBinaryOp(genericOp);
}

LogicalResult SingleWorkgroupReduction::matchAndRewrite(
    linalg::GenericOp genericOp, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Operation *op = genericOp.getOperation();
  auto originalInputType = op->getOperand(0).getType().cast<MemRefType>();
  auto originalOutputType = op->getOperand(1).getType().cast<MemRefType>();

  auto binaryOpKind = matchAsPerformingReduction(genericOp);
  if (!binaryOpKind)
    return failure();

  // Query the shader interface for local workgroup size to make sure the
  // invocation configuration fits with the input memref's shape.
  DenseIntElementsAttr localSize = spirv::lookupLocalWorkGroupSize(genericOp);
  if (!localSize)
    return failure();

  if ((*localSize.begin()).getSExtValue() != originalInputType.getDimSize(0))
    return failure();
  if (llvm::any_of(llvm::drop_begin(localSize.getValues<APInt>(), 1),
                   [](const APInt &size) { return !size.isOneValue(); }))
    return failure();

  // TODO: Query the target environment to make sure the current
  // workload fits in a local workgroup.

  Value convertedInput = adaptor.getOperands()[0];
  Value convertedOutput = adaptor.getOperands()[1];
  Location loc = genericOp.getLoc();

  auto *typeConverter = getTypeConverter<SPIRVTypeConverter>();
  auto indexType = typeConverter->getIndexType();

  // Get the invocation ID.
  Value x = getLocalInvocationDimSize(genericOp, /*dim=*/0, indexType, loc,
                                      &rewriter);

  // TODO: Load to Workgroup storage class first.


  // Get the input element accessed by this invocation.
  Value inputElementPtr = spirv::getElementPtr(
      *typeConverter, originalInputType, convertedInput, {x}, loc, rewriter);
  Value inputElement = rewriter.create<spirv::LoadOp>(loc, inputElementPtr);

  // Perform the group reduction operation.
  Value groupOperation;
#define CREATE_GROUP_NON_UNIFORM_BIN_OP(opKind, spvOp)                         \
  case linalg::RegionMatcher::BinaryOpKind::opKind: {                          \
    groupOperation = rewriter.create<spirv::spvOp>(                            \
        loc, originalInputType.getElementType(), spirv::Scope::Subgroup,       \
        spirv::GroupOperation::Reduce, inputElement,                           \
        /*cluster_size=*/nullptr);                                             \
  } break
  switch (*binaryOpKind) {
    CREATE_GROUP_NON_UNIFORM_BIN_OP(IAdd, GroupNonUniformIAddOp);
  }
#undef CREATE_GROUP_NON_UNIFORM_BIN_OP

  // Get the output element accessed by this reduction.
  Value zero = spirv::ConstantOp::getZero(indexType, loc, rewriter);
  SmallVector<Value, 1> zeroIndices(originalOutputType.getRank(), zero);
  Value outputElementPtr =
      spirv::getElementPtr(*typeConverter, originalOutputType, convertedOutput,
                           zeroIndices, loc, rewriter);

  // Write out the final reduction result. This should be only conducted by one
  // invocation. We use spv.GroupNonUniformElect to find the invocation with the
  // lowest ID.
  //
  // ```
  // if (spv.GroupNonUniformElect) { output = ... }
  // ```

  Value condition = rewriter.create<spirv::GroupNonUniformElectOp>(
      loc, spirv::Scope::Subgroup);

  auto createAtomicOp = [&](OpBuilder &builder) {
#define CREATE_ATOMIC_BIN_OP(opKind, spvOp)                                    \
  case linalg::RegionMatcher::BinaryOpKind::opKind: {                          \
    builder.create<spirv::spvOp>(loc, outputElementPtr, spirv::Scope::Device,  \
                                 spirv::MemorySemantics::AcquireRelease,       \
                                 groupOperation);                              \
  } break
    switch (*binaryOpKind) { CREATE_ATOMIC_BIN_OP(IAdd, AtomicIAddOp); }
#undef CREATE_ATOMIC_BIN_OP
  };

  spirv::SelectionOp::createIfThen(loc, condition, createAtomicOp, rewriter);

  rewriter.eraseOp(genericOp);
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateLinalgToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  patterns.add<SingleWorkgroupReduction>(typeConverter, patterns.getContext());
}
