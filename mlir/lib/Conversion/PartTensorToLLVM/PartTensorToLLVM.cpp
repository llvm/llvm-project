//===- PartTensorToLLVM.cpp - conversion from PartTensor to LLVM dialect
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/PartTensorToLLVM/PartTensorToLLVM.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTPARTTENSORTOLLVMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::part_tensor;

template <typename T>
static Type getPtrToElementType(T containerType, LLVMTypeConverter &lowering) {
  return LLVMPointerType::get(
      lowering.convertType(containerType.getElementType()));
}

namespace {
// GetPartitionOp produces and LLVM::ReturnOp.
class GetPartitionOpConversion
    : public ConvertOpToLLVMPattern<part_tensor::GetPartitionOp> {
public:
  using ConvertOpToLLVMPattern<
      part_tensor::GetPartitionOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(part_tensor::GetPartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(false);
    auto module = op->getParentOfType<ModuleOp>();
    auto rtFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("get_partitions");
    if (!rtFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto rtFuncTy = LLVM::LLVMFunctionType::get(getVoidType(), {});
      rtFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                 "get_partitions", rtFuncTy);
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, rtFunc,
                                              adaptor.getOperands());
    return success();
  }
};
} // namespace

/// Populate the given list with patterns that convert from PartTensor to LLVM.
void mlir::populatePartTensorToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GetPartitionOpConversion>(converter);
}

namespace {
struct ConvertPartTensorToLLVMPass
    : public impl::ConvertPartTensorToLLVMPassBase<
          ConvertPartTensorToLLVMPass> {

  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void ConvertPartTensorToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // Convert to the LLVM IR dialect using the converter defined above.
  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions options(&getContext());
  options.useOpaquePointers = useOpaquePointers;
  LLVMTypeConverter converter(&getContext(), options);
  // populatePartTensorToLLVMConversionPatterns(converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);

  // mlir::populatePartTensorToLLVMConversionPatterns(converter, patterns);
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  // target.addLegalOp<llvm::CallOp>();
  patterns.add<GetPartitionOpConversion>(converter);
  applyFullConversion(module, target, std::move(patterns));
  // LLVMConversionTarget target(getContext());
  // target.addLegalOp<func::FuncOp>();
  // if (failed(applyPartialConversion(op, target, std::move(patterns))))
  //   signalPassFailure();
}
