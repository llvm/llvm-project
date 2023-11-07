//===- PartTensorPasses.cpp - Pass for autogen part tensor code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include "mlir/Dialect/PartTensor/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_PARTTENSORCONVERSIONPASS
#include "mlir/Dialect/PartTensor/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::part_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Passes implementation.
//===----------------------------------------------------------------------===//

struct PartTensorConversionPass
    : public impl::PartTensorConversionPassBase<PartTensorConversionPass> {

  PartTensorConversionPass() = default;
  PartTensorConversionPass(const PartTensorConversionPass &pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    PartTensorTypeToPtrConverter converter;
    ConversionTarget target(*ctx);
    // Everything in the part dialect must go!
    //  target.addIllegalDialect<PartTensorDialect>();
    //  // All dynamic rules below accept new function, call, return, and
    //  various
    //  // tensor and bufferization operations as legal output of the rewriting
    //  // provided that all part tensor types have been fully rewritten.
    //  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    //    return converter.isSignatureLegal(op.getFunctionType());
    //  });
    //  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    //    return converter.isSignatureLegal(op.getCalleeType());
    //  });
    //  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    //    return converter.isLegal(op.getOperandTypes());
    //  });
    //  target.addDynamicallyLegalOp<tensor::DimOp>([&](tensor::DimOp op) {
    //    return converter.isLegal(op.getOperandTypes());
    //  });
    //  target.addDynamicallyLegalOp<tensor::CastOp>([&](tensor::CastOp op) {
    //    return converter.isLegal(op.getSource().getType()) &&
    //           converter.isLegal(op.getDest().getType());
    //  });
    //  target.addDynamicallyLegalOp<tensor::ExpandShapeOp>(
    //      [&](tensor::ExpandShapeOp op) {
    //        return converter.isLegal(op.getSrc().getType()) &&
    //               converter.isLegal(op.getResult().getType());
    //      });
    //  target.addDynamicallyLegalOp<tensor::CollapseShapeOp>(
    //      [&](tensor::CollapseShapeOp op) {
    //        return converter.isLegal(op.getSrc().getType()) &&
    //               converter.isLegal(op.getResult().getType());
    //      });
    //  target.addDynamicallyLegalOp<bufferization::AllocTensorOp>(
    //      [&](bufferization::AllocTensorOp op) {
    //        return converter.isLegal(op.getType());
    //      });
    //  target.addDynamicallyLegalOp<bufferization::DeallocTensorOp>(
    //      [&](bufferization::DeallocTensorOp op) {
    //        return converter.isLegal(op.getTensor().getType());
    //      });
    //  // The following operations and dialects may be introduced by the
    //  // rewriting rules, and are therefore marked as legal.
    //  target.addLegalOp<complex::ConstantOp, complex::NotEqualOp,
    //  linalg::FillOp,
    //                    linalg::YieldOp, tensor::ExtractOp>();
    //  target.addLegalDialect<
    //      arith::ArithDialect, bufferization::BufferizationDialect,
    //      LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect>();
    //  // Populate with rules and apply rewriting rules.
    //  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
    //                                                                 converter);
    //  populateCallOpTypeConversionPattern(patterns, converter);
    //  scf::populateSCFStructuralTypeConversionsAndLegality(converter,
    //  patterns,
    //                                                       target);
    populatePartTensorConversionPatterns(converter, patterns);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createPartTensorConversionPass() {
  return std::make_unique<PartTensorConversionPass>();
}
