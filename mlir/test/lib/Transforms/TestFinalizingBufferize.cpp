//===- TestFinalizingBufferize.cpp - Finalizing bufferization ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that exercises the functionality of finalizing
// bufferizations.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Bufferize.h"

using namespace mlir;

namespace {
/// This pass is a test for "finalizing" bufferize conversions.
///
/// A "finalizing" bufferize conversion is one that performs a "full" conversion
/// and expects all tensors to be gone from the program. This in particular
/// involves rewriting funcs (including block arguments of the contained
/// region), calls, and returns. The unique property of finalizing bufferization
/// passes is that they cannot be done via a local transformation with suitable
/// materializations to ensure composability (as other bufferization passes do).
/// For example, if a call is rewritten, the callee needs to be rewritten
/// otherwise the IR will end up invalid. Thus, finalizing bufferization passes
/// require an atomic change to the entire program (e.g. the whole module).
///
/// `allowMemrefFunctionResults` informs the buffer finalization policy to allow
/// functions that have memref typed results. Patterns involved with converting
/// func/call/return respect the finalization policy to ensure a consistent
/// atomic conversion of the entire module. `allowMemrefFunctionResults` also
/// allows memref typed results to escape from the deallocation.
///
/// TODO: Split out BufferizeFinalizationPolicy from BufferizeTypeConverter.
template <bool allowMemrefFunctionResults>
struct TestFinalizingBufferizePass
    : mlir::PassWrapper<TestFinalizingBufferizePass<allowMemrefFunctionResults>,
                        OperationPass<ModuleOp>> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TestDialect>();
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = this->getContext();
    ConversionTarget target(context);
    BufferizeTypeConverter converter;

    // Mark all Standard operations legal.
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<linalg::CopyOp>();
    target.addLegalOp<MakeTupleOp>();
    target.addLegalOp<GetTupleElementOp>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<ModuleTerminatorOp>();

    // Mark Standard Return operations illegal as long as one operand is tensor.
    target.addDynamicallyLegalOp<mlir::ReturnOp>([&](mlir::ReturnOp returnOp) {
      return converter.isLegal(returnOp.getOperandTypes());
    });

    // Mark Standard Call Operation illegal as long as it operates on tensor.
    target.addDynamicallyLegalOp<mlir::CallOp>(
        [&](mlir::CallOp callOp) { return converter.isLegal(callOp); });

    // Mark the function whose arguments are in tensor-type illegal.
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType()) &&
             converter.isLegal(&funcOp.getBody());
    });

    auto kind = allowMemrefFunctionResults
                    ? BufferizeTypeConverter::KeepAsFunctionResult
                    : BufferizeTypeConverter::AppendToArgumentsList;
    converter.setResultConversionKind<RankedTensorType, MemRefType>(kind);
    converter.setResultConversionKind<UnrankedTensorType, UnrankedMemRefType>(
        kind);

    converter.addDecomposeTypeConversion(
        [](TupleType tupleType, SmallVectorImpl<Type> &types) {
          tupleType.getFlattenedTypes(types);
          return success();
        });

    converter.addArgumentMaterialization(
        [](OpBuilder &builder, TupleType resultType, ValueRange inputs,
           Location loc) -> Optional<Value> {
          if (inputs.size() == 1)
            return llvm::None;
          TypeRange TypeRange = inputs.getTypes();
          SmallVector<Type, 2> types(TypeRange.begin(), TypeRange.end());
          TupleType tuple = TupleType::get(types, builder.getContext());
          mlir::Value value = builder.create<MakeTupleOp>(loc, tuple, inputs);
          return value;
        });

    converter.addDecomposeValueConversion([](OpBuilder &builder, Location loc,
                                             TupleType resultType, Value value,
                                             SmallVectorImpl<Value> &values) {
      for (unsigned i = 0, e = resultType.size(); i < e; ++i) {
        Value res = builder.create<GetTupleElementOp>(
            loc, resultType.getType(i), value, builder.getI32IntegerAttr(i));
        values.push_back(res);
      }
      return success();
    });

    OwningRewritePatternList patterns;
    populateWithBufferizeOpConversionPatterns<mlir::ReturnOp, mlir::ReturnOp,
                                              linalg::CopyOp>(
        &context, converter, patterns);
    if (failed(applyFullConversion(this->getOperation(), target,
                                   std::move(patterns))))
      this->signalPassFailure();
  };
};
} // end anonymous namespace

namespace mlir {
void registerTestFinalizingBufferizePass() {
  PassRegistration<
      TestFinalizingBufferizePass</*allowMemrefFunctionResults=*/false>>(
      "test-finalizing-bufferize", "Tests finalizing bufferize conversions");
}

void registerTestPreparationPassWithAllowedMemrefResults() {
  PassRegistration<
      TestFinalizingBufferizePass</*allowMemrefFunctionResults=*/true>>(
      "test-finalizing-bufferize-with-allowed-memref-results",
      "Tests finalizing buffierize conversions, allowing functions to have "
      "memref typed results.");
}
} // end namespace mlir
