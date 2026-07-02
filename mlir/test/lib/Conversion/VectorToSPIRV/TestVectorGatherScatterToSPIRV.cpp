//===- TestVectorGatherScatterToSPIRV.cpp - Test gather/scatter ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace {

struct TestVectorGatherScatterToSPIRV
    : PassWrapper<TestVectorGatherScatterToSPIRV, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestVectorGatherScatterToSPIRV)

  StringRef getArgument() const final {
    return "test-vector-gather-scatter-to-spirv";
  }

  StringRef getDescription() const final {
    return "Test lowering of vector.gather/vector.scatter to "
           "spirv.INTEL.MaskedGather/MaskedScatter "
           "(SPV_INTEL_masked_gather_scatter)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, spirv::SPIRVDialect,
                    ub::UBDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *op = getOperation();

    auto targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    SPIRVTypeConverter typeConverter(targetAttr);

    target->addLegalOp<UnrealizedConversionCastOp>();

    RewritePatternSet patterns(context);
    populateVectorToSPIRVPatterns(typeConverter, patterns);
    populateVectorGatherScatterToSPIRVPatterns(typeConverter, patterns);
    ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace test {
void registerTestVectorGatherScatterToSPIRV() {
  PassRegistration<TestVectorGatherScatterToSPIRV>();
}
} // namespace test
} // namespace mlir
