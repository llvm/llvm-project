//===- ConvertToSPIRVPass.cpp - MLIR SPIR-V Conversion --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/IndexToSPIRV/IndexToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/UBToSPIRV/UBToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

#define DEBUG_TYPE "test-convert-to-spirv"

using namespace mlir;

namespace {

/// Map memRef memory space to SPIR-V storage class.
void mapToMemRef(Operation *op, spirv::TargetEnvAttr &targetAttr) {
  spirv::TargetEnv targetEnv(targetAttr);
  bool targetEnvSupportsKernelCapability =
      targetEnv.allows(spirv::Capability::Kernel);
  spirv::MemorySpaceToStorageClassMap memorySpaceMap =
      targetEnvSupportsKernelCapability
          ? spirv::mapMemorySpaceToOpenCLStorageClass
          : spirv::mapMemorySpaceToVulkanStorageClass;
  spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);
  spirv::convertMemRefTypesAndAttrs(op, converter);
}

/// Populate patterns for each dialect.
void populateConvertToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                    ScfToSPIRVContext &scfToSPIRVContext,
                                    RewritePatternSet &patterns) {
  arith::populateCeilFloorDivExpandOpsPatterns(patterns);
  arith::populateArithToSPIRVPatterns(typeConverter, patterns);
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateGPUToSPIRVPatterns(typeConverter, patterns);
  index::populateIndexToSPIRVPatterns(typeConverter, patterns);
  populateMemRefToSPIRVPatterns(typeConverter, patterns);
  populateVectorToSPIRVPatterns(typeConverter, patterns);
  populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);
  ub::populateUBToSPIRVConversionPatterns(typeConverter, patterns);
}

/// A pass to perform the SPIR-V conversion.
struct TestConvertToSPIRVPass final
    : PassWrapper<TestConvertToSPIRVPass, OperationPass<>> {
  Option<bool> runSignatureConversion{
      *this, "run-signature-conversion",
      llvm::cl::desc(
          "Run function signature conversion to convert vector types"),
      llvm::cl::init(true)};
  Option<bool> runVectorUnrolling{
      *this, "run-vector-unrolling",
      llvm::cl::desc(
          "Run vector unrolling to convert vector types in function bodies"),
      llvm::cl::init(true)};
  Option<bool> convertGPUModules{
      *this, "convert-gpu-modules",
      llvm::cl::desc("Clone and convert GPU modules"), llvm::cl::init(false)};
  Option<bool> nestInGPUModule{
      *this, "nest-in-gpu-module",
      llvm::cl::desc("Put converted SPIR-V module inside the gpu.module "
                     "instead of alongside it."),
      llvm::cl::init(false)};

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestConvertToSPIRVPass)

  StringRef getArgument() const final { return "test-convert-to-spirv"; }
  StringRef getDescription() const final {
    return "Conversion to SPIR-V pass only used for internal tests.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
    registry.insert<vector::VectorDialect>();
  }

  TestConvertToSPIRVPass() = default;
  TestConvertToSPIRVPass(bool convertGPUModules, bool nestInGPUModule) {
    this->convertGPUModules = convertGPUModules;
    this->nestInGPUModule = nestInGPUModule;
  };
  TestConvertToSPIRVPass(const TestConvertToSPIRVPass &) {}

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = &getContext();

    // Unroll vectors in function signatures to native size.
    if (runSignatureConversion && failed(spirv::unrollVectorsInSignatures(op)))
      return signalPassFailure();

    // Unroll vectors in function bodies to native size.
    if (runVectorUnrolling && failed(spirv::unrollVectorsInFuncBodies(op)))
      return signalPassFailure();

    // Generic conversion.
    if (!convertGPUModules) {
      spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
      std::unique_ptr<ConversionTarget> target =
          SPIRVConversionTarget::get(targetAttr);
      SPIRVTypeConverter typeConverter(targetAttr);
      RewritePatternSet patterns(context);
      ScfToSPIRVContext scfToSPIRVContext;
      mapToMemRef(op, targetAttr);
      populateConvertToSPIRVPatterns(typeConverter, scfToSPIRVContext,
                                     patterns);
      if (failed(applyPartialConversion(op, *target, std::move(patterns))))
        return signalPassFailure();
      return;
    }

    // Clone each GPU kernel module for conversion, given that the GPU
    // launch op still needs the original GPU kernel module.
    SmallVector<Operation *, 1> gpuModules;
    OpBuilder builder(context);
    op->walk([&](gpu::GPUModuleOp gpuModule) {
      if (nestInGPUModule)
        builder.setInsertionPointToStart(gpuModule.getBody());
      else
        builder.setInsertionPoint(gpuModule);
      gpuModules.push_back(builder.clone(*gpuModule));
    });
    // Run conversion for each module independently as they can have
    // different TargetEnv attributes.
    for (Operation *gpuModule : gpuModules) {
      spirv::TargetEnvAttr targetAttr =
          spirv::lookupTargetEnvOrDefault(gpuModule);
      std::unique_ptr<ConversionTarget> target =
          SPIRVConversionTarget::get(targetAttr);
      SPIRVTypeConverter typeConverter(targetAttr);
      RewritePatternSet patterns(context);
      ScfToSPIRVContext scfToSPIRVContext;
      mapToMemRef(gpuModule, targetAttr);
      populateConvertToSPIRVPatterns(typeConverter, scfToSPIRVContext,
                                     patterns);
      if (failed(applyFullConversion(gpuModule, *target, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::test {
void registerTestConvertToSPIRVPass() {
  PassRegistration<TestConvertToSPIRVPass>();
}
std::unique_ptr<Pass> createTestConvertToSPIRVPass(bool convertGPUModules,
                                                   bool nestInGPUModule) {
  return std::make_unique<TestConvertToSPIRVPass>(convertGPUModules,
                                                  nestInGPUModule);
}
} // namespace mlir::test
