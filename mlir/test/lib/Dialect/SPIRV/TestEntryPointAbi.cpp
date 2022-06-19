//===- TestAvailability.cpp - Test pass for setting Entry point ABI info --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that sets the spv.entry_point_abi attribute on
// functions that are to be lowered as entry point functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Pass to set the spv.entry_point_abi
struct TestSpirvEntryPointABIPass
    : public PassWrapper<TestSpirvEntryPointABIPass,
                         OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSpirvEntryPointABIPass)

  StringRef getArgument() const final { return "test-spirv-entry-point-abi"; }
  StringRef getDescription() const final {
    return "Set the spv.entry_point_abi attribute on GPU kernel function "
           "within the "
           "module, intended for testing only";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }
  TestSpirvEntryPointABIPass() = default;
  TestSpirvEntryPointABIPass(const TestSpirvEntryPointABIPass &) {}
  void runOnOperation() override;

private:
  Pass::ListOption<int32_t> workgroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Workgroup size to use for all gpu.func kernels in the module, "
          "specified with x-dimension first, y-dimension next and z-dimension "
          "last. Unspecified dimensions will be set to 1"),
      llvm::cl::ZeroOrMore};
};
} // namespace

void TestSpirvEntryPointABIPass::runOnOperation() {
  gpu::GPUModuleOp gpuModule = getOperation();
  MLIRContext *context = &getContext();
  StringRef attrName = spirv::getEntryPointABIAttrName();
  for (gpu::GPUFuncOp gpuFunc : gpuModule.getOps<gpu::GPUFuncOp>()) {
    if (!gpu::GPUDialect::isKernel(gpuFunc) || gpuFunc->getAttr(attrName))
      continue;
    SmallVector<int32_t, 3> workgroupSizeVec(workgroupSize.begin(),
                                             workgroupSize.end());
    workgroupSizeVec.resize(3, 1);
    gpuFunc->setAttr(attrName,
                     spirv::getEntryPointABIAttr(workgroupSizeVec, context));
  }
}

namespace mlir {
void registerTestSpirvEntryPointABIPass() {
  PassRegistration<TestSpirvEntryPointABIPass>();
}
} // namespace mlir
