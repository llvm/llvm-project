//===- TestConvertGPUKernelToHsaco.cpp - Test gpu kernel hsaco lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

#if MLIR_ROCM_CONVERSIONS_ENABLED
namespace {
class TestSerializeToHsacoPass
    : public PassWrapper<TestSerializeToHsacoPass, gpu::SerializeToBlobPass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSerializeToHsacoPass)

  StringRef getArgument() const final { return "test-gpu-to-hsaco"; }
  StringRef getDescription() const final {
    return "Lower GPU kernel function to HSAco binary annotations";
  }
  TestSerializeToHsacoPass();

private:
  void getDependentDialects(DialectRegistry &registry) const override;

  // Serializes ROCDL IR to HSACO.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override;
};
} // namespace

TestSerializeToHsacoPass::TestSerializeToHsacoPass() {
  this->triple = "amdgcn-amd-amdhsa";
  this->chip = "gfx900";
}

void TestSerializeToHsacoPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerROCDLDialectTranslation(registry);
  gpu::SerializeToBlobPass::getDependentDialects(registry);
}

std::unique_ptr<std::vector<char>>
TestSerializeToHsacoPass::serializeISA(const std::string &) {
  std::string data = "HSACO";
  return std::make_unique<std::vector<char>>(data.begin(), data.end());
}

namespace mlir {
namespace test {
// Register test pass to serialize GPU module to a HSAco binary annotation.
void registerTestGpuSerializeToHsacoPass() {
  PassRegistration<TestSerializeToHsacoPass>([] {
    // Initialize LLVM AMDGPU backend.
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();

    return std::make_unique<TestSerializeToHsacoPass>();
  });
}
} // namespace test
} // namespace mlir
#endif // MLIR_ROCM_CONVERSIONS_ENABLED
