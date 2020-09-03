//===- TestConvertGPUKernelToHsaco.cpp - Test gpu kernel hsaco lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/ROCDLIR.h"
#include "llvm/Support/TargetSelect.h"
using namespace mlir;

#if MLIR_ROCM_CONVERSIONS_ENABLED
static OwnedBlob compileIsaToHsacoForTesting(const std::string &, Location,
                                             StringRef) {
  const char data[] = "HSACO";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

namespace mlir {
void registerTestConvertGPUKernelToHsacoPass() {
  PassPipelineRegistration<>(
      "test-kernel-to-hsaco",
      "Convert all kernel functions to ROCm hsaco blobs",
      [](OpPassManager &pm) {
        // Initialize LLVM AMDGPU backend.
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();

        pm.addPass(createConvertGPUKernelToBlobPass(
            translateModuleToROCDLIR, compileIsaToHsacoForTesting,
            "amdgcn-amd-amdhsa", "gfx900", "-code-object-v3", "rocdl.hsaco"));
      });
}
} // namespace mlir
#endif
