//===- TestConvertGPUKernelToCubin.cpp - Test gpu kernel cubin lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/NVVMIR.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

#if MLIR_CUDA_CONVERSIONS_ENABLED
static OwnedBlob compilePtxToCubinForTesting(const std::string &, Location,
                                             StringRef) {
  const char data[] = "CUBIN";
  return std::make_unique<std::vector<char>>(data, data + sizeof(data) - 1);
}

namespace mlir {
namespace test {
void registerTestConvertGPUKernelToCubinPass() {
  PassPipelineRegistration<>(
      "test-kernel-to-cubin",
      "Convert all kernel functions to CUDA cubin blobs",
      [](OpPassManager &pm) {
        // Initialize LLVM NVPTX backend.
        LLVMInitializeNVPTXTarget();
        LLVMInitializeNVPTXTargetInfo();
        LLVMInitializeNVPTXTargetMC();
        LLVMInitializeNVPTXAsmPrinter();

        pm.addPass(createConvertGPUKernelToBlobPass(
            translateModuleToNVVMIR, compilePtxToCubinForTesting,
            "nvptx64-nvidia-cuda", "sm_35", "+ptx60", "nvvm.cubin"));
      });
}
} // namespace test
} // namespace mlir
#endif
