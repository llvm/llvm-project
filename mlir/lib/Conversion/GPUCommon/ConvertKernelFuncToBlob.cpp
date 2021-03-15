//===- ConvertKernelFuncToBlob.cpp - MLIR GPU lowering passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu kernel functions into a
// corresponding binary blob that can be executed on a GPU. Currently
// only translates the function itself but no dependencies.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace {

/// A pass converting tagged kernel modules to a blob with target instructions.
///
/// If tagged as a kernel module, each contained function is translated to
/// user-specified IR. A user provided BlobGenerator then compiles the IR to
/// GPU binary code, which is then attached as an attribute to the function.
/// The function body is erased.
class GpuKernelToBlobPass
    : public PassWrapper<GpuKernelToBlobPass, gpu::SerializeToBlobPass> {
public:
  GpuKernelToBlobPass(LoweringCallback loweringCallback,
                      BlobGenerator blobGenerator, StringRef triple,
                      StringRef targetChip, StringRef features,
                      StringRef gpuBinaryAnnotation)
      : loweringCallback(loweringCallback), blobGenerator(blobGenerator) {
    if (!triple.empty())
      this->triple = triple.str();
    if (!targetChip.empty())
      this->chip = targetChip.str();
    if (!features.empty())
      this->features = features.str();
    if (!gpuBinaryAnnotation.empty())
      this->gpuBinaryAnnotation = gpuBinaryAnnotation.str();
  }

private:
  // Translates the 'getOperation()' result to an LLVM module.
  // Note: when this class is removed, this function no longer needs to be
  // virtual.
  std::unique_ptr<llvm::Module>
  translateToLLVMIR(llvm::LLVMContext &llvmContext) override {
    return loweringCallback(getOperation(), llvmContext, "LLVMDialectModule");
  }

  // Serializes the target ISA to binary form.
  std::unique_ptr<std::vector<char>>
  serializeISA(const std::string &isa) override {
    return blobGenerator(isa, getOperation().getLoc(),
                         getOperation().getName());
  }

  LoweringCallback loweringCallback;
  BlobGenerator blobGenerator;
};

} // anonymous namespace

std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
mlir::createConvertGPUKernelToBlobPass(LoweringCallback loweringCallback,
                                       BlobGenerator blobGenerator,
                                       StringRef triple, StringRef targetChip,
                                       StringRef features,
                                       StringRef gpuBinaryAnnotation) {
  return std::make_unique<GpuKernelToBlobPass>(loweringCallback, blobGenerator,
                                               triple, targetChip, features,
                                               gpuBinaryAnnotation);
}
