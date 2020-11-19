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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

namespace {

/// A pass converting tagged kernel modules to a blob with target instructions.
///
/// If tagged as a kernel module, each contained function is translated to
/// user-specified IR. A user provided BlobGenerator then compiles the IR to
/// GPU binary code, which is then attached as an attribute to the function.
/// The function body is erased.
class GpuKernelToBlobPass
    : public PassWrapper<GpuKernelToBlobPass, OperationPass<gpu::GPUModuleOp>> {
public:
  GpuKernelToBlobPass(LoweringCallback loweringCallback,
                      BlobGenerator blobGenerator, StringRef triple,
                      StringRef targetChip, StringRef features,
                      StringRef gpuBinaryAnnotation)
      : loweringCallback(loweringCallback), blobGenerator(blobGenerator),
        triple(triple), targetChip(targetChip), features(features),
        blobAnnotation(gpuBinaryAnnotation) {}

  void runOnOperation() override {
    gpu::GPUModuleOp module = getOperation();

    // Lower the module to an LLVM IR module using a separate context to enable
    // multi-threaded processing.
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmModule =
        loweringCallback(module, llvmContext, "LLVMDialectModule");
    if (!llvmModule)
      return signalPassFailure();

    // Translate the llvm module to a target blob and attach the result as
    // attribute to the module.
    if (auto blobAttr = translateGPUModuleToBinaryAnnotation(
            *llvmModule, module.getLoc(), module.getName()))
      module.setAttr(blobAnnotation, blobAttr);
    else
      signalPassFailure();
  }

private:
  std::string translateModuleToISA(llvm::Module &module,
                                   llvm::TargetMachine &targetMachine);

  /// Converts llvmModule to a blob with target instructions using the
  /// user-provided generator. Location is used for error reporting and name is
  /// forwarded to the blob generator to use in its logging mechanisms.
  OwnedBlob convertModuleToBlob(llvm::Module &llvmModule, Location loc,
                                StringRef name);

  /// Translates llvmModule to a blob with target instructions and returns the
  /// result as attribute.
  StringAttr translateGPUModuleToBinaryAnnotation(llvm::Module &llvmModule,
                                                  Location loc, StringRef name);

  LoweringCallback loweringCallback;
  BlobGenerator blobGenerator;
  llvm::Triple triple;
  StringRef targetChip;
  StringRef features;
  StringRef blobAnnotation;
};

} // anonymous namespace

std::string
GpuKernelToBlobPass::translateModuleToISA(llvm::Module &module,
                                          llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CGFT_AssemblyFile);
    codegenPasses.run(module);
  }

  return targetISA;
}

OwnedBlob GpuKernelToBlobPass::convertModuleToBlob(llvm::Module &llvmModule,
                                                   Location loc,
                                                   StringRef name) {
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  {
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget("", triple, error);
    if (target == nullptr) {
      emitError(loc, "cannot initialize target triple");
      return {};
    }
    targetMachine.reset(target->createTargetMachine(triple.str(), targetChip,
                                                    features, {}, {}));
  }

  llvmModule.setDataLayout(targetMachine->createDataLayout());

  auto targetISA = translateModuleToISA(llvmModule, *targetMachine);

  return blobGenerator(targetISA, loc, name);
}

StringAttr GpuKernelToBlobPass::translateGPUModuleToBinaryAnnotation(
    llvm::Module &llvmModule, Location loc, StringRef name) {
  auto blob = convertModuleToBlob(llvmModule, loc, name);
  if (!blob)
    return {};
  return StringAttr::get({blob->data(), blob->size()}, loc->getContext());
}

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
