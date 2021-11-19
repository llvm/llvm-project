//===- SerializeToBlob.cpp - MLIR GPU lowering pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a base class for a pass to serialize a gpu module
// into a binary blob that can be executed on a GPU. The binary blob is added
// as a string attribute to the gpu module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

using namespace mlir;

std::string gpu::getDefaultGpuBinaryAnnotation() { return "gpu.binary"; }

gpu::SerializeToBlobPass::SerializeToBlobPass(TypeID passID)
    : OperationPass<gpu::GPUModuleOp>(passID) {}

gpu::SerializeToBlobPass::SerializeToBlobPass(const SerializeToBlobPass &other)
    : OperationPass<gpu::GPUModuleOp>(other) {}

Optional<std::string>
gpu::SerializeToBlobPass::translateToISA(llvm::Module &llvmModule,
                                         llvm::TargetMachine &targetMachine) {
  llvmModule.setDataLayout(targetMachine.createDataLayout());

  if (failed(optimizeLlvm(llvmModule, targetMachine)))
    return llvm::None;

  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CGFT_AssemblyFile))
      return llvm::None;

    codegenPasses.run(llvmModule);
  }
  return stream.str();
}

void gpu::SerializeToBlobPass::runOnOperation() {
  // Lower the module to an LLVM IR module using a separate context to enable
  // multi-threaded processing.
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateToLLVMIR(llvmContext);
  if (!llvmModule)
    return signalPassFailure();

  // Lower the LLVM IR module to target ISA.
  std::unique_ptr<llvm::TargetMachine> targetMachine = createTargetMachine();
  if (!targetMachine)
    return signalPassFailure();

  Optional<std::string> maybeTargetISA =
      translateToISA(*llvmModule, *targetMachine);

  if (!maybeTargetISA.hasValue())
    return signalPassFailure();

  std::string targetISA = std::move(maybeTargetISA.getValue());

  // Serialize the target ISA.
  std::unique_ptr<std::vector<char>> blob = serializeISA(targetISA);
  if (!blob)
    return signalPassFailure();

  // Add the blob as module attribute.
  auto attr =
      StringAttr::get(&getContext(), StringRef(blob->data(), blob->size()));
  getOperation()->setAttr(gpuBinaryAnnotation, attr);
}

LogicalResult
gpu::SerializeToBlobPass::optimizeLlvm(llvm::Module &llvmModule,
                                       llvm::TargetMachine &targetMachine) {
  // TODO: If serializeToCubin ends up defining optimizations, factor them
  // into here from SerializeToHsaco
  return success();
}

void gpu::SerializeToBlobPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerLLVMDialectTranslation(registry);
  OperationPass<gpu::GPUModuleOp>::getDependentDialects(registry);
}

std::unique_ptr<llvm::TargetMachine>
gpu::SerializeToBlobPass::createTargetMachine() {
  Location loc = getOperation().getLoc();
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return {};
  }
  llvm::TargetMachine *machine =
      target->createTargetMachine(triple, chip, features, {}, {});
  if (!machine) {
    emitError(loc, "failed to create target machine");
    return {};
  }

  return std::unique_ptr<llvm::TargetMachine>{machine};
}

std::unique_ptr<llvm::Module>
gpu::SerializeToBlobPass::translateToLLVMIR(llvm::LLVMContext &llvmContext) {
  return translateModuleToLLVMIR(getOperation(), llvmContext,
                                 "LLVMDialectModule");
}
