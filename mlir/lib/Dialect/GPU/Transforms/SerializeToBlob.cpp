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

#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include <string>
#include <optional>

#define DEBUG_TYPE "serialize-to-blob"

using namespace mlir;

std::string gpu::getDefaultGpuBinaryAnnotation() { return "gpu.binary"; }

gpu::SerializeToBlobPass::SerializeToBlobPass(TypeID passID)
    : OperationPass<gpu::GPUModuleOp>(passID) {}

gpu::SerializeToBlobPass::SerializeToBlobPass(const SerializeToBlobPass &other)
    : OperationPass<gpu::GPUModuleOp>(other) {}

std::optional<std::string>
gpu::SerializeToBlobPass::translateToISA(llvm::Module &llvmModule,
                                         llvm::TargetMachine &targetMachine) {
  llvmModule.setDataLayout(targetMachine.createDataLayout());

  if (failed(optimizeLlvm(llvmModule, targetMachine)))
    return std::nullopt;

  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CGFT_AssemblyFile))
      return std::nullopt;

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

  std::optional<std::string> maybeTargetISA =
      translateToISA(*llvmModule, *targetMachine);

  if (!maybeTargetISA.has_value())
    return signalPassFailure();

  std::string targetISA = std::move(*maybeTargetISA);

  LLVM_DEBUG({
    llvm::dbgs() << "ISA for module: " << getOperation().getNameAttr() << "\n";
    llvm::dbgs() << targetISA << "\n";
    llvm::dbgs().flush();
  });

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
  int optLevel = this->optLevel.getValue();
  if (optLevel < 0 || optLevel > 3)
    return getOperation().emitError()
           << "invalid optimization level " << optLevel;

  targetMachine.setOptLevel(static_cast<llvm::CodeGenOpt::Level>(optLevel));

  auto transformer =
      makeOptimizingTransformer(optLevel, /*sizeLevel=*/0, &targetMachine);
  auto error = transformer(&llvmModule);
  if (error) {
    InFlightDiagnostic mlirError = getOperation()->emitError();
    llvm::handleAllErrors(
        std::move(error), [&mlirError](const llvm::ErrorInfoBase &ei) {
          mlirError << "could not optimize LLVM IR: " << ei.message();
        });
    return mlirError;
  }
  return success();
}

void gpu::SerializeToBlobPass::getDependentDialects(
    DialectRegistry &registry) const {
  registerGPUDialectTranslation(registry);
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
