//===- Target.cpp - MLIR LLVM AGXIR target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines AGXIR target related functions including registration
// calls for the `#agxir.target` compilation attribute: GPU modules are
// compiled with the experimental AGX LLVM backend.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/AGXIR/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "mlir/Target/LLVMIR/Dialect/AGXIR/AGXIRToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <optional>

using namespace mlir;
using namespace mlir::AGXIR;

#if MLIR_AGX_TARGET_ENABLED
extern "C" {
void LLVMInitializeAGXTargetInfo();
void LLVMInitializeAGXTarget();
void LLVMInitializeAGXTargetMC();
void LLVMInitializeAGXAsmParser();
void LLVMInitializeAGXAsmPrinter();
}
#endif

namespace {
class AGXIRTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<AGXIRTargetAttrImpl> {
public:
  std::optional<gpu::SerializedObject>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const gpu::SerializedObject &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

void mlir::AGXIR::registerAGXIRTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, AGXIR::AGXIRDialect *dialect) {
    AGXIRTargetAttr::attachInterface<AGXIRTargetAttrImpl>(*ctx);
  });
}

void mlir::AGXIR::registerAGXIRTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerAGXIRTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

#if MLIR_AGX_TARGET_ENABLED
namespace {
/// Serializes a GPU module to AGX assembly, an ELF object or LLVM bitcode.
class AGXSerializer : public LLVM::ModuleToObject {
public:
  AGXSerializer(Operation &module, AGXIRTargetAttr target,
                const gpu::TargetOptions &targetOptions)
      : ModuleToObject(module, target.getTriple(), target.getChip(),
                       target.getFeatures(), target.getO(),
                       targetOptions.getInitialLlvmIRCallback(),
                       targetOptions.getLinkedLlvmIRCallback(),
                       targetOptions.getOptimizedLlvmIRCallback(),
                       targetOptions.getISACallback()),
        targetOptions(targetOptions),
        librariesToLink(targetOptions.getLibrariesToLink()) {
    if (target.getLink())
      librariesToLink.append(target.getLink().begin(), target.getLink().end());
  }

  static void init() {
    static llvm::once_flag initializeBackendOnce;
    llvm::call_once(initializeBackendOnce, []() {
      LLVMInitializeAGXTargetInfo();
      LLVMInitializeAGXTarget();
      LLVMInitializeAGXTargetMC();
      LLVMInitializeAGXAsmParser();
      LLVMInitializeAGXAsmPrinter();
    });
  }

protected:
  std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module) override {
    SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
    if (failed(loadBitcodeFilesFromList(module.getContext(), librariesToLink,
                                        bcFiles, true)))
      return std::nullopt;
    return std::move(bcFiles);
  }

  FailureOr<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override {
    if (targetOptions.getCompilationTarget() ==
        gpu::CompilationTarget::Offload)
      return ModuleToObject::moduleToObject(llvmModule);

    FailureOr<llvm::TargetMachine *> targetMachine =
        getOrCreateTargetMachine();
    if (failed(targetMachine))
      return getOperation().emitError()
             << "target machine unavailable for triple " << triple
             << ", can't compile with LLVM";

    if (targetOptions.getCompilationTarget() ==
        gpu::CompilationTarget::Assembly) {
      FailureOr<SmallString<0>> isa =
          translateModuleToISA(llvmModule, **targetMachine,
                               [&]() { return getOperation().emitError(); });
      if (failed(isa))
        return failure();
      if (isaCallback)
        isaCallback(*isa);
      return SmallVector<char, 0>(isa->begin(), isa->end());
    }

    // The ELF object, with the machine code of each kernel in .text.
    SmallVector<char, 0> object;
    {
      llvm::raw_svector_ostream stream(object);
      llvm::legacy::PassManager codegenPasses;
      if ((*targetMachine)
              ->addPassesToEmitFile(codegenPasses, stream, nullptr,
                                    llvm::CodeGenFileType::ObjectFile))
        return getOperation().emitError()
               << "the AGX target cannot emit object files";
      codegenPasses.run(llvmModule);
    }
    return object;
  }

private:
  gpu::TargetOptions targetOptions;
  SmallVector<Attribute> librariesToLink;
};
} // namespace
#endif // MLIR_AGX_TARGET_ENABLED

std::optional<gpu::SerializedObject>
AGXIRTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                       const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("module must be a GPU module");
    return std::nullopt;
  }
#if MLIR_AGX_TARGET_ENABLED
  AGXSerializer serializer(*module, cast<AGXIRTargetAttr>(attribute),
                           options);
  AGXSerializer::init();
  std::optional<SmallVector<char, 0>> binary = serializer.run();
  if (!binary)
    return std::nullopt;
  return gpu::SerializedObject{std::move(*binary)};
#else
  module->emitError("the `AGX` target was not built. Please enable it when "
                    "building LLVM (LLVM_EXPERIMENTAL_TARGETS_TO_BUILD=AGX)");
  return std::nullopt;
#endif // MLIR_AGX_TARGET_ENABLED
}

Attribute
AGXIRTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                  const gpu::SerializedObject &object,
                                  const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  // There is no fat binary format: use the ELF object.
  if (format > gpu::CompilationTarget::Binary)
    format = gpu::CompilationTarget::Binary;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(
          StringRef(object.getObject().data(), object.getObject().size())),
      DictionaryAttr{}, gpu::KernelTableAttr{});
}
