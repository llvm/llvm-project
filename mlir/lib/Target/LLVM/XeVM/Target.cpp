//===-- Target.cpp - MLIR LLVM XeVM target compilation ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines XeVM target related functions including registration
// calls for the `#xevm.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/XeVM/Target.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/XeVMDialect.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/Target/LLVM/XeVM/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Config/Targets.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

// FIXME: One of the headers uses `.inc` file from the build directory, this
// does not work for installation (i.e., DCMAKE_INSTALL_PREFIX) caching as build
// directory will not be cached. Since float atomics are not yet supported by
// the backend anyway, we can afford to temporarily comment this section.

// #if LLVM_HAS_SPIRV_TARGET
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
// #include "SPIRVTargetMachine.h"
// #pragma GCC diagnostic pop

// #include "SPIRVCommandLine.h"
// #endif // LLVM_HAS_SPIRV_TARGET

#include <set>

using namespace mlir;

namespace {
// XeVM implementation of the gpu:TargetAttrInterface.
class XeVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<XeVMTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const SmallVector<char, 0> &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

void mlir::xevm::registerXeVMTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::xevm::XeVMDialect *dialect) {
        mlir::xevm::XeVMTargetAttr::attachInterface<XeVMTargetAttrImpl>(*ctx);
      });
}

void mlir::xevm::registerXeVMTargetInterfaceExternalModels(
    MLIRContext &context) {
  DialectRegistry registry;
  registerXeVMTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

mlir::xevm::SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, mlir::xevm::XeVMTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), "", {}, target.getO()),
      target(target) {}

void mlir::xevm::SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
#if LLVM_HAS_SPIRV_TARGET
    LLVMInitializeSPIRVTarget();
    LLVMInitializeSPIRVTargetInfo();
    LLVMInitializeSPIRVTargetMC();
    LLVMInitializeSPIRVAsmPrinter();
#endif
  });
}

mlir::xevm::XeVMTargetAttr
mlir::xevm::SerializeGPUModuleBase::getTarget() const {
  return target;
}

namespace {
class SpirSerializer : public mlir::xevm::SerializeGPUModuleBase {
public:
  SpirSerializer(Operation &module, mlir::xevm::XeVMTargetAttr target,
                 const gpu::TargetOptions &targetOptions)
      : mlir::xevm::SerializeGPUModuleBase(module, target, targetOptions) {}

  gpu::GPUModuleOp getOperation();

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

private:
  std::optional<std::string>
  translateToSPIRVBinary(llvm::Module &llvmModule,
                         llvm::TargetMachine &targetMachine);
  gpu::TargetOptions targetOptions;
};
} // namespace

gpu::GPUModuleOp SpirSerializer::getOperation() {
  return dyn_cast<gpu::GPUModuleOp>(
      &mlir::xevm::SerializeGPUModuleBase::getOperation());
}

std::optional<SmallVector<char, 0>>
SpirSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is `offload`.
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return mlir::xevm::SerializeGPUModuleBase::moduleToObject(llvmModule);

#if !LLVM_HAS_SPIRV_TARGET
  getOperation()->emitError(
      "The `SPIRV` target was not built. Please enable it when building LLVM.");
  return std::nullopt;
#endif // LLVM_HAS_SPIRV_TARGET

  std::optional<llvm::TargetMachine *> targetMachine =
      getOrCreateTargetMachine();
  if (!targetMachine) {
    getOperation().emitError() << "Target Machine unavailable for triple "
                               << triple << ", can't compile with LLVM\n";
    return std::nullopt;
  }

  //===----------------------------------------------------------------------===//
  // Workaround to enable spirv extensions that are not added to target machine
  // by default.

  // FIXME: see fixme comment above SPIRV headers.
  // #if LLVM_HAS_SPIRV_TARGET
  //   std::set<llvm::SPIRV::Extension::Extension> AllowedExtIds{
  //       llvm::SPIRV::Extension::Extension::SPV_EXT_shader_atomic_float_add,
  //       llvm::SPIRV::Extension::Extension::SPV_EXT_shader_atomic_float16_add};
  //   llvm::SPIRVTargetMachine *STM =
  //       static_cast<llvm::SPIRVTargetMachine *>(targetMachine.value());
  //   const_cast<llvm::SPIRVSubtarget *>(STM->getSubtargetImpl())
  //       ->initAvailableExtensions(AllowedExtIds);
  // #endif // LLVM_HAS_SPIRV_TARGET

  //===----------------------------------------------------------------------===//

  // Return SPIRV if the compilation target is `assembly`.
  if (targetOptions.getCompilationTarget() ==
      gpu::CompilationTarget::Assembly) {
    std::optional<std::string> serializedISA =
        translateToISA(llvmModule, **targetMachine);
    if (!serializedISA) {
      getOperation().emitError() << "Failed translating the module to ISA.";
      return std::nullopt;
    }
    // Make sure to include the null terminator.
    StringRef bin(serializedISA->c_str(), serializedISA->size() + 1);
    return SmallVector<char, 0>(bin.begin(), bin.end());
  }

  std::optional<std::string> serializedSPIRVBinary =
      translateToSPIRVBinary(llvmModule, **targetMachine);
  if (!serializedSPIRVBinary) {
    getOperation().emitError() << "Failed translating the module to Binary.";
    return std::nullopt;
  }
  if (serializedSPIRVBinary->size() % 4) {
    getOperation().emitError() << "SPIRV code size must be a multiple of 4.";
    return std::nullopt;
  }
  StringRef bin(serializedSPIRVBinary->c_str(), serializedSPIRVBinary->size());
  return SmallVector<char, 0>(bin.begin(), bin.end());
}

std::optional<std::string>
SpirSerializer::translateToSPIRVBinary(llvm::Module &llvmModule,
                                       llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  llvm::raw_string_ostream stream(targetISA);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                          llvm::CodeGenFileType::ObjectFile))
      return std::nullopt;

    codegenPasses.run(llvmModule);
  }
  return targetISA;
}

std::optional<SmallVector<char, 0>>
XeVMTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  if (!module)
    return std::nullopt;
  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("expected to be a gpu.module op");
    return std::nullopt;
  }
  gpuMod.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (funcOp->hasAttr(gpu::GPUDialect::getKernelFuncAttrName())) {
      funcOp.setIntelReqdSubGroupSize(16);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  SpirSerializer serializer(
      *module, cast<mlir::xevm::XeVMTargetAttr>(attribute), options);
  serializer.init();

#if !LLVM_HAS_SPIRV_TARGET
  module->emitError("Cannot run `TargetRegistry::lookupTarget()` for SPIRV "
                    "without having the target built.");
#endif

  return serializer.run();
}

Attribute
XeVMTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                 const SmallVector<char, 0> &object,
                                 const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.data(), object.size())),
      objectProps, /*kernels=*/nullptr);
}
