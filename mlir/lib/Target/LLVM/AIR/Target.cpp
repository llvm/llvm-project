//===- Target.cpp - MLIR LLVM AIR target compilation ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines AIR target related functions including registration calls
// for the `#air.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/AIR/Target.h"

#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVM/AIR/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::air;

// Defined by the in-tree LLVM AIR backend (llvm/lib/Target/AIR).
extern "C" void LLVMInitializeAIRTarget();
extern "C" void LLVMInitializeAIRTargetInfo();
extern "C" void LLVMInitializeAIRTargetMC();
extern "C" void LLVMInitializeAIRAsmPrinter();

namespace {
// Implementation of the `TargetAttrInterface` model.
class AIRTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<AIRTargetAttrImpl> {
public:
  std::optional<mlir::gpu::SerializedObject>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const mlir::gpu::SerializedObject &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

void mlir::air::registerAIRTargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, air::AIRDialect *dialect) {
    AIRTargetAttr::attachInterface<AIRTargetAttrImpl>(*ctx);
  });
}

void mlir::air::registerAIRTargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  registerAIRTargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, AIRTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(), "",
                     target.getO()),
      target(target), targetOptions(targetOptions) {}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
    LLVMInitializeAIRTargetInfo();
    LLVMInitializeAIRTarget();
    LLVMInitializeAIRTargetMC();
    LLVMInitializeAIRAsmPrinter();
  });
}

AIRTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

gpu::GPUModuleOp SerializeGPUModuleBase::getGPUModuleOp() {
  return dyn_cast<gpu::GPUModuleOp>(&SerializeGPUModuleBase::getOperation());
}

namespace {
class AIRSerializer : public SerializeGPUModuleBase {
public:
  AIRSerializer(Operation &module, AIRTargetAttr target,
                const gpu::TargetOptions &targetOptions)
      : SerializeGPUModuleBase(module, target, targetOptions) {}

  /// Drives the AIR LLVM backend codegen pipeline on `llvmModule`, emitting the
  /// `.metallib` container bytes.
  FailureOr<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;
};
} // namespace

static void
propagateWorkgroupAttributionSizes(Operation *gpuModule,
                                   llvm::Module &llvmModule) {
  gpuModule->walk([&](LLVM::LLVMFuncOp funcOp) {
    llvm::Function *llvmFn = llvmModule.getFunction(funcOp.getName());
    if (!llvmFn || llvmFn->isDeclaration())
      return;
    unsigned n = std::min<unsigned>(funcOp.getNumArguments(), llvmFn->arg_size());
    for (unsigned i = 0; i < n; ++i) {
      auto wg = funcOp.getArgAttrOfType<LLVM::WorkgroupAttributionAttr>(
          i, "llvm.workgroup_attribution");
      if (!wg)
        continue;
      uint64_t numElems = wg.getNumElements().getValue().getZExtValue();
      llvmFn->getArg(i)->addAttr(llvm::Attribute::get(
          llvmModule.getContext(), "air.wg.num_elems",
          llvm::utostr(numElems)));
    }
  });
}

FailureOr<SmallVector<char, 0>>
AIRSerializer::moduleToObject(llvm::Module &llvmModule) {
  // Return LLVM IR if the compilation target is `offload`.
  if (targetOptions.getCompilationTarget() == gpu::CompilationTarget::Offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule);

  propagateWorkgroupAttributionSizes(&getOperation(), llvmModule);

  if (const char *dir = ::getenv("AIR_DUMP_LLVM_IR")) {
    std::error_code ec;
    std::string path =
        std::string(dir) + "/" + llvmModule.getName().str() + ".ll";
    llvm::raw_fd_ostream os(path, ec);
    if (!ec)
      llvmModule.print(os, nullptr);
  }

  FailureOr<llvm::TargetMachine *> targetMachine = getOrCreateTargetMachine();
  if (failed(targetMachine))
    return getOperation().emitError()
           << "Target Machine unavailable for triple " << triple
           << ", can't emit AIR\n";

  // Run the complete AIR backend pipeline (AIRFromGenericGPU, AIRSystemValues,
  // AIRPrepare, the AIR writer/embedder, ...) and collect the emitted
  // `.metallib` bytes into an in-memory buffer.
  SmallVector<char, 0> metallib;
  {
    llvm::raw_svector_ostream stream(metallib);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    if ((*targetMachine)
            ->addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                  llvm::CodeGenFileType::ObjectFile))
      return getOperation().emitError()
             << "AIR target machine cannot emit a `.metallib` object";

    codegenPasses.run(llvmModule);
  }
  return metallib;
}

std::optional<mlir::gpu::SerializedObject>
AIRTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                     const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
  AIRSerializer serializer(*module, cast<AIRTargetAttr>(attribute), options);
  serializer.init();
  std::optional<SmallVector<char, 0>> result = serializer.run();
  if (!result)
    return std::nullopt;
  return gpu::SerializedObject{std::move(*result)};
}

Attribute
AIRTargetAttrImpl::createObject(Attribute attribute, Operation *module,
                                const mlir::gpu::SerializedObject &object,
                                const gpu::TargetOptions &options) const {
  auto target = cast<AIRTargetAttr>(attribute);
  gpu::CompilationTarget format = options.getCompilationTarget();
  DictionaryAttr objectProps;
  Builder builder(attribute.getContext());
  SmallVector<NamedAttribute> properties;
  if (format == gpu::CompilationTarget::Assembly)
    properties.push_back(
        builder.getNamedAttr("O", builder.getI32IntegerAttr(target.getO())));

  if (!properties.empty())
    objectProps = builder.getDictionaryAttr(properties);

  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(
          StringRef(object.getObject().data(), object.getObject().size())),
      objectProps, /*kernels=*/nullptr);
}
