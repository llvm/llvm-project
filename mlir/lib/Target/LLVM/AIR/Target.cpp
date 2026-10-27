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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
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
#if MLIR_AIR_AGX_ENABLED
// Defined by the AGX LLVM backend (llvm/lib/Target/AGX), for AGX chips.
extern "C" void LLVMInitializeAGXTarget();
extern "C" void LLVMInitializeAGXTargetInfo();
extern "C" void LLVMInitializeAGXTargetMC();
extern "C" void LLVMInitializeAGXAsmPrinter();
#endif

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
    // An AGX chip is not an AIR processor: AIR's own passes run for any.
    : ModuleToObject(module, target.getTriple(),
                     target.isAGX() ? "" : target.getChip(), "",
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
  /// `.metallib` container bytes, or for an AGX chip an AGX object.
  FailureOr<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule) override;

  /// For an AGX chip the module is optimized by the AGX backend, once it is
  /// AGX IR, in agxObject.
  LogicalResult optimizeModule(llvm::Module &module, int optLevel) override {
    if (getTarget().isAGX())
      return success();
    return SerializeGPUModuleBase::optimizeModule(module, optLevel);
  }

private:
  /// AIR's portable passes (with the AIR target machine), then the AGX
  /// backend: agx-lower-air at the start of its optimization pipeline, and
  /// code generation to an ELF object.
  FailureOr<SmallVector<char, 0>> agxObject(llvm::Module &llvmModule);
};

/// Runs a textual new pass manager pipeline with `tm`'s passes.
LogicalResult runPipeline(llvm::TargetMachine &tm, llvm::Module &module,
                          StringRef pipeline, std::string &error) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;
  llvm::PassBuilder pb(&tm);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);
  llvm::ModulePassManager mpm;
  if (llvm::Error e = pb.parsePassPipeline(mpm, pipeline)) {
    error = llvm::toString(std::move(e));
    return failure();
  }
  mpm.run(module, mam);
  return success();
}
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

  if (getTarget().isAGX())
    return agxObject(llvmModule);

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

FailureOr<SmallVector<char, 0>>
AIRSerializer::agxObject(llvm::Module &llvmModule) {
#if !MLIR_AIR_AGX_ENABLED
  return getOperation().emitError()
         << "the AGX target is not built, so there is no AGX object for "
         << getTarget().getChip();
#else
  static llvm::once_flag initializeAGXOnce;
  llvm::call_once(initializeAGXOnce, []() {
    LLVMInitializeAGXTargetInfo();
    LLVMInitializeAGXTarget();
    LLVMInitializeAGXTargetMC();
    LLVMInitializeAGXAsmPrinter();
  });
  FailureOr<llvm::TargetMachine *> airMachine = getOrCreateTargetMachine();
  if (failed(airMachine))
    return getOperation().emitError()
           << "Target Machine unavailable for triple " << triple;
  std::string error;
  if (failed(runPipeline(**airMachine, llvmModule, "air-portable", error)))
    return getOperation().emitError() << "AIR's portable passes: " << error;

  llvm::Triple agxTriple("agx");
  const llvm::Target *agx = llvm::TargetRegistry::lookupTarget(agxTriple, error);
  if (!agx)
    return getOperation().emitError() << error;
  int optLevel = getTarget().getO();
  std::unique_ptr<llvm::TargetMachine> agxMachine(agx->createTargetMachine(
      agxTriple, getTarget().getChip(), getTarget().getFeatures(), {}, {},
      std::nullopt, static_cast<llvm::CodeGenOptLevel>(optLevel)));
  if (!agxMachine)
    return getOperation().emitError()
           << "no AGX target machine for " << getTarget().getChip();
  // agx-lower-air runs at the start of the default pipelines.
  if (failed(runPipeline(*agxMachine, llvmModule,
                         "default<O" + std::to_string(optLevel) + ">",
                         error)))
    return getOperation().emitError() << "AGX optimization: " << error;

  SmallVector<char, 0> object;
  {
    llvm::raw_svector_ostream stream(object);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    if (agxMachine->addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                        llvm::CodeGenFileType::ObjectFile))
      return getOperation().emitError()
             << "the AGX target machine cannot emit an object";
    codegenPasses.run(llvmModule);
  }
  return object;
#endif
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
