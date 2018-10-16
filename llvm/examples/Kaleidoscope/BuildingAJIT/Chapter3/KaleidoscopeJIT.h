//===- KaleidoscopeJIT.h - A simple JIT for Kaleidoscope --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains a simple JIT definition for use in the kaleidoscope tutorials.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
#define LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace llvm {
namespace orc {

class KaleidoscopeJIT {
private:
  ExecutionSession ES;
  std::map<VModuleKey, std::shared_ptr<SymbolResolver>> Resolvers;
  std::unique_ptr<TargetMachine> TM;
  const DataLayout DL;
  LegacyRTDyldObjectLinkingLayer ObjectLayer;
  LegacyIRCompileLayer<decltype(ObjectLayer), SimpleCompiler> CompileLayer;

  using OptimizeFunction =
      std::function<std::unique_ptr<Module>(std::unique_ptr<Module>)>;

  LegacyIRTransformLayer<decltype(CompileLayer), OptimizeFunction> OptimizeLayer;

  std::unique_ptr<JITCompileCallbackManager> CompileCallbackManager;
  LegacyCompileOnDemandLayer<decltype(OptimizeLayer)> CODLayer;

public:
  KaleidoscopeJIT()
      : TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        ObjectLayer(ES,
                    [this](VModuleKey K) {
                      return LegacyRTDyldObjectLinkingLayer::Resources{
                          std::make_shared<SectionMemoryManager>(),
                          Resolvers[K]};
                    }),
        CompileLayer(ObjectLayer, SimpleCompiler(*TM)),
        OptimizeLayer(CompileLayer,
                      [this](std::unique_ptr<Module> M) {
                        return optimizeModule(std::move(M));
                      }),
        CompileCallbackManager(cantFail(orc::createLocalCompileCallbackManager(
            TM->getTargetTriple(), ES, 0))),
        CODLayer(ES, OptimizeLayer,
                 [&](orc::VModuleKey K) { return Resolvers[K]; },
                 [&](orc::VModuleKey K, std::shared_ptr<SymbolResolver> R) {
                   Resolvers[K] = std::move(R);
                 },
                 [](Function &F) { return std::set<Function *>({&F}); },
                 *CompileCallbackManager,
                 orc::createLocalIndirectStubsManagerBuilder(
                     TM->getTargetTriple())) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  TargetMachine &getTargetMachine() { return *TM; }

  VModuleKey addModule(std::unique_ptr<Module> M) {
    // Create a new VModuleKey.
    VModuleKey K = ES.allocateVModule();

    // Build a resolver and associate it with the new key.
    Resolvers[K] = createLegacyLookupResolver(
        ES,
        [this](const std::string &Name) -> JITSymbol {
          if (auto Sym = CompileLayer.findSymbol(Name, false))
            return Sym;
          else if (auto Err = Sym.takeError())
            return std::move(Err);
          if (auto SymAddr =
                  RTDyldMemoryManager::getSymbolAddressInProcess(Name))
            return JITSymbol(SymAddr, JITSymbolFlags::Exported);
          return nullptr;
        },
        [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); });

    // Add the module to the JIT with the new key.
    cantFail(CODLayer.addModule(K, std::move(M)));
    return K;
  }

  JITSymbol findSymbol(const std::string Name) {
    std::string MangledName;
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    return CODLayer.findSymbol(MangledNameStream.str(), true);
  }

  void removeModule(VModuleKey K) {
    cantFail(CODLayer.removeModule(K));
  }

private:
  std::unique_ptr<Module> optimizeModule(std::unique_ptr<Module> M) {
    // Create a function pass manager.
    auto FPM = llvm::make_unique<legacy::FunctionPassManager>(M.get());

    // Add some optimizations.
    FPM->add(createInstructionCombiningPass());
    FPM->add(createReassociatePass());
    FPM->add(createGVNPass());
    FPM->add(createCFGSimplificationPass());
    FPM->doInitialization();

    // Run the optimizations over all functions in the module being added to
    // the JIT.
    for (auto &F : *M)
      FPM->run(F);

    return M;
  }
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_KALEIDOSCOPEJIT_H
