//===----- LLJIT.h -- An ORC-based JIT for compiling LLVM IR ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for 3Bdetails.
//
//===----------------------------------------------------------------------===//
//
// An ORC-based JIT for compiling LLVM IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_LLJIT_H
#define LLVM_EXECUTIONENGINE_ORC_LLJIT_H

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
namespace orc {

/// A pre-fabricated ORC JIT stack that can serve as an alternative to MCJIT.
class LLJIT {
public:
  /// Create an LLJIT instance.
  static Expected<std::unique_ptr<LLJIT>>
  Create(std::unique_ptr<ExecutionSession> ES,
         std::unique_ptr<TargetMachine> TM, DataLayout DL);

  /// Returns a reference to the ExecutionSession for this JIT instance.
  ExecutionSession &getExecutionSession() { return *ES; }

  /// Returns a reference to the VSO representing the JIT'd main program.
  VSO &getMainVSO() { return Main; }

  /// Convenience method for defining an absolute symbol.
  Error defineAbsolute(StringRef Name, JITEvaluatedSymbol Address);

  /// Adds an IR module to the given VSO.
  Error addIRModule(VSO &V, std::unique_ptr<Module> M);

  /// Adds an IR module to the Main VSO.
  Error addIRModule(std::unique_ptr<Module> M) {
    return addIRModule(Main, std::move(M));
  }

  /// Look up a symbol in VSO V by the symbol's linker-mangled name (to look up
  /// symbols based on their IR name use the lookup function instead).
  Expected<JITEvaluatedSymbol> lookupLinkerMangled(VSO &V, StringRef Name);

  /// Look up a symbol in the main VSO by the symbol's linker-mangled name (to
  /// look up symbols based on their IR name use the lookup function instead).
  Expected<JITEvaluatedSymbol> lookupLinkerMangled(StringRef Name) {
    return lookupLinkerMangled(Main, Name);
  }

  /// Look up a symbol in VSO V based on its IR symbol name.
  Expected<JITEvaluatedSymbol> lookup(VSO &V, StringRef UnmangledName) {
    return lookupLinkerMangled(V, mangle(UnmangledName));
  }

  /// Look up a symbol in the main VSO based on its IR symbol name.
  Expected<JITEvaluatedSymbol> lookup(StringRef UnmangledName) {
    return lookup(Main, UnmangledName);
  }

  /// Runs all not-yet-run static constructors.
  Error runConstructors() { return CtorRunner.run(); }

  /// Runs all not-yet-run static destructors.
  Error runDestructors() { return DtorRunner.run(); }

protected:
  LLJIT(std::unique_ptr<ExecutionSession> ES, std::unique_ptr<TargetMachine> TM,
        DataLayout DL);

  std::shared_ptr<SymbolResolver> takeSymbolResolver(VModuleKey K);
  RTDyldObjectLinkingLayer2::Resources getRTDyldResources(VModuleKey K);

  std::string mangle(StringRef UnmangledName);

  std::unique_ptr<SymbolResolver> createResolverFor(VSO &V);

  Error applyDataLayout(Module &M);

  void recordCtorDtors(Module &M);

  std::unique_ptr<ExecutionSession> ES;
  VSO &Main;

  std::unique_ptr<TargetMachine> TM;
  DataLayout DL;

  std::map<VSO *, VSOList> VSOLookupOrder;

  RTDyldObjectLinkingLayer2 ObjLinkingLayer;
  IRCompileLayer2 CompileLayer;

  std::map<VModuleKey, std::shared_ptr<orc::SymbolResolver>> Resolvers;
  CtorDtorRunner2 CtorRunner, DtorRunner;
};

/// An extended version of LLJIT that supports lazy function-at-a-time
/// compilation of LLVM IR.
class LLLazyJIT : public LLJIT {
public:
  /// Create an LLLazyJIT instance.
  static Expected<std::unique_ptr<LLLazyJIT>>
  Create(std::unique_ptr<ExecutionSession> ES,
         std::unique_ptr<TargetMachine> TM, DataLayout DL, LLVMContext &Ctx);

  /// Set an IR transform (e.g. pass manager pipeline) to run on each function
  /// when it is compiled.
  void setLazyCompileTransform(IRTransformLayer2::TransformFunction Transform) {
    TransformLayer.setTransform(std::move(Transform));
  }

  /// Add a module to be lazily compiled to VSO V.
  Error addLazyIRModule(VSO &V, std::unique_ptr<Module> M);

  /// Add a module to be lazily compiled to the main VSO.
  Error addLazyIRModule(std::unique_ptr<Module> M) {
    return addLazyIRModule(Main, std::move(M));
  }

private:
  LLLazyJIT(std::unique_ptr<ExecutionSession> ES,
            std::unique_ptr<TargetMachine> TM, DataLayout DL, LLVMContext &Ctx,
            std::unique_ptr<JITCompileCallbackManager> CCMgr,
            std::function<std::unique_ptr<IndirectStubsManager>()> ISMBuilder);

  std::shared_ptr<SymbolResolver> getSymbolResolver(VModuleKey K);

  void setSymbolResolver(VModuleKey K, std::shared_ptr<SymbolResolver> R);

  std::unique_ptr<JITCompileCallbackManager> CCMgr;
  std::function<std::unique_ptr<IndirectStubsManager>()> ISMBuilder;

  IRTransformLayer2 TransformLayer;
  CompileOnDemandLayer2 CODLayer;
};

} // End namespace orc
} // End namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_LLJIT_H
