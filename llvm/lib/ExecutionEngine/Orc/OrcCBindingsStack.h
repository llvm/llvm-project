//===- OrcCBindingsStack.h - Orc JIT stack for C bindings -----*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
#define LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H

#include "llvm-c/OrcBindings.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace llvm {

class OrcCBindingsStack;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OrcCBindingsStack, LLVMOrcJITStackRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

namespace detail {

// FIXME: Kill this off once the Layer concept becomes an interface.
class GenericLayer {
public:
  virtual ~GenericLayer() = default;

  virtual JITSymbol findSymbolIn(orc::VModuleKey K, const std::string &Name,
                                 bool ExportedSymbolsOnly) = 0;
  virtual Error removeModule(orc::VModuleKey K) = 0;
  };

  template <typename LayerT> class GenericLayerImpl : public GenericLayer {
  public:
    GenericLayerImpl(LayerT &Layer) : Layer(Layer) {}

    JITSymbol findSymbolIn(orc::VModuleKey K, const std::string &Name,
                           bool ExportedSymbolsOnly) override {
      return Layer.findSymbolIn(K, Name, ExportedSymbolsOnly);
    }

    Error removeModule(orc::VModuleKey K) override {
      return Layer.removeModule(K);
    }

  private:
    LayerT &Layer;
  };

  template <>
  class GenericLayerImpl<orc::RTDyldObjectLinkingLayer> : public GenericLayer {
  private:
    using LayerT = orc::RTDyldObjectLinkingLayer;
  public:
    GenericLayerImpl(LayerT &Layer) : Layer(Layer) {}

    JITSymbol findSymbolIn(orc::VModuleKey K, const std::string &Name,
                           bool ExportedSymbolsOnly) override {
      return Layer.findSymbolIn(K, Name, ExportedSymbolsOnly);
    }

    Error removeModule(orc::VModuleKey K) override {
      return Layer.removeObject(K);
    }

  private:
    LayerT &Layer;
  };

  template <typename LayerT>
  std::unique_ptr<GenericLayerImpl<LayerT>> createGenericLayer(LayerT &Layer) {
    return llvm::make_unique<GenericLayerImpl<LayerT>>(Layer);
  }

} // end namespace detail

class OrcCBindingsStack {
public:

  using CompileCallbackMgr = orc::JITCompileCallbackManager;
  using ObjLayerT = orc::RTDyldObjectLinkingLayer;
  using CompileLayerT = orc::IRCompileLayer<ObjLayerT, orc::SimpleCompiler>;
  using CODLayerT =
        orc::CompileOnDemandLayer<CompileLayerT, CompileCallbackMgr>;

  using CallbackManagerBuilder =
      std::function<std::unique_ptr<CompileCallbackMgr>()>;

  using IndirectStubsManagerBuilder = CODLayerT::IndirectStubsManagerBuilderT;

private:

  using OwningObject = object::OwningBinary<object::ObjectFile>;

  class CBindingsResolver : public orc::SymbolResolver {
  public:
    CBindingsResolver(OrcCBindingsStack &Stack,
                      LLVMOrcSymbolResolverFn ExternalResolver,
                      void *ExternalResolverCtx)
        : Stack(Stack), ExternalResolver(std::move(ExternalResolver)),
          ExternalResolverCtx(std::move(ExternalResolverCtx)) {}

    orc::SymbolNameSet lookupFlags(orc::SymbolFlagsMap &SymbolFlags,
                                   const orc::SymbolNameSet &Symbols) override {
      orc::SymbolNameSet SymbolsNotFound;

      for (auto &S : Symbols) {
        if (auto Sym = findSymbol(*S))
          SymbolFlags[S] = Sym.getFlags();
        else if (auto Err = Sym.takeError()) {
          Stack.reportError(std::move(Err));
          return orc::SymbolNameSet();
        } else
          SymbolsNotFound.insert(S);
      }

      return SymbolsNotFound;
    }

    orc::SymbolNameSet
    lookup(std::shared_ptr<orc::AsynchronousSymbolQuery> Query,
           orc::SymbolNameSet Symbols) override {
      orc::SymbolNameSet UnresolvedSymbols;

      for (auto &S : Symbols) {
        if (auto Sym = findSymbol(*S)) {
          if (auto Addr = Sym.getAddress())
            Query->resolve(S, JITEvaluatedSymbol(*Addr, Sym.getFlags()));
          else {
            Query->notifyMaterializationFailed(Addr.takeError());
            return orc::SymbolNameSet();
          }
        } else if (auto Err = Sym.takeError()) {
          Query->notifyMaterializationFailed(std::move(Err));
          return orc::SymbolNameSet();
        } else
          UnresolvedSymbols.insert(S);
      }

      return UnresolvedSymbols;
    }

  private:
    JITSymbol findSymbol(const std::string &Name) {
      // Search order:
      // 1. JIT'd symbols.
      // 2. Runtime overrides.
      // 3. External resolver (if present).

      if (auto Sym = Stack.CODLayer.findSymbol(Name, true))
        return Sym;
      else if (auto Err = Sym.takeError())
        return Sym.takeError();

      if (auto Sym = Stack.CXXRuntimeOverrides.searchOverrides(Name))
        return Sym;

      if (ExternalResolver)
        return JITSymbol(ExternalResolver(Name.c_str(), ExternalResolverCtx),
                         JITSymbolFlags::Exported);

      return JITSymbol(nullptr);
    }

    OrcCBindingsStack &Stack;
    LLVMOrcSymbolResolverFn ExternalResolver;
    void *ExternalResolverCtx = nullptr;
  };

public:

  OrcCBindingsStack(TargetMachine &TM,
                    std::unique_ptr<CompileCallbackMgr> CCMgr,
                    IndirectStubsManagerBuilder IndirectStubsMgrBuilder)
      : DL(TM.createDataLayout()),
        IndirectStubsMgr(IndirectStubsMgrBuilder()), CCMgr(std::move(CCMgr)),
        ObjectLayer(ES,
                    [this](orc::VModuleKey K) {
                      auto ResolverI = Resolvers.find(K);
                      assert(ResolverI != Resolvers.end() &&
                             "No resolver for module K");
                      auto Resolver = std::move(ResolverI->second);
                      Resolvers.erase(ResolverI);
                      return ObjLayerT::Resources{
                          std::make_shared<SectionMemoryManager>(), Resolver};
                    }),
        CompileLayer(ObjectLayer, orc::SimpleCompiler(TM)),
        CODLayer(ES, CompileLayer,
                 [this](orc::VModuleKey K) {
                   auto ResolverI = Resolvers.find(K);
                   assert(ResolverI != Resolvers.end() &&
                          "No resolver for module K");
                   return ResolverI->second;
                 },
                 [this](orc::VModuleKey K,
                        std::shared_ptr<orc::SymbolResolver> Resolver) {
                   assert(!Resolvers.count(K) && "Resolver already present");
                   Resolvers[K] = std::move(Resolver);
                 },
                 [](Function &F) { return std::set<Function *>({&F}); },
                 *this->CCMgr, std::move(IndirectStubsMgrBuilder), false),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); }) {}

  LLVMOrcErrorCode shutdown() {
    // Run any destructors registered with __cxa_atexit.
    CXXRuntimeOverrides.runDestructors();
    // Run any IR destructors.
    for (auto &DtorRunner : IRStaticDestructorRunners)
      if (auto Err = DtorRunner.runViaLayer(*this))
        return mapError(std::move(Err));
    return LLVMOrcErrSuccess;
  }

  std::string mangle(StringRef Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  template <typename PtrTy>
  static PtrTy fromTargetAddress(JITTargetAddress Addr) {
    return reinterpret_cast<PtrTy>(static_cast<uintptr_t>(Addr));
  }


  LLVMOrcErrorCode
  createLazyCompileCallback(JITTargetAddress &RetAddr,
                            LLVMOrcLazyCompileCallbackFn Callback,
                            void *CallbackCtx) {
    if (auto CCInfoOrErr = CCMgr->getCompileCallback()) {
      auto &CCInfo = *CCInfoOrErr;
      CCInfo.setCompileAction([=]() -> JITTargetAddress {
          return Callback(wrap(this), CallbackCtx);
        });
      RetAddr = CCInfo.getAddress();
      return LLVMOrcErrSuccess;
    } else
      return mapError(CCInfoOrErr.takeError());
  }

  LLVMOrcErrorCode createIndirectStub(StringRef StubName,
                                      JITTargetAddress Addr) {
    return mapError(
        IndirectStubsMgr->createStub(StubName, Addr, JITSymbolFlags::Exported));
  }

  LLVMOrcErrorCode setIndirectStubPointer(StringRef Name,
                                          JITTargetAddress Addr) {
    return mapError(IndirectStubsMgr->updatePointer(Name, Addr));
  }
  template <typename LayerT>
  LLVMOrcErrorCode
  addIRModule(orc::VModuleKey &RetKey, LayerT &Layer, std::unique_ptr<Module> M,
              std::unique_ptr<RuntimeDyld::MemoryManager> MemMgr,
              LLVMOrcSymbolResolverFn ExternalResolver,
              void *ExternalResolverCtx) {

    // Attach a data-layout if one isn't already present.
    if (M->getDataLayout().isDefault())
      M->setDataLayout(DL);

    // Record the static constructors and destructors. We have to do this before
    // we hand over ownership of the module to the JIT.
    std::vector<std::string> CtorNames, DtorNames;
    for (auto Ctor : orc::getConstructors(*M))
      CtorNames.push_back(mangle(Ctor.Func->getName()));
    for (auto Dtor : orc::getDestructors(*M))
      DtorNames.push_back(mangle(Dtor.Func->getName()));

    // Add the module to the JIT.
    RetKey = ES.allocateVModule();
    Resolvers[RetKey] = std::make_shared<CBindingsResolver>(
        *this, ExternalResolver, ExternalResolverCtx);
    if (auto Err = Layer.addModule(RetKey, std::move(M)))
      return mapError(std::move(Err));

    KeyLayers[RetKey] = detail::createGenericLayer(Layer);

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    orc::CtorDtorRunner<OrcCBindingsStack> CtorRunner(std::move(CtorNames),
                                                      RetKey);
    if (auto Err = CtorRunner.runViaLayer(*this))
      return mapError(std::move(Err));

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), RetKey);

    return LLVMOrcErrSuccess;
  }

  LLVMOrcErrorCode addIRModuleEager(orc::VModuleKey &RetKey,
                                    std::unique_ptr<Module> M,
                                    LLVMOrcSymbolResolverFn ExternalResolver,
                                    void *ExternalResolverCtx) {
    return addIRModule(RetKey, CompileLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  LLVMOrcErrorCode addIRModuleLazy(orc::VModuleKey &RetKey,
                                   std::unique_ptr<Module> M,
                                   LLVMOrcSymbolResolverFn ExternalResolver,
                                   void *ExternalResolverCtx) {
    return addIRModule(RetKey, CODLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  LLVMOrcErrorCode removeModule(orc::VModuleKey K) {
    // FIXME: Should error release the module key?
    if (auto Err = KeyLayers[K]->removeModule(K))
      return mapError(std::move(Err));
    ES.releaseVModule(K);
    KeyLayers.erase(K);
    return LLVMOrcErrSuccess;
  }

  LLVMOrcErrorCode addObject(orc::VModuleKey &RetKey,
                             std::unique_ptr<MemoryBuffer> ObjBuffer,
                             LLVMOrcSymbolResolverFn ExternalResolver,
                             void *ExternalResolverCtx) {
    if (auto Obj = object::ObjectFile::createObjectFile(
            ObjBuffer->getMemBufferRef())) {

      RetKey = ES.allocateVModule();
      Resolvers[RetKey] = std::make_shared<CBindingsResolver>(
          *this, ExternalResolver, ExternalResolverCtx);

      if (auto Err = ObjectLayer.addObject(RetKey, std::move(ObjBuffer)))
        return mapError(std::move(Err));

      KeyLayers[RetKey] = detail::createGenericLayer(ObjectLayer);

      return LLVMOrcErrSuccess;
    } else
      return mapError(Obj.takeError());
  }

  JITSymbol findSymbol(const std::string &Name,
                                 bool ExportedSymbolsOnly) {
    if (auto Sym = IndirectStubsMgr->findStub(Name, ExportedSymbolsOnly))
      return Sym;
    return CODLayer.findSymbol(mangle(Name), ExportedSymbolsOnly);
  }

  JITSymbol findSymbolIn(orc::VModuleKey K, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return KeyLayers[K]->findSymbolIn(K, Name, ExportedSymbolsOnly);
  }

  LLVMOrcErrorCode findSymbolAddress(JITTargetAddress &RetAddr,
                                     const std::string &Name,
                                     bool ExportedSymbolsOnly) {
    RetAddr = 0;
    if (auto Sym = findSymbol(Name, ExportedSymbolsOnly)) {
      // Successful lookup, non-null symbol:
      if (auto AddrOrErr = Sym.getAddress()) {
        RetAddr = *AddrOrErr;
        return LLVMOrcErrSuccess;
      } else
        return mapError(AddrOrErr.takeError());
    } else if (auto Err = Sym.takeError()) {
      // Lookup failure - report error.
      return mapError(std::move(Err));
    }
    // Otherwise we had a successful lookup but got a null result. We already
    // set RetAddr to '0' above, so just return success.
    return LLVMOrcErrSuccess;
  }

  const std::string &getErrorMessage() const { return ErrMsg; }

private:

  LLVMOrcErrorCode mapError(Error Err) {
    LLVMOrcErrorCode Result = LLVMOrcErrSuccess;
    handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
      // Handler of last resort.
      Result = LLVMOrcErrGeneric;
      ErrMsg = "";
      raw_string_ostream ErrStream(ErrMsg);
      EIB.log(ErrStream);
    });
    return Result;
  }

  void reportError(Error Err) {
    // FIXME: Report errors on the execution session.
    logAllUnhandledErrors(std::move(Err), errs(), "ORC error: ");
  };

  orc::ExecutionSession ES;

  DataLayout DL;
  SectionMemoryManager CCMgrMemMgr;

  std::unique_ptr<orc::IndirectStubsManager> IndirectStubsMgr;

  std::unique_ptr<CompileCallbackMgr> CCMgr;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  CODLayerT CODLayer;

  std::map<orc::VModuleKey, std::unique_ptr<detail::GenericLayer>> KeyLayers;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<OrcCBindingsStack>> IRStaticDestructorRunners;
  std::string ErrMsg;

  std::map<orc::VModuleKey, std::shared_ptr<orc::SymbolResolver>> Resolvers;
};

} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
