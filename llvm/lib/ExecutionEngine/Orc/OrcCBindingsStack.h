//===--- OrcCBindingsStack.h - Orc JIT stack for C bindings ---*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
#define LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm-c/OrcBindings.h"

namespace llvm {

class OrcCBindingsStack;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OrcCBindingsStack, LLVMOrcJITStackRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

class OrcCBindingsStack {
public:

  typedef orc::JITCompileCallbackManager CompileCallbackMgr;
  typedef orc::ObjectLinkingLayer<> ObjLayerT;
  typedef orc::IRCompileLayer<ObjLayerT> CompileLayerT;
  typedef orc::CompileOnDemandLayer<CompileLayerT, CompileCallbackMgr> CODLayerT;

  typedef std::function<std::unique_ptr<CompileCallbackMgr>()>
    CallbackManagerBuilder;

  typedef CODLayerT::IndirectStubsManagerBuilderT IndirectStubsManagerBuilder;

private:

  class GenericHandle {
  public:
    virtual ~GenericHandle() {}
    virtual orc::JITSymbol findSymbolIn(const std::string &Name,
                                        bool ExportedSymbolsOnly) = 0;
    virtual void removeModule() = 0;
  };

  template <typename LayerT>
  class GenericHandleImpl : public GenericHandle {
  public:
    GenericHandleImpl(LayerT &Layer, typename LayerT::ModuleSetHandleT Handle)
      : Layer(Layer), Handle(std::move(Handle)) {}

    orc::JITSymbol findSymbolIn(const std::string &Name,
                                bool ExportedSymbolsOnly) override {
      return Layer.findSymbolIn(Handle, Name, ExportedSymbolsOnly);
    }

    void removeModule() override {
      return Layer.removeModuleSet(Handle);
    }

  private:
    LayerT &Layer;
    typename LayerT::ModuleSetHandleT Handle;
  };

  template <typename LayerT>
  std::unique_ptr<GenericHandleImpl<LayerT>>
  createGenericHandle(LayerT &Layer, typename LayerT::ModuleSetHandleT Handle) {
    return llvm::make_unique<GenericHandleImpl<LayerT>>(Layer,
                                                        std::move(Handle));
  }

public:

  // We need a 'ModuleSetHandleT' to conform to the layer concept.
  typedef unsigned ModuleSetHandleT;

  typedef unsigned ModuleHandleT;

  static std::unique_ptr<CompileCallbackMgr> createCompileCallbackMgr(Triple T);
  static IndirectStubsManagerBuilder createIndirectStubsMgrBuilder(Triple T);

  OrcCBindingsStack(TargetMachine &TM,
		    std::unique_ptr<CompileCallbackMgr> CCMgr, 
                    IndirectStubsManagerBuilder IndirectStubsMgrBuilder)
    : DL(TM.createDataLayout()), CCMgr(std::move(CCMgr)),
      ObjectLayer(),
      CompileLayer(ObjectLayer, orc::SimpleCompiler(TM)),
      CODLayer(CompileLayer,
               [](Function &F) { std::set<Function*> S; S.insert(&F); return S; },
               *this->CCMgr, std::move(IndirectStubsMgrBuilder), false),
      IndirectStubsMgr(IndirectStubsMgrBuilder()),
      CXXRuntimeOverrides([this](const std::string &S) { return mangle(S); }) {}

  ~OrcCBindingsStack() {
    // Run any destructors registered with __cxa_atexit.
    CXXRuntimeOverrides.runDestructors();
    // Run any IR destructors.
    for (auto &DtorRunner : IRStaticDestructorRunners)
      DtorRunner.runViaLayer(*this);
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
  static PtrTy fromTargetAddress(orc::TargetAddress Addr) {
    return reinterpret_cast<PtrTy>(static_cast<uintptr_t>(Addr));
  }

  orc::TargetAddress
  createLazyCompileCallback(LLVMOrcLazyCompileCallbackFn Callback,
                            void *CallbackCtx) {
    auto CCInfo = CCMgr->getCompileCallback();
    CCInfo.setCompileAction(
      [=]() -> orc::TargetAddress {
        return Callback(wrap(this), CallbackCtx);
      });
    return CCInfo.getAddress();
  }

  void createIndirectStub(StringRef StubName, orc::TargetAddress Addr) {
    IndirectStubsMgr->createStub(StubName, Addr, JITSymbolFlags::Exported);
  }

  void setIndirectStubPointer(StringRef Name, orc::TargetAddress Addr) {
    IndirectStubsMgr->updatePointer(Name, Addr);
  }

  std::shared_ptr<RuntimeDyld::SymbolResolver>
  createResolver(LLVMOrcSymbolResolverFn ExternalResolver,
                 void *ExternalResolverCtx) {
    auto Resolver = orc::createLambdaResolver(
      [this, ExternalResolver, ExternalResolverCtx](const std::string &Name) {
        // Search order:
        // 1. JIT'd symbols.
        // 2. Runtime overrides.
        // 3. External resolver (if present).

        if (auto Sym = CODLayer.findSymbol(Name, true))
          return RuntimeDyld::SymbolInfo(Sym.getAddress(),
                                         Sym.getFlags());
        if (auto Sym = CXXRuntimeOverrides.searchOverrides(Name))
          return Sym;

        if (ExternalResolver)
          return RuntimeDyld::SymbolInfo(ExternalResolver(Name.c_str(),
                                                          ExternalResolverCtx),
                                         llvm::JITSymbolFlags::Exported);

        return RuntimeDyld::SymbolInfo(nullptr);
      },
      [](const std::string &Name) {
        return RuntimeDyld::SymbolInfo(nullptr);
      }
    );

    return std::shared_ptr<RuntimeDyld::SymbolResolver>(std::move(Resolver));
  }

  template <typename LayerT>
  ModuleHandleT addIRModule(LayerT &Layer,
                            Module *M,
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

    // Create the resolver.
    auto Resolver = createResolver(ExternalResolver, ExternalResolverCtx);

    // Add the module to the JIT.
    std::vector<Module*> S;
    S.push_back(std::move(M));

    auto LH = Layer.addModuleSet(std::move(S), std::move(MemMgr),
                                 std::move(Resolver));
    ModuleHandleT H = createHandle(Layer, LH);

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    orc::CtorDtorRunner<OrcCBindingsStack> CtorRunner(std::move(CtorNames), H);
    CtorRunner.runViaLayer(*this);

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), H);

    return H;
  }

  ModuleHandleT addIRModuleEager(Module* M,
                                 LLVMOrcSymbolResolverFn ExternalResolver,
                                 void *ExternalResolverCtx) {
    return addIRModule(CompileLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  ModuleHandleT addIRModuleLazy(Module* M,
                                LLVMOrcSymbolResolverFn ExternalResolver,
                                void *ExternalResolverCtx) {
    return addIRModule(CODLayer, std::move(M), nullptr,
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  void removeModule(ModuleHandleT H) {
    GenericHandles[H]->removeModule();
    GenericHandles[H] = nullptr;
    FreeHandleIndexes.push_back(H);
  }

  orc::JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    if (auto Sym = IndirectStubsMgr->findStub(Name, ExportedSymbolsOnly))
      return Sym;
    return CODLayer.findSymbol(mangle(Name), ExportedSymbolsOnly);
  }

  orc::JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                              bool ExportedSymbolsOnly) {
    return GenericHandles[H]->findSymbolIn(Name, ExportedSymbolsOnly);
  }

private:

  template <typename LayerT>
  unsigned createHandle(LayerT &Layer,
                        typename LayerT::ModuleSetHandleT Handle) {
    unsigned NewHandle;
    if (!FreeHandleIndexes.empty()) {
      NewHandle = FreeHandleIndexes.back();
      FreeHandleIndexes.pop_back();
      GenericHandles[NewHandle] = createGenericHandle(Layer, std::move(Handle));
      return NewHandle;
    } else {
      NewHandle = GenericHandles.size();
      GenericHandles.push_back(createGenericHandle(Layer, std::move(Handle)));
    }
    return NewHandle;
  }

  DataLayout DL;
  SectionMemoryManager CCMgrMemMgr;

  std::unique_ptr<CompileCallbackMgr> CCMgr;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  CODLayerT CODLayer;

  std::unique_ptr<orc::IndirectStubsManager> IndirectStubsMgr;

  std::vector<std::unique_ptr<GenericHandle>> GenericHandles;
  std::vector<unsigned> FreeHandleIndexes;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<OrcCBindingsStack>> IRStaticDestructorRunners;
};

} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
