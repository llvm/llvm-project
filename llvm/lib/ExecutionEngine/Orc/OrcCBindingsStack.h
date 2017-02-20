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

#include "llvm-c/OrcBindings.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

namespace llvm {

class OrcCBindingsStack;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OrcCBindingsStack, LLVMOrcJITStackRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

class OrcCBindingsStack {
public:
  typedef orc::JITCompileCallbackManager CompileCallbackMgr;
  typedef orc::RTDyldObjectLinkingLayer<> ObjLayerT;
  typedef orc::IRCompileLayer<ObjLayerT> CompileLayerT;
  typedef orc::CompileOnDemandLayer<CompileLayerT, CompileCallbackMgr>
      CODLayerT;

  typedef std::function<std::unique_ptr<CompileCallbackMgr>()>
      CallbackManagerBuilder;

  typedef CODLayerT::IndirectStubsManagerBuilderT IndirectStubsManagerBuilder;

private:
  class GenericHandle {
  public:
    virtual ~GenericHandle() {}
    virtual JITSymbol findSymbolIn(const std::string &Name,
                                   bool ExportedSymbolsOnly) = 0;
    virtual void removeModule() = 0;
  };

  template <typename LayerT> class GenericHandleImpl : public GenericHandle {
  public:
    GenericHandleImpl(LayerT &Layer, typename LayerT::ModuleSetHandleT Handle)
        : Layer(Layer), Handle(std::move(Handle)) {}

    JITSymbol findSymbolIn(const std::string &Name,
                           bool ExportedSymbolsOnly) override {
      return Layer.findSymbolIn(Handle, Name, ExportedSymbolsOnly);
    }

    void removeModule() override { return Layer.removeModuleSet(Handle); }

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

  OrcCBindingsStack(TargetMachine &TM,
                    std::unique_ptr<CompileCallbackMgr> CCMgr,
                    IndirectStubsManagerBuilder IndirectStubsMgrBuilder)
      : DL(TM.createDataLayout()), IndirectStubsMgr(IndirectStubsMgrBuilder()),
        CCMgr(std::move(CCMgr)), ObjectLayer(),
        CompileLayer(ObjectLayer, orc::SimpleCompiler(TM)),
        CODLayer(CompileLayer,
                 [](Function &F) { return std::set<Function *>({&F}); },
                 *this->CCMgr, std::move(IndirectStubsMgrBuilder), false),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); }) {}

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
  static PtrTy fromTargetAddress(JITTargetAddress Addr) {
    return reinterpret_cast<PtrTy>(static_cast<uintptr_t>(Addr));
  }

  JITTargetAddress
  createLazyCompileCallback(LLVMOrcLazyCompileCallbackFn Callback,
                            void *CallbackCtx) {
    auto CCInfo = CCMgr->getCompileCallback();
    CCInfo.setCompileAction([=]() -> JITTargetAddress {
      return Callback(wrap(this), CallbackCtx);
    });
    return CCInfo.getAddress();
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

  std::unique_ptr<JITSymbolResolver>
  createResolver(LLVMOrcSymbolResolverFn ExternalResolver,
                 void *ExternalResolverCtx) {
    return orc::createLambdaResolver(
        [this, ExternalResolver, ExternalResolverCtx](const std::string &Name)
            -> JITSymbol {
          // Search order:
          // 1. JIT'd symbols.
          // 2. Runtime overrides.
          // 3. External resolver (if present).

          if (auto Sym = CODLayer.findSymbol(Name, true))
            return Sym;
          if (auto Sym = CXXRuntimeOverrides.searchOverrides(Name))
            return Sym;

          if (ExternalResolver)
            return JITSymbol(
                ExternalResolver(Name.c_str(), ExternalResolverCtx),
                llvm::JITSymbolFlags::Exported);

          return JITSymbol(nullptr);
        },
        [](const std::string &Name) {
          return JITSymbol(nullptr);
        });
  }

  template <typename LayerT>
  ModuleHandleT addIRModule(LayerT &Layer, Module *M,
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
    std::vector<Module *> S;
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

  ModuleHandleT addIRModuleEager(Module *M,
                                 LLVMOrcSymbolResolverFn ExternalResolver,
                                 void *ExternalResolverCtx) {
    return addIRModule(CompileLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  ModuleHandleT addIRModuleLazy(Module *M,
                                LLVMOrcSymbolResolverFn ExternalResolver,
                                void *ExternalResolverCtx) {
    return addIRModule(CODLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  void removeModule(ModuleHandleT H) {
    GenericHandles[H]->removeModule();
    GenericHandles[H] = nullptr;
    FreeHandleIndexes.push_back(H);
  }

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    if (auto Sym = IndirectStubsMgr->findStub(Name, ExportedSymbolsOnly))
      return Sym;
    return CODLayer.findSymbol(mangle(Name), ExportedSymbolsOnly);
  }

  JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return GenericHandles[H]->findSymbolIn(Name, ExportedSymbolsOnly);
  }

  const std::string &getErrorMessage() const { return ErrMsg; }

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

  DataLayout DL;
  SectionMemoryManager CCMgrMemMgr;

  std::unique_ptr<orc::IndirectStubsManager> IndirectStubsMgr;

  std::unique_ptr<CompileCallbackMgr> CCMgr;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  CODLayerT CODLayer;

  std::vector<std::unique_ptr<GenericHandle>> GenericHandles;
  std::vector<unsigned> FreeHandleIndexes;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<OrcCBindingsStack>> IRStaticDestructorRunners;
  std::string ErrMsg;
};

} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
