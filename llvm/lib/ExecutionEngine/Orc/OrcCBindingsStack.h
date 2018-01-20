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
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace llvm {

class OrcCBindingsStack;

DEFINE_SIMPLE_CONVERSION_FUNCTIONS(std::shared_ptr<Module>,
                                   LLVMSharedModuleRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(OrcCBindingsStack, LLVMOrcJITStackRef)
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(TargetMachine, LLVMTargetMachineRef)

namespace detail {


  class GenericHandle {
  public:
    virtual ~GenericHandle() = default;

    virtual JITSymbol findSymbolIn(const std::string &Name,
                                   bool ExportedSymbolsOnly) = 0;
    virtual Error removeModule() = 0;
  };

  template <typename LayerT> class GenericHandleImpl : public GenericHandle {
  public:
    GenericHandleImpl(LayerT &Layer, typename LayerT::ModuleHandleT Handle)
        : Layer(Layer), Handle(std::move(Handle)) {}

    JITSymbol findSymbolIn(const std::string &Name,
                           bool ExportedSymbolsOnly) override {
      return Layer.findSymbolIn(Handle, Name, ExportedSymbolsOnly);
    }

    Error removeModule() override { return Layer.removeModule(Handle); }

  private:
    LayerT &Layer;
    typename LayerT::ModuleHandleT Handle;
  };

  template <>
  class GenericHandleImpl<orc::RTDyldObjectLinkingLayer>
    : public GenericHandle {
  private:
    using LayerT = orc::RTDyldObjectLinkingLayer;
  public:

    GenericHandleImpl(LayerT &Layer, typename LayerT::ObjHandleT Handle)
        : Layer(Layer), Handle(std::move(Handle)) {}

    JITSymbol findSymbolIn(const std::string &Name,
                           bool ExportedSymbolsOnly) override {
      return Layer.findSymbolIn(Handle, Name, ExportedSymbolsOnly);
    }

    Error removeModule() override { return Layer.removeObject(Handle); }

  private:
    LayerT &Layer;
    typename LayerT::ObjHandleT Handle;
  };


  template <typename LayerT, typename HandleT>
  std::unique_ptr<GenericHandleImpl<LayerT>>
  createGenericHandle(LayerT &Layer, HandleT Handle) {
    return llvm::make_unique<GenericHandleImpl<LayerT>>(Layer,
                                                        std::move(Handle));
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

public:
  using ModuleHandleT = unsigned;

  OrcCBindingsStack(TargetMachine &TM,
                    std::unique_ptr<CompileCallbackMgr> CCMgr,
                    IndirectStubsManagerBuilder IndirectStubsMgrBuilder)
      : DL(TM.createDataLayout()), IndirectStubsMgr(IndirectStubsMgrBuilder()),
        CCMgr(std::move(CCMgr)),
        ObjectLayer(
          []() {
            return std::make_shared<SectionMemoryManager>();
          }),
        CompileLayer(ObjectLayer, orc::SimpleCompiler(TM)),
        CODLayer(CompileLayer,
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

  std::shared_ptr<LegacyJITSymbolResolver>
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
          else if (auto Err = Sym.takeError())
            return Sym.takeError();

          if (auto Sym = CXXRuntimeOverrides.searchOverrides(Name))
            return Sym;

          if (ExternalResolver)
            return JITSymbol(
                ExternalResolver(Name.c_str(), ExternalResolverCtx),
                JITSymbolFlags::Exported);

          return JITSymbol(nullptr);
        },
        [](const std::string &Name) -> JITSymbol {
          return JITSymbol(nullptr);
        });
  }

  template <typename LayerT>
  LLVMOrcErrorCode
  addIRModule(ModuleHandleT &RetHandle, LayerT &Layer,
              std::shared_ptr<Module> M,
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
    ModuleHandleT H;
    if (auto LHOrErr = Layer.addModule(std::move(M), std::move(Resolver)))
      H = createHandle(Layer, *LHOrErr);
    else
      return mapError(LHOrErr.takeError());

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    orc::CtorDtorRunner<OrcCBindingsStack> CtorRunner(std::move(CtorNames), H);
    if (auto Err = CtorRunner.runViaLayer(*this))
      return mapError(std::move(Err));

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), H);

    RetHandle = H;
    return LLVMOrcErrSuccess;
  }

  LLVMOrcErrorCode addIRModuleEager(ModuleHandleT &RetHandle,
                                    std::shared_ptr<Module> M,
                                    LLVMOrcSymbolResolverFn ExternalResolver,
                                    void *ExternalResolverCtx) {
    return addIRModule(RetHandle, CompileLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  LLVMOrcErrorCode addIRModuleLazy(ModuleHandleT &RetHandle,
                                   std::shared_ptr<Module> M,
                                   LLVMOrcSymbolResolverFn ExternalResolver,
                                   void *ExternalResolverCtx) {
    return addIRModule(RetHandle, CODLayer, std::move(M),
                       llvm::make_unique<SectionMemoryManager>(),
                       std::move(ExternalResolver), ExternalResolverCtx);
  }

  LLVMOrcErrorCode removeModule(ModuleHandleT H) {
    if (auto Err = GenericHandles[H]->removeModule())
      return mapError(std::move(Err));
    GenericHandles[H] = nullptr;
    FreeHandleIndexes.push_back(H);
    return LLVMOrcErrSuccess;
  }

  LLVMOrcErrorCode addObject(ModuleHandleT &RetHandle,
                             std::unique_ptr<MemoryBuffer> ObjBuffer,
                             LLVMOrcSymbolResolverFn ExternalResolver,
                             void *ExternalResolverCtx) {
    if (auto ObjOrErr =
        object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef())) {
      auto &Obj = *ObjOrErr;
      auto OwningObj =
        std::make_shared<OwningObject>(std::move(Obj), std::move(ObjBuffer));

      // Create the resolver.
      auto Resolver = createResolver(ExternalResolver, ExternalResolverCtx);

      ModuleHandleT H;
      if (auto HOrErr = ObjectLayer.addObject(std::move(OwningObj),
                                              std::move(Resolver)))
        H = createHandle(ObjectLayer, *HOrErr);
      else
        return mapError(HOrErr.takeError());

      RetHandle = H;

      return LLVMOrcErrSuccess;
    } else
      return mapError(ObjOrErr.takeError());
  }

  JITSymbol findSymbol(const std::string &Name,
                                 bool ExportedSymbolsOnly) {
    if (auto Sym = IndirectStubsMgr->findStub(Name, ExportedSymbolsOnly))
      return Sym;
    return CODLayer.findSymbol(mangle(Name), ExportedSymbolsOnly);
  }

  JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return GenericHandles[H]->findSymbolIn(Name, ExportedSymbolsOnly);
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
  template <typename LayerT, typename HandleT>
  unsigned createHandle(LayerT &Layer, HandleT Handle) {
    unsigned NewHandle;
    if (!FreeHandleIndexes.empty()) {
      NewHandle = FreeHandleIndexes.back();
      FreeHandleIndexes.pop_back();
      GenericHandles[NewHandle] =
        detail::createGenericHandle(Layer, std::move(Handle));
      return NewHandle;
    } else {
      NewHandle = GenericHandles.size();
      GenericHandles.push_back(
        detail::createGenericHandle(Layer, std::move(Handle)));
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

  std::vector<std::unique_ptr<detail::GenericHandle>> GenericHandles;
  std::vector<unsigned> FreeHandleIndexes;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<OrcCBindingsStack>> IRStaticDestructorRunners;
  std::string ErrMsg;
};

} // end namespace llvm

#endif // LLVM_LIB_EXECUTIONENGINE_ORC_ORCCBINDINGSSTACK_H
