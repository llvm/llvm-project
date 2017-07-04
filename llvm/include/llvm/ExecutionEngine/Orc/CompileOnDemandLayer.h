//===- CompileOnDemandLayer.h - Compile each function on demand -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// JIT layer for breaking up modules and inserting callbacks to allow
// individual functions to be compiled on demand.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
#define LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class Value;

namespace orc {

/// @brief Compile-on-demand layer.
///
///   When a module is added to this layer a stub is created for each of its
/// function definitions. The stubs and other global values are immediately
/// added to the layer below. When a stub is called it triggers the extraction
/// of the function body from the original module. The extracted body is then
/// compiled and executed.
template <typename BaseLayerT,
          typename CompileCallbackMgrT = JITCompileCallbackManager,
          typename IndirectStubsMgrT = IndirectStubsManager>
class CompileOnDemandLayer {
private:
  template <typename MaterializerFtor>
  class LambdaMaterializer final : public ValueMaterializer {
  public:
    LambdaMaterializer(MaterializerFtor M) : M(std::move(M)) {}

    Value *materialize(Value *V) final { return M(V); }

  private:
    MaterializerFtor M;
  };

  template <typename MaterializerFtor>
  LambdaMaterializer<MaterializerFtor>
  createLambdaMaterializer(MaterializerFtor M) {
    return LambdaMaterializer<MaterializerFtor>(std::move(M));
  }

  using BaseLayerModuleHandleT = typename BaseLayerT::ModuleHandleT;

  // Provide type-erasure for the Modules and MemoryManagers.
  template <typename ResourceT>
  class ResourceOwner {
  public:
    ResourceOwner() = default;
    ResourceOwner(const ResourceOwner &) = delete;
    ResourceOwner &operator=(const ResourceOwner &) = delete;
    virtual ~ResourceOwner() = default;

    virtual ResourceT& getResource() const = 0;
  };

  template <typename ResourceT, typename ResourcePtrT>
  class ResourceOwnerImpl : public ResourceOwner<ResourceT> {
  public:
    ResourceOwnerImpl(ResourcePtrT ResourcePtr)
      : ResourcePtr(std::move(ResourcePtr)) {}

    ResourceT& getResource() const override { return *ResourcePtr; }

  private:
    ResourcePtrT ResourcePtr;
  };

  template <typename ResourceT, typename ResourcePtrT>
  std::unique_ptr<ResourceOwner<ResourceT>>
  wrapOwnership(ResourcePtrT ResourcePtr) {
    using RO = ResourceOwnerImpl<ResourceT, ResourcePtrT>;
    return llvm::make_unique<RO>(std::move(ResourcePtr));
  }

  class StaticGlobalRenamer {
  public:
    StaticGlobalRenamer() = default;
    StaticGlobalRenamer(StaticGlobalRenamer &&) = default;
    StaticGlobalRenamer &operator=(StaticGlobalRenamer &&) = default;

    void rename(Module &M) {
      for (auto &F : M)
        if (F.hasLocalLinkage())
          F.setName("$static." + Twine(NextId++));
      for (auto &G : M.globals())
        if (G.hasLocalLinkage())
          G.setName("$static." + Twine(NextId++));
    }

  private:
    unsigned NextId = 0;
  };

  struct LogicalDylib {
    using SymbolResolverFtor = std::function<JITSymbol(const std::string&)>;

    using ModuleAdderFtor =
      std::function<typename BaseLayerT::ModuleHandleT(
                    BaseLayerT&,
                    std::unique_ptr<Module>,
                    std::unique_ptr<JITSymbolResolver>)>;

    struct SourceModuleEntry {
      std::shared_ptr<Module> SourceMod;
      std::set<Function*> StubsToClone;
    };

    using SourceModulesList = std::vector<SourceModuleEntry>;
    using SourceModuleHandle = typename SourceModulesList::size_type;

    SourceModuleHandle
    addSourceModule(std::shared_ptr<Module> M) {
      SourceModuleHandle H = SourceModules.size();
      SourceModules.push_back(SourceModuleEntry());
      SourceModules.back().SourceMod = std::move(M);
      return H;
    }

    Module& getSourceModule(SourceModuleHandle H) {
      return *SourceModules[H].SourceMod;
    }

    std::set<Function*>& getStubsToClone(SourceModuleHandle H) {
      return SourceModules[H].StubsToClone;
    }

    JITSymbol findSymbol(BaseLayerT &BaseLayer, const std::string &Name,
                         bool ExportedSymbolsOnly) {
      if (auto Sym = StubsMgr->findStub(Name, ExportedSymbolsOnly))
        return Sym;
      for (auto BLH : BaseLayerHandles)
        if (auto Sym = BaseLayer.findSymbolIn(BLH, Name, ExportedSymbolsOnly))
          return Sym;
      return nullptr;
    }

    void removeModulesFromBaseLayer(BaseLayerT &BaseLayer) {
      for (auto &BLH : BaseLayerHandles)
        BaseLayer.removeModule(BLH);
    }

    std::shared_ptr<JITSymbolResolver> ExternalSymbolResolver;
    std::unique_ptr<IndirectStubsMgrT> StubsMgr;
    StaticGlobalRenamer StaticRenamer;
    SourceModulesList SourceModules;
    std::vector<BaseLayerModuleHandleT> BaseLayerHandles;
  };

  using LogicalDylibList = std::list<LogicalDylib>;

public:

  /// @brief Handle to loaded module.
  using ModuleHandleT = typename LogicalDylibList::iterator;

  /// @brief Module partitioning functor.
  using PartitioningFtor = std::function<std::set<Function*>(Function&)>;

  /// @brief Builder for IndirectStubsManagers.
  using IndirectStubsManagerBuilderT =
      std::function<std::unique_ptr<IndirectStubsMgrT>()>;

  /// @brief Construct a compile-on-demand layer instance.
  CompileOnDemandLayer(BaseLayerT &BaseLayer, PartitioningFtor Partition,
                       CompileCallbackMgrT &CallbackMgr,
                       IndirectStubsManagerBuilderT CreateIndirectStubsManager,
                       bool CloneStubsIntoPartitions = true)
      : BaseLayer(BaseLayer), Partition(std::move(Partition)),
        CompileCallbackMgr(CallbackMgr),
        CreateIndirectStubsManager(std::move(CreateIndirectStubsManager)),
        CloneStubsIntoPartitions(CloneStubsIntoPartitions) {}

  ~CompileOnDemandLayer() {
    while (!LogicalDylibs.empty())
      removeModule(LogicalDylibs.begin());
  }

  /// @brief Add a module to the compile-on-demand layer.
  ModuleHandleT addModule(std::shared_ptr<Module> M,
                          std::shared_ptr<JITSymbolResolver> Resolver) {

    LogicalDylibs.push_back(LogicalDylib());
    auto &LD = LogicalDylibs.back();
    LD.ExternalSymbolResolver = std::move(Resolver);
    LD.StubsMgr = CreateIndirectStubsManager();

    // Process each of the modules in this module set.
    addLogicalModule(LD, std::move(M));

    return std::prev(LogicalDylibs.end());
  }

  /// @brief Add extra modules to an existing logical module.
  void addExtraModule(ModuleHandleT H, std::shared_ptr<Module> M) {
    addLogicalModule(*H, std::move(M));
  }

  /// @brief Remove the module represented by the given handle.
  ///
  ///   This will remove all modules in the layers below that were derived from
  /// the module represented by H.
  void removeModule(ModuleHandleT H) {
    H->removeModulesFromBaseLayer(BaseLayer);
    LogicalDylibs.erase(H);
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto LDI = LogicalDylibs.begin(), LDE = LogicalDylibs.end();
         LDI != LDE; ++LDI) {
      if (auto Sym = LDI->StubsMgr->findStub(Name, ExportedSymbolsOnly))
        return Sym;
      if (auto Sym = findSymbolIn(LDI, Name, ExportedSymbolsOnly))
        return Sym;
    }
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  JITSymbol findSymbolIn(ModuleHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return H->findSymbol(BaseLayer, Name, ExportedSymbolsOnly);
  }

  /// @brief Update the stub for the given function to point at FnBodyAddr.
  /// This can be used to support re-optimization.
  /// @return true if the function exists and the stub is updated, false
  ///         otherwise.
  //
  // FIXME: We should track and free associated resources (unused compile
  //        callbacks, uncompiled IR, and no-longer-needed/reachable function
  //        implementations).
  // FIXME: Return Error once the JIT APIs are Errorized.
  bool updatePointer(std::string FuncName, JITTargetAddress FnBodyAddr) {
    //Find out which logical dylib contains our symbol
    auto LDI = LogicalDylibs.begin();
    for (auto LDE = LogicalDylibs.end(); LDI != LDE; ++LDI) {
      if (auto LMResources = LDI->getLogicalModuleResourcesForSymbol(FuncName, false)) {
        Module &SrcM = LMResources->SourceModule->getResource();
        std::string CalledFnName = mangle(FuncName, SrcM.getDataLayout());
        if (auto EC = LMResources->StubsMgr->updatePointer(CalledFnName, FnBodyAddr))
          return false;
        else
          return true;
      }
    }
    return false;
  }

private:

  void addLogicalModule(LogicalDylib &LD, std::shared_ptr<Module> SrcMPtr) {

    // Rename all static functions / globals to $static.X :
    // This will unique the names across all modules in the logical dylib,
    // simplifying symbol lookup.
    LD.StaticRenamer.rename(*SrcMPtr);

    // Bump the linkage and rename any anonymous/privote members in SrcM to
    // ensure that everything will resolve properly after we partition SrcM.
    makeAllSymbolsExternallyAccessible(*SrcMPtr);

    // Create a logical module handle for SrcM within the logical dylib.
    Module &SrcM = *SrcMPtr;
    auto LMId = LD.addSourceModule(std::move(SrcMPtr));

    // Create stub functions.
    const DataLayout &DL = SrcM.getDataLayout();
    {
      typename IndirectStubsMgrT::StubInitsMap StubInits;
      for (auto &F : SrcM) {
        // Skip declarations.
        if (F.isDeclaration())
          continue;

        // Skip weak functions for which we already have definitions.
        auto MangledName = mangle(F.getName(), DL);
        if (F.hasWeakLinkage() || F.hasLinkOnceLinkage())
          if (auto Sym = LD.findSymbol(BaseLayer, MangledName, false))
            continue;

        // Record all functions defined by this module.
        if (CloneStubsIntoPartitions)
          LD.getStubsToClone(LMId).insert(&F);

        // Create a callback, associate it with the stub for the function,
        // and set the compile action to compile the partition containing the
        // function.
        auto CCInfo = CompileCallbackMgr.getCompileCallback();
        StubInits[MangledName] =
          std::make_pair(CCInfo.getAddress(),
                         JITSymbolFlags::fromGlobalValue(F));
        CCInfo.setCompileAction([this, &LD, LMId, &F]() {
          return this->extractAndCompile(LD, LMId, F);
        });
      }

      auto EC = LD.StubsMgr->createStubs(StubInits);
      (void)EC;
      // FIXME: This should be propagated back to the user. Stub creation may
      //        fail for remote JITs.
      assert(!EC && "Error generating stubs");
    }

    // If this module doesn't contain any globals, aliases, or module flags then
    // we can bail out early and avoid the overhead of creating and managing an
    // empty globals module.
    if (SrcM.global_empty() && SrcM.alias_empty() &&
        !SrcM.getModuleFlagsMetadata())
      return;

    // Create the GlobalValues module.
    auto GVsM = llvm::make_unique<Module>((SrcM.getName() + ".globals").str(),
                                          SrcM.getContext());
    GVsM->setDataLayout(DL);

    ValueToValueMapTy VMap;

    // Clone global variable decls.
    for (auto &GV : SrcM.globals())
      if (!GV.isDeclaration() && !VMap.count(&GV))
        cloneGlobalVariableDecl(*GVsM, GV, &VMap);

    // And the aliases.
    for (auto &A : SrcM.aliases())
      if (!VMap.count(&A))
        cloneGlobalAliasDecl(*GVsM, A, VMap);

    // Clone the module flags.
    cloneModuleFlagsMetadata(*GVsM, SrcM, VMap);

    // Now we need to clone the GV and alias initializers.

    // Initializers may refer to functions declared (but not defined) in this
    // module. Build a materializer to clone decls on demand.
    auto Materializer = createLambdaMaterializer(
      [&LD, &GVsM](Value *V) -> Value* {
        if (auto *F = dyn_cast<Function>(V)) {
          // Decls in the original module just get cloned.
          if (F->isDeclaration())
            return cloneFunctionDecl(*GVsM, *F);

          // Definitions in the original module (which we have emitted stubs
          // for at this point) get turned into a constant alias to the stub
          // instead.
          const DataLayout &DL = GVsM->getDataLayout();
          std::string FName = mangle(F->getName(), DL);
          auto StubSym = LD.StubsMgr->findStub(FName, false);
          unsigned PtrBitWidth = DL.getPointerTypeSizeInBits(F->getType());
          ConstantInt *StubAddr =
            ConstantInt::get(GVsM->getContext(),
                             APInt(PtrBitWidth, StubSym.getAddress()));
          Constant *Init = ConstantExpr::getCast(Instruction::IntToPtr,
                                                 StubAddr, F->getType());
          return GlobalAlias::create(F->getFunctionType(),
                                     F->getType()->getAddressSpace(),
                                     F->getLinkage(), F->getName(),
                                     Init, GVsM.get());
        }
        // else....
        return nullptr;
      });

    // Clone the global variable initializers.
    for (auto &GV : SrcM.globals())
      if (!GV.isDeclaration())
        moveGlobalVariableInitializer(GV, VMap, &Materializer);

    // Clone the global alias initializers.
    for (auto &A : SrcM.aliases()) {
      auto *NewA = cast<GlobalAlias>(VMap[&A]);
      assert(NewA && "Alias not cloned?");
      Value *Init = MapValue(A.getAliasee(), VMap, RF_None, nullptr,
                             &Materializer);
      NewA->setAliasee(cast<Constant>(Init));
    }

    // Build a resolver for the globals module and add it to the base layer.
    auto GVsResolver = createLambdaResolver(
        [this, &LD](const std::string &Name) {
          if (auto Sym = LD.StubsMgr->findStub(Name, false))
            return Sym;
          if (auto Sym = LD.findSymbol(BaseLayer, Name, false))
            return Sym;
          return LD.ExternalSymbolResolver->findSymbolInLogicalDylib(Name);
        },
        [&LD](const std::string &Name) {
          return LD.ExternalSymbolResolver->findSymbol(Name);
        });

    auto GVsH = BaseLayer.addModule(std::move(GVsM), std::move(GVsResolver));
    LD.BaseLayerHandles.push_back(GVsH);
  }

  static std::string mangle(StringRef Name, const DataLayout &DL) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  JITTargetAddress
  extractAndCompile(LogicalDylib &LD,
                    typename LogicalDylib::SourceModuleHandle LMId,
                    Function &F) {
    Module &SrcM = LD.getSourceModule(LMId);

    // If F is a declaration we must already have compiled it.
    if (F.isDeclaration())
      return 0;

    // Grab the name of the function being called here.
    std::string CalledFnName = mangle(F.getName(), SrcM.getDataLayout());

    auto Part = Partition(F);
    auto PartH = emitPartition(LD, LMId, Part);

    JITTargetAddress CalledAddr = 0;
    for (auto *SubF : Part) {
      std::string FnName = mangle(SubF->getName(), SrcM.getDataLayout());
      auto FnBodySym = BaseLayer.findSymbolIn(PartH, FnName, false);
      assert(FnBodySym && "Couldn't find function body.");

      JITTargetAddress FnBodyAddr = FnBodySym.getAddress();

      // If this is the function we're calling record the address so we can
      // return it from this function.
      if (SubF == &F)
        CalledAddr = FnBodyAddr;

      // Update the function body pointer for the stub.
      if (auto EC = LD.StubsMgr->updatePointer(FnName, FnBodyAddr))
        return 0;
    }

    LD.BaseLayerHandles.push_back(PartH);

    return CalledAddr;
  }

  template <typename PartitionT>
  BaseLayerModuleHandleT
  emitPartition(LogicalDylib &LD,
                typename LogicalDylib::SourceModuleHandle LMId,
                const PartitionT &Part) {
    Module &SrcM = LD.getSourceModule(LMId);

    // Create the module.
    std::string NewName = SrcM.getName();
    for (auto *F : Part) {
      NewName += ".";
      NewName += F->getName();
    }

    auto M = llvm::make_unique<Module>(NewName, SrcM.getContext());
    M->setDataLayout(SrcM.getDataLayout());
    ValueToValueMapTy VMap;

    auto Materializer = createLambdaMaterializer([&LD, &LMId,
                                                  &M](Value *V) -> Value * {
      if (auto *GV = dyn_cast<GlobalVariable>(V))
        return cloneGlobalVariableDecl(*M, *GV);

      if (auto *F = dyn_cast<Function>(V)) {
        // Check whether we want to clone an available_externally definition.
        if (!LD.getStubsToClone(LMId).count(F))
          return cloneFunctionDecl(*M, *F);

        // Ok - we want an inlinable stub. For that to work we need a decl
        // for the stub pointer.
        auto *StubPtr = createImplPointer(*F->getType(), *M,
                                          F->getName() + "$stub_ptr", nullptr);
        auto *ClonedF = cloneFunctionDecl(*M, *F);
        makeStub(*ClonedF, *StubPtr);
        ClonedF->setLinkage(GlobalValue::AvailableExternallyLinkage);
        ClonedF->addFnAttr(Attribute::AlwaysInline);
        return ClonedF;
      }

      if (auto *A = dyn_cast<GlobalAlias>(V)) {
        auto *Ty = A->getValueType();
        if (Ty->isFunctionTy())
          return Function::Create(cast<FunctionType>(Ty),
                                  GlobalValue::ExternalLinkage, A->getName(),
                                  M.get());

        return new GlobalVariable(*M, Ty, false, GlobalValue::ExternalLinkage,
                                  nullptr, A->getName(), nullptr,
                                  GlobalValue::NotThreadLocal,
                                  A->getType()->getAddressSpace());
      }

      return nullptr;
    });

    // Create decls in the new module.
    for (auto *F : Part)
      cloneFunctionDecl(*M, *F, &VMap);

    // Move the function bodies.
    for (auto *F : Part)
      moveFunctionBody(*F, VMap, &Materializer);

    // Create memory manager and symbol resolver.
    auto Resolver = createLambdaResolver(
        [this, &LD](const std::string &Name) {
          if (auto Sym = LD.findSymbol(BaseLayer, Name, false))
            return Sym;
          return LD.ExternalSymbolResolver->findSymbolInLogicalDylib(Name);
        },
        [&LD](const std::string &Name) {
          return LD.ExternalSymbolResolver->findSymbol(Name);
        });

    return BaseLayer.addModule(std::move(M), std::move(Resolver));
  }

  BaseLayerT &BaseLayer;
  PartitioningFtor Partition;
  CompileCallbackMgrT &CompileCallbackMgr;
  IndirectStubsManagerBuilderT CreateIndirectStubsManager;

  LogicalDylibList LogicalDylibs;
  bool CloneStubsIntoPartitions;
};

} // end namespace orc

} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
