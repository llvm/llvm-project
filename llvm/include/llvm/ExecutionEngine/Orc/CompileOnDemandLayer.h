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

#include "IndirectionUtils.h"
#include "LambdaResolver.h"
#include "LogicalDylib.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <list>
#include <memory>
#include <set>

#include "llvm/Support/Debug.h"

namespace llvm {
namespace orc {

/// @brief Compile-on-demand layer.
///
///   When a module is added to this layer a stub is created for each of its
/// function definitions. The stubs and other global values are immediately
/// added to the layer below. When a stub is called it triggers the extraction
/// of the function body from the original module. The extracted body is then
/// compiled and executed.
template <typename BaseLayerT,
          typename CompileCallbackMgrT = JITCompileCallbackManagerBase,
          typename IndirectStubsMgrT = IndirectStubsManagerBase>
class CompileOnDemandLayer {
private:

  template <typename MaterializerFtor>
  class LambdaMaterializer final : public ValueMaterializer {
  public:
    LambdaMaterializer(MaterializerFtor M) : M(std::move(M)) {}
    Value* materializeValueFor(Value *V) final {
      return M(V);
    }
  private:
    MaterializerFtor M;
  };

  template <typename MaterializerFtor>
  LambdaMaterializer<MaterializerFtor>
  createLambdaMaterializer(MaterializerFtor M) {
    return LambdaMaterializer<MaterializerFtor>(std::move(M));
  }

  typedef typename BaseLayerT::ModuleSetHandleT BaseLayerModuleSetHandleT;

  class ModuleOwner {
  public:
    ModuleOwner() = default;
    ModuleOwner(const ModuleOwner&) = delete;
    ModuleOwner& operator=(const ModuleOwner&) = delete;
    virtual ~ModuleOwner() { }
    virtual Module& getModule() const = 0;
  };

  template <typename ModulePtrT>
  class ModuleOwnerImpl : public ModuleOwner {
  public:
    ModuleOwnerImpl(ModulePtrT ModulePtr) : ModulePtr(std::move(ModulePtr)) {}
    Module& getModule() const override { return *ModulePtr; }
  private:
    ModulePtrT ModulePtr;
  };

  template <typename ModulePtrT>
  std::unique_ptr<ModuleOwner> wrapOwnership(ModulePtrT ModulePtr) {
    return llvm::make_unique<ModuleOwnerImpl<ModulePtrT>>(std::move(ModulePtr));
  }

  struct LogicalModuleResources {
    std::unique_ptr<ModuleOwner> SourceModuleOwner;
    std::set<const Function*> StubsToClone;
    std::unique_ptr<IndirectStubsMgrT> StubsMgr;

    LogicalModuleResources() = default;

    // Explicit move constructor to make MSVC happy.
    LogicalModuleResources(LogicalModuleResources &&Other)
        : SourceModuleOwner(std::move(Other.SourceModuleOwner)),
          StubsToClone(std::move(Other.StubsToClone)),
          StubsMgr(std::move(Other.StubsMgr)) {}

    // Explicit move assignment to make MSVC happy.
    LogicalModuleResources& operator=(LogicalModuleResources &&Other) {
      SourceModuleOwner = std::move(Other.SourceModuleOwner);
      StubsToClone = std::move(Other.StubsToClone);
      StubsMgr = std::move(Other.StubsMgr);
    }

    JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
      if (Name.endswith("$stub_ptr") && !ExportedSymbolsOnly) {
        assert(!ExportedSymbolsOnly && "Stubs are never exported");
        return StubsMgr->findPointer(Name.drop_back(9));
      }
      return StubsMgr->findStub(Name, ExportedSymbolsOnly);
    }

  };



  struct LogicalDylibResources {
    typedef std::function<RuntimeDyld::SymbolInfo(const std::string&)>
      SymbolResolverFtor;
    SymbolResolverFtor ExternalSymbolResolver;
  };

  typedef LogicalDylib<BaseLayerT, LogicalModuleResources,
                       LogicalDylibResources> CODLogicalDylib;

  typedef typename CODLogicalDylib::LogicalModuleHandle LogicalModuleHandle;
  typedef std::list<CODLogicalDylib> LogicalDylibList;

public:

  /// @brief Handle to a set of loaded modules.
  typedef typename LogicalDylibList::iterator ModuleSetHandleT;

  /// @brief Module partitioning functor.
  typedef std::function<std::set<Function*>(Function&)> PartitioningFtor;

  /// @brief Builder for IndirectStubsManagers.
  typedef std::function<std::unique_ptr<IndirectStubsMgrT>()>
    IndirectStubsManagerBuilderT;

  /// @brief Construct a compile-on-demand layer instance.
  CompileOnDemandLayer(BaseLayerT &BaseLayer, PartitioningFtor Partition,
                       CompileCallbackMgrT &CallbackMgr,
                       IndirectStubsManagerBuilderT CreateIndirectStubsManager,
                       bool CloneStubsIntoPartitions = true)
      : BaseLayer(BaseLayer),  Partition(Partition),
        CompileCallbackMgr(CallbackMgr),
        CreateIndirectStubsManager(std::move(CreateIndirectStubsManager)),
        CloneStubsIntoPartitions(CloneStubsIntoPartitions) {}

  /// @brief Add a module to the compile-on-demand layer.
  template <typename ModuleSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ModuleSetHandleT addModuleSet(ModuleSetT Ms,
                                MemoryManagerPtrT MemMgr,
                                SymbolResolverPtrT Resolver) {

    assert(MemMgr == nullptr &&
           "User supplied memory managers not supported with COD yet.");

    LogicalDylibs.push_back(CODLogicalDylib(BaseLayer));
    auto &LDResources = LogicalDylibs.back().getDylibResources();

    LDResources.ExternalSymbolResolver =
      [Resolver](const std::string &Name) {
        return Resolver->findSymbol(Name);
      };

    // Process each of the modules in this module set.
    for (auto &M : Ms)
      addLogicalModule(LogicalDylibs.back(), std::move(M));

    return std::prev(LogicalDylibs.end());
  }

  /// @brief Remove the module represented by the given handle.
  ///
  ///   This will remove all modules in the layers below that were derived from
  /// the module represented by H.
  void removeModuleSet(ModuleSetHandleT H) {
    LogicalDylibs.erase(H);
  }

  /// @brief Search for the given named symbol.
  /// @param Name The name of the symbol to search for.
  /// @param ExportedSymbolsOnly If true, search only for exported symbols.
  /// @return A handle for the given named symbol, if it exists.
  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto LDI = LogicalDylibs.begin(), LDE = LogicalDylibs.end();
         LDI != LDE; ++LDI)
      if (auto Symbol = findSymbolIn(LDI, Name, ExportedSymbolsOnly))
        return Symbol;
    return BaseLayer.findSymbol(Name, ExportedSymbolsOnly);
  }

  /// @brief Get the address of a symbol provided by this layer, or some layer
  ///        below this one.
  JITSymbol findSymbolIn(ModuleSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    return H->findSymbol(Name, ExportedSymbolsOnly);
  }

private:

  template <typename ModulePtrT>
  void addLogicalModule(CODLogicalDylib &LD, ModulePtrT SrcMPtr) {

    // Bump the linkage and rename any anonymous/privote members in SrcM to
    // ensure that everything will resolve properly after we partition SrcM.
    makeAllSymbolsExternallyAccessible(*SrcMPtr);

    // Create a logical module handle for SrcM within the logical dylib.
    auto LMH = LD.createLogicalModule();
    auto &LMResources =  LD.getLogicalModuleResources(LMH);

    LMResources.SourceModuleOwner = wrapOwnership(std::move(SrcMPtr));

    Module &SrcM = LMResources.SourceModuleOwner->getModule();

    // Create the GlobalValues module.
    const DataLayout &DL = SrcM.getDataLayout();
    auto GVsM = llvm::make_unique<Module>((SrcM.getName() + ".globals").str(),
                                          SrcM.getContext());
    GVsM->setDataLayout(DL);

    // Create function stubs.
    ValueToValueMapTy VMap;
    {
      typename IndirectStubsMgrT::StubInitsMap StubInits;
      for (auto &F : SrcM) {
        // Skip declarations.
        if (F.isDeclaration())
          continue;

        // Record all functions defined by this module.
        if (CloneStubsIntoPartitions)
          LMResources.StubsToClone.insert(&F);

        // Create a callback, associate it with the stub for the function,
        // and set the compile action to compile the partition containing the
        // function.
        auto CCInfo = CompileCallbackMgr.getCompileCallback();
        StubInits[mangle(F.getName(), DL)] =
          std::make_pair(CCInfo.getAddress(),
                         JITSymbolBase::flagsFromGlobalValue(F));
        CCInfo.setCompileAction([this, &LD, LMH, &F]() {
          return this->extractAndCompile(LD, LMH, F);
        });
      }

      LMResources.StubsMgr = CreateIndirectStubsManager();
      auto EC = LMResources.StubsMgr->createStubs(StubInits);
      (void)EC;
      // FIXME: This should be propagated back to the user. Stub creation may
      //        fail for remote JITs.
      assert(!EC && "Error generating stubs");
    }

    // Clone global variable decls.
    for (auto &GV : SrcM.globals())
      if (!GV.isDeclaration() && !VMap.count(&GV))
        cloneGlobalVariableDecl(*GVsM, GV, &VMap);

    // And the aliases.
    for (auto &A : SrcM.aliases())
      if (!VMap.count(&A))
        cloneGlobalAliasDecl(*GVsM, A, VMap);

    // Now we need to clone the GV and alias initializers.

    // Initializers may refer to functions declared (but not defined) in this
    // module. Build a materializer to clone decls on demand.
    auto Materializer = createLambdaMaterializer(
      [this, &GVsM, &LMResources](Value *V) -> Value* {
        if (auto *F = dyn_cast<Function>(V)) {
          // Decls in the original module just get cloned.
          if (F->isDeclaration())
            return cloneFunctionDecl(*GVsM, *F);

          // Definitions in the original module (which we have emitted stubs
          // for at this point) get turned into a constant alias to the stub
          // instead.
          const DataLayout &DL = GVsM->getDataLayout();
          std::string FName = mangle(F->getName(), DL);
          auto StubSym = LMResources.StubsMgr->findStub(FName, false);
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
        [&LD, LMH](const std::string &Name) {
          auto &LMResources = LD.getLogicalModuleResources(LMH);
          if (auto Sym = LMResources.StubsMgr->findStub(Name, false))
            return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());
          return LD.getDylibResources().ExternalSymbolResolver(Name);
        },
        [](const std::string &Name) {
          return RuntimeDyld::SymbolInfo(nullptr);
        });

    std::vector<std::unique_ptr<Module>> GVsMSet;
    GVsMSet.push_back(std::move(GVsM));
    auto GVsH =
      BaseLayer.addModuleSet(std::move(GVsMSet),
                             llvm::make_unique<SectionMemoryManager>(),
                             std::move(GVsResolver));
    LD.addToLogicalModule(LMH, GVsH);
  }

  static std::string mangle(StringRef Name, const DataLayout &DL) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  TargetAddress extractAndCompile(CODLogicalDylib &LD,
                                  LogicalModuleHandle LMH,
                                  Function &F) {
    auto &LMResources = LD.getLogicalModuleResources(LMH);
    Module &SrcM = LMResources.SourceModuleOwner->getModule();

    // If F is a declaration we must already have compiled it.
    if (F.isDeclaration())
      return 0;

    // Grab the name of the function being called here.
    std::string CalledFnName = mangle(F.getName(), SrcM.getDataLayout());

    auto Part = Partition(F);
    auto PartH = emitPartition(LD, LMH, Part);

    TargetAddress CalledAddr = 0;
    for (auto *SubF : Part) {
      std::string FnName = mangle(SubF->getName(), SrcM.getDataLayout());
      auto FnBodySym = BaseLayer.findSymbolIn(PartH, FnName, false);
      assert(FnBodySym && "Couldn't find function body.");

      TargetAddress FnBodyAddr = FnBodySym.getAddress();

      // If this is the function we're calling record the address so we can
      // return it from this function.
      if (SubF == &F)
        CalledAddr = FnBodyAddr;

      // Update the function body pointer for the stub.
      if (auto EC = LMResources.StubsMgr->updatePointer(FnName, FnBodyAddr))
        return 0;
    }

    return CalledAddr;
  }

  template <typename PartitionT>
  BaseLayerModuleSetHandleT emitPartition(CODLogicalDylib &LD,
                                          LogicalModuleHandle LMH,
                                          const PartitionT &Part) {
    auto &LMResources = LD.getLogicalModuleResources(LMH);
    Module &SrcM = LMResources.SourceModuleOwner->getModule();

    // Create the module.
    std::string NewName = SrcM.getName();
    for (auto *F : Part) {
      NewName += ".";
      NewName += F->getName();
    }

    auto M = llvm::make_unique<Module>(NewName, SrcM.getContext());
    M->setDataLayout(SrcM.getDataLayout());
    ValueToValueMapTy VMap;

    auto Materializer = createLambdaMaterializer([this, &LMResources, &M,
                                                  &VMap](Value *V) -> Value * {
      if (auto *GV = dyn_cast<GlobalVariable>(V))
        return cloneGlobalVariableDecl(*M, *GV);

      if (auto *F = dyn_cast<Function>(V)) {
        // Check whether we want to clone an available_externally definition.
        if (!LMResources.StubsToClone.count(F))
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
    auto MemMgr = llvm::make_unique<SectionMemoryManager>();
    auto Resolver = createLambdaResolver(
        [this, &LD, LMH](const std::string &Name) {
          if (auto Symbol = LD.findSymbolInternally(LMH, Name))
            return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                           Symbol.getFlags());
          return LD.getDylibResources().ExternalSymbolResolver(Name);
        },
        [this, &LD, LMH](const std::string &Name) {
          if (auto Symbol = LD.findSymbolInternally(LMH, Name))
            return RuntimeDyld::SymbolInfo(Symbol.getAddress(),
                                           Symbol.getFlags());
          return RuntimeDyld::SymbolInfo(nullptr);
        });
    std::vector<std::unique_ptr<Module>> PartMSet;
    PartMSet.push_back(std::move(M));
    return BaseLayer.addModuleSet(std::move(PartMSet), std::move(MemMgr),
                                  std::move(Resolver));
  }

  BaseLayerT &BaseLayer;
  PartitioningFtor Partition;
  CompileCallbackMgrT &CompileCallbackMgr;
  IndirectStubsManagerBuilderT CreateIndirectStubsManager;

  LogicalDylibList LogicalDylibs;
  bool CloneStubsIntoPartitions;
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_COMPILEONDEMANDLAYER_H
