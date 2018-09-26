//===----- CompileOnDemandLayer.cpp - Lazily emit IR on first call --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

template <typename MaterializerFtor>
class LambdaValueMaterializer final : public ValueMaterializer {
public:
  LambdaValueMaterializer(MaterializerFtor M) : M(std::move(M)) {}

  Value *materialize(Value *V) final { return M(V); }

private:
  MaterializerFtor M;
};

template <typename MaterializerFtor>
LambdaValueMaterializer<MaterializerFtor>
createLambdaValueMaterializer(MaterializerFtor M) {
  return LambdaValueMaterializer<MaterializerFtor>(std::move(M));
}
} // namespace

static void extractAliases(MaterializationResponsibility &R, Module &M,
                           MangleAndInterner &Mangle) {
  SymbolAliasMap Aliases;

  std::vector<GlobalAlias *> ModAliases;
  for (auto &A : M.aliases())
    ModAliases.push_back(&A);

  for (auto *A : ModAliases) {
    Constant *Aliasee = A->getAliasee();
    assert(A->hasName() && "Anonymous alias?");
    assert(Aliasee->hasName() && "Anonymous aliasee");
    std::string AliasName = A->getName();

    Aliases[Mangle(AliasName)] = SymbolAliasMapEntry(
        {Mangle(Aliasee->getName()), JITSymbolFlags::fromGlobalValue(*A)});

    if (isa<Function>(Aliasee)) {
      auto *F = cloneFunctionDecl(M, *cast<Function>(Aliasee));
      A->replaceAllUsesWith(F);
      A->eraseFromParent();
      F->setName(AliasName);
    } else if (isa<GlobalValue>(Aliasee)) {
      auto *G = cloneGlobalVariableDecl(M, *cast<GlobalVariable>(Aliasee));
      A->replaceAllUsesWith(G);
      A->eraseFromParent();
      G->setName(AliasName);
    }
  }

  R.replace(symbolAliases(std::move(Aliases)));
}

static ThreadSafeModule extractAndClone(ThreadSafeModule &TSM, StringRef Suffix,
                                        GVPredicate ShouldCloneDefinition) {

  auto DeleteClonedDefsAndPromoteDeclLinkages = [](GlobalValue &GV) {
    // Delete the definition and bump the linkage in the source module.
    if (isa<Function>(GV)) {
      auto &F = cast<Function>(GV);
      F.deleteBody();
      F.setPersonalityFn(nullptr);
    } else if (isa<GlobalVariable>(GV)) {
      cast<GlobalVariable>(GV).setInitializer(nullptr);
    } else
      llvm_unreachable("Unsupported global type");

    GV.setLinkage(GlobalValue::ExternalLinkage);
  };

  auto NewTSMod = cloneToNewContext(TSM, ShouldCloneDefinition,
                                    DeleteClonedDefsAndPromoteDeclLinkages);
  auto &M = *NewTSMod.getModule();
  M.setModuleIdentifier((M.getModuleIdentifier() + Suffix).str());

  return NewTSMod;
}

static ThreadSafeModule extractGlobals(ThreadSafeModule &TSM) {
  return extractAndClone(TSM, ".globals", [](const GlobalValue &GV) {
    return isa<GlobalVariable>(GV);
  });
}

namespace llvm {
namespace orc {

class ExtractingIRMaterializationUnit : public IRMaterializationUnit {
public:
  ExtractingIRMaterializationUnit(ExecutionSession &ES,
                                  CompileOnDemandLayer2 &Parent,
                                  ThreadSafeModule TSM)
      : IRMaterializationUnit(ES, std::move(TSM)), Parent(Parent) {}

  ExtractingIRMaterializationUnit(ThreadSafeModule TSM,
                                  SymbolFlagsMap SymbolFlags,
                                  SymbolNameToDefinitionMap SymbolToDefinition,
                                  CompileOnDemandLayer2 &Parent)
      : IRMaterializationUnit(std::move(TSM), std::move(SymbolFlags),
                              std::move(SymbolToDefinition)),
        Parent(Parent) {}

private:
  void materialize(MaterializationResponsibility R) override {
    // FIXME: Need a 'notify lazy-extracting/emitting' callback to tie the
    //        extracted module key, extracted module, and source module key
    //        together. This could be used, for example, to provide a specific
    //        memory manager instance to the linking layer.

    auto RequestedSymbols = R.getRequestedSymbols();

    // Extract the requested functions into a new module.
    ThreadSafeModule ExtractedFunctionsModule;
    if (!RequestedSymbols.empty()) {
      std::string Suffix;
      std::set<const GlobalValue *> FunctionsToClone;
      for (auto &Name : RequestedSymbols) {
        auto I = SymbolToDefinition.find(Name);
        assert(I != SymbolToDefinition.end() && I->second != nullptr &&
               "Should have a non-null definition");
        FunctionsToClone.insert(I->second);
        Suffix += ".";
        Suffix += *Name;
      }

      std::lock_guard<std::mutex> Lock(SourceModuleMutex);
      ExtractedFunctionsModule =
          extractAndClone(TSM, Suffix, [&](const GlobalValue &GV) -> bool {
            return FunctionsToClone.count(&GV);
          });
    }

    // Build a new ExtractingIRMaterializationUnit to delegate the unrequested
    // symbols to.
    SymbolFlagsMap DelegatedSymbolFlags;
    IRMaterializationUnit::SymbolNameToDefinitionMap
        DelegatedSymbolToDefinition;
    for (auto &KV : SymbolToDefinition) {
      if (RequestedSymbols.count(KV.first))
        continue;
      DelegatedSymbolFlags[KV.first] =
          JITSymbolFlags::fromGlobalValue(*KV.second);
      DelegatedSymbolToDefinition[KV.first] = KV.second;
    }

    if (!DelegatedSymbolFlags.empty()) {
      assert(DelegatedSymbolFlags.size() ==
                 DelegatedSymbolToDefinition.size() &&
             "SymbolFlags and SymbolToDefinition should have the same number "
             "of entries");
      R.replace(llvm::make_unique<ExtractingIRMaterializationUnit>(
          std::move(TSM), std::move(DelegatedSymbolFlags),
          std::move(DelegatedSymbolToDefinition), Parent));
    }

    if (ExtractedFunctionsModule)
      Parent.emitExtractedFunctionsModule(std::move(R),
                                          std::move(ExtractedFunctionsModule));
  }

  void discard(const JITDylib &V, SymbolStringPtr Name) override {
    // All original symbols were materialized by the CODLayer and should be
    // final. The function bodies provided by M should never be overridden.
    llvm_unreachable("Discard should never be called on an "
                     "ExtractingIRMaterializationUnit");
  }

  mutable std::mutex SourceModuleMutex;
  CompileOnDemandLayer2 &Parent;
};

CompileOnDemandLayer2::CompileOnDemandLayer2(
    ExecutionSession &ES, IRLayer &BaseLayer, LazyCallThroughManager &LCTMgr,
    IndirectStubsManagerBuilder BuildIndirectStubsManager)
    : IRLayer(ES), BaseLayer(BaseLayer), LCTMgr(LCTMgr),
      BuildIndirectStubsManager(std::move(BuildIndirectStubsManager)) {}

Error CompileOnDemandLayer2::add(JITDylib &V, VModuleKey K,
                                 ThreadSafeModule TSM) {
  return IRLayer::add(V, K, std::move(TSM));
}

void CompileOnDemandLayer2::emit(MaterializationResponsibility R, VModuleKey K,
                                 ThreadSafeModule TSM) {
  auto &ES = getExecutionSession();
  assert(TSM && "M should not be null");
  auto &M = *TSM.getModule();

  for (auto &GV : M.global_values())
    if (GV.hasWeakLinkage())
      GV.setLinkage(GlobalValue::ExternalLinkage);

  MangleAndInterner Mangle(ES, M.getDataLayout());

  extractAliases(R, *TSM.getModule(), Mangle);

  auto GlobalsModule = extractGlobals(TSM);

  // Delete the bodies of any available externally functions and build the
  // lazy reexports alias map.
  std::map<SymbolStringPtr, std::pair<JITTargetAddress, JITSymbolFlags>>
      StubCallbacksAndLinkages;
  auto &TargetJD = R.getTargetJITDylib();
  auto &Resources = getPerDylibResources(TargetJD);
  auto &ImplD = Resources.getImplDylib();

  SymbolAliasMap LazyReexports;
  for (auto &F : M.functions()) {
    if (F.isDeclaration())
      continue;

    if (F.hasAvailableExternallyLinkage()) {
      F.deleteBody();
      F.setPersonalityFn(nullptr);
      continue;
    }

    auto Flags = JITSymbolFlags::fromGlobalValue(F);
    assert(Flags.isCallable() && "Non-callable definition in functions module");

    auto MangledName = Mangle(F.getName());
    LazyReexports[MangledName] = SymbolAliasMapEntry(MangledName, Flags);
  }

  // Add the functions module to the implementation dylib using an extracting
  // materialization unit.
  if (auto Err =
          ImplD.define(llvm::make_unique<ExtractingIRMaterializationUnit>(
              ES, *this, std::move(TSM)))) {
    ES.reportError(std::move(Err));
    R.failMaterialization();
    return;
  }

  // Handle responsibility for function symbols by returning lazy reexports.
  auto &ISMgr = Resources.getISManager();
  R.replace(lazyReexports(LCTMgr, ISMgr, ImplD, LazyReexports));

  BaseLayer.emit(std::move(R), std::move(K), std::move(GlobalsModule));
}

CompileOnDemandLayer2::PerDylibResources &
CompileOnDemandLayer2::getPerDylibResources(JITDylib &TargetD) {
  auto I = DylibResources.find(&TargetD);
  if (I == DylibResources.end()) {
    auto &ImplD =
        getExecutionSession().createJITDylib(TargetD.getName() + ".impl");
    TargetD.withSearchOrderDo([&](const JITDylibList &TargetSearchOrder) {
      ImplD.setSearchOrder(TargetSearchOrder, false);
    });
    PerDylibResources PDR(ImplD, BuildIndirectStubsManager());
    I = DylibResources.insert(std::make_pair(&TargetD, std::move(PDR))).first;
  }

  return I->second;
}

void CompileOnDemandLayer2::emitExtractedFunctionsModule(
    MaterializationResponsibility R, ThreadSafeModule TSM) {
  auto K = getExecutionSession().allocateVModule();
  BaseLayer.emit(std::move(R), std::move(K), std::move(TSM));
}

} // end namespace orc
} // end namespace llvm
