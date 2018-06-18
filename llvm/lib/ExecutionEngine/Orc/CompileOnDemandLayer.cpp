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
#include "llvm/Support/raw_ostream.h"

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

static std::unique_ptr<Module> extractGlobals(Module &M) {
  // FIXME: Add alias support.

  if (M.global_empty() && M.alias_empty() && !M.getModuleFlagsMetadata())
    return nullptr;

  auto GlobalsModule = llvm::make_unique<Module>(
      (M.getName() + ".globals").str(), M.getContext());
  GlobalsModule->setDataLayout(M.getDataLayout());

  ValueToValueMapTy VMap;

  for (auto &GV : M.globals())
    if (!GV.isDeclaration() && !VMap.count(&GV))
      cloneGlobalVariableDecl(*GlobalsModule, GV, &VMap);

  // Clone the module flags.
  cloneModuleFlagsMetadata(*GlobalsModule, M, VMap);

  auto Materializer = createLambdaValueMaterializer([&](Value *V) -> Value * {
    if (auto *F = dyn_cast<Function>(V))
      return cloneFunctionDecl(*GlobalsModule, *F);
    return nullptr;
  });

  // Move the global variable initializers.
  for (auto &GV : M.globals()) {
    if (!GV.isDeclaration())
      moveGlobalVariableInitializer(GV, VMap, &Materializer);
    GV.setInitializer(nullptr);
  }

  return GlobalsModule;
}

namespace llvm {
namespace orc {

class ExtractingIRMaterializationUnit : public IRMaterializationUnit {
public:
  ExtractingIRMaterializationUnit(
      ExecutionSession &ES, CompileOnDemandLayer2 &Parent,
      std::unique_ptr<Module> M,
      std::shared_ptr<SymbolResolver> BackingResolver)
      : IRMaterializationUnit(ES, std::move(M)), Parent(Parent),
        BackingResolver(std::move(BackingResolver)) {}

  ExtractingIRMaterializationUnit(
      std::unique_ptr<Module> M, SymbolFlagsMap SymbolFlags,
      SymbolNameToDefinitionMap SymbolToDefinition,
      CompileOnDemandLayer2 &Parent,
      std::shared_ptr<SymbolResolver> BackingResolver)
      : IRMaterializationUnit(std::move(M), std::move(SymbolFlags),
                              std::move(SymbolToDefinition)),
        Parent(Parent), BackingResolver(std::move(BackingResolver)) {}

private:
  void materialize(MaterializationResponsibility R) override {
    // FIXME: Need a 'notify lazy-extracting/emitting' callback to tie the
    //        extracted module key, extracted module, and source module key
    //        together. This could be used, for example, to provide a specific
    //        memory manager instance to the linking layer.

    // FIXME: The derived constructor should *only* look for the names of
    //        original function definitions in the target VSO. All other
    //        symbols should be looked up in the backing resolver.

    // Find the functions that have been requested.
    auto RequestedSymbols = R.getRequestedSymbols();

    // Extract them into a new module.
    auto ExtractedFunctionsModule =
        Parent.extractFunctions(*M, RequestedSymbols, SymbolToDefinition);

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
      R.delegate(llvm::make_unique<ExtractingIRMaterializationUnit>(
          std::move(M), std::move(DelegatedSymbolFlags),
          std::move(DelegatedSymbolToDefinition), Parent, BackingResolver));
    }

    Parent.emitExtractedFunctionsModule(
        std::move(R), std::move(ExtractedFunctionsModule), BackingResolver);
  }

  void discard(const VSO &V, SymbolStringPtr Name) override {
    // All original symbols were materialized by the CODLayer and should be
    // final. The function bodies provided by M should never be overridden.
    llvm_unreachable("Discard should never be called on an "
                     "ExtractingIRMaterializationUnit");
  }

  CompileOnDemandLayer2 &Parent;
  std::shared_ptr<SymbolResolver> BackingResolver;
};

CompileOnDemandLayer2::CompileOnDemandLayer2(
    ExecutionSession &ES, IRLayer &BaseLayer, JITCompileCallbackManager &CCMgr,
    IndirectStubsManagerBuilder BuildIndirectStubsManager,
    GetSymbolResolverFunction GetSymbolResolver,
    SetSymbolResolverFunction SetSymbolResolver,
    GetAvailableContextFunction GetAvailableContext)
    : IRLayer(ES), BaseLayer(BaseLayer), CCMgr(CCMgr),
      BuildIndirectStubsManager(std::move(BuildIndirectStubsManager)),
      GetSymbolResolver(std::move(GetSymbolResolver)),
      SetSymbolResolver(std::move(SetSymbolResolver)),
      GetAvailableContext(std::move(GetAvailableContext)) {}

Error CompileOnDemandLayer2::add(VSO &V, VModuleKey K,
                                 std::unique_ptr<Module> M) {
  makeAllSymbolsExternallyAccessible(*M);
  return IRLayer::add(V, K, std::move(M));
}

void CompileOnDemandLayer2::emit(MaterializationResponsibility R, VModuleKey K,
                                 std::unique_ptr<Module> M) {
  auto &ES = getExecutionSession();
  assert(M && "M should not be null");

  for (auto &GV : M->global_values())
    if (GV.hasWeakLinkage())
      GV.setLinkage(GlobalValue::ExternalLinkage);

  auto GlobalsModule = extractGlobals(*M);

  MangleAndInterner Mangle(ES, M->getDataLayout());

  // Delete the bodies of any available externally functions, rename the
  // rest, and build the compile callbacks.
  std::map<SymbolStringPtr, std::pair<JITTargetAddress, JITSymbolFlags>>
      StubCallbacksAndLinkages;
  auto &TargetVSO = R.getTargetVSO();

  for (auto &F : M->functions()) {
    if (F.isDeclaration())
      continue;

    if (F.hasAvailableExternallyLinkage()) {
      F.deleteBody();
      continue;
    }

    assert(F.hasName() && "Function should have a name");
    auto StubName = Mangle(F.getName());
    F.setName(F.getName() + "$body");
    auto BodyName = Mangle(F.getName());
    if (auto CallbackAddr = CCMgr.getCompileCallback(
            [BodyName, &TargetVSO, &ES]() -> JITTargetAddress {
              if (auto Sym = lookup({&TargetVSO}, BodyName))
                return Sym->getAddress();
              else {
                ES.reportError(Sym.takeError());
                return 0;
              }
            })) {
      auto Flags = JITSymbolFlags::fromGlobalValue(F);
      Flags &= ~JITSymbolFlags::Weak;
      StubCallbacksAndLinkages[std::move(StubName)] =
          std::make_pair(*CallbackAddr, Flags);
    } else {
      ES.reportError(CallbackAddr.takeError());
      R.failMaterialization();
      return;
    }
  }

  // Build the stub inits map.
  IndirectStubsManager::StubInitsMap StubInits;
  for (auto &KV : StubCallbacksAndLinkages)
    StubInits[*KV.first] = KV.second;

  // Build the function-body-extracting materialization unit.
  if (auto Err = R.getTargetVSO().define(
          llvm::make_unique<ExtractingIRMaterializationUnit>(
              ES, *this, std::move(M), GetSymbolResolver(K)))) {
    ES.reportError(std::move(Err));
    R.failMaterialization();
    return;
  }

  // Build the stubs.
  // FIXME: Remove function bodies materialization unit if stub creation fails.
  auto &StubsMgr = getStubsManager(TargetVSO);
  if (auto Err = StubsMgr.createStubs(StubInits)) {
    ES.reportError(std::move(Err));
    R.failMaterialization();
    return;
  }

  // Resolve and finalize stubs.
  SymbolMap ResolvedStubs;
  for (auto &KV : StubCallbacksAndLinkages) {
    if (auto Sym = StubsMgr.findStub(*KV.first, false))
      ResolvedStubs[KV.first] = Sym;
    else
      llvm_unreachable("Stub went missing");
  }

  R.resolve(ResolvedStubs);

  BaseLayer.emit(std::move(R), std::move(K), std::move(GlobalsModule));
}

IndirectStubsManager &CompileOnDemandLayer2::getStubsManager(const VSO &V) {
  std::lock_guard<std::mutex> Lock(CODLayerMutex);
  StubManagersMap::iterator I = StubsMgrs.find(&V);
  if (I == StubsMgrs.end())
    I = StubsMgrs.insert(std::make_pair(&V, BuildIndirectStubsManager())).first;
  return *I->second;
}

std::unique_ptr<Module> CompileOnDemandLayer2::extractFunctions(
    Module &M, const SymbolNameSet &SymbolNames,
    const SymbolNameToDefinitionMap &SymbolToDefinition) {
  assert(!SymbolNames.empty() && "Can not extract an empty function set");

  std::string ExtractedModName;
  {
    raw_string_ostream ExtractedModNameStream(ExtractedModName);
    ExtractedModNameStream << M.getName();
    for (auto &Name : SymbolNames)
      ExtractedModNameStream << "." << *Name;
  }

  auto ExtractedFunctionsModule =
      llvm::make_unique<Module>(ExtractedModName, GetAvailableContext());
  ExtractedFunctionsModule->setDataLayout(M.getDataLayout());

  ValueToValueMapTy VMap;

  auto Materializer = createLambdaValueMaterializer([&](Value *V) -> Value * {
    if (auto *F = dyn_cast<Function>(V))
      return cloneFunctionDecl(*ExtractedFunctionsModule, *F);
    else if (auto *GV = dyn_cast<GlobalVariable>(V))
      return cloneGlobalVariableDecl(*ExtractedFunctionsModule, *GV);
    return nullptr;
  });

  std::vector<std::pair<Function *, Function *>> OrigToNew;
  for (auto &FunctionName : SymbolNames) {
    assert(SymbolToDefinition.count(FunctionName) &&
           "No definition for symbol");
    auto *OrigF = cast<Function>(SymbolToDefinition.find(FunctionName)->second);
    auto *NewF = cloneFunctionDecl(*ExtractedFunctionsModule, *OrigF, &VMap);
    OrigToNew.push_back(std::make_pair(OrigF, NewF));
  }

  for (auto &KV : OrigToNew)
    moveFunctionBody(*KV.first, VMap, &Materializer, KV.second);

  return ExtractedFunctionsModule;
}

void CompileOnDemandLayer2::emitExtractedFunctionsModule(
    MaterializationResponsibility R, std::unique_ptr<Module> M,
    std::shared_ptr<SymbolResolver> Resolver) {
  auto &TargetVSO = R.getTargetVSO();
  auto K = getExecutionSession().allocateVModule();

  auto ExtractedFunctionsResolver = createSymbolResolver(
      [=](SymbolFlagsMap &Flags, const SymbolNameSet &Symbols) {
        return Resolver->lookupFlags(Flags, Symbols);
      },
      [=, &TargetVSO](std::shared_ptr<AsynchronousSymbolQuery> Query,
                      SymbolNameSet Symbols) {
        auto RemainingSymbols = TargetVSO.lookup(Query, std::move(Symbols));
        return Resolver->lookup(std::move(Query), std::move(RemainingSymbols));
      });

  SetSymbolResolver(K, std::move(ExtractedFunctionsResolver));
  BaseLayer.emit(std::move(R), std::move(K), std::move(M));
}

} // end namespace orc
} // end namespace llvm
