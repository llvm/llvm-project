//===----- CompileOnDemandLayer.cpp - Lazily emit IR on first call --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

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

static std::unique_ptr<Module>
extractAndClone(Module &M, LLVMContext &NewContext, StringRef Suffix,
                function_ref<bool(const GlobalValue *)> ShouldCloneDefinition) {
  SmallVector<char, 1> ClonedModuleBuffer;

  {
    std::set<GlobalValue *> ClonedDefsInSrc;
    ValueToValueMapTy VMap;
    auto Tmp = CloneModule(M, VMap, [&](const GlobalValue *GV) {
      if (ShouldCloneDefinition(GV)) {
        ClonedDefsInSrc.insert(const_cast<GlobalValue *>(GV));
        return true;
      }
      return false;
    });

    for (auto *GV : ClonedDefsInSrc) {
      // Delete the definition and bump the linkage in the source module.
      if (isa<Function>(GV)) {
        auto &F = *cast<Function>(GV);
        F.deleteBody();
        F.setPersonalityFn(nullptr);
      } else if (isa<GlobalVariable>(GV)) {
        cast<GlobalVariable>(GV)->setInitializer(nullptr);
      } else
        llvm_unreachable("Unsupported global type");

      GV->setLinkage(GlobalValue::ExternalLinkage);
    }

    BitcodeWriter BCWriter(ClonedModuleBuffer);

    BCWriter.writeModule(*Tmp);
    BCWriter.writeSymtab();
    BCWriter.writeStrtab();
  }

  MemoryBufferRef ClonedModuleBufferRef(
      StringRef(ClonedModuleBuffer.data(), ClonedModuleBuffer.size()),
      "cloned module buffer");

  auto ClonedModule =
      cantFail(parseBitcodeFile(ClonedModuleBufferRef, NewContext));
  ClonedModule->setModuleIdentifier((M.getName() + Suffix).str());
  return ClonedModule;
}

static std::unique_ptr<Module> extractGlobals(Module &M,
                                              LLVMContext &NewContext) {
  return extractAndClone(M, NewContext, ".globals", [](const GlobalValue *GV) {
    return isa<GlobalVariable>(GV);
  });
}

namespace llvm {
namespace orc {

class ExtractingIRMaterializationUnit : public IRMaterializationUnit {
public:
  ExtractingIRMaterializationUnit(ExecutionSession &ES,
                                  CompileOnDemandLayer2 &Parent,
                                  std::unique_ptr<Module> M)
      : IRMaterializationUnit(ES, std::move(M)), Parent(Parent) {}

  ExtractingIRMaterializationUnit(std::unique_ptr<Module> M,
                                  SymbolFlagsMap SymbolFlags,
                                  SymbolNameToDefinitionMap SymbolToDefinition,
                                  CompileOnDemandLayer2 &Parent)
      : IRMaterializationUnit(std::move(M), std::move(SymbolFlags),
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
    std::unique_ptr<Module> ExtractedFunctionsModule;
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
          extractAndClone(*M, Parent.GetAvailableContext(), Suffix,
                          [&](const GlobalValue *GV) -> bool {
                            return FunctionsToClone.count(GV);
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
          std::move(M), std::move(DelegatedSymbolFlags),
          std::move(DelegatedSymbolToDefinition), Parent));
    }

    if (ExtractedFunctionsModule)
      Parent.emitExtractedFunctionsModule(std::move(R),
                                          std::move(ExtractedFunctionsModule));
  }

  void discard(const VSO &V, SymbolStringPtr Name) override {
    // All original symbols were materialized by the CODLayer and should be
    // final. The function bodies provided by M should never be overridden.
    llvm_unreachable("Discard should never be called on an "
                     "ExtractingIRMaterializationUnit");
  }

  mutable std::mutex SourceModuleMutex;
  CompileOnDemandLayer2 &Parent;
};

CompileOnDemandLayer2::CompileOnDemandLayer2(
    ExecutionSession &ES, IRLayer &BaseLayer, JITCompileCallbackManager &CCMgr,
    IndirectStubsManagerBuilder BuildIndirectStubsManager,
    GetAvailableContextFunction GetAvailableContext)
    : IRLayer(ES), BaseLayer(BaseLayer), CCMgr(CCMgr),
      BuildIndirectStubsManager(std::move(BuildIndirectStubsManager)),
      GetAvailableContext(std::move(GetAvailableContext)) {}

Error CompileOnDemandLayer2::add(VSO &V, VModuleKey K,
                                 std::unique_ptr<Module> M) {
  return IRLayer::add(V, K, std::move(M));
}

void CompileOnDemandLayer2::emit(MaterializationResponsibility R, VModuleKey K,
                                 std::unique_ptr<Module> M) {
  auto &ES = getExecutionSession();
  assert(M && "M should not be null");

  for (auto &GV : M->global_values())
    if (GV.hasWeakLinkage())
      GV.setLinkage(GlobalValue::ExternalLinkage);

  MangleAndInterner Mangle(ES, M->getDataLayout());

  extractAliases(R, *M, Mangle);

  auto GlobalsModule = extractGlobals(*M, GetAvailableContext());

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
      F.setPersonalityFn(nullptr);
      continue;
    }

    assert(F.hasName() && "Function should have a name");
    std::string StubUnmangledName = F.getName();
    F.setName(F.getName() + "$body");
    auto StubDecl = cloneFunctionDecl(*M, F);
    StubDecl->setName(StubUnmangledName);
    StubDecl->setPersonalityFn(nullptr);
    StubDecl->setLinkage(GlobalValue::ExternalLinkage);
    F.replaceAllUsesWith(StubDecl);

    auto StubName = Mangle(StubUnmangledName);
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
          llvm::make_unique<ExtractingIRMaterializationUnit>(ES, *this,
                                                             std::move(M)))) {
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

void CompileOnDemandLayer2::emitExtractedFunctionsModule(
    MaterializationResponsibility R, std::unique_ptr<Module> M) {
  auto K = getExecutionSession().allocateVModule();
  BaseLayer.emit(std::move(R), std::move(K), std::move(M));
}

} // end namespace orc
} // end namespace llvm
