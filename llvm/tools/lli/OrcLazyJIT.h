//===- OrcLazyJIT.h - Basic Orc-based JIT for lazy execution ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simple Orc-based JIT. Uses the compile-on-demand layer to break up and
// lazily compile modules.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLI_ORCLAZYJIT_H
#define LLVM_TOOLS_LLI_ORCLAZYJIT_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace llvm {

class OrcLazyJIT {
public:

  using CompileCallbackMgr = orc::JITCompileCallbackManager;
  using ObjLayerT = orc::RTDyldObjectLinkingLayer;
  using CompileLayerT = orc::IRCompileLayer<ObjLayerT, orc::SimpleCompiler>;
  using TransformFtor =
      std::function<std::unique_ptr<Module>(std::unique_ptr<Module>)>;
  using IRDumpLayerT = orc::IRTransformLayer<CompileLayerT, TransformFtor>;
  using CODLayerT = orc::CompileOnDemandLayer<IRDumpLayerT, CompileCallbackMgr>;
  using IndirectStubsManagerBuilder = CODLayerT::IndirectStubsManagerBuilderT;

  OrcLazyJIT(std::unique_ptr<TargetMachine> TM,
             std::unique_ptr<CompileCallbackMgr> CCMgr,
             IndirectStubsManagerBuilder IndirectStubsMgrBuilder,
             bool InlineStubs)
      : TM(std::move(TM)),
        DL(this->TM->createDataLayout()),
        CCMgr(std::move(CCMgr)),
        ObjectLayer(ES,
                    [this](orc::VModuleKey K) {
                      auto ResolverI = Resolvers.find(K);
                      assert(ResolverI != Resolvers.end() &&
                             "Missing resolver for module K");
                      auto Resolver = std::move(ResolverI->second);
                      Resolvers.erase(ResolverI);
                      return ObjLayerT::Resources{
                          std::make_shared<SectionMemoryManager>(),
                          std::move(Resolver)};
                    }),
        CompileLayer(ObjectLayer, orc::SimpleCompiler(*this->TM)),
        IRDumpLayer(CompileLayer, createDebugDumper()),
        CODLayer(
            ES, IRDumpLayer,
            [&](orc::VModuleKey K) {
              auto ResolverI = Resolvers.find(K);
              assert(ResolverI != Resolvers.end() &&
                     "Missing resolver for module K");
              auto Resolver = std::move(ResolverI->second);
              Resolvers.erase(ResolverI);
              return Resolver;
            },
            [&](orc::VModuleKey K, std::shared_ptr<orc::SymbolResolver> R) {
              assert(!Resolvers.count(K) && "Resolver already present");
              Resolvers[K] = std::move(R);
            },
            extractSingleFunction, *this->CCMgr,
            std::move(IndirectStubsMgrBuilder), InlineStubs),
        CXXRuntimeOverrides(
            [this](const std::string &S) { return mangle(S); }) {}

  ~OrcLazyJIT() {
    // Run any destructors registered with __cxa_atexit.
    CXXRuntimeOverrides.runDestructors();
    // Run any IR destructors.
    for (auto &DtorRunner : IRStaticDestructorRunners)
      if (auto Err = DtorRunner.runViaLayer(CODLayer)) {
        // FIXME: OrcLazyJIT should probably take a "shutdownError" callback to
        //        report these errors on.
        report_fatal_error(std::move(Err));
      }
  }

  Error addModule(std::unique_ptr<Module> M) {
    if (M->getDataLayout().isDefault())
      M->setDataLayout(DL);

    // Rename, bump linkage and record static constructors and destructors.
    // We have to do this before we hand over ownership of the module to the
    // JIT.
    std::vector<std::string> CtorNames, DtorNames;
    {
      unsigned CtorId = 0, DtorId = 0;
      for (auto Ctor : orc::getConstructors(*M)) {
        std::string NewCtorName = ("$static_ctor." + Twine(CtorId++)).str();
        Ctor.Func->setName(NewCtorName);
        Ctor.Func->setLinkage(GlobalValue::ExternalLinkage);
        Ctor.Func->setVisibility(GlobalValue::HiddenVisibility);
        CtorNames.push_back(mangle(NewCtorName));
      }
      for (auto Dtor : orc::getDestructors(*M)) {
        std::string NewDtorName = ("$static_dtor." + Twine(DtorId++)).str();
        Dtor.Func->setLinkage(GlobalValue::ExternalLinkage);
        Dtor.Func->setVisibility(GlobalValue::HiddenVisibility);
        DtorNames.push_back(mangle(Dtor.Func->getName()));
        Dtor.Func->setName(NewDtorName);
      }
    }

    // Symbol resolution order:
    //   1) Search the JIT symbols.
    //   2) Check for C++ runtime overrides.
    //   3) Search the host process (LLI)'s symbol table.
    if (!ModulesKey) {
      auto LegacyLookupInDylib = [this](const std::string &Name) -> JITSymbol {
        if (auto Sym = CODLayer.findSymbol(Name, true))
          return Sym;
        else if (auto Err = Sym.takeError())
          return std::move(Err);
        return CXXRuntimeOverrides.searchOverrides(Name);
      };

      auto LegacyLookup =
          [LegacyLookupInDylib](const std::string &Name) -> JITSymbol {
        if (auto Sym = LegacyLookupInDylib(Name))
          return Sym;
        else if (auto Err = Sym.takeError())
          return std::move(Err);

        if (auto Addr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
          return JITSymbol(Addr, JITSymbolFlags::Exported);

        return nullptr;
      };

      ModulesKey = ES.allocateVModule();
      assert(!Resolvers.count(*ModulesKey) && "Resolver already present");
      Resolvers[*ModulesKey] = orc::createSymbolResolver(
          [LegacyLookupInDylib](orc::SymbolFlagsMap &SymbolFlags,
                                const orc::SymbolNameSet &Symbols) {
            auto NotFoundViaLegacyLookup = lookupFlagsWithLegacyFn(
                SymbolFlags, Symbols, LegacyLookupInDylib);
            if (!NotFoundViaLegacyLookup) {
              logAllUnhandledErrors(NotFoundViaLegacyLookup.takeError(), errs(),
                                    "OrcLazyJIT lookupFlags error: ");
              SymbolFlags.clear();
              return orc::SymbolNameSet();
            }
            return std::move(*NotFoundViaLegacyLookup);
          },
          [LegacyLookup](std::shared_ptr<orc::AsynchronousSymbolQuery> Query,
                         orc::SymbolNameSet Symbols) {
            return lookupWithLegacyFn(*Query, Symbols, LegacyLookup);
          });

      // Add the module to the JIT.
      if (auto Err = CODLayer.addModule(*ModulesKey, std::move(M)))
        return Err;

    } else if (auto Err = CODLayer.addExtraModule(*ModulesKey, std::move(M)))
      return Err;

    // Run the static constructors, and save the static destructor runner for
    // execution when the JIT is torn down.
    orc::CtorDtorRunner<CODLayerT> CtorRunner(std::move(CtorNames),
                                              *ModulesKey);
    if (auto Err = CtorRunner.runViaLayer(CODLayer))
      return Err;

    IRStaticDestructorRunners.emplace_back(std::move(DtorNames), *ModulesKey);

    return Error::success();
  }

  JITSymbol findSymbol(const std::string &Name) {
    return CODLayer.findSymbol(mangle(Name), true);
  }

  JITSymbol findSymbolIn(orc::VModuleKey K, const std::string &Name) {
    return CODLayer.findSymbolIn(K, mangle(Name), true);
  }

private:
  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      raw_string_ostream MangledNameStream(MangledName);
      Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  static std::set<Function*> extractSingleFunction(Function &F) {
    std::set<Function*> Partition;
    Partition.insert(&F);
    return Partition;
  }

  static TransformFtor createDebugDumper();

  orc::SymbolStringPool SSP;
  orc::ExecutionSession ES;

  std::map<orc::VModuleKey, std::shared_ptr<orc::SymbolResolver>> Resolvers;

  std::unique_ptr<TargetMachine> TM;
  DataLayout DL;
  SectionMemoryManager CCMgrMemMgr;

  std::unique_ptr<CompileCallbackMgr> CCMgr;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  IRDumpLayerT IRDumpLayer;
  CODLayerT CODLayer;

  orc::LocalCXXRuntimeOverrides CXXRuntimeOverrides;
  std::vector<orc::CtorDtorRunner<CODLayerT>> IRStaticDestructorRunners;
  llvm::Optional<orc::VModuleKey> ModulesKey;
};

int runOrcLazyJIT(std::vector<std::unique_ptr<Module>> Ms,
                  const std::vector<std::string> &Args);

} // end namespace llvm

#endif // LLVM_TOOLS_LLI_ORCLAZYJIT_H
