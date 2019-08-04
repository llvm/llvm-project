//===-- Speculation.h - Speculative Compilation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition to support speculative compilation when laziness is
// enabled.
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SPECULATION_H
#define LLVM_EXECUTIONENGINE_ORC_SPECULATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace llvm {
namespace orc {

class Speculator;

// Track the Impls (JITDylib,Symbols) of Symbols while lazy call through
// trampolines are created. Operations are guarded by locks tp ensure that Imap
// stays in consistent state after read/write

class ImplSymbolMap {
  friend class Speculator;

public:
  using AliaseeDetails = std::pair<SymbolStringPtr, JITDylib *>;
  using Alias = SymbolStringPtr;
  using ImapTy = DenseMap<Alias, AliaseeDetails>;
  void trackImpls(SymbolAliasMap ImplMaps, JITDylib *SrcJD);

private:
  // FIX ME: find a right way to distinguish the pre-compile Symbols, and update
  // the callsite
  Optional<AliaseeDetails> getImplFor(const SymbolStringPtr &StubSymbol) {
    std::lock_guard<std::mutex> Lockit(ConcurrentAccess);
    auto Position = Maps.find(StubSymbol);
    if (Position != Maps.end())
      return Position->getSecond();
    else
      return None;
  }

  std::mutex ConcurrentAccess;
  ImapTy Maps;
};

// Defines Speculator Concept,
class Speculator {
public:
  using TargetFAddr = JITTargetAddress;
  using FunctionCandidatesMap = DenseMap<SymbolStringPtr, SymbolNameSet>;
  using StubAddrLikelies = DenseMap<TargetFAddr, SymbolNameSet>;

private:
  void registerSymbolsWithAddr(TargetFAddr ImplAddr,
                               SymbolNameSet likelySymbols) {
    std::lock_guard<std::mutex> Lockit(ConcurrentAccess);
    GlobalSpecMap.insert({ImplAddr, std::move(likelySymbols)});
  }

  void launchCompile(JITTargetAddress FAddr) {
    SymbolNameSet CandidateSet;
    // Copy CandidateSet is necessary, to avoid unsynchronized access to
    // the datastructure.
    {
      std::lock_guard<std::mutex> Lockit(ConcurrentAccess);
      auto It = GlobalSpecMap.find(FAddr);
      // Kill this when jump on first call instrumentation is in place;
      auto Iv = AlreadyExecuted.insert(FAddr);
      if (It == GlobalSpecMap.end() || Iv.second == false)
        return;
      else
        CandidateSet = It->getSecond();
    }

    // Try to distinguish pre-compiled symbols!
    for (auto &Callee : CandidateSet) {
      auto ImplSymbol = AliaseeImplTable.getImplFor(Callee);
      if (!ImplSymbol.hasValue())
        continue;
      const auto &ImplSymbolName = ImplSymbol.getPointer()->first;
      auto *ImplJD = ImplSymbol.getPointer()->second;
      ES.lookup(JITDylibSearchList({{ImplJD, true}}),
                SymbolNameSet({ImplSymbolName}), SymbolState::Ready,
                [this](Expected<SymbolMap> Result) {
                  if (auto Err = Result.takeError())
                    ES.reportError(std::move(Err));
                },
                NoDependenciesToRegister);
    }
  }

public:
  Speculator(ImplSymbolMap &Impl, ExecutionSession &ref)
      : AliaseeImplTable(Impl), ES(ref), GlobalSpecMap(0) {}
  Speculator(const Speculator &) = delete;
  Speculator(Speculator &&) = delete;
  Speculator &operator=(const Speculator &) = delete;
  Speculator &operator=(Speculator &&) = delete;
  ~Speculator() {}

  // Speculatively compile likely functions for the given Stub Address.
  // destination of __orc_speculate_for jump
  void speculateFor(TargetFAddr StubAddr) { launchCompile(StubAddr); }

  // FIXME : Register with Stub Address, after JITLink Fix.
  void registerSymbols(FunctionCandidatesMap Candidates, JITDylib *JD) {
    for (auto &SymPair : Candidates) {
      auto Target = SymPair.first;
      auto Likely = SymPair.second;

      auto OnReadyFixUp = [Likely, Target,
                           this](Expected<SymbolMap> ReadySymbol) {
        if (ReadySymbol) {
          auto RAddr = (*ReadySymbol)[Target].getAddress();
          registerSymbolsWithAddr(RAddr, std::move(Likely));
        } else
          this->getES().reportError(ReadySymbol.takeError());
      };
      // Include non-exported symbols also.
      ES.lookup(JITDylibSearchList({{JD, true}}), SymbolNameSet({Target}),
                SymbolState::Ready, OnReadyFixUp, NoDependenciesToRegister);
    }
  }

  ExecutionSession &getES() { return ES; }

private:
  std::mutex ConcurrentAccess;
  ImplSymbolMap &AliaseeImplTable;
  ExecutionSession &ES;
  DenseSet<TargetFAddr> AlreadyExecuted;
  StubAddrLikelies GlobalSpecMap;
};
// replace DenseMap with Pair
class IRSpeculationLayer : public IRLayer {
public:
  using IRlikiesStrRef = Optional<DenseMap<StringRef, DenseSet<StringRef>>>;
  using ResultEval =
      std::function<IRlikiesStrRef(Function &, FunctionAnalysisManager &)>;
  using TargetAndLikelies = DenseMap<SymbolStringPtr, SymbolNameSet>;

  IRSpeculationLayer(ExecutionSession &ES, IRCompileLayer &BaseLayer,
                     Speculator &Spec, ResultEval Interpreter)
      : IRLayer(ES), NextLayer(BaseLayer), S(Spec), QueryAnalysis(Interpreter) {
    PB.registerFunctionAnalyses(FAM);
  }

  template <
      typename AnalysisTy,
      typename std::enable_if<
          std::is_base_of<AnalysisInfoMixin<AnalysisTy>, AnalysisTy>::value,
          bool>::type = true>
  void registerAnalysis() {
    FAM.registerPass([]() { return AnalysisTy(); });
  }

  void emit(MaterializationResponsibility R, ThreadSafeModule TSM);

private:
  TargetAndLikelies
  internToJITSymbols(DenseMap<StringRef, DenseSet<StringRef>> IRNames) {
    assert(!IRNames.empty() && "No IRNames received to Intern?");
    TargetAndLikelies InternedNames;
    DenseSet<SymbolStringPtr> TargetJITNames;
    ExecutionSession &Es = getExecutionSession();
    for (auto &NamePair : IRNames) {
      for (auto &TargetNames : NamePair.second)
        TargetJITNames.insert(Es.intern(TargetNames));

      InternedNames.insert(
          {Es.intern(NamePair.first), std::move(TargetJITNames)});
    }
    return InternedNames;
  }

  IRCompileLayer &NextLayer;
  Speculator &S;
  PassBuilder PB;
  FunctionAnalysisManager FAM;
  ResultEval QueryAnalysis;
};

// Runtime Function Interface
extern "C" {
void __orc_speculate_for(Speculator *, uint64_t stub_id);
}

} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SPECULATION_H
