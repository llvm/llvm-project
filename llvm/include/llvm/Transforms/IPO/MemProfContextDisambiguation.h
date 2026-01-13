//==- MemProfContextDisambiguation.h - Context Disambiguation ----*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements support for context disambiguation of allocation calls for profile
// guided heap optimization using memprof metadata. See implementation file for
// details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H
#define LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H

#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>

namespace llvm {
class GlobalValueSummary;
class Module;
class OptimizationRemarkEmitter;

class MemProfContextDisambiguation
    : public PassInfoMixin<MemProfContextDisambiguation> {
  /// Run the context disambiguator on \p M, returns true if any changes made.
  bool processModule(
      Module &M,
      function_ref<OptimizationRemarkEmitter &(Function *)> OREGetter);

  /// In the ThinLTO backend, apply the cloning decisions in ImportSummary to
  /// the IR.
  bool applyImport(Module &M);

  // Builds the symtab and analysis used for ICP during ThinLTO backends.
  bool initializeIndirectCallPromotionInfo(Module &M);

  // Data structure for saving indirect call profile info for use in ICP with
  // cloning.
  struct ICallAnalysisData {
    CallBase *CB;
    std::vector<InstrProfValueData> CandidateProfileData;
    uint32_t NumCandidates;
    uint64_t TotalCount;
    size_t CallsiteInfoStartIndex;
  };

  // Record information needed for ICP of an indirect call, depending on its
  // profile information and the clone information recorded in the corresponding
  // CallsiteInfo records. The SI iterator point to the current iteration point
  // through AllCallsites in this function, and will be updated in this method
  // as we iterate through profiled targets. The number of clones recorded for
  // this indirect call is returned. The necessary information is recorded in
  // the ICallAnalysisInfo list for later ICP.
  unsigned recordICPInfo(CallBase *CB, ArrayRef<CallsiteInfo> AllCallsites,
                         ArrayRef<CallsiteInfo>::iterator &SI,
                         SmallVector<ICallAnalysisData> &ICallAnalysisInfo);

  // Actually performs any needed ICP in the function, using the information
  // recorded in the ICallAnalysisInfo list.
  void performICP(Module &M, ArrayRef<CallsiteInfo> AllCallsites,
                  ArrayRef<std::unique_ptr<ValueToValueMapTy>> VMaps,
                  ArrayRef<ICallAnalysisData> ICallAnalysisInfo,
                  OptimizationRemarkEmitter &ORE);

  /// Import summary containing cloning decisions for the ThinLTO backend.
  const ModuleSummaryIndex *ImportSummary;

  // Owns the import summary specified by internal options for testing the
  // ThinLTO backend via opt (to simulate distributed ThinLTO).
  std::unique_ptr<ModuleSummaryIndex> ImportSummaryForTesting;

  // Whether we are building with SamplePGO. This is needed for correctly
  // updating profile metadata on speculatively promoted calls.
  bool isSamplePGO;

  // Used when performing indirect call analysis and promotion when cloning in
  // the ThinLTO backend during applyImport.
  std::unique_ptr<InstrProfSymtab> Symtab;
  std::unique_ptr<ICallPromotionAnalysis> ICallAnalysis;

public:
  MemProfContextDisambiguation(const ModuleSummaryIndex *Summary = nullptr,
                               bool isSamplePGO = false);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  void run(ModuleSummaryIndex &Index,
           function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
               isPrevailing);
};

/// Strips MemProf attributes and metadata. Can be invoked by the pass pipeline
/// when we don't have an index that has recorded that we are linking with
/// allocation libraries containing the necessary APIs for downstream
/// transformations.
class MemProfRemoveInfo : public PassInfoMixin<MemProfRemoveInfo> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H
