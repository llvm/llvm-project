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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <functional>

namespace llvm {
class GlobalValueSummary;
class LLVMContext;
class Module;
class OptimizationRemarkEmitter;

class MemProfContextDisambiguation
    : public OptionalPassInfoMixin<MemProfContextDisambiguation> {
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
    // The number of clones recorded for the call.
    unsigned NumClones;
    // The CallsiteInfo summary record for each profiled target in
    // CandidateProfileData (in the same order), or null if there is none, in
    // which case all clones of the call should call the original target.
    SmallVector<const CallsiteInfo *> Callsites;
  };

  // Record information needed for ICP of an indirect call, depending on its
  // profile information and the clone information recorded in the
  // CallsiteInfo records for its profiled targets. GetCallsite is invoked once
  // for each profiled target, in order, and returns the corresponding record
  // (or null if there is none). The number of clones recorded for this
  // indirect call is returned (0 if there are no records). The necessary
  // information is recorded in the ICallAnalysisInfo list for later ICP if any
  // clone of the call should call a cloned target.
  unsigned
  recordICPInfo(CallBase *CB,
                function_ref<const CallsiteInfo *(const InstrProfValueData &)>
                    GetCallsite,
                SmallVector<ICallAnalysisData> &ICallAnalysisInfo);

  // Actually performs any needed ICP in the function, using the information
  // recorded in the ICallAnalysisInfo list.
  void performICP(Module &M, ArrayRef<std::unique_ptr<ValueToValueMapTy>> VMaps,
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
  LLVM_ABI
  MemProfContextDisambiguation(const ModuleSummaryIndex *Summary = nullptr,
                               bool isSamplePGO = false);

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  LLVM_ABI void
  run(ModuleSummaryIndex &Index,
      function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
          isPrevailing,
      LLVMContext &Ctx,
      function_ref<void(StringRef, StringRef, const Twine &)> EmitRemark =
          nullptr);
};

/// Strips MemProf attributes and metadata. Can be invoked by the pass pipeline
/// when we don't have an index that has recorded that we are linking with
/// allocation libraries containing the necessary APIs for downstream
/// transformations.
class MemProfRemoveInfo : public OptionalPassInfoMixin<MemProfRemoveInfo> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_MEMPROF_CONTEXT_DISAMBIGUATION_H
