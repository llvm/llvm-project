//===- MemProfUse.cpp - memory allocation profile use pass --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MemProfUsePass which reads memory profiling data
// and uses it to add metadata to instructions to guide optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/MemProfUse.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/StaticDataProfileInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/MemProfCommon.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/HashBuilder.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/Utils/LongestCommonSequence.h"
#include <map>
#include <set>

using namespace llvm;
using namespace llvm::memprof;

#define DEBUG_TYPE "memprof"

namespace llvm {
extern cl::opt<bool> PGOWarnMissing;
extern cl::opt<bool> NoPGOWarnMismatch;
extern cl::opt<bool> NoPGOWarnMismatchComdatWeak;
} // namespace llvm

// By default disable matching of allocation profiles onto operator new that
// already explicitly pass a hot/cold hint, since we don't currently
// override these hints anyway.
static cl::opt<bool> ClMemProfMatchHotColdNew(
    "memprof-match-hot-cold-new",
    cl::desc(
        "Match allocation profiles onto existing hot/cold operator new calls"),
    cl::Hidden, cl::init(false));

static cl::opt<bool>
    ClPrintMemProfMatchInfo("memprof-print-match-info",
                            cl::desc("Print matching stats for each allocation "
                                     "context in this module's profiles"),
                            cl::Hidden, cl::init(false));

static cl::opt<bool> PrintMatchedAllocStack(
    "memprof-print-matched-alloc-stack",
    cl::desc("Print full stack context for matched "
             "allocations with -memprof-print-match-info."),
    cl::Hidden, cl::init(false));

static cl::opt<bool>
    PrintFunctionGuids("memprof-print-function-guids",
                       cl::desc("Print function GUIDs computed for matching"),
                       cl::Hidden, cl::init(false));

static cl::opt<bool>
    SalvageStaleProfile("memprof-salvage-stale-profile",
                        cl::desc("Salvage stale MemProf profile"),
                        cl::init(false), cl::Hidden);

static cl::opt<bool> ClMemProfAttachCalleeGuids(
    "memprof-attach-calleeguids",
    cl::desc(
        "Attach calleeguids as value profile metadata for indirect calls."),
    cl::init(true), cl::Hidden);

static cl::opt<unsigned> MinMatchedColdBytePercent(
    "memprof-matching-cold-threshold", cl::init(100), cl::Hidden,
    cl::desc("Min percent of cold bytes matched to hint allocation cold"));

static cl::opt<bool> AnnotateStaticDataSectionPrefix(
    "memprof-annotate-static-data-prefix", cl::init(false), cl::Hidden,
    cl::desc("If true, annotate the static data section prefix"));

// Matching statistics
STATISTIC(NumOfMemProfMissing, "Number of functions without memory profile.");
STATISTIC(NumOfMemProfMismatch,
          "Number of functions having mismatched memory profile hash.");
STATISTIC(NumOfMemProfFunc, "Number of functions having valid memory profile.");
STATISTIC(NumOfMemProfAllocContextProfiles,
          "Number of alloc contexts in memory profile.");
STATISTIC(NumOfMemProfCallSiteProfiles,
          "Number of callsites in memory profile.");
STATISTIC(NumOfMemProfMatchedAllocContexts,
          "Number of matched memory profile alloc contexts.");
STATISTIC(NumOfMemProfMatchedAllocs,
          "Number of matched memory profile allocs.");
STATISTIC(NumOfMemProfMatchedCallSites,
          "Number of matched memory profile callsites.");
STATISTIC(NumOfMemProfHotGlobalVars,
          "Number of global vars annotated with 'hot' section prefix.");
STATISTIC(NumOfMemProfColdGlobalVars,
          "Number of global vars annotated with 'unlikely' section prefix.");
STATISTIC(NumOfMemProfUnknownGlobalVars,
          "Number of global vars with unknown hotness (no section prefix).");
STATISTIC(NumOfMemProfExplicitSectionGlobalVars,
          "Number of global vars with user-specified section (not annotated).");

static void addCallsiteMetadata(Instruction &I,
                                ArrayRef<uint64_t> InlinedCallStack,
                                LLVMContext &Ctx) {
  I.setMetadata(LLVMContext::MD_callsite,
                buildCallstackMetadata(InlinedCallStack, Ctx));
}

static uint64_t computeStackId(GlobalValue::GUID Function, uint32_t LineOffset,
                               uint32_t Column) {
  llvm::HashBuilder<llvm::TruncatedBLAKE3<8>, llvm::endianness::little>
      HashBuilder;
  HashBuilder.add(Function, LineOffset, Column);
  llvm::BLAKE3Result<8> Hash = HashBuilder.final();
  uint64_t Id;
  std::memcpy(&Id, Hash.data(), sizeof(Hash));
  return Id;
}

static uint64_t computeStackId(const memprof::Frame &Frame) {
  return computeStackId(Frame.Function, Frame.LineOffset, Frame.Column);
}

static AllocationType getAllocType(const AllocationInfo *AllocInfo) {
  return getAllocType(AllocInfo->Info.getTotalLifetimeAccessDensity(),
                      AllocInfo->Info.getAllocCount(),
                      AllocInfo->Info.getTotalLifetime());
}

static AllocationType addCallStack(CallStackTrie &AllocTrie,
                                   const AllocationInfo *AllocInfo,
                                   uint64_t FullStackId) {
  SmallVector<uint64_t> StackIds;
  for (const auto &StackFrame : AllocInfo->CallStack)
    StackIds.push_back(computeStackId(StackFrame));
  auto AllocType = getAllocType(AllocInfo);
  std::vector<ContextTotalSize> ContextSizeInfo;
  if (recordContextSizeInfoForAnalysis()) {
    auto TotalSize = AllocInfo->Info.getTotalSize();
    assert(TotalSize);
    assert(FullStackId != 0);
    ContextSizeInfo.push_back({FullStackId, TotalSize});
  }
  AllocTrie.addCallStack(AllocType, StackIds, std::move(ContextSizeInfo));
  return AllocType;
}

// Return true if InlinedCallStack, computed from a call instruction's debug
// info, is a prefix of ProfileCallStack, a list of Frames from profile data
// (either the allocation data or a callsite).
static bool
stackFrameIncludesInlinedCallStack(ArrayRef<Frame> ProfileCallStack,
                                   ArrayRef<uint64_t> InlinedCallStack) {
  return ProfileCallStack.size() >= InlinedCallStack.size() &&
         llvm::equal(ProfileCallStack.take_front(InlinedCallStack.size()),
                     InlinedCallStack, [](const Frame &F, uint64_t StackId) {
                       return computeStackId(F) == StackId;
                     });
}

static bool isAllocationWithHotColdVariant(const Function *Callee,
                                           const TargetLibraryInfo &TLI) {
  if (!Callee)
    return false;
  LibFunc Func;
  if (!TLI.getLibFunc(*Callee, Func))
    return false;
  switch (Func) {
  case LibFunc_Znwm:
  case LibFunc_ZnwmRKSt9nothrow_t:
  case LibFunc_ZnwmSt11align_val_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t:
  case LibFunc_Znam:
  case LibFunc_ZnamRKSt9nothrow_t:
  case LibFunc_ZnamSt11align_val_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t:
  case LibFunc_size_returning_new:
  case LibFunc_size_returning_new_aligned:
    return true;
  case LibFunc_Znwm12__hot_cold_t:
  case LibFunc_ZnwmRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_ZnwmSt11align_val_t12__hot_cold_t:
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_Znam12__hot_cold_t:
  case LibFunc_ZnamRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_ZnamSt11align_val_t12__hot_cold_t:
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
  case LibFunc_size_returning_new_hot_cold:
  case LibFunc_size_returning_new_aligned_hot_cold:
    return ClMemProfMatchHotColdNew;
  default:
    return false;
  }
}

static void HandleUnsupportedAnnotationKinds(GlobalVariable &GVar,
                                             AnnotationKind Kind) {
  assert(Kind != llvm::memprof::AnnotationKind::AnnotationOK &&
         "Should not handle AnnotationOK here");
  SmallString<32> Reason;
  switch (Kind) {
  case llvm::memprof::AnnotationKind::ExplicitSection:
    ++NumOfMemProfExplicitSectionGlobalVars;
    Reason.append("explicit section name");
    break;
  case llvm::memprof::AnnotationKind::DeclForLinker:
    Reason.append("linker declaration");
    break;
  case llvm::memprof::AnnotationKind::ReservedName:
    Reason.append("name starts with `llvm.`");
    break;
  default:
    llvm_unreachable("Unexpected annotation kind");
  }
  LLVM_DEBUG(dbgs() << "Skip annotation for " << GVar.getName() << " due to "
                    << Reason << ".\n");
}

// Structure for tracking info about matched allocation contexts for use with
// -memprof-print-match-info and -memprof-print-matched-alloc-stack.
struct AllocMatchInfo {
  // Total size in bytes of matched context.
  uint64_t TotalSize = 0;
  // Matched allocation's type.
  AllocationType AllocType = AllocationType::None;
  // Number of frames matched to the allocation itself (values will be >1 in
  // cases where allocation was already inlined). Use a set because there can
  // be multiple inlined instances and each may have a different inline depth.
  // Use std::set to iterate in sorted order when printing.
  std::set<unsigned> MatchedFramesSet;
  // The full call stack of the allocation, for cases where requested via
  // -memprof-print-matched-alloc-stack.
  std::vector<Frame> CallStack;

  // Caller responsible for inserting the matched frames and the call stack when
  // appropriate.
  AllocMatchInfo(uint64_t TotalSize, AllocationType AllocType)
      : TotalSize(TotalSize), AllocType(AllocType) {}
};

DenseMap<uint64_t, SmallVector<CallEdgeTy, 0>>
memprof::extractCallsFromIR(Module &M, const TargetLibraryInfo &TLI,
                            function_ref<bool(uint64_t)> IsPresentInProfile) {
  DenseMap<uint64_t, SmallVector<CallEdgeTy, 0>> Calls;

  auto GetOffset = [](const DILocation *DIL) {
    return (DIL->getLine() - DIL->getScope()->getSubprogram()->getLine()) &
           0xffff;
  };

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    for (auto &BB : F) {
      for (auto &I : BB) {
        if (!isa<CallBase>(&I) || isa<IntrinsicInst>(&I))
          continue;

        auto *CB = dyn_cast<CallBase>(&I);
        auto *CalledFunction = CB->getCalledFunction();
        // Disregard indirect calls and intrinsics.
        if (!CalledFunction || CalledFunction->isIntrinsic())
          continue;

        StringRef CalleeName = CalledFunction->getName();
        // True if we are calling a heap allocation function that supports
        // hot/cold variants.
        bool IsAlloc = isAllocationWithHotColdVariant(CalledFunction, TLI);
        // True for the first iteration below, indicating that we are looking at
        // a leaf node.
        bool IsLeaf = true;
        for (const DILocation *DIL = I.getDebugLoc(); DIL;
             DIL = DIL->getInlinedAt()) {
          StringRef CallerName = DIL->getSubprogramLinkageName();
          assert(!CallerName.empty() &&
                 "Be sure to enable -fdebug-info-for-profiling");
          uint64_t CallerGUID = memprof::getGUID(CallerName);
          uint64_t CalleeGUID = memprof::getGUID(CalleeName);
          // Pretend that we are calling a function with GUID == 0 if we are
          // in the inline stack leading to a heap allocation function.
          if (IsAlloc) {
            if (IsLeaf) {
              // For leaf nodes, set CalleeGUID to 0 without consulting
              // IsPresentInProfile.
              CalleeGUID = 0;
            } else if (!IsPresentInProfile(CalleeGUID)) {
              // In addition to the leaf case above, continue to set CalleeGUID
              // to 0 as long as we don't see CalleeGUID in the profile.
              CalleeGUID = 0;
            } else {
              // Once we encounter a callee that exists in the profile, stop
              // setting CalleeGUID to 0.
              IsAlloc = false;
            }
          }

          LineLocation Loc = {GetOffset(DIL), DIL->getColumn()};
          Calls[CallerGUID].emplace_back(Loc, CalleeGUID);
          CalleeName = CallerName;
          IsLeaf = false;
        }
      }
    }
  }

  // Sort each call list by the source location.
  for (auto &[CallerGUID, CallList] : Calls) {
    llvm::sort(CallList);
    CallList.erase(llvm::unique(CallList), CallList.end());
  }

  return Calls;
}

DenseMap<uint64_t, LocToLocMap>
memprof::computeUndriftMap(Module &M, IndexedInstrProfReader *MemProfReader,
                           const TargetLibraryInfo &TLI) {
  DenseMap<uint64_t, LocToLocMap> UndriftMaps;

  DenseMap<uint64_t, SmallVector<memprof::CallEdgeTy, 0>> CallsFromProfile =
      MemProfReader->getMemProfCallerCalleePairs();
  DenseMap<uint64_t, SmallVector<memprof::CallEdgeTy, 0>> CallsFromIR =
      extractCallsFromIR(M, TLI, [&](uint64_t GUID) {
        return CallsFromProfile.contains(GUID);
      });

  // Compute an undrift map for each CallerGUID.
  for (const auto &[CallerGUID, IRAnchors] : CallsFromIR) {
    auto It = CallsFromProfile.find(CallerGUID);
    if (It == CallsFromProfile.end())
      continue;
    const auto &ProfileAnchors = It->second;

    LocToLocMap Matchings;
    longestCommonSequence<LineLocation, GlobalValue::GUID>(
        ProfileAnchors, IRAnchors, std::equal_to<GlobalValue::GUID>(),
        [&](LineLocation A, LineLocation B) { Matchings.try_emplace(A, B); });
    [[maybe_unused]] bool Inserted =
        UndriftMaps.try_emplace(CallerGUID, std::move(Matchings)).second;

    // The insertion must succeed because we visit each GUID exactly once.
    assert(Inserted);
  }

  return UndriftMaps;
}

// Given a MemProfRecord, undrift all the source locations present in the
// record in place.
static void
undriftMemProfRecord(const DenseMap<uint64_t, LocToLocMap> &UndriftMaps,
                     memprof::MemProfRecord &MemProfRec) {
  // Undrift a call stack in place.
  auto UndriftCallStack = [&](std::vector<Frame> &CallStack) {
    for (auto &F : CallStack) {
      auto I = UndriftMaps.find(F.Function);
      if (I == UndriftMaps.end())
        continue;
      auto J = I->second.find(LineLocation(F.LineOffset, F.Column));
      if (J == I->second.end())
        continue;
      auto &NewLoc = J->second;
      F.LineOffset = NewLoc.LineOffset;
      F.Column = NewLoc.Column;
    }
  };

  for (auto &AS : MemProfRec.AllocSites)
    UndriftCallStack(AS.CallStack);

  for (auto &CS : MemProfRec.CallSites)
    UndriftCallStack(CS.Frames);
}

// Helper function to process CalleeGuids and create value profile metadata
static void addVPMetadata(Module &M, Instruction &I,
                          ArrayRef<GlobalValue::GUID> CalleeGuids) {
  if (!ClMemProfAttachCalleeGuids || CalleeGuids.empty())
    return;

  // Prepare the vector of value data, initializing from any existing
  // value-profile metadata present on the instruction so that we merge the
  // new CalleeGuids into the existing entries.
  SmallVector<InstrProfValueData> VDs;
  uint64_t TotalCount = 0;

  if (I.getMetadata(LLVMContext::MD_prof)) {
    // Read all existing entries so we can merge them. Use a large
    // MaxNumValueData to retrieve all existing entries.
    VDs = getValueProfDataFromInst(I, IPVK_IndirectCallTarget,
                                   /*MaxNumValueData=*/UINT32_MAX, TotalCount);
  }

  // Save the original size for use later in detecting whether any were added.
  const size_t OriginalSize = VDs.size();

  // Initialize the set of existing guids with the original list.
  DenseSet<uint64_t> ExistingValues(
      llvm::from_range,
      llvm::map_range(
          VDs, [](const InstrProfValueData &Entry) { return Entry.Value; }));

  // Merge CalleeGuids into list of existing VDs, by appending any that are not
  // already included.
  VDs.reserve(OriginalSize + CalleeGuids.size());
  for (auto G : CalleeGuids) {
    if (!ExistingValues.insert(G).second)
      continue;
    InstrProfValueData NewEntry;
    NewEntry.Value = G;
    // For MemProf, we don't have actual call counts, so we assign
    // a weight of 1 to each potential target.
    // TODO: Consider making this weight configurable or increasing it to
    // improve effectiveness for ICP.
    NewEntry.Count = 1;
    TotalCount += NewEntry.Count;
    VDs.push_back(NewEntry);
  }

  // Update the VP metadata if we added any new callee GUIDs to the list.
  assert(VDs.size() >= OriginalSize);
  if (VDs.size() == OriginalSize)
    return;

  // First clear the existing !prof.
  I.setMetadata(LLVMContext::MD_prof, nullptr);

  // No need to sort the updated VDs as all appended entries have the same count
  // of 1, which is no larger than any existing entries. The incoming list of
  // CalleeGuids should already be deterministic for a given profile.
  annotateValueSite(M, I, VDs, TotalCount, IPVK_IndirectCallTarget, VDs.size());
}

static void handleAllocSite(
    Instruction &I, CallBase *CI, ArrayRef<uint64_t> InlinedCallStack,
    LLVMContext &Ctx, OptimizationRemarkEmitter &ORE, uint64_t MaxColdSize,
    const std::set<const AllocationInfo *> &AllocInfoSet,
    std::map<uint64_t, AllocMatchInfo> &FullStackIdToAllocMatchInfo) {
  // TODO: Remove this once the profile creation logic deduplicates contexts
  // that are the same other than the IsInlineFrame bool. Until then, keep the
  // largest.
  DenseMap<uint64_t, const AllocationInfo *> UniqueFullContextIdAllocInfo;
  for (auto *AllocInfo : AllocInfoSet) {
    auto FullStackId = computeFullStackId(AllocInfo->CallStack);
    auto [It, Inserted] =
        UniqueFullContextIdAllocInfo.insert({FullStackId, AllocInfo});
    // If inserted entry, done.
    if (Inserted)
      continue;
    // Keep the larger one, or the noncold one if they are the same size.
    auto CurSize = It->second->Info.getTotalSize();
    auto NewSize = AllocInfo->Info.getTotalSize();
    if ((CurSize > NewSize) ||
        (CurSize == NewSize &&
         getAllocType(AllocInfo) != AllocationType::NotCold))
      continue;
    It->second = AllocInfo;
  }
  // We may match this instruction's location list to multiple MIB
  // contexts. Add them to a Trie specialized for trimming the contexts to
  // the minimal needed to disambiguate contexts with unique behavior.
  CallStackTrie AllocTrie(&ORE, MaxColdSize);
  uint64_t TotalSize = 0;
  uint64_t TotalColdSize = 0;
  for (auto &[FullStackId, AllocInfo] : UniqueFullContextIdAllocInfo) {
    // Check the full inlined call stack against this one.
    // If we found and thus matched all frames on the call, include
    // this MIB.
    if (stackFrameIncludesInlinedCallStack(AllocInfo->CallStack,
                                           InlinedCallStack)) {
      NumOfMemProfMatchedAllocContexts++;
      auto AllocType = addCallStack(AllocTrie, AllocInfo, FullStackId);
      TotalSize += AllocInfo->Info.getTotalSize();
      if (AllocType == AllocationType::Cold)
        TotalColdSize += AllocInfo->Info.getTotalSize();
      // Record information about the allocation if match info printing
      // was requested.
      if (ClPrintMemProfMatchInfo) {
        assert(FullStackId != 0);
        auto [Iter, Inserted] = FullStackIdToAllocMatchInfo.try_emplace(
            FullStackId,
            AllocMatchInfo(AllocInfo->Info.getTotalSize(), AllocType));
        // Always insert the new matched frame count, since it may differ.
        Iter->second.MatchedFramesSet.insert(InlinedCallStack.size());
        if (Inserted && PrintMatchedAllocStack)
          Iter->second.CallStack.insert(Iter->second.CallStack.begin(),
                                        AllocInfo->CallStack.begin(),
                                        AllocInfo->CallStack.end());
      }
      ORE.emit(
          OptimizationRemark(DEBUG_TYPE, "MemProfUse", CI)
          << ore::NV("AllocationCall", CI) << " in function "
          << ore::NV("Caller", CI->getFunction())
          << " matched alloc context with alloc type "
          << ore::NV("Attribute", getAllocTypeAttributeString(AllocType))
          << " total size " << ore::NV("Size", AllocInfo->Info.getTotalSize())
          << " full context id " << ore::NV("Context", FullStackId)
          << " frame count " << ore::NV("Frames", InlinedCallStack.size()));
    }
  }
  // If the threshold for the percent of cold bytes is less than 100%,
  // and not all bytes are cold, see if we should still hint this
  // allocation as cold without context sensitivity.
  if (TotalColdSize < TotalSize && MinMatchedColdBytePercent < 100 &&
      TotalColdSize * 100 >= MinMatchedColdBytePercent * TotalSize) {
    AllocTrie.addSingleAllocTypeAttribute(CI, AllocationType::Cold, "dominant");
    return;
  }

  // We might not have matched any to the full inlined call stack.
  // But if we did, create and attach metadata, or a function attribute if
  // all contexts have identical profiled behavior.
  if (!AllocTrie.empty()) {
    NumOfMemProfMatchedAllocs++;
    // MemprofMDAttached will be false if a function attribute was
    // attached.
    bool MemprofMDAttached = AllocTrie.buildAndAttachMIBMetadata(CI);
    assert(MemprofMDAttached == I.hasMetadata(LLVMContext::MD_memprof));
    if (MemprofMDAttached) {
      // Add callsite metadata for the instruction's location list so that
      // it simpler later on to identify which part of the MIB contexts
      // are from this particular instruction (including during inlining,
      // when the callsite metadata will be updated appropriately).
      // FIXME: can this be changed to strip out the matching stack
      // context ids from the MIB contexts and not add any callsite
      // metadata here to save space?
      addCallsiteMetadata(I, InlinedCallStack, Ctx);
    }
  }
}

// Helper struct for maintaining refs to callsite data. As an alternative we
// could store a pointer to the CallSiteInfo struct but we also need the frame
// index. Using ArrayRefs instead makes it a little easier to read.
struct CallSiteEntry {
  // Subset of frames for the corresponding CallSiteInfo.
  ArrayRef<Frame> Frames;
  // Potential targets for indirect calls.
  ArrayRef<GlobalValue::GUID> CalleeGuids;
};

static void handleCallSite(Instruction &I, const Function *CalledFunction,
                           ArrayRef<uint64_t> InlinedCallStack,
                           const std::vector<CallSiteEntry> &CallSiteEntries,
                           Module &M,
                           std::set<std::vector<uint64_t>> &MatchedCallSites,
                           OptimizationRemarkEmitter &ORE) {
  auto &Ctx = M.getContext();
  // Set of Callee GUIDs to attach to indirect calls. We accumulate all of them
  // to support cases where the instuction's inlined frames match multiple call
  // site entries, which can happen if the profile was collected from a binary
  // where this instruction was eventually inlined into multiple callers.
  SetVector<GlobalValue::GUID> CalleeGuids;
  bool CallsiteMDAdded = false;
  for (const auto &CallSiteEntry : CallSiteEntries) {
    // If we found and thus matched all frames on the call, create and
    // attach call stack metadata.
    if (stackFrameIncludesInlinedCallStack(CallSiteEntry.Frames,
                                           InlinedCallStack)) {
      NumOfMemProfMatchedCallSites++;
      // Only need to find one with a matching call stack and add a single
      // callsite metadata.
      if (!CallsiteMDAdded) {
        addCallsiteMetadata(I, InlinedCallStack, Ctx);

        // Accumulate call site matching information upon request.
        if (ClPrintMemProfMatchInfo) {
          std::vector<uint64_t> CallStack;
          append_range(CallStack, InlinedCallStack);
          MatchedCallSites.insert(std::move(CallStack));
        }
        ORE.emit(OptimizationRemark(DEBUG_TYPE, "MemProfUse", &I)
                 << ore::NV("CallSite", &I) << " in function "
                 << ore::NV("Caller", I.getFunction())
                 << " matched callsite with frame count "
                 << ore::NV("Frames", InlinedCallStack.size()));

        // If this is a direct call, we're done.
        if (CalledFunction)
          break;
        CallsiteMDAdded = true;
      }

      assert(!CalledFunction && "Didn't expect direct call");

      // Collect Callee GUIDs from all matching CallSiteEntries.
      CalleeGuids.insert(CallSiteEntry.CalleeGuids.begin(),
                         CallSiteEntry.CalleeGuids.end());
    }
  }
  // Try to attach indirect call metadata if possible.
  addVPMetadata(M, I, CalleeGuids.getArrayRef());
}

static void
readMemprof(Module &M, Function &F, IndexedInstrProfReader *MemProfReader,
            const TargetLibraryInfo &TLI,
            std::map<uint64_t, AllocMatchInfo> &FullStackIdToAllocMatchInfo,
            std::set<std::vector<uint64_t>> &MatchedCallSites,
            DenseMap<uint64_t, LocToLocMap> &UndriftMaps,
            OptimizationRemarkEmitter &ORE, uint64_t MaxColdSize) {
  auto &Ctx = M.getContext();
  // Previously we used getIRPGOFuncName() here. If F is local linkage,
  // getIRPGOFuncName() returns FuncName with prefix 'FileName;'. But
  // llvm-profdata uses FuncName in dwarf to create GUID which doesn't
  // contain FileName's prefix. It caused local linkage function can't
  // find MemProfRecord. So we use getName() now.
  // 'unique-internal-linkage-names' can make MemProf work better for local
  // linkage function.
  auto FuncName = F.getName();
  auto FuncGUID = Function::getGUIDAssumingExternalLinkage(FuncName);
  if (PrintFunctionGuids)
    errs() << "MemProf: Function GUID " << FuncGUID << " is " << FuncName
           << "\n";
  std::optional<memprof::MemProfRecord> MemProfRec;
  auto Err = MemProfReader->getMemProfRecord(FuncGUID).moveInto(MemProfRec);
  if (Err) {
    handleAllErrors(std::move(Err), [&](const InstrProfError &IPE) {
      auto Err = IPE.get();
      bool SkipWarning = false;
      LLVM_DEBUG(dbgs() << "Error in reading profile for Func " << FuncName
                        << ": ");
      if (Err == instrprof_error::unknown_function) {
        NumOfMemProfMissing++;
        SkipWarning = !PGOWarnMissing;
        LLVM_DEBUG(dbgs() << "unknown function");
      } else if (Err == instrprof_error::hash_mismatch) {
        NumOfMemProfMismatch++;
        SkipWarning =
            NoPGOWarnMismatch ||
            (NoPGOWarnMismatchComdatWeak &&
             (F.hasComdat() ||
              F.getLinkage() == GlobalValue::AvailableExternallyLinkage));
        LLVM_DEBUG(dbgs() << "hash mismatch (skip=" << SkipWarning << ")");
      }

      if (SkipWarning)
        return;

      std::string Msg = (IPE.message() + Twine(" ") + F.getName().str() +
                         Twine(" Hash = ") + std::to_string(FuncGUID))
                            .str();

      Ctx.diagnose(
          DiagnosticInfoPGOProfile(M.getName().data(), Msg, DS_Warning));
    });
    return;
  }

  NumOfMemProfFunc++;

  // If requested, undrfit MemProfRecord so that the source locations in it
  // match those in the IR.
  if (SalvageStaleProfile)
    undriftMemProfRecord(UndriftMaps, *MemProfRec);

  // Detect if there are non-zero column numbers in the profile. If not,
  // treat all column numbers as 0 when matching (i.e. ignore any non-zero
  // columns in the IR). The profiled binary might have been built with
  // column numbers disabled, for example.
  bool ProfileHasColumns = false;

  // Build maps of the location hash to all profile data with that leaf location
  // (allocation info and the callsites).
  std::map<uint64_t, std::set<const AllocationInfo *>> LocHashToAllocInfo;

  // For the callsites we need to record slices of the frame array (see comments
  // below where the map entries are added) along with their CalleeGuids.
  std::map<uint64_t, std::vector<CallSiteEntry>> LocHashToCallSites;
  for (auto &AI : MemProfRec->AllocSites) {
    NumOfMemProfAllocContextProfiles++;
    // Associate the allocation info with the leaf frame. The later matching
    // code will match any inlined call sequences in the IR with a longer prefix
    // of call stack frames.
    uint64_t StackId = computeStackId(AI.CallStack[0]);
    LocHashToAllocInfo[StackId].insert(&AI);
    ProfileHasColumns |= AI.CallStack[0].Column;
  }
  for (auto &CS : MemProfRec->CallSites) {
    NumOfMemProfCallSiteProfiles++;
    // Need to record all frames from leaf up to and including this function,
    // as any of these may or may not have been inlined at this point.
    unsigned Idx = 0;
    for (auto &StackFrame : CS.Frames) {
      uint64_t StackId = computeStackId(StackFrame);
      ArrayRef<Frame> FrameSlice = ArrayRef<Frame>(CS.Frames).drop_front(Idx++);
      // The callee guids for the slice containing all frames (due to the
      // increment above Idx is now 1) comes from the CalleeGuids recorded in
      // the CallSite. For the slices not containing the leaf-most frame, the
      // callee guid is simply the function GUID of the prior frame.
      LocHashToCallSites[StackId].push_back(
          {FrameSlice, (Idx == 1 ? CS.CalleeGuids
                                 : ArrayRef<GlobalValue::GUID>(
                                       CS.Frames[Idx - 2].Function))});

      ProfileHasColumns |= StackFrame.Column;
      // Once we find this function, we can stop recording.
      if (StackFrame.Function == FuncGUID)
        break;
    }
    assert(Idx <= CS.Frames.size() && CS.Frames[Idx - 1].Function == FuncGUID);
  }

  auto GetOffset = [](const DILocation *DIL) {
    return (DIL->getLine() - DIL->getScope()->getSubprogram()->getLine()) &
           0xffff;
  };

  // Now walk the instructions, looking up the associated profile data using
  // debug locations.
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (I.isDebugOrPseudoInst())
        continue;
      // We are only interested in calls (allocation or interior call stack
      // context calls).
      auto *CI = dyn_cast<CallBase>(&I);
      if (!CI)
        continue;
      auto *CalledFunction = CI->getCalledFunction();
      if (CalledFunction && CalledFunction->isIntrinsic())
        continue;
      // List of call stack ids computed from the location hashes on debug
      // locations (leaf to inlined at root).
      SmallVector<uint64_t, 8> InlinedCallStack;
      // Was the leaf location found in one of the profile maps?
      bool LeafFound = false;
      // If leaf was found in a map, iterators pointing to its location in both
      // of the maps. It might exist in neither, one, or both (the latter case
      // can happen because we don't currently have discriminators to
      // distinguish the case when a single line/col maps to both an allocation
      // and another callsite).
      auto AllocInfoIter = LocHashToAllocInfo.end();
      auto CallSitesIter = LocHashToCallSites.end();
      for (const DILocation *DIL = I.getDebugLoc(); DIL != nullptr;
           DIL = DIL->getInlinedAt()) {
        // Use C++ linkage name if possible. Need to compile with
        // -fdebug-info-for-profiling to get linkage name.
        StringRef Name = DIL->getScope()->getSubprogram()->getLinkageName();
        if (Name.empty())
          Name = DIL->getScope()->getSubprogram()->getName();
        auto CalleeGUID = Function::getGUIDAssumingExternalLinkage(Name);
        auto StackId = computeStackId(CalleeGUID, GetOffset(DIL),
                                      ProfileHasColumns ? DIL->getColumn() : 0);
        // Check if we have found the profile's leaf frame. If yes, collect
        // the rest of the call's inlined context starting here. If not, see if
        // we find a match further up the inlined context (in case the profile
        // was missing debug frames at the leaf).
        if (!LeafFound) {
          AllocInfoIter = LocHashToAllocInfo.find(StackId);
          CallSitesIter = LocHashToCallSites.find(StackId);
          if (AllocInfoIter != LocHashToAllocInfo.end() ||
              CallSitesIter != LocHashToCallSites.end())
            LeafFound = true;
        }
        if (LeafFound)
          InlinedCallStack.push_back(StackId);
      }
      // If leaf not in either of the maps, skip inst.
      if (!LeafFound)
        continue;

      // First add !memprof metadata from allocation info, if we found the
      // instruction's leaf location in that map, and if the rest of the
      // instruction's locations match the prefix Frame locations on an
      // allocation context with the same leaf.
      if (AllocInfoIter != LocHashToAllocInfo.end() &&
          // Only consider allocations which support hinting.
          isAllocationWithHotColdVariant(CI->getCalledFunction(), TLI))
        handleAllocSite(I, CI, InlinedCallStack, Ctx, ORE, MaxColdSize,
                        AllocInfoIter->second, FullStackIdToAllocMatchInfo);
      else if (CallSitesIter != LocHashToCallSites.end())
        // Otherwise, add callsite metadata. If we reach here then we found the
        // instruction's leaf location in the callsites map and not the
        // allocation map.
        handleCallSite(I, CalledFunction, InlinedCallStack,
                       CallSitesIter->second, M, MatchedCallSites, ORE);
    }
  }
}

MemProfUsePass::MemProfUsePass(std::string MemoryProfileFile,
                               IntrusiveRefCntPtr<vfs::FileSystem> FS)
    : MemoryProfileFileName(MemoryProfileFile), FS(FS) {
  if (!FS)
    this->FS = vfs::getRealFileSystem();
}

PreservedAnalyses MemProfUsePass::run(Module &M, ModuleAnalysisManager &AM) {
  // Return immediately if the module doesn't contain any function or global
  // variables.
  if (M.empty() && M.globals().empty())
    return PreservedAnalyses::all();

  LLVM_DEBUG(dbgs() << "Read in memory profile:\n");
  auto &Ctx = M.getContext();
  auto ReaderOrErr = IndexedInstrProfReader::create(MemoryProfileFileName, *FS);
  if (Error E = ReaderOrErr.takeError()) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
      Ctx.diagnose(
          DiagnosticInfoPGOProfile(MemoryProfileFileName.data(), EI.message()));
    });
    return PreservedAnalyses::all();
  }

  std::unique_ptr<IndexedInstrProfReader> MemProfReader =
      std::move(ReaderOrErr.get());
  if (!MemProfReader) {
    Ctx.diagnose(DiagnosticInfoPGOProfile(
        MemoryProfileFileName.data(), StringRef("Cannot get MemProfReader")));
    return PreservedAnalyses::all();
  }

  if (!MemProfReader->hasMemoryProfile()) {
    Ctx.diagnose(DiagnosticInfoPGOProfile(MemoryProfileFileName.data(),
                                          "Not a memory profile"));
    return PreservedAnalyses::all();
  }

  const bool Changed =
      annotateGlobalVariables(M, MemProfReader->getDataAccessProfileData());

  // If the module doesn't contain any function, return after we process all
  // global variables.
  if (M.empty())
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(*M.begin());
  DenseMap<uint64_t, LocToLocMap> UndriftMaps;
  if (SalvageStaleProfile)
    UndriftMaps = computeUndriftMap(M, MemProfReader.get(), TLI);

  // Map from the stack hash of each matched allocation context in the function
  // profiles to match info such as the total profiled size (bytes), allocation
  // type, number of frames matched to the allocation itself, and the full array
  // of call stack ids.
  std::map<uint64_t, AllocMatchInfo> FullStackIdToAllocMatchInfo;

  // Set of the matched call sites, each expressed as a sequence of an inline
  // call stack.
  std::set<std::vector<uint64_t>> MatchedCallSites;

  uint64_t MaxColdSize = 0;
  if (auto *MemProfSum = MemProfReader->getMemProfSummary())
    MaxColdSize = MemProfSum->getMaxColdTotalSize();

  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    const TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
    readMemprof(M, F, MemProfReader.get(), TLI, FullStackIdToAllocMatchInfo,
                MatchedCallSites, UndriftMaps, ORE, MaxColdSize);
  }

  if (ClPrintMemProfMatchInfo) {
    for (const auto &[Id, Info] : FullStackIdToAllocMatchInfo) {
      for (auto Frames : Info.MatchedFramesSet) {
        // TODO: To reduce verbosity, should we change the existing message
        // so that we emit a list of matched frame counts in a single message
        // about the context (instead of one message per frame count?
        errs() << "MemProf " << getAllocTypeAttributeString(Info.AllocType)
               << " context with id " << Id << " has total profiled size "
               << Info.TotalSize << " is matched with " << Frames << " frames";
        if (PrintMatchedAllocStack) {
          errs() << " and call stack";
          for (auto &F : Info.CallStack)
            errs() << " " << computeStackId(F);
        }
        errs() << "\n";
      }
    }

    for (const auto &CallStack : MatchedCallSites) {
      errs() << "MemProf callsite match for inline call stack";
      for (uint64_t StackId : CallStack)
        errs() << " " << StackId;
      errs() << "\n";
    }
  }

  return PreservedAnalyses::none();
}

bool MemProfUsePass::annotateGlobalVariables(
    Module &M, const memprof::DataAccessProfData *DataAccessProf) {
  if (!AnnotateStaticDataSectionPrefix || M.globals().empty())
    return false;

  if (!DataAccessProf) {
    M.addModuleFlag(Module::Warning, "EnableDataAccessProf", 0U);
    // FIXME: Add a diagnostic message without failing the compilation when
    // data access profile payload is not available.
    return false;
  }
  M.addModuleFlag(Module::Warning, "EnableDataAccessProf", 1U);

  bool Changed = false;
  // Iterate all global variables in the module and annotate them based on
  // data access profiles. Note it's up to the linker to decide how to map input
  // sections to output sections, and one conservative practice is to map
  // unlikely-prefixed ones to unlikely output section, and map the rest
  // (hot-prefixed or prefix-less) to the canonical output section.
  for (GlobalVariable &GVar : M.globals()) {
    assert(!GVar.getSectionPrefix().has_value() &&
           "GVar shouldn't have section prefix yet");
    auto Kind = llvm::memprof::getAnnotationKind(GVar);
    if (Kind != llvm::memprof::AnnotationKind::AnnotationOK) {
      HandleUnsupportedAnnotationKinds(GVar, Kind);
      continue;
    }

    StringRef Name = GVar.getName();
    // Skip string literals as their mangled names don't stay stable across
    // binary releases.
    // TODO: Track string content hash in the profiles and compute it inside the
    // compiler to categeorize the hotness string literals.
    if (Name.starts_with(".str")) {
      LLVM_DEBUG(dbgs() << "Skip annotating string literal " << Name << "\n");
      continue;
    }

    // DataAccessProfRecord's get* methods will canonicalize the name under the
    // hood before looking it up, so optimizer doesn't need to do it.
    std::optional<DataAccessProfRecord> Record =
        DataAccessProf->getProfileRecord(Name);
    // Annotate a global variable as hot if it has non-zero sampled count, and
    // annotate it as cold if it's seen in the profiled binary
    // file but doesn't have any access sample.
    // For logging, optimization remark emitter requires a llvm::Function, but
    // it's not well defined how to associate a global variable with a function.
    // So we just print out the static data section prefix in LLVM_DEBUG.
    if (Record && Record->AccessCount > 0) {
      ++NumOfMemProfHotGlobalVars;
      Changed |= GVar.setSectionPrefix("hot");
      LLVM_DEBUG(dbgs() << "Global variable " << Name
                        << " is annotated as hot\n");
    } else if (DataAccessProf->isKnownColdSymbol(Name)) {
      ++NumOfMemProfColdGlobalVars;
      Changed |= GVar.setSectionPrefix("unlikely");
      Changed = true;
      LLVM_DEBUG(dbgs() << "Global variable " << Name
                        << " is annotated as unlikely\n");
    } else {
      ++NumOfMemProfUnknownGlobalVars;
      LLVM_DEBUG(dbgs() << "Global variable " << Name << " is not annotated\n");
    }
  }

  return Changed;
}
