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

static AllocationType addCallStack(CallStackTrie &AllocTrie,
                                   const AllocationInfo *AllocInfo,
                                   uint64_t FullStackId) {
  SmallVector<uint64_t> StackIds;
  for (const auto &StackFrame : AllocInfo->CallStack)
    StackIds.push_back(computeStackId(StackFrame));
  auto AllocType = getAllocType(AllocInfo->Info.getTotalLifetimeAccessDensity(),
                                AllocInfo->Info.getAllocCount(),
                                AllocInfo->Info.getTotalLifetime());
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

struct AllocMatchInfo {
  uint64_t TotalSize = 0;
  AllocationType AllocType = AllocationType::None;
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

  if (I.getMetadata(LLVMContext::MD_prof)) {
    uint64_t Unused;
    // TODO: When merging is implemented, increase this to a typical ICP value
    // (e.g., 3-6) For now, we only need to check if existing data exists, so 1
    // is sufficient
    auto ExistingVD = getValueProfDataFromInst(I, IPVK_IndirectCallTarget,
                                               /*MaxNumValueData=*/1, Unused);
    // We don't know how to merge value profile data yet.
    if (!ExistingVD.empty()) {
      return;
    }
  }

  SmallVector<InstrProfValueData, 4> VDs;
  uint64_t TotalCount = 0;

  for (const GlobalValue::GUID CalleeGUID : CalleeGuids) {
    InstrProfValueData VD;
    VD.Value = CalleeGUID;
    // For MemProf, we don't have actual call counts, so we assign
    // a weight of 1 to each potential target.
    // TODO: Consider making this weight configurable or increasing it to
    // improve effectiveness for ICP.
    VD.Count = 1;
    VDs.push_back(VD);
    TotalCount += VD.Count;
  }

  if (!VDs.empty()) {
    annotateValueSite(M, I, VDs, TotalCount, IPVK_IndirectCallTarget,
                      VDs.size());
  }
}

static void
handleAllocSite(Instruction &I, CallBase *CI,
                ArrayRef<uint64_t> InlinedCallStack, LLVMContext &Ctx,
                OptimizationRemarkEmitter &ORE, uint64_t MaxColdSize,
                const std::set<const AllocationInfo *> &AllocInfoSet,
                std::map<std::pair<uint64_t, unsigned>, AllocMatchInfo>
                    &FullStackIdToAllocMatchInfo) {
  // We may match this instruction's location list to multiple MIB
  // contexts. Add them to a Trie specialized for trimming the contexts to
  // the minimal needed to disambiguate contexts with unique behavior.
  CallStackTrie AllocTrie(&ORE, MaxColdSize);
  uint64_t TotalSize = 0;
  uint64_t TotalColdSize = 0;
  for (auto *AllocInfo : AllocInfoSet) {
    // Check the full inlined call stack against this one.
    // If we found and thus matched all frames on the call, include
    // this MIB.
    if (stackFrameIncludesInlinedCallStack(AllocInfo->CallStack,
                                           InlinedCallStack)) {
      NumOfMemProfMatchedAllocContexts++;
      uint64_t FullStackId = 0;
      if (ClPrintMemProfMatchInfo || recordContextSizeInfoForAnalysis())
        FullStackId = computeFullStackId(AllocInfo->CallStack);
      auto AllocType = addCallStack(AllocTrie, AllocInfo, FullStackId);
      TotalSize += AllocInfo->Info.getTotalSize();
      if (AllocType == AllocationType::Cold)
        TotalColdSize += AllocInfo->Info.getTotalSize();
      // Record information about the allocation if match info printing
      // was requested.
      if (ClPrintMemProfMatchInfo) {
        assert(FullStackId != 0);
        FullStackIdToAllocMatchInfo[std::make_pair(FullStackId,
                                                   InlinedCallStack.size())] = {
            AllocInfo->Info.getTotalSize(), AllocType};
      }
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

  // Only compare Frame contents.
  // Use pointer-based equality instead of ArrayRef's operator== which does
  // element-wise comparison. We want to check if it's the same slice of the
  // underlying array, not just equivalent content.
  bool operator==(const CallSiteEntry &Other) const {
    return Frames.data() == Other.Frames.data() &&
           Frames.size() == Other.Frames.size();
  }
};

struct CallSiteEntryHash {
  size_t operator()(const CallSiteEntry &Entry) const {
    return computeFullStackId(Entry.Frames);
  }
};

static void handleCallSite(
    Instruction &I, const Function *CalledFunction,
    ArrayRef<uint64_t> InlinedCallStack,
    const std::unordered_set<CallSiteEntry, CallSiteEntryHash> &CallSiteEntries,
    Module &M, std::set<std::vector<uint64_t>> &MatchedCallSites) {
  auto &Ctx = M.getContext();
  for (const auto &CallSiteEntry : CallSiteEntries) {
    // If we found and thus matched all frames on the call, create and
    // attach call stack metadata.
    if (stackFrameIncludesInlinedCallStack(CallSiteEntry.Frames,
                                           InlinedCallStack)) {
      NumOfMemProfMatchedCallSites++;
      addCallsiteMetadata(I, InlinedCallStack, Ctx);

      // Try to attach indirect call metadata if possible.
      if (!CalledFunction)
        addVPMetadata(M, I, CallSiteEntry.CalleeGuids);

      // Only need to find one with a matching call stack and add a single
      // callsite metadata.

      // Accumulate call site matching information upon request.
      if (ClPrintMemProfMatchInfo) {
        std::vector<uint64_t> CallStack;
        append_range(CallStack, InlinedCallStack);
        MatchedCallSites.insert(std::move(CallStack));
      }
      break;
    }
  }
}

static void readMemprof(Module &M, Function &F,
                        IndexedInstrProfReader *MemProfReader,
                        const TargetLibraryInfo &TLI,
                        std::map<std::pair<uint64_t, unsigned>, AllocMatchInfo>
                            &FullStackIdToAllocMatchInfo,
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
  std::map<uint64_t, std::unordered_set<CallSiteEntry, CallSiteEntryHash>>
      LocHashToCallSites;
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
      ArrayRef<GlobalValue::GUID> CalleeGuids(CS.CalleeGuids);
      LocHashToCallSites[StackId].insert({FrameSlice, CalleeGuids});

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
                       CallSitesIter->second, M, MatchedCallSites);
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

  // Map from the stack hash and matched frame count of each allocation context
  // in the function profiles to the total profiled size (bytes) and allocation
  // type.
  std::map<std::pair<uint64_t, unsigned>, AllocMatchInfo>
      FullStackIdToAllocMatchInfo;

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
    for (const auto &[IdLengthPair, Info] : FullStackIdToAllocMatchInfo) {
      auto [Id, Length] = IdLengthPair;
      errs() << "MemProf " << getAllocTypeAttributeString(Info.AllocType)
             << " context with id " << Id << " has total profiled size "
             << Info.TotalSize << " is matched with " << Length << " frames\n";
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

// Returns true iff the global variable has custom section either by
// __attribute__((section("name")))
// (https://clang.llvm.org/docs/AttributeReference.html#section-declspec-allocate)
// or #pragma clang section directives
// (https://clang.llvm.org/docs/LanguageExtensions.html#specifying-section-names-for-global-objects-pragma-clang-section).
static bool hasExplicitSectionName(const GlobalVariable &GVar) {
  if (GVar.hasSection())
    return true;

  auto Attrs = GVar.getAttributes();
  if (Attrs.hasAttribute("bss-section") || Attrs.hasAttribute("data-section") ||
      Attrs.hasAttribute("relro-section") ||
      Attrs.hasAttribute("rodata-section"))
    return true;
  return false;
}

bool MemProfUsePass::annotateGlobalVariables(
    Module &M, const memprof::DataAccessProfData *DataAccessProf) {
  if (!AnnotateStaticDataSectionPrefix || M.globals().empty())
    return false;

  if (!DataAccessProf) {
    M.getContext().diagnose(DiagnosticInfoPGOProfile(
        MemoryProfileFileName.data(),
        StringRef("Data access profiles not found in memprof. Ignore "
                  "-memprof-annotate-static-data-prefix."),
        DS_Warning));
    return false;
  }

  bool Changed = false;
  // Iterate all global variables in the module and annotate them based on
  // data access profiles. Note it's up to the linker to decide how to map input
  // sections to output sections, and one conservative practice is to map
  // unlikely-prefixed ones to unlikely output section, and map the rest
  // (hot-prefixed or prefix-less) to the canonical output section.
  for (GlobalVariable &GVar : M.globals()) {
    assert(!GVar.getSectionPrefix().has_value() &&
           "GVar shouldn't have section prefix yet");
    if (GVar.isDeclarationForLinker())
      continue;

    if (hasExplicitSectionName(GVar)) {
      ++NumOfMemProfExplicitSectionGlobalVars;
      LLVM_DEBUG(dbgs() << "Global variable " << GVar.getName()
                        << " has explicit section name. Skip annotating.\n");
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
      GVar.setSectionPrefix("hot");
      Changed = true;
      LLVM_DEBUG(dbgs() << "Global variable " << Name
                        << " is annotated as hot\n");
    } else if (DataAccessProf->isKnownColdSymbol(Name)) {
      ++NumOfMemProfColdGlobalVars;
      GVar.setSectionPrefix("unlikely");
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
