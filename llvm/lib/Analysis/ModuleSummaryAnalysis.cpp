//===- ModuleSummaryAnalysis.cpp - Module summary index builder -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass builds a ModuleSummaryIndex object for the module, to be written
// to bitcode or LLVM assembly.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "module-summary-analysis"

// Walk through the operands of a given User via worklist iteration and populate
// the set of GlobalValue references encountered. Invoked either on an
// Instruction or a GlobalVariable (which walks its initializer).
static void findRefEdges(ModuleSummaryIndex &Index, const User *CurUser,
                         SetVector<ValueInfo> &RefEdges,
                         SmallPtrSet<const User *, 8> &Visited) {
  SmallVector<const User *, 32> Worklist;
  Worklist.push_back(CurUser);

  while (!Worklist.empty()) {
    const User *U = Worklist.pop_back_val();

    if (!Visited.insert(U).second)
      continue;

    ImmutableCallSite CS(U);

    for (const auto &OI : U->operands()) {
      const User *Operand = dyn_cast<User>(OI);
      if (!Operand)
        continue;
      if (isa<BlockAddress>(Operand))
        continue;
      if (auto *GV = dyn_cast<GlobalValue>(Operand)) {
        // We have a reference to a global value. This should be added to
        // the reference set unless it is a callee. Callees are handled
        // specially by WriteFunction and are added to a separate list.
        if (!(CS && CS.isCallee(&OI)))
          RefEdges.insert(Index.getOrInsertValueInfo(GV));
        continue;
      }
      Worklist.push_back(Operand);
    }
  }
}

static CalleeInfo::HotnessType getHotness(uint64_t ProfileCount,
                                          ProfileSummaryInfo *PSI) {
  if (!PSI)
    return CalleeInfo::HotnessType::Unknown;
  if (PSI->isHotCount(ProfileCount))
    return CalleeInfo::HotnessType::Hot;
  if (PSI->isColdCount(ProfileCount))
    return CalleeInfo::HotnessType::Cold;
  return CalleeInfo::HotnessType::None;
}

static bool isNonRenamableLocal(const GlobalValue &GV) {
  return GV.hasSection() && GV.hasLocalLinkage();
}

/// Determine whether this call has all constant integer arguments (excluding
/// "this") and summarize it to VCalls or ConstVCalls as appropriate.
static void addVCallToSet(DevirtCallSite Call, GlobalValue::GUID Guid,
                          SetVector<FunctionSummary::VFuncId> &VCalls,
                          SetVector<FunctionSummary::ConstVCall> &ConstVCalls) {
  std::vector<uint64_t> Args;
  // Start from the second argument to skip the "this" pointer.
  for (auto &Arg : make_range(Call.CS.arg_begin() + 1, Call.CS.arg_end())) {
    auto *CI = dyn_cast<ConstantInt>(Arg);
    if (!CI || CI->getBitWidth() > 64) {
      VCalls.insert({Guid, Call.Offset});
      return;
    }
    Args.push_back(CI->getZExtValue());
  }
  ConstVCalls.insert({{Guid, Call.Offset}, std::move(Args)});
}

/// If this intrinsic call requires that we add information to the function
/// summary, do so via the non-constant reference arguments.
static void addIntrinsicToSummary(
    const CallInst *CI, SetVector<GlobalValue::GUID> &TypeTests,
    SetVector<FunctionSummary::VFuncId> &TypeTestAssumeVCalls,
    SetVector<FunctionSummary::VFuncId> &TypeCheckedLoadVCalls,
    SetVector<FunctionSummary::ConstVCall> &TypeTestAssumeConstVCalls,
    SetVector<FunctionSummary::ConstVCall> &TypeCheckedLoadConstVCalls) {
  switch (CI->getCalledFunction()->getIntrinsicID()) {
  case Intrinsic::type_test: {
    auto *TypeMDVal = cast<MetadataAsValue>(CI->getArgOperand(1));
    auto *TypeId = dyn_cast<MDString>(TypeMDVal->getMetadata());
    if (!TypeId)
      break;
    GlobalValue::GUID Guid = GlobalValue::getGUID(TypeId->getString());

    // Produce a summary from type.test intrinsics. We only summarize type.test
    // intrinsics that are used other than by an llvm.assume intrinsic.
    // Intrinsics that are assumed are relevant only to the devirtualization
    // pass, not the type test lowering pass.
    bool HasNonAssumeUses = llvm::any_of(CI->uses(), [](const Use &CIU) {
      auto *AssumeCI = dyn_cast<CallInst>(CIU.getUser());
      if (!AssumeCI)
        return true;
      Function *F = AssumeCI->getCalledFunction();
      return !F || F->getIntrinsicID() != Intrinsic::assume;
    });
    if (HasNonAssumeUses)
      TypeTests.insert(Guid);

    SmallVector<DevirtCallSite, 4> DevirtCalls;
    SmallVector<CallInst *, 4> Assumes;
    findDevirtualizableCallsForTypeTest(DevirtCalls, Assumes, CI);
    for (auto &Call : DevirtCalls)
      addVCallToSet(Call, Guid, TypeTestAssumeVCalls,
                    TypeTestAssumeConstVCalls);

    break;
  }

  case Intrinsic::type_checked_load: {
    auto *TypeMDVal = cast<MetadataAsValue>(CI->getArgOperand(2));
    auto *TypeId = dyn_cast<MDString>(TypeMDVal->getMetadata());
    if (!TypeId)
      break;
    GlobalValue::GUID Guid = GlobalValue::getGUID(TypeId->getString());

    SmallVector<DevirtCallSite, 4> DevirtCalls;
    SmallVector<Instruction *, 4> LoadedPtrs;
    SmallVector<Instruction *, 4> Preds;
    bool HasNonCallUses = false;
    findDevirtualizableCallsForTypeCheckedLoad(DevirtCalls, LoadedPtrs, Preds,
                                               HasNonCallUses, CI);
    // Any non-call uses of the result of llvm.type.checked.load will
    // prevent us from optimizing away the llvm.type.test.
    if (HasNonCallUses)
      TypeTests.insert(Guid);
    for (auto &Call : DevirtCalls)
      addVCallToSet(Call, Guid, TypeCheckedLoadVCalls,
                    TypeCheckedLoadConstVCalls);

    break;
  }
  default:
    break;
  }
}

static void
computeFunctionSummary(ModuleSummaryIndex &Index, const Module &M,
                       const Function &F, BlockFrequencyInfo *BFI,
                       ProfileSummaryInfo *PSI, bool HasLocalsInUsed,
                       DenseSet<GlobalValue::GUID> &CantBePromoted) {
  // Summary not currently supported for anonymous functions, they should
  // have been named.
  assert(F.hasName());

  unsigned NumInsts = 0;
  // Map from callee ValueId to profile count. Used to accumulate profile
  // counts for all static calls to a given callee.
  MapVector<ValueInfo, CalleeInfo> CallGraphEdges;
  SetVector<ValueInfo> RefEdges;
  SetVector<GlobalValue::GUID> TypeTests;
  SetVector<FunctionSummary::VFuncId> TypeTestAssumeVCalls,
      TypeCheckedLoadVCalls;
  SetVector<FunctionSummary::ConstVCall> TypeTestAssumeConstVCalls,
      TypeCheckedLoadConstVCalls;
  ICallPromotionAnalysis ICallAnalysis;

  bool HasInlineAsmMaybeReferencingInternal = false;
  SmallPtrSet<const User *, 8> Visited;
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB) {
      if (isa<DbgInfoIntrinsic>(I))
        continue;
      ++NumInsts;
      findRefEdges(Index, &I, RefEdges, Visited);
      auto CS = ImmutableCallSite(&I);
      if (!CS)
        continue;

      const auto *CI = dyn_cast<CallInst>(&I);
      // Since we don't know exactly which local values are referenced in inline
      // assembly, conservatively mark the function as possibly referencing
      // a local value from inline assembly to ensure we don't export a
      // reference (which would require renaming and promotion of the
      // referenced value).
      if (HasLocalsInUsed && CI && CI->isInlineAsm())
        HasInlineAsmMaybeReferencingInternal = true;

      auto *CalledValue = CS.getCalledValue();
      auto *CalledFunction = CS.getCalledFunction();
      // Check if this is an alias to a function. If so, get the
      // called aliasee for the checks below.
      if (auto *GA = dyn_cast<GlobalAlias>(CalledValue)) {
        assert(!CalledFunction && "Expected null called function in callsite for alias");
        CalledFunction = dyn_cast<Function>(GA->getBaseObject());
      }
      // Check if this is a direct call to a known function or a known
      // intrinsic, or an indirect call with profile data.
      if (CalledFunction) {
        if (CI && CalledFunction->isIntrinsic()) {
          addIntrinsicToSummary(
              CI, TypeTests, TypeTestAssumeVCalls, TypeCheckedLoadVCalls,
              TypeTestAssumeConstVCalls, TypeCheckedLoadConstVCalls);
          continue;
        }
        // We should have named any anonymous globals
        assert(CalledFunction->hasName());
        auto ScaledCount = PSI->getProfileCount(&I, BFI);
        auto Hotness = ScaledCount ? getHotness(ScaledCount.getValue(), PSI)
                                   : CalleeInfo::HotnessType::Unknown;

        // Use the original CalledValue, in case it was an alias. We want
        // to record the call edge to the alias in that case. Eventually
        // an alias summary will be created to associate the alias and
        // aliasee.
        CallGraphEdges[Index.getOrInsertValueInfo(
                           cast<GlobalValue>(CalledValue))]
            .updateHotness(Hotness);
      } else {
        // Skip inline assembly calls.
        if (CI && CI->isInlineAsm())
          continue;
        // Skip direct calls.
        if (!CS.getCalledValue() || isa<Constant>(CS.getCalledValue()))
          continue;

        uint32_t NumVals, NumCandidates;
        uint64_t TotalCount;
        auto CandidateProfileData =
            ICallAnalysis.getPromotionCandidatesForInstruction(
                &I, NumVals, TotalCount, NumCandidates);
        for (auto &Candidate : CandidateProfileData)
          CallGraphEdges[Index.getOrInsertValueInfo(Candidate.Value)]
              .updateHotness(getHotness(Candidate.Count, PSI));
      }
    }

  // Explicit add hot edges to enforce importing for designated GUIDs for
  // sample PGO, to enable the same inlines as the profiled optimized binary.
  for (auto &I : F.getImportGUIDs())
    CallGraphEdges[Index.getOrInsertValueInfo(I)].updateHotness(
        CalleeInfo::HotnessType::Critical);

  bool NonRenamableLocal = isNonRenamableLocal(F);
  bool NotEligibleForImport =
      NonRenamableLocal || HasInlineAsmMaybeReferencingInternal ||
      // Inliner doesn't handle variadic functions.
      // FIXME: refactor this to use the same code that inliner is using.
      F.isVarArg();
  GlobalValueSummary::GVFlags Flags(F.getLinkage(), NotEligibleForImport,
                                    /* Live = */ false);
  FunctionSummary::FFlags FunFlags{
      F.hasFnAttribute(Attribute::ReadNone),
      F.hasFnAttribute(Attribute::ReadOnly),
      F.hasFnAttribute(Attribute::NoRecurse),
      F.returnDoesNotAlias(),
  };
  auto FuncSummary = llvm::make_unique<FunctionSummary>(
      Flags, NumInsts, FunFlags, RefEdges.takeVector(),
      CallGraphEdges.takeVector(), TypeTests.takeVector(),
      TypeTestAssumeVCalls.takeVector(), TypeCheckedLoadVCalls.takeVector(),
      TypeTestAssumeConstVCalls.takeVector(),
      TypeCheckedLoadConstVCalls.takeVector());
  if (NonRenamableLocal)
    CantBePromoted.insert(F.getGUID());
  Index.addGlobalValueSummary(F.getName(), std::move(FuncSummary));
}

static void
computeVariableSummary(ModuleSummaryIndex &Index, const GlobalVariable &V,
                       DenseSet<GlobalValue::GUID> &CantBePromoted) {
  SetVector<ValueInfo> RefEdges;
  SmallPtrSet<const User *, 8> Visited;
  findRefEdges(Index, &V, RefEdges, Visited);
  bool NonRenamableLocal = isNonRenamableLocal(V);
  GlobalValueSummary::GVFlags Flags(V.getLinkage(), NonRenamableLocal,
                                    /* Live = */ false);
  auto GVarSummary =
      llvm::make_unique<GlobalVarSummary>(Flags, RefEdges.takeVector());
  if (NonRenamableLocal)
    CantBePromoted.insert(V.getGUID());
  Index.addGlobalValueSummary(V.getName(), std::move(GVarSummary));
}

static void
computeAliasSummary(ModuleSummaryIndex &Index, const GlobalAlias &A,
                    DenseSet<GlobalValue::GUID> &CantBePromoted) {
  bool NonRenamableLocal = isNonRenamableLocal(A);
  GlobalValueSummary::GVFlags Flags(A.getLinkage(), NonRenamableLocal,
                                    /* Live = */ false);
  auto AS = llvm::make_unique<AliasSummary>(Flags, ArrayRef<ValueInfo>{});
  auto *Aliasee = A.getBaseObject();
  auto *AliaseeSummary = Index.getGlobalValueSummary(*Aliasee);
  assert(AliaseeSummary && "Alias expects aliasee summary to be parsed");
  AS->setAliasee(AliaseeSummary);
  if (NonRenamableLocal)
    CantBePromoted.insert(A.getGUID());
  Index.addGlobalValueSummary(A.getName(), std::move(AS));
}

// Set LiveRoot flag on entries matching the given value name.
static void setLiveRoot(ModuleSummaryIndex &Index, StringRef Name) {
  if (ValueInfo VI = Index.getValueInfo(GlobalValue::getGUID(Name)))
    for (auto &Summary : VI.getSummaryList())
      Summary->setLive(true);
}

ModuleSummaryIndex llvm::buildModuleSummaryIndex(
    const Module &M,
    std::function<BlockFrequencyInfo *(const Function &F)> GetBFICallback,
    ProfileSummaryInfo *PSI) {
  assert(PSI);
  ModuleSummaryIndex Index;

  // Identify the local values in the llvm.used and llvm.compiler.used sets,
  // which should not be exported as they would then require renaming and
  // promotion, but we may have opaque uses e.g. in inline asm. We collect them
  // here because we use this information to mark functions containing inline
  // assembly calls as not importable.
  SmallPtrSet<GlobalValue *, 8> LocalsUsed;
  SmallPtrSet<GlobalValue *, 8> Used;
  // First collect those in the llvm.used set.
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ false);
  // Next collect those in the llvm.compiler.used set.
  collectUsedGlobalVariables(M, Used, /*CompilerUsed*/ true);
  DenseSet<GlobalValue::GUID> CantBePromoted;
  for (auto *V : Used) {
    if (V->hasLocalLinkage()) {
      LocalsUsed.insert(V);
      CantBePromoted.insert(V->getGUID());
    }
  }

  // Compute summaries for all functions defined in module, and save in the
  // index.
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    BlockFrequencyInfo *BFI = nullptr;
    std::unique_ptr<BlockFrequencyInfo> BFIPtr;
    if (GetBFICallback)
      BFI = GetBFICallback(F);
    else if (F.getEntryCount().hasValue()) {
      LoopInfo LI{DominatorTree(const_cast<Function &>(F))};
      BranchProbabilityInfo BPI{F, LI};
      BFIPtr = llvm::make_unique<BlockFrequencyInfo>(F, BPI, LI);
      BFI = BFIPtr.get();
    }

    computeFunctionSummary(Index, M, F, BFI, PSI, !LocalsUsed.empty(),
                           CantBePromoted);
  }

  // Set live flag for all personality functions. That allows to
  // preserve them during DCE.
  for (const llvm::Function &F : M)
    if (!F.isDeclaration() && F.hasPersonalityFn())
      setLiveRoot(Index, F.getPersonalityFn()->getName());

  // Compute summaries for all variables defined in module, and save in the
  // index.
  for (const GlobalVariable &G : M.globals()) {
    if (G.isDeclaration())
      continue;
    computeVariableSummary(Index, G, CantBePromoted);
  }

  // Compute summaries for all aliases defined in module, and save in the
  // index.
  for (const GlobalAlias &A : M.aliases())
    computeAliasSummary(Index, A, CantBePromoted);

  for (auto *V : LocalsUsed) {
    auto *Summary = Index.getGlobalValueSummary(*V);
    assert(Summary && "Missing summary for global value");
    Summary->setNotEligibleToImport();
  }

  // The linker doesn't know about these LLVM produced values, so we need
  // to flag them as live in the index to ensure index-based dead value
  // analysis treats them as live roots of the analysis.
  setLiveRoot(Index, "llvm.used");
  setLiveRoot(Index, "llvm.compiler.used");
  setLiveRoot(Index, "llvm.global_ctors");
  setLiveRoot(Index, "llvm.global_dtors");
  setLiveRoot(Index, "llvm.global.annotations");

  if (!M.getModuleInlineAsm().empty()) {
    // Collect the local values defined by module level asm, and set up
    // summaries for these symbols so that they can be marked as NoRename,
    // to prevent export of any use of them in regular IR that would require
    // renaming within the module level asm. Note we don't need to create a
    // summary for weak or global defs, as they don't need to be flagged as
    // NoRename, and defs in module level asm can't be imported anyway.
    // Also, any values used but not defined within module level asm should
    // be listed on the llvm.used or llvm.compiler.used global and marked as
    // referenced from there.
    ModuleSymbolTable::CollectAsmSymbols(
        M, [&M, &Index, &CantBePromoted](StringRef Name,
                                         object::BasicSymbolRef::Flags Flags) {
          // Symbols not marked as Weak or Global are local definitions.
          if (Flags & (object::BasicSymbolRef::SF_Weak |
                       object::BasicSymbolRef::SF_Global))
            return;
          GlobalValue *GV = M.getNamedValue(Name);
          if (!GV)
            return;
          assert(GV->isDeclaration() && "Def in module asm already has definition");
          GlobalValueSummary::GVFlags GVFlags(GlobalValue::InternalLinkage,
                                              /* NotEligibleToImport = */ true,
                                              /* Live = */ true);
          CantBePromoted.insert(GlobalValue::getGUID(Name));
          // Create the appropriate summary type.
          if (Function *F = dyn_cast<Function>(GV)) {
            std::unique_ptr<FunctionSummary> Summary =
                llvm::make_unique<FunctionSummary>(
                    GVFlags, 0,
                    FunctionSummary::FFlags{
                        F->hasFnAttribute(Attribute::ReadNone),
                        F->hasFnAttribute(Attribute::ReadOnly),
                        F->hasFnAttribute(Attribute::NoRecurse),
                        F->returnDoesNotAlias()},
                    ArrayRef<ValueInfo>{}, ArrayRef<FunctionSummary::EdgeTy>{},
                    ArrayRef<GlobalValue::GUID>{},
                    ArrayRef<FunctionSummary::VFuncId>{},
                    ArrayRef<FunctionSummary::VFuncId>{},
                    ArrayRef<FunctionSummary::ConstVCall>{},
                    ArrayRef<FunctionSummary::ConstVCall>{});
            Index.addGlobalValueSummary(Name, std::move(Summary));
          } else {
            std::unique_ptr<GlobalVarSummary> Summary =
                llvm::make_unique<GlobalVarSummary>(GVFlags,
                                                    ArrayRef<ValueInfo>{});
            Index.addGlobalValueSummary(Name, std::move(Summary));
          }
        });
  }

  bool IsThinLTO = true;
  if (auto *MD =
          mdconst::extract_or_null<ConstantInt>(M.getModuleFlag("ThinLTO")))
    IsThinLTO = MD->getZExtValue();

  for (auto &GlobalList : Index) {
    // Ignore entries for references that are undefined in the current module.
    if (GlobalList.second.SummaryList.empty())
      continue;

    assert(GlobalList.second.SummaryList.size() == 1 &&
           "Expected module's index to have one summary per GUID");
    auto &Summary = GlobalList.second.SummaryList[0];
    if (!IsThinLTO) {
      Summary->setNotEligibleToImport();
      continue;
    }

    bool AllRefsCanBeExternallyReferenced =
        llvm::all_of(Summary->refs(), [&](const ValueInfo &VI) {
          return !CantBePromoted.count(VI.getGUID());
        });
    if (!AllRefsCanBeExternallyReferenced) {
      Summary->setNotEligibleToImport();
      continue;
    }

    if (auto *FuncSummary = dyn_cast<FunctionSummary>(Summary.get())) {
      bool AllCallsCanBeExternallyReferenced = llvm::all_of(
          FuncSummary->calls(), [&](const FunctionSummary::EdgeTy &Edge) {
            return !CantBePromoted.count(Edge.first.getGUID());
          });
      if (!AllCallsCanBeExternallyReferenced)
        Summary->setNotEligibleToImport();
    }
  }

  return Index;
}

AnalysisKey ModuleSummaryIndexAnalysis::Key;

ModuleSummaryIndex
ModuleSummaryIndexAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  ProfileSummaryInfo &PSI = AM.getResult<ProfileSummaryAnalysis>(M);
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  return buildModuleSummaryIndex(
      M,
      [&FAM](const Function &F) {
        return &FAM.getResult<BlockFrequencyAnalysis>(
            *const_cast<Function *>(&F));
      },
      &PSI);
}

char ModuleSummaryIndexWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(ModuleSummaryIndexWrapperPass, "module-summary-analysis",
                      "Module Summary Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_END(ModuleSummaryIndexWrapperPass, "module-summary-analysis",
                    "Module Summary Analysis", false, true)

ModulePass *llvm::createModuleSummaryIndexWrapperPass() {
  return new ModuleSummaryIndexWrapperPass();
}

ModuleSummaryIndexWrapperPass::ModuleSummaryIndexWrapperPass()
    : ModulePass(ID) {
  initializeModuleSummaryIndexWrapperPassPass(*PassRegistry::getPassRegistry());
}

bool ModuleSummaryIndexWrapperPass::runOnModule(Module &M) {
  auto &PSI = *getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  Index = buildModuleSummaryIndex(
      M,
      [this](const Function &F) {
        return &(this->getAnalysis<BlockFrequencyInfoWrapperPass>(
                         *const_cast<Function *>(&F))
                     .getBFI());
      },
      &PSI);
  return false;
}

bool ModuleSummaryIndexWrapperPass::doFinalization(Module &M) {
  Index.reset();
  return false;
}

void ModuleSummaryIndexWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
}
