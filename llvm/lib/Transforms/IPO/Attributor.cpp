//===- Attributor.cpp - Module-wide attribute deduction -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an interprocedural pass that deduces and/or propagates
// attributes. This is done in an abstract interpretation style fixpoint
// iteration. See the Attributor.h file comment and the class descriptions in
// that file for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Attributor.h"

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#ifdef EXPENSIVE_CHECKS
#include "llvm/IR/Verifier.h"
#endif

#include <cassert>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "attributor"

DEBUG_COUNTER(ManifestDBGCounter, "attributor-manifest",
              "Determine what attributes are manifested in the IR");

STATISTIC(NumFnDeleted, "Number of function deleted");
STATISTIC(NumFnWithExactDefinition,
          "Number of functions with exact definitions");
STATISTIC(NumFnWithoutExactDefinition,
          "Number of functions without exact definitions");
STATISTIC(NumFnShallowWrappersCreated, "Number of shallow wrappers created");
STATISTIC(NumAttributesTimedOut,
          "Number of abstract attributes timed out before fixpoint");
STATISTIC(NumAttributesValidFixpoint,
          "Number of abstract attributes in a valid fixpoint state");
STATISTIC(NumAttributesManifested,
          "Number of abstract attributes manifested in IR");

// TODO: Determine a good default value.
//
// In the LLVM-TS and SPEC2006, 32 seems to not induce compile time overheads
// (when run with the first 5 abstract attributes). The results also indicate
// that we never reach 32 iterations but always find a fixpoint sooner.
//
// This will become more evolved once we perform two interleaved fixpoint
// iterations: bottom-up and top-down.
static cl::opt<unsigned>
    SetFixpointIterations("attributor-max-iterations", cl::Hidden,
                          cl::desc("Maximal number of fixpoint iterations."),
                          cl::init(32));

static cl::opt<unsigned, true> MaxInitializationChainLengthX(
    "attributor-max-initialization-chain-length", cl::Hidden,
    cl::desc(
        "Maximal number of chained initializations (to avoid stack overflows)"),
    cl::location(MaxInitializationChainLength), cl::init(1024));
unsigned llvm::MaxInitializationChainLength;

static cl::opt<bool> VerifyMaxFixpointIterations(
    "attributor-max-iterations-verify", cl::Hidden,
    cl::desc("Verify that max-iterations is a tight bound for a fixpoint"),
    cl::init(false));

static cl::opt<bool> AnnotateDeclarationCallSites(
    "attributor-annotate-decl-cs", cl::Hidden,
    cl::desc("Annotate call sites of function declarations."), cl::init(false));

static cl::opt<bool> EnableHeapToStack("enable-heap-to-stack-conversion",
                                       cl::init(true), cl::Hidden);

static cl::opt<bool>
    AllowShallowWrappers("attributor-allow-shallow-wrappers", cl::Hidden,
                         cl::desc("Allow the Attributor to create shallow "
                                  "wrappers for non-exact definitions."),
                         cl::init(false));

static cl::opt<bool>
    AllowDeepWrapper("attributor-allow-deep-wrappers", cl::Hidden,
                     cl::desc("Allow the Attributor to use IP information "
                              "derived from non-exact functions via cloning"),
                     cl::init(false));

// These options can only used for debug builds.
#ifndef NDEBUG
static cl::list<std::string>
    SeedAllowList("attributor-seed-allow-list", cl::Hidden,
                  cl::desc("Comma seperated list of attribute names that are "
                           "allowed to be seeded."),
                  cl::CommaSeparated);

static cl::list<std::string> FunctionSeedAllowList(
    "attributor-function-seed-allow-list", cl::Hidden,
    cl::desc("Comma seperated list of function names that are "
             "allowed to be seeded."),
    cl::CommaSeparated);
#endif

static cl::opt<bool>
    DumpDepGraph("attributor-dump-dep-graph", cl::Hidden,
                 cl::desc("Dump the dependency graph to dot files."),
                 cl::init(false));

static cl::opt<std::string> DepGraphDotFileNamePrefix(
    "attributor-depgraph-dot-filename-prefix", cl::Hidden,
    cl::desc("The prefix used for the CallGraph dot file names."));

static cl::opt<bool> ViewDepGraph("attributor-view-dep-graph", cl::Hidden,
                                  cl::desc("View the dependency graph."),
                                  cl::init(false));

static cl::opt<bool> PrintDependencies("attributor-print-dep", cl::Hidden,
                                       cl::desc("Print attribute dependencies"),
                                       cl::init(false));

static cl::opt<bool> EnableCallSiteSpecific(
    "attributor-enable-call-site-specific-deduction", cl::Hidden,
    cl::desc("Allow the Attributor to do call site specific analysis"),
    cl::init(false));

static cl::opt<bool>
    PrintCallGraph("attributor-print-call-graph", cl::Hidden,
                   cl::desc("Print Attributor's internal call graph"),
                   cl::init(false));

static cl::opt<bool> SimplifyAllLoads("attributor-simplify-all-loads",
                                      cl::Hidden,
                                      cl::desc("Try to simplify all loads."),
                                      cl::init(true));

/// Logic operators for the change status enum class.
///
///{
ChangeStatus llvm::operator|(ChangeStatus L, ChangeStatus R) {
  return L == ChangeStatus::CHANGED ? L : R;
}
ChangeStatus &llvm::operator|=(ChangeStatus &L, ChangeStatus R) {
  L = L | R;
  return L;
}
ChangeStatus llvm::operator&(ChangeStatus L, ChangeStatus R) {
  return L == ChangeStatus::UNCHANGED ? L : R;
}
ChangeStatus &llvm::operator&=(ChangeStatus &L, ChangeStatus R) {
  L = L & R;
  return L;
}
///}

bool AA::isNoSyncInst(Attributor &A, const Instruction &I,
                      const AbstractAttribute &QueryingAA) {
  // We are looking for volatile instructions or non-relaxed atomics.
  if (const auto *CB = dyn_cast<CallBase>(&I)) {
    if (CB->hasFnAttr(Attribute::NoSync))
      return true;

    // Non-convergent and readnone imply nosync.
    if (!CB->isConvergent() && !CB->mayReadOrWriteMemory())
      return true;

    if (AANoSync::isNoSyncIntrinsic(&I))
      return true;

    const auto &NoSyncAA = A.getAAFor<AANoSync>(
        QueryingAA, IRPosition::callsite_function(*CB), DepClassTy::OPTIONAL);
    return NoSyncAA.isAssumedNoSync();
  }

  if (!I.mayReadOrWriteMemory())
    return true;

  return !I.isVolatile() && !AANoSync::isNonRelaxedAtomic(&I);
}

bool AA::isDynamicallyUnique(Attributor &A, const AbstractAttribute &QueryingAA,
                             const Value &V, bool ForAnalysisOnly) {
  // TODO: See the AAInstanceInfo class comment.
  if (!ForAnalysisOnly)
    return false;
  auto &InstanceInfoAA = A.getAAFor<AAInstanceInfo>(
      QueryingAA, IRPosition::value(V), DepClassTy::OPTIONAL);
  return InstanceInfoAA.isAssumedUniqueForAnalysis();
}

Constant *AA::getInitialValueForObj(Value &Obj, Type &Ty,
                                    const TargetLibraryInfo *TLI) {
  if (isa<AllocaInst>(Obj))
    return UndefValue::get(&Ty);
  if (isAllocationFn(&Obj, TLI))
    return getInitialValueOfAllocation(&cast<CallBase>(Obj), TLI, &Ty);
  auto *GV = dyn_cast<GlobalVariable>(&Obj);
  if (!GV)
    return nullptr;
  if (!GV->hasLocalLinkage() && !(GV->isConstant() && GV->hasInitializer()))
    return nullptr;
  if (!GV->hasInitializer())
    return UndefValue::get(&Ty);
  return dyn_cast_or_null<Constant>(getWithType(*GV->getInitializer(), Ty));
}

bool AA::isValidInScope(const Value &V, const Function *Scope) {
  if (isa<Constant>(V))
    return true;
  if (auto *I = dyn_cast<Instruction>(&V))
    return I->getFunction() == Scope;
  if (auto *A = dyn_cast<Argument>(&V))
    return A->getParent() == Scope;
  return false;
}

bool AA::isValidAtPosition(const AA::ValueAndContext &VAC,
                           InformationCache &InfoCache) {
  if (isa<Constant>(VAC.getValue()) || VAC.getValue() == VAC.getCtxI())
    return true;
  const Function *Scope = nullptr;
  const Instruction *CtxI = VAC.getCtxI();
  if (CtxI)
    Scope = CtxI->getFunction();
  if (auto *A = dyn_cast<Argument>(VAC.getValue()))
    return A->getParent() == Scope;
  if (auto *I = dyn_cast<Instruction>(VAC.getValue())) {
    if (I->getFunction() == Scope) {
      if (const DominatorTree *DT =
              InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(
                  *Scope))
        return DT->dominates(I, CtxI);
      // Local dominance check mostly for the old PM passes.
      if (CtxI && I->getParent() == CtxI->getParent())
        return llvm::any_of(
            make_range(I->getIterator(), I->getParent()->end()),
            [&](const Instruction &AfterI) { return &AfterI == CtxI; });
    }
  }
  return false;
}

Value *AA::getWithType(Value &V, Type &Ty) {
  if (V.getType() == &Ty)
    return &V;
  if (isa<PoisonValue>(V))
    return PoisonValue::get(&Ty);
  if (isa<UndefValue>(V))
    return UndefValue::get(&Ty);
  if (auto *C = dyn_cast<Constant>(&V)) {
    if (C->isNullValue())
      return Constant::getNullValue(&Ty);
    if (C->getType()->isPointerTy() && Ty.isPointerTy())
      return ConstantExpr::getPointerCast(C, &Ty);
    if (C->getType()->getPrimitiveSizeInBits() >= Ty.getPrimitiveSizeInBits()) {
      if (C->getType()->isIntegerTy() && Ty.isIntegerTy())
        return ConstantExpr::getTrunc(C, &Ty, /* OnlyIfReduced */ true);
      if (C->getType()->isFloatingPointTy() && Ty.isFloatingPointTy())
        return ConstantExpr::getFPTrunc(C, &Ty, /* OnlyIfReduced */ true);
    }
  }
  return nullptr;
}

Optional<Value *>
AA::combineOptionalValuesInAAValueLatice(const Optional<Value *> &A,
                                         const Optional<Value *> &B, Type *Ty) {
  if (A == B)
    return A;
  if (!B.hasValue())
    return A;
  if (*B == nullptr)
    return nullptr;
  if (!A.hasValue())
    return Ty ? getWithType(**B, *Ty) : nullptr;
  if (*A == nullptr)
    return nullptr;
  if (!Ty)
    Ty = (*A)->getType();
  if (isa_and_nonnull<UndefValue>(*A))
    return getWithType(**B, *Ty);
  if (isa<UndefValue>(*B))
    return A;
  if (*A && *B && *A == getWithType(**B, *Ty))
    return A;
  return nullptr;
}

template <bool IsLoad, typename Ty>
static bool getPotentialCopiesOfMemoryValue(
    Attributor &A, Ty &I, SmallSetVector<Value *, 4> &PotentialCopies,
    SmallSetVector<Instruction *, 4> &PotentialValueOrigins,
    const AbstractAttribute &QueryingAA, bool &UsedAssumedInformation,
    bool OnlyExact) {
  LLVM_DEBUG(dbgs() << "Trying to determine the potential copies of " << I
                    << " (only exact: " << OnlyExact << ")\n";);

  Value &Ptr = *I.getPointerOperand();
  SmallVector<Value *, 8> Objects;
  if (!AA::getAssumedUnderlyingObjects(A, Ptr, Objects, QueryingAA, &I,
                                       UsedAssumedInformation)) {
    LLVM_DEBUG(
        dbgs() << "Underlying objects stored into could not be determined\n";);
    return false;
  }

  // Containers to remember the pointer infos and new copies while we are not
  // sure that we can find all of them. If we abort we want to avoid spurious
  // dependences and potential copies in the provided container.
  SmallVector<const AAPointerInfo *> PIs;
  SmallVector<Value *> NewCopies;
  SmallVector<Instruction *> NewCopyOrigins;

  const auto *TLI =
      A.getInfoCache().getTargetLibraryInfoForFunction(*I.getFunction());
  for (Value *Obj : Objects) {
    LLVM_DEBUG(dbgs() << "Visit underlying object " << *Obj << "\n");
    if (isa<UndefValue>(Obj))
      continue;
    if (isa<ConstantPointerNull>(Obj)) {
      // A null pointer access can be undefined but any offset from null may
      // be OK. We do not try to optimize the latter.
      if (!NullPointerIsDefined(I.getFunction(),
                                Ptr.getType()->getPointerAddressSpace()) &&
          A.getAssumedSimplified(Ptr, QueryingAA, UsedAssumedInformation) ==
              Obj)
        continue;
      LLVM_DEBUG(
          dbgs() << "Underlying object is a valid nullptr, giving up.\n";);
      return false;
    }
    // TODO: Use assumed noalias return.
    if (!isa<AllocaInst>(Obj) && !isa<GlobalVariable>(Obj) &&
        !(IsLoad ? isAllocationFn(Obj, TLI) : isNoAliasCall(Obj))) {
      LLVM_DEBUG(dbgs() << "Underlying object is not supported yet: " << *Obj
                        << "\n";);
      return false;
    }
    if (auto *GV = dyn_cast<GlobalVariable>(Obj))
      if (!GV->hasLocalLinkage() &&
          !(GV->isConstant() && GV->hasInitializer())) {
        LLVM_DEBUG(dbgs() << "Underlying object is global with external "
                             "linkage, not supported yet: "
                          << *Obj << "\n";);
        return false;
      }

    if (IsLoad) {
      Value *InitialValue = AA::getInitialValueForObj(*Obj, *I.getType(), TLI);
      if (!InitialValue)
        return false;
      NewCopies.push_back(InitialValue);
      NewCopyOrigins.push_back(nullptr);
    }

    auto CheckAccess = [&](const AAPointerInfo::Access &Acc, bool IsExact) {
      if ((IsLoad && !Acc.isWrite()) || (!IsLoad && !Acc.isRead()))
        return true;
      if (IsLoad && Acc.isWrittenValueYetUndetermined())
        return true;
      if (OnlyExact && !IsExact &&
          !isa_and_nonnull<UndefValue>(Acc.getWrittenValue())) {
        LLVM_DEBUG(dbgs() << "Non exact access " << *Acc.getRemoteInst()
                          << ", abort!\n");
        return false;
      }
      if (IsLoad) {
        assert(isa<LoadInst>(I) && "Expected load or store instruction only!");
        if (!Acc.isWrittenValueUnknown()) {
          NewCopies.push_back(Acc.getWrittenValue());
          NewCopyOrigins.push_back(Acc.getRemoteInst());
          return true;
        }
        auto *SI = dyn_cast<StoreInst>(Acc.getRemoteInst());
        if (!SI) {
          LLVM_DEBUG(dbgs() << "Underlying object written through a non-store "
                               "instruction not supported yet: "
                            << *Acc.getRemoteInst() << "\n";);
          return false;
        }
        NewCopies.push_back(SI->getValueOperand());
        NewCopyOrigins.push_back(SI);
      } else {
        assert(isa<StoreInst>(I) && "Expected load or store instruction only!");
        auto *LI = dyn_cast<LoadInst>(Acc.getRemoteInst());
        if (!LI && OnlyExact) {
          LLVM_DEBUG(dbgs() << "Underlying object read through a non-load "
                               "instruction not supported yet: "
                            << *Acc.getRemoteInst() << "\n";);
          return false;
        }
        NewCopies.push_back(Acc.getRemoteInst());
      }
      return true;
    };

    auto &PI = A.getAAFor<AAPointerInfo>(QueryingAA, IRPosition::value(*Obj),
                                         DepClassTy::NONE);
    if (!PI.forallInterferingAccesses(A, QueryingAA, I, CheckAccess)) {
      LLVM_DEBUG(
          dbgs()
          << "Failed to verify all interfering accesses for underlying object: "
          << *Obj << "\n");
      return false;
    }
    PIs.push_back(&PI);
  }

  // Only if we were successful collection all potential copies we record
  // dependences (on non-fix AAPointerInfo AAs). We also only then modify the
  // given PotentialCopies container.
  for (auto *PI : PIs) {
    if (!PI->getState().isAtFixpoint())
      UsedAssumedInformation = true;
    A.recordDependence(*PI, QueryingAA, DepClassTy::OPTIONAL);
  }
  PotentialCopies.insert(NewCopies.begin(), NewCopies.end());
  PotentialValueOrigins.insert(NewCopyOrigins.begin(), NewCopyOrigins.end());

  return true;
}

bool AA::getPotentiallyLoadedValues(
    Attributor &A, LoadInst &LI, SmallSetVector<Value *, 4> &PotentialValues,
    SmallSetVector<Instruction *, 4> &PotentialValueOrigins,
    const AbstractAttribute &QueryingAA, bool &UsedAssumedInformation,
    bool OnlyExact) {
  return getPotentialCopiesOfMemoryValue</* IsLoad */ true>(
      A, LI, PotentialValues, PotentialValueOrigins, QueryingAA,
      UsedAssumedInformation, OnlyExact);
}

bool AA::getPotentialCopiesOfStoredValue(
    Attributor &A, StoreInst &SI, SmallSetVector<Value *, 4> &PotentialCopies,
    const AbstractAttribute &QueryingAA, bool &UsedAssumedInformation,
    bool OnlyExact) {
  SmallSetVector<Instruction *, 4> PotentialValueOrigins;
  return getPotentialCopiesOfMemoryValue</* IsLoad */ false>(
      A, SI, PotentialCopies, PotentialValueOrigins, QueryingAA,
      UsedAssumedInformation, OnlyExact);
}

static bool isAssumedReadOnlyOrReadNone(Attributor &A, const IRPosition &IRP,
                                        const AbstractAttribute &QueryingAA,
                                        bool RequireReadNone, bool &IsKnown) {

  IRPosition::Kind Kind = IRP.getPositionKind();
  if (Kind == IRPosition::IRP_FUNCTION || Kind == IRPosition::IRP_CALL_SITE) {
    const auto &MemLocAA =
        A.getAAFor<AAMemoryLocation>(QueryingAA, IRP, DepClassTy::NONE);
    if (MemLocAA.isAssumedReadNone()) {
      IsKnown = MemLocAA.isKnownReadNone();
      if (!IsKnown)
        A.recordDependence(MemLocAA, QueryingAA, DepClassTy::OPTIONAL);
      return true;
    }
  }

  const auto &MemBehaviorAA =
      A.getAAFor<AAMemoryBehavior>(QueryingAA, IRP, DepClassTy::NONE);
  if (MemBehaviorAA.isAssumedReadNone() ||
      (!RequireReadNone && MemBehaviorAA.isAssumedReadOnly())) {
    IsKnown = RequireReadNone ? MemBehaviorAA.isKnownReadNone()
                              : MemBehaviorAA.isKnownReadOnly();
    if (!IsKnown)
      A.recordDependence(MemBehaviorAA, QueryingAA, DepClassTy::OPTIONAL);
    return true;
  }

  return false;
}

bool AA::isAssumedReadOnly(Attributor &A, const IRPosition &IRP,
                           const AbstractAttribute &QueryingAA, bool &IsKnown) {
  return isAssumedReadOnlyOrReadNone(A, IRP, QueryingAA,
                                     /* RequireReadNone */ false, IsKnown);
}
bool AA::isAssumedReadNone(Attributor &A, const IRPosition &IRP,
                           const AbstractAttribute &QueryingAA, bool &IsKnown) {
  return isAssumedReadOnlyOrReadNone(A, IRP, QueryingAA,
                                     /* RequireReadNone */ true, IsKnown);
}

static bool
isPotentiallyReachable(Attributor &A, const Instruction &FromI,
                       const Instruction *ToI, const Function &ToFn,
                       const AbstractAttribute &QueryingAA,
                       std::function<bool(const Function &F)> GoBackwardsCB) {
  LLVM_DEBUG(dbgs() << "[AA] isPotentiallyReachable @" << ToFn.getName()
                    << " from " << FromI << " [GBCB: " << bool(GoBackwardsCB)
                    << "]\n");

  SmallPtrSet<const Instruction *, 8> Visited;
  SmallVector<const Instruction *> Worklist;
  Worklist.push_back(&FromI);

  const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
      QueryingAA, IRPosition::function(ToFn), DepClassTy::OPTIONAL);
  while (!Worklist.empty()) {
    const Instruction *CurFromI = Worklist.pop_back_val();
    if (!Visited.insert(CurFromI).second)
      continue;

    const Function *FromFn = CurFromI->getFunction();
    if (FromFn == &ToFn) {
      if (!ToI)
        return true;
      LLVM_DEBUG(dbgs() << "[AA] check " << *ToI << " from " << *CurFromI
                        << " intraprocedurally\n");
      const auto &ReachabilityAA = A.getAAFor<AAReachability>(
          QueryingAA, IRPosition::function(ToFn), DepClassTy::OPTIONAL);
      bool Result = ReachabilityAA.isAssumedReachable(A, *CurFromI, *ToI);
      LLVM_DEBUG(dbgs() << "[AA] " << *CurFromI << " "
                        << (Result ? "can potentially " : "cannot ") << "reach "
                        << *ToI << " [Intra]\n");
      if (Result)
        return true;
      if (NoRecurseAA.isAssumedNoRecurse())
        continue;
    }

    // TODO: If we can go arbitrarily backwards we will eventually reach an
    // entry point that can reach ToI. Only once this takes a set of blocks
    // through which we cannot go, or once we track internal functions not
    // accessible from the outside, it makes sense to perform backwards analysis
    // in the absence of a GoBackwardsCB.
    if (!GoBackwardsCB) {
      LLVM_DEBUG(dbgs() << "[AA] check @" << ToFn.getName() << " from "
                        << *CurFromI << " is not checked backwards, abort\n");
      return true;
    }

    // Check if the current instruction is already known to reach the ToFn.
    const auto &FnReachabilityAA = A.getAAFor<AAFunctionReachability>(
        QueryingAA, IRPosition::function(*FromFn), DepClassTy::OPTIONAL);
    bool Result = FnReachabilityAA.instructionCanReach(
        A, *CurFromI, ToFn, /* UseBackwards */ false);
    LLVM_DEBUG(dbgs() << "[AA] " << *CurFromI << " in @" << FromFn->getName()
                      << " " << (Result ? "can potentially " : "cannot ")
                      << "reach @" << ToFn.getName() << " [FromFn]\n");
    if (Result)
      return true;

    // If we do not go backwards from the FromFn we are done here and so far we
    // could not find a way to reach ToFn/ToI.
    if (!GoBackwardsCB(*FromFn))
      continue;

    LLVM_DEBUG(dbgs() << "Stepping backwards to the call sites of @"
                      << FromFn->getName() << "\n");

    auto CheckCallSite = [&](AbstractCallSite ACS) {
      CallBase *CB = ACS.getInstruction();
      if (!CB)
        return false;

      if (isa<InvokeInst>(CB))
        return false;

      Instruction *Inst = CB->getNextNonDebugInstruction();
      Worklist.push_back(Inst);
      return true;
    };

    bool UsedAssumedInformation = false;
    Result = !A.checkForAllCallSites(CheckCallSite, *FromFn,
                                     /* RequireAllCallSites */ true,
                                     &QueryingAA, UsedAssumedInformation);
    if (Result) {
      LLVM_DEBUG(dbgs() << "[AA] stepping back to call sites from " << *CurFromI
                        << " in @" << FromFn->getName()
                        << " failed, give up\n");
      return true;
    }

    LLVM_DEBUG(dbgs() << "[AA] stepped back to call sites from " << *CurFromI
                      << " in @" << FromFn->getName()
                      << " worklist size is: " << Worklist.size() << "\n");
  }
  return false;
}

bool AA::isPotentiallyReachable(
    Attributor &A, const Instruction &FromI, const Instruction &ToI,
    const AbstractAttribute &QueryingAA,
    std::function<bool(const Function &F)> GoBackwardsCB) {
  LLVM_DEBUG(dbgs() << "[AA] isPotentiallyReachable " << ToI << " from "
                    << FromI << " [GBCB: " << bool(GoBackwardsCB) << "]\n");
  const Function *ToFn = ToI.getFunction();
  return ::isPotentiallyReachable(A, FromI, &ToI, *ToFn, QueryingAA,
                                  GoBackwardsCB);
}

bool AA::isPotentiallyReachable(
    Attributor &A, const Instruction &FromI, const Function &ToFn,
    const AbstractAttribute &QueryingAA,
    std::function<bool(const Function &F)> GoBackwardsCB) {
  return ::isPotentiallyReachable(A, FromI, /* ToI */ nullptr, ToFn, QueryingAA,
                                  GoBackwardsCB);
}

/// Return true if \p New is equal or worse than \p Old.
static bool isEqualOrWorse(const Attribute &New, const Attribute &Old) {
  if (!Old.isIntAttribute())
    return true;

  return Old.getValueAsInt() >= New.getValueAsInt();
}

/// Return true if the information provided by \p Attr was added to the
/// attribute list \p Attrs. This is only the case if it was not already present
/// in \p Attrs at the position describe by \p PK and \p AttrIdx.
static bool addIfNotExistent(LLVMContext &Ctx, const Attribute &Attr,
                             AttributeList &Attrs, int AttrIdx,
                             bool ForceReplace = false) {

  if (Attr.isEnumAttribute()) {
    Attribute::AttrKind Kind = Attr.getKindAsEnum();
    if (Attrs.hasAttributeAtIndex(AttrIdx, Kind))
      if (!ForceReplace &&
          isEqualOrWorse(Attr, Attrs.getAttributeAtIndex(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.addAttributeAtIndex(Ctx, AttrIdx, Attr);
    return true;
  }
  if (Attr.isStringAttribute()) {
    StringRef Kind = Attr.getKindAsString();
    if (Attrs.hasAttributeAtIndex(AttrIdx, Kind))
      if (!ForceReplace &&
          isEqualOrWorse(Attr, Attrs.getAttributeAtIndex(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.addAttributeAtIndex(Ctx, AttrIdx, Attr);
    return true;
  }
  if (Attr.isIntAttribute()) {
    Attribute::AttrKind Kind = Attr.getKindAsEnum();
    if (Attrs.hasAttributeAtIndex(AttrIdx, Kind))
      if (!ForceReplace &&
          isEqualOrWorse(Attr, Attrs.getAttributeAtIndex(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.removeAttributeAtIndex(Ctx, AttrIdx, Kind);
    Attrs = Attrs.addAttributeAtIndex(Ctx, AttrIdx, Attr);
    return true;
  }

  llvm_unreachable("Expected enum or string attribute!");
}

Argument *IRPosition::getAssociatedArgument() const {
  if (getPositionKind() == IRP_ARGUMENT)
    return cast<Argument>(&getAnchorValue());

  // Not an Argument and no argument number means this is not a call site
  // argument, thus we cannot find a callback argument to return.
  int ArgNo = getCallSiteArgNo();
  if (ArgNo < 0)
    return nullptr;

  // Use abstract call sites to make the connection between the call site
  // values and the ones in callbacks. If a callback was found that makes use
  // of the underlying call site operand, we want the corresponding callback
  // callee argument and not the direct callee argument.
  Optional<Argument *> CBCandidateArg;
  SmallVector<const Use *, 4> CallbackUses;
  const auto &CB = cast<CallBase>(getAnchorValue());
  AbstractCallSite::getCallbackUses(CB, CallbackUses);
  for (const Use *U : CallbackUses) {
    AbstractCallSite ACS(U);
    assert(ACS && ACS.isCallbackCall());
    if (!ACS.getCalledFunction())
      continue;

    for (unsigned u = 0, e = ACS.getNumArgOperands(); u < e; u++) {

      // Test if the underlying call site operand is argument number u of the
      // callback callee.
      if (ACS.getCallArgOperandNo(u) != ArgNo)
        continue;

      assert(ACS.getCalledFunction()->arg_size() > u &&
             "ACS mapped into var-args arguments!");
      if (CBCandidateArg) {
        CBCandidateArg = nullptr;
        break;
      }
      CBCandidateArg = ACS.getCalledFunction()->getArg(u);
    }
  }

  // If we found a unique callback candidate argument, return it.
  if (CBCandidateArg.hasValue() && CBCandidateArg.getValue())
    return CBCandidateArg.getValue();

  // If no callbacks were found, or none used the underlying call site operand
  // exclusively, use the direct callee argument if available.
  const Function *Callee = CB.getCalledFunction();
  if (Callee && Callee->arg_size() > unsigned(ArgNo))
    return Callee->getArg(ArgNo);

  return nullptr;
}

ChangeStatus AbstractAttribute::update(Attributor &A) {
  ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
  if (getState().isAtFixpoint())
    return HasChanged;

  LLVM_DEBUG(dbgs() << "[Attributor] Update: " << *this << "\n");

  HasChanged = updateImpl(A);

  LLVM_DEBUG(dbgs() << "[Attributor] Update " << HasChanged << " " << *this
                    << "\n");

  return HasChanged;
}

ChangeStatus
IRAttributeManifest::manifestAttrs(Attributor &A, const IRPosition &IRP,
                                   const ArrayRef<Attribute> &DeducedAttrs,
                                   bool ForceReplace) {
  Function *ScopeFn = IRP.getAnchorScope();
  IRPosition::Kind PK = IRP.getPositionKind();

  // In the following some generic code that will manifest attributes in
  // DeducedAttrs if they improve the current IR. Due to the different
  // annotation positions we use the underlying AttributeList interface.

  AttributeList Attrs;
  switch (PK) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
    return ChangeStatus::UNCHANGED;
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_FUNCTION:
  case IRPosition::IRP_RETURNED:
    Attrs = ScopeFn->getAttributes();
    break;
  case IRPosition::IRP_CALL_SITE:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
    Attrs = cast<CallBase>(IRP.getAnchorValue()).getAttributes();
    break;
  }

  ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
  LLVMContext &Ctx = IRP.getAnchorValue().getContext();
  for (const Attribute &Attr : DeducedAttrs) {
    if (!addIfNotExistent(Ctx, Attr, Attrs, IRP.getAttrIdx(), ForceReplace))
      continue;

    HasChanged = ChangeStatus::CHANGED;
  }

  if (HasChanged == ChangeStatus::UNCHANGED)
    return HasChanged;

  switch (PK) {
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_FUNCTION:
  case IRPosition::IRP_RETURNED:
    ScopeFn->setAttributes(Attrs);
    break;
  case IRPosition::IRP_CALL_SITE:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
    cast<CallBase>(IRP.getAnchorValue()).setAttributes(Attrs);
    break;
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
    break;
  }

  return HasChanged;
}

const IRPosition IRPosition::EmptyKey(DenseMapInfo<void *>::getEmptyKey());
const IRPosition
    IRPosition::TombstoneKey(DenseMapInfo<void *>::getTombstoneKey());

SubsumingPositionIterator::SubsumingPositionIterator(const IRPosition &IRP) {
  IRPositions.emplace_back(IRP);

  // Helper to determine if operand bundles on a call site are benin or
  // potentially problematic. We handle only llvm.assume for now.
  auto CanIgnoreOperandBundles = [](const CallBase &CB) {
    return (isa<IntrinsicInst>(CB) &&
            cast<IntrinsicInst>(CB).getIntrinsicID() == Intrinsic ::assume);
  };

  const auto *CB = dyn_cast<CallBase>(&IRP.getAnchorValue());
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_FUNCTION:
    return;
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_RETURNED:
    IRPositions.emplace_back(IRPosition::function(*IRP.getAnchorScope()));
    return;
  case IRPosition::IRP_CALL_SITE:
    assert(CB && "Expected call site!");
    // TODO: We need to look at the operand bundles similar to the redirection
    //       in CallBase.
    if (!CB->hasOperandBundles() || CanIgnoreOperandBundles(*CB))
      if (const Function *Callee = CB->getCalledFunction())
        IRPositions.emplace_back(IRPosition::function(*Callee));
    return;
  case IRPosition::IRP_CALL_SITE_RETURNED:
    assert(CB && "Expected call site!");
    // TODO: We need to look at the operand bundles similar to the redirection
    //       in CallBase.
    if (!CB->hasOperandBundles() || CanIgnoreOperandBundles(*CB)) {
      if (const Function *Callee = CB->getCalledFunction()) {
        IRPositions.emplace_back(IRPosition::returned(*Callee));
        IRPositions.emplace_back(IRPosition::function(*Callee));
        for (const Argument &Arg : Callee->args())
          if (Arg.hasReturnedAttr()) {
            IRPositions.emplace_back(
                IRPosition::callsite_argument(*CB, Arg.getArgNo()));
            IRPositions.emplace_back(
                IRPosition::value(*CB->getArgOperand(Arg.getArgNo())));
            IRPositions.emplace_back(IRPosition::argument(Arg));
          }
      }
    }
    IRPositions.emplace_back(IRPosition::callsite_function(*CB));
    return;
  case IRPosition::IRP_CALL_SITE_ARGUMENT: {
    assert(CB && "Expected call site!");
    // TODO: We need to look at the operand bundles similar to the redirection
    //       in CallBase.
    if (!CB->hasOperandBundles() || CanIgnoreOperandBundles(*CB)) {
      const Function *Callee = CB->getCalledFunction();
      if (Callee) {
        if (Argument *Arg = IRP.getAssociatedArgument())
          IRPositions.emplace_back(IRPosition::argument(*Arg));
        IRPositions.emplace_back(IRPosition::function(*Callee));
      }
    }
    IRPositions.emplace_back(IRPosition::value(IRP.getAssociatedValue()));
    return;
  }
  }
}

bool IRPosition::hasAttr(ArrayRef<Attribute::AttrKind> AKs,
                         bool IgnoreSubsumingPositions, Attributor *A) const {
  SmallVector<Attribute, 4> Attrs;
  for (const IRPosition &EquivIRP : SubsumingPositionIterator(*this)) {
    for (Attribute::AttrKind AK : AKs)
      if (EquivIRP.getAttrsFromIRAttr(AK, Attrs))
        return true;
    // The first position returned by the SubsumingPositionIterator is
    // always the position itself. If we ignore subsuming positions we
    // are done after the first iteration.
    if (IgnoreSubsumingPositions)
      break;
  }
  if (A)
    for (Attribute::AttrKind AK : AKs)
      if (getAttrsFromAssumes(AK, Attrs, *A))
        return true;
  return false;
}

void IRPosition::getAttrs(ArrayRef<Attribute::AttrKind> AKs,
                          SmallVectorImpl<Attribute> &Attrs,
                          bool IgnoreSubsumingPositions, Attributor *A) const {
  for (const IRPosition &EquivIRP : SubsumingPositionIterator(*this)) {
    for (Attribute::AttrKind AK : AKs)
      EquivIRP.getAttrsFromIRAttr(AK, Attrs);
    // The first position returned by the SubsumingPositionIterator is
    // always the position itself. If we ignore subsuming positions we
    // are done after the first iteration.
    if (IgnoreSubsumingPositions)
      break;
  }
  if (A)
    for (Attribute::AttrKind AK : AKs)
      getAttrsFromAssumes(AK, Attrs, *A);
}

bool IRPosition::getAttrsFromIRAttr(Attribute::AttrKind AK,
                                    SmallVectorImpl<Attribute> &Attrs) const {
  if (getPositionKind() == IRP_INVALID || getPositionKind() == IRP_FLOAT)
    return false;

  AttributeList AttrList;
  if (const auto *CB = dyn_cast<CallBase>(&getAnchorValue()))
    AttrList = CB->getAttributes();
  else
    AttrList = getAssociatedFunction()->getAttributes();

  bool HasAttr = AttrList.hasAttributeAtIndex(getAttrIdx(), AK);
  if (HasAttr)
    Attrs.push_back(AttrList.getAttributeAtIndex(getAttrIdx(), AK));
  return HasAttr;
}

bool IRPosition::getAttrsFromAssumes(Attribute::AttrKind AK,
                                     SmallVectorImpl<Attribute> &Attrs,
                                     Attributor &A) const {
  assert(getPositionKind() != IRP_INVALID && "Did expect a valid position!");
  Value &AssociatedValue = getAssociatedValue();

  const Assume2KnowledgeMap &A2K =
      A.getInfoCache().getKnowledgeMap().lookup({&AssociatedValue, AK});

  // Check if we found any potential assume use, if not we don't need to create
  // explorer iterators.
  if (A2K.empty())
    return false;

  LLVMContext &Ctx = AssociatedValue.getContext();
  unsigned AttrsSize = Attrs.size();
  MustBeExecutedContextExplorer &Explorer =
      A.getInfoCache().getMustBeExecutedContextExplorer();
  auto EIt = Explorer.begin(getCtxI()), EEnd = Explorer.end(getCtxI());
  for (auto &It : A2K)
    if (Explorer.findInContextOf(It.first, EIt, EEnd))
      Attrs.push_back(Attribute::get(Ctx, AK, It.second.Max));
  return AttrsSize != Attrs.size();
}

void IRPosition::verify() {
#ifdef EXPENSIVE_CHECKS
  switch (getPositionKind()) {
  case IRP_INVALID:
    assert((CBContext == nullptr) &&
           "Invalid position must not have CallBaseContext!");
    assert(!Enc.getOpaqueValue() &&
           "Expected a nullptr for an invalid position!");
    return;
  case IRP_FLOAT:
    assert((!isa<Argument>(&getAssociatedValue())) &&
           "Expected specialized kind for argument values!");
    return;
  case IRP_RETURNED:
    assert(isa<Function>(getAsValuePtr()) &&
           "Expected function for a 'returned' position!");
    assert(getAsValuePtr() == &getAssociatedValue() &&
           "Associated value mismatch!");
    return;
  case IRP_CALL_SITE_RETURNED:
    assert((CBContext == nullptr) &&
           "'call site returned' position must not have CallBaseContext!");
    assert((isa<CallBase>(getAsValuePtr())) &&
           "Expected call base for 'call site returned' position!");
    assert(getAsValuePtr() == &getAssociatedValue() &&
           "Associated value mismatch!");
    return;
  case IRP_CALL_SITE:
    assert((CBContext == nullptr) &&
           "'call site function' position must not have CallBaseContext!");
    assert((isa<CallBase>(getAsValuePtr())) &&
           "Expected call base for 'call site function' position!");
    assert(getAsValuePtr() == &getAssociatedValue() &&
           "Associated value mismatch!");
    return;
  case IRP_FUNCTION:
    assert(isa<Function>(getAsValuePtr()) &&
           "Expected function for a 'function' position!");
    assert(getAsValuePtr() == &getAssociatedValue() &&
           "Associated value mismatch!");
    return;
  case IRP_ARGUMENT:
    assert(isa<Argument>(getAsValuePtr()) &&
           "Expected argument for a 'argument' position!");
    assert(getAsValuePtr() == &getAssociatedValue() &&
           "Associated value mismatch!");
    return;
  case IRP_CALL_SITE_ARGUMENT: {
    assert((CBContext == nullptr) &&
           "'call site argument' position must not have CallBaseContext!");
    Use *U = getAsUsePtr();
    (void)U; // Silence unused variable warning.
    assert(U && "Expected use for a 'call site argument' position!");
    assert(isa<CallBase>(U->getUser()) &&
           "Expected call base user for a 'call site argument' position!");
    assert(cast<CallBase>(U->getUser())->isArgOperand(U) &&
           "Expected call base argument operand for a 'call site argument' "
           "position");
    assert(cast<CallBase>(U->getUser())->getArgOperandNo(U) ==
               unsigned(getCallSiteArgNo()) &&
           "Argument number mismatch!");
    assert(U->get() == &getAssociatedValue() && "Associated value mismatch!");
    return;
  }
  }
#endif
}

Optional<Constant *>
Attributor::getAssumedConstant(const IRPosition &IRP,
                               const AbstractAttribute &AA,
                               bool &UsedAssumedInformation) {
  // First check all callbacks provided by outside AAs. If any of them returns
  // a non-null value that is different from the associated value, or None, we
  // assume it's simplified.
  for (auto &CB : SimplificationCallbacks.lookup(IRP)) {
    Optional<Value *> SimplifiedV = CB(IRP, &AA, UsedAssumedInformation);
    if (!SimplifiedV)
      return llvm::None;
    if (isa_and_nonnull<Constant>(*SimplifiedV))
      return cast<Constant>(*SimplifiedV);
    return nullptr;
  }
  if (auto *C = dyn_cast<Constant>(&IRP.getAssociatedValue()))
    return C;
  const auto &ValueSimplifyAA =
      getAAFor<AAValueSimplify>(AA, IRP, DepClassTy::NONE);
  Optional<Value *> SimplifiedV =
      ValueSimplifyAA.getAssumedSimplifiedValue(*this);
  bool IsKnown = ValueSimplifyAA.isAtFixpoint();
  UsedAssumedInformation |= !IsKnown;
  if (!SimplifiedV) {
    recordDependence(ValueSimplifyAA, AA, DepClassTy::OPTIONAL);
    return llvm::None;
  }
  if (isa_and_nonnull<UndefValue>(SimplifiedV.getValue())) {
    recordDependence(ValueSimplifyAA, AA, DepClassTy::OPTIONAL);
    return UndefValue::get(IRP.getAssociatedType());
  }
  Constant *CI = dyn_cast_or_null<Constant>(SimplifiedV.getValue());
  if (CI)
    CI = dyn_cast_or_null<Constant>(
        AA::getWithType(*CI, *IRP.getAssociatedType()));
  if (CI)
    recordDependence(ValueSimplifyAA, AA, DepClassTy::OPTIONAL);
  return CI;
}

Optional<Value *>
Attributor::getAssumedSimplified(const IRPosition &IRP,
                                 const AbstractAttribute *AA,
                                 bool &UsedAssumedInformation) {
  // First check all callbacks provided by outside AAs. If any of them returns
  // a non-null value that is different from the associated value, or None, we
  // assume it's simplified.
  for (auto &CB : SimplificationCallbacks.lookup(IRP))
    return CB(IRP, AA, UsedAssumedInformation);

  // If no high-level/outside simplification occurred, use AAValueSimplify.
  const auto &ValueSimplifyAA =
      getOrCreateAAFor<AAValueSimplify>(IRP, AA, DepClassTy::NONE);
  Optional<Value *> SimplifiedV =
      ValueSimplifyAA.getAssumedSimplifiedValue(*this);
  bool IsKnown = ValueSimplifyAA.isAtFixpoint();
  UsedAssumedInformation |= !IsKnown;
  if (!SimplifiedV) {
    if (AA)
      recordDependence(ValueSimplifyAA, *AA, DepClassTy::OPTIONAL);
    return llvm::None;
  }
  if (*SimplifiedV == nullptr)
    return const_cast<Value *>(&IRP.getAssociatedValue());
  if (Value *SimpleV =
          AA::getWithType(**SimplifiedV, *IRP.getAssociatedType())) {
    if (AA)
      recordDependence(ValueSimplifyAA, *AA, DepClassTy::OPTIONAL);
    return SimpleV;
  }
  return const_cast<Value *>(&IRP.getAssociatedValue());
}

Optional<Value *> Attributor::translateArgumentToCallSiteContent(
    Optional<Value *> V, CallBase &CB, const AbstractAttribute &AA,
    bool &UsedAssumedInformation) {
  if (!V)
    return V;
  if (*V == nullptr || isa<Constant>(*V))
    return V;
  if (auto *Arg = dyn_cast<Argument>(*V))
    if (CB.getCalledFunction() == Arg->getParent())
      if (!Arg->hasPointeeInMemoryValueAttr())
        return getAssumedSimplified(
            IRPosition::callsite_argument(CB, Arg->getArgNo()), AA,
            UsedAssumedInformation);
  return nullptr;
}

Attributor::~Attributor() {
  // The abstract attributes are allocated via the BumpPtrAllocator Allocator,
  // thus we cannot delete them. We can, and want to, destruct them though.
  for (auto &DepAA : DG.SyntheticRoot.Deps) {
    AbstractAttribute *AA = cast<AbstractAttribute>(DepAA.getPointer());
    AA->~AbstractAttribute();
  }
}

bool Attributor::isAssumedDead(const AbstractAttribute &AA,
                               const AAIsDead *FnLivenessAA,
                               bool &UsedAssumedInformation,
                               bool CheckBBLivenessOnly, DepClassTy DepClass) {
  const IRPosition &IRP = AA.getIRPosition();
  if (!Functions.count(IRP.getAnchorScope()))
    return false;
  return isAssumedDead(IRP, &AA, FnLivenessAA, UsedAssumedInformation,
                       CheckBBLivenessOnly, DepClass);
}

bool Attributor::isAssumedDead(const Use &U,
                               const AbstractAttribute *QueryingAA,
                               const AAIsDead *FnLivenessAA,
                               bool &UsedAssumedInformation,
                               bool CheckBBLivenessOnly, DepClassTy DepClass) {
  Instruction *UserI = dyn_cast<Instruction>(U.getUser());
  if (!UserI)
    return isAssumedDead(IRPosition::value(*U.get()), QueryingAA, FnLivenessAA,
                         UsedAssumedInformation, CheckBBLivenessOnly, DepClass);

  if (auto *CB = dyn_cast<CallBase>(UserI)) {
    // For call site argument uses we can check if the argument is
    // unused/dead.
    if (CB->isArgOperand(&U)) {
      const IRPosition &CSArgPos =
          IRPosition::callsite_argument(*CB, CB->getArgOperandNo(&U));
      return isAssumedDead(CSArgPos, QueryingAA, FnLivenessAA,
                           UsedAssumedInformation, CheckBBLivenessOnly,
                           DepClass);
    }
  } else if (ReturnInst *RI = dyn_cast<ReturnInst>(UserI)) {
    const IRPosition &RetPos = IRPosition::returned(*RI->getFunction());
    return isAssumedDead(RetPos, QueryingAA, FnLivenessAA,
                         UsedAssumedInformation, CheckBBLivenessOnly, DepClass);
  } else if (PHINode *PHI = dyn_cast<PHINode>(UserI)) {
    BasicBlock *IncomingBB = PHI->getIncomingBlock(U);
    return isAssumedDead(*IncomingBB->getTerminator(), QueryingAA, FnLivenessAA,
                         UsedAssumedInformation, CheckBBLivenessOnly, DepClass);
  } else if (StoreInst *SI = dyn_cast<StoreInst>(UserI)) {
    if (!CheckBBLivenessOnly && SI->getPointerOperand() != U.get()) {
      const IRPosition IRP = IRPosition::inst(*SI);
      const AAIsDead &IsDeadAA =
          getOrCreateAAFor<AAIsDead>(IRP, QueryingAA, DepClassTy::NONE);
      if (IsDeadAA.isRemovableStore()) {
        if (QueryingAA)
          recordDependence(IsDeadAA, *QueryingAA, DepClass);
        if (!IsDeadAA.isKnown(AAIsDead::IS_REMOVABLE))
          UsedAssumedInformation = true;
        return true;
      }
    }
  }

  return isAssumedDead(IRPosition::inst(*UserI), QueryingAA, FnLivenessAA,
                       UsedAssumedInformation, CheckBBLivenessOnly, DepClass);
}

bool Attributor::isAssumedDead(const Instruction &I,
                               const AbstractAttribute *QueryingAA,
                               const AAIsDead *FnLivenessAA,
                               bool &UsedAssumedInformation,
                               bool CheckBBLivenessOnly, DepClassTy DepClass) {
  const IRPosition::CallBaseContext *CBCtx =
      QueryingAA ? QueryingAA->getCallBaseContext() : nullptr;

  if (ManifestAddedBlocks.contains(I.getParent()))
    return false;

  if (!FnLivenessAA || FnLivenessAA->getAnchorScope() != I.getFunction())
    FnLivenessAA = &getOrCreateAAFor<AAIsDead>(
        IRPosition::function(*I.getFunction(), CBCtx), QueryingAA,
        DepClassTy::NONE);

  // If we have a context instruction and a liveness AA we use it.
  if (CheckBBLivenessOnly ? FnLivenessAA->isAssumedDead(I.getParent())
                          : FnLivenessAA->isAssumedDead(&I)) {
    if (QueryingAA)
      recordDependence(*FnLivenessAA, *QueryingAA, DepClass);
    if (!FnLivenessAA->isKnownDead(&I))
      UsedAssumedInformation = true;
    return true;
  }

  if (CheckBBLivenessOnly)
    return false;

  const IRPosition IRP = IRPosition::inst(I, CBCtx);
  const AAIsDead &IsDeadAA =
      getOrCreateAAFor<AAIsDead>(IRP, QueryingAA, DepClassTy::NONE);
  // Don't check liveness for AAIsDead.
  if (QueryingAA == &IsDeadAA)
    return false;

  if (IsDeadAA.isAssumedDead()) {
    if (QueryingAA)
      recordDependence(IsDeadAA, *QueryingAA, DepClass);
    if (!IsDeadAA.isKnownDead())
      UsedAssumedInformation = true;
    return true;
  }

  return false;
}

bool Attributor::isAssumedDead(const IRPosition &IRP,
                               const AbstractAttribute *QueryingAA,
                               const AAIsDead *FnLivenessAA,
                               bool &UsedAssumedInformation,
                               bool CheckBBLivenessOnly, DepClassTy DepClass) {
  Instruction *CtxI = IRP.getCtxI();
  if (CtxI &&
      isAssumedDead(*CtxI, QueryingAA, FnLivenessAA, UsedAssumedInformation,
                    /* CheckBBLivenessOnly */ true,
                    CheckBBLivenessOnly ? DepClass : DepClassTy::OPTIONAL))
    return true;

  if (CheckBBLivenessOnly)
    return false;

  // If we haven't succeeded we query the specific liveness info for the IRP.
  const AAIsDead *IsDeadAA;
  if (IRP.getPositionKind() == IRPosition::IRP_CALL_SITE)
    IsDeadAA = &getOrCreateAAFor<AAIsDead>(
        IRPosition::callsite_returned(cast<CallBase>(IRP.getAssociatedValue())),
        QueryingAA, DepClassTy::NONE);
  else
    IsDeadAA = &getOrCreateAAFor<AAIsDead>(IRP, QueryingAA, DepClassTy::NONE);
  // Don't check liveness for AAIsDead.
  if (QueryingAA == IsDeadAA)
    return false;

  if (IsDeadAA->isAssumedDead()) {
    if (QueryingAA)
      recordDependence(*IsDeadAA, *QueryingAA, DepClass);
    if (!IsDeadAA->isKnownDead())
      UsedAssumedInformation = true;
    return true;
  }

  return false;
}

bool Attributor::isAssumedDead(const BasicBlock &BB,
                               const AbstractAttribute *QueryingAA,
                               const AAIsDead *FnLivenessAA,
                               DepClassTy DepClass) {
  if (!FnLivenessAA || FnLivenessAA->getAnchorScope() != BB.getParent())
    FnLivenessAA = &getOrCreateAAFor<AAIsDead>(
        IRPosition::function(*BB.getParent()), QueryingAA, DepClassTy::NONE);
  if (FnLivenessAA->isAssumedDead(&BB)) {
    if (QueryingAA)
      recordDependence(*FnLivenessAA, *QueryingAA, DepClass);
    return true;
  }

  return false;
}

bool Attributor::checkForAllUses(
    function_ref<bool(const Use &, bool &)> Pred,
    const AbstractAttribute &QueryingAA, const Value &V,
    bool CheckBBLivenessOnly, DepClassTy LivenessDepClass,
    bool IgnoreDroppableUses,
    function_ref<bool(const Use &OldU, const Use &NewU)> EquivalentUseCB) {

  // Check the trivial case first as it catches void values.
  if (V.use_empty())
    return true;

  const IRPosition &IRP = QueryingAA.getIRPosition();
  SmallVector<const Use *, 16> Worklist;
  SmallPtrSet<const Use *, 16> Visited;

  for (const Use &U : V.uses())
    Worklist.push_back(&U);

  LLVM_DEBUG(dbgs() << "[Attributor] Got " << Worklist.size()
                    << " initial uses to check\n");

  const Function *ScopeFn = IRP.getAnchorScope();
  const auto *LivenessAA =
      ScopeFn ? &getAAFor<AAIsDead>(QueryingAA, IRPosition::function(*ScopeFn),
                                    DepClassTy::NONE)
              : nullptr;

  while (!Worklist.empty()) {
    const Use *U = Worklist.pop_back_val();
    if (isa<PHINode>(U->getUser()) && !Visited.insert(U).second)
      continue;
    LLVM_DEBUG({
      if (auto *Fn = dyn_cast<Function>(U->getUser()))
        dbgs() << "[Attributor] Check use: " << **U << " in " << Fn->getName()
               << "\n";
      else
        dbgs() << "[Attributor] Check use: " << **U << " in " << *U->getUser()
               << "\n";
    });
    bool UsedAssumedInformation = false;
    if (isAssumedDead(*U, &QueryingAA, LivenessAA, UsedAssumedInformation,
                      CheckBBLivenessOnly, LivenessDepClass)) {
      LLVM_DEBUG(dbgs() << "[Attributor] Dead use, skip!\n");
      continue;
    }
    if (IgnoreDroppableUses && U->getUser()->isDroppable()) {
      LLVM_DEBUG(dbgs() << "[Attributor] Droppable user, skip!\n");
      continue;
    }

    if (auto *SI = dyn_cast<StoreInst>(U->getUser())) {
      if (&SI->getOperandUse(0) == U) {
        if (!Visited.insert(U).second)
          continue;
        SmallSetVector<Value *, 4> PotentialCopies;
        if (AA::getPotentialCopiesOfStoredValue(
                *this, *SI, PotentialCopies, QueryingAA, UsedAssumedInformation,
                /* OnlyExact */ true)) {
          LLVM_DEBUG(dbgs() << "[Attributor] Value is stored, continue with "
                            << PotentialCopies.size()
                            << " potential copies instead!\n");
          for (Value *PotentialCopy : PotentialCopies)
            for (const Use &CopyUse : PotentialCopy->uses()) {
              if (EquivalentUseCB && !EquivalentUseCB(*U, CopyUse)) {
                LLVM_DEBUG(dbgs() << "[Attributor] Potential copy was "
                                     "rejected by the equivalence call back: "
                                  << *CopyUse << "!\n");
                return false;
              }
              Worklist.push_back(&CopyUse);
            }
          continue;
        }
      }
    }

    bool Follow = false;
    if (!Pred(*U, Follow))
      return false;
    if (!Follow)
      continue;
    for (const Use &UU : U->getUser()->uses())
      Worklist.push_back(&UU);
  }

  return true;
}

bool Attributor::checkForAllCallSites(function_ref<bool(AbstractCallSite)> Pred,
                                      const AbstractAttribute &QueryingAA,
                                      bool RequireAllCallSites,
                                      bool &UsedAssumedInformation) {
  // We can try to determine information from
  // the call sites. However, this is only possible all call sites are known,
  // hence the function has internal linkage.
  const IRPosition &IRP = QueryingAA.getIRPosition();
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction) {
    LLVM_DEBUG(dbgs() << "[Attributor] No function associated with " << IRP
                      << "\n");
    return false;
  }

  return checkForAllCallSites(Pred, *AssociatedFunction, RequireAllCallSites,
                              &QueryingAA, UsedAssumedInformation);
}

bool Attributor::checkForAllCallSites(function_ref<bool(AbstractCallSite)> Pred,
                                      const Function &Fn,
                                      bool RequireAllCallSites,
                                      const AbstractAttribute *QueryingAA,
                                      bool &UsedAssumedInformation) {
  if (RequireAllCallSites && !Fn.hasLocalLinkage()) {
    LLVM_DEBUG(
        dbgs()
        << "[Attributor] Function " << Fn.getName()
        << " has no internal linkage, hence not all call sites are known\n");
    return false;
  }

  SmallVector<const Use *, 8> Uses(make_pointer_range(Fn.uses()));
  for (unsigned u = 0; u < Uses.size(); ++u) {
    const Use &U = *Uses[u];
    LLVM_DEBUG({
      if (auto *Fn = dyn_cast<Function>(U))
        dbgs() << "[Attributor] Check use: " << Fn->getName() << " in "
               << *U.getUser() << "\n";
      else
        dbgs() << "[Attributor] Check use: " << *U << " in " << *U.getUser()
               << "\n";
    });
    if (isAssumedDead(U, QueryingAA, nullptr, UsedAssumedInformation,
                      /* CheckBBLivenessOnly */ true)) {
      LLVM_DEBUG(dbgs() << "[Attributor] Dead use, skip!\n");
      continue;
    }
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(U.getUser())) {
      if (CE->isCast() && CE->getType()->isPointerTy()) {
        LLVM_DEBUG(
            dbgs() << "[Attributor] Use, is constant cast expression, add "
                   << CE->getNumUses()
                   << " uses of that expression instead!\n");
        for (const Use &CEU : CE->uses())
          Uses.push_back(&CEU);
        continue;
      }
    }

    AbstractCallSite ACS(&U);
    if (!ACS) {
      LLVM_DEBUG(dbgs() << "[Attributor] Function " << Fn.getName()
                        << " has non call site use " << *U.get() << " in "
                        << *U.getUser() << "\n");
      // BlockAddress users are allowed.
      if (isa<BlockAddress>(U.getUser()))
        continue;
      return false;
    }

    const Use *EffectiveUse =
        ACS.isCallbackCall() ? &ACS.getCalleeUseForCallback() : &U;
    if (!ACS.isCallee(EffectiveUse)) {
      if (!RequireAllCallSites) {
        LLVM_DEBUG(dbgs() << "[Attributor] User " << *EffectiveUse->getUser()
                          << " is not a call of " << Fn.getName()
                          << ", skip use\n");
        continue;
      }
      LLVM_DEBUG(dbgs() << "[Attributor] User " << *EffectiveUse->getUser()
                        << " is an invalid use of " << Fn.getName() << "\n");
      return false;
    }

    // Make sure the arguments that can be matched between the call site and the
    // callee argee on their type. It is unlikely they do not and it doesn't
    // make sense for all attributes to know/care about this.
    assert(&Fn == ACS.getCalledFunction() && "Expected known callee");
    unsigned MinArgsParams =
        std::min(size_t(ACS.getNumArgOperands()), Fn.arg_size());
    for (unsigned u = 0; u < MinArgsParams; ++u) {
      Value *CSArgOp = ACS.getCallArgOperand(u);
      if (CSArgOp && Fn.getArg(u)->getType() != CSArgOp->getType()) {
        LLVM_DEBUG(
            dbgs() << "[Attributor] Call site / callee argument type mismatch ["
                   << u << "@" << Fn.getName() << ": "
                   << *Fn.getArg(u)->getType() << " vs. "
                   << *ACS.getCallArgOperand(u)->getType() << "\n");
        return false;
      }
    }

    if (Pred(ACS))
      continue;

    LLVM_DEBUG(dbgs() << "[Attributor] Call site callback failed for "
                      << *ACS.getInstruction() << "\n");
    return false;
  }

  return true;
}

bool Attributor::shouldPropagateCallBaseContext(const IRPosition &IRP) {
  // TODO: Maintain a cache of Values that are
  // on the pathway from a Argument to a Instruction that would effect the
  // liveness/return state etc.
  return EnableCallSiteSpecific;
}

bool Attributor::checkForAllReturnedValuesAndReturnInsts(
    function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)> Pred,
    const AbstractAttribute &QueryingAA) {

  const IRPosition &IRP = QueryingAA.getIRPosition();
  // Since we need to provide return instructions we have to have an exact
  // definition.
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  // If this is a call site query we use the call site specific return values
  // and liveness information.
  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(*AssociatedFunction);
  const auto &AARetVal =
      getAAFor<AAReturnedValues>(QueryingAA, QueryIRP, DepClassTy::REQUIRED);
  if (!AARetVal.getState().isValidState())
    return false;

  return AARetVal.checkForAllReturnedValuesAndReturnInsts(Pred);
}

bool Attributor::checkForAllReturnedValues(
    function_ref<bool(Value &)> Pred, const AbstractAttribute &QueryingAA) {

  const IRPosition &IRP = QueryingAA.getIRPosition();
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(
      *AssociatedFunction, QueryingAA.getCallBaseContext());
  const auto &AARetVal =
      getAAFor<AAReturnedValues>(QueryingAA, QueryIRP, DepClassTy::REQUIRED);
  if (!AARetVal.getState().isValidState())
    return false;

  return AARetVal.checkForAllReturnedValuesAndReturnInsts(
      [&](Value &RV, const SmallSetVector<ReturnInst *, 4> &) {
        return Pred(RV);
      });
}

static bool checkForAllInstructionsImpl(
    Attributor *A, InformationCache::OpcodeInstMapTy &OpcodeInstMap,
    function_ref<bool(Instruction &)> Pred, const AbstractAttribute *QueryingAA,
    const AAIsDead *LivenessAA, const ArrayRef<unsigned> &Opcodes,
    bool &UsedAssumedInformation, bool CheckBBLivenessOnly = false,
    bool CheckPotentiallyDead = false) {
  for (unsigned Opcode : Opcodes) {
    // Check if we have instructions with this opcode at all first.
    auto *Insts = OpcodeInstMap.lookup(Opcode);
    if (!Insts)
      continue;

    for (Instruction *I : *Insts) {
      // Skip dead instructions.
      if (A && !CheckPotentiallyDead &&
          A->isAssumedDead(IRPosition::inst(*I), QueryingAA, LivenessAA,
                           UsedAssumedInformation, CheckBBLivenessOnly)) {
        LLVM_DEBUG(dbgs() << "[Attributor] Instruction " << *I
                          << " is potentially dead, skip!\n";);
        continue;
      }

      if (!Pred(*I))
        return false;
    }
  }
  return true;
}

bool Attributor::checkForAllInstructions(function_ref<bool(Instruction &)> Pred,
                                         const Function *Fn,
                                         const AbstractAttribute &QueryingAA,
                                         const ArrayRef<unsigned> &Opcodes,
                                         bool &UsedAssumedInformation,
                                         bool CheckBBLivenessOnly,
                                         bool CheckPotentiallyDead) {
  // Since we need to provide instructions we have to have an exact definition.
  if (!Fn || Fn->isDeclaration())
    return false;

  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(*Fn);
  const auto *LivenessAA =
      (CheckBBLivenessOnly || CheckPotentiallyDead)
          ? nullptr
          : &(getAAFor<AAIsDead>(QueryingAA, QueryIRP, DepClassTy::NONE));

  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(*Fn);
  if (!checkForAllInstructionsImpl(this, OpcodeInstMap, Pred, &QueryingAA,
                                   LivenessAA, Opcodes, UsedAssumedInformation,
                                   CheckBBLivenessOnly, CheckPotentiallyDead))
    return false;

  return true;
}

bool Attributor::checkForAllInstructions(function_ref<bool(Instruction &)> Pred,
                                         const AbstractAttribute &QueryingAA,
                                         const ArrayRef<unsigned> &Opcodes,
                                         bool &UsedAssumedInformation,
                                         bool CheckBBLivenessOnly,
                                         bool CheckPotentiallyDead) {
  const IRPosition &IRP = QueryingAA.getIRPosition();
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  return checkForAllInstructions(Pred, AssociatedFunction, QueryingAA, Opcodes,
                                 UsedAssumedInformation, CheckBBLivenessOnly,
                                 CheckPotentiallyDead);
}

bool Attributor::checkForAllReadWriteInstructions(
    function_ref<bool(Instruction &)> Pred, AbstractAttribute &QueryingAA,
    bool &UsedAssumedInformation) {

  const Function *AssociatedFunction =
      QueryingAA.getIRPosition().getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(*AssociatedFunction);
  const auto &LivenessAA =
      getAAFor<AAIsDead>(QueryingAA, QueryIRP, DepClassTy::NONE);

  for (Instruction *I :
       InfoCache.getReadOrWriteInstsForFunction(*AssociatedFunction)) {
    // Skip dead instructions.
    if (isAssumedDead(IRPosition::inst(*I), &QueryingAA, &LivenessAA,
                      UsedAssumedInformation))
      continue;

    if (!Pred(*I))
      return false;
  }

  return true;
}

void Attributor::runTillFixpoint() {
  TimeTraceScope TimeScope("Attributor::runTillFixpoint");
  LLVM_DEBUG(dbgs() << "[Attributor] Identified and initialized "
                    << DG.SyntheticRoot.Deps.size()
                    << " abstract attributes.\n");

  // Now that all abstract attributes are collected and initialized we start
  // the abstract analysis.

  unsigned IterationCounter = 1;
  unsigned MaxIterations =
      Configuration.MaxFixpointIterations.value_or(SetFixpointIterations);

  SmallVector<AbstractAttribute *, 32> ChangedAAs;
  SetVector<AbstractAttribute *> Worklist, InvalidAAs;
  Worklist.insert(DG.SyntheticRoot.begin(), DG.SyntheticRoot.end());

  do {
    // Remember the size to determine new attributes.
    size_t NumAAs = DG.SyntheticRoot.Deps.size();
    LLVM_DEBUG(dbgs() << "\n\n[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist size: " << Worklist.size() << "\n");

    // For invalid AAs we can fix dependent AAs that have a required dependence,
    // thereby folding long dependence chains in a single step without the need
    // to run updates.
    for (unsigned u = 0; u < InvalidAAs.size(); ++u) {
      AbstractAttribute *InvalidAA = InvalidAAs[u];

      // Check the dependences to fast track invalidation.
      LLVM_DEBUG(dbgs() << "[Attributor] InvalidAA: " << *InvalidAA << " has "
                        << InvalidAA->Deps.size()
                        << " required & optional dependences\n");
      while (!InvalidAA->Deps.empty()) {
        const auto &Dep = InvalidAA->Deps.back();
        InvalidAA->Deps.pop_back();
        AbstractAttribute *DepAA = cast<AbstractAttribute>(Dep.getPointer());
        if (Dep.getInt() == unsigned(DepClassTy::OPTIONAL)) {
          LLVM_DEBUG(dbgs() << " - recompute: " << *DepAA);
          Worklist.insert(DepAA);
          continue;
        }
        LLVM_DEBUG(dbgs() << " - invalidate: " << *DepAA);
        DepAA->getState().indicatePessimisticFixpoint();
        assert(DepAA->getState().isAtFixpoint() && "Expected fixpoint state!");
        if (!DepAA->getState().isValidState())
          InvalidAAs.insert(DepAA);
        else
          ChangedAAs.push_back(DepAA);
      }
    }

    // Add all abstract attributes that are potentially dependent on one that
    // changed to the work list.
    for (AbstractAttribute *ChangedAA : ChangedAAs)
      while (!ChangedAA->Deps.empty()) {
        Worklist.insert(
            cast<AbstractAttribute>(ChangedAA->Deps.back().getPointer()));
        ChangedAA->Deps.pop_back();
      }

    LLVM_DEBUG(dbgs() << "[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist+Dependent size: " << Worklist.size()
                      << "\n");

    // Reset the changed and invalid set.
    ChangedAAs.clear();
    InvalidAAs.clear();

    // Update all abstract attribute in the work list and record the ones that
    // changed.
    for (AbstractAttribute *AA : Worklist) {
      const auto &AAState = AA->getState();
      if (!AAState.isAtFixpoint())
        if (updateAA(*AA) == ChangeStatus::CHANGED)
          ChangedAAs.push_back(AA);

      // Use the InvalidAAs vector to propagate invalid states fast transitively
      // without requiring updates.
      if (!AAState.isValidState())
        InvalidAAs.insert(AA);
    }

    // Add attributes to the changed set if they have been created in the last
    // iteration.
    ChangedAAs.append(DG.SyntheticRoot.begin() + NumAAs,
                      DG.SyntheticRoot.end());

    // Reset the work list and repopulate with the changed abstract attributes.
    // Note that dependent ones are added above.
    Worklist.clear();
    Worklist.insert(ChangedAAs.begin(), ChangedAAs.end());
    Worklist.insert(QueryAAsAwaitingUpdate.begin(),
                    QueryAAsAwaitingUpdate.end());
    QueryAAsAwaitingUpdate.clear();

  } while (!Worklist.empty() &&
           (IterationCounter++ < MaxIterations || VerifyMaxFixpointIterations));

  if (IterationCounter > MaxIterations && !Functions.empty()) {
    auto Remark = [&](OptimizationRemarkMissed ORM) {
      return ORM << "Attributor did not reach a fixpoint after "
                 << ore::NV("Iterations", MaxIterations) << " iterations.";
    };
    Function *F = Functions.front();
    emitRemark<OptimizationRemarkMissed>(F, "FixedPoint", Remark);
  }

  LLVM_DEBUG(dbgs() << "\n[Attributor] Fixpoint iteration done after: "
                    << IterationCounter << "/" << MaxIterations
                    << " iterations\n");

  // Reset abstract arguments not settled in a sound fixpoint by now. This
  // happens when we stopped the fixpoint iteration early. Note that only the
  // ones marked as "changed" *and* the ones transitively depending on them
  // need to be reverted to a pessimistic state. Others might not be in a
  // fixpoint state but we can use the optimistic results for them anyway.
  SmallPtrSet<AbstractAttribute *, 32> Visited;
  for (unsigned u = 0; u < ChangedAAs.size(); u++) {
    AbstractAttribute *ChangedAA = ChangedAAs[u];
    if (!Visited.insert(ChangedAA).second)
      continue;

    AbstractState &State = ChangedAA->getState();
    if (!State.isAtFixpoint()) {
      State.indicatePessimisticFixpoint();

      NumAttributesTimedOut++;
    }

    while (!ChangedAA->Deps.empty()) {
      ChangedAAs.push_back(
          cast<AbstractAttribute>(ChangedAA->Deps.back().getPointer()));
      ChangedAA->Deps.pop_back();
    }
  }

  LLVM_DEBUG({
    if (!Visited.empty())
      dbgs() << "\n[Attributor] Finalized " << Visited.size()
             << " abstract attributes.\n";
  });

  if (VerifyMaxFixpointIterations && IterationCounter != MaxIterations) {
    errs() << "\n[Attributor] Fixpoint iteration done after: "
           << IterationCounter << "/" << MaxIterations << " iterations\n";
    llvm_unreachable("The fixpoint was not reached with exactly the number of "
                     "specified iterations!");
  }
}

void Attributor::registerForUpdate(AbstractAttribute &AA) {
  assert(AA.isQueryAA() &&
         "Non-query AAs should not be required to register for updates!");
  QueryAAsAwaitingUpdate.insert(&AA);
}

ChangeStatus Attributor::manifestAttributes() {
  TimeTraceScope TimeScope("Attributor::manifestAttributes");
  size_t NumFinalAAs = DG.SyntheticRoot.Deps.size();

  unsigned NumManifested = 0;
  unsigned NumAtFixpoint = 0;
  ChangeStatus ManifestChange = ChangeStatus::UNCHANGED;
  for (auto &DepAA : DG.SyntheticRoot.Deps) {
    AbstractAttribute *AA = cast<AbstractAttribute>(DepAA.getPointer());
    AbstractState &State = AA->getState();

    // If there is not already a fixpoint reached, we can now take the
    // optimistic state. This is correct because we enforced a pessimistic one
    // on abstract attributes that were transitively dependent on a changed one
    // already above.
    if (!State.isAtFixpoint())
      State.indicateOptimisticFixpoint();

    // We must not manifest Attributes that use Callbase info.
    if (AA->hasCallBaseContext())
      continue;
    // If the state is invalid, we do not try to manifest it.
    if (!State.isValidState())
      continue;

    if (AA->getCtxI() && !isRunOn(*AA->getAnchorScope()))
      continue;

    // Skip dead code.
    bool UsedAssumedInformation = false;
    if (isAssumedDead(*AA, nullptr, UsedAssumedInformation,
                      /* CheckBBLivenessOnly */ true))
      continue;
    // Check if the manifest debug counter that allows skipping manifestation of
    // AAs
    if (!DebugCounter::shouldExecute(ManifestDBGCounter))
      continue;
    // Manifest the state and record if we changed the IR.
    ChangeStatus LocalChange = AA->manifest(*this);
    if (LocalChange == ChangeStatus::CHANGED && AreStatisticsEnabled())
      AA->trackStatistics();
    LLVM_DEBUG(dbgs() << "[Attributor] Manifest " << LocalChange << " : " << *AA
                      << "\n");

    ManifestChange = ManifestChange | LocalChange;

    NumAtFixpoint++;
    NumManifested += (LocalChange == ChangeStatus::CHANGED);
  }

  (void)NumManifested;
  (void)NumAtFixpoint;
  LLVM_DEBUG(dbgs() << "\n[Attributor] Manifested " << NumManifested
                    << " arguments while " << NumAtFixpoint
                    << " were in a valid fixpoint state\n");

  NumAttributesManifested += NumManifested;
  NumAttributesValidFixpoint += NumAtFixpoint;

  (void)NumFinalAAs;
  if (NumFinalAAs != DG.SyntheticRoot.Deps.size()) {
    for (unsigned u = NumFinalAAs; u < DG.SyntheticRoot.Deps.size(); ++u)
      errs() << "Unexpected abstract attribute: "
             << cast<AbstractAttribute>(DG.SyntheticRoot.Deps[u].getPointer())
             << " :: "
             << cast<AbstractAttribute>(DG.SyntheticRoot.Deps[u].getPointer())
                    ->getIRPosition()
                    .getAssociatedValue()
             << "\n";
    llvm_unreachable("Expected the final number of abstract attributes to "
                     "remain unchanged!");
  }
  return ManifestChange;
}

void Attributor::identifyDeadInternalFunctions() {
  // Early exit if we don't intend to delete functions.
  if (!Configuration.DeleteFns)
    return;

  // Identify dead internal functions and delete them. This happens outside
  // the other fixpoint analysis as we might treat potentially dead functions
  // as live to lower the number of iterations. If they happen to be dead, the
  // below fixpoint loop will identify and eliminate them.
  SmallVector<Function *, 8> InternalFns;
  for (Function *F : Functions)
    if (F->hasLocalLinkage())
      InternalFns.push_back(F);

  SmallPtrSet<Function *, 8> LiveInternalFns;
  bool FoundLiveInternal = true;
  while (FoundLiveInternal) {
    FoundLiveInternal = false;
    for (unsigned u = 0, e = InternalFns.size(); u < e; ++u) {
      Function *F = InternalFns[u];
      if (!F)
        continue;

      bool UsedAssumedInformation = false;
      if (checkForAllCallSites(
              [&](AbstractCallSite ACS) {
                Function *Callee = ACS.getInstruction()->getFunction();
                return ToBeDeletedFunctions.count(Callee) ||
                       (Functions.count(Callee) && Callee->hasLocalLinkage() &&
                        !LiveInternalFns.count(Callee));
              },
              *F, true, nullptr, UsedAssumedInformation)) {
        continue;
      }

      LiveInternalFns.insert(F);
      InternalFns[u] = nullptr;
      FoundLiveInternal = true;
    }
  }

  for (unsigned u = 0, e = InternalFns.size(); u < e; ++u)
    if (Function *F = InternalFns[u])
      ToBeDeletedFunctions.insert(F);
}

ChangeStatus Attributor::cleanupIR() {
  TimeTraceScope TimeScope("Attributor::cleanupIR");
  // Delete stuff at the end to avoid invalid references and a nice order.
  LLVM_DEBUG(dbgs() << "\n[Attributor] Delete/replace at least "
                    << ToBeDeletedFunctions.size() << " functions and "
                    << ToBeDeletedBlocks.size() << " blocks and "
                    << ToBeDeletedInsts.size() << " instructions and "
                    << ToBeChangedValues.size() << " values and "
                    << ToBeChangedUses.size() << " uses. To insert "
                    << ToBeChangedToUnreachableInsts.size() << " unreachables."
                    << "Preserve manifest added " << ManifestAddedBlocks.size()
                    << " blocks\n");

  SmallVector<WeakTrackingVH, 32> DeadInsts;
  SmallVector<Instruction *, 32> TerminatorsToFold;

  auto ReplaceUse = [&](Use *U, Value *NewV) {
    Value *OldV = U->get();

    // If we plan to replace NewV we need to update it at this point.
    do {
      const auto &Entry = ToBeChangedValues.lookup(NewV);
      if (!Entry.first)
        break;
      NewV = Entry.first;
    } while (true);

    Instruction *I = dyn_cast<Instruction>(U->getUser());
    assert((!I || isRunOn(*I->getFunction())) &&
           "Cannot replace an instruction outside the current SCC!");

    // Do not replace uses in returns if the value is a must-tail call we will
    // not delete.
    if (auto *RI = dyn_cast_or_null<ReturnInst>(I)) {
      if (auto *CI = dyn_cast<CallInst>(OldV->stripPointerCasts()))
        if (CI->isMustTailCall() && !ToBeDeletedInsts.count(CI))
          return;
      // If we rewrite a return and the new value is not an argument, strip the
      // `returned` attribute as it is wrong now.
      if (!isa<Argument>(NewV))
        for (auto &Arg : RI->getFunction()->args())
          Arg.removeAttr(Attribute::Returned);
    }

    // Do not perform call graph altering changes outside the SCC.
    if (auto *CB = dyn_cast_or_null<CallBase>(I))
      if (CB->isCallee(U))
        return;

    LLVM_DEBUG(dbgs() << "Use " << *NewV << " in " << *U->getUser()
                      << " instead of " << *OldV << "\n");
    U->set(NewV);

    if (Instruction *I = dyn_cast<Instruction>(OldV)) {
      CGModifiedFunctions.insert(I->getFunction());
      if (!isa<PHINode>(I) && !ToBeDeletedInsts.count(I) &&
          isInstructionTriviallyDead(I))
        DeadInsts.push_back(I);
    }
    if (isa<UndefValue>(NewV) && isa<CallBase>(U->getUser())) {
      auto *CB = cast<CallBase>(U->getUser());
      if (CB->isArgOperand(U)) {
        unsigned Idx = CB->getArgOperandNo(U);
        CB->removeParamAttr(Idx, Attribute::NoUndef);
        Function *Fn = CB->getCalledFunction();
        if (Fn && Fn->arg_size() > Idx)
          Fn->removeParamAttr(Idx, Attribute::NoUndef);
      }
    }
    if (isa<Constant>(NewV) && isa<BranchInst>(U->getUser())) {
      Instruction *UserI = cast<Instruction>(U->getUser());
      if (isa<UndefValue>(NewV)) {
        ToBeChangedToUnreachableInsts.insert(UserI);
      } else {
        TerminatorsToFold.push_back(UserI);
      }
    }
  };

  for (auto &It : ToBeChangedUses) {
    Use *U = It.first;
    Value *NewV = It.second;
    ReplaceUse(U, NewV);
  }

  SmallVector<Use *, 4> Uses;
  for (auto &It : ToBeChangedValues) {
    Value *OldV = It.first;
    auto &Entry = It.second;
    Value *NewV = Entry.first;
    Uses.clear();
    for (auto &U : OldV->uses())
      if (Entry.second || !U.getUser()->isDroppable())
        Uses.push_back(&U);
    for (Use *U : Uses) {
      if (auto *I = dyn_cast<Instruction>(U->getUser()))
        if (!isRunOn(*I->getFunction()))
          continue;
      ReplaceUse(U, NewV);
    }
  }

  for (auto &V : InvokeWithDeadSuccessor)
    if (InvokeInst *II = dyn_cast_or_null<InvokeInst>(V)) {
      assert(isRunOn(*II->getFunction()) &&
             "Cannot replace an invoke outside the current SCC!");
      bool UnwindBBIsDead = II->hasFnAttr(Attribute::NoUnwind);
      bool NormalBBIsDead = II->hasFnAttr(Attribute::NoReturn);
      bool Invoke2CallAllowed =
          !AAIsDead::mayCatchAsynchronousExceptions(*II->getFunction());
      assert((UnwindBBIsDead || NormalBBIsDead) &&
             "Invoke does not have dead successors!");
      BasicBlock *BB = II->getParent();
      BasicBlock *NormalDestBB = II->getNormalDest();
      if (UnwindBBIsDead) {
        Instruction *NormalNextIP = &NormalDestBB->front();
        if (Invoke2CallAllowed) {
          changeToCall(II);
          NormalNextIP = BB->getTerminator();
        }
        if (NormalBBIsDead)
          ToBeChangedToUnreachableInsts.insert(NormalNextIP);
      } else {
        assert(NormalBBIsDead && "Broken invariant!");
        if (!NormalDestBB->getUniquePredecessor())
          NormalDestBB = SplitBlockPredecessors(NormalDestBB, {BB}, ".dead");
        ToBeChangedToUnreachableInsts.insert(&NormalDestBB->front());
      }
    }
  for (Instruction *I : TerminatorsToFold) {
    assert(isRunOn(*I->getFunction()) &&
           "Cannot replace a terminator outside the current SCC!");
    CGModifiedFunctions.insert(I->getFunction());
    ConstantFoldTerminator(I->getParent());
  }
  for (auto &V : ToBeChangedToUnreachableInsts)
    if (Instruction *I = dyn_cast_or_null<Instruction>(V)) {
      assert(isRunOn(*I->getFunction()) &&
             "Cannot replace an instruction outside the current SCC!");
      CGModifiedFunctions.insert(I->getFunction());
      changeToUnreachable(I);
    }

  for (auto &V : ToBeDeletedInsts) {
    if (Instruction *I = dyn_cast_or_null<Instruction>(V)) {
      if (auto *CB = dyn_cast<CallBase>(I)) {
        assert(isRunOn(*I->getFunction()) &&
               "Cannot delete an instruction outside the current SCC!");
        if (!isa<IntrinsicInst>(CB))
          Configuration.CGUpdater.removeCallSite(*CB);
      }
      I->dropDroppableUses();
      CGModifiedFunctions.insert(I->getFunction());
      if (!I->getType()->isVoidTy())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      if (!isa<PHINode>(I) && isInstructionTriviallyDead(I))
        DeadInsts.push_back(I);
      else
        I->eraseFromParent();
    }
  }

  llvm::erase_if(DeadInsts, [&](WeakTrackingVH I) { return !I; });

  LLVM_DEBUG({
    dbgs() << "[Attributor] DeadInsts size: " << DeadInsts.size() << "\n";
    for (auto &I : DeadInsts)
      if (I)
        dbgs() << "  - " << *I << "\n";
  });

  RecursivelyDeleteTriviallyDeadInstructions(DeadInsts);

  if (unsigned NumDeadBlocks = ToBeDeletedBlocks.size()) {
    SmallVector<BasicBlock *, 8> ToBeDeletedBBs;
    ToBeDeletedBBs.reserve(NumDeadBlocks);
    for (BasicBlock *BB : ToBeDeletedBlocks) {
      assert(isRunOn(*BB->getParent()) &&
             "Cannot delete a block outside the current SCC!");
      CGModifiedFunctions.insert(BB->getParent());
      // Do not delete BBs added during manifests of AAs.
      if (ManifestAddedBlocks.contains(BB))
        continue;
      ToBeDeletedBBs.push_back(BB);
    }
    // Actually we do not delete the blocks but squash them into a single
    // unreachable but untangling branches that jump here is something we need
    // to do in a more generic way.
    detachDeadBlocks(ToBeDeletedBBs, nullptr);
  }

  identifyDeadInternalFunctions();

  // Rewrite the functions as requested during manifest.
  ChangeStatus ManifestChange = rewriteFunctionSignatures(CGModifiedFunctions);

  for (Function *Fn : CGModifiedFunctions)
    if (!ToBeDeletedFunctions.count(Fn) && Functions.count(Fn))
      Configuration.CGUpdater.reanalyzeFunction(*Fn);

  for (Function *Fn : ToBeDeletedFunctions) {
    if (!Functions.count(Fn))
      continue;
    Configuration.CGUpdater.removeFunction(*Fn);
  }

  if (!ToBeChangedUses.empty())
    ManifestChange = ChangeStatus::CHANGED;

  if (!ToBeChangedToUnreachableInsts.empty())
    ManifestChange = ChangeStatus::CHANGED;

  if (!ToBeDeletedFunctions.empty())
    ManifestChange = ChangeStatus::CHANGED;

  if (!ToBeDeletedBlocks.empty())
    ManifestChange = ChangeStatus::CHANGED;

  if (!ToBeDeletedInsts.empty())
    ManifestChange = ChangeStatus::CHANGED;

  if (!InvokeWithDeadSuccessor.empty())
    ManifestChange = ChangeStatus::CHANGED;

  if (!DeadInsts.empty())
    ManifestChange = ChangeStatus::CHANGED;

  NumFnDeleted += ToBeDeletedFunctions.size();

  LLVM_DEBUG(dbgs() << "[Attributor] Deleted " << ToBeDeletedFunctions.size()
                    << " functions after manifest.\n");

#ifdef EXPENSIVE_CHECKS
  for (Function *F : Functions) {
    if (ToBeDeletedFunctions.count(F))
      continue;
    assert(!verifyFunction(*F, &errs()) && "Module verification failed!");
  }
#endif

  return ManifestChange;
}

ChangeStatus Attributor::run() {
  TimeTraceScope TimeScope("Attributor::run");
  AttributorCallGraph ACallGraph(*this);

  if (PrintCallGraph)
    ACallGraph.populateAll();

  Phase = AttributorPhase::UPDATE;
  runTillFixpoint();

  // dump graphs on demand
  if (DumpDepGraph)
    DG.dumpGraph();

  if (ViewDepGraph)
    DG.viewGraph();

  if (PrintDependencies)
    DG.print();

  Phase = AttributorPhase::MANIFEST;
  ChangeStatus ManifestChange = manifestAttributes();

  Phase = AttributorPhase::CLEANUP;
  ChangeStatus CleanupChange = cleanupIR();

  if (PrintCallGraph)
    ACallGraph.print();

  return ManifestChange | CleanupChange;
}

ChangeStatus Attributor::updateAA(AbstractAttribute &AA) {
  TimeTraceScope TimeScope(
      AA.getName() + std::to_string(AA.getIRPosition().getPositionKind()) +
      "::updateAA");
  assert(Phase == AttributorPhase::UPDATE &&
         "We can update AA only in the update stage!");

  // Use a new dependence vector for this update.
  DependenceVector DV;
  DependenceStack.push_back(&DV);

  auto &AAState = AA.getState();
  ChangeStatus CS = ChangeStatus::UNCHANGED;
  bool UsedAssumedInformation = false;
  if (!isAssumedDead(AA, nullptr, UsedAssumedInformation,
                     /* CheckBBLivenessOnly */ true))
    CS = AA.update(*this);

  if (!AA.isQueryAA() && DV.empty()) {
    // If the attribute did not query any non-fix information, the state
    // will not change and we can indicate that right away.
    AAState.indicateOptimisticFixpoint();
  }

  if (!AAState.isAtFixpoint())
    rememberDependences();

  // Verify the stack was used properly, that is we pop the dependence vector we
  // put there earlier.
  DependenceVector *PoppedDV = DependenceStack.pop_back_val();
  (void)PoppedDV;
  assert(PoppedDV == &DV && "Inconsistent usage of the dependence stack!");

  return CS;
}

void Attributor::createShallowWrapper(Function &F) {
  assert(!F.isDeclaration() && "Cannot create a wrapper around a declaration!");

  Module &M = *F.getParent();
  LLVMContext &Ctx = M.getContext();
  FunctionType *FnTy = F.getFunctionType();

  Function *Wrapper =
      Function::Create(FnTy, F.getLinkage(), F.getAddressSpace(), F.getName());
  F.setName(""); // set the inside function anonymous
  M.getFunctionList().insert(F.getIterator(), Wrapper);

  F.setLinkage(GlobalValue::InternalLinkage);

  F.replaceAllUsesWith(Wrapper);
  assert(F.use_empty() && "Uses remained after wrapper was created!");

  // Move the COMDAT section to the wrapper.
  // TODO: Check if we need to keep it for F as well.
  Wrapper->setComdat(F.getComdat());
  F.setComdat(nullptr);

  // Copy all metadata and attributes but keep them on F as well.
  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  F.getAllMetadata(MDs);
  for (auto MDIt : MDs)
    Wrapper->addMetadata(MDIt.first, *MDIt.second);
  Wrapper->setAttributes(F.getAttributes());

  // Create the call in the wrapper.
  BasicBlock *EntryBB = BasicBlock::Create(Ctx, "entry", Wrapper);

  SmallVector<Value *, 8> Args;
  Argument *FArgIt = F.arg_begin();
  for (Argument &Arg : Wrapper->args()) {
    Args.push_back(&Arg);
    Arg.setName((FArgIt++)->getName());
  }

  CallInst *CI = CallInst::Create(&F, Args, "", EntryBB);
  CI->setTailCall(true);
  CI->addFnAttr(Attribute::NoInline);
  ReturnInst::Create(Ctx, CI->getType()->isVoidTy() ? nullptr : CI, EntryBB);

  NumFnShallowWrappersCreated++;
}

bool Attributor::isInternalizable(Function &F) {
  if (F.isDeclaration() || F.hasLocalLinkage() ||
      GlobalValue::isInterposableLinkage(F.getLinkage()))
    return false;
  return true;
}

Function *Attributor::internalizeFunction(Function &F, bool Force) {
  if (!AllowDeepWrapper && !Force)
    return nullptr;
  if (!isInternalizable(F))
    return nullptr;

  SmallPtrSet<Function *, 2> FnSet = {&F};
  DenseMap<Function *, Function *> InternalizedFns;
  internalizeFunctions(FnSet, InternalizedFns);

  return InternalizedFns[&F];
}

bool Attributor::internalizeFunctions(SmallPtrSetImpl<Function *> &FnSet,
                                      DenseMap<Function *, Function *> &FnMap) {
  for (Function *F : FnSet)
    if (!Attributor::isInternalizable(*F))
      return false;

  FnMap.clear();
  // Generate the internalized version of each function.
  for (Function *F : FnSet) {
    Module &M = *F->getParent();
    FunctionType *FnTy = F->getFunctionType();

    // Create a copy of the current function
    Function *Copied =
        Function::Create(FnTy, F->getLinkage(), F->getAddressSpace(),
                         F->getName() + ".internalized");
    ValueToValueMapTy VMap;
    auto *NewFArgIt = Copied->arg_begin();
    for (auto &Arg : F->args()) {
      auto ArgName = Arg.getName();
      NewFArgIt->setName(ArgName);
      VMap[&Arg] = &(*NewFArgIt++);
    }
    SmallVector<ReturnInst *, 8> Returns;

    // Copy the body of the original function to the new one
    CloneFunctionInto(Copied, F, VMap,
                      CloneFunctionChangeType::LocalChangesOnly, Returns);

    // Set the linakage and visibility late as CloneFunctionInto has some
    // implicit requirements.
    Copied->setVisibility(GlobalValue::DefaultVisibility);
    Copied->setLinkage(GlobalValue::PrivateLinkage);

    // Copy metadata
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F->getAllMetadata(MDs);
    for (auto MDIt : MDs)
      if (!Copied->hasMetadata())
        Copied->addMetadata(MDIt.first, *MDIt.second);

    M.getFunctionList().insert(F->getIterator(), Copied);
    Copied->setDSOLocal(true);
    FnMap[F] = Copied;
  }

  // Replace all uses of the old function with the new internalized function
  // unless the caller is a function that was just internalized.
  for (Function *F : FnSet) {
    auto &InternalizedFn = FnMap[F];
    auto IsNotInternalized = [&](Use &U) -> bool {
      if (auto *CB = dyn_cast<CallBase>(U.getUser()))
        return !FnMap.lookup(CB->getCaller());
      return false;
    };
    F->replaceUsesWithIf(InternalizedFn, IsNotInternalized);
  }

  return true;
}

bool Attributor::isValidFunctionSignatureRewrite(
    Argument &Arg, ArrayRef<Type *> ReplacementTypes) {

  if (!Configuration.RewriteSignatures)
    return false;

  Function *Fn = Arg.getParent();
  auto CallSiteCanBeChanged = [Fn](AbstractCallSite ACS) {
    // Forbid the call site to cast the function return type. If we need to
    // rewrite these functions we need to re-create a cast for the new call site
    // (if the old had uses).
    if (!ACS.getCalledFunction() ||
        ACS.getInstruction()->getType() !=
            ACS.getCalledFunction()->getReturnType())
      return false;
    if (ACS.getCalledOperand()->getType() != Fn->getType())
      return false;
    // Forbid must-tail calls for now.
    return !ACS.isCallbackCall() && !ACS.getInstruction()->isMustTailCall();
  };

  // Avoid var-arg functions for now.
  if (Fn->isVarArg()) {
    LLVM_DEBUG(dbgs() << "[Attributor] Cannot rewrite var-args functions\n");
    return false;
  }

  // Avoid functions with complicated argument passing semantics.
  AttributeList FnAttributeList = Fn->getAttributes();
  if (FnAttributeList.hasAttrSomewhere(Attribute::Nest) ||
      FnAttributeList.hasAttrSomewhere(Attribute::StructRet) ||
      FnAttributeList.hasAttrSomewhere(Attribute::InAlloca) ||
      FnAttributeList.hasAttrSomewhere(Attribute::Preallocated)) {
    LLVM_DEBUG(
        dbgs() << "[Attributor] Cannot rewrite due to complex attribute\n");
    return false;
  }

  // Avoid callbacks for now.
  bool UsedAssumedInformation = false;
  if (!checkForAllCallSites(CallSiteCanBeChanged, *Fn, true, nullptr,
                            UsedAssumedInformation)) {
    LLVM_DEBUG(dbgs() << "[Attributor] Cannot rewrite all call sites\n");
    return false;
  }

  auto InstPred = [](Instruction &I) {
    if (auto *CI = dyn_cast<CallInst>(&I))
      return !CI->isMustTailCall();
    return true;
  };

  // Forbid must-tail calls for now.
  // TODO:
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(*Fn);
  if (!checkForAllInstructionsImpl(nullptr, OpcodeInstMap, InstPred, nullptr,
                                   nullptr, {Instruction::Call},
                                   UsedAssumedInformation)) {
    LLVM_DEBUG(dbgs() << "[Attributor] Cannot rewrite due to instructions\n");
    return false;
  }

  return true;
}

bool Attributor::registerFunctionSignatureRewrite(
    Argument &Arg, ArrayRef<Type *> ReplacementTypes,
    ArgumentReplacementInfo::CalleeRepairCBTy &&CalleeRepairCB,
    ArgumentReplacementInfo::ACSRepairCBTy &&ACSRepairCB) {
  LLVM_DEBUG(dbgs() << "[Attributor] Register new rewrite of " << Arg << " in "
                    << Arg.getParent()->getName() << " with "
                    << ReplacementTypes.size() << " replacements\n");
  assert(isValidFunctionSignatureRewrite(Arg, ReplacementTypes) &&
         "Cannot register an invalid rewrite");

  Function *Fn = Arg.getParent();
  SmallVectorImpl<std::unique_ptr<ArgumentReplacementInfo>> &ARIs =
      ArgumentReplacementMap[Fn];
  if (ARIs.empty())
    ARIs.resize(Fn->arg_size());

  // If we have a replacement already with less than or equal new arguments,
  // ignore this request.
  std::unique_ptr<ArgumentReplacementInfo> &ARI = ARIs[Arg.getArgNo()];
  if (ARI && ARI->getNumReplacementArgs() <= ReplacementTypes.size()) {
    LLVM_DEBUG(dbgs() << "[Attributor] Existing rewrite is preferred\n");
    return false;
  }

  // If we have a replacement already but we like the new one better, delete
  // the old.
  ARI.reset();

  LLVM_DEBUG(dbgs() << "[Attributor] Register new rewrite of " << Arg << " in "
                    << Arg.getParent()->getName() << " with "
                    << ReplacementTypes.size() << " replacements\n");

  // Remember the replacement.
  ARI.reset(new ArgumentReplacementInfo(*this, Arg, ReplacementTypes,
                                        std::move(CalleeRepairCB),
                                        std::move(ACSRepairCB)));

  return true;
}

bool Attributor::shouldSeedAttribute(AbstractAttribute &AA) {
  bool Result = true;
#ifndef NDEBUG
  if (SeedAllowList.size() != 0)
    Result = llvm::is_contained(SeedAllowList, AA.getName());
  Function *Fn = AA.getAnchorScope();
  if (FunctionSeedAllowList.size() != 0 && Fn)
    Result &= llvm::is_contained(FunctionSeedAllowList, Fn->getName());
#endif
  return Result;
}

ChangeStatus Attributor::rewriteFunctionSignatures(
    SmallSetVector<Function *, 8> &ModifiedFns) {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  for (auto &It : ArgumentReplacementMap) {
    Function *OldFn = It.getFirst();

    // Deleted functions do not require rewrites.
    if (!Functions.count(OldFn) || ToBeDeletedFunctions.count(OldFn))
      continue;

    const SmallVectorImpl<std::unique_ptr<ArgumentReplacementInfo>> &ARIs =
        It.getSecond();
    assert(ARIs.size() == OldFn->arg_size() && "Inconsistent state!");

    SmallVector<Type *, 16> NewArgumentTypes;
    SmallVector<AttributeSet, 16> NewArgumentAttributes;

    // Collect replacement argument types and copy over existing attributes.
    AttributeList OldFnAttributeList = OldFn->getAttributes();
    for (Argument &Arg : OldFn->args()) {
      if (const std::unique_ptr<ArgumentReplacementInfo> &ARI =
              ARIs[Arg.getArgNo()]) {
        NewArgumentTypes.append(ARI->ReplacementTypes.begin(),
                                ARI->ReplacementTypes.end());
        NewArgumentAttributes.append(ARI->getNumReplacementArgs(),
                                     AttributeSet());
      } else {
        NewArgumentTypes.push_back(Arg.getType());
        NewArgumentAttributes.push_back(
            OldFnAttributeList.getParamAttrs(Arg.getArgNo()));
      }
    }

    uint64_t LargestVectorWidth = 0;
    for (auto *I : NewArgumentTypes)
      if (auto *VT = dyn_cast<llvm::VectorType>(I))
        LargestVectorWidth = std::max(
            LargestVectorWidth, VT->getPrimitiveSizeInBits().getKnownMinSize());

    FunctionType *OldFnTy = OldFn->getFunctionType();
    Type *RetTy = OldFnTy->getReturnType();

    // Construct the new function type using the new arguments types.
    FunctionType *NewFnTy =
        FunctionType::get(RetTy, NewArgumentTypes, OldFnTy->isVarArg());

    LLVM_DEBUG(dbgs() << "[Attributor] Function rewrite '" << OldFn->getName()
                      << "' from " << *OldFn->getFunctionType() << " to "
                      << *NewFnTy << "\n");

    // Create the new function body and insert it into the module.
    Function *NewFn = Function::Create(NewFnTy, OldFn->getLinkage(),
                                       OldFn->getAddressSpace(), "");
    Functions.insert(NewFn);
    OldFn->getParent()->getFunctionList().insert(OldFn->getIterator(), NewFn);
    NewFn->takeName(OldFn);
    NewFn->copyAttributesFrom(OldFn);

    // Patch the pointer to LLVM function in debug info descriptor.
    NewFn->setSubprogram(OldFn->getSubprogram());
    OldFn->setSubprogram(nullptr);

    // Recompute the parameter attributes list based on the new arguments for
    // the function.
    LLVMContext &Ctx = OldFn->getContext();
    NewFn->setAttributes(AttributeList::get(
        Ctx, OldFnAttributeList.getFnAttrs(), OldFnAttributeList.getRetAttrs(),
        NewArgumentAttributes));
    AttributeFuncs::updateMinLegalVectorWidthAttr(*NewFn, LargestVectorWidth);

    // Since we have now created the new function, splice the body of the old
    // function right into the new function, leaving the old rotting hulk of the
    // function empty.
    NewFn->getBasicBlockList().splice(NewFn->begin(),
                                      OldFn->getBasicBlockList());

    // Fixup block addresses to reference new function.
    SmallVector<BlockAddress *, 8u> BlockAddresses;
    for (User *U : OldFn->users())
      if (auto *BA = dyn_cast<BlockAddress>(U))
        BlockAddresses.push_back(BA);
    for (auto *BA : BlockAddresses)
      BA->replaceAllUsesWith(BlockAddress::get(NewFn, BA->getBasicBlock()));

    // Set of all "call-like" instructions that invoke the old function mapped
    // to their new replacements.
    SmallVector<std::pair<CallBase *, CallBase *>, 8> CallSitePairs;

    // Callback to create a new "call-like" instruction for a given one.
    auto CallSiteReplacementCreator = [&](AbstractCallSite ACS) {
      CallBase *OldCB = cast<CallBase>(ACS.getInstruction());
      const AttributeList &OldCallAttributeList = OldCB->getAttributes();

      // Collect the new argument operands for the replacement call site.
      SmallVector<Value *, 16> NewArgOperands;
      SmallVector<AttributeSet, 16> NewArgOperandAttributes;
      for (unsigned OldArgNum = 0; OldArgNum < ARIs.size(); ++OldArgNum) {
        unsigned NewFirstArgNum = NewArgOperands.size();
        (void)NewFirstArgNum; // only used inside assert.
        if (const std::unique_ptr<ArgumentReplacementInfo> &ARI =
                ARIs[OldArgNum]) {
          if (ARI->ACSRepairCB)
            ARI->ACSRepairCB(*ARI, ACS, NewArgOperands);
          assert(ARI->getNumReplacementArgs() + NewFirstArgNum ==
                     NewArgOperands.size() &&
                 "ACS repair callback did not provide as many operand as new "
                 "types were registered!");
          // TODO: Exose the attribute set to the ACS repair callback
          NewArgOperandAttributes.append(ARI->ReplacementTypes.size(),
                                         AttributeSet());
        } else {
          NewArgOperands.push_back(ACS.getCallArgOperand(OldArgNum));
          NewArgOperandAttributes.push_back(
              OldCallAttributeList.getParamAttrs(OldArgNum));
        }
      }

      assert(NewArgOperands.size() == NewArgOperandAttributes.size() &&
             "Mismatch # argument operands vs. # argument operand attributes!");
      assert(NewArgOperands.size() == NewFn->arg_size() &&
             "Mismatch # argument operands vs. # function arguments!");

      SmallVector<OperandBundleDef, 4> OperandBundleDefs;
      OldCB->getOperandBundlesAsDefs(OperandBundleDefs);

      // Create a new call or invoke instruction to replace the old one.
      CallBase *NewCB;
      if (InvokeInst *II = dyn_cast<InvokeInst>(OldCB)) {
        NewCB =
            InvokeInst::Create(NewFn, II->getNormalDest(), II->getUnwindDest(),
                               NewArgOperands, OperandBundleDefs, "", OldCB);
      } else {
        auto *NewCI = CallInst::Create(NewFn, NewArgOperands, OperandBundleDefs,
                                       "", OldCB);
        NewCI->setTailCallKind(cast<CallInst>(OldCB)->getTailCallKind());
        NewCB = NewCI;
      }

      // Copy over various properties and the new attributes.
      NewCB->copyMetadata(*OldCB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});
      NewCB->setCallingConv(OldCB->getCallingConv());
      NewCB->takeName(OldCB);
      NewCB->setAttributes(AttributeList::get(
          Ctx, OldCallAttributeList.getFnAttrs(),
          OldCallAttributeList.getRetAttrs(), NewArgOperandAttributes));

      AttributeFuncs::updateMinLegalVectorWidthAttr(*NewCB->getCaller(),
                                                    LargestVectorWidth);

      CallSitePairs.push_back({OldCB, NewCB});
      return true;
    };

    // Use the CallSiteReplacementCreator to create replacement call sites.
    bool UsedAssumedInformation = false;
    bool Success = checkForAllCallSites(CallSiteReplacementCreator, *OldFn,
                                        true, nullptr, UsedAssumedInformation);
    (void)Success;
    assert(Success && "Assumed call site replacement to succeed!");

    // Rewire the arguments.
    Argument *OldFnArgIt = OldFn->arg_begin();
    Argument *NewFnArgIt = NewFn->arg_begin();
    for (unsigned OldArgNum = 0; OldArgNum < ARIs.size();
         ++OldArgNum, ++OldFnArgIt) {
      if (const std::unique_ptr<ArgumentReplacementInfo> &ARI =
              ARIs[OldArgNum]) {
        if (ARI->CalleeRepairCB)
          ARI->CalleeRepairCB(*ARI, *NewFn, NewFnArgIt);
        if (ARI->ReplacementTypes.empty())
          OldFnArgIt->replaceAllUsesWith(
              PoisonValue::get(OldFnArgIt->getType()));
        NewFnArgIt += ARI->ReplacementTypes.size();
      } else {
        NewFnArgIt->takeName(&*OldFnArgIt);
        OldFnArgIt->replaceAllUsesWith(&*NewFnArgIt);
        ++NewFnArgIt;
      }
    }

    // Eliminate the instructions *after* we visited all of them.
    for (auto &CallSitePair : CallSitePairs) {
      CallBase &OldCB = *CallSitePair.first;
      CallBase &NewCB = *CallSitePair.second;
      assert(OldCB.getType() == NewCB.getType() &&
             "Cannot handle call sites with different types!");
      ModifiedFns.insert(OldCB.getFunction());
      Configuration.CGUpdater.replaceCallSite(OldCB, NewCB);
      OldCB.replaceAllUsesWith(&NewCB);
      OldCB.eraseFromParent();
    }

    // Replace the function in the call graph (if any).
    Configuration.CGUpdater.replaceFunctionWith(*OldFn, *NewFn);

    // If the old function was modified and needed to be reanalyzed, the new one
    // does now.
    if (ModifiedFns.remove(OldFn))
      ModifiedFns.insert(NewFn);

    Changed = ChangeStatus::CHANGED;
  }

  return Changed;
}

void InformationCache::initializeInformationCache(const Function &CF,
                                                  FunctionInfo &FI) {
  // As we do not modify the function here we can remove the const
  // withouth breaking implicit assumptions. At the end of the day, we could
  // initialize the cache eagerly which would look the same to the users.
  Function &F = const_cast<Function &>(CF);

  // Walk all instructions to find interesting instructions that might be
  // queried by abstract attributes during their initialization or update.
  // This has to happen before we create attributes.

  DenseMap<const Value *, Optional<short>> AssumeUsesMap;

  // Add \p V to the assume uses map which track the number of uses outside of
  // "visited" assumes. If no outside uses are left the value is added to the
  // assume only use vector.
  auto AddToAssumeUsesMap = [&](const Value &V) -> void {
    SmallVector<const Instruction *> Worklist;
    if (auto *I = dyn_cast<Instruction>(&V))
      Worklist.push_back(I);
    while (!Worklist.empty()) {
      const Instruction *I = Worklist.pop_back_val();
      Optional<short> &NumUses = AssumeUsesMap[I];
      if (!NumUses.hasValue())
        NumUses = I->getNumUses();
      NumUses = NumUses.getValue() - /* this assume */ 1;
      if (NumUses.getValue() != 0)
        continue;
      AssumeOnlyValues.insert(I);
      for (const Value *Op : I->operands())
        if (auto *OpI = dyn_cast<Instruction>(Op))
          Worklist.push_back(OpI);
    }
  };

  for (Instruction &I : instructions(&F)) {
    bool IsInterestingOpcode = false;

    // To allow easy access to all instructions in a function with a given
    // opcode we store them in the InfoCache. As not all opcodes are interesting
    // to concrete attributes we only cache the ones that are as identified in
    // the following switch.
    // Note: There are no concrete attributes now so this is initially empty.
    switch (I.getOpcode()) {
    default:
      assert(!isa<CallBase>(&I) &&
             "New call base instruction type needs to be known in the "
             "Attributor.");
      break;
    case Instruction::Call:
      // Calls are interesting on their own, additionally:
      // For `llvm.assume` calls we also fill the KnowledgeMap as we find them.
      // For `must-tail` calls we remember the caller and callee.
      if (auto *Assume = dyn_cast<AssumeInst>(&I)) {
        fillMapFromAssume(*Assume, KnowledgeMap);
        AddToAssumeUsesMap(*Assume->getArgOperand(0));
      } else if (cast<CallInst>(I).isMustTailCall()) {
        FI.ContainsMustTailCall = true;
        if (const Function *Callee = cast<CallInst>(I).getCalledFunction())
          getFunctionInfo(*Callee).CalledViaMustTail = true;
      }
      LLVM_FALLTHROUGH;
    case Instruction::CallBr:
    case Instruction::Invoke:
    case Instruction::CleanupRet:
    case Instruction::CatchSwitch:
    case Instruction::AtomicRMW:
    case Instruction::AtomicCmpXchg:
    case Instruction::Br:
    case Instruction::Resume:
    case Instruction::Ret:
    case Instruction::Load:
      // The alignment of a pointer is interesting for loads.
    case Instruction::Store:
      // The alignment of a pointer is interesting for stores.
    case Instruction::Alloca:
    case Instruction::AddrSpaceCast:
      IsInterestingOpcode = true;
    }
    if (IsInterestingOpcode) {
      auto *&Insts = FI.OpcodeInstMap[I.getOpcode()];
      if (!Insts)
        Insts = new (Allocator) InstructionVectorTy();
      Insts->push_back(&I);
    }
    if (I.mayReadOrWriteMemory())
      FI.RWInsts.push_back(&I);
  }

  if (F.hasFnAttribute(Attribute::AlwaysInline) &&
      isInlineViable(F).isSuccess())
    InlineableFunctions.insert(&F);
}

AAResults *InformationCache::getAAResultsForFunction(const Function &F) {
  return AG.getAnalysis<AAManager>(F);
}

InformationCache::FunctionInfo::~FunctionInfo() {
  // The instruction vectors are allocated using a BumpPtrAllocator, we need to
  // manually destroy them.
  for (auto &It : OpcodeInstMap)
    It.getSecond()->~InstructionVectorTy();
}

void Attributor::recordDependence(const AbstractAttribute &FromAA,
                                  const AbstractAttribute &ToAA,
                                  DepClassTy DepClass) {
  if (DepClass == DepClassTy::NONE)
    return;
  // If we are outside of an update, thus before the actual fixpoint iteration
  // started (= when we create AAs), we do not track dependences because we will
  // put all AAs into the initial worklist anyway.
  if (DependenceStack.empty())
    return;
  if (FromAA.getState().isAtFixpoint())
    return;
  DependenceStack.back()->push_back({&FromAA, &ToAA, DepClass});
}

void Attributor::rememberDependences() {
  assert(!DependenceStack.empty() && "No dependences to remember!");

  for (DepInfo &DI : *DependenceStack.back()) {
    assert((DI.DepClass == DepClassTy::REQUIRED ||
            DI.DepClass == DepClassTy::OPTIONAL) &&
           "Expected required or optional dependence (1 bit)!");
    auto &DepAAs = const_cast<AbstractAttribute &>(*DI.FromAA).Deps;
    DepAAs.push_back(AbstractAttribute::DepTy(
        const_cast<AbstractAttribute *>(DI.ToAA), unsigned(DI.DepClass)));
  }
}

void Attributor::identifyDefaultAbstractAttributes(Function &F) {
  if (!VisitedFunctions.insert(&F).second)
    return;
  if (F.isDeclaration())
    return;

  // In non-module runs we need to look at the call sites of a function to
  // determine if it is part of a must-tail call edge. This will influence what
  // attributes we can derive.
  InformationCache::FunctionInfo &FI = InfoCache.getFunctionInfo(F);
  if (!isModulePass() && !FI.CalledViaMustTail) {
    for (const Use &U : F.uses())
      if (const auto *CB = dyn_cast<CallBase>(U.getUser()))
        if (CB->isCallee(&U) && CB->isMustTailCall())
          FI.CalledViaMustTail = true;
  }

  IRPosition FPos = IRPosition::function(F);

  // Check for dead BasicBlocks in every function.
  // We need dead instruction detection because we do not want to deal with
  // broken IR in which SSA rules do not apply.
  getOrCreateAAFor<AAIsDead>(FPos);

  // Every function might be "will-return".
  getOrCreateAAFor<AAWillReturn>(FPos);

  // Every function might contain instructions that cause "undefined behavior".
  getOrCreateAAFor<AAUndefinedBehavior>(FPos);

  // Every function can be nounwind.
  getOrCreateAAFor<AANoUnwind>(FPos);

  // Every function might be marked "nosync"
  getOrCreateAAFor<AANoSync>(FPos);

  // Every function might be "no-free".
  getOrCreateAAFor<AANoFree>(FPos);

  // Every function might be "no-return".
  getOrCreateAAFor<AANoReturn>(FPos);

  // Every function might be "no-recurse".
  getOrCreateAAFor<AANoRecurse>(FPos);

  // Every function might be "readnone/readonly/writeonly/...".
  getOrCreateAAFor<AAMemoryBehavior>(FPos);

  // Every function can be "readnone/argmemonly/inaccessiblememonly/...".
  getOrCreateAAFor<AAMemoryLocation>(FPos);

  // Every function can track active assumptions.
  getOrCreateAAFor<AAAssumptionInfo>(FPos);

  // Every function might be applicable for Heap-To-Stack conversion.
  if (EnableHeapToStack)
    getOrCreateAAFor<AAHeapToStack>(FPos);

  // Return attributes are only appropriate if the return type is non void.
  Type *ReturnType = F.getReturnType();
  if (!ReturnType->isVoidTy()) {
    // Argument attribute "returned" --- Create only one per function even
    // though it is an argument attribute.
    getOrCreateAAFor<AAReturnedValues>(FPos);

    IRPosition RetPos = IRPosition::returned(F);

    // Every returned value might be dead.
    getOrCreateAAFor<AAIsDead>(RetPos);

    // Every function might be simplified.
    bool UsedAssumedInformation = false;
    getAssumedSimplified(RetPos, nullptr, UsedAssumedInformation);

    // Every returned value might be marked noundef.
    getOrCreateAAFor<AANoUndef>(RetPos);

    if (ReturnType->isPointerTy()) {

      // Every function with pointer return type might be marked align.
      getOrCreateAAFor<AAAlign>(RetPos);

      // Every function with pointer return type might be marked nonnull.
      getOrCreateAAFor<AANonNull>(RetPos);

      // Every function with pointer return type might be marked noalias.
      getOrCreateAAFor<AANoAlias>(RetPos);

      // Every function with pointer return type might be marked
      // dereferenceable.
      getOrCreateAAFor<AADereferenceable>(RetPos);
    }
  }

  for (Argument &Arg : F.args()) {
    IRPosition ArgPos = IRPosition::argument(Arg);

    // Every argument might be simplified. We have to go through the Attributor
    // interface though as outside AAs can register custom simplification
    // callbacks.
    bool UsedAssumedInformation = false;
    getAssumedSimplified(ArgPos, /* AA */ nullptr, UsedAssumedInformation);

    // Every argument might be dead.
    getOrCreateAAFor<AAIsDead>(ArgPos);

    // Every argument might be marked noundef.
    getOrCreateAAFor<AANoUndef>(ArgPos);

    if (Arg.getType()->isPointerTy()) {
      // Every argument with pointer type might be marked nonnull.
      getOrCreateAAFor<AANonNull>(ArgPos);

      // Every argument with pointer type might be marked noalias.
      getOrCreateAAFor<AANoAlias>(ArgPos);

      // Every argument with pointer type might be marked dereferenceable.
      getOrCreateAAFor<AADereferenceable>(ArgPos);

      // Every argument with pointer type might be marked align.
      getOrCreateAAFor<AAAlign>(ArgPos);

      // Every argument with pointer type might be marked nocapture.
      getOrCreateAAFor<AANoCapture>(ArgPos);

      // Every argument with pointer type might be marked
      // "readnone/readonly/writeonly/..."
      getOrCreateAAFor<AAMemoryBehavior>(ArgPos);

      // Every argument with pointer type might be marked nofree.
      getOrCreateAAFor<AANoFree>(ArgPos);

      // Every argument with pointer type might be privatizable (or promotable)
      getOrCreateAAFor<AAPrivatizablePtr>(ArgPos);
    }
  }

  auto CallSitePred = [&](Instruction &I) -> bool {
    auto &CB = cast<CallBase>(I);
    IRPosition CBInstPos = IRPosition::inst(CB);
    IRPosition CBFnPos = IRPosition::callsite_function(CB);

    // Call sites might be dead if they do not have side effects and no live
    // users. The return value might be dead if there are no live users.
    getOrCreateAAFor<AAIsDead>(CBInstPos);

    Function *Callee = CB.getCalledFunction();
    // TODO: Even if the callee is not known now we might be able to simplify
    //       the call/callee.
    if (!Callee)
      return true;

    // Every call site can track active assumptions.
    getOrCreateAAFor<AAAssumptionInfo>(CBFnPos);

    // Skip declarations except if annotations on their call sites were
    // explicitly requested.
    if (!AnnotateDeclarationCallSites && Callee->isDeclaration() &&
        !Callee->hasMetadata(LLVMContext::MD_callback))
      return true;

    if (!Callee->getReturnType()->isVoidTy() && !CB.use_empty()) {

      IRPosition CBRetPos = IRPosition::callsite_returned(CB);
      bool UsedAssumedInformation = false;
      getAssumedSimplified(CBRetPos, nullptr, UsedAssumedInformation);
    }

    for (int I = 0, E = CB.arg_size(); I < E; ++I) {

      IRPosition CBArgPos = IRPosition::callsite_argument(CB, I);

      // Every call site argument might be dead.
      getOrCreateAAFor<AAIsDead>(CBArgPos);

      // Call site argument might be simplified. We have to go through the
      // Attributor interface though as outside AAs can register custom
      // simplification callbacks.
      bool UsedAssumedInformation = false;
      getAssumedSimplified(CBArgPos, /* AA */ nullptr, UsedAssumedInformation);

      // Every call site argument might be marked "noundef".
      getOrCreateAAFor<AANoUndef>(CBArgPos);

      if (!CB.getArgOperand(I)->getType()->isPointerTy())
        continue;

      // Call site argument attribute "non-null".
      getOrCreateAAFor<AANonNull>(CBArgPos);

      // Call site argument attribute "nocapture".
      getOrCreateAAFor<AANoCapture>(CBArgPos);

      // Call site argument attribute "no-alias".
      getOrCreateAAFor<AANoAlias>(CBArgPos);

      // Call site argument attribute "dereferenceable".
      getOrCreateAAFor<AADereferenceable>(CBArgPos);

      // Call site argument attribute "align".
      getOrCreateAAFor<AAAlign>(CBArgPos);

      // Call site argument attribute
      // "readnone/readonly/writeonly/..."
      getOrCreateAAFor<AAMemoryBehavior>(CBArgPos);

      // Call site argument attribute "nofree".
      getOrCreateAAFor<AANoFree>(CBArgPos);
    }
    return true;
  };

  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);
  bool Success;
  bool UsedAssumedInformation = false;
  Success = checkForAllInstructionsImpl(
      nullptr, OpcodeInstMap, CallSitePred, nullptr, nullptr,
      {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
       (unsigned)Instruction::Call},
      UsedAssumedInformation);
  (void)Success;
  assert(Success && "Expected the check call to be successful!");

  auto LoadStorePred = [&](Instruction &I) -> bool {
    if (isa<LoadInst>(I)) {
      getOrCreateAAFor<AAAlign>(
          IRPosition::value(*cast<LoadInst>(I).getPointerOperand()));
      if (SimplifyAllLoads)
        getAssumedSimplified(IRPosition::value(I), nullptr,
                             UsedAssumedInformation);
    } else {
      auto &SI = cast<StoreInst>(I);
      getOrCreateAAFor<AAIsDead>(IRPosition::inst(I));
      getAssumedSimplified(IRPosition::value(*SI.getValueOperand()), nullptr,
                           UsedAssumedInformation);
      getOrCreateAAFor<AAAlign>(IRPosition::value(*SI.getPointerOperand()));
    }
    return true;
  };
  Success = checkForAllInstructionsImpl(
      nullptr, OpcodeInstMap, LoadStorePred, nullptr, nullptr,
      {(unsigned)Instruction::Load, (unsigned)Instruction::Store},
      UsedAssumedInformation);
  (void)Success;
  assert(Success && "Expected the check call to be successful!");
}

/// Helpers to ease debugging through output streams and print calls.
///
///{
raw_ostream &llvm::operator<<(raw_ostream &OS, ChangeStatus S) {
  return OS << (S == ChangeStatus::CHANGED ? "changed" : "unchanged");
}

raw_ostream &llvm::operator<<(raw_ostream &OS, IRPosition::Kind AP) {
  switch (AP) {
  case IRPosition::IRP_INVALID:
    return OS << "inv";
  case IRPosition::IRP_FLOAT:
    return OS << "flt";
  case IRPosition::IRP_RETURNED:
    return OS << "fn_ret";
  case IRPosition::IRP_CALL_SITE_RETURNED:
    return OS << "cs_ret";
  case IRPosition::IRP_FUNCTION:
    return OS << "fn";
  case IRPosition::IRP_CALL_SITE:
    return OS << "cs";
  case IRPosition::IRP_ARGUMENT:
    return OS << "arg";
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
    return OS << "cs_arg";
  }
  llvm_unreachable("Unknown attribute position!");
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const IRPosition &Pos) {
  const Value &AV = Pos.getAssociatedValue();
  OS << "{" << Pos.getPositionKind() << ":" << AV.getName() << " ["
     << Pos.getAnchorValue().getName() << "@" << Pos.getCallSiteArgNo() << "]";

  if (Pos.hasCallBaseContext())
    OS << "[cb_context:" << *Pos.getCallBaseContext() << "]";
  return OS << "}";
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const IntegerRangeState &S) {
  OS << "range-state(" << S.getBitWidth() << ")<";
  S.getKnown().print(OS);
  OS << " / ";
  S.getAssumed().print(OS);
  OS << ">";

  return OS << static_cast<const AbstractState &>(S);
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractState &S) {
  return OS << (!S.isValidState() ? "top" : (S.isAtFixpoint() ? "fix" : ""));
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractAttribute &AA) {
  AA.print(OS);
  return OS;
}

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              const PotentialConstantIntValuesState &S) {
  OS << "set-state(< {";
  if (!S.isValidState())
    OS << "full-set";
  else {
    for (auto &It : S.getAssumedSet())
      OS << It << ", ";
    if (S.undefIsContained())
      OS << "undef ";
  }
  OS << "} >)";

  return OS;
}

void AbstractAttribute::print(raw_ostream &OS) const {
  OS << "[";
  OS << getName();
  OS << "] for CtxI ";

  if (auto *I = getCtxI()) {
    OS << "'";
    I->print(OS);
    OS << "'";
  } else
    OS << "<<null inst>>";

  OS << " at position " << getIRPosition() << " with state " << getAsStr()
     << '\n';
}

void AbstractAttribute::printWithDeps(raw_ostream &OS) const {
  print(OS);

  for (const auto &DepAA : Deps) {
    auto *AA = DepAA.getPointer();
    OS << "  updates ";
    AA->print(OS);
  }

  OS << '\n';
}

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              const AAPointerInfo::Access &Acc) {
  OS << " [" << Acc.getKind() << "] " << *Acc.getRemoteInst();
  if (Acc.getLocalInst() != Acc.getRemoteInst())
    OS << " via " << *Acc.getLocalInst();
  if (Acc.getContent()) {
    if (*Acc.getContent())
      OS << " [" << **Acc.getContent() << "]";
    else
      OS << " [ <unknown> ]";
  }
  return OS;
}
///}

/// ----------------------------------------------------------------------------
///                       Pass (Manager) Boilerplate
/// ----------------------------------------------------------------------------

static bool runAttributorOnFunctions(InformationCache &InfoCache,
                                     SetVector<Function *> &Functions,
                                     AnalysisGetter &AG,
                                     CallGraphUpdater &CGUpdater,
                                     bool DeleteFns, bool IsModulePass) {
  if (Functions.empty())
    return false;

  LLVM_DEBUG({
    dbgs() << "[Attributor] Run on module with " << Functions.size()
           << " functions:\n";
    for (Function *Fn : Functions)
      dbgs() << "  - " << Fn->getName() << "\n";
  });

  // Create an Attributor and initially empty information cache that is filled
  // while we identify default attribute opportunities.
  AttributorConfig AC(CGUpdater);
  AC.IsModulePass = IsModulePass;
  AC.DeleteFns = DeleteFns;
  Attributor A(Functions, InfoCache, AC);

  // Create shallow wrappers for all functions that are not IPO amendable
  if (AllowShallowWrappers)
    for (Function *F : Functions)
      if (!A.isFunctionIPOAmendable(*F))
        Attributor::createShallowWrapper(*F);

  // Internalize non-exact functions
  // TODO: for now we eagerly internalize functions without calculating the
  //       cost, we need a cost interface to determine whether internalizing
  //       a function is "benefitial"
  if (AllowDeepWrapper) {
    unsigned FunSize = Functions.size();
    for (unsigned u = 0; u < FunSize; u++) {
      Function *F = Functions[u];
      if (!F->isDeclaration() && !F->isDefinitionExact() && F->getNumUses() &&
          !GlobalValue::isInterposableLinkage(F->getLinkage())) {
        Function *NewF = Attributor::internalizeFunction(*F);
        assert(NewF && "Could not internalize function.");
        Functions.insert(NewF);

        // Update call graph
        CGUpdater.replaceFunctionWith(*F, *NewF);
        for (const Use &U : NewF->uses())
          if (CallBase *CB = dyn_cast<CallBase>(U.getUser())) {
            auto *CallerF = CB->getCaller();
            CGUpdater.reanalyzeFunction(*CallerF);
          }
      }
    }
  }

  for (Function *F : Functions) {
    if (F->hasExactDefinition())
      NumFnWithExactDefinition++;
    else
      NumFnWithoutExactDefinition++;

    // We look at internal functions only on-demand but if any use is not a
    // direct call or outside the current set of analyzed functions, we have
    // to do it eagerly.
    if (F->hasLocalLinkage()) {
      if (llvm::all_of(F->uses(), [&Functions](const Use &U) {
            const auto *CB = dyn_cast<CallBase>(U.getUser());
            return CB && CB->isCallee(&U) &&
                   Functions.count(const_cast<Function *>(CB->getCaller()));
          }))
        continue;
    }

    // Populate the Attributor with abstract attribute opportunities in the
    // function and the information cache with IR information.
    A.identifyDefaultAbstractAttributes(*F);
  }

  ChangeStatus Changed = A.run();

  LLVM_DEBUG(dbgs() << "[Attributor] Done with " << Functions.size()
                    << " functions, result: " << Changed << ".\n");
  return Changed == ChangeStatus::CHANGED;
}

void AADepGraph::viewGraph() { llvm::ViewGraph(this, "Dependency Graph"); }

void AADepGraph::dumpGraph() {
  static std::atomic<int> CallTimes;
  std::string Prefix;

  if (!DepGraphDotFileNamePrefix.empty())
    Prefix = DepGraphDotFileNamePrefix;
  else
    Prefix = "dep_graph";
  std::string Filename =
      Prefix + "_" + std::to_string(CallTimes.load()) + ".dot";

  outs() << "Dependency graph dump to " << Filename << ".\n";

  std::error_code EC;

  raw_fd_ostream File(Filename, EC, sys::fs::OF_TextWithCRLF);
  if (!EC)
    llvm::WriteGraph(File, this);

  CallTimes++;
}

void AADepGraph::print() {
  for (auto DepAA : SyntheticRoot.Deps)
    cast<AbstractAttribute>(DepAA.getPointer())->printWithDeps(outs());
}

PreservedAnalyses AttributorPass::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  AnalysisGetter AG(FAM);

  SetVector<Function *> Functions;
  for (Function &F : M)
    Functions.insert(&F);

  CallGraphUpdater CGUpdater;
  BumpPtrAllocator Allocator;
  InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ nullptr);
  if (runAttributorOnFunctions(InfoCache, Functions, AG, CGUpdater,
                               /* DeleteFns */ true, /* IsModulePass */ true)) {
    // FIXME: Think about passes we will preserve and add them here.
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}

PreservedAnalyses AttributorCGSCCPass::run(LazyCallGraph::SCC &C,
                                           CGSCCAnalysisManager &AM,
                                           LazyCallGraph &CG,
                                           CGSCCUpdateResult &UR) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();
  AnalysisGetter AG(FAM);

  SetVector<Function *> Functions;
  for (LazyCallGraph::Node &N : C)
    Functions.insert(&N.getFunction());

  if (Functions.empty())
    return PreservedAnalyses::all();

  Module &M = *Functions.back()->getParent();
  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);
  BumpPtrAllocator Allocator;
  InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ &Functions);
  if (runAttributorOnFunctions(InfoCache, Functions, AG, CGUpdater,
                               /* DeleteFns */ false,
                               /* IsModulePass */ false)) {
    // FIXME: Think about passes we will preserve and add them here.
    PreservedAnalyses PA;
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    return PA;
  }
  return PreservedAnalyses::all();
}

namespace llvm {

template <> struct GraphTraits<AADepGraphNode *> {
  using NodeRef = AADepGraphNode *;
  using DepTy = PointerIntPair<AADepGraphNode *, 1>;
  using EdgeRef = PointerIntPair<AADepGraphNode *, 1>;

  static NodeRef getEntryNode(AADepGraphNode *DGN) { return DGN; }
  static NodeRef DepGetVal(DepTy &DT) { return DT.getPointer(); }

  using ChildIteratorType =
      mapped_iterator<TinyPtrVector<DepTy>::iterator, decltype(&DepGetVal)>;
  using ChildEdgeIteratorType = TinyPtrVector<DepTy>::iterator;

  static ChildIteratorType child_begin(NodeRef N) { return N->child_begin(); }

  static ChildIteratorType child_end(NodeRef N) { return N->child_end(); }
};

template <>
struct GraphTraits<AADepGraph *> : public GraphTraits<AADepGraphNode *> {
  static NodeRef getEntryNode(AADepGraph *DG) { return DG->GetEntryNode(); }

  using nodes_iterator =
      mapped_iterator<TinyPtrVector<DepTy>::iterator, decltype(&DepGetVal)>;

  static nodes_iterator nodes_begin(AADepGraph *DG) { return DG->begin(); }

  static nodes_iterator nodes_end(AADepGraph *DG) { return DG->end(); }
};

template <> struct DOTGraphTraits<AADepGraph *> : public DefaultDOTGraphTraits {
  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  static std::string getNodeLabel(const AADepGraphNode *Node,
                                  const AADepGraph *DG) {
    std::string AAString;
    raw_string_ostream O(AAString);
    Node->print(O);
    return AAString;
  }
};

} // end namespace llvm

namespace {

struct AttributorLegacyPass : public ModulePass {
  static char ID;

  AttributorLegacyPass() : ModulePass(ID) {
    initializeAttributorLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;

    AnalysisGetter AG;
    SetVector<Function *> Functions;
    for (Function &F : M)
      Functions.insert(&F);

    CallGraphUpdater CGUpdater;
    BumpPtrAllocator Allocator;
    InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ nullptr);
    return runAttributorOnFunctions(InfoCache, Functions, AG, CGUpdater,
                                    /* DeleteFns*/ true,
                                    /* IsModulePass */ true);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // FIXME: Think about passes we will preserve and add them here.
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};

struct AttributorCGSCCLegacyPass : public CallGraphSCCPass {
  static char ID;

  AttributorCGSCCLegacyPass() : CallGraphSCCPass(ID) {
    initializeAttributorCGSCCLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnSCC(CallGraphSCC &SCC) override {
    if (skipSCC(SCC))
      return false;

    SetVector<Function *> Functions;
    for (CallGraphNode *CGN : SCC)
      if (Function *Fn = CGN->getFunction())
        if (!Fn->isDeclaration())
          Functions.insert(Fn);

    if (Functions.empty())
      return false;

    AnalysisGetter AG;
    CallGraph &CG = const_cast<CallGraph &>(SCC.getCallGraph());
    CallGraphUpdater CGUpdater;
    CGUpdater.initialize(CG, SCC);
    Module &M = *Functions.back()->getParent();
    BumpPtrAllocator Allocator;
    InformationCache InfoCache(M, AG, Allocator, /* CGSCC */ &Functions);
    return runAttributorOnFunctions(InfoCache, Functions, AG, CGUpdater,
                                    /* DeleteFns */ false,
                                    /* IsModulePass */ false);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // FIXME: Think about passes we will preserve and add them here.
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    CallGraphSCCPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

Pass *llvm::createAttributorLegacyPass() { return new AttributorLegacyPass(); }
Pass *llvm::createAttributorCGSCCLegacyPass() {
  return new AttributorCGSCCLegacyPass();
}

char AttributorLegacyPass::ID = 0;
char AttributorCGSCCLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(AttributorLegacyPass, "attributor",
                      "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AttributorLegacyPass, "attributor",
                    "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_BEGIN(AttributorCGSCCLegacyPass, "attributor-cgscc",
                      "Deduce and propagate attributes (CGSCC pass)", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(AttributorCGSCCLegacyPass, "attributor-cgscc",
                    "Deduce and propagate attributes (CGSCC pass)", false,
                    false)
