//===- AttributorAttributes.cpp - Attributes for Attributor deduction -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See the Attributor.h file comment and the class descriptions in that file for
// more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Attributor.h"

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/Utils/Local.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "attributor"

static cl::opt<bool> ManifestInternal(
    "attributor-manifest-internal", cl::Hidden,
    cl::desc("Manifest Attributor internal string attributes."),
    cl::init(false));

static cl::opt<int> MaxHeapToStackSize("max-heap-to-stack-size", cl::init(128),
                                       cl::Hidden);

template <>
unsigned llvm::PotentialConstantIntValuesState::MaxPotentialValues = 0;

static cl::opt<unsigned, true> MaxPotentialValues(
    "attributor-max-potential-values", cl::Hidden,
    cl::desc("Maximum number of potential values to be "
             "tracked for each position."),
    cl::location(llvm::PotentialConstantIntValuesState::MaxPotentialValues),
    cl::init(7));

STATISTIC(NumAAs, "Number of abstract attributes created");

// Some helper macros to deal with statistics tracking.
//
// Usage:
// For simple IR attribute tracking overload trackStatistics in the abstract
// attribute and choose the right STATS_DECLTRACK_********* macro,
// e.g.,:
//  void trackStatistics() const override {
//    STATS_DECLTRACK_ARG_ATTR(returned)
//  }
// If there is a single "increment" side one can use the macro
// STATS_DECLTRACK with a custom message. If there are multiple increment
// sides, STATS_DECL and STATS_TRACK can also be used separately.
//
#define BUILD_STAT_MSG_IR_ATTR(TYPE, NAME)                                     \
  ("Number of " #TYPE " marked '" #NAME "'")
#define BUILD_STAT_NAME(NAME, TYPE) NumIR##TYPE##_##NAME
#define STATS_DECL_(NAME, MSG) STATISTIC(NAME, MSG);
#define STATS_DECL(NAME, TYPE, MSG)                                            \
  STATS_DECL_(BUILD_STAT_NAME(NAME, TYPE), MSG);
#define STATS_TRACK(NAME, TYPE) ++(BUILD_STAT_NAME(NAME, TYPE));
#define STATS_DECLTRACK(NAME, TYPE, MSG)                                       \
  {                                                                            \
    STATS_DECL(NAME, TYPE, MSG)                                                \
    STATS_TRACK(NAME, TYPE)                                                    \
  }
#define STATS_DECLTRACK_ARG_ATTR(NAME)                                         \
  STATS_DECLTRACK(NAME, Arguments, BUILD_STAT_MSG_IR_ATTR(arguments, NAME))
#define STATS_DECLTRACK_CSARG_ATTR(NAME)                                       \
  STATS_DECLTRACK(NAME, CSArguments,                                           \
                  BUILD_STAT_MSG_IR_ATTR(call site arguments, NAME))
#define STATS_DECLTRACK_FN_ATTR(NAME)                                          \
  STATS_DECLTRACK(NAME, Function, BUILD_STAT_MSG_IR_ATTR(functions, NAME))
#define STATS_DECLTRACK_CS_ATTR(NAME)                                          \
  STATS_DECLTRACK(NAME, CS, BUILD_STAT_MSG_IR_ATTR(call site, NAME))
#define STATS_DECLTRACK_FNRET_ATTR(NAME)                                       \
  STATS_DECLTRACK(NAME, FunctionReturn,                                        \
                  BUILD_STAT_MSG_IR_ATTR(function returns, NAME))
#define STATS_DECLTRACK_CSRET_ATTR(NAME)                                       \
  STATS_DECLTRACK(NAME, CSReturn,                                              \
                  BUILD_STAT_MSG_IR_ATTR(call site returns, NAME))
#define STATS_DECLTRACK_FLOATING_ATTR(NAME)                                    \
  STATS_DECLTRACK(NAME, Floating,                                              \
                  ("Number of floating values known to be '" #NAME "'"))

// Specialization of the operator<< for abstract attributes subclasses. This
// disambiguates situations where multiple operators are applicable.
namespace llvm {
#define PIPE_OPERATOR(CLASS)                                                   \
  raw_ostream &operator<<(raw_ostream &OS, const CLASS &AA) {                  \
    return OS << static_cast<const AbstractAttribute &>(AA);                   \
  }

PIPE_OPERATOR(AAIsDead)
PIPE_OPERATOR(AANoUnwind)
PIPE_OPERATOR(AANoSync)
PIPE_OPERATOR(AANoRecurse)
PIPE_OPERATOR(AAWillReturn)
PIPE_OPERATOR(AANoReturn)
PIPE_OPERATOR(AAReturnedValues)
PIPE_OPERATOR(AANonNull)
PIPE_OPERATOR(AANoAlias)
PIPE_OPERATOR(AADereferenceable)
PIPE_OPERATOR(AAAlign)
PIPE_OPERATOR(AANoCapture)
PIPE_OPERATOR(AAValueSimplify)
PIPE_OPERATOR(AANoFree)
PIPE_OPERATOR(AAHeapToStack)
PIPE_OPERATOR(AAReachability)
PIPE_OPERATOR(AAMemoryBehavior)
PIPE_OPERATOR(AAMemoryLocation)
PIPE_OPERATOR(AAValueConstantRange)
PIPE_OPERATOR(AAPrivatizablePtr)
PIPE_OPERATOR(AAUndefinedBehavior)
PIPE_OPERATOR(AAPotentialValues)
PIPE_OPERATOR(AANoUndef)

#undef PIPE_OPERATOR
} // namespace llvm

namespace {

static Optional<ConstantInt *>
getAssumedConstantInt(Attributor &A, const Value &V,
                      const AbstractAttribute &AA,
                      bool &UsedAssumedInformation) {
  Optional<Constant *> C = A.getAssumedConstant(V, AA, UsedAssumedInformation);
  if (C.hasValue())
    return dyn_cast_or_null<ConstantInt>(C.getValue());
  return llvm::None;
}

/// Get pointer operand of memory accessing instruction. If \p I is
/// not a memory accessing instruction, return nullptr. If \p AllowVolatile,
/// is set to false and the instruction is volatile, return nullptr.
static const Value *getPointerOperand(const Instruction *I,
                                      bool AllowVolatile) {
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    if (!AllowVolatile && LI->isVolatile())
      return nullptr;
    return LI->getPointerOperand();
  }

  if (auto *SI = dyn_cast<StoreInst>(I)) {
    if (!AllowVolatile && SI->isVolatile())
      return nullptr;
    return SI->getPointerOperand();
  }

  if (auto *CXI = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!AllowVolatile && CXI->isVolatile())
      return nullptr;
    return CXI->getPointerOperand();
  }

  if (auto *RMWI = dyn_cast<AtomicRMWInst>(I)) {
    if (!AllowVolatile && RMWI->isVolatile())
      return nullptr;
    return RMWI->getPointerOperand();
  }

  return nullptr;
}

/// Helper function to create a pointer of type \p ResTy, based on \p Ptr, and
/// advanced by \p Offset bytes. To aid later analysis the method tries to build
/// getelement pointer instructions that traverse the natural type of \p Ptr if
/// possible. If that fails, the remaining offset is adjusted byte-wise, hence
/// through a cast to i8*.
///
/// TODO: This could probably live somewhere more prominantly if it doesn't
///       already exist.
static Value *constructPointer(Type *ResTy, Value *Ptr, int64_t Offset,
                               IRBuilder<NoFolder> &IRB, const DataLayout &DL) {
  assert(Offset >= 0 && "Negative offset not supported yet!");
  LLVM_DEBUG(dbgs() << "Construct pointer: " << *Ptr << " + " << Offset
                    << "-bytes as " << *ResTy << "\n");

  // The initial type we are trying to traverse to get nice GEPs.
  Type *Ty = Ptr->getType();

  SmallVector<Value *, 4> Indices;
  std::string GEPName = Ptr->getName().str();
  while (Offset) {
    uint64_t Idx, Rem;

    if (auto *STy = dyn_cast<StructType>(Ty)) {
      const StructLayout *SL = DL.getStructLayout(STy);
      if (int64_t(SL->getSizeInBytes()) < Offset)
        break;
      Idx = SL->getElementContainingOffset(Offset);
      assert(Idx < STy->getNumElements() && "Offset calculation error!");
      Rem = Offset - SL->getElementOffset(Idx);
      Ty = STy->getElementType(Idx);
    } else if (auto *PTy = dyn_cast<PointerType>(Ty)) {
      Ty = PTy->getElementType();
      if (!Ty->isSized())
        break;
      uint64_t ElementSize = DL.getTypeAllocSize(Ty);
      assert(ElementSize && "Expected type with size!");
      Idx = Offset / ElementSize;
      Rem = Offset % ElementSize;
    } else {
      // Non-aggregate type, we cast and make byte-wise progress now.
      break;
    }

    LLVM_DEBUG(errs() << "Ty: " << *Ty << " Offset: " << Offset
                      << " Idx: " << Idx << " Rem: " << Rem << "\n");

    GEPName += "." + std::to_string(Idx);
    Indices.push_back(ConstantInt::get(IRB.getInt32Ty(), Idx));
    Offset = Rem;
  }

  // Create a GEP if we collected indices above.
  if (Indices.size())
    Ptr = IRB.CreateGEP(Ptr, Indices, GEPName);

  // If an offset is left we use byte-wise adjustment.
  if (Offset) {
    Ptr = IRB.CreateBitCast(Ptr, IRB.getInt8PtrTy());
    Ptr = IRB.CreateGEP(Ptr, IRB.getInt32(Offset),
                        GEPName + ".b" + Twine(Offset));
  }

  // Ensure the result has the requested type.
  Ptr = IRB.CreateBitOrPointerCast(Ptr, ResTy, Ptr->getName() + ".cast");

  LLVM_DEBUG(dbgs() << "Constructed pointer: " << *Ptr << "\n");
  return Ptr;
}

/// Recursively visit all values that might become \p IRP at some point. This
/// will be done by looking through cast instructions, selects, phis, and calls
/// with the "returned" attribute. Once we cannot look through the value any
/// further, the callback \p VisitValueCB is invoked and passed the current
/// value, the \p State, and a flag to indicate if we stripped anything.
/// Stripped means that we unpacked the value associated with \p IRP at least
/// once. Note that the value used for the callback may still be the value
/// associated with \p IRP (due to PHIs). To limit how much effort is invested,
/// we will never visit more values than specified by \p MaxValues.
template <typename AAType, typename StateTy>
static bool genericValueTraversal(
    Attributor &A, IRPosition IRP, const AAType &QueryingAA, StateTy &State,
    function_ref<bool(Value &, const Instruction *, StateTy &, bool)>
        VisitValueCB,
    const Instruction *CtxI, bool UseValueSimplify = true, int MaxValues = 16,
    function_ref<Value *(Value *)> StripCB = nullptr) {

  const AAIsDead *LivenessAA = nullptr;
  if (IRP.getAnchorScope())
    LivenessAA = &A.getAAFor<AAIsDead>(
        QueryingAA, IRPosition::function(*IRP.getAnchorScope()),
        DepClassTy::NONE);
  bool AnyDead = false;

  using Item = std::pair<Value *, const Instruction *>;
  SmallSet<Item, 16> Visited;
  SmallVector<Item, 16> Worklist;
  Worklist.push_back({&IRP.getAssociatedValue(), CtxI});

  int Iteration = 0;
  do {
    Item I = Worklist.pop_back_val();
    Value *V = I.first;
    CtxI = I.second;
    if (StripCB)
      V = StripCB(V);

    // Check if we should process the current value. To prevent endless
    // recursion keep a record of the values we followed!
    if (!Visited.insert(I).second)
      continue;

    // Make sure we limit the compile time for complex expressions.
    if (Iteration++ >= MaxValues)
      return false;

    // Explicitly look through calls with a "returned" attribute if we do
    // not have a pointer as stripPointerCasts only works on them.
    Value *NewV = nullptr;
    if (V->getType()->isPointerTy()) {
      NewV = V->stripPointerCasts();
    } else {
      auto *CB = dyn_cast<CallBase>(V);
      if (CB && CB->getCalledFunction()) {
        for (Argument &Arg : CB->getCalledFunction()->args())
          if (Arg.hasReturnedAttr()) {
            NewV = CB->getArgOperand(Arg.getArgNo());
            break;
          }
      }
    }
    if (NewV && NewV != V) {
      Worklist.push_back({NewV, CtxI});
      continue;
    }

    // Look through select instructions, visit both potential values.
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      Worklist.push_back({SI->getTrueValue(), CtxI});
      Worklist.push_back({SI->getFalseValue(), CtxI});
      continue;
    }

    // Look through phi nodes, visit all live operands.
    if (auto *PHI = dyn_cast<PHINode>(V)) {
      assert(LivenessAA &&
             "Expected liveness in the presence of instructions!");
      for (unsigned u = 0, e = PHI->getNumIncomingValues(); u < e; u++) {
        BasicBlock *IncomingBB = PHI->getIncomingBlock(u);
        if (A.isAssumedDead(*IncomingBB->getTerminator(), &QueryingAA,
                            LivenessAA,
                            /* CheckBBLivenessOnly */ true)) {
          AnyDead = true;
          continue;
        }
        Worklist.push_back(
            {PHI->getIncomingValue(u), IncomingBB->getTerminator()});
      }
      continue;
    }

    if (UseValueSimplify && !isa<Constant>(V)) {
      bool UsedAssumedInformation = false;
      Optional<Constant *> C =
          A.getAssumedConstant(*V, QueryingAA, UsedAssumedInformation);
      if (!C.hasValue())
        continue;
      if (Value *NewV = C.getValue()) {
        Worklist.push_back({NewV, CtxI});
        continue;
      }
    }

    // Once a leaf is reached we inform the user through the callback.
    if (!VisitValueCB(*V, CtxI, State, Iteration > 1))
      return false;
  } while (!Worklist.empty());

  // If we actually used liveness information so we have to record a dependence.
  if (AnyDead)
    A.recordDependence(*LivenessAA, QueryingAA, DepClassTy::OPTIONAL);

  // All values have been visited.
  return true;
}

const Value *stripAndAccumulateMinimalOffsets(
    Attributor &A, const AbstractAttribute &QueryingAA, const Value *Val,
    const DataLayout &DL, APInt &Offset, bool AllowNonInbounds,
    bool UseAssumed = false) {

  auto AttributorAnalysis = [&](Value &V, APInt &ROffset) -> bool {
    const IRPosition &Pos = IRPosition::value(V);
    // Only track dependence if we are going to use the assumed info.
    const AAValueConstantRange &ValueConstantRangeAA =
        A.getAAFor<AAValueConstantRange>(QueryingAA, Pos,
                                         UseAssumed ? DepClassTy::OPTIONAL
                                                    : DepClassTy::NONE);
    ConstantRange Range = UseAssumed ? ValueConstantRangeAA.getAssumed()
                                     : ValueConstantRangeAA.getKnown();
    // We can only use the lower part of the range because the upper part can
    // be higher than what the value can really be.
    ROffset = Range.getSignedMin();
    return true;
  };

  return Val->stripAndAccumulateConstantOffsets(DL, Offset, AllowNonInbounds,
                                                AttributorAnalysis);
}

static const Value *getMinimalBaseOfAccsesPointerOperand(
    Attributor &A, const AbstractAttribute &QueryingAA, const Instruction *I,
    int64_t &BytesOffset, const DataLayout &DL, bool AllowNonInbounds = false) {
  const Value *Ptr = getPointerOperand(I, /* AllowVolatile */ false);
  if (!Ptr)
    return nullptr;
  APInt OffsetAPInt(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
  const Value *Base = stripAndAccumulateMinimalOffsets(
      A, QueryingAA, Ptr, DL, OffsetAPInt, AllowNonInbounds);

  BytesOffset = OffsetAPInt.getSExtValue();
  return Base;
}

static const Value *
getBasePointerOfAccessPointerOperand(const Instruction *I, int64_t &BytesOffset,
                                     const DataLayout &DL,
                                     bool AllowNonInbounds = false) {
  const Value *Ptr = getPointerOperand(I, /* AllowVolatile */ false);
  if (!Ptr)
    return nullptr;

  return GetPointerBaseWithConstantOffset(Ptr, BytesOffset, DL,
                                          AllowNonInbounds);
}

/// Helper function to clamp a state \p S of type \p StateType with the
/// information in \p R and indicate/return if \p S did change (as-in update is
/// required to be run again).
template <typename StateType>
ChangeStatus clampStateAndIndicateChange(StateType &S, const StateType &R) {
  auto Assumed = S.getAssumed();
  S ^= R;
  return Assumed == S.getAssumed() ? ChangeStatus::UNCHANGED
                                   : ChangeStatus::CHANGED;
}

/// Clamp the information known for all returned values of a function
/// (identified by \p QueryingAA) into \p S.
template <typename AAType, typename StateType = typename AAType::StateType>
static void clampReturnedValueStates(
    Attributor &A, const AAType &QueryingAA, StateType &S,
    const IRPosition::CallBaseContext *CBContext = nullptr) {
  LLVM_DEBUG(dbgs() << "[Attributor] Clamp return value states for "
                    << QueryingAA << " into " << S << "\n");

  assert((QueryingAA.getIRPosition().getPositionKind() ==
              IRPosition::IRP_RETURNED ||
          QueryingAA.getIRPosition().getPositionKind() ==
              IRPosition::IRP_CALL_SITE_RETURNED) &&
         "Can only clamp returned value states for a function returned or call "
         "site returned position!");

  // Use an optional state as there might not be any return values and we want
  // to join (IntegerState::operator&) the state of all there are.
  Optional<StateType> T;

  // Callback for each possibly returned value.
  auto CheckReturnValue = [&](Value &RV) -> bool {
    const IRPosition &RVPos = IRPosition::value(RV, CBContext);
    const AAType &AA =
        A.getAAFor<AAType>(QueryingAA, RVPos, DepClassTy::REQUIRED);
    LLVM_DEBUG(dbgs() << "[Attributor] RV: " << RV << " AA: " << AA.getAsStr()
                      << " @ " << RVPos << "\n");
    const StateType &AAS = AA.getState();
    if (T.hasValue())
      *T &= AAS;
    else
      T = AAS;
    LLVM_DEBUG(dbgs() << "[Attributor] AA State: " << AAS << " RV State: " << T
                      << "\n");
    return T->isValidState();
  };

  if (!A.checkForAllReturnedValues(CheckReturnValue, QueryingAA))
    S.indicatePessimisticFixpoint();
  else if (T.hasValue())
    S ^= *T;
}

/// Helper class for generic deduction: return value -> returned position.
template <typename AAType, typename BaseType,
          typename StateType = typename BaseType::StateType,
          bool PropagateCallBaseContext = false>
struct AAReturnedFromReturnedValues : public BaseType {
  AAReturnedFromReturnedValues(const IRPosition &IRP, Attributor &A)
      : BaseType(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    StateType S(StateType::getBestState(this->getState()));
    clampReturnedValueStates<AAType, StateType>(
        A, *this, S,
        PropagateCallBaseContext ? this->getCallBaseContext() : nullptr);
    // TODO: If we know we visited all returned values, thus no are assumed
    // dead, we can take the known information from the state T.
    return clampStateAndIndicateChange<StateType>(this->getState(), S);
  }
};

/// Clamp the information known at all call sites for a given argument
/// (identified by \p QueryingAA) into \p S.
template <typename AAType, typename StateType = typename AAType::StateType>
static void clampCallSiteArgumentStates(Attributor &A, const AAType &QueryingAA,
                                        StateType &S) {
  LLVM_DEBUG(dbgs() << "[Attributor] Clamp call site argument states for "
                    << QueryingAA << " into " << S << "\n");

  assert(QueryingAA.getIRPosition().getPositionKind() ==
             IRPosition::IRP_ARGUMENT &&
         "Can only clamp call site argument states for an argument position!");

  // Use an optional state as there might not be any return values and we want
  // to join (IntegerState::operator&) the state of all there are.
  Optional<StateType> T;

  // The argument number which is also the call site argument number.
  unsigned ArgNo = QueryingAA.getIRPosition().getCallSiteArgNo();

  auto CallSiteCheck = [&](AbstractCallSite ACS) {
    const IRPosition &ACSArgPos = IRPosition::callsite_argument(ACS, ArgNo);
    // Check if a coresponding argument was found or if it is on not associated
    // (which can happen for callback calls).
    if (ACSArgPos.getPositionKind() == IRPosition::IRP_INVALID)
      return false;

    const AAType &AA =
        A.getAAFor<AAType>(QueryingAA, ACSArgPos, DepClassTy::REQUIRED);
    LLVM_DEBUG(dbgs() << "[Attributor] ACS: " << *ACS.getInstruction()
                      << " AA: " << AA.getAsStr() << " @" << ACSArgPos << "\n");
    const StateType &AAS = AA.getState();
    if (T.hasValue())
      *T &= AAS;
    else
      T = AAS;
    LLVM_DEBUG(dbgs() << "[Attributor] AA State: " << AAS << " CSA State: " << T
                      << "\n");
    return T->isValidState();
  };

  bool AllCallSitesKnown;
  if (!A.checkForAllCallSites(CallSiteCheck, QueryingAA, true,
                              AllCallSitesKnown))
    S.indicatePessimisticFixpoint();
  else if (T.hasValue())
    S ^= *T;
}

/// This function is the bridge between argument position and the call base
/// context.
template <typename AAType, typename BaseType,
          typename StateType = typename AAType::StateType>
bool getArgumentStateFromCallBaseContext(Attributor &A,
                                         BaseType &QueryingAttribute,
                                         IRPosition &Pos, StateType &State) {
  assert((Pos.getPositionKind() == IRPosition::IRP_ARGUMENT) &&
         "Expected an 'argument' position !");
  const CallBase *CBContext = Pos.getCallBaseContext();
  if (!CBContext)
    return false;

  int ArgNo = Pos.getCallSiteArgNo();
  assert(ArgNo >= 0 && "Invalid Arg No!");

  const auto &AA = A.getAAFor<AAType>(
      QueryingAttribute, IRPosition::callsite_argument(*CBContext, ArgNo),
      DepClassTy::REQUIRED);
  const StateType &CBArgumentState =
      static_cast<const StateType &>(AA.getState());

  LLVM_DEBUG(dbgs() << "[Attributor] Briding Call site context to argument"
                    << "Position:" << Pos << "CB Arg state:" << CBArgumentState
                    << "\n");

  // NOTE: If we want to do call site grouping it should happen here.
  State ^= CBArgumentState;
  return true;
}

/// Helper class for generic deduction: call site argument -> argument position.
template <typename AAType, typename BaseType,
          typename StateType = typename AAType::StateType,
          bool BridgeCallBaseContext = false>
struct AAArgumentFromCallSiteArguments : public BaseType {
  AAArgumentFromCallSiteArguments(const IRPosition &IRP, Attributor &A)
      : BaseType(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    StateType S = StateType::getBestState(this->getState());

    if (BridgeCallBaseContext) {
      bool Success =
          getArgumentStateFromCallBaseContext<AAType, BaseType, StateType>(
              A, *this, this->getIRPosition(), S);
      if (Success)
        return clampStateAndIndicateChange<StateType>(this->getState(), S);
    }
    clampCallSiteArgumentStates<AAType, StateType>(A, *this, S);

    // TODO: If we know we visited all incoming values, thus no are assumed
    // dead, we can take the known information from the state T.
    return clampStateAndIndicateChange<StateType>(this->getState(), S);
  }
};

/// Helper class for generic replication: function returned -> cs returned.
template <typename AAType, typename BaseType,
          typename StateType = typename BaseType::StateType,
          bool IntroduceCallBaseContext = false>
struct AACallSiteReturnedFromReturned : public BaseType {
  AACallSiteReturnedFromReturned(const IRPosition &IRP, Attributor &A)
      : BaseType(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    assert(this->getIRPosition().getPositionKind() ==
               IRPosition::IRP_CALL_SITE_RETURNED &&
           "Can only wrap function returned positions for call site returned "
           "positions!");
    auto &S = this->getState();

    const Function *AssociatedFunction =
        this->getIRPosition().getAssociatedFunction();
    if (!AssociatedFunction)
      return S.indicatePessimisticFixpoint();

    CallBase &CBContext = static_cast<CallBase &>(this->getAnchorValue());
    if (IntroduceCallBaseContext)
      LLVM_DEBUG(dbgs() << "[Attributor] Introducing call base context:"
                        << CBContext << "\n");

    IRPosition FnPos = IRPosition::returned(
        *AssociatedFunction, IntroduceCallBaseContext ? &CBContext : nullptr);
    const AAType &AA = A.getAAFor<AAType>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(S, AA.getState());
  }
};

/// Helper function to accumulate uses.
template <class AAType, typename StateType = typename AAType::StateType>
static void followUsesInContext(AAType &AA, Attributor &A,
                                MustBeExecutedContextExplorer &Explorer,
                                const Instruction *CtxI,
                                SetVector<const Use *> &Uses,
                                StateType &State) {
  auto EIt = Explorer.begin(CtxI), EEnd = Explorer.end(CtxI);
  for (unsigned u = 0; u < Uses.size(); ++u) {
    const Use *U = Uses[u];
    if (const Instruction *UserI = dyn_cast<Instruction>(U->getUser())) {
      bool Found = Explorer.findInContextOf(UserI, EIt, EEnd);
      if (Found && AA.followUseInMBEC(A, U, UserI, State))
        for (const Use &Us : UserI->uses())
          Uses.insert(&Us);
    }
  }
}

/// Use the must-be-executed-context around \p I to add information into \p S.
/// The AAType class is required to have `followUseInMBEC` method with the
/// following signature and behaviour:
///
/// bool followUseInMBEC(Attributor &A, const Use *U, const Instruction *I)
/// U - Underlying use.
/// I - The user of the \p U.
/// Returns true if the value should be tracked transitively.
///
template <class AAType, typename StateType = typename AAType::StateType>
static void followUsesInMBEC(AAType &AA, Attributor &A, StateType &S,
                             Instruction &CtxI) {

  // Container for (transitive) uses of the associated value.
  SetVector<const Use *> Uses;
  for (const Use &U : AA.getIRPosition().getAssociatedValue().uses())
    Uses.insert(&U);

  MustBeExecutedContextExplorer &Explorer =
      A.getInfoCache().getMustBeExecutedContextExplorer();

  followUsesInContext<AAType>(AA, A, Explorer, &CtxI, Uses, S);

  if (S.isAtFixpoint())
    return;

  SmallVector<const BranchInst *, 4> BrInsts;
  auto Pred = [&](const Instruction *I) {
    if (const BranchInst *Br = dyn_cast<BranchInst>(I))
      if (Br->isConditional())
        BrInsts.push_back(Br);
    return true;
  };

  // Here, accumulate conditional branch instructions in the context. We
  // explore the child paths and collect the known states. The disjunction of
  // those states can be merged to its own state. Let ParentState_i be a state
  // to indicate the known information for an i-th branch instruction in the
  // context. ChildStates are created for its successors respectively.
  //
  // ParentS_1 = ChildS_{1, 1} /\ ChildS_{1, 2} /\ ... /\ ChildS_{1, n_1}
  // ParentS_2 = ChildS_{2, 1} /\ ChildS_{2, 2} /\ ... /\ ChildS_{2, n_2}
  //      ...
  // ParentS_m = ChildS_{m, 1} /\ ChildS_{m, 2} /\ ... /\ ChildS_{m, n_m}
  //
  // Known State |= ParentS_1 \/ ParentS_2 \/... \/ ParentS_m
  //
  // FIXME: Currently, recursive branches are not handled. For example, we
  // can't deduce that ptr must be dereferenced in below function.
  //
  // void f(int a, int c, int *ptr) {
  //    if(a)
  //      if (b) {
  //        *ptr = 0;
  //      } else {
  //        *ptr = 1;
  //      }
  //    else {
  //      if (b) {
  //        *ptr = 0;
  //      } else {
  //        *ptr = 1;
  //      }
  //    }
  // }

  Explorer.checkForAllContext(&CtxI, Pred);
  for (const BranchInst *Br : BrInsts) {
    StateType ParentState;

    // The known state of the parent state is a conjunction of children's
    // known states so it is initialized with a best state.
    ParentState.indicateOptimisticFixpoint();

    for (const BasicBlock *BB : Br->successors()) {
      StateType ChildState;

      size_t BeforeSize = Uses.size();
      followUsesInContext(AA, A, Explorer, &BB->front(), Uses, ChildState);

      // Erase uses which only appear in the child.
      for (auto It = Uses.begin() + BeforeSize; It != Uses.end();)
        It = Uses.erase(It);

      ParentState &= ChildState;
    }

    // Use only known state.
    S += ParentState;
  }
}

/// -----------------------NoUnwind Function Attribute--------------------------

struct AANoUnwindImpl : AANoUnwind {
  AANoUnwindImpl(const IRPosition &IRP, Attributor &A) : AANoUnwind(IRP, A) {}

  const std::string getAsStr() const override {
    return getAssumed() ? "nounwind" : "may-unwind";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto Opcodes = {
        (unsigned)Instruction::Invoke,      (unsigned)Instruction::CallBr,
        (unsigned)Instruction::Call,        (unsigned)Instruction::CleanupRet,
        (unsigned)Instruction::CatchSwitch, (unsigned)Instruction::Resume};

    auto CheckForNoUnwind = [&](Instruction &I) {
      if (!I.mayThrow())
        return true;

      if (const auto *CB = dyn_cast<CallBase>(&I)) {
        const auto &NoUnwindAA = A.getAAFor<AANoUnwind>(
            *this, IRPosition::callsite_function(*CB), DepClassTy::REQUIRED);
        return NoUnwindAA.isAssumedNoUnwind();
      }
      return false;
    };

    if (!A.checkForAllInstructions(CheckForNoUnwind, *this, Opcodes))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }
};

struct AANoUnwindFunction final : public AANoUnwindImpl {
  AANoUnwindFunction(const IRPosition &IRP, Attributor &A)
      : AANoUnwindImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(nounwind) }
};

/// NoUnwind attribute deduction for a call sites.
struct AANoUnwindCallSite final : AANoUnwindImpl {
  AANoUnwindCallSite(const IRPosition &IRP, Attributor &A)
      : AANoUnwindImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoUnwindImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<AANoUnwind>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(nounwind); }
};

/// --------------------- Function Return Values -------------------------------

/// "Attribute" that collects all potential returned values and the return
/// instructions that they arise from.
///
/// If there is a unique returned value R, the manifest method will:
///   - mark R with the "returned" attribute, if R is an argument.
class AAReturnedValuesImpl : public AAReturnedValues, public AbstractState {

  /// Mapping of values potentially returned by the associated function to the
  /// return instructions that might return them.
  MapVector<Value *, SmallSetVector<ReturnInst *, 4>> ReturnedValues;

  /// Mapping to remember the number of returned values for a call site such
  /// that we can avoid updates if nothing changed.
  DenseMap<const CallBase *, unsigned> NumReturnedValuesPerKnownAA;

  /// Set of unresolved calls returned by the associated function.
  SmallSetVector<CallBase *, 4> UnresolvedCalls;

  /// State flags
  ///
  ///{
  bool IsFixed = false;
  bool IsValidState = true;
  ///}

public:
  AAReturnedValuesImpl(const IRPosition &IRP, Attributor &A)
      : AAReturnedValues(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // Reset the state.
    IsFixed = false;
    IsValidState = true;
    ReturnedValues.clear();

    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration()) {
      indicatePessimisticFixpoint();
      return;
    }
    assert(!F->getReturnType()->isVoidTy() &&
           "Did not expect a void return type!");

    // The map from instruction opcodes to those instructions in the function.
    auto &OpcodeInstMap = A.getInfoCache().getOpcodeInstMapForFunction(*F);

    // Look through all arguments, if one is marked as returned we are done.
    for (Argument &Arg : F->args()) {
      if (Arg.hasReturnedAttr()) {
        auto &ReturnInstSet = ReturnedValues[&Arg];
        if (auto *Insts = OpcodeInstMap.lookup(Instruction::Ret))
          for (Instruction *RI : *Insts)
            ReturnInstSet.insert(cast<ReturnInst>(RI));

        indicateOptimisticFixpoint();
        return;
      }
    }

    if (!A.isFunctionIPOAmendable(*F))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override;

  /// See AbstractAttribute::getState(...).
  AbstractState &getState() override { return *this; }

  /// See AbstractAttribute::getState(...).
  const AbstractState &getState() const override { return *this; }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;

  llvm::iterator_range<iterator> returned_values() override {
    return llvm::make_range(ReturnedValues.begin(), ReturnedValues.end());
  }

  llvm::iterator_range<const_iterator> returned_values() const override {
    return llvm::make_range(ReturnedValues.begin(), ReturnedValues.end());
  }

  const SmallSetVector<CallBase *, 4> &getUnresolvedCalls() const override {
    return UnresolvedCalls;
  }

  /// Return the number of potential return values, -1 if unknown.
  size_t getNumReturnValues() const override {
    return isValidState() ? ReturnedValues.size() : -1;
  }

  /// Return an assumed unique return value if a single candidate is found. If
  /// there cannot be one, return a nullptr. If it is not clear yet, return the
  /// Optional::NoneType.
  Optional<Value *> getAssumedUniqueReturnValue(Attributor &A) const;

  /// See AbstractState::checkForAllReturnedValues(...).
  bool checkForAllReturnedValuesAndReturnInsts(
      function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)> Pred)
      const override;

  /// Pretty print the attribute similar to the IR representation.
  const std::string getAsStr() const override;

  /// See AbstractState::isAtFixpoint().
  bool isAtFixpoint() const override { return IsFixed; }

  /// See AbstractState::isValidState().
  bool isValidState() const override { return IsValidState; }

  /// See AbstractState::indicateOptimisticFixpoint(...).
  ChangeStatus indicateOptimisticFixpoint() override {
    IsFixed = true;
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    IsFixed = true;
    IsValidState = false;
    return ChangeStatus::CHANGED;
  }
};

ChangeStatus AAReturnedValuesImpl::manifest(Attributor &A) {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  // Bookkeeping.
  assert(isValidState());
  STATS_DECLTRACK(KnownReturnValues, FunctionReturn,
                  "Number of function with known return values");

  // Check if we have an assumed unique return value that we could manifest.
  Optional<Value *> UniqueRV = getAssumedUniqueReturnValue(A);

  if (!UniqueRV.hasValue() || !UniqueRV.getValue())
    return Changed;

  // Bookkeeping.
  STATS_DECLTRACK(UniqueReturnValue, FunctionReturn,
                  "Number of function with unique return");

  // Callback to replace the uses of CB with the constant C.
  auto ReplaceCallSiteUsersWith = [&A](CallBase &CB, Constant &C) {
    if (CB.use_empty())
      return ChangeStatus::UNCHANGED;
    if (A.changeValueAfterManifest(CB, C))
      return ChangeStatus::CHANGED;
    return ChangeStatus::UNCHANGED;
  };

  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.getValue())) {
    if (UniqueRVArg->getType()->canLosslesslyBitCastTo(
            getAssociatedFunction()->getReturnType())) {
      getIRPosition() = IRPosition::argument(*UniqueRVArg);
      Changed = IRAttribute::manifest(A);
    }
  } else if (auto *RVC = dyn_cast<Constant>(UniqueRV.getValue())) {
    // We can replace the returned value with the unique returned constant.
    Value &AnchorValue = getAnchorValue();
    if (Function *F = dyn_cast<Function>(&AnchorValue)) {
      for (const Use &U : F->uses())
        if (CallBase *CB = dyn_cast<CallBase>(U.getUser()))
          if (CB->isCallee(&U)) {
            Constant *RVCCast =
                CB->getType() == RVC->getType()
                    ? RVC
                    : ConstantExpr::getTruncOrBitCast(RVC, CB->getType());
            Changed = ReplaceCallSiteUsersWith(*CB, *RVCCast) | Changed;
          }
    } else {
      assert(isa<CallBase>(AnchorValue) &&
             "Expcected a function or call base anchor!");
      Constant *RVCCast =
          AnchorValue.getType() == RVC->getType()
              ? RVC
              : ConstantExpr::getTruncOrBitCast(RVC, AnchorValue.getType());
      Changed = ReplaceCallSiteUsersWith(cast<CallBase>(AnchorValue), *RVCCast);
    }
    if (Changed == ChangeStatus::CHANGED)
      STATS_DECLTRACK(UniqueConstantReturnValue, FunctionReturn,
                      "Number of function returns replaced by constant return");
  }

  return Changed;
}

const std::string AAReturnedValuesImpl::getAsStr() const {
  return (isAtFixpoint() ? "returns(#" : "may-return(#") +
         (isValidState() ? std::to_string(getNumReturnValues()) : "?") +
         ")[#UC: " + std::to_string(UnresolvedCalls.size()) + "]";
}

Optional<Value *>
AAReturnedValuesImpl::getAssumedUniqueReturnValue(Attributor &A) const {
  // If checkForAllReturnedValues provides a unique value, ignoring potential
  // undef values that can also be present, it is assumed to be the actual
  // return value and forwarded to the caller of this method. If there are
  // multiple, a nullptr is returned indicating there cannot be a unique
  // returned value.
  Optional<Value *> UniqueRV;

  auto Pred = [&](Value &RV) -> bool {
    // If we found a second returned value and neither the current nor the saved
    // one is an undef, there is no unique returned value. Undefs are special
    // since we can pretend they have any value.
    if (UniqueRV.hasValue() && UniqueRV != &RV &&
        !(isa<UndefValue>(RV) || isa<UndefValue>(UniqueRV.getValue()))) {
      UniqueRV = nullptr;
      return false;
    }

    // Do not overwrite a value with an undef.
    if (!UniqueRV.hasValue() || !isa<UndefValue>(RV))
      UniqueRV = &RV;

    return true;
  };

  if (!A.checkForAllReturnedValues(Pred, *this))
    UniqueRV = nullptr;

  return UniqueRV;
}

bool AAReturnedValuesImpl::checkForAllReturnedValuesAndReturnInsts(
    function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)> Pred)
    const {
  if (!isValidState())
    return false;

  // Check all returned values but ignore call sites as long as we have not
  // encountered an overdefined one during an update.
  for (auto &It : ReturnedValues) {
    Value *RV = It.first;

    CallBase *CB = dyn_cast<CallBase>(RV);
    if (CB && !UnresolvedCalls.count(CB))
      continue;

    if (!Pred(*RV, It.second))
      return false;
  }

  return true;
}

ChangeStatus AAReturnedValuesImpl::updateImpl(Attributor &A) {
  size_t NumUnresolvedCalls = UnresolvedCalls.size();
  bool Changed = false;

  // State used in the value traversals starting in returned values.
  struct RVState {
    // The map in which we collect return values -> return instrs.
    decltype(ReturnedValues) &RetValsMap;
    // The flag to indicate a change.
    bool &Changed;
    // The return instrs we come from.
    SmallSetVector<ReturnInst *, 4> RetInsts;
  };

  // Callback for a leaf value returned by the associated function.
  auto VisitValueCB = [](Value &Val, const Instruction *, RVState &RVS,
                         bool) -> bool {
    auto Size = RVS.RetValsMap[&Val].size();
    RVS.RetValsMap[&Val].insert(RVS.RetInsts.begin(), RVS.RetInsts.end());
    bool Inserted = RVS.RetValsMap[&Val].size() != Size;
    RVS.Changed |= Inserted;
    LLVM_DEBUG({
      if (Inserted)
        dbgs() << "[AAReturnedValues] 1 Add new returned value " << Val
               << " => " << RVS.RetInsts.size() << "\n";
    });
    return true;
  };

  // Helper method to invoke the generic value traversal.
  auto VisitReturnedValue = [&](Value &RV, RVState &RVS,
                                const Instruction *CtxI) {
    IRPosition RetValPos = IRPosition::value(RV);
    return genericValueTraversal<AAReturnedValues, RVState>(
        A, RetValPos, *this, RVS, VisitValueCB, CtxI,
        /* UseValueSimplify */ false);
  };

  // Callback for all "return intructions" live in the associated function.
  auto CheckReturnInst = [this, &VisitReturnedValue, &Changed](Instruction &I) {
    ReturnInst &Ret = cast<ReturnInst>(I);
    RVState RVS({ReturnedValues, Changed, {}});
    RVS.RetInsts.insert(&Ret);
    return VisitReturnedValue(*Ret.getReturnValue(), RVS, &I);
  };

  // Start by discovering returned values from all live returned instructions in
  // the associated function.
  if (!A.checkForAllInstructions(CheckReturnInst, *this, {Instruction::Ret}))
    return indicatePessimisticFixpoint();

  // Once returned values "directly" present in the code are handled we try to
  // resolve returned calls. To avoid modifications to the ReturnedValues map
  // while we iterate over it we kept record of potential new entries in a copy
  // map, NewRVsMap.
  decltype(ReturnedValues) NewRVsMap;

  auto HandleReturnValue = [&](Value *RV,
                               SmallSetVector<ReturnInst *, 4> &RIs) {
    LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned value: " << *RV << " by #"
                      << RIs.size() << " RIs\n");
    CallBase *CB = dyn_cast<CallBase>(RV);
    if (!CB || UnresolvedCalls.count(CB))
      return;

    if (!CB->getCalledFunction()) {
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Unresolved call: " << *CB
                        << "\n");
      UnresolvedCalls.insert(CB);
      return;
    }

    // TODO: use the function scope once we have call site AAReturnedValues.
    const auto &RetValAA = A.getAAFor<AAReturnedValues>(
        *this, IRPosition::function(*CB->getCalledFunction()),
        DepClassTy::REQUIRED);
    LLVM_DEBUG(dbgs() << "[AAReturnedValues] Found another AAReturnedValues: "
                      << RetValAA << "\n");

    // Skip dead ends, thus if we do not know anything about the returned
    // call we mark it as unresolved and it will stay that way.
    if (!RetValAA.getState().isValidState()) {
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Unresolved call: " << *CB
                        << "\n");
      UnresolvedCalls.insert(CB);
      return;
    }

    // Do not try to learn partial information. If the callee has unresolved
    // return values we will treat the call as unresolved/opaque.
    auto &RetValAAUnresolvedCalls = RetValAA.getUnresolvedCalls();
    if (!RetValAAUnresolvedCalls.empty()) {
      UnresolvedCalls.insert(CB);
      return;
    }

    // Now check if we can track transitively returned values. If possible, thus
    // if all return value can be represented in the current scope, do so.
    bool Unresolved = false;
    for (auto &RetValAAIt : RetValAA.returned_values()) {
      Value *RetVal = RetValAAIt.first;
      if (isa<Argument>(RetVal) || isa<CallBase>(RetVal) ||
          isa<Constant>(RetVal))
        continue;
      // Anything that did not fit in the above categories cannot be resolved,
      // mark the call as unresolved.
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] transitively returned value "
                           "cannot be translated: "
                        << *RetVal << "\n");
      UnresolvedCalls.insert(CB);
      Unresolved = true;
      break;
    }

    if (Unresolved)
      return;

    // Now track transitively returned values.
    unsigned &NumRetAA = NumReturnedValuesPerKnownAA[CB];
    if (NumRetAA == RetValAA.getNumReturnValues()) {
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Skip call as it has not "
                           "changed since it was seen last\n");
      return;
    }
    NumRetAA = RetValAA.getNumReturnValues();

    for (auto &RetValAAIt : RetValAA.returned_values()) {
      Value *RetVal = RetValAAIt.first;
      if (Argument *Arg = dyn_cast<Argument>(RetVal)) {
        // Arguments are mapped to call site operands and we begin the traversal
        // again.
        bool Unused = false;
        RVState RVS({NewRVsMap, Unused, RetValAAIt.second});
        VisitReturnedValue(*CB->getArgOperand(Arg->getArgNo()), RVS, CB);
        continue;
      }
      if (isa<CallBase>(RetVal)) {
        // Call sites are resolved by the callee attribute over time, no need to
        // do anything for us.
        continue;
      }
      if (isa<Constant>(RetVal)) {
        // Constants are valid everywhere, we can simply take them.
        NewRVsMap[RetVal].insert(RIs.begin(), RIs.end());
        continue;
      }
    }
  };

  for (auto &It : ReturnedValues)
    HandleReturnValue(It.first, It.second);

  // Because processing the new information can again lead to new return values
  // we have to be careful and iterate until this iteration is complete. The
  // idea is that we are in a stable state at the end of an update. All return
  // values have been handled and properly categorized. We might not update
  // again if we have not requested a non-fix attribute so we cannot "wait" for
  // the next update to analyze a new return value.
  while (!NewRVsMap.empty()) {
    auto It = std::move(NewRVsMap.back());
    NewRVsMap.pop_back();

    assert(!It.second.empty() && "Entry does not add anything.");
    auto &ReturnInsts = ReturnedValues[It.first];
    for (ReturnInst *RI : It.second)
      if (ReturnInsts.insert(RI)) {
        LLVM_DEBUG(dbgs() << "[AAReturnedValues] Add new returned value "
                          << *It.first << " => " << *RI << "\n");
        HandleReturnValue(It.first, ReturnInsts);
        Changed = true;
      }
  }

  Changed |= (NumUnresolvedCalls != UnresolvedCalls.size());
  return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
}

struct AAReturnedValuesFunction final : public AAReturnedValuesImpl {
  AAReturnedValuesFunction(const IRPosition &IRP, Attributor &A)
      : AAReturnedValuesImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(returned) }
};

/// Returned values information for a call sites.
struct AAReturnedValuesCallSite final : AAReturnedValuesImpl {
  AAReturnedValuesCallSite(const IRPosition &IRP, Attributor &A)
      : AAReturnedValuesImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites instead of
    //       redirecting requests to the callee.
    llvm_unreachable("Abstract attributes for returned values are not "
                     "supported for call sites yet!");
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

/// ------------------------ NoSync Function Attribute -------------------------

struct AANoSyncImpl : AANoSync {
  AANoSyncImpl(const IRPosition &IRP, Attributor &A) : AANoSync(IRP, A) {}

  const std::string getAsStr() const override {
    return getAssumed() ? "nosync" : "may-sync";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// Helper function used to determine whether an instruction is non-relaxed
  /// atomic. In other words, if an atomic instruction does not have unordered
  /// or monotonic ordering
  static bool isNonRelaxedAtomic(Instruction *I);

  /// Helper function used to determine whether an instruction is volatile.
  static bool isVolatile(Instruction *I);

  /// Helper function uset to check if intrinsic is volatile (memcpy, memmove,
  /// memset).
  static bool isNoSyncIntrinsic(Instruction *I);
};

bool AANoSyncImpl::isNonRelaxedAtomic(Instruction *I) {
  if (!I->isAtomic())
    return false;

  AtomicOrdering Ordering;
  switch (I->getOpcode()) {
  case Instruction::AtomicRMW:
    Ordering = cast<AtomicRMWInst>(I)->getOrdering();
    break;
  case Instruction::Store:
    Ordering = cast<StoreInst>(I)->getOrdering();
    break;
  case Instruction::Load:
    Ordering = cast<LoadInst>(I)->getOrdering();
    break;
  case Instruction::Fence: {
    auto *FI = cast<FenceInst>(I);
    if (FI->getSyncScopeID() == SyncScope::SingleThread)
      return false;
    Ordering = FI->getOrdering();
    break;
  }
  case Instruction::AtomicCmpXchg: {
    AtomicOrdering Success = cast<AtomicCmpXchgInst>(I)->getSuccessOrdering();
    AtomicOrdering Failure = cast<AtomicCmpXchgInst>(I)->getFailureOrdering();
    // Only if both are relaxed, than it can be treated as relaxed.
    // Otherwise it is non-relaxed.
    if (Success != AtomicOrdering::Unordered &&
        Success != AtomicOrdering::Monotonic)
      return true;
    if (Failure != AtomicOrdering::Unordered &&
        Failure != AtomicOrdering::Monotonic)
      return true;
    return false;
  }
  default:
    llvm_unreachable(
        "New atomic operations need to be known in the attributor.");
  }

  // Relaxed.
  if (Ordering == AtomicOrdering::Unordered ||
      Ordering == AtomicOrdering::Monotonic)
    return false;
  return true;
}

/// Checks if an intrinsic is nosync. Currently only checks mem* intrinsics.
/// FIXME: We should ipmrove the handling of intrinsics.
bool AANoSyncImpl::isNoSyncIntrinsic(Instruction *I) {
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    /// Element wise atomic memory intrinsics are can only be unordered,
    /// therefore nosync.
    case Intrinsic::memset_element_unordered_atomic:
    case Intrinsic::memmove_element_unordered_atomic:
    case Intrinsic::memcpy_element_unordered_atomic:
      return true;
    case Intrinsic::memset:
    case Intrinsic::memmove:
    case Intrinsic::memcpy:
      if (!cast<MemIntrinsic>(II)->isVolatile())
        return true;
      return false;
    default:
      return false;
    }
  }
  return false;
}

bool AANoSyncImpl::isVolatile(Instruction *I) {
  assert(!isa<CallBase>(I) && "Calls should not be checked here");

  switch (I->getOpcode()) {
  case Instruction::AtomicRMW:
    return cast<AtomicRMWInst>(I)->isVolatile();
  case Instruction::Store:
    return cast<StoreInst>(I)->isVolatile();
  case Instruction::Load:
    return cast<LoadInst>(I)->isVolatile();
  case Instruction::AtomicCmpXchg:
    return cast<AtomicCmpXchgInst>(I)->isVolatile();
  default:
    return false;
  }
}

ChangeStatus AANoSyncImpl::updateImpl(Attributor &A) {

  auto CheckRWInstForNoSync = [&](Instruction &I) {
    /// We are looking for volatile instructions or Non-Relaxed atomics.
    /// FIXME: We should improve the handling of intrinsics.

    if (isa<IntrinsicInst>(&I) && isNoSyncIntrinsic(&I))
      return true;

    if (const auto *CB = dyn_cast<CallBase>(&I)) {
      if (CB->hasFnAttr(Attribute::NoSync))
        return true;

      const auto &NoSyncAA = A.getAAFor<AANoSync>(
          *this, IRPosition::callsite_function(*CB), DepClassTy::REQUIRED);
      return NoSyncAA.isAssumedNoSync();
    }

    if (!isVolatile(&I) && !isNonRelaxedAtomic(&I))
      return true;

    return false;
  };

  auto CheckForNoSync = [&](Instruction &I) {
    // At this point we handled all read/write effects and they are all
    // nosync, so they can be skipped.
    if (I.mayReadOrWriteMemory())
      return true;

    // non-convergent and readnone imply nosync.
    return !cast<CallBase>(I).isConvergent();
  };

  if (!A.checkForAllReadWriteInstructions(CheckRWInstForNoSync, *this) ||
      !A.checkForAllCallLikeInstructions(CheckForNoSync, *this))
    return indicatePessimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

struct AANoSyncFunction final : public AANoSyncImpl {
  AANoSyncFunction(const IRPosition &IRP, Attributor &A)
      : AANoSyncImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(nosync) }
};

/// NoSync attribute deduction for a call sites.
struct AANoSyncCallSite final : AANoSyncImpl {
  AANoSyncCallSite(const IRPosition &IRP, Attributor &A)
      : AANoSyncImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoSyncImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<AANoSync>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(nosync); }
};

/// ------------------------ No-Free Attributes ----------------------------

struct AANoFreeImpl : public AANoFree {
  AANoFreeImpl(const IRPosition &IRP, Attributor &A) : AANoFree(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto CheckForNoFree = [&](Instruction &I) {
      const auto &CB = cast<CallBase>(I);
      if (CB.hasFnAttr(Attribute::NoFree))
        return true;

      const auto &NoFreeAA = A.getAAFor<AANoFree>(
          *this, IRPosition::callsite_function(CB), DepClassTy::REQUIRED);
      return NoFreeAA.isAssumedNoFree();
    };

    if (!A.checkForAllCallLikeInstructions(CheckForNoFree, *this))
      return indicatePessimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nofree" : "may-free";
  }
};

struct AANoFreeFunction final : public AANoFreeImpl {
  AANoFreeFunction(const IRPosition &IRP, Attributor &A)
      : AANoFreeImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(nofree) }
};

/// NoFree attribute deduction for a call sites.
struct AANoFreeCallSite final : AANoFreeImpl {
  AANoFreeCallSite(const IRPosition &IRP, Attributor &A)
      : AANoFreeImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoFreeImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<AANoFree>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(nofree); }
};

/// NoFree attribute for floating values.
struct AANoFreeFloating : AANoFreeImpl {
  AANoFreeFloating(const IRPosition &IRP, Attributor &A)
      : AANoFreeImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override{STATS_DECLTRACK_FLOATING_ATTR(nofree)}

  /// See Abstract Attribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const IRPosition &IRP = getIRPosition();

    const auto &NoFreeAA = A.getAAFor<AANoFree>(
        *this, IRPosition::function_scope(IRP), DepClassTy::OPTIONAL);
    if (NoFreeAA.isAssumedNoFree())
      return ChangeStatus::UNCHANGED;

    Value &AssociatedValue = getIRPosition().getAssociatedValue();
    auto Pred = [&](const Use &U, bool &Follow) -> bool {
      Instruction *UserI = cast<Instruction>(U.getUser());
      if (auto *CB = dyn_cast<CallBase>(UserI)) {
        if (CB->isBundleOperand(&U))
          return false;
        if (!CB->isArgOperand(&U))
          return true;
        unsigned ArgNo = CB->getArgOperandNo(&U);

        const auto &NoFreeArg = A.getAAFor<AANoFree>(
            *this, IRPosition::callsite_argument(*CB, ArgNo),
            DepClassTy::REQUIRED);
        return NoFreeArg.isAssumedNoFree();
      }

      if (isa<GetElementPtrInst>(UserI) || isa<BitCastInst>(UserI) ||
          isa<PHINode>(UserI) || isa<SelectInst>(UserI)) {
        Follow = true;
        return true;
      }
      if (isa<ReturnInst>(UserI))
        return true;

      // Unknown user.
      return false;
    };
    if (!A.checkForAllUses(Pred, *this, AssociatedValue))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }
};

/// NoFree attribute for a call site argument.
struct AANoFreeArgument final : AANoFreeFloating {
  AANoFreeArgument(const IRPosition &IRP, Attributor &A)
      : AANoFreeFloating(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nofree) }
};

/// NoFree attribute for call site arguments.
struct AANoFreeCallSiteArgument final : AANoFreeFloating {
  AANoFreeCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AANoFreeFloating(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Argument *Arg = getAssociatedArgument();
    if (!Arg)
      return indicatePessimisticFixpoint();
    const IRPosition &ArgPos = IRPosition::argument(*Arg);
    auto &ArgAA = A.getAAFor<AANoFree>(*this, ArgPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), ArgAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override{STATS_DECLTRACK_CSARG_ATTR(nofree)};
};

/// NoFree attribute for function return value.
struct AANoFreeReturned final : AANoFreeFloating {
  AANoFreeReturned(const IRPosition &IRP, Attributor &A)
      : AANoFreeFloating(IRP, A) {
    llvm_unreachable("NoFree is not applicable to function returns!");
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    llvm_unreachable("NoFree is not applicable to function returns!");
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("NoFree is not applicable to function returns!");
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

/// NoFree attribute deduction for a call site return value.
struct AANoFreeCallSiteReturned final : AANoFreeFloating {
  AANoFreeCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AANoFreeFloating(IRP, A) {}

  ChangeStatus manifest(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }
  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(nofree) }
};

/// ------------------------ NonNull Argument Attribute ------------------------
static int64_t getKnownNonNullAndDerefBytesForUse(
    Attributor &A, const AbstractAttribute &QueryingAA, Value &AssociatedValue,
    const Use *U, const Instruction *I, bool &IsNonNull, bool &TrackUse) {
  TrackUse = false;

  const Value *UseV = U->get();
  if (!UseV->getType()->isPointerTy())
    return 0;

  // We need to follow common pointer manipulation uses to the accesses they
  // feed into. We can try to be smart to avoid looking through things we do not
  // like for now, e.g., non-inbounds GEPs.
  if (isa<CastInst>(I)) {
    TrackUse = true;
    return 0;
  }

  if (isa<GetElementPtrInst>(I)) {
    TrackUse = true;
    return 0;
  }

  Type *PtrTy = UseV->getType();
  const Function *F = I->getFunction();
  bool NullPointerIsDefined =
      F ? llvm::NullPointerIsDefined(F, PtrTy->getPointerAddressSpace()) : true;
  const DataLayout &DL = A.getInfoCache().getDL();
  if (const auto *CB = dyn_cast<CallBase>(I)) {
    if (CB->isBundleOperand(U)) {
      if (RetainedKnowledge RK = getKnowledgeFromUse(
              U, {Attribute::NonNull, Attribute::Dereferenceable})) {
        IsNonNull |=
            (RK.AttrKind == Attribute::NonNull || !NullPointerIsDefined);
        return RK.ArgValue;
      }
      return 0;
    }

    if (CB->isCallee(U)) {
      IsNonNull |= !NullPointerIsDefined;
      return 0;
    }

    unsigned ArgNo = CB->getArgOperandNo(U);
    IRPosition IRP = IRPosition::callsite_argument(*CB, ArgNo);
    // As long as we only use known information there is no need to track
    // dependences here.
    auto &DerefAA =
        A.getAAFor<AADereferenceable>(QueryingAA, IRP, DepClassTy::NONE);
    IsNonNull |= DerefAA.isKnownNonNull();
    return DerefAA.getKnownDereferenceableBytes();
  }

  int64_t Offset;
  const Value *Base =
      getMinimalBaseOfAccsesPointerOperand(A, QueryingAA, I, Offset, DL);
  if (Base) {
    if (Base == &AssociatedValue &&
        getPointerOperand(I, /* AllowVolatile */ false) == UseV) {
      int64_t DerefBytes =
          (int64_t)DL.getTypeStoreSize(PtrTy->getPointerElementType()) + Offset;

      IsNonNull |= !NullPointerIsDefined;
      return std::max(int64_t(0), DerefBytes);
    }
  }

  /// Corner case when an offset is 0.
  Base = getBasePointerOfAccessPointerOperand(I, Offset, DL,
                                              /*AllowNonInbounds*/ true);
  if (Base) {
    if (Offset == 0 && Base == &AssociatedValue &&
        getPointerOperand(I, /* AllowVolatile */ false) == UseV) {
      int64_t DerefBytes =
          (int64_t)DL.getTypeStoreSize(PtrTy->getPointerElementType());
      IsNonNull |= !NullPointerIsDefined;
      return std::max(int64_t(0), DerefBytes);
    }
  }

  return 0;
}

struct AANonNullImpl : AANonNull {
  AANonNullImpl(const IRPosition &IRP, Attributor &A)
      : AANonNull(IRP, A),
        NullIsDefined(NullPointerIsDefined(
            getAnchorScope(),
            getAssociatedValue().getType()->getPointerAddressSpace())) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Value &V = getAssociatedValue();
    if (!NullIsDefined &&
        hasAttr({Attribute::NonNull, Attribute::Dereferenceable},
                /* IgnoreSubsumingPositions */ false, &A)) {
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<ConstantPointerNull>(V)) {
      indicatePessimisticFixpoint();
      return;
    }

    AANonNull::initialize(A);

    bool CanBeNull = true;
    if (V.getPointerDereferenceableBytes(A.getDataLayout(), CanBeNull)) {
      if (!CanBeNull) {
        indicateOptimisticFixpoint();
        return;
      }
    }

    if (isa<GlobalValue>(&getAssociatedValue())) {
      indicatePessimisticFixpoint();
      return;
    }

    if (Instruction *CtxI = getCtxI())
      followUsesInMBEC(*this, A, getState(), *CtxI);
  }

  /// See followUsesInMBEC
  bool followUseInMBEC(Attributor &A, const Use *U, const Instruction *I,
                       AANonNull::StateType &State) {
    bool IsNonNull = false;
    bool TrackUse = false;
    getKnownNonNullAndDerefBytesForUse(A, *this, getAssociatedValue(), U, I,
                                       IsNonNull, TrackUse);
    State.setKnown(IsNonNull);
    return TrackUse;
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nonnull" : "may-null";
  }

  /// Flag to determine if the underlying value can be null and still allow
  /// valid accesses.
  const bool NullIsDefined;
};

/// NonNull attribute for a floating value.
struct AANonNullFloating : public AANonNullImpl {
  AANonNullFloating(const IRPosition &IRP, Attributor &A)
      : AANonNullImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    DominatorTree *DT = nullptr;
    AssumptionCache *AC = nullptr;
    InformationCache &InfoCache = A.getInfoCache();
    if (const Function *Fn = getAnchorScope()) {
      DT = InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(*Fn);
      AC = InfoCache.getAnalysisResultForFunction<AssumptionAnalysis>(*Fn);
    }

    auto VisitValueCB = [&](Value &V, const Instruction *CtxI,
                            AANonNull::StateType &T, bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AANonNull>(*this, IRPosition::value(V),
                                             DepClassTy::REQUIRED);
      if (!Stripped && this == &AA) {
        if (!isKnownNonZero(&V, DL, 0, AC, CtxI, DT))
          T.indicatePessimisticFixpoint();
      } else {
        // Use abstract attribute information.
        const AANonNull::StateType &NS = AA.getState();
        T ^= NS;
      }
      return T.isValidState();
    };

    StateType T;
    if (!genericValueTraversal<AANonNull, StateType>(
            A, getIRPosition(), *this, T, VisitValueCB, getCtxI()))
      return indicatePessimisticFixpoint();

    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(nonnull) }
};

/// NonNull attribute for function return value.
struct AANonNullReturned final
    : AAReturnedFromReturnedValues<AANonNull, AANonNull> {
  AANonNullReturned(const IRPosition &IRP, Attributor &A)
      : AAReturnedFromReturnedValues<AANonNull, AANonNull>(IRP, A) {}

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nonnull" : "may-null";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(nonnull) }
};

/// NonNull attribute for function argument.
struct AANonNullArgument final
    : AAArgumentFromCallSiteArguments<AANonNull, AANonNullImpl> {
  AANonNullArgument(const IRPosition &IRP, Attributor &A)
      : AAArgumentFromCallSiteArguments<AANonNull, AANonNullImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nonnull) }
};

struct AANonNullCallSiteArgument final : AANonNullFloating {
  AANonNullCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AANonNullFloating(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(nonnull) }
};

/// NonNull attribute for a call site return position.
struct AANonNullCallSiteReturned final
    : AACallSiteReturnedFromReturned<AANonNull, AANonNullImpl> {
  AANonNullCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AACallSiteReturnedFromReturned<AANonNull, AANonNullImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(nonnull) }
};

/// ------------------------ No-Recurse Attributes ----------------------------

struct AANoRecurseImpl : public AANoRecurse {
  AANoRecurseImpl(const IRPosition &IRP, Attributor &A) : AANoRecurse(IRP, A) {}

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "norecurse" : "may-recurse";
  }
};

struct AANoRecurseFunction final : AANoRecurseImpl {
  AANoRecurseFunction(const IRPosition &IRP, Attributor &A)
      : AANoRecurseImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoRecurseImpl::initialize(A);
    if (const Function *F = getAnchorScope())
      if (A.getInfoCache().getSccSize(*F) != 1)
        indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {

    // If all live call sites are known to be no-recurse, we are as well.
    auto CallSitePred = [&](AbstractCallSite ACS) {
      const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
          *this, IRPosition::function(*ACS.getInstruction()->getFunction()),
          DepClassTy::NONE);
      return NoRecurseAA.isKnownNoRecurse();
    };
    bool AllCallSitesKnown;
    if (A.checkForAllCallSites(CallSitePred, *this, true, AllCallSitesKnown)) {
      // If we know all call sites and all are known no-recurse, we are done.
      // If all known call sites, which might not be all that exist, are known
      // to be no-recurse, we are not done but we can continue to assume
      // no-recurse. If one of the call sites we have not visited will become
      // live, another update is triggered.
      if (AllCallSitesKnown)
        indicateOptimisticFixpoint();
      return ChangeStatus::UNCHANGED;
    }

    // If the above check does not hold anymore we look at the calls.
    auto CheckForNoRecurse = [&](Instruction &I) {
      const auto &CB = cast<CallBase>(I);
      if (CB.hasFnAttr(Attribute::NoRecurse))
        return true;

      const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
          *this, IRPosition::callsite_function(CB), DepClassTy::REQUIRED);
      if (!NoRecurseAA.isAssumedNoRecurse())
        return false;

      // Recursion to the same function
      if (CB.getCalledFunction() == getAnchorScope())
        return false;

      return true;
    };

    if (!A.checkForAllCallLikeInstructions(CheckForNoRecurse, *this))
      return indicatePessimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(norecurse) }
};

/// NoRecurse attribute deduction for a call sites.
struct AANoRecurseCallSite final : AANoRecurseImpl {
  AANoRecurseCallSite(const IRPosition &IRP, Attributor &A)
      : AANoRecurseImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoRecurseImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<AANoRecurse>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(norecurse); }
};

/// -------------------- Undefined-Behavior Attributes ------------------------

struct AAUndefinedBehaviorImpl : public AAUndefinedBehavior {
  AAUndefinedBehaviorImpl(const IRPosition &IRP, Attributor &A)
      : AAUndefinedBehavior(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  // through a pointer (i.e. also branches etc.)
  ChangeStatus updateImpl(Attributor &A) override {
    const size_t UBPrevSize = KnownUBInsts.size();
    const size_t NoUBPrevSize = AssumedNoUBInsts.size();

    auto InspectMemAccessInstForUB = [&](Instruction &I) {
      // Skip instructions that are already saved.
      if (AssumedNoUBInsts.count(&I) || KnownUBInsts.count(&I))
        return true;

      // If we reach here, we know we have an instruction
      // that accesses memory through a pointer operand,
      // for which getPointerOperand() should give it to us.
      const Value *PtrOp = getPointerOperand(&I, /* AllowVolatile */ true);
      assert(PtrOp &&
             "Expected pointer operand of memory accessing instruction");

      // Either we stopped and the appropriate action was taken,
      // or we got back a simplified value to continue.
      Optional<Value *> SimplifiedPtrOp = stopOnUndefOrAssumed(A, PtrOp, &I);
      if (!SimplifiedPtrOp.hasValue())
        return true;
      const Value *PtrOpVal = SimplifiedPtrOp.getValue();

      // A memory access through a pointer is considered UB
      // only if the pointer has constant null value.
      // TODO: Expand it to not only check constant values.
      if (!isa<ConstantPointerNull>(PtrOpVal)) {
        AssumedNoUBInsts.insert(&I);
        return true;
      }
      const Type *PtrTy = PtrOpVal->getType();

      // Because we only consider instructions inside functions,
      // assume that a parent function exists.
      const Function *F = I.getFunction();

      // A memory access using constant null pointer is only considered UB
      // if null pointer is _not_ defined for the target platform.
      if (llvm::NullPointerIsDefined(F, PtrTy->getPointerAddressSpace()))
        AssumedNoUBInsts.insert(&I);
      else
        KnownUBInsts.insert(&I);
      return true;
    };

    auto InspectBrInstForUB = [&](Instruction &I) {
      // A conditional branch instruction is considered UB if it has `undef`
      // condition.

      // Skip instructions that are already saved.
      if (AssumedNoUBInsts.count(&I) || KnownUBInsts.count(&I))
        return true;

      // We know we have a branch instruction.
      auto BrInst = cast<BranchInst>(&I);

      // Unconditional branches are never considered UB.
      if (BrInst->isUnconditional())
        return true;

      // Either we stopped and the appropriate action was taken,
      // or we got back a simplified value to continue.
      Optional<Value *> SimplifiedCond =
          stopOnUndefOrAssumed(A, BrInst->getCondition(), BrInst);
      if (!SimplifiedCond.hasValue())
        return true;
      AssumedNoUBInsts.insert(&I);
      return true;
    };

    auto InspectCallSiteForUB = [&](Instruction &I) {
      // Check whether a callsite always cause UB or not

      // Skip instructions that are already saved.
      if (AssumedNoUBInsts.count(&I) || KnownUBInsts.count(&I))
        return true;

      // Check nonnull and noundef argument attribute violation for each
      // callsite.
      CallBase &CB = cast<CallBase>(I);
      Function *Callee = CB.getCalledFunction();
      if (!Callee)
        return true;
      for (unsigned idx = 0; idx < CB.getNumArgOperands(); idx++) {
        // If current argument is known to be simplified to null pointer and the
        // corresponding argument position is known to have nonnull attribute,
        // the argument is poison. Furthermore, if the argument is poison and
        // the position is known to have noundef attriubte, this callsite is
        // considered UB.
        if (idx >= Callee->arg_size())
          break;
        Value *ArgVal = CB.getArgOperand(idx);
        if (!ArgVal)
          continue;
        // Here, we handle three cases.
        //   (1) Not having a value means it is dead. (we can replace the value
        //       with undef)
        //   (2) Simplified to undef. The argument violate noundef attriubte.
        //   (3) Simplified to null pointer where known to be nonnull.
        //       The argument is a poison value and violate noundef attribute.
        IRPosition CalleeArgumentIRP = IRPosition::callsite_argument(CB, idx);
        auto &NoUndefAA =
            A.getAAFor<AANoUndef>(*this, CalleeArgumentIRP, DepClassTy::NONE);
        if (!NoUndefAA.isKnownNoUndef())
          continue;
        auto &ValueSimplifyAA = A.getAAFor<AAValueSimplify>(
            *this, IRPosition::value(*ArgVal), DepClassTy::NONE);
        if (!ValueSimplifyAA.isKnown())
          continue;
        Optional<Value *> SimplifiedVal =
            ValueSimplifyAA.getAssumedSimplifiedValue(A);
        if (!SimplifiedVal.hasValue() ||
            isa<UndefValue>(*SimplifiedVal.getValue())) {
          KnownUBInsts.insert(&I);
          continue;
        }
        if (!ArgVal->getType()->isPointerTy() ||
            !isa<ConstantPointerNull>(*SimplifiedVal.getValue()))
          continue;
        auto &NonNullAA =
            A.getAAFor<AANonNull>(*this, CalleeArgumentIRP, DepClassTy::NONE);
        if (NonNullAA.isKnownNonNull())
          KnownUBInsts.insert(&I);
      }
      return true;
    };

    auto InspectReturnInstForUB =
        [&](Value &V, const SmallSetVector<ReturnInst *, 4> RetInsts) {
          // Check if a return instruction always cause UB or not
          // Note: It is guaranteed that the returned position of the anchor
          //       scope has noundef attribute when this is called.
          //       We also ensure the return position is not "assumed dead"
          //       because the returned value was then potentially simplified to
          //       `undef` in AAReturnedValues without removing the `noundef`
          //       attribute yet.

          // When the returned position has noundef attriubte, UB occur in the
          // following cases.
          //   (1) Returned value is known to be undef.
          //   (2) The value is known to be a null pointer and the returned
          //       position has nonnull attribute (because the returned value is
          //       poison).
          bool FoundUB = false;
          if (isa<UndefValue>(V)) {
            FoundUB = true;
          } else {
            if (isa<ConstantPointerNull>(V)) {
              auto &NonNullAA = A.getAAFor<AANonNull>(
                  *this, IRPosition::returned(*getAnchorScope()),
                  DepClassTy::NONE);
              if (NonNullAA.isKnownNonNull())
                FoundUB = true;
            }
          }

          if (FoundUB)
            for (ReturnInst *RI : RetInsts)
              KnownUBInsts.insert(RI);
          return true;
        };

    A.checkForAllInstructions(InspectMemAccessInstForUB, *this,
                              {Instruction::Load, Instruction::Store,
                               Instruction::AtomicCmpXchg,
                               Instruction::AtomicRMW},
                              /* CheckBBLivenessOnly */ true);
    A.checkForAllInstructions(InspectBrInstForUB, *this, {Instruction::Br},
                              /* CheckBBLivenessOnly */ true);
    A.checkForAllCallLikeInstructions(InspectCallSiteForUB, *this);

    // If the returned position of the anchor scope has noundef attriubte, check
    // all returned instructions.
    if (!getAnchorScope()->getReturnType()->isVoidTy()) {
      const IRPosition &ReturnIRP = IRPosition::returned(*getAnchorScope());
      if (!A.isAssumedDead(ReturnIRP, this, nullptr)) {
        auto &RetPosNoUndefAA =
            A.getAAFor<AANoUndef>(*this, ReturnIRP, DepClassTy::NONE);
        if (RetPosNoUndefAA.isKnownNoUndef())
          A.checkForAllReturnedValuesAndReturnInsts(InspectReturnInstForUB,
                                                    *this);
      }
    }

    if (NoUBPrevSize != AssumedNoUBInsts.size() ||
        UBPrevSize != KnownUBInsts.size())
      return ChangeStatus::CHANGED;
    return ChangeStatus::UNCHANGED;
  }

  bool isKnownToCauseUB(Instruction *I) const override {
    return KnownUBInsts.count(I);
  }

  bool isAssumedToCauseUB(Instruction *I) const override {
    // In simple words, if an instruction is not in the assumed to _not_
    // cause UB, then it is assumed UB (that includes those
    // in the KnownUBInsts set). The rest is boilerplate
    // is to ensure that it is one of the instructions we test
    // for UB.

    switch (I->getOpcode()) {
    case Instruction::Load:
    case Instruction::Store:
    case Instruction::AtomicCmpXchg:
    case Instruction::AtomicRMW:
      return !AssumedNoUBInsts.count(I);
    case Instruction::Br: {
      auto BrInst = cast<BranchInst>(I);
      if (BrInst->isUnconditional())
        return false;
      return !AssumedNoUBInsts.count(I);
    } break;
    default:
      return false;
    }
    return false;
  }

  ChangeStatus manifest(Attributor &A) override {
    if (KnownUBInsts.empty())
      return ChangeStatus::UNCHANGED;
    for (Instruction *I : KnownUBInsts)
      A.changeToUnreachableAfterManifest(I);
    return ChangeStatus::CHANGED;
  }

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "undefined-behavior" : "no-ub";
  }

  /// Note: The correctness of this analysis depends on the fact that the
  /// following 2 sets will stop changing after some point.
  /// "Change" here means that their size changes.
  /// The size of each set is monotonically increasing
  /// (we only add items to them) and it is upper bounded by the number of
  /// instructions in the processed function (we can never save more
  /// elements in either set than this number). Hence, at some point,
  /// they will stop increasing.
  /// Consequently, at some point, both sets will have stopped
  /// changing, effectively making the analysis reach a fixpoint.

  /// Note: These 2 sets are disjoint and an instruction can be considered
  /// one of 3 things:
  /// 1) Known to cause UB (AAUndefinedBehavior could prove it) and put it in
  ///    the KnownUBInsts set.
  /// 2) Assumed to cause UB (in every updateImpl, AAUndefinedBehavior
  ///    has a reason to assume it).
  /// 3) Assumed to not cause UB. very other instruction - AAUndefinedBehavior
  ///    could not find a reason to assume or prove that it can cause UB,
  ///    hence it assumes it doesn't. We have a set for these instructions
  ///    so that we don't reprocess them in every update.
  ///    Note however that instructions in this set may cause UB.

protected:
  /// A set of all live instructions _known_ to cause UB.
  SmallPtrSet<Instruction *, 8> KnownUBInsts;

private:
  /// A set of all the (live) instructions that are assumed to _not_ cause UB.
  SmallPtrSet<Instruction *, 8> AssumedNoUBInsts;

  // Should be called on updates in which if we're processing an instruction
  // \p I that depends on a value \p V, one of the following has to happen:
  // - If the value is assumed, then stop.
  // - If the value is known but undef, then consider it UB.
  // - Otherwise, do specific processing with the simplified value.
  // We return None in the first 2 cases to signify that an appropriate
  // action was taken and the caller should stop.
  // Otherwise, we return the simplified value that the caller should
  // use for specific processing.
  Optional<Value *> stopOnUndefOrAssumed(Attributor &A, const Value *V,
                                         Instruction *I) {
    const auto &ValueSimplifyAA = A.getAAFor<AAValueSimplify>(
        *this, IRPosition::value(*V), DepClassTy::REQUIRED);
    Optional<Value *> SimplifiedV =
        ValueSimplifyAA.getAssumedSimplifiedValue(A);
    if (!ValueSimplifyAA.isKnown()) {
      // Don't depend on assumed values.
      return llvm::None;
    }
    if (!SimplifiedV.hasValue()) {
      // If it is known (which we tested above) but it doesn't have a value,
      // then we can assume `undef` and hence the instruction is UB.
      KnownUBInsts.insert(I);
      return llvm::None;
    }
    Value *Val = SimplifiedV.getValue();
    if (isa<UndefValue>(Val)) {
      KnownUBInsts.insert(I);
      return llvm::None;
    }
    return Val;
  }
};

struct AAUndefinedBehaviorFunction final : AAUndefinedBehaviorImpl {
  AAUndefinedBehaviorFunction(const IRPosition &IRP, Attributor &A)
      : AAUndefinedBehaviorImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECL(UndefinedBehaviorInstruction, Instruction,
               "Number of instructions known to have UB");
    BUILD_STAT_NAME(UndefinedBehaviorInstruction, Instruction) +=
        KnownUBInsts.size();
  }
};

/// ------------------------ Will-Return Attributes ----------------------------

// Helper function that checks whether a function has any cycle which we don't
// know if it is bounded or not.
// Loops with maximum trip count are considered bounded, any other cycle not.
static bool mayContainUnboundedCycle(Function &F, Attributor &A) {
  ScalarEvolution *SE =
      A.getInfoCache().getAnalysisResultForFunction<ScalarEvolutionAnalysis>(F);
  LoopInfo *LI = A.getInfoCache().getAnalysisResultForFunction<LoopAnalysis>(F);
  // If either SCEV or LoopInfo is not available for the function then we assume
  // any cycle to be unbounded cycle.
  // We use scc_iterator which uses Tarjan algorithm to find all the maximal
  // SCCs.To detect if there's a cycle, we only need to find the maximal ones.
  if (!SE || !LI) {
    for (scc_iterator<Function *> SCCI = scc_begin(&F); !SCCI.isAtEnd(); ++SCCI)
      if (SCCI.hasCycle())
        return true;
    return false;
  }

  // If there's irreducible control, the function may contain non-loop cycles.
  if (mayContainIrreducibleControl(F, LI))
    return true;

  // Any loop that does not have a max trip count is considered unbounded cycle.
  for (auto *L : LI->getLoopsInPreorder()) {
    if (!SE->getSmallConstantMaxTripCount(L))
      return true;
  }
  return false;
}

struct AAWillReturnImpl : public AAWillReturn {
  AAWillReturnImpl(const IRPosition &IRP, Attributor &A)
      : AAWillReturn(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAWillReturn::initialize(A);

    Function *F = getAnchorScope();
    if (!F || F->isDeclaration() || mayContainUnboundedCycle(*F, A))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto CheckForWillReturn = [&](Instruction &I) {
      IRPosition IPos = IRPosition::callsite_function(cast<CallBase>(I));
      const auto &WillReturnAA =
          A.getAAFor<AAWillReturn>(*this, IPos, DepClassTy::REQUIRED);
      if (WillReturnAA.isKnownWillReturn())
        return true;
      if (!WillReturnAA.isAssumedWillReturn())
        return false;
      const auto &NoRecurseAA =
          A.getAAFor<AANoRecurse>(*this, IPos, DepClassTy::REQUIRED);
      return NoRecurseAA.isAssumedNoRecurse();
    };

    if (!A.checkForAllCallLikeInstructions(CheckForWillReturn, *this))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "willreturn" : "may-noreturn";
  }
};

struct AAWillReturnFunction final : AAWillReturnImpl {
  AAWillReturnFunction(const IRPosition &IRP, Attributor &A)
      : AAWillReturnImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(willreturn) }
};

/// WillReturn attribute deduction for a call sites.
struct AAWillReturnCallSite final : AAWillReturnImpl {
  AAWillReturnCallSite(const IRPosition &IRP, Attributor &A)
      : AAWillReturnImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAWillReturn::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || !A.isFunctionIPOAmendable(*F))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<AAWillReturn>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(willreturn); }
};

/// -------------------AAReachability Attribute--------------------------

struct AAReachabilityImpl : AAReachability {
  AAReachabilityImpl(const IRPosition &IRP, Attributor &A)
      : AAReachability(IRP, A) {}

  const std::string getAsStr() const override {
    // TODO: Return the number of reachable queries.
    return "reachable";
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override { indicatePessimisticFixpoint(); }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return indicatePessimisticFixpoint();
  }
};

struct AAReachabilityFunction final : public AAReachabilityImpl {
  AAReachabilityFunction(const IRPosition &IRP, Attributor &A)
      : AAReachabilityImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(reachable); }
};

/// ------------------------ NoAlias Argument Attribute ------------------------

struct AANoAliasImpl : AANoAlias {
  AANoAliasImpl(const IRPosition &IRP, Attributor &A) : AANoAlias(IRP, A) {
    assert(getAssociatedType()->isPointerTy() &&
           "Noalias is a pointer attribute");
  }

  const std::string getAsStr() const override {
    return getAssumed() ? "noalias" : "may-alias";
  }
};

/// NoAlias attribute for a floating value.
struct AANoAliasFloating final : AANoAliasImpl {
  AANoAliasFloating(const IRPosition &IRP, Attributor &A)
      : AANoAliasImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoAliasImpl::initialize(A);
    Value *Val = &getAssociatedValue();
    do {
      CastInst *CI = dyn_cast<CastInst>(Val);
      if (!CI)
        break;
      Value *Base = CI->getOperand(0);
      if (!Base->hasOneUse())
        break;
      Val = Base;
    } while (true);

    if (!Val->getType()->isPointerTy()) {
      indicatePessimisticFixpoint();
      return;
    }

    if (isa<AllocaInst>(Val))
      indicateOptimisticFixpoint();
    else if (isa<ConstantPointerNull>(Val) &&
             !NullPointerIsDefined(getAnchorScope(),
                                   Val->getType()->getPointerAddressSpace()))
      indicateOptimisticFixpoint();
    else if (Val != &getAssociatedValue()) {
      const auto &ValNoAliasAA = A.getAAFor<AANoAlias>(
          *this, IRPosition::value(*Val), DepClassTy::OPTIONAL);
      if (ValNoAliasAA.isKnownNoAlias())
        indicateOptimisticFixpoint();
    }
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Implement this.
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(noalias)
  }
};

/// NoAlias attribute for an argument.
struct AANoAliasArgument final
    : AAArgumentFromCallSiteArguments<AANoAlias, AANoAliasImpl> {
  using Base = AAArgumentFromCallSiteArguments<AANoAlias, AANoAliasImpl>;
  AANoAliasArgument(const IRPosition &IRP, Attributor &A) : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Base::initialize(A);
    // See callsite argument attribute and callee argument attribute.
    if (hasAttr({Attribute::ByVal}))
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::update(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // We have to make sure no-alias on the argument does not break
    // synchronization when this is a callback argument, see also [1] below.
    // If synchronization cannot be affected, we delegate to the base updateImpl
    // function, otherwise we give up for now.

    // If the function is no-sync, no-alias cannot break synchronization.
    const auto &NoSyncAA =
        A.getAAFor<AANoSync>(*this, IRPosition::function_scope(getIRPosition()),
                             DepClassTy::OPTIONAL);
    if (NoSyncAA.isAssumedNoSync())
      return Base::updateImpl(A);

    // If the argument is read-only, no-alias cannot break synchronization.
    const auto &MemBehaviorAA = A.getAAFor<AAMemoryBehavior>(
        *this, getIRPosition(), DepClassTy::OPTIONAL);
    if (MemBehaviorAA.isAssumedReadOnly())
      return Base::updateImpl(A);

    // If the argument is never passed through callbacks, no-alias cannot break
    // synchronization.
    bool AllCallSitesKnown;
    if (A.checkForAllCallSites(
            [](AbstractCallSite ACS) { return !ACS.isCallbackCall(); }, *this,
            true, AllCallSitesKnown))
      return Base::updateImpl(A);

    // TODO: add no-alias but make sure it doesn't break synchronization by
    // introducing fake uses. See:
    // [1] Compiler Optimizations for OpenMP, J. Doerfert and H. Finkel,
    //     International Workshop on OpenMP 2018,
    //     http://compilers.cs.uni-saarland.de/people/doerfert/par_opt18.pdf

    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(noalias) }
};

struct AANoAliasCallSiteArgument final : AANoAliasImpl {
  AANoAliasCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AANoAliasImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // See callsite argument attribute and callee argument attribute.
    const auto &CB = cast<CallBase>(getAnchorValue());
    if (CB.paramHasAttr(getCallSiteArgNo(), Attribute::NoAlias))
      indicateOptimisticFixpoint();
    Value &Val = getAssociatedValue();
    if (isa<ConstantPointerNull>(Val) &&
        !NullPointerIsDefined(getAnchorScope(),
                              Val.getType()->getPointerAddressSpace()))
      indicateOptimisticFixpoint();
  }

  /// Determine if the underlying value may alias with the call site argument
  /// \p OtherArgNo of \p ICS (= the underlying call site).
  bool mayAliasWithArgument(Attributor &A, AAResults *&AAR,
                            const AAMemoryBehavior &MemBehaviorAA,
                            const CallBase &CB, unsigned OtherArgNo) {
    // We do not need to worry about aliasing with the underlying IRP.
    if (this->getCalleeArgNo() == (int)OtherArgNo)
      return false;

    // If it is not a pointer or pointer vector we do not alias.
    const Value *ArgOp = CB.getArgOperand(OtherArgNo);
    if (!ArgOp->getType()->isPtrOrPtrVectorTy())
      return false;

    auto &CBArgMemBehaviorAA = A.getAAFor<AAMemoryBehavior>(
        *this, IRPosition::callsite_argument(CB, OtherArgNo), DepClassTy::NONE);

    // If the argument is readnone, there is no read-write aliasing.
    if (CBArgMemBehaviorAA.isAssumedReadNone()) {
      A.recordDependence(CBArgMemBehaviorAA, *this, DepClassTy::OPTIONAL);
      return false;
    }

    // If the argument is readonly and the underlying value is readonly, there
    // is no read-write aliasing.
    bool IsReadOnly = MemBehaviorAA.isAssumedReadOnly();
    if (CBArgMemBehaviorAA.isAssumedReadOnly() && IsReadOnly) {
      A.recordDependence(MemBehaviorAA, *this, DepClassTy::OPTIONAL);
      A.recordDependence(CBArgMemBehaviorAA, *this, DepClassTy::OPTIONAL);
      return false;
    }

    // We have to utilize actual alias analysis queries so we need the object.
    if (!AAR)
      AAR = A.getInfoCache().getAAResultsForFunction(*getAnchorScope());

    // Try to rule it out at the call site.
    bool IsAliasing = !AAR || !AAR->isNoAlias(&getAssociatedValue(), ArgOp);
    LLVM_DEBUG(dbgs() << "[NoAliasCSArg] Check alias between "
                         "callsite arguments: "
                      << getAssociatedValue() << " " << *ArgOp << " => "
                      << (IsAliasing ? "" : "no-") << "alias \n");

    return IsAliasing;
  }

  bool
  isKnownNoAliasDueToNoAliasPreservation(Attributor &A, AAResults *&AAR,
                                         const AAMemoryBehavior &MemBehaviorAA,
                                         const AANoAlias &NoAliasAA) {
    // We can deduce "noalias" if the following conditions hold.
    // (i)   Associated value is assumed to be noalias in the definition.
    // (ii)  Associated value is assumed to be no-capture in all the uses
    //       possibly executed before this callsite.
    // (iii) There is no other pointer argument which could alias with the
    //       value.

    bool AssociatedValueIsNoAliasAtDef = NoAliasAA.isAssumedNoAlias();
    if (!AssociatedValueIsNoAliasAtDef) {
      LLVM_DEBUG(dbgs() << "[AANoAlias] " << getAssociatedValue()
                        << " is not no-alias at the definition\n");
      return false;
    }

    A.recordDependence(NoAliasAA, *this, DepClassTy::OPTIONAL);

    const IRPosition &VIRP = IRPosition::value(getAssociatedValue());
    const Function *ScopeFn = VIRP.getAnchorScope();
    auto &NoCaptureAA = A.getAAFor<AANoCapture>(*this, VIRP, DepClassTy::NONE);
    // Check whether the value is captured in the scope using AANoCapture.
    //      Look at CFG and check only uses possibly executed before this
    //      callsite.
    auto UsePred = [&](const Use &U, bool &Follow) -> bool {
      Instruction *UserI = cast<Instruction>(U.getUser());

      // If UserI is the curr instruction and there is a single potential use of
      // the value in UserI we allow the use.
      // TODO: We should inspect the operands and allow those that cannot alias
      //       with the value.
      if (UserI == getCtxI() && UserI->getNumOperands() == 1)
        return true;

      if (ScopeFn) {
        const auto &ReachabilityAA = A.getAAFor<AAReachability>(
            *this, IRPosition::function(*ScopeFn), DepClassTy::OPTIONAL);

        if (!ReachabilityAA.isAssumedReachable(A, *UserI, *getCtxI()))
          return true;

        if (auto *CB = dyn_cast<CallBase>(UserI)) {
          if (CB->isArgOperand(&U)) {

            unsigned ArgNo = CB->getArgOperandNo(&U);

            const auto &NoCaptureAA = A.getAAFor<AANoCapture>(
                *this, IRPosition::callsite_argument(*CB, ArgNo),
                DepClassTy::OPTIONAL);

            if (NoCaptureAA.isAssumedNoCapture())
              return true;
          }
        }
      }

      // For cases which can potentially have more users
      if (isa<GetElementPtrInst>(U) || isa<BitCastInst>(U) || isa<PHINode>(U) ||
          isa<SelectInst>(U)) {
        Follow = true;
        return true;
      }

      LLVM_DEBUG(dbgs() << "[AANoAliasCSArg] Unknown user: " << *U << "\n");
      return false;
    };

    if (!NoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
      if (!A.checkForAllUses(UsePred, *this, getAssociatedValue())) {
        LLVM_DEBUG(
            dbgs() << "[AANoAliasCSArg] " << getAssociatedValue()
                   << " cannot be noalias as it is potentially captured\n");
        return false;
      }
    }
    A.recordDependence(NoCaptureAA, *this, DepClassTy::OPTIONAL);

    // Check there is no other pointer argument which could alias with the
    // value passed at this call site.
    // TODO: AbstractCallSite
    const auto &CB = cast<CallBase>(getAnchorValue());
    for (unsigned OtherArgNo = 0; OtherArgNo < CB.getNumArgOperands();
         OtherArgNo++)
      if (mayAliasWithArgument(A, AAR, MemBehaviorAA, CB, OtherArgNo))
        return false;

    return true;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // If the argument is readnone we are done as there are no accesses via the
    // argument.
    auto &MemBehaviorAA =
        A.getAAFor<AAMemoryBehavior>(*this, getIRPosition(), DepClassTy::NONE);
    if (MemBehaviorAA.isAssumedReadNone()) {
      A.recordDependence(MemBehaviorAA, *this, DepClassTy::OPTIONAL);
      return ChangeStatus::UNCHANGED;
    }

    const IRPosition &VIRP = IRPosition::value(getAssociatedValue());
    const auto &NoAliasAA =
        A.getAAFor<AANoAlias>(*this, VIRP, DepClassTy::NONE);

    AAResults *AAR = nullptr;
    if (isKnownNoAliasDueToNoAliasPreservation(A, AAR, MemBehaviorAA,
                                               NoAliasAA)) {
      LLVM_DEBUG(
          dbgs() << "[AANoAlias] No-Alias deduced via no-alias preservation\n");
      return ChangeStatus::UNCHANGED;
    }

    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(noalias) }
};

/// NoAlias attribute for function return value.
struct AANoAliasReturned final : AANoAliasImpl {
  AANoAliasReturned(const IRPosition &IRP, Attributor &A)
      : AANoAliasImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoAliasImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  virtual ChangeStatus updateImpl(Attributor &A) override {

    auto CheckReturnValue = [&](Value &RV) -> bool {
      if (Constant *C = dyn_cast<Constant>(&RV))
        if (C->isNullValue() || isa<UndefValue>(C))
          return true;

      /// For now, we can only deduce noalias if we have call sites.
      /// FIXME: add more support.
      if (!isa<CallBase>(&RV))
        return false;

      const IRPosition &RVPos = IRPosition::value(RV);
      const auto &NoAliasAA =
          A.getAAFor<AANoAlias>(*this, RVPos, DepClassTy::REQUIRED);
      if (!NoAliasAA.isAssumedNoAlias())
        return false;

      const auto &NoCaptureAA =
          A.getAAFor<AANoCapture>(*this, RVPos, DepClassTy::REQUIRED);
      return NoCaptureAA.isAssumedNoCaptureMaybeReturned();
    };

    if (!A.checkForAllReturnedValues(CheckReturnValue, *this))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(noalias) }
};

/// NoAlias attribute deduction for a call site return value.
struct AANoAliasCallSiteReturned final : AANoAliasImpl {
  AANoAliasCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AANoAliasImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoAliasImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::returned(*F);
    auto &FnAA = A.getAAFor<AANoAlias>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(noalias); }
};

/// -------------------AAIsDead Function Attribute-----------------------

struct AAIsDeadValueImpl : public AAIsDead {
  AAIsDeadValueImpl(const IRPosition &IRP, Attributor &A) : AAIsDead(IRP, A) {}

  /// See AAIsDead::isAssumedDead().
  bool isAssumedDead() const override { return getAssumed(); }

  /// See AAIsDead::isKnownDead().
  bool isKnownDead() const override { return getKnown(); }

  /// See AAIsDead::isAssumedDead(BasicBlock *).
  bool isAssumedDead(const BasicBlock *BB) const override { return false; }

  /// See AAIsDead::isKnownDead(BasicBlock *).
  bool isKnownDead(const BasicBlock *BB) const override { return false; }

  /// See AAIsDead::isAssumedDead(Instruction *I).
  bool isAssumedDead(const Instruction *I) const override {
    return I == getCtxI() && isAssumedDead();
  }

  /// See AAIsDead::isKnownDead(Instruction *I).
  bool isKnownDead(const Instruction *I) const override {
    return isAssumedDead(I) && getKnown();
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return isAssumedDead() ? "assumed-dead" : "assumed-live";
  }

  /// Check if all uses are assumed dead.
  bool areAllUsesAssumedDead(Attributor &A, Value &V) {
    auto UsePred = [&](const Use &U, bool &Follow) { return false; };
    // Explicitly set the dependence class to required because we want a long
    // chain of N dependent instructions to be considered live as soon as one is
    // without going through N update cycles. This is not required for
    // correctness.
    return A.checkForAllUses(UsePred, *this, V, DepClassTy::REQUIRED);
  }

  /// Determine if \p I is assumed to be side-effect free.
  bool isAssumedSideEffectFree(Attributor &A, Instruction *I) {
    if (!I || wouldInstructionBeTriviallyDead(I))
      return true;

    auto *CB = dyn_cast<CallBase>(I);
    if (!CB || isa<IntrinsicInst>(CB))
      return false;

    const IRPosition &CallIRP = IRPosition::callsite_function(*CB);
    const auto &NoUnwindAA =
        A.getAndUpdateAAFor<AANoUnwind>(*this, CallIRP, DepClassTy::NONE);
    if (!NoUnwindAA.isAssumedNoUnwind())
      return false;
    if (!NoUnwindAA.isKnownNoUnwind())
      A.recordDependence(NoUnwindAA, *this, DepClassTy::OPTIONAL);

    const auto &MemBehaviorAA =
        A.getAndUpdateAAFor<AAMemoryBehavior>(*this, CallIRP, DepClassTy::NONE);
    if (MemBehaviorAA.isAssumedReadOnly()) {
      if (!MemBehaviorAA.isKnownReadOnly())
        A.recordDependence(MemBehaviorAA, *this, DepClassTy::OPTIONAL);
      return true;
    }
    return false;
  }
};

struct AAIsDeadFloating : public AAIsDeadValueImpl {
  AAIsDeadFloating(const IRPosition &IRP, Attributor &A)
      : AAIsDeadValueImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (isa<UndefValue>(getAssociatedValue())) {
      indicatePessimisticFixpoint();
      return;
    }

    Instruction *I = dyn_cast<Instruction>(&getAssociatedValue());
    if (!isAssumedSideEffectFree(A, I))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    Instruction *I = dyn_cast<Instruction>(&getAssociatedValue());
    if (!isAssumedSideEffectFree(A, I))
      return indicatePessimisticFixpoint();

    if (!areAllUsesAssumedDead(A, getAssociatedValue()))
      return indicatePessimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    Value &V = getAssociatedValue();
    if (auto *I = dyn_cast<Instruction>(&V)) {
      // If we get here we basically know the users are all dead. We check if
      // isAssumedSideEffectFree returns true here again because it might not be
      // the case and only the users are dead but the instruction (=call) is
      // still needed.
      if (isAssumedSideEffectFree(A, I) && !isa<InvokeInst>(I)) {
        A.deleteAfterManifest(*I);
        return ChangeStatus::CHANGED;
      }
    }
    if (V.use_empty())
      return ChangeStatus::UNCHANGED;

    bool UsedAssumedInformation = false;
    Optional<Constant *> C =
        A.getAssumedConstant(V, *this, UsedAssumedInformation);
    if (C.hasValue() && C.getValue())
      return ChangeStatus::UNCHANGED;

    // Replace the value with undef as it is dead but keep droppable uses around
    // as they provide information we don't want to give up on just yet.
    UndefValue &UV = *UndefValue::get(V.getType());
    bool AnyChange =
        A.changeValueAfterManifest(V, UV, /* ChangeDropppable */ false);
    return AnyChange ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(IsDead)
  }
};

struct AAIsDeadArgument : public AAIsDeadFloating {
  AAIsDeadArgument(const IRPosition &IRP, Attributor &A)
      : AAIsDeadFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (!A.isFunctionIPOAmendable(*getAnchorScope()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = AAIsDeadFloating::manifest(A);
    Argument &Arg = *getAssociatedArgument();
    if (A.isValidFunctionSignatureRewrite(Arg, /* ReplacementTypes */ {}))
      if (A.registerFunctionSignatureRewrite(
              Arg, /* ReplacementTypes */ {},
              Attributor::ArgumentReplacementInfo::CalleeRepairCBTy{},
              Attributor::ArgumentReplacementInfo::ACSRepairCBTy{})) {
        Arg.dropDroppableUses();
        return ChangeStatus::CHANGED;
      }
    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(IsDead) }
};

struct AAIsDeadCallSiteArgument : public AAIsDeadValueImpl {
  AAIsDeadCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAIsDeadValueImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (isa<UndefValue>(getAssociatedValue()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Argument *Arg = getAssociatedArgument();
    if (!Arg)
      return indicatePessimisticFixpoint();
    const IRPosition &ArgPos = IRPosition::argument(*Arg);
    auto &ArgAA = A.getAAFor<AAIsDead>(*this, ArgPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), ArgAA.getState());
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAnchorValue());
    Use &U = CB.getArgOperandUse(getCallSiteArgNo());
    assert(!isa<UndefValue>(U.get()) &&
           "Expected undef values to be filtered out!");
    UndefValue &UV = *UndefValue::get(U->getType());
    if (A.changeUseAfterManifest(U, UV))
      return ChangeStatus::CHANGED;
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(IsDead) }
};

struct AAIsDeadCallSiteReturned : public AAIsDeadFloating {
  AAIsDeadCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAIsDeadFloating(IRP, A), IsAssumedSideEffectFree(true) {}

  /// See AAIsDead::isAssumedDead().
  bool isAssumedDead() const override {
    return AAIsDeadFloating::isAssumedDead() && IsAssumedSideEffectFree;
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (isa<UndefValue>(getAssociatedValue())) {
      indicatePessimisticFixpoint();
      return;
    }

    // We track this separately as a secondary state.
    IsAssumedSideEffectFree = isAssumedSideEffectFree(A, getCtxI());
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    if (IsAssumedSideEffectFree && !isAssumedSideEffectFree(A, getCtxI())) {
      IsAssumedSideEffectFree = false;
      Changed = ChangeStatus::CHANGED;
    }

    if (!areAllUsesAssumedDead(A, getAssociatedValue()))
      return indicatePessimisticFixpoint();
    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (IsAssumedSideEffectFree)
      STATS_DECLTRACK_CSRET_ATTR(IsDead)
    else
      STATS_DECLTRACK_CSRET_ATTR(UnusedResult)
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return isAssumedDead()
               ? "assumed-dead"
               : (getAssumed() ? "assumed-dead-users" : "assumed-live");
  }

private:
  bool IsAssumedSideEffectFree;
};

struct AAIsDeadReturned : public AAIsDeadValueImpl {
  AAIsDeadReturned(const IRPosition &IRP, Attributor &A)
      : AAIsDeadValueImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {

    A.checkForAllInstructions([](Instruction &) { return true; }, *this,
                              {Instruction::Ret});

    auto PredForCallSite = [&](AbstractCallSite ACS) {
      if (ACS.isCallbackCall() || !ACS.getInstruction())
        return false;
      return areAllUsesAssumedDead(A, *ACS.getInstruction());
    };

    bool AllCallSitesKnown;
    if (!A.checkForAllCallSites(PredForCallSite, *this, true,
                                AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // TODO: Rewrite the signature to return void?
    bool AnyChange = false;
    UndefValue &UV = *UndefValue::get(getAssociatedFunction()->getReturnType());
    auto RetInstPred = [&](Instruction &I) {
      ReturnInst &RI = cast<ReturnInst>(I);
      if (!isa<UndefValue>(RI.getReturnValue()))
        AnyChange |= A.changeUseAfterManifest(RI.getOperandUse(0), UV);
      return true;
    };
    A.checkForAllInstructions(RetInstPred, *this, {Instruction::Ret});
    return AnyChange ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(IsDead) }
};

struct AAIsDeadFunction : public AAIsDead {
  AAIsDeadFunction(const IRPosition &IRP, Attributor &A) : AAIsDead(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    const Function *F = getAnchorScope();
    if (F && !F->isDeclaration()) {
      // We only want to compute liveness once. If the function is not part of
      // the SCC, skip it.
      if (A.isRunOn(*const_cast<Function *>(F))) {
        ToBeExploredFrom.insert(&F->getEntryBlock().front());
        assumeLive(A, F->getEntryBlock());
      } else {
        indicatePessimisticFixpoint();
      }
    }
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return "Live[#BB " + std::to_string(AssumedLiveBlocks.size()) + "/" +
           std::to_string(getAnchorScope()->size()) + "][#TBEP " +
           std::to_string(ToBeExploredFrom.size()) + "][#KDE " +
           std::to_string(KnownDeadEnds.size()) + "]";
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    assert(getState().isValidState() &&
           "Attempted to manifest an invalid state!");

    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
    Function &F = *getAnchorScope();

    if (AssumedLiveBlocks.empty()) {
      A.deleteAfterManifest(F);
      return ChangeStatus::CHANGED;
    }

    // Flag to determine if we can change an invoke to a call assuming the
    // callee is nounwind. This is not possible if the personality of the
    // function allows to catch asynchronous exceptions.
    bool Invoke2CallAllowed = !mayCatchAsynchronousExceptions(F);

    KnownDeadEnds.set_union(ToBeExploredFrom);
    for (const Instruction *DeadEndI : KnownDeadEnds) {
      auto *CB = dyn_cast<CallBase>(DeadEndI);
      if (!CB)
        continue;
      const auto &NoReturnAA = A.getAndUpdateAAFor<AANoReturn>(
          *this, IRPosition::callsite_function(*CB), DepClassTy::OPTIONAL);
      bool MayReturn = !NoReturnAA.isAssumedNoReturn();
      if (MayReturn && (!Invoke2CallAllowed || !isa<InvokeInst>(CB)))
        continue;

      if (auto *II = dyn_cast<InvokeInst>(DeadEndI))
        A.registerInvokeWithDeadSuccessor(const_cast<InvokeInst &>(*II));
      else
        A.changeToUnreachableAfterManifest(
            const_cast<Instruction *>(DeadEndI->getNextNode()));
      HasChanged = ChangeStatus::CHANGED;
    }

    STATS_DECL(AAIsDead, BasicBlock, "Number of dead basic blocks deleted.");
    for (BasicBlock &BB : F)
      if (!AssumedLiveBlocks.count(&BB)) {
        A.deleteAfterManifest(BB);
        ++BUILD_STAT_NAME(AAIsDead, BasicBlock);
      }

    return HasChanged;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  bool isEdgeDead(const BasicBlock *From, const BasicBlock *To) const override {
    return !AssumedLiveEdges.count(std::make_pair(From, To));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

  /// Returns true if the function is assumed dead.
  bool isAssumedDead() const override { return false; }

  /// See AAIsDead::isKnownDead().
  bool isKnownDead() const override { return false; }

  /// See AAIsDead::isAssumedDead(BasicBlock *).
  bool isAssumedDead(const BasicBlock *BB) const override {
    assert(BB->getParent() == getAnchorScope() &&
           "BB must be in the same anchor scope function.");

    if (!getAssumed())
      return false;
    return !AssumedLiveBlocks.count(BB);
  }

  /// See AAIsDead::isKnownDead(BasicBlock *).
  bool isKnownDead(const BasicBlock *BB) const override {
    return getKnown() && isAssumedDead(BB);
  }

  /// See AAIsDead::isAssumed(Instruction *I).
  bool isAssumedDead(const Instruction *I) const override {
    assert(I->getParent()->getParent() == getAnchorScope() &&
           "Instruction must be in the same anchor scope function.");

    if (!getAssumed())
      return false;

    // If it is not in AssumedLiveBlocks then it for sure dead.
    // Otherwise, it can still be after noreturn call in a live block.
    if (!AssumedLiveBlocks.count(I->getParent()))
      return true;

    // If it is not after a liveness barrier it is live.
    const Instruction *PrevI = I->getPrevNode();
    while (PrevI) {
      if (KnownDeadEnds.count(PrevI) || ToBeExploredFrom.count(PrevI))
        return true;
      PrevI = PrevI->getPrevNode();
    }
    return false;
  }

  /// See AAIsDead::isKnownDead(Instruction *I).
  bool isKnownDead(const Instruction *I) const override {
    return getKnown() && isAssumedDead(I);
  }

  /// Assume \p BB is (partially) live now and indicate to the Attributor \p A
  /// that internal function called from \p BB should now be looked at.
  bool assumeLive(Attributor &A, const BasicBlock &BB) {
    if (!AssumedLiveBlocks.insert(&BB).second)
      return false;

    // We assume that all of BB is (probably) live now and if there are calls to
    // internal functions we will assume that those are now live as well. This
    // is a performance optimization for blocks with calls to a lot of internal
    // functions. It can however cause dead functions to be treated as live.
    for (const Instruction &I : BB)
      if (const auto *CB = dyn_cast<CallBase>(&I))
        if (const Function *F = CB->getCalledFunction())
          if (F->hasLocalLinkage())
            A.markLiveInternalFunction(*F);
    return true;
  }

  /// Collection of instructions that need to be explored again, e.g., we
  /// did assume they do not transfer control to (one of their) successors.
  SmallSetVector<const Instruction *, 8> ToBeExploredFrom;

  /// Collection of instructions that are known to not transfer control.
  SmallSetVector<const Instruction *, 8> KnownDeadEnds;

  /// Collection of all assumed live edges
  DenseSet<std::pair<const BasicBlock *, const BasicBlock *>> AssumedLiveEdges;

  /// Collection of all assumed live BasicBlocks.
  DenseSet<const BasicBlock *> AssumedLiveBlocks;
};

static bool
identifyAliveSuccessors(Attributor &A, const CallBase &CB,
                        AbstractAttribute &AA,
                        SmallVectorImpl<const Instruction *> &AliveSuccessors) {
  const IRPosition &IPos = IRPosition::callsite_function(CB);

  const auto &NoReturnAA =
      A.getAndUpdateAAFor<AANoReturn>(AA, IPos, DepClassTy::OPTIONAL);
  if (NoReturnAA.isAssumedNoReturn())
    return !NoReturnAA.isKnownNoReturn();
  if (CB.isTerminator())
    AliveSuccessors.push_back(&CB.getSuccessor(0)->front());
  else
    AliveSuccessors.push_back(CB.getNextNode());
  return false;
}

static bool
identifyAliveSuccessors(Attributor &A, const InvokeInst &II,
                        AbstractAttribute &AA,
                        SmallVectorImpl<const Instruction *> &AliveSuccessors) {
  bool UsedAssumedInformation =
      identifyAliveSuccessors(A, cast<CallBase>(II), AA, AliveSuccessors);

  // First, determine if we can change an invoke to a call assuming the
  // callee is nounwind. This is not possible if the personality of the
  // function allows to catch asynchronous exceptions.
  if (AAIsDeadFunction::mayCatchAsynchronousExceptions(*II.getFunction())) {
    AliveSuccessors.push_back(&II.getUnwindDest()->front());
  } else {
    const IRPosition &IPos = IRPosition::callsite_function(II);
    const auto &AANoUnw =
        A.getAndUpdateAAFor<AANoUnwind>(AA, IPos, DepClassTy::OPTIONAL);
    if (AANoUnw.isAssumedNoUnwind()) {
      UsedAssumedInformation |= !AANoUnw.isKnownNoUnwind();
    } else {
      AliveSuccessors.push_back(&II.getUnwindDest()->front());
    }
  }
  return UsedAssumedInformation;
}

static bool
identifyAliveSuccessors(Attributor &A, const BranchInst &BI,
                        AbstractAttribute &AA,
                        SmallVectorImpl<const Instruction *> &AliveSuccessors) {
  bool UsedAssumedInformation = false;
  if (BI.getNumSuccessors() == 1) {
    AliveSuccessors.push_back(&BI.getSuccessor(0)->front());
  } else {
    Optional<ConstantInt *> CI = getAssumedConstantInt(
        A, *BI.getCondition(), AA, UsedAssumedInformation);
    if (!CI.hasValue()) {
      // No value yet, assume both edges are dead.
    } else if (CI.getValue()) {
      const BasicBlock *SuccBB =
          BI.getSuccessor(1 - CI.getValue()->getZExtValue());
      AliveSuccessors.push_back(&SuccBB->front());
    } else {
      AliveSuccessors.push_back(&BI.getSuccessor(0)->front());
      AliveSuccessors.push_back(&BI.getSuccessor(1)->front());
      UsedAssumedInformation = false;
    }
  }
  return UsedAssumedInformation;
}

static bool
identifyAliveSuccessors(Attributor &A, const SwitchInst &SI,
                        AbstractAttribute &AA,
                        SmallVectorImpl<const Instruction *> &AliveSuccessors) {
  bool UsedAssumedInformation = false;
  Optional<ConstantInt *> CI =
      getAssumedConstantInt(A, *SI.getCondition(), AA, UsedAssumedInformation);
  if (!CI.hasValue()) {
    // No value yet, assume all edges are dead.
  } else if (CI.getValue()) {
    for (auto &CaseIt : SI.cases()) {
      if (CaseIt.getCaseValue() == CI.getValue()) {
        AliveSuccessors.push_back(&CaseIt.getCaseSuccessor()->front());
        return UsedAssumedInformation;
      }
    }
    AliveSuccessors.push_back(&SI.getDefaultDest()->front());
    return UsedAssumedInformation;
  } else {
    for (const BasicBlock *SuccBB : successors(SI.getParent()))
      AliveSuccessors.push_back(&SuccBB->front());
  }
  return UsedAssumedInformation;
}

ChangeStatus AAIsDeadFunction::updateImpl(Attributor &A) {
  ChangeStatus Change = ChangeStatus::UNCHANGED;

  LLVM_DEBUG(dbgs() << "[AAIsDead] Live [" << AssumedLiveBlocks.size() << "/"
                    << getAnchorScope()->size() << "] BBs and "
                    << ToBeExploredFrom.size() << " exploration points and "
                    << KnownDeadEnds.size() << " known dead ends\n");

  // Copy and clear the list of instructions we need to explore from. It is
  // refilled with instructions the next update has to look at.
  SmallVector<const Instruction *, 8> Worklist(ToBeExploredFrom.begin(),
                                               ToBeExploredFrom.end());
  decltype(ToBeExploredFrom) NewToBeExploredFrom;

  SmallVector<const Instruction *, 8> AliveSuccessors;
  while (!Worklist.empty()) {
    const Instruction *I = Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "[AAIsDead] Exploration inst: " << *I << "\n");

    // Fast forward for uninteresting instructions. We could look for UB here
    // though.
    while (!I->isTerminator() && !isa<CallBase>(I)) {
      Change = ChangeStatus::CHANGED;
      I = I->getNextNode();
    }

    AliveSuccessors.clear();

    bool UsedAssumedInformation = false;
    switch (I->getOpcode()) {
    // TODO: look for (assumed) UB to backwards propagate "deadness".
    default:
      assert(I->isTerminator() &&
             "Expected non-terminators to be handled already!");
      for (const BasicBlock *SuccBB : successors(I->getParent()))
        AliveSuccessors.push_back(&SuccBB->front());
      break;
    case Instruction::Call:
      UsedAssumedInformation = identifyAliveSuccessors(A, cast<CallInst>(*I),
                                                       *this, AliveSuccessors);
      break;
    case Instruction::Invoke:
      UsedAssumedInformation = identifyAliveSuccessors(A, cast<InvokeInst>(*I),
                                                       *this, AliveSuccessors);
      break;
    case Instruction::Br:
      UsedAssumedInformation = identifyAliveSuccessors(A, cast<BranchInst>(*I),
                                                       *this, AliveSuccessors);
      break;
    case Instruction::Switch:
      UsedAssumedInformation = identifyAliveSuccessors(A, cast<SwitchInst>(*I),
                                                       *this, AliveSuccessors);
      break;
    }

    if (UsedAssumedInformation) {
      NewToBeExploredFrom.insert(I);
    } else {
      Change = ChangeStatus::CHANGED;
      if (AliveSuccessors.empty() ||
          (I->isTerminator() && AliveSuccessors.size() < I->getNumSuccessors()))
        KnownDeadEnds.insert(I);
    }

    LLVM_DEBUG(dbgs() << "[AAIsDead] #AliveSuccessors: "
                      << AliveSuccessors.size() << " UsedAssumedInformation: "
                      << UsedAssumedInformation << "\n");

    for (const Instruction *AliveSuccessor : AliveSuccessors) {
      if (!I->isTerminator()) {
        assert(AliveSuccessors.size() == 1 &&
               "Non-terminator expected to have a single successor!");
        Worklist.push_back(AliveSuccessor);
      } else {
        // record the assumed live edge
        AssumedLiveEdges.insert(
            std::make_pair(I->getParent(), AliveSuccessor->getParent()));
        if (assumeLive(A, *AliveSuccessor->getParent()))
          Worklist.push_back(AliveSuccessor);
      }
    }
  }

  ToBeExploredFrom = std::move(NewToBeExploredFrom);

  // If we know everything is live there is no need to query for liveness.
  // Instead, indicating a pessimistic fixpoint will cause the state to be
  // "invalid" and all queries to be answered conservatively without lookups.
  // To be in this state we have to (1) finished the exploration and (3) not
  // discovered any non-trivial dead end and (2) not ruled unreachable code
  // dead.
  if (ToBeExploredFrom.empty() &&
      getAnchorScope()->size() == AssumedLiveBlocks.size() &&
      llvm::all_of(KnownDeadEnds, [](const Instruction *DeadEndI) {
        return DeadEndI->isTerminator() && DeadEndI->getNumSuccessors() == 0;
      }))
    return indicatePessimisticFixpoint();
  return Change;
}

/// Liveness information for a call sites.
struct AAIsDeadCallSite final : AAIsDeadFunction {
  AAIsDeadCallSite(const IRPosition &IRP, Attributor &A)
      : AAIsDeadFunction(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites instead of
    //       redirecting requests to the callee.
    llvm_unreachable("Abstract attributes for liveness are not "
                     "supported for call sites yet!");
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

/// -------------------- Dereferenceable Argument Attribute --------------------

template <>
ChangeStatus clampStateAndIndicateChange<DerefState>(DerefState &S,
                                                     const DerefState &R) {
  ChangeStatus CS0 =
      clampStateAndIndicateChange(S.DerefBytesState, R.DerefBytesState);
  ChangeStatus CS1 = clampStateAndIndicateChange(S.GlobalState, R.GlobalState);
  return CS0 | CS1;
}

struct AADereferenceableImpl : AADereferenceable {
  AADereferenceableImpl(const IRPosition &IRP, Attributor &A)
      : AADereferenceable(IRP, A) {}
  using StateType = DerefState;

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    SmallVector<Attribute, 4> Attrs;
    getAttrs({Attribute::Dereferenceable, Attribute::DereferenceableOrNull},
             Attrs, /* IgnoreSubsumingPositions */ false, &A);
    for (const Attribute &Attr : Attrs)
      takeKnownDerefBytesMaximum(Attr.getValueAsInt());

    const IRPosition &IRP = this->getIRPosition();
    NonNullAA = &A.getAAFor<AANonNull>(*this, IRP, DepClassTy::NONE);

    bool CanBeNull;
    takeKnownDerefBytesMaximum(
        IRP.getAssociatedValue().getPointerDereferenceableBytes(
            A.getDataLayout(), CanBeNull));

    bool IsFnInterface = IRP.isFnInterfaceKind();
    Function *FnScope = IRP.getAnchorScope();
    if (IsFnInterface && (!FnScope || !A.isFunctionIPOAmendable(*FnScope))) {
      indicatePessimisticFixpoint();
      return;
    }

    if (Instruction *CtxI = getCtxI())
      followUsesInMBEC(*this, A, getState(), *CtxI);
  }

  /// See AbstractAttribute::getState()
  /// {
  StateType &getState() override { return *this; }
  const StateType &getState() const override { return *this; }
  /// }

  /// Helper function for collecting accessed bytes in must-be-executed-context
  void addAccessedBytesForUse(Attributor &A, const Use *U, const Instruction *I,
                              DerefState &State) {
    const Value *UseV = U->get();
    if (!UseV->getType()->isPointerTy())
      return;

    Type *PtrTy = UseV->getType();
    const DataLayout &DL = A.getDataLayout();
    int64_t Offset;
    if (const Value *Base = getBasePointerOfAccessPointerOperand(
            I, Offset, DL, /*AllowNonInbounds*/ true)) {
      if (Base == &getAssociatedValue() &&
          getPointerOperand(I, /* AllowVolatile */ false) == UseV) {
        uint64_t Size = DL.getTypeStoreSize(PtrTy->getPointerElementType());
        State.addAccessedBytes(Offset, Size);
      }
    }
  }

  /// See followUsesInMBEC
  bool followUseInMBEC(Attributor &A, const Use *U, const Instruction *I,
                       AADereferenceable::StateType &State) {
    bool IsNonNull = false;
    bool TrackUse = false;
    int64_t DerefBytes = getKnownNonNullAndDerefBytesForUse(
        A, *this, getAssociatedValue(), U, I, IsNonNull, TrackUse);
    LLVM_DEBUG(dbgs() << "[AADereferenceable] Deref bytes: " << DerefBytes
                      << " for instruction " << *I << "\n");

    addAccessedBytesForUse(A, U, I, State);
    State.takeKnownDerefBytesMaximum(DerefBytes);
    return TrackUse;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Change = AADereferenceable::manifest(A);
    if (isAssumedNonNull() && hasAttr(Attribute::DereferenceableOrNull)) {
      removeAttrs({Attribute::DereferenceableOrNull});
      return ChangeStatus::CHANGED;
    }
    return Change;
  }

  void getDeducedAttributes(LLVMContext &Ctx,
                            SmallVectorImpl<Attribute> &Attrs) const override {
    // TODO: Add *_globally support
    if (isAssumedNonNull())
      Attrs.emplace_back(Attribute::getWithDereferenceableBytes(
          Ctx, getAssumedDereferenceableBytes()));
    else
      Attrs.emplace_back(Attribute::getWithDereferenceableOrNullBytes(
          Ctx, getAssumedDereferenceableBytes()));
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    if (!getAssumedDereferenceableBytes())
      return "unknown-dereferenceable";
    return std::string("dereferenceable") +
           (isAssumedNonNull() ? "" : "_or_null") +
           (isAssumedGlobal() ? "_globally" : "") + "<" +
           std::to_string(getKnownDereferenceableBytes()) + "-" +
           std::to_string(getAssumedDereferenceableBytes()) + ">";
  }
};

/// Dereferenceable attribute for a floating value.
struct AADereferenceableFloating : AADereferenceableImpl {
  AADereferenceableFloating(const IRPosition &IRP, Attributor &A)
      : AADereferenceableImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    auto VisitValueCB = [&](const Value &V, const Instruction *, DerefState &T,
                            bool Stripped) -> bool {
      unsigned IdxWidth =
          DL.getIndexSizeInBits(V.getType()->getPointerAddressSpace());
      APInt Offset(IdxWidth, 0);
      const Value *Base =
          stripAndAccumulateMinimalOffsets(A, *this, &V, DL, Offset, false);

      const auto &AA = A.getAAFor<AADereferenceable>(
          *this, IRPosition::value(*Base), DepClassTy::REQUIRED);
      int64_t DerefBytes = 0;
      if (!Stripped && this == &AA) {
        // Use IR information if we did not strip anything.
        // TODO: track globally.
        bool CanBeNull;
        DerefBytes = Base->getPointerDereferenceableBytes(DL, CanBeNull);
        T.GlobalState.indicatePessimisticFixpoint();
      } else {
        const DerefState &DS = AA.getState();
        DerefBytes = DS.DerefBytesState.getAssumed();
        T.GlobalState &= DS.GlobalState;
      }

      // For now we do not try to "increase" dereferenceability due to negative
      // indices as we first have to come up with code to deal with loops and
      // for overflows of the dereferenceable bytes.
      int64_t OffsetSExt = Offset.getSExtValue();
      if (OffsetSExt < 0)
        OffsetSExt = 0;

      T.takeAssumedDerefBytesMinimum(
          std::max(int64_t(0), DerefBytes - OffsetSExt));

      if (this == &AA) {
        if (!Stripped) {
          // If nothing was stripped IR information is all we got.
          T.takeKnownDerefBytesMaximum(
              std::max(int64_t(0), DerefBytes - OffsetSExt));
          T.indicatePessimisticFixpoint();
        } else if (OffsetSExt > 0) {
          // If something was stripped but there is circular reasoning we look
          // for the offset. If it is positive we basically decrease the
          // dereferenceable bytes in a circluar loop now, which will simply
          // drive them down to the known value in a very slow way which we
          // can accelerate.
          T.indicatePessimisticFixpoint();
        }
      }

      return T.isValidState();
    };

    DerefState T;
    if (!genericValueTraversal<AADereferenceable, DerefState>(
            A, getIRPosition(), *this, T, VisitValueCB, getCtxI()))
      return indicatePessimisticFixpoint();

    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute for a return value.
struct AADereferenceableReturned final
    : AAReturnedFromReturnedValues<AADereferenceable, AADereferenceableImpl> {
  AADereferenceableReturned(const IRPosition &IRP, Attributor &A)
      : AAReturnedFromReturnedValues<AADereferenceable, AADereferenceableImpl>(
            IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute for an argument
struct AADereferenceableArgument final
    : AAArgumentFromCallSiteArguments<AADereferenceable,
                                      AADereferenceableImpl> {
  using Base =
      AAArgumentFromCallSiteArguments<AADereferenceable, AADereferenceableImpl>;
  AADereferenceableArgument(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute for a call site argument.
struct AADereferenceableCallSiteArgument final : AADereferenceableFloating {
  AADereferenceableCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AADereferenceableFloating(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute deduction for a call site return value.
struct AADereferenceableCallSiteReturned final
    : AACallSiteReturnedFromReturned<AADereferenceable, AADereferenceableImpl> {
  using Base =
      AACallSiteReturnedFromReturned<AADereferenceable, AADereferenceableImpl>;
  AADereferenceableCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(dereferenceable);
  }
};

// ------------------------ Align Argument Attribute ------------------------

static unsigned getKnownAlignForUse(Attributor &A, AAAlign &QueryingAA,
                                    Value &AssociatedValue, const Use *U,
                                    const Instruction *I, bool &TrackUse) {
  // We need to follow common pointer manipulation uses to the accesses they
  // feed into.
  if (isa<CastInst>(I)) {
    // Follow all but ptr2int casts.
    TrackUse = !isa<PtrToIntInst>(I);
    return 0;
  }
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    if (GEP->hasAllConstantIndices())
      TrackUse = true;
    return 0;
  }

  MaybeAlign MA;
  if (const auto *CB = dyn_cast<CallBase>(I)) {
    if (CB->isBundleOperand(U) || CB->isCallee(U))
      return 0;

    unsigned ArgNo = CB->getArgOperandNo(U);
    IRPosition IRP = IRPosition::callsite_argument(*CB, ArgNo);
    // As long as we only use known information there is no need to track
    // dependences here.
    auto &AlignAA = A.getAAFor<AAAlign>(QueryingAA, IRP, DepClassTy::NONE);
    MA = MaybeAlign(AlignAA.getKnownAlign());
  }

  const DataLayout &DL = A.getDataLayout();
  const Value *UseV = U->get();
  if (auto *SI = dyn_cast<StoreInst>(I)) {
    if (SI->getPointerOperand() == UseV)
      MA = SI->getAlign();
  } else if (auto *LI = dyn_cast<LoadInst>(I)) {
    if (LI->getPointerOperand() == UseV)
      MA = LI->getAlign();
  }

  if (!MA || *MA <= QueryingAA.getKnownAlign())
    return 0;

  unsigned Alignment = MA->value();
  int64_t Offset;

  if (const Value *Base = GetPointerBaseWithConstantOffset(UseV, Offset, DL)) {
    if (Base == &AssociatedValue) {
      // BasePointerAddr + Offset = Alignment * Q for some integer Q.
      // So we can say that the maximum power of two which is a divisor of
      // gcd(Offset, Alignment) is an alignment.

      uint32_t gcd =
          greatestCommonDivisor(uint32_t(abs((int32_t)Offset)), Alignment);
      Alignment = llvm::PowerOf2Floor(gcd);
    }
  }

  return Alignment;
}

struct AAAlignImpl : AAAlign {
  AAAlignImpl(const IRPosition &IRP, Attributor &A) : AAAlign(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    SmallVector<Attribute, 4> Attrs;
    getAttrs({Attribute::Alignment}, Attrs);
    for (const Attribute &Attr : Attrs)
      takeKnownMaximum(Attr.getValueAsInt());

    Value &V = getAssociatedValue();
    // TODO: This is a HACK to avoid getPointerAlignment to introduce a ptr2int
    //       use of the function pointer. This was caused by D73131. We want to
    //       avoid this for function pointers especially because we iterate
    //       their uses and int2ptr is not handled. It is not a correctness
    //       problem though!
    if (!V.getType()->getPointerElementType()->isFunctionTy())
      takeKnownMaximum(V.getPointerAlignment(A.getDataLayout()).value());

    if (getIRPosition().isFnInterfaceKind() &&
        (!getAnchorScope() ||
         !A.isFunctionIPOAmendable(*getAssociatedFunction()))) {
      indicatePessimisticFixpoint();
      return;
    }

    if (Instruction *CtxI = getCtxI())
      followUsesInMBEC(*this, A, getState(), *CtxI);
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus LoadStoreChanged = ChangeStatus::UNCHANGED;

    // Check for users that allow alignment annotations.
    Value &AssociatedValue = getAssociatedValue();
    for (const Use &U : AssociatedValue.uses()) {
      if (auto *SI = dyn_cast<StoreInst>(U.getUser())) {
        if (SI->getPointerOperand() == &AssociatedValue)
          if (SI->getAlignment() < getAssumedAlign()) {
            STATS_DECLTRACK(AAAlign, Store,
                            "Number of times alignment added to a store");
            SI->setAlignment(Align(getAssumedAlign()));
            LoadStoreChanged = ChangeStatus::CHANGED;
          }
      } else if (auto *LI = dyn_cast<LoadInst>(U.getUser())) {
        if (LI->getPointerOperand() == &AssociatedValue)
          if (LI->getAlignment() < getAssumedAlign()) {
            LI->setAlignment(Align(getAssumedAlign()));
            STATS_DECLTRACK(AAAlign, Load,
                            "Number of times alignment added to a load");
            LoadStoreChanged = ChangeStatus::CHANGED;
          }
      }
    }

    ChangeStatus Changed = AAAlign::manifest(A);

    Align InheritAlign =
        getAssociatedValue().getPointerAlignment(A.getDataLayout());
    if (InheritAlign >= getAssumedAlign())
      return LoadStoreChanged;
    return Changed | LoadStoreChanged;
  }

  // TODO: Provide a helper to determine the implied ABI alignment and check in
  //       the existing manifest method and a new one for AAAlignImpl that value
  //       to avoid making the alignment explicit if it did not improve.

  /// See AbstractAttribute::getDeducedAttributes
  virtual void
  getDeducedAttributes(LLVMContext &Ctx,
                       SmallVectorImpl<Attribute> &Attrs) const override {
    if (getAssumedAlign() > 1)
      Attrs.emplace_back(
          Attribute::getWithAlignment(Ctx, Align(getAssumedAlign())));
  }

  /// See followUsesInMBEC
  bool followUseInMBEC(Attributor &A, const Use *U, const Instruction *I,
                       AAAlign::StateType &State) {
    bool TrackUse = false;

    unsigned int KnownAlign =
        getKnownAlignForUse(A, *this, getAssociatedValue(), U, I, TrackUse);
    State.takeKnownMaximum(KnownAlign);

    return TrackUse;
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumedAlign() ? ("align<" + std::to_string(getKnownAlign()) +
                                "-" + std::to_string(getAssumedAlign()) + ">")
                             : "unknown-align";
  }
};

/// Align attribute for a floating value.
struct AAAlignFloating : AAAlignImpl {
  AAAlignFloating(const IRPosition &IRP, Attributor &A) : AAAlignImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    auto VisitValueCB = [&](Value &V, const Instruction *,
                            AAAlign::StateType &T, bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AAAlign>(*this, IRPosition::value(V),
                                           DepClassTy::REQUIRED);
      if (!Stripped && this == &AA) {
        int64_t Offset;
        unsigned Alignment = 1;
        if (const Value *Base =
                GetPointerBaseWithConstantOffset(&V, Offset, DL)) {
          Align PA = Base->getPointerAlignment(DL);
          // BasePointerAddr + Offset = Alignment * Q for some integer Q.
          // So we can say that the maximum power of two which is a divisor of
          // gcd(Offset, Alignment) is an alignment.

          uint32_t gcd = greatestCommonDivisor(uint32_t(abs((int32_t)Offset)),
                                               uint32_t(PA.value()));
          Alignment = llvm::PowerOf2Floor(gcd);
        } else {
          Alignment = V.getPointerAlignment(DL).value();
        }
        // Use only IR information if we did not strip anything.
        T.takeKnownMaximum(Alignment);
        T.indicatePessimisticFixpoint();
      } else {
        // Use abstract attribute information.
        const AAAlign::StateType &DS = AA.getState();
        T ^= DS;
      }
      return T.isValidState();
    };

    StateType T;
    if (!genericValueTraversal<AAAlign, StateType>(A, getIRPosition(), *this, T,
                                                   VisitValueCB, getCtxI()))
      return indicatePessimisticFixpoint();

    // TODO: If we know we visited all incoming values, thus no are assumed
    // dead, we can take the known information from the state T.
    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FLOATING_ATTR(align) }
};

/// Align attribute for function return value.
struct AAAlignReturned final
    : AAReturnedFromReturnedValues<AAAlign, AAAlignImpl> {
  using Base = AAReturnedFromReturnedValues<AAAlign, AAAlignImpl>;
  AAAlignReturned(const IRPosition &IRP, Attributor &A) : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Base::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(aligned) }
};

/// Align attribute for function argument.
struct AAAlignArgument final
    : AAArgumentFromCallSiteArguments<AAAlign, AAAlignImpl> {
  using Base = AAArgumentFromCallSiteArguments<AAAlign, AAAlignImpl>;
  AAAlignArgument(const IRPosition &IRP, Attributor &A) : Base(IRP, A) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // If the associated argument is involved in a must-tail call we give up
    // because we would need to keep the argument alignments of caller and
    // callee in-sync. Just does not seem worth the trouble right now.
    if (A.getInfoCache().isInvolvedInMustTailCall(*getAssociatedArgument()))
      return ChangeStatus::UNCHANGED;
    return Base::manifest(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(aligned) }
};

struct AAAlignCallSiteArgument final : AAAlignFloating {
  AAAlignCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAAlignFloating(IRP, A) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // If the associated argument is involved in a must-tail call we give up
    // because we would need to keep the argument alignments of caller and
    // callee in-sync. Just does not seem worth the trouble right now.
    if (Argument *Arg = getAssociatedArgument())
      if (A.getInfoCache().isInvolvedInMustTailCall(*Arg))
        return ChangeStatus::UNCHANGED;
    ChangeStatus Changed = AAAlignImpl::manifest(A);
    Align InheritAlign =
        getAssociatedValue().getPointerAlignment(A.getDataLayout());
    if (InheritAlign >= getAssumedAlign())
      Changed = ChangeStatus::UNCHANGED;
    return Changed;
  }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = AAAlignFloating::updateImpl(A);
    if (Argument *Arg = getAssociatedArgument()) {
      // We only take known information from the argument
      // so we do not need to track a dependence.
      const auto &ArgAlignAA = A.getAAFor<AAAlign>(
          *this, IRPosition::argument(*Arg), DepClassTy::NONE);
      takeKnownMaximum(ArgAlignAA.getKnownAlign());
    }
    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(aligned) }
};

/// Align attribute deduction for a call site return value.
struct AAAlignCallSiteReturned final
    : AACallSiteReturnedFromReturned<AAAlign, AAAlignImpl> {
  using Base = AACallSiteReturnedFromReturned<AAAlign, AAAlignImpl>;
  AAAlignCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Base::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(align); }
};

/// ------------------ Function No-Return Attribute ----------------------------
struct AANoReturnImpl : public AANoReturn {
  AANoReturnImpl(const IRPosition &IRP, Attributor &A) : AANoReturn(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoReturn::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "noreturn" : "may-return";
  }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  virtual ChangeStatus updateImpl(Attributor &A) override {
    auto CheckForNoReturn = [](Instruction &) { return false; };
    if (!A.checkForAllInstructions(CheckForNoReturn, *this,
                                   {(unsigned)Instruction::Ret}))
      return indicatePessimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }
};

struct AANoReturnFunction final : AANoReturnImpl {
  AANoReturnFunction(const IRPosition &IRP, Attributor &A)
      : AANoReturnImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(noreturn) }
};

/// NoReturn attribute deduction for a call sites.
struct AANoReturnCallSite final : AANoReturnImpl {
  AANoReturnCallSite(const IRPosition &IRP, Attributor &A)
      : AANoReturnImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoReturnImpl::initialize(A);
    if (Function *F = getAssociatedFunction()) {
      const IRPosition &FnPos = IRPosition::function(*F);
      auto &FnAA = A.getAAFor<AANoReturn>(*this, FnPos, DepClassTy::REQUIRED);
      if (!FnAA.isAssumedNoReturn())
        indicatePessimisticFixpoint();
    }
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA = A.getAAFor<AANoReturn>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(noreturn); }
};

/// ----------------------- Variable Capturing ---------------------------------

/// A class to hold the state of for no-capture attributes.
struct AANoCaptureImpl : public AANoCapture {
  AANoCaptureImpl(const IRPosition &IRP, Attributor &A) : AANoCapture(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr(getAttrKind(), /* IgnoreSubsumingPositions */ true)) {
      indicateOptimisticFixpoint();
      return;
    }
    Function *AnchorScope = getAnchorScope();
    if (isFnInterfaceKind() &&
        (!AnchorScope || !A.isFunctionIPOAmendable(*AnchorScope))) {
      indicatePessimisticFixpoint();
      return;
    }

    // You cannot "capture" null in the default address space.
    if (isa<ConstantPointerNull>(getAssociatedValue()) &&
        getAssociatedValue().getType()->getPointerAddressSpace() == 0) {
      indicateOptimisticFixpoint();
      return;
    }

    const Function *F =
        isArgumentPosition() ? getAssociatedFunction() : AnchorScope;

    // Check what state the associated function can actually capture.
    if (F)
      determineFunctionCaptureCapabilities(getIRPosition(), *F, *this);
    else
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// see AbstractAttribute::isAssumedNoCaptureMaybeReturned(...).
  virtual void
  getDeducedAttributes(LLVMContext &Ctx,
                       SmallVectorImpl<Attribute> &Attrs) const override {
    if (!isAssumedNoCaptureMaybeReturned())
      return;

    if (isArgumentPosition()) {
      if (isAssumedNoCapture())
        Attrs.emplace_back(Attribute::get(Ctx, Attribute::NoCapture));
      else if (ManifestInternal)
        Attrs.emplace_back(Attribute::get(Ctx, "no-capture-maybe-returned"));
    }
  }

  /// Set the NOT_CAPTURED_IN_MEM and NOT_CAPTURED_IN_RET bits in \p Known
  /// depending on the ability of the function associated with \p IRP to capture
  /// state in memory and through "returning/throwing", respectively.
  static void determineFunctionCaptureCapabilities(const IRPosition &IRP,
                                                   const Function &F,
                                                   BitIntegerState &State) {
    // TODO: Once we have memory behavior attributes we should use them here.

    // If we know we cannot communicate or write to memory, we do not care about
    // ptr2int anymore.
    if (F.onlyReadsMemory() && F.doesNotThrow() &&
        F.getReturnType()->isVoidTy()) {
      State.addKnownBits(NO_CAPTURE);
      return;
    }

    // A function cannot capture state in memory if it only reads memory, it can
    // however return/throw state and the state might be influenced by the
    // pointer value, e.g., loading from a returned pointer might reveal a bit.
    if (F.onlyReadsMemory())
      State.addKnownBits(NOT_CAPTURED_IN_MEM);

    // A function cannot communicate state back if it does not through
    // exceptions and doesn not return values.
    if (F.doesNotThrow() && F.getReturnType()->isVoidTy())
      State.addKnownBits(NOT_CAPTURED_IN_RET);

    // Check existing "returned" attributes.
    int ArgNo = IRP.getCalleeArgNo();
    if (F.doesNotThrow() && ArgNo >= 0) {
      for (unsigned u = 0, e = F.arg_size(); u < e; ++u)
        if (F.hasParamAttribute(u, Attribute::Returned)) {
          if (u == unsigned(ArgNo))
            State.removeAssumedBits(NOT_CAPTURED_IN_RET);
          else if (F.onlyReadsMemory())
            State.addKnownBits(NO_CAPTURE);
          else
            State.addKnownBits(NOT_CAPTURED_IN_RET);
          break;
        }
    }
  }

  /// See AbstractState::getAsStr().
  const std::string getAsStr() const override {
    if (isKnownNoCapture())
      return "known not-captured";
    if (isAssumedNoCapture())
      return "assumed not-captured";
    if (isKnownNoCaptureMaybeReturned())
      return "known not-captured-maybe-returned";
    if (isAssumedNoCaptureMaybeReturned())
      return "assumed not-captured-maybe-returned";
    return "assumed-captured";
  }
};

/// Attributor-aware capture tracker.
struct AACaptureUseTracker final : public CaptureTracker {

  /// Create a capture tracker that can lookup in-flight abstract attributes
  /// through the Attributor \p A.
  ///
  /// If a use leads to a potential capture, \p CapturedInMemory is set and the
  /// search is stopped. If a use leads to a return instruction,
  /// \p CommunicatedBack is set to true and \p CapturedInMemory is not changed.
  /// If a use leads to a ptr2int which may capture the value,
  /// \p CapturedInInteger is set. If a use is found that is currently assumed
  /// "no-capture-maybe-returned", the user is added to the \p PotentialCopies
  /// set. All values in \p PotentialCopies are later tracked as well. For every
  /// explored use we decrement \p RemainingUsesToExplore. Once it reaches 0,
  /// the search is stopped with \p CapturedInMemory and \p CapturedInInteger
  /// conservatively set to true.
  AACaptureUseTracker(Attributor &A, AANoCapture &NoCaptureAA,
                      const AAIsDead &IsDeadAA, AANoCapture::StateType &State,
                      SmallVectorImpl<const Value *> &PotentialCopies,
                      unsigned &RemainingUsesToExplore)
      : A(A), NoCaptureAA(NoCaptureAA), IsDeadAA(IsDeadAA), State(State),
        PotentialCopies(PotentialCopies),
        RemainingUsesToExplore(RemainingUsesToExplore) {}

  /// Determine if \p V maybe captured. *Also updates the state!*
  bool valueMayBeCaptured(const Value *V) {
    if (V->getType()->isPointerTy()) {
      PointerMayBeCaptured(V, this);
    } else {
      State.indicatePessimisticFixpoint();
    }
    return State.isAssumed(AANoCapture::NO_CAPTURE_MAYBE_RETURNED);
  }

  /// See CaptureTracker::tooManyUses().
  void tooManyUses() override {
    State.removeAssumedBits(AANoCapture::NO_CAPTURE);
  }

  bool isDereferenceableOrNull(Value *O, const DataLayout &DL) override {
    if (CaptureTracker::isDereferenceableOrNull(O, DL))
      return true;
    const auto &DerefAA = A.getAAFor<AADereferenceable>(
        NoCaptureAA, IRPosition::value(*O), DepClassTy::OPTIONAL);
    return DerefAA.getAssumedDereferenceableBytes();
  }

  /// See CaptureTracker::captured(...).
  bool captured(const Use *U) override {
    Instruction *UInst = cast<Instruction>(U->getUser());
    LLVM_DEBUG(dbgs() << "Check use: " << *U->get() << " in " << *UInst
                      << "\n");

    // Because we may reuse the tracker multiple times we keep track of the
    // number of explored uses ourselves as well.
    if (RemainingUsesToExplore-- == 0) {
      LLVM_DEBUG(dbgs() << " - too many uses to explore!\n");
      return isCapturedIn(/* Memory */ true, /* Integer */ true,
                          /* Return */ true);
    }

    // Deal with ptr2int by following uses.
    if (isa<PtrToIntInst>(UInst)) {
      LLVM_DEBUG(dbgs() << " - ptr2int assume the worst!\n");
      return valueMayBeCaptured(UInst);
    }

    // Explicitly catch return instructions.
    if (isa<ReturnInst>(UInst))
      return isCapturedIn(/* Memory */ false, /* Integer */ false,
                          /* Return */ true);

    // For now we only use special logic for call sites. However, the tracker
    // itself knows about a lot of other non-capturing cases already.
    auto *CB = dyn_cast<CallBase>(UInst);
    if (!CB || !CB->isArgOperand(U))
      return isCapturedIn(/* Memory */ true, /* Integer */ true,
                          /* Return */ true);

    unsigned ArgNo = CB->getArgOperandNo(U);
    const IRPosition &CSArgPos = IRPosition::callsite_argument(*CB, ArgNo);
    // If we have a abstract no-capture attribute for the argument we can use
    // it to justify a non-capture attribute here. This allows recursion!
    auto &ArgNoCaptureAA =
        A.getAAFor<AANoCapture>(NoCaptureAA, CSArgPos, DepClassTy::REQUIRED);
    if (ArgNoCaptureAA.isAssumedNoCapture())
      return isCapturedIn(/* Memory */ false, /* Integer */ false,
                          /* Return */ false);
    if (ArgNoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
      addPotentialCopy(*CB);
      return isCapturedIn(/* Memory */ false, /* Integer */ false,
                          /* Return */ false);
    }

    // Lastly, we could not find a reason no-capture can be assumed so we don't.
    return isCapturedIn(/* Memory */ true, /* Integer */ true,
                        /* Return */ true);
  }

  /// Register \p CS as potential copy of the value we are checking.
  void addPotentialCopy(CallBase &CB) { PotentialCopies.push_back(&CB); }

  /// See CaptureTracker::shouldExplore(...).
  bool shouldExplore(const Use *U) override {
    // Check liveness and ignore droppable users.
    return !U->getUser()->isDroppable() &&
           !A.isAssumedDead(*U, &NoCaptureAA, &IsDeadAA);
  }

  /// Update the state according to \p CapturedInMem, \p CapturedInInt, and
  /// \p CapturedInRet, then return the appropriate value for use in the
  /// CaptureTracker::captured() interface.
  bool isCapturedIn(bool CapturedInMem, bool CapturedInInt,
                    bool CapturedInRet) {
    LLVM_DEBUG(dbgs() << " - captures [Mem " << CapturedInMem << "|Int "
                      << CapturedInInt << "|Ret " << CapturedInRet << "]\n");
    if (CapturedInMem)
      State.removeAssumedBits(AANoCapture::NOT_CAPTURED_IN_MEM);
    if (CapturedInInt)
      State.removeAssumedBits(AANoCapture::NOT_CAPTURED_IN_INT);
    if (CapturedInRet)
      State.removeAssumedBits(AANoCapture::NOT_CAPTURED_IN_RET);
    return !State.isAssumed(AANoCapture::NO_CAPTURE_MAYBE_RETURNED);
  }

private:
  /// The attributor providing in-flight abstract attributes.
  Attributor &A;

  /// The abstract attribute currently updated.
  AANoCapture &NoCaptureAA;

  /// The abstract liveness state.
  const AAIsDead &IsDeadAA;

  /// The state currently updated.
  AANoCapture::StateType &State;

  /// Set of potential copies of the tracked value.
  SmallVectorImpl<const Value *> &PotentialCopies;

  /// Global counter to limit the number of explored uses.
  unsigned &RemainingUsesToExplore;
};

ChangeStatus AANoCaptureImpl::updateImpl(Attributor &A) {
  const IRPosition &IRP = getIRPosition();
  const Value *V = isArgumentPosition() ? IRP.getAssociatedArgument()
                                        : &IRP.getAssociatedValue();
  if (!V)
    return indicatePessimisticFixpoint();

  const Function *F =
      isArgumentPosition() ? IRP.getAssociatedFunction() : IRP.getAnchorScope();
  assert(F && "Expected a function!");
  const IRPosition &FnPos = IRPosition::function(*F);
  const auto &IsDeadAA = A.getAAFor<AAIsDead>(*this, FnPos, DepClassTy::NONE);

  AANoCapture::StateType T;

  // Readonly means we cannot capture through memory.
  const auto &FnMemAA =
      A.getAAFor<AAMemoryBehavior>(*this, FnPos, DepClassTy::NONE);
  if (FnMemAA.isAssumedReadOnly()) {
    T.addKnownBits(NOT_CAPTURED_IN_MEM);
    if (FnMemAA.isKnownReadOnly())
      addKnownBits(NOT_CAPTURED_IN_MEM);
    else
      A.recordDependence(FnMemAA, *this, DepClassTy::OPTIONAL);
  }

  // Make sure all returned values are different than the underlying value.
  // TODO: we could do this in a more sophisticated way inside
  //       AAReturnedValues, e.g., track all values that escape through returns
  //       directly somehow.
  auto CheckReturnedArgs = [&](const AAReturnedValues &RVAA) {
    bool SeenConstant = false;
    for (auto &It : RVAA.returned_values()) {
      if (isa<Constant>(It.first)) {
        if (SeenConstant)
          return false;
        SeenConstant = true;
      } else if (!isa<Argument>(It.first) ||
                 It.first == getAssociatedArgument())
        return false;
    }
    return true;
  };

  const auto &NoUnwindAA =
      A.getAAFor<AANoUnwind>(*this, FnPos, DepClassTy::OPTIONAL);
  if (NoUnwindAA.isAssumedNoUnwind()) {
    bool IsVoidTy = F->getReturnType()->isVoidTy();
    const AAReturnedValues *RVAA =
        IsVoidTy ? nullptr
                 : &A.getAAFor<AAReturnedValues>(*this, FnPos,

                                                 DepClassTy::OPTIONAL);
    if (IsVoidTy || CheckReturnedArgs(*RVAA)) {
      T.addKnownBits(NOT_CAPTURED_IN_RET);
      if (T.isKnown(NOT_CAPTURED_IN_MEM))
        return ChangeStatus::UNCHANGED;
      if (NoUnwindAA.isKnownNoUnwind() &&
          (IsVoidTy || RVAA->getState().isAtFixpoint())) {
        addKnownBits(NOT_CAPTURED_IN_RET);
        if (isKnown(NOT_CAPTURED_IN_MEM))
          return indicateOptimisticFixpoint();
      }
    }
  }

  // Use the CaptureTracker interface and logic with the specialized tracker,
  // defined in AACaptureUseTracker, that can look at in-flight abstract
  // attributes and directly updates the assumed state.
  SmallVector<const Value *, 4> PotentialCopies;
  unsigned RemainingUsesToExplore =
      getDefaultMaxUsesToExploreForCaptureTracking();
  AACaptureUseTracker Tracker(A, *this, IsDeadAA, T, PotentialCopies,
                              RemainingUsesToExplore);

  // Check all potential copies of the associated value until we can assume
  // none will be captured or we have to assume at least one might be.
  unsigned Idx = 0;
  PotentialCopies.push_back(V);
  while (T.isAssumed(NO_CAPTURE_MAYBE_RETURNED) && Idx < PotentialCopies.size())
    Tracker.valueMayBeCaptured(PotentialCopies[Idx++]);

  AANoCapture::StateType &S = getState();
  auto Assumed = S.getAssumed();
  S.intersectAssumedBits(T.getAssumed());
  if (!isAssumedNoCaptureMaybeReturned())
    return indicatePessimisticFixpoint();
  return Assumed == S.getAssumed() ? ChangeStatus::UNCHANGED
                                   : ChangeStatus::CHANGED;
}

/// NoCapture attribute for function arguments.
struct AANoCaptureArgument final : AANoCaptureImpl {
  AANoCaptureArgument(const IRPosition &IRP, Attributor &A)
      : AANoCaptureImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nocapture) }
};

/// NoCapture attribute for call site arguments.
struct AANoCaptureCallSiteArgument final : AANoCaptureImpl {
  AANoCaptureCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AANoCaptureImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (Argument *Arg = getAssociatedArgument())
      if (Arg->hasByValAttr())
        indicateOptimisticFixpoint();
    AANoCaptureImpl::initialize(A);
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Argument *Arg = getAssociatedArgument();
    if (!Arg)
      return indicatePessimisticFixpoint();
    const IRPosition &ArgPos = IRPosition::argument(*Arg);
    auto &ArgAA = A.getAAFor<AANoCapture>(*this, ArgPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), ArgAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override{STATS_DECLTRACK_CSARG_ATTR(nocapture)};
};

/// NoCapture attribute for floating values.
struct AANoCaptureFloating final : AANoCaptureImpl {
  AANoCaptureFloating(const IRPosition &IRP, Attributor &A)
      : AANoCaptureImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(nocapture)
  }
};

/// NoCapture attribute for function return value.
struct AANoCaptureReturned final : AANoCaptureImpl {
  AANoCaptureReturned(const IRPosition &IRP, Attributor &A)
      : AANoCaptureImpl(IRP, A) {
    llvm_unreachable("NoCapture is not applicable to function returns!");
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    llvm_unreachable("NoCapture is not applicable to function returns!");
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("NoCapture is not applicable to function returns!");
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

/// NoCapture attribute deduction for a call site return value.
struct AANoCaptureCallSiteReturned final : AANoCaptureImpl {
  AANoCaptureCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AANoCaptureImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    const Function *F = getAnchorScope();
    // Check what state the associated function can actually capture.
    determineFunctionCaptureCapabilities(getIRPosition(), *F, *this);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(nocapture)
  }
};

/// ------------------ Value Simplify Attribute ----------------------------
struct AAValueSimplifyImpl : AAValueSimplify {
  AAValueSimplifyImpl(const IRPosition &IRP, Attributor &A)
      : AAValueSimplify(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (getAssociatedValue().getType()->isVoidTy())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? (getKnown() ? "simplified" : "maybe-simple")
                        : "not-simple";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

  /// See AAValueSimplify::getAssumedSimplifiedValue()
  Optional<Value *> getAssumedSimplifiedValue(Attributor &A) const override {
    if (!getAssumed())
      return const_cast<Value *>(&getAssociatedValue());
    return SimplifiedAssociatedValue;
  }

  /// Helper function for querying AAValueSimplify and updating candicate.
  /// \param QueryingValue Value trying to unify with SimplifiedValue
  /// \param AccumulatedSimplifiedValue Current simplification result.
  static bool checkAndUpdate(Attributor &A, const AbstractAttribute &QueryingAA,
                             Value &QueryingValue,
                             Optional<Value *> &AccumulatedSimplifiedValue) {
    // FIXME: Add a typecast support.

    auto &ValueSimplifyAA = A.getAAFor<AAValueSimplify>(
        QueryingAA, IRPosition::value(QueryingValue), DepClassTy::REQUIRED);

    Optional<Value *> QueryingValueSimplified =
        ValueSimplifyAA.getAssumedSimplifiedValue(A);

    if (!QueryingValueSimplified.hasValue())
      return true;

    if (!QueryingValueSimplified.getValue())
      return false;

    Value &QueryingValueSimplifiedUnwrapped =
        *QueryingValueSimplified.getValue();

    if (AccumulatedSimplifiedValue.hasValue() &&
        !isa<UndefValue>(AccumulatedSimplifiedValue.getValue()) &&
        !isa<UndefValue>(QueryingValueSimplifiedUnwrapped))
      return AccumulatedSimplifiedValue == QueryingValueSimplified;
    if (AccumulatedSimplifiedValue.hasValue() &&
        isa<UndefValue>(QueryingValueSimplifiedUnwrapped))
      return true;

    LLVM_DEBUG(dbgs() << "[ValueSimplify] " << QueryingValue
                      << " is assumed to be "
                      << QueryingValueSimplifiedUnwrapped << "\n");

    AccumulatedSimplifiedValue = QueryingValueSimplified;
    return true;
  }

  /// Returns a candidate is found or not
  template <typename AAType> bool askSimplifiedValueFor(Attributor &A) {
    if (!getAssociatedValue().getType()->isIntegerTy())
      return false;

    const auto &AA =
        A.getAAFor<AAType>(*this, getIRPosition(), DepClassTy::NONE);

    Optional<ConstantInt *> COpt = AA.getAssumedConstantInt(A);

    if (!COpt.hasValue()) {
      SimplifiedAssociatedValue = llvm::None;
      A.recordDependence(AA, *this, DepClassTy::OPTIONAL);
      return true;
    }
    if (auto *C = COpt.getValue()) {
      SimplifiedAssociatedValue = C;
      A.recordDependence(AA, *this, DepClassTy::OPTIONAL);
      return true;
    }
    return false;
  }

  bool askSimplifiedValueForOtherAAs(Attributor &A) {
    if (askSimplifiedValueFor<AAValueConstantRange>(A))
      return true;
    if (askSimplifiedValueFor<AAPotentialValues>(A))
      return true;
    return false;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    if (SimplifiedAssociatedValue.hasValue() &&
        !SimplifiedAssociatedValue.getValue())
      return Changed;

    Value &V = getAssociatedValue();
    auto *C = SimplifiedAssociatedValue.hasValue()
                  ? dyn_cast<Constant>(SimplifiedAssociatedValue.getValue())
                  : UndefValue::get(V.getType());
    if (C) {
      // We can replace the AssociatedValue with the constant.
      if (!V.user_empty() && &V != C && V.getType() == C->getType()) {
        LLVM_DEBUG(dbgs() << "[ValueSimplify] " << V << " -> " << *C
                          << " :: " << *this << "\n");
        if (A.changeValueAfterManifest(V, *C))
          Changed = ChangeStatus::CHANGED;
      }
    }

    return Changed | AAValueSimplify::manifest(A);
  }

  /// See AbstractState::indicatePessimisticFixpoint(...).
  ChangeStatus indicatePessimisticFixpoint() override {
    // NOTE: Associated value will be returned in a pessimistic fixpoint and is
    // regarded as known. That's why`indicateOptimisticFixpoint` is called.
    SimplifiedAssociatedValue = &getAssociatedValue();
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

protected:
  // An assumed simplified value. Initially, it is set to Optional::None, which
  // means that the value is not clear under current assumption. If in the
  // pessimistic state, getAssumedSimplifiedValue doesn't return this value but
  // returns orignal associated value.
  Optional<Value *> SimplifiedAssociatedValue;
};

struct AAValueSimplifyArgument final : AAValueSimplifyImpl {
  AAValueSimplifyArgument(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyImpl(IRP, A) {}

  void initialize(Attributor &A) override {
    AAValueSimplifyImpl::initialize(A);
    if (!getAnchorScope() || getAnchorScope()->isDeclaration())
      indicatePessimisticFixpoint();
    if (hasAttr({Attribute::InAlloca, Attribute::Preallocated,
                 Attribute::StructRet, Attribute::Nest},
                /* IgnoreSubsumingPositions */ true))
      indicatePessimisticFixpoint();

    // FIXME: This is a hack to prevent us from propagating function poiner in
    // the new pass manager CGSCC pass as it creates call edges the
    // CallGraphUpdater cannot handle yet.
    Value &V = getAssociatedValue();
    if (V.getType()->isPointerTy() &&
        V.getType()->getPointerElementType()->isFunctionTy() &&
        !A.isModulePass())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // Byval is only replacable if it is readonly otherwise we would write into
    // the replaced value and not the copy that byval creates implicitly.
    Argument *Arg = getAssociatedArgument();
    if (Arg->hasByValAttr()) {
      // TODO: We probably need to verify synchronization is not an issue, e.g.,
      //       there is no race by not copying a constant byval.
      const auto &MemAA = A.getAAFor<AAMemoryBehavior>(*this, getIRPosition(),
                                                       DepClassTy::REQUIRED);
      if (!MemAA.isAssumedReadOnly())
        return indicatePessimisticFixpoint();
    }

    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    auto PredForCallSite = [&](AbstractCallSite ACS) {
      const IRPosition &ACSArgPos =
          IRPosition::callsite_argument(ACS, getCallSiteArgNo());
      // Check if a coresponding argument was found or if it is on not
      // associated (which can happen for callback calls).
      if (ACSArgPos.getPositionKind() == IRPosition::IRP_INVALID)
        return false;

      // We can only propagate thread independent values through callbacks.
      // This is different to direct/indirect call sites because for them we
      // know the thread executing the caller and callee is the same. For
      // callbacks this is not guaranteed, thus a thread dependent value could
      // be different for the caller and callee, making it invalid to propagate.
      Value &ArgOp = ACSArgPos.getAssociatedValue();
      if (ACS.isCallbackCall())
        if (auto *C = dyn_cast<Constant>(&ArgOp))
          if (C->isThreadDependent())
            return false;
      return checkAndUpdate(A, *this, ArgOp, SimplifiedAssociatedValue);
    };

    bool AllCallSitesKnown;
    if (!A.checkForAllCallSites(PredForCallSite, *this, true,
                                AllCallSitesKnown))
      if (!askSimplifiedValueForOtherAAs(A))
        return indicatePessimisticFixpoint();

    // If a candicate was found in this update, return CHANGED.
    return HasValueBefore == SimplifiedAssociatedValue.hasValue()
               ? ChangeStatus::UNCHANGED
               : ChangeStatus ::CHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(value_simplify)
  }
};

struct AAValueSimplifyReturned : AAValueSimplifyImpl {
  AAValueSimplifyReturned(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    auto PredForReturned = [&](Value &V) {
      return checkAndUpdate(A, *this, V, SimplifiedAssociatedValue);
    };

    if (!A.checkForAllReturnedValues(PredForReturned, *this))
      if (!askSimplifiedValueForOtherAAs(A))
        return indicatePessimisticFixpoint();

    // If a candicate was found in this update, return CHANGED.
    return HasValueBefore == SimplifiedAssociatedValue.hasValue()
               ? ChangeStatus::UNCHANGED
               : ChangeStatus ::CHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    if (SimplifiedAssociatedValue.hasValue() &&
        !SimplifiedAssociatedValue.getValue())
      return Changed;

    Value &V = getAssociatedValue();
    auto *C = SimplifiedAssociatedValue.hasValue()
                  ? dyn_cast<Constant>(SimplifiedAssociatedValue.getValue())
                  : UndefValue::get(V.getType());
    if (C) {
      auto PredForReturned =
          [&](Value &V, const SmallSetVector<ReturnInst *, 4> &RetInsts) {
            // We can replace the AssociatedValue with the constant.
            if (&V == C || V.getType() != C->getType() || isa<UndefValue>(V))
              return true;

            for (ReturnInst *RI : RetInsts) {
              if (RI->getFunction() != getAnchorScope())
                continue;
              auto *RC = C;
              if (RC->getType() != RI->getReturnValue()->getType())
                RC = ConstantExpr::getBitCast(RC,
                                              RI->getReturnValue()->getType());
              LLVM_DEBUG(dbgs() << "[ValueSimplify] " << V << " -> " << *RC
                                << " in " << *RI << " :: " << *this << "\n");
              if (A.changeUseAfterManifest(RI->getOperandUse(0), *RC))
                Changed = ChangeStatus::CHANGED;
            }
            return true;
          };
      A.checkForAllReturnedValuesAndReturnInsts(PredForReturned, *this);
    }

    return Changed | AAValueSimplify::manifest(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(value_simplify)
  }
};

struct AAValueSimplifyFloating : AAValueSimplifyImpl {
  AAValueSimplifyFloating(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // FIXME: This might have exposed a SCC iterator update bug in the old PM.
    //        Needs investigation.
    // AAValueSimplifyImpl::initialize(A);
    Value &V = getAnchorValue();

    // TODO: add other stuffs
    if (isa<Constant>(V))
      indicatePessimisticFixpoint();
  }

  /// Check if \p ICmp is an equality comparison (==/!=) with at least one
  /// nullptr. If so, try to simplify it using AANonNull on the other operand.
  /// Return true if successful, in that case SimplifiedAssociatedValue will be
  /// updated and \p Changed is set appropriately.
  bool checkForNullPtrCompare(Attributor &A, ICmpInst *ICmp,
                              ChangeStatus &Changed) {
    if (!ICmp)
      return false;
    if (!ICmp->isEquality())
      return false;

    // This is a comparison with == or !-. We check for nullptr now.
    bool Op0IsNull = isa<ConstantPointerNull>(ICmp->getOperand(0));
    bool Op1IsNull = isa<ConstantPointerNull>(ICmp->getOperand(1));
    if (!Op0IsNull && !Op1IsNull)
      return false;

    LLVMContext &Ctx = ICmp->getContext();
    // Check for `nullptr ==/!= nullptr` first:
    if (Op0IsNull && Op1IsNull) {
      Value *NewVal = ConstantInt::get(
          Type::getInt1Ty(Ctx), ICmp->getPredicate() == CmpInst::ICMP_EQ);
      assert(!SimplifiedAssociatedValue.hasValue() &&
             "Did not expect non-fixed value for constant comparison");
      SimplifiedAssociatedValue = NewVal;
      indicateOptimisticFixpoint();
      Changed = ChangeStatus::CHANGED;
      return true;
    }

    // Left is the nullptr ==/!= non-nullptr case. We'll use AANonNull on the
    // non-nullptr operand and if we assume it's non-null we can conclude the
    // result of the comparison.
    assert((Op0IsNull || Op1IsNull) &&
           "Expected nullptr versus non-nullptr comparison at this point");

    // The index is the operand that we assume is not null.
    unsigned PtrIdx = Op0IsNull;
    auto &PtrNonNullAA = A.getAAFor<AANonNull>(
        *this, IRPosition::value(*ICmp->getOperand(PtrIdx)),
        DepClassTy::REQUIRED);
    if (!PtrNonNullAA.isAssumedNonNull())
      return false;

    // The new value depends on the predicate, true for != and false for ==.
    Value *NewVal = ConstantInt::get(Type::getInt1Ty(Ctx),
                                     ICmp->getPredicate() == CmpInst::ICMP_NE);

    assert((!SimplifiedAssociatedValue.hasValue() ||
            SimplifiedAssociatedValue == NewVal) &&
           "Did not expect to change value for zero-comparison");

    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();
    SimplifiedAssociatedValue = NewVal;

    if (PtrNonNullAA.isKnownNonNull())
      indicateOptimisticFixpoint();

    Changed = HasValueBefore ? ChangeStatus::UNCHANGED : ChangeStatus ::CHANGED;
    return true;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    ChangeStatus Changed;
    if (checkForNullPtrCompare(A, dyn_cast<ICmpInst>(&getAnchorValue()),
                               Changed))
      return Changed;

    auto VisitValueCB = [&](Value &V, const Instruction *CtxI, bool &,
                            bool Stripped) -> bool {
      auto &AA = A.getAAFor<AAValueSimplify>(*this, IRPosition::value(V),
                                             DepClassTy::REQUIRED);
      if (!Stripped && this == &AA) {
        // TODO: Look the instruction and check recursively.

        LLVM_DEBUG(dbgs() << "[ValueSimplify] Can't be stripped more : " << V
                          << "\n");
        return false;
      }
      return checkAndUpdate(A, *this, V, SimplifiedAssociatedValue);
    };

    bool Dummy = false;
    if (!genericValueTraversal<AAValueSimplify, bool>(
            A, getIRPosition(), *this, Dummy, VisitValueCB, getCtxI(),
            /* UseValueSimplify */ false))
      if (!askSimplifiedValueForOtherAAs(A))
        return indicatePessimisticFixpoint();

    // If a candicate was found in this update, return CHANGED.

    return HasValueBefore == SimplifiedAssociatedValue.hasValue()
               ? ChangeStatus::UNCHANGED
               : ChangeStatus ::CHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(value_simplify)
  }
};

struct AAValueSimplifyFunction : AAValueSimplifyImpl {
  AAValueSimplifyFunction(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    SimplifiedAssociatedValue = &getAnchorValue();
    indicateOptimisticFixpoint();
  }
  /// See AbstractAttribute::initialize(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable(
        "AAValueSimplify(Function|CallSite)::updateImpl will not be called");
  }
  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FN_ATTR(value_simplify)
  }
};

struct AAValueSimplifyCallSite : AAValueSimplifyFunction {
  AAValueSimplifyCallSite(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyFunction(IRP, A) {}
  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(value_simplify)
  }
};

struct AAValueSimplifyCallSiteReturned : AAValueSimplifyReturned {
  AAValueSimplifyCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyReturned(IRP, A) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    return AAValueSimplifyImpl::manifest(A);
  }

  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(value_simplify)
  }
};
struct AAValueSimplifyCallSiteArgument : AAValueSimplifyFloating {
  AAValueSimplifyCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyFloating(IRP, A) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    if (SimplifiedAssociatedValue.hasValue() &&
        !SimplifiedAssociatedValue.getValue())
      return Changed;

    Value &V = getAssociatedValue();
    auto *C = SimplifiedAssociatedValue.hasValue()
                  ? dyn_cast<Constant>(SimplifiedAssociatedValue.getValue())
                  : UndefValue::get(V.getType());
    if (C) {
      Use &U = cast<CallBase>(&getAnchorValue())
                   ->getArgOperandUse(getCallSiteArgNo());
      // We can replace the AssociatedValue with the constant.
      if (&V != C && V.getType() == C->getType()) {
        if (A.changeUseAfterManifest(U, *C))
          Changed = ChangeStatus::CHANGED;
      }
    }

    return Changed | AAValueSimplify::manifest(A);
  }

  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(value_simplify)
  }
};

/// ----------------------- Heap-To-Stack Conversion ---------------------------
struct AAHeapToStackImpl : public AAHeapToStack {
  AAHeapToStackImpl(const IRPosition &IRP, Attributor &A)
      : AAHeapToStack(IRP, A) {}

  const std::string getAsStr() const override {
    return "[H2S] Mallocs: " + std::to_string(MallocCalls.size());
  }

  ChangeStatus manifest(Attributor &A) override {
    assert(getState().isValidState() &&
           "Attempted to manifest an invalid state!");

    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
    Function *F = getAnchorScope();
    const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);

    for (Instruction *MallocCall : MallocCalls) {
      // This malloc cannot be replaced.
      if (BadMallocCalls.count(MallocCall))
        continue;

      for (Instruction *FreeCall : FreesForMalloc[MallocCall]) {
        LLVM_DEBUG(dbgs() << "H2S: Removing free call: " << *FreeCall << "\n");
        A.deleteAfterManifest(*FreeCall);
        HasChanged = ChangeStatus::CHANGED;
      }

      LLVM_DEBUG(dbgs() << "H2S: Removing malloc call: " << *MallocCall
                        << "\n");

      Align Alignment;
      Value *Size;
      if (isCallocLikeFn(MallocCall, TLI)) {
        auto *Num = MallocCall->getOperand(0);
        auto *SizeT = MallocCall->getOperand(1);
        IRBuilder<> B(MallocCall);
        Size = B.CreateMul(Num, SizeT, "h2s.calloc.size");
      } else if (isAlignedAllocLikeFn(MallocCall, TLI)) {
        Size = MallocCall->getOperand(1);
        Alignment = MaybeAlign(cast<ConstantInt>(MallocCall->getOperand(0))
                                   ->getValue()
                                   .getZExtValue())
                        .valueOrOne();
      } else {
        Size = MallocCall->getOperand(0);
      }

      unsigned AS = cast<PointerType>(MallocCall->getType())->getAddressSpace();
      Instruction *AI =
          new AllocaInst(Type::getInt8Ty(F->getContext()), AS, Size, Alignment,
                         "", MallocCall->getNextNode());

      if (AI->getType() != MallocCall->getType())
        AI = new BitCastInst(AI, MallocCall->getType(), "malloc_bc",
                             AI->getNextNode());

      A.changeValueAfterManifest(*MallocCall, *AI);

      if (auto *II = dyn_cast<InvokeInst>(MallocCall)) {
        auto *NBB = II->getNormalDest();
        BranchInst::Create(NBB, MallocCall->getParent());
        A.deleteAfterManifest(*MallocCall);
      } else {
        A.deleteAfterManifest(*MallocCall);
      }

      // Zero out the allocated memory if it was a calloc.
      if (isCallocLikeFn(MallocCall, TLI)) {
        auto *BI = new BitCastInst(AI, MallocCall->getType(), "calloc_bc",
                                   AI->getNextNode());
        Value *Ops[] = {
            BI, ConstantInt::get(F->getContext(), APInt(8, 0, false)), Size,
            ConstantInt::get(Type::getInt1Ty(F->getContext()), false)};

        Type *Tys[] = {BI->getType(), MallocCall->getOperand(0)->getType()};
        Module *M = F->getParent();
        Function *Fn = Intrinsic::getDeclaration(M, Intrinsic::memset, Tys);
        CallInst::Create(Fn, Ops, "", BI->getNextNode());
      }
      HasChanged = ChangeStatus::CHANGED;
    }

    return HasChanged;
  }

  /// Collection of all malloc calls in a function.
  SmallSetVector<Instruction *, 4> MallocCalls;

  /// Collection of malloc calls that cannot be converted.
  DenseSet<const Instruction *> BadMallocCalls;

  /// A map for each malloc call to the set of associated free calls.
  DenseMap<Instruction *, SmallPtrSet<Instruction *, 4>> FreesForMalloc;

  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AAHeapToStackImpl::updateImpl(Attributor &A) {
  const Function *F = getAnchorScope();
  const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);

  MustBeExecutedContextExplorer &Explorer =
      A.getInfoCache().getMustBeExecutedContextExplorer();

  auto FreeCheck = [&](Instruction &I) {
    const auto &Frees = FreesForMalloc.lookup(&I);
    if (Frees.size() != 1)
      return false;
    Instruction *UniqueFree = *Frees.begin();
    return Explorer.findInContextOf(UniqueFree, I.getNextNode());
  };

  auto UsesCheck = [&](Instruction &I) {
    bool ValidUsesOnly = true;
    bool MustUse = true;
    auto Pred = [&](const Use &U, bool &Follow) -> bool {
      Instruction *UserI = cast<Instruction>(U.getUser());
      if (isa<LoadInst>(UserI))
        return true;
      if (auto *SI = dyn_cast<StoreInst>(UserI)) {
        if (SI->getValueOperand() == U.get()) {
          LLVM_DEBUG(dbgs()
                     << "[H2S] escaping store to memory: " << *UserI << "\n");
          ValidUsesOnly = false;
        } else {
          // A store into the malloc'ed memory is fine.
        }
        return true;
      }
      if (auto *CB = dyn_cast<CallBase>(UserI)) {
        if (!CB->isArgOperand(&U) || CB->isLifetimeStartOrEnd())
          return true;
        // Record malloc.
        if (isFreeCall(UserI, TLI)) {
          if (MustUse) {
            FreesForMalloc[&I].insert(UserI);
          } else {
            LLVM_DEBUG(dbgs() << "[H2S] free potentially on different mallocs: "
                              << *UserI << "\n");
            ValidUsesOnly = false;
          }
          return true;
        }

        unsigned ArgNo = CB->getArgOperandNo(&U);

        const auto &NoCaptureAA = A.getAAFor<AANoCapture>(
            *this, IRPosition::callsite_argument(*CB, ArgNo),
            DepClassTy::REQUIRED);

        // If a callsite argument use is nofree, we are fine.
        const auto &ArgNoFreeAA = A.getAAFor<AANoFree>(
            *this, IRPosition::callsite_argument(*CB, ArgNo),
            DepClassTy::REQUIRED);

        if (!NoCaptureAA.isAssumedNoCapture() ||
            !ArgNoFreeAA.isAssumedNoFree()) {
          LLVM_DEBUG(dbgs() << "[H2S] Bad user: " << *UserI << "\n");
          ValidUsesOnly = false;
        }
        return true;
      }

      if (isa<GetElementPtrInst>(UserI) || isa<BitCastInst>(UserI) ||
          isa<PHINode>(UserI) || isa<SelectInst>(UserI)) {
        MustUse &= !(isa<PHINode>(UserI) || isa<SelectInst>(UserI));
        Follow = true;
        return true;
      }
      // Unknown user for which we can not track uses further (in a way that
      // makes sense).
      LLVM_DEBUG(dbgs() << "[H2S] Unknown user: " << *UserI << "\n");
      ValidUsesOnly = false;
      return true;
    };
    A.checkForAllUses(Pred, *this, I);
    return ValidUsesOnly;
  };

  auto MallocCallocCheck = [&](Instruction &I) {
    if (BadMallocCalls.count(&I))
      return true;

    bool IsMalloc = isMallocLikeFn(&I, TLI);
    bool IsAlignedAllocLike = isAlignedAllocLikeFn(&I, TLI);
    bool IsCalloc = !IsMalloc && isCallocLikeFn(&I, TLI);
    if (!IsMalloc && !IsAlignedAllocLike && !IsCalloc) {
      BadMallocCalls.insert(&I);
      return true;
    }

    if (IsMalloc) {
      if (MaxHeapToStackSize == -1) {
        if (UsesCheck(I) || FreeCheck(I)) {
          MallocCalls.insert(&I);
          return true;
        }
      }
      if (auto *Size = dyn_cast<ConstantInt>(I.getOperand(0)))
        if (Size->getValue().ule(MaxHeapToStackSize))
          if (UsesCheck(I) || FreeCheck(I)) {
            MallocCalls.insert(&I);
            return true;
          }
    } else if (IsAlignedAllocLike && isa<ConstantInt>(I.getOperand(0))) {
      if (MaxHeapToStackSize == -1) {
        if (UsesCheck(I) || FreeCheck(I)) {
          MallocCalls.insert(&I);
          return true;
        }
      }
      // Only if the alignment and sizes are constant.
      if (auto *Size = dyn_cast<ConstantInt>(I.getOperand(1)))
        if (Size->getValue().ule(MaxHeapToStackSize))
          if (UsesCheck(I) || FreeCheck(I)) {
            MallocCalls.insert(&I);
            return true;
          }
    } else if (IsCalloc) {
      if (MaxHeapToStackSize == -1) {
        if (UsesCheck(I) || FreeCheck(I)) {
          MallocCalls.insert(&I);
          return true;
        }
      }
      bool Overflow = false;
      if (auto *Num = dyn_cast<ConstantInt>(I.getOperand(0)))
        if (auto *Size = dyn_cast<ConstantInt>(I.getOperand(1)))
          if ((Size->getValue().umul_ov(Num->getValue(), Overflow))
                  .ule(MaxHeapToStackSize))
            if (!Overflow && (UsesCheck(I) || FreeCheck(I))) {
              MallocCalls.insert(&I);
              return true;
            }
    }

    BadMallocCalls.insert(&I);
    return true;
  };

  size_t NumBadMallocs = BadMallocCalls.size();

  A.checkForAllCallLikeInstructions(MallocCallocCheck, *this);

  if (NumBadMallocs != BadMallocCalls.size())
    return ChangeStatus::CHANGED;

  return ChangeStatus::UNCHANGED;
}

struct AAHeapToStackFunction final : public AAHeapToStackImpl {
  AAHeapToStackFunction(const IRPosition &IRP, Attributor &A)
      : AAHeapToStackImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics().
  void trackStatistics() const override {
    STATS_DECL(
        MallocCalls, Function,
        "Number of malloc/calloc/aligned_alloc calls converted to allocas");
    for (auto *C : MallocCalls)
      if (!BadMallocCalls.count(C))
        ++BUILD_STAT_NAME(MallocCalls, Function);
  }
};

/// ----------------------- Privatizable Pointers ------------------------------
struct AAPrivatizablePtrImpl : public AAPrivatizablePtr {
  AAPrivatizablePtrImpl(const IRPosition &IRP, Attributor &A)
      : AAPrivatizablePtr(IRP, A), PrivatizableType(llvm::None) {}

  ChangeStatus indicatePessimisticFixpoint() override {
    AAPrivatizablePtr::indicatePessimisticFixpoint();
    PrivatizableType = nullptr;
    return ChangeStatus::CHANGED;
  }

  /// Identify the type we can chose for a private copy of the underlying
  /// argument. None means it is not clear yet, nullptr means there is none.
  virtual Optional<Type *> identifyPrivatizableType(Attributor &A) = 0;

  /// Return a privatizable type that encloses both T0 and T1.
  /// TODO: This is merely a stub for now as we should manage a mapping as well.
  Optional<Type *> combineTypes(Optional<Type *> T0, Optional<Type *> T1) {
    if (!T0.hasValue())
      return T1;
    if (!T1.hasValue())
      return T0;
    if (T0 == T1)
      return T0;
    return nullptr;
  }

  Optional<Type *> getPrivatizableType() const override {
    return PrivatizableType;
  }

  const std::string getAsStr() const override {
    return isAssumedPrivatizablePtr() ? "[priv]" : "[no-priv]";
  }

protected:
  Optional<Type *> PrivatizableType;
};

// TODO: Do this for call site arguments (probably also other values) as well.

struct AAPrivatizablePtrArgument final : public AAPrivatizablePtrImpl {
  AAPrivatizablePtrArgument(const IRPosition &IRP, Attributor &A)
      : AAPrivatizablePtrImpl(IRP, A) {}

  /// See AAPrivatizablePtrImpl::identifyPrivatizableType(...)
  Optional<Type *> identifyPrivatizableType(Attributor &A) override {
    // If this is a byval argument and we know all the call sites (so we can
    // rewrite them), there is no need to check them explicitly.
    bool AllCallSitesKnown;
    if (getIRPosition().hasAttr(Attribute::ByVal) &&
        A.checkForAllCallSites([](AbstractCallSite ACS) { return true; }, *this,
                               true, AllCallSitesKnown))
      return getAssociatedValue().getType()->getPointerElementType();

    Optional<Type *> Ty;
    unsigned ArgNo = getIRPosition().getCallSiteArgNo();

    // Make sure the associated call site argument has the same type at all call
    // sites and it is an allocation we know is safe to privatize, for now that
    // means we only allow alloca instructions.
    // TODO: We can additionally analyze the accesses in the callee to  create
    //       the type from that information instead. That is a little more
    //       involved and will be done in a follow up patch.
    auto CallSiteCheck = [&](AbstractCallSite ACS) {
      IRPosition ACSArgPos = IRPosition::callsite_argument(ACS, ArgNo);
      // Check if a coresponding argument was found or if it is one not
      // associated (which can happen for callback calls).
      if (ACSArgPos.getPositionKind() == IRPosition::IRP_INVALID)
        return false;

      // Check that all call sites agree on a type.
      auto &PrivCSArgAA =
          A.getAAFor<AAPrivatizablePtr>(*this, ACSArgPos, DepClassTy::REQUIRED);
      Optional<Type *> CSTy = PrivCSArgAA.getPrivatizableType();

      LLVM_DEBUG({
        dbgs() << "[AAPrivatizablePtr] ACSPos: " << ACSArgPos << ", CSTy: ";
        if (CSTy.hasValue() && CSTy.getValue())
          CSTy.getValue()->print(dbgs());
        else if (CSTy.hasValue())
          dbgs() << "<nullptr>";
        else
          dbgs() << "<none>";
      });

      Ty = combineTypes(Ty, CSTy);

      LLVM_DEBUG({
        dbgs() << " : New Type: ";
        if (Ty.hasValue() && Ty.getValue())
          Ty.getValue()->print(dbgs());
        else if (Ty.hasValue())
          dbgs() << "<nullptr>";
        else
          dbgs() << "<none>";
        dbgs() << "\n";
      });

      return !Ty.hasValue() || Ty.getValue();
    };

    if (!A.checkForAllCallSites(CallSiteCheck, *this, true, AllCallSitesKnown))
      return nullptr;
    return Ty;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    PrivatizableType = identifyPrivatizableType(A);
    if (!PrivatizableType.hasValue())
      return ChangeStatus::UNCHANGED;
    if (!PrivatizableType.getValue())
      return indicatePessimisticFixpoint();

    // The dependence is optional so we don't give up once we give up on the
    // alignment.
    A.getAAFor<AAAlign>(*this, IRPosition::value(getAssociatedValue()),
                        DepClassTy::OPTIONAL);

    // Avoid arguments with padding for now.
    if (!getIRPosition().hasAttr(Attribute::ByVal) &&
        !ArgumentPromotionPass::isDenselyPacked(PrivatizableType.getValue(),
                                                A.getInfoCache().getDL())) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] Padding detected\n");
      return indicatePessimisticFixpoint();
    }

    // Verify callee and caller agree on how the promoted argument would be
    // passed.
    // TODO: The use of the ArgumentPromotion interface here is ugly, we need a
    // specialized form of TargetTransformInfo::areFunctionArgsABICompatible
    // which doesn't require the arguments ArgumentPromotion wanted to pass.
    Function &Fn = *getIRPosition().getAnchorScope();
    SmallPtrSet<Argument *, 1> ArgsToPromote, Dummy;
    ArgsToPromote.insert(getAssociatedArgument());
    const auto *TTI =
        A.getInfoCache().getAnalysisResultForFunction<TargetIRAnalysis>(Fn);
    if (!TTI ||
        !ArgumentPromotionPass::areFunctionArgsABICompatible(
            Fn, *TTI, ArgsToPromote, Dummy) ||
        ArgsToPromote.empty()) {
      LLVM_DEBUG(
          dbgs() << "[AAPrivatizablePtr] ABI incompatibility detected for "
                 << Fn.getName() << "\n");
      return indicatePessimisticFixpoint();
    }

    // Collect the types that will replace the privatizable type in the function
    // signature.
    SmallVector<Type *, 16> ReplacementTypes;
    identifyReplacementTypes(PrivatizableType.getValue(), ReplacementTypes);

    // Register a rewrite of the argument.
    Argument *Arg = getAssociatedArgument();
    if (!A.isValidFunctionSignatureRewrite(*Arg, ReplacementTypes)) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] Rewrite not valid\n");
      return indicatePessimisticFixpoint();
    }

    unsigned ArgNo = Arg->getArgNo();

    // Helper to check if for the given call site the associated argument is
    // passed to a callback where the privatization would be different.
    auto IsCompatiblePrivArgOfCallback = [&](CallBase &CB) {
      SmallVector<const Use *, 4> CallbackUses;
      AbstractCallSite::getCallbackUses(CB, CallbackUses);
      for (const Use *U : CallbackUses) {
        AbstractCallSite CBACS(U);
        assert(CBACS && CBACS.isCallbackCall());
        for (Argument &CBArg : CBACS.getCalledFunction()->args()) {
          int CBArgNo = CBACS.getCallArgOperandNo(CBArg);

          LLVM_DEBUG({
            dbgs()
                << "[AAPrivatizablePtr] Argument " << *Arg
                << "check if can be privatized in the context of its parent ("
                << Arg->getParent()->getName()
                << ")\n[AAPrivatizablePtr] because it is an argument in a "
                   "callback ("
                << CBArgNo << "@" << CBACS.getCalledFunction()->getName()
                << ")\n[AAPrivatizablePtr] " << CBArg << " : "
                << CBACS.getCallArgOperand(CBArg) << " vs "
                << CB.getArgOperand(ArgNo) << "\n"
                << "[AAPrivatizablePtr] " << CBArg << " : "
                << CBACS.getCallArgOperandNo(CBArg) << " vs " << ArgNo << "\n";
          });

          if (CBArgNo != int(ArgNo))
            continue;
          const auto &CBArgPrivAA = A.getAAFor<AAPrivatizablePtr>(
              *this, IRPosition::argument(CBArg), DepClassTy::REQUIRED);
          if (CBArgPrivAA.isValidState()) {
            auto CBArgPrivTy = CBArgPrivAA.getPrivatizableType();
            if (!CBArgPrivTy.hasValue())
              continue;
            if (CBArgPrivTy.getValue() == PrivatizableType)
              continue;
          }

          LLVM_DEBUG({
            dbgs() << "[AAPrivatizablePtr] Argument " << *Arg
                   << " cannot be privatized in the context of its parent ("
                   << Arg->getParent()->getName()
                   << ")\n[AAPrivatizablePtr] because it is an argument in a "
                      "callback ("
                   << CBArgNo << "@" << CBACS.getCalledFunction()->getName()
                   << ").\n[AAPrivatizablePtr] for which the argument "
                      "privatization is not compatible.\n";
          });
          return false;
        }
      }
      return true;
    };

    // Helper to check if for the given call site the associated argument is
    // passed to a direct call where the privatization would be different.
    auto IsCompatiblePrivArgOfDirectCS = [&](AbstractCallSite ACS) {
      CallBase *DC = cast<CallBase>(ACS.getInstruction());
      int DCArgNo = ACS.getCallArgOperandNo(ArgNo);
      assert(DCArgNo >= 0 && unsigned(DCArgNo) < DC->getNumArgOperands() &&
             "Expected a direct call operand for callback call operand");

      LLVM_DEBUG({
        dbgs() << "[AAPrivatizablePtr] Argument " << *Arg
               << " check if be privatized in the context of its parent ("
               << Arg->getParent()->getName()
               << ")\n[AAPrivatizablePtr] because it is an argument in a "
                  "direct call of ("
               << DCArgNo << "@" << DC->getCalledFunction()->getName()
               << ").\n";
      });

      Function *DCCallee = DC->getCalledFunction();
      if (unsigned(DCArgNo) < DCCallee->arg_size()) {
        const auto &DCArgPrivAA = A.getAAFor<AAPrivatizablePtr>(
            *this, IRPosition::argument(*DCCallee->getArg(DCArgNo)),
            DepClassTy::REQUIRED);
        if (DCArgPrivAA.isValidState()) {
          auto DCArgPrivTy = DCArgPrivAA.getPrivatizableType();
          if (!DCArgPrivTy.hasValue())
            return true;
          if (DCArgPrivTy.getValue() == PrivatizableType)
            return true;
        }
      }

      LLVM_DEBUG({
        dbgs() << "[AAPrivatizablePtr] Argument " << *Arg
               << " cannot be privatized in the context of its parent ("
               << Arg->getParent()->getName()
               << ")\n[AAPrivatizablePtr] because it is an argument in a "
                  "direct call of ("
               << ACS.getInstruction()->getCalledFunction()->getName()
               << ").\n[AAPrivatizablePtr] for which the argument "
                  "privatization is not compatible.\n";
      });
      return false;
    };

    // Helper to check if the associated argument is used at the given abstract
    // call site in a way that is incompatible with the privatization assumed
    // here.
    auto IsCompatiblePrivArgOfOtherCallSite = [&](AbstractCallSite ACS) {
      if (ACS.isDirectCall())
        return IsCompatiblePrivArgOfCallback(*ACS.getInstruction());
      if (ACS.isCallbackCall())
        return IsCompatiblePrivArgOfDirectCS(ACS);
      return false;
    };

    bool AllCallSitesKnown;
    if (!A.checkForAllCallSites(IsCompatiblePrivArgOfOtherCallSite, *this, true,
                                AllCallSitesKnown))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }

  /// Given a type to private \p PrivType, collect the constituates (which are
  /// used) in \p ReplacementTypes.
  static void
  identifyReplacementTypes(Type *PrivType,
                           SmallVectorImpl<Type *> &ReplacementTypes) {
    // TODO: For now we expand the privatization type to the fullest which can
    //       lead to dead arguments that need to be removed later.
    assert(PrivType && "Expected privatizable type!");

    // Traverse the type, extract constituate types on the outermost level.
    if (auto *PrivStructType = dyn_cast<StructType>(PrivType)) {
      for (unsigned u = 0, e = PrivStructType->getNumElements(); u < e; u++)
        ReplacementTypes.push_back(PrivStructType->getElementType(u));
    } else if (auto *PrivArrayType = dyn_cast<ArrayType>(PrivType)) {
      ReplacementTypes.append(PrivArrayType->getNumElements(),
                              PrivArrayType->getElementType());
    } else {
      ReplacementTypes.push_back(PrivType);
    }
  }

  /// Initialize \p Base according to the type \p PrivType at position \p IP.
  /// The values needed are taken from the arguments of \p F starting at
  /// position \p ArgNo.
  static void createInitialization(Type *PrivType, Value &Base, Function &F,
                                   unsigned ArgNo, Instruction &IP) {
    assert(PrivType && "Expected privatizable type!");

    IRBuilder<NoFolder> IRB(&IP);
    const DataLayout &DL = F.getParent()->getDataLayout();

    // Traverse the type, build GEPs and stores.
    if (auto *PrivStructType = dyn_cast<StructType>(PrivType)) {
      const StructLayout *PrivStructLayout = DL.getStructLayout(PrivStructType);
      for (unsigned u = 0, e = PrivStructType->getNumElements(); u < e; u++) {
        Type *PointeeTy = PrivStructType->getElementType(u)->getPointerTo();
        Value *Ptr = constructPointer(
            PointeeTy, &Base, PrivStructLayout->getElementOffset(u), IRB, DL);
        new StoreInst(F.getArg(ArgNo + u), Ptr, &IP);
      }
    } else if (auto *PrivArrayType = dyn_cast<ArrayType>(PrivType)) {
      Type *PointeeTy = PrivArrayType->getElementType();
      Type *PointeePtrTy = PointeeTy->getPointerTo();
      uint64_t PointeeTySize = DL.getTypeStoreSize(PointeeTy);
      for (unsigned u = 0, e = PrivArrayType->getNumElements(); u < e; u++) {
        Value *Ptr =
            constructPointer(PointeePtrTy, &Base, u * PointeeTySize, IRB, DL);
        new StoreInst(F.getArg(ArgNo + u), Ptr, &IP);
      }
    } else {
      new StoreInst(F.getArg(ArgNo), &Base, &IP);
    }
  }

  /// Extract values from \p Base according to the type \p PrivType at the
  /// call position \p ACS. The values are appended to \p ReplacementValues.
  void createReplacementValues(Align Alignment, Type *PrivType,
                               AbstractCallSite ACS, Value *Base,
                               SmallVectorImpl<Value *> &ReplacementValues) {
    assert(Base && "Expected base value!");
    assert(PrivType && "Expected privatizable type!");
    Instruction *IP = ACS.getInstruction();

    IRBuilder<NoFolder> IRB(IP);
    const DataLayout &DL = IP->getModule()->getDataLayout();

    if (Base->getType()->getPointerElementType() != PrivType)
      Base = BitCastInst::CreateBitOrPointerCast(Base, PrivType->getPointerTo(),
                                                 "", ACS.getInstruction());

    // Traverse the type, build GEPs and loads.
    if (auto *PrivStructType = dyn_cast<StructType>(PrivType)) {
      const StructLayout *PrivStructLayout = DL.getStructLayout(PrivStructType);
      for (unsigned u = 0, e = PrivStructType->getNumElements(); u < e; u++) {
        Type *PointeeTy = PrivStructType->getElementType(u);
        Value *Ptr =
            constructPointer(PointeeTy->getPointerTo(), Base,
                             PrivStructLayout->getElementOffset(u), IRB, DL);
        LoadInst *L = new LoadInst(PointeeTy, Ptr, "", IP);
        L->setAlignment(Alignment);
        ReplacementValues.push_back(L);
      }
    } else if (auto *PrivArrayType = dyn_cast<ArrayType>(PrivType)) {
      Type *PointeeTy = PrivArrayType->getElementType();
      uint64_t PointeeTySize = DL.getTypeStoreSize(PointeeTy);
      Type *PointeePtrTy = PointeeTy->getPointerTo();
      for (unsigned u = 0, e = PrivArrayType->getNumElements(); u < e; u++) {
        Value *Ptr =
            constructPointer(PointeePtrTy, Base, u * PointeeTySize, IRB, DL);
        LoadInst *L = new LoadInst(PointeeTy, Ptr, "", IP);
        L->setAlignment(Alignment);
        ReplacementValues.push_back(L);
      }
    } else {
      LoadInst *L = new LoadInst(PrivType, Base, "", IP);
      L->setAlignment(Alignment);
      ReplacementValues.push_back(L);
    }
  }

  /// See AbstractAttribute::manifest(...)
  ChangeStatus manifest(Attributor &A) override {
    if (!PrivatizableType.hasValue())
      return ChangeStatus::UNCHANGED;
    assert(PrivatizableType.getValue() && "Expected privatizable type!");

    // Collect all tail calls in the function as we cannot allow new allocas to
    // escape into tail recursion.
    // TODO: Be smarter about new allocas escaping into tail calls.
    SmallVector<CallInst *, 16> TailCalls;
    if (!A.checkForAllInstructions(
            [&](Instruction &I) {
              CallInst &CI = cast<CallInst>(I);
              if (CI.isTailCall())
                TailCalls.push_back(&CI);
              return true;
            },
            *this, {Instruction::Call}))
      return ChangeStatus::UNCHANGED;

    Argument *Arg = getAssociatedArgument();
    // Query AAAlign attribute for alignment of associated argument to
    // determine the best alignment of loads.
    const auto &AlignAA =
        A.getAAFor<AAAlign>(*this, IRPosition::value(*Arg), DepClassTy::NONE);

    // Callback to repair the associated function. A new alloca is placed at the
    // beginning and initialized with the values passed through arguments. The
    // new alloca replaces the use of the old pointer argument.
    Attributor::ArgumentReplacementInfo::CalleeRepairCBTy FnRepairCB =
        [=](const Attributor::ArgumentReplacementInfo &ARI,
            Function &ReplacementFn, Function::arg_iterator ArgIt) {
          BasicBlock &EntryBB = ReplacementFn.getEntryBlock();
          Instruction *IP = &*EntryBB.getFirstInsertionPt();
          Instruction *AI = new AllocaInst(PrivatizableType.getValue(), 0,
                                           Arg->getName() + ".priv", IP);
          createInitialization(PrivatizableType.getValue(), *AI, ReplacementFn,
                               ArgIt->getArgNo(), *IP);

          if (AI->getType() != Arg->getType())
            AI =
                BitCastInst::CreateBitOrPointerCast(AI, Arg->getType(), "", IP);
          Arg->replaceAllUsesWith(AI);

          for (CallInst *CI : TailCalls)
            CI->setTailCall(false);
        };

    // Callback to repair a call site of the associated function. The elements
    // of the privatizable type are loaded prior to the call and passed to the
    // new function version.
    Attributor::ArgumentReplacementInfo::ACSRepairCBTy ACSRepairCB =
        [=, &AlignAA](const Attributor::ArgumentReplacementInfo &ARI,
                      AbstractCallSite ACS,
                      SmallVectorImpl<Value *> &NewArgOperands) {
          // When no alignment is specified for the load instruction,
          // natural alignment is assumed.
          createReplacementValues(
              assumeAligned(AlignAA.getAssumedAlign()),
              PrivatizableType.getValue(), ACS,
              ACS.getCallArgOperand(ARI.getReplacedArg().getArgNo()),
              NewArgOperands);
        };

    // Collect the types that will replace the privatizable type in the function
    // signature.
    SmallVector<Type *, 16> ReplacementTypes;
    identifyReplacementTypes(PrivatizableType.getValue(), ReplacementTypes);

    // Register a rewrite of the argument.
    if (A.registerFunctionSignatureRewrite(*Arg, ReplacementTypes,
                                           std::move(FnRepairCB),
                                           std::move(ACSRepairCB)))
      return ChangeStatus::CHANGED;
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(privatizable_ptr);
  }
};

struct AAPrivatizablePtrFloating : public AAPrivatizablePtrImpl {
  AAPrivatizablePtrFloating(const IRPosition &IRP, Attributor &A)
      : AAPrivatizablePtrImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  virtual void initialize(Attributor &A) override {
    // TODO: We can privatize more than arguments.
    indicatePessimisticFixpoint();
  }

  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("AAPrivatizablePtr(Floating|Returned|CallSiteReturned)::"
                     "updateImpl will not be called");
  }

  /// See AAPrivatizablePtrImpl::identifyPrivatizableType(...)
  Optional<Type *> identifyPrivatizableType(Attributor &A) override {
    Value *Obj = getUnderlyingObject(&getAssociatedValue());
    if (!Obj) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] No underlying object found!\n");
      return nullptr;
    }

    if (auto *AI = dyn_cast<AllocaInst>(Obj))
      if (auto *CI = dyn_cast<ConstantInt>(AI->getArraySize()))
        if (CI->isOne())
          return Obj->getType()->getPointerElementType();
    if (auto *Arg = dyn_cast<Argument>(Obj)) {
      auto &PrivArgAA = A.getAAFor<AAPrivatizablePtr>(
          *this, IRPosition::argument(*Arg), DepClassTy::REQUIRED);
      if (PrivArgAA.isAssumedPrivatizablePtr())
        return Obj->getType()->getPointerElementType();
    }

    LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] Underlying object neither valid "
                         "alloca nor privatizable argument: "
                      << *Obj << "!\n");
    return nullptr;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(privatizable_ptr);
  }
};

struct AAPrivatizablePtrCallSiteArgument final
    : public AAPrivatizablePtrFloating {
  AAPrivatizablePtrCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAPrivatizablePtrFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (getIRPosition().hasAttr(Attribute::ByVal))
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    PrivatizableType = identifyPrivatizableType(A);
    if (!PrivatizableType.hasValue())
      return ChangeStatus::UNCHANGED;
    if (!PrivatizableType.getValue())
      return indicatePessimisticFixpoint();

    const IRPosition &IRP = getIRPosition();
    auto &NoCaptureAA =
        A.getAAFor<AANoCapture>(*this, IRP, DepClassTy::REQUIRED);
    if (!NoCaptureAA.isAssumedNoCapture()) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] pointer might be captured!\n");
      return indicatePessimisticFixpoint();
    }

    auto &NoAliasAA = A.getAAFor<AANoAlias>(*this, IRP, DepClassTy::REQUIRED);
    if (!NoAliasAA.isAssumedNoAlias()) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] pointer might alias!\n");
      return indicatePessimisticFixpoint();
    }

    const auto &MemBehaviorAA =
        A.getAAFor<AAMemoryBehavior>(*this, IRP, DepClassTy::REQUIRED);
    if (!MemBehaviorAA.isAssumedReadOnly()) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] pointer is written!\n");
      return indicatePessimisticFixpoint();
    }

    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(privatizable_ptr);
  }
};

struct AAPrivatizablePtrCallSiteReturned final
    : public AAPrivatizablePtrFloating {
  AAPrivatizablePtrCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAPrivatizablePtrFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // TODO: We can privatize more than arguments.
    indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(privatizable_ptr);
  }
};

struct AAPrivatizablePtrReturned final : public AAPrivatizablePtrFloating {
  AAPrivatizablePtrReturned(const IRPosition &IRP, Attributor &A)
      : AAPrivatizablePtrFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // TODO: We can privatize more than arguments.
    indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(privatizable_ptr);
  }
};

/// -------------------- Memory Behavior Attributes ----------------------------
/// Includes read-none, read-only, and write-only.
/// ----------------------------------------------------------------------------
struct AAMemoryBehaviorImpl : public AAMemoryBehavior {
  AAMemoryBehaviorImpl(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehavior(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    intersectAssumedBits(BEST_STATE);
    getKnownStateFromValue(getIRPosition(), getState());
    AAMemoryBehavior::initialize(A);
  }

  /// Return the memory behavior information encoded in the IR for \p IRP.
  static void getKnownStateFromValue(const IRPosition &IRP,
                                     BitIntegerState &State,
                                     bool IgnoreSubsumingPositions = false) {
    SmallVector<Attribute, 2> Attrs;
    IRP.getAttrs(AttrKinds, Attrs, IgnoreSubsumingPositions);
    for (const Attribute &Attr : Attrs) {
      switch (Attr.getKindAsEnum()) {
      case Attribute::ReadNone:
        State.addKnownBits(NO_ACCESSES);
        break;
      case Attribute::ReadOnly:
        State.addKnownBits(NO_WRITES);
        break;
      case Attribute::WriteOnly:
        State.addKnownBits(NO_READS);
        break;
      default:
        llvm_unreachable("Unexpected attribute!");
      }
    }

    if (auto *I = dyn_cast<Instruction>(&IRP.getAnchorValue())) {
      if (!I->mayReadFromMemory())
        State.addKnownBits(NO_READS);
      if (!I->mayWriteToMemory())
        State.addKnownBits(NO_WRITES);
    }
  }

  /// See AbstractAttribute::getDeducedAttributes(...).
  void getDeducedAttributes(LLVMContext &Ctx,
                            SmallVectorImpl<Attribute> &Attrs) const override {
    assert(Attrs.size() == 0);
    if (isAssumedReadNone())
      Attrs.push_back(Attribute::get(Ctx, Attribute::ReadNone));
    else if (isAssumedReadOnly())
      Attrs.push_back(Attribute::get(Ctx, Attribute::ReadOnly));
    else if (isAssumedWriteOnly())
      Attrs.push_back(Attribute::get(Ctx, Attribute::WriteOnly));
    assert(Attrs.size() <= 1);
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    if (hasAttr(Attribute::ReadNone, /* IgnoreSubsumingPositions */ true))
      return ChangeStatus::UNCHANGED;

    const IRPosition &IRP = getIRPosition();

    // Check if we would improve the existing attributes first.
    SmallVector<Attribute, 4> DeducedAttrs;
    getDeducedAttributes(IRP.getAnchorValue().getContext(), DeducedAttrs);
    if (llvm::all_of(DeducedAttrs, [&](const Attribute &Attr) {
          return IRP.hasAttr(Attr.getKindAsEnum(),
                             /* IgnoreSubsumingPositions */ true);
        }))
      return ChangeStatus::UNCHANGED;

    // Clear existing attributes.
    IRP.removeAttrs(AttrKinds);

    // Use the generic manifest method.
    return IRAttribute::manifest(A);
  }

  /// See AbstractState::getAsStr().
  const std::string getAsStr() const override {
    if (isAssumedReadNone())
      return "readnone";
    if (isAssumedReadOnly())
      return "readonly";
    if (isAssumedWriteOnly())
      return "writeonly";
    return "may-read/write";
  }

  /// The set of IR attributes AAMemoryBehavior deals with.
  static const Attribute::AttrKind AttrKinds[3];
};

const Attribute::AttrKind AAMemoryBehaviorImpl::AttrKinds[] = {
    Attribute::ReadNone, Attribute::ReadOnly, Attribute::WriteOnly};

/// Memory behavior attribute for a floating value.
struct AAMemoryBehaviorFloating : AAMemoryBehaviorImpl {
  AAMemoryBehaviorFloating(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehaviorImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAMemoryBehaviorImpl::initialize(A);
    addUsesOf(A, getAssociatedValue());
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_FLOATING_ATTR(readnone)
    else if (isAssumedReadOnly())
      STATS_DECLTRACK_FLOATING_ATTR(readonly)
    else if (isAssumedWriteOnly())
      STATS_DECLTRACK_FLOATING_ATTR(writeonly)
  }

private:
  /// Return true if users of \p UserI might access the underlying
  /// variable/location described by \p U and should therefore be analyzed.
  bool followUsersOfUseIn(Attributor &A, const Use *U,
                          const Instruction *UserI);

  /// Update the state according to the effect of use \p U in \p UserI.
  void analyzeUseIn(Attributor &A, const Use *U, const Instruction *UserI);

protected:
  /// Add the uses of \p V to the `Uses` set we look at during the update step.
  void addUsesOf(Attributor &A, const Value &V);

  /// Container for (transitive) uses of the associated argument.
  SmallVector<const Use *, 8> Uses;

  /// Set to remember the uses we already traversed.
  SmallPtrSet<const Use *, 8> Visited;
};

/// Memory behavior attribute for function argument.
struct AAMemoryBehaviorArgument : AAMemoryBehaviorFloating {
  AAMemoryBehaviorArgument(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehaviorFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    intersectAssumedBits(BEST_STATE);
    const IRPosition &IRP = getIRPosition();
    // TODO: Make IgnoreSubsumingPositions a property of an IRAttribute so we
    // can query it when we use has/getAttr. That would allow us to reuse the
    // initialize of the base class here.
    bool HasByVal =
        IRP.hasAttr({Attribute::ByVal}, /* IgnoreSubsumingPositions */ true);
    getKnownStateFromValue(IRP, getState(),
                           /* IgnoreSubsumingPositions */ HasByVal);

    // Initialize the use vector with all direct uses of the associated value.
    Argument *Arg = getAssociatedArgument();
    if (!Arg || !A.isFunctionIPOAmendable(*(Arg->getParent()))) {
      indicatePessimisticFixpoint();
    } else {
      addUsesOf(A, *Arg);
    }
  }

  ChangeStatus manifest(Attributor &A) override {
    // TODO: Pointer arguments are not supported on vectors of pointers yet.
    if (!getAssociatedValue().getType()->isPointerTy())
      return ChangeStatus::UNCHANGED;

    // TODO: From readattrs.ll: "inalloca parameters are always
    //                           considered written"
    if (hasAttr({Attribute::InAlloca, Attribute::Preallocated})) {
      removeKnownBits(NO_WRITES);
      removeAssumedBits(NO_WRITES);
    }
    return AAMemoryBehaviorFloating::manifest(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_ARG_ATTR(readnone)
    else if (isAssumedReadOnly())
      STATS_DECLTRACK_ARG_ATTR(readonly)
    else if (isAssumedWriteOnly())
      STATS_DECLTRACK_ARG_ATTR(writeonly)
  }
};

struct AAMemoryBehaviorCallSiteArgument final : AAMemoryBehaviorArgument {
  AAMemoryBehaviorCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehaviorArgument(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // If we don't have an associated attribute this is either a variadic call
    // or an indirect call, either way, nothing to do here.
    Argument *Arg = getAssociatedArgument();
    if (!Arg) {
      indicatePessimisticFixpoint();
      return;
    }
    if (Arg->hasByValAttr()) {
      addKnownBits(NO_WRITES);
      removeKnownBits(NO_READS);
      removeAssumedBits(NO_READS);
    }
    AAMemoryBehaviorArgument::initialize(A);
    if (getAssociatedFunction()->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Argument *Arg = getAssociatedArgument();
    const IRPosition &ArgPos = IRPosition::argument(*Arg);
    auto &ArgAA =
        A.getAAFor<AAMemoryBehavior>(*this, ArgPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), ArgAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_CSARG_ATTR(readnone)
    else if (isAssumedReadOnly())
      STATS_DECLTRACK_CSARG_ATTR(readonly)
    else if (isAssumedWriteOnly())
      STATS_DECLTRACK_CSARG_ATTR(writeonly)
  }
};

/// Memory behavior attribute for a call site return position.
struct AAMemoryBehaviorCallSiteReturned final : AAMemoryBehaviorFloating {
  AAMemoryBehaviorCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehaviorFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAMemoryBehaviorImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // We do not annotate returned values.
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

/// An AA to represent the memory behavior function attributes.
struct AAMemoryBehaviorFunction final : public AAMemoryBehaviorImpl {
  AAMemoryBehaviorFunction(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehaviorImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(Attributor &A).
  virtual ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    Function &F = cast<Function>(getAnchorValue());
    if (isAssumedReadNone()) {
      F.removeFnAttr(Attribute::ArgMemOnly);
      F.removeFnAttr(Attribute::InaccessibleMemOnly);
      F.removeFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    }
    return AAMemoryBehaviorImpl::manifest(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_FN_ATTR(readnone)
    else if (isAssumedReadOnly())
      STATS_DECLTRACK_FN_ATTR(readonly)
    else if (isAssumedWriteOnly())
      STATS_DECLTRACK_FN_ATTR(writeonly)
  }
};

/// AAMemoryBehavior attribute for call sites.
struct AAMemoryBehaviorCallSite final : AAMemoryBehaviorImpl {
  AAMemoryBehaviorCallSite(const IRPosition &IRP, Attributor &A)
      : AAMemoryBehaviorImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAMemoryBehaviorImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA =
        A.getAAFor<AAMemoryBehavior>(*this, FnPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), FnAA.getState());
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_CS_ATTR(readnone)
    else if (isAssumedReadOnly())
      STATS_DECLTRACK_CS_ATTR(readonly)
    else if (isAssumedWriteOnly())
      STATS_DECLTRACK_CS_ATTR(writeonly)
  }
};

ChangeStatus AAMemoryBehaviorFunction::updateImpl(Attributor &A) {

  // The current assumed state used to determine a change.
  auto AssumedState = getAssumed();

  auto CheckRWInst = [&](Instruction &I) {
    // If the instruction has an own memory behavior state, use it to restrict
    // the local state. No further analysis is required as the other memory
    // state is as optimistic as it gets.
    if (const auto *CB = dyn_cast<CallBase>(&I)) {
      const auto &MemBehaviorAA = A.getAAFor<AAMemoryBehavior>(
          *this, IRPosition::callsite_function(*CB), DepClassTy::REQUIRED);
      intersectAssumedBits(MemBehaviorAA.getAssumed());
      return !isAtFixpoint();
    }

    // Remove access kind modifiers if necessary.
    if (I.mayReadFromMemory())
      removeAssumedBits(NO_READS);
    if (I.mayWriteToMemory())
      removeAssumedBits(NO_WRITES);
    return !isAtFixpoint();
  };

  if (!A.checkForAllReadWriteInstructions(CheckRWInst, *this))
    return indicatePessimisticFixpoint();

  return (AssumedState != getAssumed()) ? ChangeStatus::CHANGED
                                        : ChangeStatus::UNCHANGED;
}

ChangeStatus AAMemoryBehaviorFloating::updateImpl(Attributor &A) {

  const IRPosition &IRP = getIRPosition();
  const IRPosition &FnPos = IRPosition::function_scope(IRP);
  AAMemoryBehavior::StateType &S = getState();

  // First, check the function scope. We take the known information and we avoid
  // work if the assumed information implies the current assumed information for
  // this attribute. This is a valid for all but byval arguments.
  Argument *Arg = IRP.getAssociatedArgument();
  AAMemoryBehavior::base_t FnMemAssumedState =
      AAMemoryBehavior::StateType::getWorstState();
  if (!Arg || !Arg->hasByValAttr()) {
    const auto &FnMemAA =
        A.getAAFor<AAMemoryBehavior>(*this, FnPos, DepClassTy::OPTIONAL);
    FnMemAssumedState = FnMemAA.getAssumed();
    S.addKnownBits(FnMemAA.getKnown());
    if ((S.getAssumed() & FnMemAA.getAssumed()) == S.getAssumed())
      return ChangeStatus::UNCHANGED;
  }

  // Make sure the value is not captured (except through "return"), if
  // it is, any information derived would be irrelevant anyway as we cannot
  // check the potential aliases introduced by the capture. However, no need
  // to fall back to anythign less optimistic than the function state.
  const auto &ArgNoCaptureAA =
      A.getAAFor<AANoCapture>(*this, IRP, DepClassTy::OPTIONAL);
  if (!ArgNoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
    S.intersectAssumedBits(FnMemAssumedState);
    return ChangeStatus::CHANGED;
  }

  // The current assumed state used to determine a change.
  auto AssumedState = S.getAssumed();

  // Liveness information to exclude dead users.
  // TODO: Take the FnPos once we have call site specific liveness information.
  const auto &LivenessAA = A.getAAFor<AAIsDead>(
      *this, IRPosition::function(*IRP.getAssociatedFunction()),
      DepClassTy::NONE);

  // Visit and expand uses until all are analyzed or a fixpoint is reached.
  for (unsigned i = 0; i < Uses.size() && !isAtFixpoint(); i++) {
    const Use *U = Uses[i];
    Instruction *UserI = cast<Instruction>(U->getUser());
    LLVM_DEBUG(dbgs() << "[AAMemoryBehavior] Use: " << **U << " in " << *UserI
                      << " [Dead: " << (A.isAssumedDead(*U, this, &LivenessAA))
                      << "]\n");
    if (A.isAssumedDead(*U, this, &LivenessAA))
      continue;

    // Droppable users, e.g., llvm::assume does not actually perform any action.
    if (UserI->isDroppable())
      continue;

    // Check if the users of UserI should also be visited.
    if (followUsersOfUseIn(A, U, UserI))
      addUsesOf(A, *UserI);

    // If UserI might touch memory we analyze the use in detail.
    if (UserI->mayReadOrWriteMemory())
      analyzeUseIn(A, U, UserI);
  }

  return (AssumedState != getAssumed()) ? ChangeStatus::CHANGED
                                        : ChangeStatus::UNCHANGED;
}

void AAMemoryBehaviorFloating::addUsesOf(Attributor &A, const Value &V) {
  SmallVector<const Use *, 8> WL;
  for (const Use &U : V.uses())
    WL.push_back(&U);

  while (!WL.empty()) {
    const Use *U = WL.pop_back_val();
    if (!Visited.insert(U).second)
      continue;

    const Instruction *UserI = cast<Instruction>(U->getUser());
    if (UserI->mayReadOrWriteMemory()) {
      Uses.push_back(U);
      continue;
    }
    if (!followUsersOfUseIn(A, U, UserI))
      continue;
    for (const Use &UU : UserI->uses())
      WL.push_back(&UU);
  }
}

bool AAMemoryBehaviorFloating::followUsersOfUseIn(Attributor &A, const Use *U,
                                                  const Instruction *UserI) {
  // The loaded value is unrelated to the pointer argument, no need to
  // follow the users of the load.
  if (isa<LoadInst>(UserI))
    return false;

  // By default we follow all uses assuming UserI might leak information on U,
  // we have special handling for call sites operands though.
  const auto *CB = dyn_cast<CallBase>(UserI);
  if (!CB || !CB->isArgOperand(U))
    return true;

  // If the use is a call argument known not to be captured, the users of
  // the call do not need to be visited because they have to be unrelated to
  // the input. Note that this check is not trivial even though we disallow
  // general capturing of the underlying argument. The reason is that the
  // call might the argument "through return", which we allow and for which we
  // need to check call users.
  if (U->get()->getType()->isPointerTy()) {
    unsigned ArgNo = CB->getArgOperandNo(U);
    const auto &ArgNoCaptureAA = A.getAAFor<AANoCapture>(
        *this, IRPosition::callsite_argument(*CB, ArgNo), DepClassTy::OPTIONAL);
    return !ArgNoCaptureAA.isAssumedNoCapture();
  }

  return true;
}

void AAMemoryBehaviorFloating::analyzeUseIn(Attributor &A, const Use *U,
                                            const Instruction *UserI) {
  assert(UserI->mayReadOrWriteMemory());

  switch (UserI->getOpcode()) {
  default:
    // TODO: Handle all atomics and other side-effect operations we know of.
    break;
  case Instruction::Load:
    // Loads cause the NO_READS property to disappear.
    removeAssumedBits(NO_READS);
    return;

  case Instruction::Store:
    // Stores cause the NO_WRITES property to disappear if the use is the
    // pointer operand. Note that we do assume that capturing was taken care of
    // somewhere else.
    if (cast<StoreInst>(UserI)->getPointerOperand() == U->get())
      removeAssumedBits(NO_WRITES);
    return;

  case Instruction::Call:
  case Instruction::CallBr:
  case Instruction::Invoke: {
    // For call sites we look at the argument memory behavior attribute (this
    // could be recursive!) in order to restrict our own state.
    const auto *CB = cast<CallBase>(UserI);

    // Give up on operand bundles.
    if (CB->isBundleOperand(U)) {
      indicatePessimisticFixpoint();
      return;
    }

    // Calling a function does read the function pointer, maybe write it if the
    // function is self-modifying.
    if (CB->isCallee(U)) {
      removeAssumedBits(NO_READS);
      break;
    }

    // Adjust the possible access behavior based on the information on the
    // argument.
    IRPosition Pos;
    if (U->get()->getType()->isPointerTy())
      Pos = IRPosition::callsite_argument(*CB, CB->getArgOperandNo(U));
    else
      Pos = IRPosition::callsite_function(*CB);
    const auto &MemBehaviorAA =
        A.getAAFor<AAMemoryBehavior>(*this, Pos, DepClassTy::OPTIONAL);
    // "assumed" has at most the same bits as the MemBehaviorAA assumed
    // and at least "known".
    intersectAssumedBits(MemBehaviorAA.getAssumed());
    return;
  }
  };

  // Generally, look at the "may-properties" and adjust the assumed state if we
  // did not trigger special handling before.
  if (UserI->mayReadFromMemory())
    removeAssumedBits(NO_READS);
  if (UserI->mayWriteToMemory())
    removeAssumedBits(NO_WRITES);
}

} // namespace

/// -------------------- Memory Locations Attributes ---------------------------
/// Includes read-none, argmemonly, inaccessiblememonly,
/// inaccessiblememorargmemonly
/// ----------------------------------------------------------------------------

std::string AAMemoryLocation::getMemoryLocationsAsStr(
    AAMemoryLocation::MemoryLocationsKind MLK) {
  if (0 == (MLK & AAMemoryLocation::NO_LOCATIONS))
    return "all memory";
  if (MLK == AAMemoryLocation::NO_LOCATIONS)
    return "no memory";
  std::string S = "memory:";
  if (0 == (MLK & AAMemoryLocation::NO_LOCAL_MEM))
    S += "stack,";
  if (0 == (MLK & AAMemoryLocation::NO_CONST_MEM))
    S += "constant,";
  if (0 == (MLK & AAMemoryLocation::NO_GLOBAL_INTERNAL_MEM))
    S += "internal global,";
  if (0 == (MLK & AAMemoryLocation::NO_GLOBAL_EXTERNAL_MEM))
    S += "external global,";
  if (0 == (MLK & AAMemoryLocation::NO_ARGUMENT_MEM))
    S += "argument,";
  if (0 == (MLK & AAMemoryLocation::NO_INACCESSIBLE_MEM))
    S += "inaccessible,";
  if (0 == (MLK & AAMemoryLocation::NO_MALLOCED_MEM))
    S += "malloced,";
  if (0 == (MLK & AAMemoryLocation::NO_UNKOWN_MEM))
    S += "unknown,";
  S.pop_back();
  return S;
}

namespace {
struct AAMemoryLocationImpl : public AAMemoryLocation {

  AAMemoryLocationImpl(const IRPosition &IRP, Attributor &A)
      : AAMemoryLocation(IRP, A), Allocator(A.Allocator) {
    for (unsigned u = 0; u < llvm::CTLog2<VALID_STATE>(); ++u)
      AccessKind2Accesses[u] = nullptr;
  }

  ~AAMemoryLocationImpl() {
    // The AccessSets are allocated via a BumpPtrAllocator, we call
    // the destructor manually.
    for (unsigned u = 0; u < llvm::CTLog2<VALID_STATE>(); ++u)
      if (AccessKind2Accesses[u])
        AccessKind2Accesses[u]->~AccessSet();
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    intersectAssumedBits(BEST_STATE);
    getKnownStateFromValue(A, getIRPosition(), getState());
    AAMemoryLocation::initialize(A);
  }

  /// Return the memory behavior information encoded in the IR for \p IRP.
  static void getKnownStateFromValue(Attributor &A, const IRPosition &IRP,
                                     BitIntegerState &State,
                                     bool IgnoreSubsumingPositions = false) {
    // For internal functions we ignore `argmemonly` and
    // `inaccessiblememorargmemonly` as we might break it via interprocedural
    // constant propagation. It is unclear if this is the best way but it is
    // unlikely this will cause real performance problems. If we are deriving
    // attributes for the anchor function we even remove the attribute in
    // addition to ignoring it.
    bool UseArgMemOnly = true;
    Function *AnchorFn = IRP.getAnchorScope();
    if (AnchorFn && A.isRunOn(*AnchorFn))
      UseArgMemOnly = !AnchorFn->hasLocalLinkage();

    SmallVector<Attribute, 2> Attrs;
    IRP.getAttrs(AttrKinds, Attrs, IgnoreSubsumingPositions);
    for (const Attribute &Attr : Attrs) {
      switch (Attr.getKindAsEnum()) {
      case Attribute::ReadNone:
        State.addKnownBits(NO_LOCAL_MEM | NO_CONST_MEM);
        break;
      case Attribute::InaccessibleMemOnly:
        State.addKnownBits(inverseLocation(NO_INACCESSIBLE_MEM, true, true));
        break;
      case Attribute::ArgMemOnly:
        if (UseArgMemOnly)
          State.addKnownBits(inverseLocation(NO_ARGUMENT_MEM, true, true));
        else
          IRP.removeAttrs({Attribute::ArgMemOnly});
        break;
      case Attribute::InaccessibleMemOrArgMemOnly:
        if (UseArgMemOnly)
          State.addKnownBits(inverseLocation(
              NO_INACCESSIBLE_MEM | NO_ARGUMENT_MEM, true, true));
        else
          IRP.removeAttrs({Attribute::InaccessibleMemOrArgMemOnly});
        break;
      default:
        llvm_unreachable("Unexpected attribute!");
      }
    }
  }

  /// See AbstractAttribute::getDeducedAttributes(...).
  void getDeducedAttributes(LLVMContext &Ctx,
                            SmallVectorImpl<Attribute> &Attrs) const override {
    assert(Attrs.size() == 0);
    if (isAssumedReadNone()) {
      Attrs.push_back(Attribute::get(Ctx, Attribute::ReadNone));
    } else if (getIRPosition().getPositionKind() == IRPosition::IRP_FUNCTION) {
      if (isAssumedInaccessibleMemOnly())
        Attrs.push_back(Attribute::get(Ctx, Attribute::InaccessibleMemOnly));
      else if (isAssumedArgMemOnly())
        Attrs.push_back(Attribute::get(Ctx, Attribute::ArgMemOnly));
      else if (isAssumedInaccessibleOrArgMemOnly())
        Attrs.push_back(
            Attribute::get(Ctx, Attribute::InaccessibleMemOrArgMemOnly));
    }
    assert(Attrs.size() <= 1);
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    const IRPosition &IRP = getIRPosition();

    // Check if we would improve the existing attributes first.
    SmallVector<Attribute, 4> DeducedAttrs;
    getDeducedAttributes(IRP.getAnchorValue().getContext(), DeducedAttrs);
    if (llvm::all_of(DeducedAttrs, [&](const Attribute &Attr) {
          return IRP.hasAttr(Attr.getKindAsEnum(),
                             /* IgnoreSubsumingPositions */ true);
        }))
      return ChangeStatus::UNCHANGED;

    // Clear existing attributes.
    IRP.removeAttrs(AttrKinds);
    if (isAssumedReadNone())
      IRP.removeAttrs(AAMemoryBehaviorImpl::AttrKinds);

    // Use the generic manifest method.
    return IRAttribute::manifest(A);
  }

  /// See AAMemoryLocation::checkForAllAccessesToMemoryKind(...).
  bool checkForAllAccessesToMemoryKind(
      function_ref<bool(const Instruction *, const Value *, AccessKind,
                        MemoryLocationsKind)>
          Pred,
      MemoryLocationsKind RequestedMLK) const override {
    if (!isValidState())
      return false;

    MemoryLocationsKind AssumedMLK = getAssumedNotAccessedLocation();
    if (AssumedMLK == NO_LOCATIONS)
      return true;

    unsigned Idx = 0;
    for (MemoryLocationsKind CurMLK = 1; CurMLK < NO_LOCATIONS;
         CurMLK *= 2, ++Idx) {
      if (CurMLK & RequestedMLK)
        continue;

      if (const AccessSet *Accesses = AccessKind2Accesses[Idx])
        for (const AccessInfo &AI : *Accesses)
          if (!Pred(AI.I, AI.Ptr, AI.Kind, CurMLK))
            return false;
    }

    return true;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    // If we give up and indicate a pessimistic fixpoint this instruction will
    // become an access for all potential access kinds:
    // TODO: Add pointers for argmemonly and globals to improve the results of
    //       checkForAllAccessesToMemoryKind.
    bool Changed = false;
    MemoryLocationsKind KnownMLK = getKnown();
    Instruction *I = dyn_cast<Instruction>(&getAssociatedValue());
    for (MemoryLocationsKind CurMLK = 1; CurMLK < NO_LOCATIONS; CurMLK *= 2)
      if (!(CurMLK & KnownMLK))
        updateStateAndAccessesMap(getState(), CurMLK, I, nullptr, Changed,
                                  getAccessKindFromInst(I));
    return AAMemoryLocation::indicatePessimisticFixpoint();
  }

protected:
  /// Helper struct to tie together an instruction that has a read or write
  /// effect with the pointer it accesses (if any).
  struct AccessInfo {

    /// The instruction that caused the access.
    const Instruction *I;

    /// The base pointer that is accessed, or null if unknown.
    const Value *Ptr;

    /// The kind of access (read/write/read+write).
    AccessKind Kind;

    bool operator==(const AccessInfo &RHS) const {
      return I == RHS.I && Ptr == RHS.Ptr && Kind == RHS.Kind;
    }
    bool operator()(const AccessInfo &LHS, const AccessInfo &RHS) const {
      if (LHS.I != RHS.I)
        return LHS.I < RHS.I;
      if (LHS.Ptr != RHS.Ptr)
        return LHS.Ptr < RHS.Ptr;
      if (LHS.Kind != RHS.Kind)
        return LHS.Kind < RHS.Kind;
      return false;
    }
  };

  /// Mapping from *single* memory location kinds, e.g., LOCAL_MEM with the
  /// value of NO_LOCAL_MEM, to the accesses encountered for this memory kind.
  using AccessSet = SmallSet<AccessInfo, 2, AccessInfo>;
  AccessSet *AccessKind2Accesses[llvm::CTLog2<VALID_STATE>()];

  /// Categorize the pointer arguments of CB that might access memory in
  /// AccessedLoc and update the state and access map accordingly.
  void
  categorizeArgumentPointerLocations(Attributor &A, CallBase &CB,
                                     AAMemoryLocation::StateType &AccessedLocs,
                                     bool &Changed);

  /// Return the kind(s) of location that may be accessed by \p V.
  AAMemoryLocation::MemoryLocationsKind
  categorizeAccessedLocations(Attributor &A, Instruction &I, bool &Changed);

  /// Return the access kind as determined by \p I.
  AccessKind getAccessKindFromInst(const Instruction *I) {
    AccessKind AK = READ_WRITE;
    if (I) {
      AK = I->mayReadFromMemory() ? READ : NONE;
      AK = AccessKind(AK | (I->mayWriteToMemory() ? WRITE : NONE));
    }
    return AK;
  }

  /// Update the state \p State and the AccessKind2Accesses given that \p I is
  /// an access of kind \p AK to a \p MLK memory location with the access
  /// pointer \p Ptr.
  void updateStateAndAccessesMap(AAMemoryLocation::StateType &State,
                                 MemoryLocationsKind MLK, const Instruction *I,
                                 const Value *Ptr, bool &Changed,
                                 AccessKind AK = READ_WRITE) {

    assert(isPowerOf2_32(MLK) && "Expected a single location set!");
    auto *&Accesses = AccessKind2Accesses[llvm::Log2_32(MLK)];
    if (!Accesses)
      Accesses = new (Allocator) AccessSet();
    Changed |= Accesses->insert(AccessInfo{I, Ptr, AK}).second;
    State.removeAssumedBits(MLK);
  }

  /// Determine the underlying locations kinds for \p Ptr, e.g., globals or
  /// arguments, and update the state and access map accordingly.
  void categorizePtrValue(Attributor &A, const Instruction &I, const Value &Ptr,
                          AAMemoryLocation::StateType &State, bool &Changed);

  /// Used to allocate access sets.
  BumpPtrAllocator &Allocator;

  /// The set of IR attributes AAMemoryLocation deals with.
  static const Attribute::AttrKind AttrKinds[4];
};

const Attribute::AttrKind AAMemoryLocationImpl::AttrKinds[] = {
    Attribute::ReadNone, Attribute::InaccessibleMemOnly, Attribute::ArgMemOnly,
    Attribute::InaccessibleMemOrArgMemOnly};

void AAMemoryLocationImpl::categorizePtrValue(
    Attributor &A, const Instruction &I, const Value &Ptr,
    AAMemoryLocation::StateType &State, bool &Changed) {
  LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Categorize pointer locations for "
                    << Ptr << " ["
                    << getMemoryLocationsAsStr(State.getAssumed()) << "]\n");

  auto StripGEPCB = [](Value *V) -> Value * {
    auto *GEP = dyn_cast<GEPOperator>(V);
    while (GEP) {
      V = GEP->getPointerOperand();
      GEP = dyn_cast<GEPOperator>(V);
    }
    return V;
  };

  auto VisitValueCB = [&](Value &V, const Instruction *,
                          AAMemoryLocation::StateType &T,
                          bool Stripped) -> bool {
    // TODO: recognize the TBAA used for constant accesses.
    MemoryLocationsKind MLK = NO_LOCATIONS;
    assert(!isa<GEPOperator>(V) && "GEPs should have been stripped.");
    if (isa<UndefValue>(V))
      return true;
    if (auto *Arg = dyn_cast<Argument>(&V)) {
      if (Arg->hasByValAttr())
        MLK = NO_LOCAL_MEM;
      else
        MLK = NO_ARGUMENT_MEM;
    } else if (auto *GV = dyn_cast<GlobalValue>(&V)) {
      // Reading constant memory is not treated as a read "effect" by the
      // function attr pass so we won't neither. Constants defined by TBAA are
      // similar. (We know we do not write it because it is constant.)
      if (auto *GVar = dyn_cast<GlobalVariable>(GV))
        if (GVar->isConstant())
          return true;

      if (GV->hasLocalLinkage())
        MLK = NO_GLOBAL_INTERNAL_MEM;
      else
        MLK = NO_GLOBAL_EXTERNAL_MEM;
    } else if (isa<ConstantPointerNull>(V) &&
               !NullPointerIsDefined(getAssociatedFunction(),
                                     V.getType()->getPointerAddressSpace())) {
      return true;
    } else if (isa<AllocaInst>(V)) {
      MLK = NO_LOCAL_MEM;
    } else if (const auto *CB = dyn_cast<CallBase>(&V)) {
      const auto &NoAliasAA = A.getAAFor<AANoAlias>(
          *this, IRPosition::callsite_returned(*CB), DepClassTy::OPTIONAL);
      if (NoAliasAA.isAssumedNoAlias())
        MLK = NO_MALLOCED_MEM;
      else
        MLK = NO_UNKOWN_MEM;
    } else {
      MLK = NO_UNKOWN_MEM;
    }

    assert(MLK != NO_LOCATIONS && "No location specified!");
    updateStateAndAccessesMap(T, MLK, &I, &V, Changed,
                              getAccessKindFromInst(&I));
    LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Ptr value cannot be categorized: "
                      << V << " -> " << getMemoryLocationsAsStr(T.getAssumed())
                      << "\n");
    return true;
  };

  if (!genericValueTraversal<AAMemoryLocation, AAMemoryLocation::StateType>(
          A, IRPosition::value(Ptr), *this, State, VisitValueCB, getCtxI(),
          /* UseValueSimplify */ true,
          /* MaxValues */ 32, StripGEPCB)) {
    LLVM_DEBUG(
        dbgs() << "[AAMemoryLocation] Pointer locations not categorized\n");
    updateStateAndAccessesMap(State, NO_UNKOWN_MEM, &I, nullptr, Changed,
                              getAccessKindFromInst(&I));
  } else {
    LLVM_DEBUG(
        dbgs()
        << "[AAMemoryLocation] Accessed locations with pointer locations: "
        << getMemoryLocationsAsStr(State.getAssumed()) << "\n");
  }
}

void AAMemoryLocationImpl::categorizeArgumentPointerLocations(
    Attributor &A, CallBase &CB, AAMemoryLocation::StateType &AccessedLocs,
    bool &Changed) {
  for (unsigned ArgNo = 0, E = CB.getNumArgOperands(); ArgNo < E; ++ArgNo) {

    // Skip non-pointer arguments.
    const Value *ArgOp = CB.getArgOperand(ArgNo);
    if (!ArgOp->getType()->isPtrOrPtrVectorTy())
      continue;

    // Skip readnone arguments.
    const IRPosition &ArgOpIRP = IRPosition::callsite_argument(CB, ArgNo);
    const auto &ArgOpMemLocationAA =
        A.getAAFor<AAMemoryBehavior>(*this, ArgOpIRP, DepClassTy::OPTIONAL);

    if (ArgOpMemLocationAA.isAssumedReadNone())
      continue;

    // Categorize potentially accessed pointer arguments as if there was an
    // access instruction with them as pointer.
    categorizePtrValue(A, CB, *ArgOp, AccessedLocs, Changed);
  }
}

AAMemoryLocation::MemoryLocationsKind
AAMemoryLocationImpl::categorizeAccessedLocations(Attributor &A, Instruction &I,
                                                  bool &Changed) {
  LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Categorize accessed locations for "
                    << I << "\n");

  AAMemoryLocation::StateType AccessedLocs;
  AccessedLocs.intersectAssumedBits(NO_LOCATIONS);

  if (auto *CB = dyn_cast<CallBase>(&I)) {

    // First check if we assume any memory is access is visible.
    const auto &CBMemLocationAA = A.getAAFor<AAMemoryLocation>(
        *this, IRPosition::callsite_function(*CB), DepClassTy::OPTIONAL);
    LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Categorize call site: " << I
                      << " [" << CBMemLocationAA << "]\n");

    if (CBMemLocationAA.isAssumedReadNone())
      return NO_LOCATIONS;

    if (CBMemLocationAA.isAssumedInaccessibleMemOnly()) {
      updateStateAndAccessesMap(AccessedLocs, NO_INACCESSIBLE_MEM, &I, nullptr,
                                Changed, getAccessKindFromInst(&I));
      return AccessedLocs.getAssumed();
    }

    uint32_t CBAssumedNotAccessedLocs =
        CBMemLocationAA.getAssumedNotAccessedLocation();

    // Set the argmemonly and global bit as we handle them separately below.
    uint32_t CBAssumedNotAccessedLocsNoArgMem =
        CBAssumedNotAccessedLocs | NO_ARGUMENT_MEM | NO_GLOBAL_MEM;

    for (MemoryLocationsKind CurMLK = 1; CurMLK < NO_LOCATIONS; CurMLK *= 2) {
      if (CBAssumedNotAccessedLocsNoArgMem & CurMLK)
        continue;
      updateStateAndAccessesMap(AccessedLocs, CurMLK, &I, nullptr, Changed,
                                getAccessKindFromInst(&I));
    }

    // Now handle global memory if it might be accessed. This is slightly tricky
    // as NO_GLOBAL_MEM has multiple bits set.
    bool HasGlobalAccesses = ((~CBAssumedNotAccessedLocs) & NO_GLOBAL_MEM);
    if (HasGlobalAccesses) {
      auto AccessPred = [&](const Instruction *, const Value *Ptr,
                            AccessKind Kind, MemoryLocationsKind MLK) {
        updateStateAndAccessesMap(AccessedLocs, MLK, &I, Ptr, Changed,
                                  getAccessKindFromInst(&I));
        return true;
      };
      if (!CBMemLocationAA.checkForAllAccessesToMemoryKind(
              AccessPred, inverseLocation(NO_GLOBAL_MEM, false, false)))
        return AccessedLocs.getWorstState();
    }

    LLVM_DEBUG(
        dbgs() << "[AAMemoryLocation] Accessed state before argument handling: "
               << getMemoryLocationsAsStr(AccessedLocs.getAssumed()) << "\n");

    // Now handle argument memory if it might be accessed.
    bool HasArgAccesses = ((~CBAssumedNotAccessedLocs) & NO_ARGUMENT_MEM);
    if (HasArgAccesses)
      categorizeArgumentPointerLocations(A, *CB, AccessedLocs, Changed);

    LLVM_DEBUG(
        dbgs() << "[AAMemoryLocation] Accessed state after argument handling: "
               << getMemoryLocationsAsStr(AccessedLocs.getAssumed()) << "\n");

    return AccessedLocs.getAssumed();
  }

  if (const Value *Ptr = getPointerOperand(&I, /* AllowVolatile */ true)) {
    LLVM_DEBUG(
        dbgs() << "[AAMemoryLocation] Categorize memory access with pointer: "
               << I << " [" << *Ptr << "]\n");
    categorizePtrValue(A, I, *Ptr, AccessedLocs, Changed);
    return AccessedLocs.getAssumed();
  }

  LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Failed to categorize instruction: "
                    << I << "\n");
  updateStateAndAccessesMap(AccessedLocs, NO_UNKOWN_MEM, &I, nullptr, Changed,
                            getAccessKindFromInst(&I));
  return AccessedLocs.getAssumed();
}

/// An AA to represent the memory behavior function attributes.
struct AAMemoryLocationFunction final : public AAMemoryLocationImpl {
  AAMemoryLocationFunction(const IRPosition &IRP, Attributor &A)
      : AAMemoryLocationImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(Attributor &A).
  virtual ChangeStatus updateImpl(Attributor &A) override {

    const auto &MemBehaviorAA =
        A.getAAFor<AAMemoryBehavior>(*this, getIRPosition(), DepClassTy::NONE);
    if (MemBehaviorAA.isAssumedReadNone()) {
      if (MemBehaviorAA.isKnownReadNone())
        return indicateOptimisticFixpoint();
      assert(isAssumedReadNone() &&
             "AAMemoryLocation was not read-none but AAMemoryBehavior was!");
      A.recordDependence(MemBehaviorAA, *this, DepClassTy::OPTIONAL);
      return ChangeStatus::UNCHANGED;
    }

    // The current assumed state used to determine a change.
    auto AssumedState = getAssumed();
    bool Changed = false;

    auto CheckRWInst = [&](Instruction &I) {
      MemoryLocationsKind MLK = categorizeAccessedLocations(A, I, Changed);
      LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Accessed locations for " << I
                        << ": " << getMemoryLocationsAsStr(MLK) << "\n");
      removeAssumedBits(inverseLocation(MLK, false, false));
      // Stop once only the valid bit set in the *not assumed location*, thus
      // once we don't actually exclude any memory locations in the state.
      return getAssumedNotAccessedLocation() != VALID_STATE;
    };

    if (!A.checkForAllReadWriteInstructions(CheckRWInst, *this))
      return indicatePessimisticFixpoint();

    Changed |= AssumedState != getAssumed();
    return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_FN_ATTR(readnone)
    else if (isAssumedArgMemOnly())
      STATS_DECLTRACK_FN_ATTR(argmemonly)
    else if (isAssumedInaccessibleMemOnly())
      STATS_DECLTRACK_FN_ATTR(inaccessiblememonly)
    else if (isAssumedInaccessibleOrArgMemOnly())
      STATS_DECLTRACK_FN_ATTR(inaccessiblememorargmemonly)
  }
};

/// AAMemoryLocation attribute for call sites.
struct AAMemoryLocationCallSite final : AAMemoryLocationImpl {
  AAMemoryLocationCallSite(const IRPosition &IRP, Attributor &A)
      : AAMemoryLocationImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAMemoryLocationImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || F->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Function *F = getAssociatedFunction();
    const IRPosition &FnPos = IRPosition::function(*F);
    auto &FnAA =
        A.getAAFor<AAMemoryLocation>(*this, FnPos, DepClassTy::REQUIRED);
    bool Changed = false;
    auto AccessPred = [&](const Instruction *I, const Value *Ptr,
                          AccessKind Kind, MemoryLocationsKind MLK) {
      updateStateAndAccessesMap(getState(), MLK, I, Ptr, Changed,
                                getAccessKindFromInst(I));
      return true;
    };
    if (!FnAA.checkForAllAccessesToMemoryKind(AccessPred, ALL_LOCATIONS))
      return indicatePessimisticFixpoint();
    return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    if (isAssumedReadNone())
      STATS_DECLTRACK_CS_ATTR(readnone)
  }
};

/// ------------------ Value Constant Range Attribute -------------------------

struct AAValueConstantRangeImpl : AAValueConstantRange {
  using StateType = IntegerRangeState;
  AAValueConstantRangeImpl(const IRPosition &IRP, Attributor &A)
      : AAValueConstantRange(IRP, A) {}

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << "range(" << getBitWidth() << ")<";
    getKnown().print(OS);
    OS << " / ";
    getAssumed().print(OS);
    OS << ">";
    return OS.str();
  }

  /// Helper function to get a SCEV expr for the associated value at program
  /// point \p I.
  const SCEV *getSCEV(Attributor &A, const Instruction *I = nullptr) const {
    if (!getAnchorScope())
      return nullptr;

    ScalarEvolution *SE =
        A.getInfoCache().getAnalysisResultForFunction<ScalarEvolutionAnalysis>(
            *getAnchorScope());

    LoopInfo *LI = A.getInfoCache().getAnalysisResultForFunction<LoopAnalysis>(
        *getAnchorScope());

    if (!SE || !LI)
      return nullptr;

    const SCEV *S = SE->getSCEV(&getAssociatedValue());
    if (!I)
      return S;

    return SE->getSCEVAtScope(S, LI->getLoopFor(I->getParent()));
  }

  /// Helper function to get a range from SCEV for the associated value at
  /// program point \p I.
  ConstantRange getConstantRangeFromSCEV(Attributor &A,
                                         const Instruction *I = nullptr) const {
    if (!getAnchorScope())
      return getWorstState(getBitWidth());

    ScalarEvolution *SE =
        A.getInfoCache().getAnalysisResultForFunction<ScalarEvolutionAnalysis>(
            *getAnchorScope());

    const SCEV *S = getSCEV(A, I);
    if (!SE || !S)
      return getWorstState(getBitWidth());

    return SE->getUnsignedRange(S);
  }

  /// Helper function to get a range from LVI for the associated value at
  /// program point \p I.
  ConstantRange
  getConstantRangeFromLVI(Attributor &A,
                          const Instruction *CtxI = nullptr) const {
    if (!getAnchorScope())
      return getWorstState(getBitWidth());

    LazyValueInfo *LVI =
        A.getInfoCache().getAnalysisResultForFunction<LazyValueAnalysis>(
            *getAnchorScope());

    if (!LVI || !CtxI)
      return getWorstState(getBitWidth());
    return LVI->getConstantRange(&getAssociatedValue(),
                                 const_cast<Instruction *>(CtxI));
  }

  /// See AAValueConstantRange::getKnownConstantRange(..).
  ConstantRange
  getKnownConstantRange(Attributor &A,
                        const Instruction *CtxI = nullptr) const override {
    if (!CtxI || CtxI == getCtxI())
      return getKnown();

    ConstantRange LVIR = getConstantRangeFromLVI(A, CtxI);
    ConstantRange SCEVR = getConstantRangeFromSCEV(A, CtxI);
    return getKnown().intersectWith(SCEVR).intersectWith(LVIR);
  }

  /// See AAValueConstantRange::getAssumedConstantRange(..).
  ConstantRange
  getAssumedConstantRange(Attributor &A,
                          const Instruction *CtxI = nullptr) const override {
    // TODO: Make SCEV use Attributor assumption.
    //       We may be able to bound a variable range via assumptions in
    //       Attributor. ex.) If x is assumed to be in [1, 3] and y is known to
    //       evolve to x^2 + x, then we can say that y is in [2, 12].

    if (!CtxI || CtxI == getCtxI())
      return getAssumed();

    ConstantRange LVIR = getConstantRangeFromLVI(A, CtxI);
    ConstantRange SCEVR = getConstantRangeFromSCEV(A, CtxI);
    return getAssumed().intersectWith(SCEVR).intersectWith(LVIR);
  }

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    // Intersect a range given by SCEV.
    intersectKnown(getConstantRangeFromSCEV(A, getCtxI()));

    // Intersect a range given by LVI.
    intersectKnown(getConstantRangeFromLVI(A, getCtxI()));
  }

  /// Helper function to create MDNode for range metadata.
  static MDNode *
  getMDNodeForConstantRange(Type *Ty, LLVMContext &Ctx,
                            const ConstantRange &AssumedConstantRange) {
    Metadata *LowAndHigh[] = {ConstantAsMetadata::get(ConstantInt::get(
                                  Ty, AssumedConstantRange.getLower())),
                              ConstantAsMetadata::get(ConstantInt::get(
                                  Ty, AssumedConstantRange.getUpper()))};
    return MDNode::get(Ctx, LowAndHigh);
  }

  /// Return true if \p Assumed is included in \p KnownRanges.
  static bool isBetterRange(const ConstantRange &Assumed, MDNode *KnownRanges) {

    if (Assumed.isFullSet())
      return false;

    if (!KnownRanges)
      return true;

    // If multiple ranges are annotated in IR, we give up to annotate assumed
    // range for now.

    // TODO:  If there exists a known range which containts assumed range, we
    // can say assumed range is better.
    if (KnownRanges->getNumOperands() > 2)
      return false;

    ConstantInt *Lower =
        mdconst::extract<ConstantInt>(KnownRanges->getOperand(0));
    ConstantInt *Upper =
        mdconst::extract<ConstantInt>(KnownRanges->getOperand(1));

    ConstantRange Known(Lower->getValue(), Upper->getValue());
    return Known.contains(Assumed) && Known != Assumed;
  }

  /// Helper function to set range metadata.
  static bool
  setRangeMetadataIfisBetterRange(Instruction *I,
                                  const ConstantRange &AssumedConstantRange) {
    auto *OldRangeMD = I->getMetadata(LLVMContext::MD_range);
    if (isBetterRange(AssumedConstantRange, OldRangeMD)) {
      if (!AssumedConstantRange.isEmptySet()) {
        I->setMetadata(LLVMContext::MD_range,
                       getMDNodeForConstantRange(I->getType(), I->getContext(),
                                                 AssumedConstantRange));
        return true;
      }
    }
    return false;
  }

  /// See AbstractAttribute::manifest()
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    ConstantRange AssumedConstantRange = getAssumedConstantRange(A);
    assert(!AssumedConstantRange.isFullSet() && "Invalid state");

    auto &V = getAssociatedValue();
    if (!AssumedConstantRange.isEmptySet() &&
        !AssumedConstantRange.isSingleElement()) {
      if (Instruction *I = dyn_cast<Instruction>(&V)) {
        assert(I == getCtxI() && "Should not annotate an instruction which is "
                                 "not the context instruction");
        if (isa<CallInst>(I) || isa<LoadInst>(I))
          if (setRangeMetadataIfisBetterRange(I, AssumedConstantRange))
            Changed = ChangeStatus::CHANGED;
      }
    }

    return Changed;
  }
};

struct AAValueConstantRangeArgument final
    : AAArgumentFromCallSiteArguments<
          AAValueConstantRange, AAValueConstantRangeImpl, IntegerRangeState,
          true /* BridgeCallBaseContext */> {
  using Base = AAArgumentFromCallSiteArguments<
      AAValueConstantRange, AAValueConstantRangeImpl, IntegerRangeState,
      true /* BridgeCallBaseContext */>;
  AAValueConstantRangeArgument(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    if (!getAnchorScope() || getAnchorScope()->isDeclaration()) {
      indicatePessimisticFixpoint();
    } else {
      Base::initialize(A);
    }
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(value_range)
  }
};

struct AAValueConstantRangeReturned
    : AAReturnedFromReturnedValues<AAValueConstantRange,
                                   AAValueConstantRangeImpl,
                                   AAValueConstantRangeImpl::StateType,
                                   /* PropogateCallBaseContext */ true> {
  using Base =
      AAReturnedFromReturnedValues<AAValueConstantRange,
                                   AAValueConstantRangeImpl,
                                   AAValueConstantRangeImpl::StateType,
                                   /* PropogateCallBaseContext */ true>;
  AAValueConstantRangeReturned(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(value_range)
  }
};

struct AAValueConstantRangeFloating : AAValueConstantRangeImpl {
  AAValueConstantRangeFloating(const IRPosition &IRP, Attributor &A)
      : AAValueConstantRangeImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAValueConstantRangeImpl::initialize(A);
    Value &V = getAssociatedValue();

    if (auto *C = dyn_cast<ConstantInt>(&V)) {
      unionAssumed(ConstantRange(C->getValue()));
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<UndefValue>(&V)) {
      // Collapse the undef state to 0.
      unionAssumed(ConstantRange(APInt(getBitWidth(), 0)));
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<CallBase>(&V))
      return;

    if (isa<BinaryOperator>(&V) || isa<CmpInst>(&V) || isa<CastInst>(&V))
      return;
    // If it is a load instruction with range metadata, use it.
    if (LoadInst *LI = dyn_cast<LoadInst>(&V))
      if (auto *RangeMD = LI->getMetadata(LLVMContext::MD_range)) {
        intersectKnown(getConstantRangeFromMetadata(*RangeMD));
        return;
      }

    // We can work with PHI and select instruction as we traverse their operands
    // during update.
    if (isa<SelectInst>(V) || isa<PHINode>(V))
      return;

    // Otherwise we give up.
    indicatePessimisticFixpoint();

    LLVM_DEBUG(dbgs() << "[AAValueConstantRange] We give up: "
                      << getAssociatedValue() << "\n");
  }

  bool calculateBinaryOperator(
      Attributor &A, BinaryOperator *BinOp, IntegerRangeState &T,
      const Instruction *CtxI,
      SmallVectorImpl<const AAValueConstantRange *> &QuerriedAAs) {
    Value *LHS = BinOp->getOperand(0);
    Value *RHS = BinOp->getOperand(1);
    // TODO: Allow non integers as well.
    if (!LHS->getType()->isIntegerTy() || !RHS->getType()->isIntegerTy())
      return false;

    auto &LHSAA = A.getAAFor<AAValueConstantRange>(
        *this, IRPosition::value(*LHS, getCallBaseContext()),
        DepClassTy::REQUIRED);
    QuerriedAAs.push_back(&LHSAA);
    auto LHSAARange = LHSAA.getAssumedConstantRange(A, CtxI);

    auto &RHSAA = A.getAAFor<AAValueConstantRange>(
        *this, IRPosition::value(*RHS, getCallBaseContext()),
        DepClassTy::REQUIRED);
    QuerriedAAs.push_back(&RHSAA);
    auto RHSAARange = RHSAA.getAssumedConstantRange(A, CtxI);

    auto AssumedRange = LHSAARange.binaryOp(BinOp->getOpcode(), RHSAARange);

    T.unionAssumed(AssumedRange);

    // TODO: Track a known state too.

    return T.isValidState();
  }

  bool calculateCastInst(
      Attributor &A, CastInst *CastI, IntegerRangeState &T,
      const Instruction *CtxI,
      SmallVectorImpl<const AAValueConstantRange *> &QuerriedAAs) {
    assert(CastI->getNumOperands() == 1 && "Expected cast to be unary!");
    // TODO: Allow non integers as well.
    Value &OpV = *CastI->getOperand(0);
    if (!OpV.getType()->isIntegerTy())
      return false;

    auto &OpAA = A.getAAFor<AAValueConstantRange>(
        *this, IRPosition::value(OpV, getCallBaseContext()),
        DepClassTy::REQUIRED);
    QuerriedAAs.push_back(&OpAA);
    T.unionAssumed(
        OpAA.getAssumed().castOp(CastI->getOpcode(), getState().getBitWidth()));
    return T.isValidState();
  }

  bool
  calculateCmpInst(Attributor &A, CmpInst *CmpI, IntegerRangeState &T,
                   const Instruction *CtxI,
                   SmallVectorImpl<const AAValueConstantRange *> &QuerriedAAs) {
    Value *LHS = CmpI->getOperand(0);
    Value *RHS = CmpI->getOperand(1);
    // TODO: Allow non integers as well.
    if (!LHS->getType()->isIntegerTy() || !RHS->getType()->isIntegerTy())
      return false;

    auto &LHSAA = A.getAAFor<AAValueConstantRange>(
        *this, IRPosition::value(*LHS, getCallBaseContext()),
        DepClassTy::REQUIRED);
    QuerriedAAs.push_back(&LHSAA);
    auto &RHSAA = A.getAAFor<AAValueConstantRange>(
        *this, IRPosition::value(*RHS, getCallBaseContext()),
        DepClassTy::REQUIRED);
    auto LHSAARange = LHSAA.getAssumedConstantRange(A, CtxI);
    auto RHSAARange = RHSAA.getAssumedConstantRange(A, CtxI);

    // If one of them is empty set, we can't decide.
    if (LHSAARange.isEmptySet() || RHSAARange.isEmptySet())
      return true;

    bool MustTrue = false, MustFalse = false;

    auto AllowedRegion =
        ConstantRange::makeAllowedICmpRegion(CmpI->getPredicate(), RHSAARange);

    auto SatisfyingRegion = ConstantRange::makeSatisfyingICmpRegion(
        CmpI->getPredicate(), RHSAARange);

    if (AllowedRegion.intersectWith(LHSAARange).isEmptySet())
      MustFalse = true;

    if (SatisfyingRegion.contains(LHSAARange))
      MustTrue = true;

    assert((!MustTrue || !MustFalse) &&
           "Either MustTrue or MustFalse should be false!");

    if (MustTrue)
      T.unionAssumed(ConstantRange(APInt(/* numBits */ 1, /* val */ 1)));
    else if (MustFalse)
      T.unionAssumed(ConstantRange(APInt(/* numBits */ 1, /* val */ 0)));
    else
      T.unionAssumed(ConstantRange(/* BitWidth */ 1, /* isFullSet */ true));

    LLVM_DEBUG(dbgs() << "[AAValueConstantRange] " << *CmpI << " " << LHSAA
                      << " " << RHSAA << "\n");

    // TODO: Track a known state too.
    return T.isValidState();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto VisitValueCB = [&](Value &V, const Instruction *CtxI,
                            IntegerRangeState &T, bool Stripped) -> bool {
      Instruction *I = dyn_cast<Instruction>(&V);
      if (!I || isa<CallBase>(I)) {

        // If the value is not instruction, we query AA to Attributor.
        const auto &AA = A.getAAFor<AAValueConstantRange>(
            *this, IRPosition::value(V), DepClassTy::REQUIRED);

        // Clamp operator is not used to utilize a program point CtxI.
        T.unionAssumed(AA.getAssumedConstantRange(A, CtxI));

        return T.isValidState();
      }

      SmallVector<const AAValueConstantRange *, 4> QuerriedAAs;
      if (auto *BinOp = dyn_cast<BinaryOperator>(I)) {
        if (!calculateBinaryOperator(A, BinOp, T, CtxI, QuerriedAAs))
          return false;
      } else if (auto *CmpI = dyn_cast<CmpInst>(I)) {
        if (!calculateCmpInst(A, CmpI, T, CtxI, QuerriedAAs))
          return false;
      } else if (auto *CastI = dyn_cast<CastInst>(I)) {
        if (!calculateCastInst(A, CastI, T, CtxI, QuerriedAAs))
          return false;
      } else {
        // Give up with other instructions.
        // TODO: Add other instructions

        T.indicatePessimisticFixpoint();
        return false;
      }

      // Catch circular reasoning in a pessimistic way for now.
      // TODO: Check how the range evolves and if we stripped anything, see also
      //       AADereferenceable or AAAlign for similar situations.
      for (const AAValueConstantRange *QueriedAA : QuerriedAAs) {
        if (QueriedAA != this)
          continue;
        // If we are in a stady state we do not need to worry.
        if (T.getAssumed() == getState().getAssumed())
          continue;
        T.indicatePessimisticFixpoint();
      }

      return T.isValidState();
    };

    IntegerRangeState T(getBitWidth());

    if (!genericValueTraversal<AAValueConstantRange, IntegerRangeState>(
            A, getIRPosition(), *this, T, VisitValueCB, getCtxI(),
            /* UseValueSimplify */ false))
      return indicatePessimisticFixpoint();

    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(value_range)
  }
};

struct AAValueConstantRangeFunction : AAValueConstantRangeImpl {
  AAValueConstantRangeFunction(const IRPosition &IRP, Attributor &A)
      : AAValueConstantRangeImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("AAValueConstantRange(Function|CallSite)::updateImpl will "
                     "not be called");
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(value_range) }
};

struct AAValueConstantRangeCallSite : AAValueConstantRangeFunction {
  AAValueConstantRangeCallSite(const IRPosition &IRP, Attributor &A)
      : AAValueConstantRangeFunction(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(value_range) }
};

struct AAValueConstantRangeCallSiteReturned
    : AACallSiteReturnedFromReturned<AAValueConstantRange,
                                     AAValueConstantRangeImpl,
                                     AAValueConstantRangeImpl::StateType,
                                     /* IntroduceCallBaseContext */ true> {
  AAValueConstantRangeCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AACallSiteReturnedFromReturned<AAValueConstantRange,
                                       AAValueConstantRangeImpl,
                                       AAValueConstantRangeImpl::StateType,
                                       /* IntroduceCallBaseContext */ true>(IRP,
                                                                            A) {
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // If it is a load instruction with range metadata, use the metadata.
    if (CallInst *CI = dyn_cast<CallInst>(&getAssociatedValue()))
      if (auto *RangeMD = CI->getMetadata(LLVMContext::MD_range))
        intersectKnown(getConstantRangeFromMetadata(*RangeMD));

    AAValueConstantRangeImpl::initialize(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(value_range)
  }
};
struct AAValueConstantRangeCallSiteArgument : AAValueConstantRangeFloating {
  AAValueConstantRangeCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAValueConstantRangeFloating(IRP, A) {}

  /// See AbstractAttribute::manifest()
  ChangeStatus manifest(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(value_range)
  }
};

/// ------------------ Potential Values Attribute -------------------------

struct AAPotentialValuesImpl : AAPotentialValues {
  using StateType = PotentialConstantIntValuesState;

  AAPotentialValuesImpl(const IRPosition &IRP, Attributor &A)
      : AAPotentialValues(IRP, A) {}

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << getState();
    return OS.str();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return indicatePessimisticFixpoint();
  }
};

struct AAPotentialValuesArgument final
    : AAArgumentFromCallSiteArguments<AAPotentialValues, AAPotentialValuesImpl,
                                      PotentialConstantIntValuesState> {
  using Base =
      AAArgumentFromCallSiteArguments<AAPotentialValues, AAPotentialValuesImpl,
                                      PotentialConstantIntValuesState>;
  AAPotentialValuesArgument(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    if (!getAnchorScope() || getAnchorScope()->isDeclaration()) {
      indicatePessimisticFixpoint();
    } else {
      Base::initialize(A);
    }
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(potential_values)
  }
};

struct AAPotentialValuesReturned
    : AAReturnedFromReturnedValues<AAPotentialValues, AAPotentialValuesImpl> {
  using Base =
      AAReturnedFromReturnedValues<AAPotentialValues, AAPotentialValuesImpl>;
  AAPotentialValuesReturned(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(potential_values)
  }
};

struct AAPotentialValuesFloating : AAPotentialValuesImpl {
  AAPotentialValuesFloating(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    Value &V = getAssociatedValue();

    if (auto *C = dyn_cast<ConstantInt>(&V)) {
      unionAssumed(C->getValue());
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<UndefValue>(&V)) {
      unionAssumedWithUndef();
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<BinaryOperator>(&V) || isa<ICmpInst>(&V) || isa<CastInst>(&V))
      return;

    if (isa<SelectInst>(V) || isa<PHINode>(V))
      return;

    indicatePessimisticFixpoint();

    LLVM_DEBUG(dbgs() << "[AAPotentialValues] We give up: "
                      << getAssociatedValue() << "\n");
  }

  static bool calculateICmpInst(const ICmpInst *ICI, const APInt &LHS,
                                const APInt &RHS) {
    ICmpInst::Predicate Pred = ICI->getPredicate();
    switch (Pred) {
    case ICmpInst::ICMP_UGT:
      return LHS.ugt(RHS);
    case ICmpInst::ICMP_SGT:
      return LHS.sgt(RHS);
    case ICmpInst::ICMP_EQ:
      return LHS.eq(RHS);
    case ICmpInst::ICMP_UGE:
      return LHS.uge(RHS);
    case ICmpInst::ICMP_SGE:
      return LHS.sge(RHS);
    case ICmpInst::ICMP_ULT:
      return LHS.ult(RHS);
    case ICmpInst::ICMP_SLT:
      return LHS.slt(RHS);
    case ICmpInst::ICMP_NE:
      return LHS.ne(RHS);
    case ICmpInst::ICMP_ULE:
      return LHS.ule(RHS);
    case ICmpInst::ICMP_SLE:
      return LHS.sle(RHS);
    default:
      llvm_unreachable("Invalid ICmp predicate!");
    }
  }

  static APInt calculateCastInst(const CastInst *CI, const APInt &Src,
                                 uint32_t ResultBitWidth) {
    Instruction::CastOps CastOp = CI->getOpcode();
    switch (CastOp) {
    default:
      llvm_unreachable("unsupported or not integer cast");
    case Instruction::Trunc:
      return Src.trunc(ResultBitWidth);
    case Instruction::SExt:
      return Src.sext(ResultBitWidth);
    case Instruction::ZExt:
      return Src.zext(ResultBitWidth);
    case Instruction::BitCast:
      return Src;
    }
  }

  static APInt calculateBinaryOperator(const BinaryOperator *BinOp,
                                       const APInt &LHS, const APInt &RHS,
                                       bool &SkipOperation, bool &Unsupported) {
    Instruction::BinaryOps BinOpcode = BinOp->getOpcode();
    // Unsupported is set to true when the binary operator is not supported.
    // SkipOperation is set to true when UB occur with the given operand pair
    // (LHS, RHS).
    // TODO: we should look at nsw and nuw keywords to handle operations
    //       that create poison or undef value.
    switch (BinOpcode) {
    default:
      Unsupported = true;
      return LHS;
    case Instruction::Add:
      return LHS + RHS;
    case Instruction::Sub:
      return LHS - RHS;
    case Instruction::Mul:
      return LHS * RHS;
    case Instruction::UDiv:
      if (RHS.isNullValue()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.udiv(RHS);
    case Instruction::SDiv:
      if (RHS.isNullValue()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.sdiv(RHS);
    case Instruction::URem:
      if (RHS.isNullValue()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.urem(RHS);
    case Instruction::SRem:
      if (RHS.isNullValue()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.srem(RHS);
    case Instruction::Shl:
      return LHS.shl(RHS);
    case Instruction::LShr:
      return LHS.lshr(RHS);
    case Instruction::AShr:
      return LHS.ashr(RHS);
    case Instruction::And:
      return LHS & RHS;
    case Instruction::Or:
      return LHS | RHS;
    case Instruction::Xor:
      return LHS ^ RHS;
    }
  }

  bool calculateBinaryOperatorAndTakeUnion(const BinaryOperator *BinOp,
                                           const APInt &LHS, const APInt &RHS) {
    bool SkipOperation = false;
    bool Unsupported = false;
    APInt Result =
        calculateBinaryOperator(BinOp, LHS, RHS, SkipOperation, Unsupported);
    if (Unsupported)
      return false;
    // If SkipOperation is true, we can ignore this operand pair (L, R).
    if (!SkipOperation)
      unionAssumed(Result);
    return isValidState();
  }

  ChangeStatus updateWithICmpInst(Attributor &A, ICmpInst *ICI) {
    auto AssumedBefore = getAssumed();
    Value *LHS = ICI->getOperand(0);
    Value *RHS = ICI->getOperand(1);
    if (!LHS->getType()->isIntegerTy() || !RHS->getType()->isIntegerTy())
      return indicatePessimisticFixpoint();

    auto &LHSAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*LHS),
                                                DepClassTy::REQUIRED);
    if (!LHSAA.isValidState())
      return indicatePessimisticFixpoint();

    auto &RHSAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*RHS),
                                                DepClassTy::REQUIRED);
    if (!RHSAA.isValidState())
      return indicatePessimisticFixpoint();

    const DenseSet<APInt> &LHSAAPVS = LHSAA.getAssumedSet();
    const DenseSet<APInt> &RHSAAPVS = RHSAA.getAssumedSet();

    // TODO: make use of undef flag to limit potential values aggressively.
    bool MaybeTrue = false, MaybeFalse = false;
    const APInt Zero(RHS->getType()->getIntegerBitWidth(), 0);
    if (LHSAA.undefIsContained() && RHSAA.undefIsContained()) {
      // The result of any comparison between undefs can be soundly replaced
      // with undef.
      unionAssumedWithUndef();
    } else if (LHSAA.undefIsContained()) {
      bool MaybeTrue = false, MaybeFalse = false;
      for (const APInt &R : RHSAAPVS) {
        bool CmpResult = calculateICmpInst(ICI, Zero, R);
        MaybeTrue |= CmpResult;
        MaybeFalse |= !CmpResult;
        if (MaybeTrue & MaybeFalse)
          return indicatePessimisticFixpoint();
      }
    } else if (RHSAA.undefIsContained()) {
      for (const APInt &L : LHSAAPVS) {
        bool CmpResult = calculateICmpInst(ICI, L, Zero);
        MaybeTrue |= CmpResult;
        MaybeFalse |= !CmpResult;
        if (MaybeTrue & MaybeFalse)
          return indicatePessimisticFixpoint();
      }
    } else {
      for (const APInt &L : LHSAAPVS) {
        for (const APInt &R : RHSAAPVS) {
          bool CmpResult = calculateICmpInst(ICI, L, R);
          MaybeTrue |= CmpResult;
          MaybeFalse |= !CmpResult;
          if (MaybeTrue & MaybeFalse)
            return indicatePessimisticFixpoint();
        }
      }
    }
    if (MaybeTrue)
      unionAssumed(APInt(/* numBits */ 1, /* val */ 1));
    if (MaybeFalse)
      unionAssumed(APInt(/* numBits */ 1, /* val */ 0));
    return AssumedBefore == getAssumed() ? ChangeStatus::UNCHANGED
                                         : ChangeStatus::CHANGED;
  }

  ChangeStatus updateWithSelectInst(Attributor &A, SelectInst *SI) {
    auto AssumedBefore = getAssumed();
    Value *LHS = SI->getTrueValue();
    Value *RHS = SI->getFalseValue();
    if (!LHS->getType()->isIntegerTy() || !RHS->getType()->isIntegerTy())
      return indicatePessimisticFixpoint();

    // TODO: Use assumed simplified condition value
    auto &LHSAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*LHS),
                                                DepClassTy::REQUIRED);
    if (!LHSAA.isValidState())
      return indicatePessimisticFixpoint();

    auto &RHSAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*RHS),
                                                DepClassTy::REQUIRED);
    if (!RHSAA.isValidState())
      return indicatePessimisticFixpoint();

    if (LHSAA.undefIsContained() && RHSAA.undefIsContained())
      // select i1 *, undef , undef => undef
      unionAssumedWithUndef();
    else {
      unionAssumed(LHSAA);
      unionAssumed(RHSAA);
    }
    return AssumedBefore == getAssumed() ? ChangeStatus::UNCHANGED
                                         : ChangeStatus::CHANGED;
  }

  ChangeStatus updateWithCastInst(Attributor &A, CastInst *CI) {
    auto AssumedBefore = getAssumed();
    if (!CI->isIntegerCast())
      return indicatePessimisticFixpoint();
    assert(CI->getNumOperands() == 1 && "Expected cast to be unary!");
    uint32_t ResultBitWidth = CI->getDestTy()->getIntegerBitWidth();
    Value *Src = CI->getOperand(0);
    auto &SrcAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*Src),
                                                DepClassTy::REQUIRED);
    if (!SrcAA.isValidState())
      return indicatePessimisticFixpoint();
    const DenseSet<APInt> &SrcAAPVS = SrcAA.getAssumedSet();
    if (SrcAA.undefIsContained())
      unionAssumedWithUndef();
    else {
      for (const APInt &S : SrcAAPVS) {
        APInt T = calculateCastInst(CI, S, ResultBitWidth);
        unionAssumed(T);
      }
    }
    return AssumedBefore == getAssumed() ? ChangeStatus::UNCHANGED
                                         : ChangeStatus::CHANGED;
  }

  ChangeStatus updateWithBinaryOperator(Attributor &A, BinaryOperator *BinOp) {
    auto AssumedBefore = getAssumed();
    Value *LHS = BinOp->getOperand(0);
    Value *RHS = BinOp->getOperand(1);
    if (!LHS->getType()->isIntegerTy() || !RHS->getType()->isIntegerTy())
      return indicatePessimisticFixpoint();

    auto &LHSAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*LHS),
                                                DepClassTy::REQUIRED);
    if (!LHSAA.isValidState())
      return indicatePessimisticFixpoint();

    auto &RHSAA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(*RHS),
                                                DepClassTy::REQUIRED);
    if (!RHSAA.isValidState())
      return indicatePessimisticFixpoint();

    const DenseSet<APInt> &LHSAAPVS = LHSAA.getAssumedSet();
    const DenseSet<APInt> &RHSAAPVS = RHSAA.getAssumedSet();
    const APInt Zero = APInt(LHS->getType()->getIntegerBitWidth(), 0);

    // TODO: make use of undef flag to limit potential values aggressively.
    if (LHSAA.undefIsContained() && RHSAA.undefIsContained()) {
      if (!calculateBinaryOperatorAndTakeUnion(BinOp, Zero, Zero))
        return indicatePessimisticFixpoint();
    } else if (LHSAA.undefIsContained()) {
      for (const APInt &R : RHSAAPVS) {
        if (!calculateBinaryOperatorAndTakeUnion(BinOp, Zero, R))
          return indicatePessimisticFixpoint();
      }
    } else if (RHSAA.undefIsContained()) {
      for (const APInt &L : LHSAAPVS) {
        if (!calculateBinaryOperatorAndTakeUnion(BinOp, L, Zero))
          return indicatePessimisticFixpoint();
      }
    } else {
      for (const APInt &L : LHSAAPVS) {
        for (const APInt &R : RHSAAPVS) {
          if (!calculateBinaryOperatorAndTakeUnion(BinOp, L, R))
            return indicatePessimisticFixpoint();
        }
      }
    }
    return AssumedBefore == getAssumed() ? ChangeStatus::UNCHANGED
                                         : ChangeStatus::CHANGED;
  }

  ChangeStatus updateWithPHINode(Attributor &A, PHINode *PHI) {
    auto AssumedBefore = getAssumed();
    for (unsigned u = 0, e = PHI->getNumIncomingValues(); u < e; u++) {
      Value *IncomingValue = PHI->getIncomingValue(u);
      auto &PotentialValuesAA = A.getAAFor<AAPotentialValues>(
          *this, IRPosition::value(*IncomingValue), DepClassTy::REQUIRED);
      if (!PotentialValuesAA.isValidState())
        return indicatePessimisticFixpoint();
      if (PotentialValuesAA.undefIsContained())
        unionAssumedWithUndef();
      else
        unionAssumed(PotentialValuesAA.getAssumed());
    }
    return AssumedBefore == getAssumed() ? ChangeStatus::UNCHANGED
                                         : ChangeStatus::CHANGED;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    Value &V = getAssociatedValue();
    Instruction *I = dyn_cast<Instruction>(&V);

    if (auto *ICI = dyn_cast<ICmpInst>(I))
      return updateWithICmpInst(A, ICI);

    if (auto *SI = dyn_cast<SelectInst>(I))
      return updateWithSelectInst(A, SI);

    if (auto *CI = dyn_cast<CastInst>(I))
      return updateWithCastInst(A, CI);

    if (auto *BinOp = dyn_cast<BinaryOperator>(I))
      return updateWithBinaryOperator(A, BinOp);

    if (auto *PHI = dyn_cast<PHINode>(I))
      return updateWithPHINode(A, PHI);

    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(potential_values)
  }
};

struct AAPotentialValuesFunction : AAPotentialValuesImpl {
  AAPotentialValuesFunction(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("AAPotentialValues(Function|CallSite)::updateImpl will "
                     "not be called");
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FN_ATTR(potential_values)
  }
};

struct AAPotentialValuesCallSite : AAPotentialValuesFunction {
  AAPotentialValuesCallSite(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesFunction(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(potential_values)
  }
};

struct AAPotentialValuesCallSiteReturned
    : AACallSiteReturnedFromReturned<AAPotentialValues, AAPotentialValuesImpl> {
  AAPotentialValuesCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AACallSiteReturnedFromReturned<AAPotentialValues,
                                       AAPotentialValuesImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(potential_values)
  }
};

struct AAPotentialValuesCallSiteArgument : AAPotentialValuesFloating {
  AAPotentialValuesCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    Value &V = getAssociatedValue();

    if (auto *C = dyn_cast<ConstantInt>(&V)) {
      unionAssumed(C->getValue());
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<UndefValue>(&V)) {
      unionAssumedWithUndef();
      indicateOptimisticFixpoint();
      return;
    }
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    Value &V = getAssociatedValue();
    auto AssumedBefore = getAssumed();
    auto &AA = A.getAAFor<AAPotentialValues>(*this, IRPosition::value(V),
                                             DepClassTy::REQUIRED);
    const auto &S = AA.getAssumed();
    unionAssumed(S);
    return AssumedBefore == getAssumed() ? ChangeStatus::UNCHANGED
                                         : ChangeStatus::CHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(potential_values)
  }
};

/// ------------------------ NoUndef Attribute ---------------------------------
struct AANoUndefImpl : AANoUndef {
  AANoUndefImpl(const IRPosition &IRP, Attributor &A) : AANoUndef(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (getIRPosition().hasAttr({Attribute::NoUndef})) {
      indicateOptimisticFixpoint();
      return;
    }
    Value &V = getAssociatedValue();
    if (isa<UndefValue>(V))
      indicatePessimisticFixpoint();
    else if (isa<FreezeInst>(V))
      indicateOptimisticFixpoint();
    else if (getPositionKind() != IRPosition::IRP_RETURNED &&
             isGuaranteedNotToBeUndefOrPoison(&V))
      indicateOptimisticFixpoint();
    else
      AANoUndef::initialize(A);
  }

  /// See followUsesInMBEC
  bool followUseInMBEC(Attributor &A, const Use *U, const Instruction *I,
                       AANoUndef::StateType &State) {
    const Value *UseV = U->get();
    const DominatorTree *DT = nullptr;
    AssumptionCache *AC = nullptr;
    InformationCache &InfoCache = A.getInfoCache();
    if (Function *F = getAnchorScope()) {
      DT = InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(*F);
      AC = InfoCache.getAnalysisResultForFunction<AssumptionAnalysis>(*F);
    }
    State.setKnown(isGuaranteedNotToBeUndefOrPoison(UseV, AC, I, DT));
    bool TrackUse = false;
    // Track use for instructions which must produce undef or poison bits when
    // at least one operand contains such bits.
    if (isa<CastInst>(*I) || isa<GetElementPtrInst>(*I))
      TrackUse = true;
    return TrackUse;
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "noundef" : "may-undef-or-poison";
  }

  ChangeStatus manifest(Attributor &A) override {
    // We don't manifest noundef attribute for dead positions because the
    // associated values with dead positions would be replaced with undef
    // values.
    if (A.isAssumedDead(getIRPosition(), nullptr, nullptr))
      return ChangeStatus::UNCHANGED;
    // A position whose simplified value does not have any value is
    // considered to be dead. We don't manifest noundef in such positions for
    // the same reason above.
    auto &ValueSimplifyAA =
        A.getAAFor<AAValueSimplify>(*this, getIRPosition(), DepClassTy::NONE);
    if (!ValueSimplifyAA.getAssumedSimplifiedValue(A).hasValue())
      return ChangeStatus::UNCHANGED;
    return AANoUndef::manifest(A);
  }
};

struct AANoUndefFloating : public AANoUndefImpl {
  AANoUndefFloating(const IRPosition &IRP, Attributor &A)
      : AANoUndefImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoUndefImpl::initialize(A);
    if (!getState().isAtFixpoint())
      if (Instruction *CtxI = getCtxI())
        followUsesInMBEC(*this, A, getState(), *CtxI);
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto VisitValueCB = [&](Value &V, const Instruction *CtxI,
                            AANoUndef::StateType &T, bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AANoUndef>(*this, IRPosition::value(V),
                                             DepClassTy::REQUIRED);
      if (!Stripped && this == &AA) {
        T.indicatePessimisticFixpoint();
      } else {
        const AANoUndef::StateType &S =
            static_cast<const AANoUndef::StateType &>(AA.getState());
        T ^= S;
      }
      return T.isValidState();
    };

    StateType T;
    if (!genericValueTraversal<AANoUndef, StateType>(
            A, getIRPosition(), *this, T, VisitValueCB, getCtxI()))
      return indicatePessimisticFixpoint();

    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(noundef) }
};

struct AANoUndefReturned final
    : AAReturnedFromReturnedValues<AANoUndef, AANoUndefImpl> {
  AANoUndefReturned(const IRPosition &IRP, Attributor &A)
      : AAReturnedFromReturnedValues<AANoUndef, AANoUndefImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(noundef) }
};

struct AANoUndefArgument final
    : AAArgumentFromCallSiteArguments<AANoUndef, AANoUndefImpl> {
  AANoUndefArgument(const IRPosition &IRP, Attributor &A)
      : AAArgumentFromCallSiteArguments<AANoUndef, AANoUndefImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(noundef) }
};

struct AANoUndefCallSiteArgument final : AANoUndefFloating {
  AANoUndefCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AANoUndefFloating(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(noundef) }
};

struct AANoUndefCallSiteReturned final
    : AACallSiteReturnedFromReturned<AANoUndef, AANoUndefImpl> {
  AANoUndefCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AACallSiteReturnedFromReturned<AANoUndef, AANoUndefImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(noundef) }
};
} // namespace

const char AAReturnedValues::ID = 0;
const char AANoUnwind::ID = 0;
const char AANoSync::ID = 0;
const char AANoFree::ID = 0;
const char AANonNull::ID = 0;
const char AANoRecurse::ID = 0;
const char AAWillReturn::ID = 0;
const char AAUndefinedBehavior::ID = 0;
const char AANoAlias::ID = 0;
const char AAReachability::ID = 0;
const char AANoReturn::ID = 0;
const char AAIsDead::ID = 0;
const char AADereferenceable::ID = 0;
const char AAAlign::ID = 0;
const char AANoCapture::ID = 0;
const char AAValueSimplify::ID = 0;
const char AAHeapToStack::ID = 0;
const char AAPrivatizablePtr::ID = 0;
const char AAMemoryBehavior::ID = 0;
const char AAMemoryLocation::ID = 0;
const char AAValueConstantRange::ID = 0;
const char AAPotentialValues::ID = 0;
const char AANoUndef::ID = 0;

// Macro magic to create the static generator function for attributes that
// follow the naming scheme.

#define SWITCH_PK_INV(CLASS, PK, POS_NAME)                                     \
  case IRPosition::PK:                                                         \
    llvm_unreachable("Cannot create " #CLASS " for a " POS_NAME " position!");

#define SWITCH_PK_CREATE(CLASS, IRP, PK, SUFFIX)                               \
  case IRPosition::PK:                                                         \
    AA = new (A.Allocator) CLASS##SUFFIX(IRP, A);                              \
    ++NumAAs;                                                                  \
    break;

#define CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(CLASS)                 \
  CLASS &CLASS::createForPosition(const IRPosition &IRP, Attributor &A) {      \
    CLASS *AA = nullptr;                                                       \
    switch (IRP.getPositionKind()) {                                           \
      SWITCH_PK_INV(CLASS, IRP_INVALID, "invalid")                             \
      SWITCH_PK_INV(CLASS, IRP_FLOAT, "floating")                              \
      SWITCH_PK_INV(CLASS, IRP_ARGUMENT, "argument")                           \
      SWITCH_PK_INV(CLASS, IRP_RETURNED, "returned")                           \
      SWITCH_PK_INV(CLASS, IRP_CALL_SITE_RETURNED, "call site returned")       \
      SWITCH_PK_INV(CLASS, IRP_CALL_SITE_ARGUMENT, "call site argument")       \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FUNCTION, Function)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE, CallSite)                    \
    }                                                                          \
    return *AA;                                                                \
  }

#define CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(CLASS)                    \
  CLASS &CLASS::createForPosition(const IRPosition &IRP, Attributor &A) {      \
    CLASS *AA = nullptr;                                                       \
    switch (IRP.getPositionKind()) {                                           \
      SWITCH_PK_INV(CLASS, IRP_INVALID, "invalid")                             \
      SWITCH_PK_INV(CLASS, IRP_FUNCTION, "function")                           \
      SWITCH_PK_INV(CLASS, IRP_CALL_SITE, "call site")                         \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FLOAT, Floating)                        \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_ARGUMENT, Argument)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_RETURNED, Returned)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE_RETURNED, CallSiteReturned)   \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE_ARGUMENT, CallSiteArgument)   \
    }                                                                          \
    return *AA;                                                                \
  }

#define CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(CLASS)                      \
  CLASS &CLASS::createForPosition(const IRPosition &IRP, Attributor &A) {      \
    CLASS *AA = nullptr;                                                       \
    switch (IRP.getPositionKind()) {                                           \
      SWITCH_PK_INV(CLASS, IRP_INVALID, "invalid")                             \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FUNCTION, Function)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE, CallSite)                    \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FLOAT, Floating)                        \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_ARGUMENT, Argument)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_RETURNED, Returned)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE_RETURNED, CallSiteReturned)   \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE_ARGUMENT, CallSiteArgument)   \
    }                                                                          \
    return *AA;                                                                \
  }

#define CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(CLASS)            \
  CLASS &CLASS::createForPosition(const IRPosition &IRP, Attributor &A) {      \
    CLASS *AA = nullptr;                                                       \
    switch (IRP.getPositionKind()) {                                           \
      SWITCH_PK_INV(CLASS, IRP_INVALID, "invalid")                             \
      SWITCH_PK_INV(CLASS, IRP_ARGUMENT, "argument")                           \
      SWITCH_PK_INV(CLASS, IRP_FLOAT, "floating")                              \
      SWITCH_PK_INV(CLASS, IRP_RETURNED, "returned")                           \
      SWITCH_PK_INV(CLASS, IRP_CALL_SITE_RETURNED, "call site returned")       \
      SWITCH_PK_INV(CLASS, IRP_CALL_SITE_ARGUMENT, "call site argument")       \
      SWITCH_PK_INV(CLASS, IRP_CALL_SITE, "call site")                         \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FUNCTION, Function)                     \
    }                                                                          \
    return *AA;                                                                \
  }

#define CREATE_NON_RET_ABSTRACT_ATTRIBUTE_FOR_POSITION(CLASS)                  \
  CLASS &CLASS::createForPosition(const IRPosition &IRP, Attributor &A) {      \
    CLASS *AA = nullptr;                                                       \
    switch (IRP.getPositionKind()) {                                           \
      SWITCH_PK_INV(CLASS, IRP_INVALID, "invalid")                             \
      SWITCH_PK_INV(CLASS, IRP_RETURNED, "returned")                           \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FUNCTION, Function)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE, CallSite)                    \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_FLOAT, Floating)                        \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_ARGUMENT, Argument)                     \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE_RETURNED, CallSiteReturned)   \
      SWITCH_PK_CREATE(CLASS, IRP, IRP_CALL_SITE_ARGUMENT, CallSiteArgument)   \
    }                                                                          \
    return *AA;                                                                \
  }

CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoUnwind)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoSync)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoRecurse)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAWillReturn)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoReturn)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAReturnedValues)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAMemoryLocation)

CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANonNull)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoAlias)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAPrivatizablePtr)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AADereferenceable)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAAlign)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoCapture)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAValueConstantRange)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAPotentialValues)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoUndef)

CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAValueSimplify)
CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAIsDead)
CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoFree)

CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAHeapToStack)
CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAReachability)
CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAUndefinedBehavior)

CREATE_NON_RET_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAMemoryBehavior)

#undef CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_NON_RET_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef SWITCH_PK_CREATE
#undef SWITCH_PK_INV
