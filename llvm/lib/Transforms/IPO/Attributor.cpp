//===- Attributor.cpp - Module-wide attribute deduction -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an inter procedural pass that deduces and/or propagating
// attributes. This is done in an abstract interpretation style fixpoint
// iteration. See the Attributor.h file comment and the class descriptions in
// that file for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Attributor.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "attributor"

STATISTIC(NumFnWithExactDefinition,
          "Number of function with exact definitions");
STATISTIC(NumFnWithoutExactDefinition,
          "Number of function without exact definitions");
STATISTIC(NumAttributesTimedOut,
          "Number of abstract attributes timed out before fixpoint");
STATISTIC(NumAttributesValidFixpoint,
          "Number of abstract attributes in a valid fixpoint state");
STATISTIC(NumAttributesManifested,
          "Number of abstract attributes manifested in IR");

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
// sides, STATS_DECL and STATS_TRACK can also be used separatly.
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

// TODO: Determine a good default value.
//
// In the LLVM-TS and SPEC2006, 32 seems to not induce compile time overheads
// (when run with the first 5 abstract attributes). The results also indicate
// that we never reach 32 iterations but always find a fixpoint sooner.
//
// This will become more evolved once we perform two interleaved fixpoint
// iterations: bottom-up and top-down.
static cl::opt<unsigned>
    MaxFixpointIterations("attributor-max-iterations", cl::Hidden,
                          cl::desc("Maximal number of fixpoint iterations."),
                          cl::init(32));
static cl::opt<bool> VerifyMaxFixpointIterations(
    "attributor-max-iterations-verify", cl::Hidden,
    cl::desc("Verify that max-iterations is a tight bound for a fixpoint"),
    cl::init(false));

static cl::opt<bool> DisableAttributor(
    "attributor-disable", cl::Hidden,
    cl::desc("Disable the attributor inter-procedural deduction pass."),
    cl::init(true));

static cl::opt<bool> ManifestInternal(
    "attributor-manifest-internal", cl::Hidden,
    cl::desc("Manifest Attributor internal string attributes."),
    cl::init(false));

static cl::opt<bool> VerifyAttributor(
    "attributor-verify", cl::Hidden,
    cl::desc("Verify the Attributor deduction and "
             "manifestation of attributes -- may issue false-positive errors"),
    cl::init(false));

static cl::opt<unsigned> DepRecInterval(
    "attributor-dependence-recompute-interval", cl::Hidden,
    cl::desc("Number of iterations until dependences are recomputed."),
    cl::init(4));

static cl::opt<bool> EnableHeapToStack("enable-heap-to-stack-conversion",
                                       cl::init(true), cl::Hidden);

static cl::opt<int> MaxHeapToStackSize("max-heap-to-stack-size",
                                       cl::init(128), cl::Hidden);

/// Logic operators for the change status enum class.
///
///{
ChangeStatus llvm::operator|(ChangeStatus l, ChangeStatus r) {
  return l == ChangeStatus::CHANGED ? l : r;
}
ChangeStatus llvm::operator&(ChangeStatus l, ChangeStatus r) {
  return l == ChangeStatus::UNCHANGED ? l : r;
}
///}

/// Recursively visit all values that might become \p IRP at some point. This
/// will be done by looking through cast instructions, selects, phis, and calls
/// with the "returned" attribute. Once we cannot look through the value any
/// further, the callback \p VisitValueCB is invoked and passed the current
/// value, the \p State, and a flag to indicate if we stripped anything. To
/// limit how much effort is invested, we will never visit more values than
/// specified by \p MaxValues.
template <typename AAType, typename StateTy>
static bool genericValueTraversal(
    Attributor &A, IRPosition IRP, const AAType &QueryingAA, StateTy &State,
    const function_ref<bool(Value &, StateTy &, bool)> &VisitValueCB,
    int MaxValues = 8) {

  const AAIsDead *LivenessAA = nullptr;
  if (IRP.getAnchorScope())
    LivenessAA = &A.getAAFor<AAIsDead>(
        QueryingAA, IRPosition::function(*IRP.getAnchorScope()),
        /* TrackDependence */ false);
  bool AnyDead = false;

  // TODO: Use Positions here to allow context sensitivity in VisitValueCB
  SmallPtrSet<Value *, 16> Visited;
  SmallVector<Value *, 16> Worklist;
  Worklist.push_back(&IRP.getAssociatedValue());

  int Iteration = 0;
  do {
    Value *V = Worklist.pop_back_val();

    // Check if we should process the current value. To prevent endless
    // recursion keep a record of the values we followed!
    if (!Visited.insert(V).second)
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
      CallSite CS(V);
      if (CS && CS.getCalledFunction()) {
        for (Argument &Arg : CS.getCalledFunction()->args())
          if (Arg.hasReturnedAttr()) {
            NewV = CS.getArgOperand(Arg.getArgNo());
            break;
          }
      }
    }
    if (NewV && NewV != V) {
      Worklist.push_back(NewV);
      continue;
    }

    // Look through select instructions, visit both potential values.
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    // Look through phi nodes, visit all live operands.
    if (auto *PHI = dyn_cast<PHINode>(V)) {
      assert(LivenessAA &&
             "Expected liveness in the presence of instructions!");
      for (unsigned u = 0, e = PHI->getNumIncomingValues(); u < e; u++) {
        const BasicBlock *IncomingBB = PHI->getIncomingBlock(u);
        if (LivenessAA->isAssumedDead(IncomingBB->getTerminator())) {
          AnyDead = true;
          continue;
        }
        Worklist.push_back(PHI->getIncomingValue(u));
      }
      continue;
    }

    // Once a leaf is reached we inform the user through the callback.
    if (!VisitValueCB(*V, State, Iteration > 1))
      return false;
  } while (!Worklist.empty());

  // If we actually used liveness information so we have to record a dependence.
  if (AnyDead)
    A.recordDependence(*LivenessAA, QueryingAA);

  // All values have been visited.
  return true;
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
                             AttributeList &Attrs, int AttrIdx) {

  if (Attr.isEnumAttribute()) {
    Attribute::AttrKind Kind = Attr.getKindAsEnum();
    if (Attrs.hasAttribute(AttrIdx, Kind))
      if (isEqualOrWorse(Attr, Attrs.getAttribute(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.addAttribute(Ctx, AttrIdx, Attr);
    return true;
  }
  if (Attr.isStringAttribute()) {
    StringRef Kind = Attr.getKindAsString();
    if (Attrs.hasAttribute(AttrIdx, Kind))
      if (isEqualOrWorse(Attr, Attrs.getAttribute(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.addAttribute(Ctx, AttrIdx, Attr);
    return true;
  }
  if (Attr.isIntAttribute()) {
    Attribute::AttrKind Kind = Attr.getKindAsEnum();
    if (Attrs.hasAttribute(AttrIdx, Kind))
      if (isEqualOrWorse(Attr, Attrs.getAttribute(AttrIdx, Kind)))
        return false;
    Attrs = Attrs.removeAttribute(Ctx, AttrIdx, Kind);
    Attrs = Attrs.addAttribute(Ctx, AttrIdx, Attr);
    return true;
  }

  llvm_unreachable("Expected enum or string attribute!");
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
IRAttributeManifest::manifestAttrs(Attributor &A, IRPosition &IRP,
                                   const ArrayRef<Attribute> &DeducedAttrs) {
  Function *ScopeFn = IRP.getAssociatedFunction();
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
    Attrs = ImmutableCallSite(&IRP.getAnchorValue()).getAttributes();
    break;
  }

  ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
  LLVMContext &Ctx = IRP.getAnchorValue().getContext();
  for (const Attribute &Attr : DeducedAttrs) {
    if (!addIfNotExistent(Ctx, Attr, Attrs, IRP.getAttrIdx()))
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
    CallSite(&IRP.getAnchorValue()).setAttributes(Attrs);
    break;
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
    break;
  }

  return HasChanged;
}

const IRPosition IRPosition::EmptyKey(255);
const IRPosition IRPosition::TombstoneKey(256);

SubsumingPositionIterator::SubsumingPositionIterator(const IRPosition &IRP) {
  IRPositions.emplace_back(IRP);

  ImmutableCallSite ICS(&IRP.getAnchorValue());
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_FUNCTION:
    return;
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_RETURNED:
    IRPositions.emplace_back(
        IRPosition::function(*IRP.getAssociatedFunction()));
    return;
  case IRPosition::IRP_CALL_SITE:
    assert(ICS && "Expected call site!");
    // TODO: We need to look at the operand bundles similar to the redirection
    //       in CallBase.
    if (!ICS.hasOperandBundles())
      if (const Function *Callee = ICS.getCalledFunction())
        IRPositions.emplace_back(IRPosition::function(*Callee));
    return;
  case IRPosition::IRP_CALL_SITE_RETURNED:
    assert(ICS && "Expected call site!");
    // TODO: We need to look at the operand bundles similar to the redirection
    //       in CallBase.
    if (!ICS.hasOperandBundles()) {
      if (const Function *Callee = ICS.getCalledFunction()) {
        IRPositions.emplace_back(IRPosition::returned(*Callee));
        IRPositions.emplace_back(IRPosition::function(*Callee));
      }
    }
    IRPositions.emplace_back(
        IRPosition::callsite_function(cast<CallBase>(*ICS.getInstruction())));
    return;
  case IRPosition::IRP_CALL_SITE_ARGUMENT: {
    int ArgNo = IRP.getArgNo();
    assert(ICS && ArgNo >= 0 && "Expected call site!");
    // TODO: We need to look at the operand bundles similar to the redirection
    //       in CallBase.
    if (!ICS.hasOperandBundles()) {
      const Function *Callee = ICS.getCalledFunction();
      if (Callee && Callee->arg_size() > unsigned(ArgNo))
        IRPositions.emplace_back(IRPosition::argument(*Callee->getArg(ArgNo)));
      if (Callee)
        IRPositions.emplace_back(IRPosition::function(*Callee));
    }
    IRPositions.emplace_back(IRPosition::value(IRP.getAssociatedValue()));
    return;
  }
  }
}

bool IRPosition::hasAttr(ArrayRef<Attribute::AttrKind> AKs) const {
  for (const IRPosition &EquivIRP : SubsumingPositionIterator(*this))
    for (Attribute::AttrKind AK : AKs)
      if (EquivIRP.getAttr(AK).getKindAsEnum() == AK)
        return true;
  return false;
}

void IRPosition::getAttrs(ArrayRef<Attribute::AttrKind> AKs,
                          SmallVectorImpl<Attribute> &Attrs) const {
  for (const IRPosition &EquivIRP : SubsumingPositionIterator(*this))
    for (Attribute::AttrKind AK : AKs) {
      const Attribute &Attr = EquivIRP.getAttr(AK);
      if (Attr.getKindAsEnum() == AK)
        Attrs.push_back(Attr);
    }
}

void IRPosition::verify() {
  switch (KindOrArgNo) {
  default:
    assert(KindOrArgNo >= 0 && "Expected argument or call site argument!");
    assert((isa<CallBase>(AnchorVal) || isa<Argument>(AnchorVal)) &&
           "Expected call base or argument for positive attribute index!");
    if (isa<Argument>(AnchorVal)) {
      assert(cast<Argument>(AnchorVal)->getArgNo() == unsigned(getArgNo()) &&
             "Argument number mismatch!");
      assert(cast<Argument>(AnchorVal) == &getAssociatedValue() &&
             "Associated value mismatch!");
    } else {
      assert(cast<CallBase>(*AnchorVal).arg_size() > unsigned(getArgNo()) &&
             "Call site argument number mismatch!");
      assert(cast<CallBase>(*AnchorVal).getArgOperand(getArgNo()) ==
                 &getAssociatedValue() &&
             "Associated value mismatch!");
    }
    break;
  case IRP_INVALID:
    assert(!AnchorVal && "Expected no value for an invalid position!");
    break;
  case IRP_FLOAT:
    assert((!isa<CallBase>(&getAssociatedValue()) &&
            !isa<Argument>(&getAssociatedValue())) &&
           "Expected specialized kind for call base and argument values!");
    break;
  case IRP_RETURNED:
    assert(isa<Function>(AnchorVal) &&
           "Expected function for a 'returned' position!");
    assert(AnchorVal == &getAssociatedValue() && "Associated value mismatch!");
    break;
  case IRP_CALL_SITE_RETURNED:
    assert((isa<CallBase>(AnchorVal)) &&
           "Expected call base for 'call site returned' position!");
    assert(AnchorVal == &getAssociatedValue() && "Associated value mismatch!");
    break;
  case IRP_CALL_SITE:
    assert((isa<CallBase>(AnchorVal)) &&
           "Expected call base for 'call site function' position!");
    assert(AnchorVal == &getAssociatedValue() && "Associated value mismatch!");
    break;
  case IRP_FUNCTION:
    assert(isa<Function>(AnchorVal) &&
           "Expected function for a 'function' position!");
    assert(AnchorVal == &getAssociatedValue() && "Associated value mismatch!");
    break;
  }
}

namespace {
/// Helper functions to clamp a state \p S of type \p StateType with the
/// information in \p R and indicate/return if \p S did change (as-in update is
/// required to be run again).
///
///{
template <typename StateType>
ChangeStatus clampStateAndIndicateChange(StateType &S, const StateType &R);

template <>
ChangeStatus clampStateAndIndicateChange<IntegerState>(IntegerState &S,
                                                       const IntegerState &R) {
  auto Assumed = S.getAssumed();
  S ^= R;
  return Assumed == S.getAssumed() ? ChangeStatus::UNCHANGED
                                   : ChangeStatus::CHANGED;
}

template <>
ChangeStatus clampStateAndIndicateChange<BooleanState>(BooleanState &S,
                                                       const BooleanState &R) {
  return clampStateAndIndicateChange<IntegerState>(S, R);
}
///}

/// Clamp the information known for all returned values of a function
/// (identified by \p QueryingAA) into \p S.
template <typename AAType, typename StateType = typename AAType::StateType>
static void clampReturnedValueStates(Attributor &A, const AAType &QueryingAA,
                                     StateType &S) {
  LLVM_DEBUG(dbgs() << "[Attributor] Clamp return value states for "
                    << static_cast<const AbstractAttribute &>(QueryingAA)
                    << " into " << S << "\n");

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
    const IRPosition &RVPos = IRPosition::value(RV);
    const AAType &AA = A.getAAFor<AAType>(QueryingAA, RVPos);
    LLVM_DEBUG(dbgs() << "[Attributor] RV: " << RV << " AA: " << AA.getAsStr()
                      << " @ " << RVPos << "\n");
    const StateType &AAS = static_cast<const StateType &>(AA.getState());
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
template <typename AAType, typename Base,
          typename StateType = typename AAType::StateType>
struct AAReturnedFromReturnedValues : public Base {
  AAReturnedFromReturnedValues(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    StateType S;
    clampReturnedValueStates<AAType, StateType>(A, *this, S);
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
                    << static_cast<const AbstractAttribute &>(QueryingAA)
                    << " into " << S << "\n");

  assert(QueryingAA.getIRPosition().getPositionKind() ==
             IRPosition::IRP_ARGUMENT &&
         "Can only clamp call site argument states for an argument position!");

  // Use an optional state as there might not be any return values and we want
  // to join (IntegerState::operator&) the state of all there are.
  Optional<StateType> T;

  // The argument number which is also the call site argument number.
  unsigned ArgNo = QueryingAA.getIRPosition().getArgNo();

  auto CallSiteCheck = [&](CallSite CS) {
    const IRPosition &CSArgPos = IRPosition::callsite_argument(CS, ArgNo);
    const AAType &AA = A.getAAFor<AAType>(QueryingAA, CSArgPos);
    LLVM_DEBUG(dbgs() << "[Attributor] CS: " << *CS.getInstruction()
                      << " AA: " << AA.getAsStr() << " @" << CSArgPos << "\n");
    const StateType &AAS = static_cast<const StateType &>(AA.getState());
    if (T.hasValue())
      *T &= AAS;
    else
      T = AAS;
    LLVM_DEBUG(dbgs() << "[Attributor] AA State: " << AAS << " CSA State: " << T
                      << "\n");
    return T->isValidState();
  };

  if (!A.checkForAllCallSites(CallSiteCheck, QueryingAA, true))
    S.indicatePessimisticFixpoint();
  else if (T.hasValue())
    S ^= *T;
}

/// Helper class for generic deduction: call site argument -> argument position.
template <typename AAType, typename Base,
          typename StateType = typename AAType::StateType>
struct AAArgumentFromCallSiteArguments : public Base {
  AAArgumentFromCallSiteArguments(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    StateType S;
    clampCallSiteArgumentStates<AAType, StateType>(A, *this, S);
    // TODO: If we know we visited all incoming values, thus no are assumed
    // dead, we can take the known information from the state T.
    return clampStateAndIndicateChange<StateType>(this->getState(), S);
  }
};

/// Helper class for generic replication: function returned -> cs returned.
template <typename AAType, typename Base>
struct AACallSiteReturnedFromReturned : public Base {
  AACallSiteReturnedFromReturned(const IRPosition &IRP) : Base(IRP) {}

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

    IRPosition FnPos = IRPosition::returned(*AssociatedFunction);
    const AAType &AA = A.getAAFor<AAType>(*this, FnPos);
    return clampStateAndIndicateChange(
        S, static_cast<const typename AAType::StateType &>(AA.getState()));
  }
};

/// -----------------------NoUnwind Function Attribute--------------------------

struct AANoUnwindImpl : AANoUnwind {
  AANoUnwindImpl(const IRPosition &IRP) : AANoUnwind(IRP) {}

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

      if (ImmutableCallSite ICS = ImmutableCallSite(&I)) {
        const auto &NoUnwindAA =
            A.getAAFor<AANoUnwind>(*this, IRPosition::callsite_function(ICS));
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
  AANoUnwindFunction(const IRPosition &IRP) : AANoUnwindImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(nounwind) }
};

/// NoUnwind attribute deduction for a call sites.
struct AANoUnwindCallSite final : AANoUnwindImpl {
  AANoUnwindCallSite(const IRPosition &IRP) : AANoUnwindImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoUnwindImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AANoUnwind>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AANoUnwind::StateType &>(FnAA.getState()));
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
  AAReturnedValuesImpl(const IRPosition &IRP) : AAReturnedValues(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // Reset the state.
    IsFixed = false;
    IsValidState = true;
    ReturnedValues.clear();

    Function *F = getAssociatedFunction();
    if (!F) {
      indicatePessimisticFixpoint();
      return;
    }

    // The map from instruction opcodes to those instructions in the function.
    auto &OpcodeInstMap = A.getInfoCache().getOpcodeInstMapForFunction(*F);

    // Look through all arguments, if one is marked as returned we are done.
    for (Argument &Arg : F->args()) {
      if (Arg.hasReturnedAttr()) {
        auto &ReturnInstSet = ReturnedValues[&Arg];
        for (Instruction *RI : OpcodeInstMap[Instruction::Ret])
          ReturnInstSet.insert(cast<ReturnInst>(RI));

        indicateOptimisticFixpoint();
        return;
      }
    }

    if (!F->hasExactDefinition())
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
      const function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)>
          &Pred) const override;

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
  auto ReplaceCallSiteUsersWith = [](CallBase &CB, Constant &C) {
    if (CB.getNumUses() == 0)
      return ChangeStatus::UNCHANGED;
    CB.replaceAllUsesWith(&C);
    return ChangeStatus::CHANGED;
  };

  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.getValue())) {
    getIRPosition() = IRPosition::argument(*UniqueRVArg);
    Changed = IRAttribute::manifest(A);
  } else if (auto *RVC = dyn_cast<Constant>(UniqueRV.getValue())) {
    // We can replace the returned value with the unique returned constant.
    Value &AnchorValue = getAnchorValue();
    if (Function *F = dyn_cast<Function>(&AnchorValue)) {
      for (const Use &U : F->uses())
        if (CallBase *CB = dyn_cast<CallBase>(U.getUser()))
          if (CB->isCallee(&U)) {
            Constant *RVCCast =
                ConstantExpr::getTruncOrBitCast(RVC, CB->getType());
            Changed = ReplaceCallSiteUsersWith(*CB, *RVCCast) | Changed;
          }
    } else {
      assert(isa<CallBase>(AnchorValue) &&
             "Expcected a function or call base anchor!");
      Constant *RVCCast =
          ConstantExpr::getTruncOrBitCast(RVC, AnchorValue.getType());
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
    const function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)>
        &Pred) const {
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
  auto VisitValueCB = [](Value &Val, RVState &RVS, bool) -> bool {
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
  auto VisitReturnedValue = [&](Value &RV, RVState &RVS) {
    IRPosition RetValPos = IRPosition::value(RV);
    return genericValueTraversal<AAReturnedValues, RVState>(A, RetValPos, *this,
                                                            RVS, VisitValueCB);
  };

  // Callback for all "return intructions" live in the associated function.
  auto CheckReturnInst = [this, &VisitReturnedValue, &Changed](Instruction &I) {
    ReturnInst &Ret = cast<ReturnInst>(I);
    RVState RVS({ReturnedValues, Changed, {}});
    RVS.RetInsts.insert(&Ret);
    return VisitReturnedValue(*Ret.getReturnValue(), RVS);
  };

  // Start by discovering returned values from all live returned instructions in
  // the associated function.
  if (!A.checkForAllInstructions(CheckReturnInst, *this, {Instruction::Ret}))
    return indicatePessimisticFixpoint();

  // Once returned values "directly" present in the code are handled we try to
  // resolve returned calls.
  decltype(ReturnedValues) NewRVsMap;
  for (auto &It : ReturnedValues) {
    LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned value: " << *It.first
                      << " by #" << It.second.size() << " RIs\n");
    CallBase *CB = dyn_cast<CallBase>(It.first);
    if (!CB || UnresolvedCalls.count(CB))
      continue;

    if (!CB->getCalledFunction()) {
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Unresolved call: " << *CB
                        << "\n");
      UnresolvedCalls.insert(CB);
      continue;
    }

    // TODO: use the function scope once we have call site AAReturnedValues.
    const auto &RetValAA = A.getAAFor<AAReturnedValues>(
        *this, IRPosition::function(*CB->getCalledFunction()));
    LLVM_DEBUG(dbgs() << "[AAReturnedValues] Found another AAReturnedValues: "
                      << static_cast<const AbstractAttribute &>(RetValAA)
                      << "\n");

    // Skip dead ends, thus if we do not know anything about the returned
    // call we mark it as unresolved and it will stay that way.
    if (!RetValAA.getState().isValidState()) {
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Unresolved call: " << *CB
                        << "\n");
      UnresolvedCalls.insert(CB);
      continue;
    }

    // Do not try to learn partial information. If the callee has unresolved
    // return values we will treat the call as unresolved/opaque.
    auto &RetValAAUnresolvedCalls = RetValAA.getUnresolvedCalls();
    if (!RetValAAUnresolvedCalls.empty()) {
      UnresolvedCalls.insert(CB);
      continue;
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
      continue;

    // Now track transitively returned values.
    unsigned &NumRetAA = NumReturnedValuesPerKnownAA[CB];
    if (NumRetAA == RetValAA.getNumReturnValues()) {
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Skip call as it has not "
                           "changed since it was seen last\n");
      continue;
    }
    NumRetAA = RetValAA.getNumReturnValues();

    for (auto &RetValAAIt : RetValAA.returned_values()) {
      Value *RetVal = RetValAAIt.first;
      if (Argument *Arg = dyn_cast<Argument>(RetVal)) {
        // Arguments are mapped to call site operands and we begin the traversal
        // again.
        bool Unused = false;
        RVState RVS({NewRVsMap, Unused, RetValAAIt.second});
        VisitReturnedValue(*CB->getArgOperand(Arg->getArgNo()), RVS);
        continue;
      } else if (isa<CallBase>(RetVal)) {
        // Call sites are resolved by the callee attribute over time, no need to
        // do anything for us.
        continue;
      } else if (isa<Constant>(RetVal)) {
        // Constants are valid everywhere, we can simply take them.
        NewRVsMap[RetVal].insert(It.second.begin(), It.second.end());
        continue;
      }
    }
  }

  // To avoid modifications to the ReturnedValues map while we iterate over it
  // we kept record of potential new entries in a copy map, NewRVsMap.
  for (auto &It : NewRVsMap) {
    assert(!It.second.empty() && "Entry does not add anything.");
    auto &ReturnInsts = ReturnedValues[It.first];
    for (ReturnInst *RI : It.second)
      if (ReturnInsts.insert(RI)) {
        LLVM_DEBUG(dbgs() << "[AAReturnedValues] Add new returned value "
                          << *It.first << " => " << *RI << "\n");
        Changed = true;
      }
  }

  Changed |= (NumUnresolvedCalls != UnresolvedCalls.size());
  return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
}

struct AAReturnedValuesFunction final : public AAReturnedValuesImpl {
  AAReturnedValuesFunction(const IRPosition &IRP) : AAReturnedValuesImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(returned) }
};

/// Returned values information for a call sites.
struct AAReturnedValuesCallSite final : AAReturnedValuesImpl {
  AAReturnedValuesCallSite(const IRPosition &IRP) : AAReturnedValuesImpl(IRP) {}

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
  AANoSyncImpl(const IRPosition &IRP) : AANoSync(IRP) {}

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
  assert(!ImmutableCallSite(I) && !isa<CallBase>(I) &&
         "Calls should not be checked here");

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
    /// FIXME: We should ipmrove the handling of intrinsics.

    if (isa<IntrinsicInst>(&I) && isNoSyncIntrinsic(&I))
      return true;

    if (ImmutableCallSite ICS = ImmutableCallSite(&I)) {
      if (ICS.hasFnAttr(Attribute::NoSync))
        return true;

      const auto &NoSyncAA =
          A.getAAFor<AANoSync>(*this, IRPosition::callsite_function(ICS));
      if (NoSyncAA.isAssumedNoSync())
        return true;
      return false;
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
    return !ImmutableCallSite(&I).isConvergent();
  };

  if (!A.checkForAllReadWriteInstructions(CheckRWInstForNoSync, *this) ||
      !A.checkForAllCallLikeInstructions(CheckForNoSync, *this))
    return indicatePessimisticFixpoint();

  return ChangeStatus::UNCHANGED;
}

struct AANoSyncFunction final : public AANoSyncImpl {
  AANoSyncFunction(const IRPosition &IRP) : AANoSyncImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(nosync) }
};

/// NoSync attribute deduction for a call sites.
struct AANoSyncCallSite final : AANoSyncImpl {
  AANoSyncCallSite(const IRPosition &IRP) : AANoSyncImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoSyncImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AANoSync>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const AANoSync::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(nosync); }
};

/// ------------------------ No-Free Attributes ----------------------------

struct AANoFreeImpl : public AANoFree {
  AANoFreeImpl(const IRPosition &IRP) : AANoFree(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto CheckForNoFree = [&](Instruction &I) {
      ImmutableCallSite ICS(&I);
      if (ICS.hasFnAttr(Attribute::NoFree))
        return true;

      const auto &NoFreeAA =
          A.getAAFor<AANoFree>(*this, IRPosition::callsite_function(ICS));
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
  AANoFreeFunction(const IRPosition &IRP) : AANoFreeImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(nofree) }
};

/// NoFree attribute deduction for a call sites.
struct AANoFreeCallSite final : AANoFreeImpl {
  AANoFreeCallSite(const IRPosition &IRP) : AANoFreeImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoFreeImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AANoFree>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const AANoFree::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(nofree); }
};

/// ------------------------ NonNull Argument Attribute ------------------------
struct AANonNullImpl : AANonNull {
  AANonNullImpl(const IRPosition &IRP) : AANonNull(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({Attribute::NonNull, Attribute::Dereferenceable}))
      indicateOptimisticFixpoint();
    else
      AANonNull::initialize(A);
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nonnull" : "may-null";
  }
};

/// NonNull attribute for a floating value.
struct AANonNullFloating : AANonNullImpl {
  AANonNullFloating(const IRPosition &IRP) : AANonNullImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANonNullImpl::initialize(A);

    if (isAtFixpoint())
      return;

    const IRPosition &IRP = getIRPosition();
    const Value &V = IRP.getAssociatedValue();
    const DataLayout &DL = A.getDataLayout();

    // TODO: This context sensitive query should be removed once we can do
    // context sensitive queries in the genericValueTraversal below.
    if (isKnownNonZero(&V, DL, 0, /* TODO: AC */ nullptr, IRP.getCtxI(),
                       /* TODO: DT */ nullptr))
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    auto VisitValueCB = [&](Value &V, AAAlign::StateType &T,
                            bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AANonNull>(*this, IRPosition::value(V));
      if (!Stripped && this == &AA) {
        if (!isKnownNonZero(&V, DL, 0, /* TODO: AC */ nullptr,
                            /* TODO: CtxI */ nullptr,
                            /* TODO: DT */ nullptr))
          T.indicatePessimisticFixpoint();
      } else {
        // Use abstract attribute information.
        const AANonNull::StateType &NS =
            static_cast<const AANonNull::StateType &>(AA.getState());
        T ^= NS;
      }
      return T.isValidState();
    };

    StateType T;
    if (!genericValueTraversal<AANonNull, StateType>(A, getIRPosition(), *this,
                                                     T, VisitValueCB))
      return indicatePessimisticFixpoint();

    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(nonnull) }
};

/// NonNull attribute for function return value.
struct AANonNullReturned final
    : AAReturnedFromReturnedValues<AANonNull, AANonNullImpl> {
  AANonNullReturned(const IRPosition &IRP)
      : AAReturnedFromReturnedValues<AANonNull, AANonNullImpl>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(nonnull) }
};

/// NonNull attribute for function argument.
struct AANonNullArgument final
    : AAArgumentFromCallSiteArguments<AANonNull, AANonNullImpl> {
  AANonNullArgument(const IRPosition &IRP)
      : AAArgumentFromCallSiteArguments<AANonNull, AANonNullImpl>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nonnull) }
};

struct AANonNullCallSiteArgument final : AANonNullFloating {
  AANonNullCallSiteArgument(const IRPosition &IRP) : AANonNullFloating(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(nonnull) }
};

/// NonNull attribute for a call site return position.
struct AANonNullCallSiteReturned final
    : AACallSiteReturnedFromReturned<AANonNull, AANonNullImpl> {
  AANonNullCallSiteReturned(const IRPosition &IRP)
      : AACallSiteReturnedFromReturned<AANonNull, AANonNullImpl>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(nonnull) }
};

/// ------------------------ No-Recurse Attributes ----------------------------

struct AANoRecurseImpl : public AANoRecurse {
  AANoRecurseImpl(const IRPosition &IRP) : AANoRecurse(IRP) {}

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "norecurse" : "may-recurse";
  }
};

struct AANoRecurseFunction final : AANoRecurseImpl {
  AANoRecurseFunction(const IRPosition &IRP) : AANoRecurseImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoRecurseImpl::initialize(A);
    if (const Function *F = getAnchorScope())
      if (A.getInfoCache().getSccSize(*F) == 1)
        return;
    indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {

    auto CheckForNoRecurse = [&](Instruction &I) {
      ImmutableCallSite ICS(&I);
      if (ICS.hasFnAttr(Attribute::NoRecurse))
        return true;

      const auto &NoRecurseAA =
          A.getAAFor<AANoRecurse>(*this, IRPosition::callsite_function(ICS));
      if (!NoRecurseAA.isAssumedNoRecurse())
        return false;

      // Recursion to the same function
      if (ICS.getCalledFunction() == getAnchorScope())
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
  AANoRecurseCallSite(const IRPosition &IRP) : AANoRecurseImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoRecurseImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AANoRecurse>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AANoRecurse::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(norecurse); }
};

/// ------------------------ Will-Return Attributes ----------------------------

// Helper function that checks whether a function has any cycle.
// TODO: Replace with more efficent code
static bool containsCycle(Function &F) {
  SmallPtrSet<BasicBlock *, 32> Visited;

  // Traverse BB by dfs and check whether successor is already visited.
  for (BasicBlock *BB : depth_first(&F)) {
    Visited.insert(BB);
    for (auto *SuccBB : successors(BB)) {
      if (Visited.count(SuccBB))
        return true;
    }
  }
  return false;
}

// Helper function that checks the function have a loop which might become an
// endless loop
// FIXME: Any cycle is regarded as endless loop for now.
//        We have to allow some patterns.
static bool containsPossiblyEndlessLoop(Function *F) {
  return !F || !F->hasExactDefinition() || containsCycle(*F);
}

struct AAWillReturnImpl : public AAWillReturn {
  AAWillReturnImpl(const IRPosition &IRP) : AAWillReturn(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAWillReturn::initialize(A);

    Function *F = getAssociatedFunction();
    if (containsPossiblyEndlessLoop(F))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto CheckForWillReturn = [&](Instruction &I) {
      IRPosition IPos = IRPosition::callsite_function(ImmutableCallSite(&I));
      const auto &WillReturnAA = A.getAAFor<AAWillReturn>(*this, IPos);
      if (WillReturnAA.isKnownWillReturn())
        return true;
      if (!WillReturnAA.isAssumedWillReturn())
        return false;
      const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(*this, IPos);
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
  AAWillReturnFunction(const IRPosition &IRP) : AAWillReturnImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(willreturn) }
};

/// WillReturn attribute deduction for a call sites.
struct AAWillReturnCallSite final : AAWillReturnImpl {
  AAWillReturnCallSite(const IRPosition &IRP) : AAWillReturnImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAWillReturnImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AAWillReturn>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AAWillReturn::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(willreturn); }
};

/// ------------------------ NoAlias Argument Attribute ------------------------

struct AANoAliasImpl : AANoAlias {
  AANoAliasImpl(const IRPosition &IRP) : AANoAlias(IRP) {}

  const std::string getAsStr() const override {
    return getAssumed() ? "noalias" : "may-alias";
  }
};

/// NoAlias attribute for a floating value.
struct AANoAliasFloating final : AANoAliasImpl {
  AANoAliasFloating(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoAliasImpl::initialize(A);
    if (isa<AllocaInst>(getAnchorValue()))
      indicateOptimisticFixpoint();
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
  AANoAliasArgument(const IRPosition &IRP)
      : AAArgumentFromCallSiteArguments<AANoAlias, AANoAliasImpl>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(noalias) }
};

struct AANoAliasCallSiteArgument final : AANoAliasImpl {
  AANoAliasCallSiteArgument(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // See callsite argument attribute and callee argument attribute.
    ImmutableCallSite ICS(&getAnchorValue());
    if (ICS.paramHasAttr(getArgNo(), Attribute::NoAlias))
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // We can deduce "noalias" if the following conditions hold.
    // (i)   Associated value is assumed to be noalias in the definition.
    // (ii)  Associated value is assumed to be no-capture in all the uses
    //       possibly executed before this callsite.
    // (iii) There is no other pointer argument which could alias with the
    //       value.

    const Value &V = getAssociatedValue();
    const IRPosition IRP = IRPosition::value(V);

    // (i) Check whether noalias holds in the definition.

    auto &NoAliasAA = A.getAAFor<AANoAlias>(*this, IRP);

    if (!NoAliasAA.isAssumedNoAlias())
      return indicatePessimisticFixpoint();

    LLVM_DEBUG(dbgs() << "[Attributor][AANoAliasCSArg] " << V
                      << " is assumed NoAlias in the definition\n");

    // (ii) Check whether the value is captured in the scope using AANoCapture.
    //      FIXME: This is conservative though, it is better to look at CFG and
    //             check only uses possibly executed before this callsite.

    auto &NoCaptureAA = A.getAAFor<AANoCapture>(*this, IRP);
    if (!NoCaptureAA.isAssumedNoCaptureMaybeReturned())
      return indicatePessimisticFixpoint();

    // (iii) Check there is no other pointer argument which could alias with the
    // value.
    ImmutableCallSite ICS(&getAnchorValue());
    for (unsigned i = 0; i < ICS.getNumArgOperands(); i++) {
      if (getArgNo() == (int)i)
        continue;
      const Value *ArgOp = ICS.getArgOperand(i);
      if (!ArgOp->getType()->isPointerTy())
        continue;

      if (const Function *F = getAnchorScope()) {
        if (AAResults *AAR = A.getInfoCache().getAAResultsForFunction(*F)) {
          LLVM_DEBUG(dbgs()
                     << "[Attributor][NoAliasCSArg] Check alias between "
                        "callsite arguments "
                     << AAR->isNoAlias(&getAssociatedValue(), ArgOp) << " "
                     << getAssociatedValue() << " " << *ArgOp << "\n");

          if (AAR->isNoAlias(&getAssociatedValue(), ArgOp))
            continue;
        }
      }
      return indicatePessimisticFixpoint();
    }

    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(noalias) }
};

/// NoAlias attribute for function return value.
struct AANoAliasReturned final : AANoAliasImpl {
  AANoAliasReturned(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  virtual ChangeStatus updateImpl(Attributor &A) override {

    auto CheckReturnValue = [&](Value &RV) -> bool {
      if (Constant *C = dyn_cast<Constant>(&RV))
        if (C->isNullValue() || isa<UndefValue>(C))
          return true;

      /// For now, we can only deduce noalias if we have call sites.
      /// FIXME: add more support.
      ImmutableCallSite ICS(&RV);
      if (!ICS)
        return false;

      const IRPosition &RVPos = IRPosition::value(RV);
      const auto &NoAliasAA = A.getAAFor<AANoAlias>(*this, RVPos);
      if (!NoAliasAA.isAssumedNoAlias())
        return false;

      const auto &NoCaptureAA = A.getAAFor<AANoCapture>(*this, RVPos);
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
  AANoAliasCallSiteReturned(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoAliasImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AANoAlias>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const AANoAlias::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(noalias); }
};

/// -------------------AAIsDead Function Attribute-----------------------

struct AAIsDeadImpl : public AAIsDead {
  AAIsDeadImpl(const IRPosition &IRP) : AAIsDead(IRP) {}

  void initialize(Attributor &A) override {
    const Function *F = getAssociatedFunction();
    if (F && !F->isDeclaration())
      exploreFromEntry(A, F);
  }

  void exploreFromEntry(Attributor &A, const Function *F) {
    ToBeExploredPaths.insert(&(F->getEntryBlock().front()));

    for (size_t i = 0; i < ToBeExploredPaths.size(); ++i)
      if (const Instruction *NextNoReturnI =
              findNextNoReturn(A, ToBeExploredPaths[i]))
        NoReturnCalls.insert(NextNoReturnI);

    // Mark the block live after we looked for no-return instructions.
    assumeLive(A, F->getEntryBlock());
  }

  /// Find the next assumed noreturn instruction in the block of \p I starting
  /// from, thus including, \p I.
  ///
  /// The caller is responsible to monitor the ToBeExploredPaths set as new
  /// instructions discovered in other basic block will be placed in there.
  ///
  /// \returns The next assumed noreturn instructions in the block of \p I
  ///          starting from, thus including, \p I.
  const Instruction *findNextNoReturn(Attributor &A, const Instruction *I);

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return "Live[#BB " + std::to_string(AssumedLiveBlocks.size()) + "/" +
           std::to_string(getAssociatedFunction()->size()) + "][#NRI " +
           std::to_string(NoReturnCalls.size()) + "]";
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    assert(getState().isValidState() &&
           "Attempted to manifest an invalid state!");

    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
    Function &F = *getAssociatedFunction();

    if (AssumedLiveBlocks.empty()) {
      A.deleteAfterManifest(F);
      return ChangeStatus::CHANGED;
    }

    // Flag to determine if we can change an invoke to a call assuming the
    // callee is nounwind. This is not possible if the personality of the
    // function allows to catch asynchronous exceptions.
    bool Invoke2CallAllowed = !mayCatchAsynchronousExceptions(F);

    for (const Instruction *NRC : NoReturnCalls) {
      Instruction *I = const_cast<Instruction *>(NRC);
      BasicBlock *BB = I->getParent();
      Instruction *SplitPos = I->getNextNode();
      // TODO: mark stuff before unreachable instructions as dead.
      if (isa_and_nonnull<UnreachableInst>(SplitPos))
        continue;

      if (auto *II = dyn_cast<InvokeInst>(I)) {
        // If we keep the invoke the split position is at the beginning of the
        // normal desitination block (it invokes a noreturn function after all).
        BasicBlock *NormalDestBB = II->getNormalDest();
        SplitPos = &NormalDestBB->front();

        /// Invoke is replaced with a call and unreachable is placed after it if
        /// the callee is nounwind and noreturn. Otherwise, we keep the invoke
        /// and only place an unreachable in the normal successor.
        if (Invoke2CallAllowed) {
          if (II->getCalledFunction()) {
            const IRPosition &IPos = IRPosition::callsite_function(*II);
            const auto &AANoUnw = A.getAAFor<AANoUnwind>(*this, IPos);
            if (AANoUnw.isAssumedNoUnwind()) {
              LLVM_DEBUG(dbgs()
                         << "[AAIsDead] Replace invoke with call inst\n");
              // We do not need an invoke (II) but instead want a call followed
              // by an unreachable. However, we do not remove II as other
              // abstract attributes might have it cached as part of their
              // results. Given that we modify the CFG anyway, we simply keep II
              // around but in a new dead block. To avoid II being live through
              // a different edge we have to ensure the block we place it in is
              // only reached from the current block of II and then not reached
              // at all when we insert the unreachable.
              SplitBlockPredecessors(NormalDestBB, {BB}, ".i2c");
              CallInst *CI = createCallMatchingInvoke(II);
              CI->insertBefore(II);
              CI->takeName(II);
              II->replaceAllUsesWith(CI);
              SplitPos = CI->getNextNode();
            }
          }
        }

        if (SplitPos == &NormalDestBB->front()) {
          // If this is an invoke of a noreturn function the edge to the normal
          // destination block is dead but not necessarily the block itself.
          // TODO: We need to move to an edge based system during deduction and
          //       also manifest.
          assert(!NormalDestBB->isLandingPad() &&
                 "Expected the normal destination not to be a landingpad!");
          BasicBlock *SplitBB =
              SplitBlockPredecessors(NormalDestBB, {BB}, ".dead");
          // The split block is live even if it contains only an unreachable
          // instruction at the end.
          assumeLive(A, *SplitBB);
          SplitPos = SplitBB->getTerminator();
        }
      }

      BB = SplitPos->getParent();
      SplitBlock(BB, SplitPos);
      changeToUnreachable(BB->getTerminator(), /* UseLLVMTrap */ false);
      HasChanged = ChangeStatus::CHANGED;
    }

    for (BasicBlock &BB : F)
      if (!AssumedLiveBlocks.count(&BB))
        A.deleteAfterManifest(BB);

    return HasChanged;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AAIsDead::isAssumedDead(BasicBlock *).
  bool isAssumedDead(const BasicBlock *BB) const override {
    assert(BB->getParent() == getAssociatedFunction() &&
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
    assert(I->getParent()->getParent() == getAssociatedFunction() &&
           "Instruction must be in the same anchor scope function.");

    if (!getAssumed())
      return false;

    // If it is not in AssumedLiveBlocks then it for sure dead.
    // Otherwise, it can still be after noreturn call in a live block.
    if (!AssumedLiveBlocks.count(I->getParent()))
      return true;

    // If it is not after a noreturn call, than it is live.
    return isAfterNoReturn(I);
  }

  /// See AAIsDead::isKnownDead(Instruction *I).
  bool isKnownDead(const Instruction *I) const override {
    return getKnown() && isAssumedDead(I);
  }

  /// Check if instruction is after noreturn call, in other words, assumed dead.
  bool isAfterNoReturn(const Instruction *I) const;

  /// Determine if \p F might catch asynchronous exceptions.
  static bool mayCatchAsynchronousExceptions(const Function &F) {
    return F.hasPersonalityFn() && !canSimplifyInvokeNoUnwind(&F);
  }

  /// Assume \p BB is (partially) live now and indicate to the Attributor \p A
  /// that internal function called from \p BB should now be looked at.
  void assumeLive(Attributor &A, const BasicBlock &BB) {
    if (!AssumedLiveBlocks.insert(&BB).second)
      return;

    // We assume that all of BB is (probably) live now and if there are calls to
    // internal functions we will assume that those are now live as well. This
    // is a performance optimization for blocks with calls to a lot of internal
    // functions. It can however cause dead functions to be treated as live.
    for (const Instruction &I : BB)
      if (ImmutableCallSite ICS = ImmutableCallSite(&I))
        if (const Function *F = ICS.getCalledFunction())
          if (F->hasInternalLinkage())
            A.markLiveInternalFunction(*F);
  }

  /// Collection of to be explored paths.
  SmallSetVector<const Instruction *, 8> ToBeExploredPaths;

  /// Collection of all assumed live BasicBlocks.
  DenseSet<const BasicBlock *> AssumedLiveBlocks;

  /// Collection of calls with noreturn attribute, assumed or knwon.
  SmallSetVector<const Instruction *, 4> NoReturnCalls;
};

struct AAIsDeadFunction final : public AAIsDeadImpl {
  AAIsDeadFunction(const IRPosition &IRP) : AAIsDeadImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECL(PartiallyDeadBlocks, Function,
               "Number of basic blocks classified as partially dead");
    BUILD_STAT_NAME(PartiallyDeadBlocks, Function) += NoReturnCalls.size();
  }
};

bool AAIsDeadImpl::isAfterNoReturn(const Instruction *I) const {
  const Instruction *PrevI = I->getPrevNode();
  while (PrevI) {
    if (NoReturnCalls.count(PrevI))
      return true;
    PrevI = PrevI->getPrevNode();
  }
  return false;
}

const Instruction *AAIsDeadImpl::findNextNoReturn(Attributor &A,
                                                  const Instruction *I) {
  const BasicBlock *BB = I->getParent();
  const Function &F = *BB->getParent();

  // Flag to determine if we can change an invoke to a call assuming the callee
  // is nounwind. This is not possible if the personality of the function allows
  // to catch asynchronous exceptions.
  bool Invoke2CallAllowed = !mayCatchAsynchronousExceptions(F);

  // TODO: We should have a function that determines if an "edge" is dead.
  //       Edges could be from an instruction to the next or from a terminator
  //       to the successor. For now, we need to special case the unwind block
  //       of InvokeInst below.

  while (I) {
    ImmutableCallSite ICS(I);

    if (ICS) {
      const IRPosition &IPos = IRPosition::callsite_function(ICS);
      // Regarless of the no-return property of an invoke instruction we only
      // learn that the regular successor is not reachable through this
      // instruction but the unwind block might still be.
      if (auto *Invoke = dyn_cast<InvokeInst>(I)) {
        // Use nounwind to justify the unwind block is dead as well.
        const auto &AANoUnw = A.getAAFor<AANoUnwind>(*this, IPos);
        if (!Invoke2CallAllowed || !AANoUnw.isAssumedNoUnwind()) {
          assumeLive(A, *Invoke->getUnwindDest());
          ToBeExploredPaths.insert(&Invoke->getUnwindDest()->front());
        }
      }

      const auto &NoReturnAA = A.getAAFor<AANoReturn>(*this, IPos);
      if (NoReturnAA.isAssumedNoReturn())
        return I;
    }

    I = I->getNextNode();
  }

  // get new paths (reachable blocks).
  for (const BasicBlock *SuccBB : successors(BB)) {
    assumeLive(A, *SuccBB);
    ToBeExploredPaths.insert(&SuccBB->front());
  }

  // No noreturn instruction found.
  return nullptr;
}

ChangeStatus AAIsDeadImpl::updateImpl(Attributor &A) {
  ChangeStatus Status = ChangeStatus::UNCHANGED;

  // Temporary collection to iterate over existing noreturn instructions. This
  // will alow easier modification of NoReturnCalls collection
  SmallVector<const Instruction *, 8> NoReturnChanged;

  for (const Instruction *I : NoReturnCalls)
    NoReturnChanged.push_back(I);

  for (const Instruction *I : NoReturnChanged) {
    size_t Size = ToBeExploredPaths.size();

    const Instruction *NextNoReturnI = findNextNoReturn(A, I);
    if (NextNoReturnI != I) {
      Status = ChangeStatus::CHANGED;
      NoReturnCalls.remove(I);
      if (NextNoReturnI)
        NoReturnCalls.insert(NextNoReturnI);
    }

    // Explore new paths.
    while (Size != ToBeExploredPaths.size()) {
      Status = ChangeStatus::CHANGED;
      if (const Instruction *NextNoReturnI =
              findNextNoReturn(A, ToBeExploredPaths[Size++]))
        NoReturnCalls.insert(NextNoReturnI);
    }
  }

  LLVM_DEBUG(dbgs() << "[AAIsDead] AssumedLiveBlocks: "
                    << AssumedLiveBlocks.size() << " Total number of blocks: "
                    << getAssociatedFunction()->size() << "\n");

  // If we know everything is live there is no need to query for liveness.
  if (NoReturnCalls.empty() &&
      getAssociatedFunction()->size() == AssumedLiveBlocks.size()) {
    // Indicating a pessimistic fixpoint will cause the state to be "invalid"
    // which will cause the Attributor to not return the AAIsDead on request,
    // which will prevent us from querying isAssumedDead().
    indicatePessimisticFixpoint();
    assert(!isValidState() && "Expected an invalid state!");
    Status = ChangeStatus::CHANGED;
  }

  return Status;
}

/// Liveness information for a call sites.
struct AAIsDeadCallSite final : AAIsDeadImpl {
  AAIsDeadCallSite(const IRPosition &IRP) : AAIsDeadImpl(IRP) {}

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
  ChangeStatus CS0 = clampStateAndIndicateChange<IntegerState>(
      S.DerefBytesState, R.DerefBytesState);
  ChangeStatus CS1 =
      clampStateAndIndicateChange<IntegerState>(S.GlobalState, R.GlobalState);
  return CS0 | CS1;
}

struct AADereferenceableImpl : AADereferenceable {
  AADereferenceableImpl(const IRPosition &IRP) : AADereferenceable(IRP) {}
  using StateType = DerefState;

  void initialize(Attributor &A) override {
    SmallVector<Attribute, 4> Attrs;
    getAttrs({Attribute::Dereferenceable, Attribute::DereferenceableOrNull},
             Attrs);
    for (const Attribute &Attr : Attrs)
      takeKnownDerefBytesMaximum(Attr.getValueAsInt());

    NonNullAA = &A.getAAFor<AANonNull>(*this, getIRPosition());

    const IRPosition &IRP = this->getIRPosition();
    bool IsFnInterface = IRP.isFnInterfaceKind();
    const Function *FnScope = IRP.getAnchorScope();
    if (IsFnInterface && (!FnScope || !FnScope->hasExactDefinition()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::getState()
  /// {
  StateType &getState() override { return *this; }
  const StateType &getState() const override { return *this; }
  /// }

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
  AADereferenceableFloating(const IRPosition &IRP)
      : AADereferenceableImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    auto VisitValueCB = [&](Value &V, DerefState &T, bool Stripped) -> bool {
      unsigned IdxWidth =
          DL.getIndexSizeInBits(V.getType()->getPointerAddressSpace());
      APInt Offset(IdxWidth, 0);
      const Value *Base =
          V.stripAndAccumulateInBoundsConstantOffsets(DL, Offset);

      const auto &AA =
          A.getAAFor<AADereferenceable>(*this, IRPosition::value(*Base));
      int64_t DerefBytes = 0;
      if (!Stripped && this == &AA) {
        // Use IR information if we did not strip anything.
        // TODO: track globally.
        bool CanBeNull;
        DerefBytes = Base->getPointerDereferenceableBytes(DL, CanBeNull);
        T.GlobalState.indicatePessimisticFixpoint();
      } else {
        const DerefState &DS = static_cast<const DerefState &>(AA.getState());
        DerefBytes = DS.DerefBytesState.getAssumed();
        T.GlobalState &= DS.GlobalState;
      }

      // For now we do not try to "increase" dereferenceability due to negative
      // indices as we first have to come up with code to deal with loops and
      // for overflows of the dereferenceable bytes.
      int64_t OffsetSExt = Offset.getSExtValue();
      if (OffsetSExt < 0)
        Offset = 0;

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
            A, getIRPosition(), *this, T, VisitValueCB))
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
    : AAReturnedFromReturnedValues<AADereferenceable, AADereferenceableImpl,
                                   DerefState> {
  AADereferenceableReturned(const IRPosition &IRP)
      : AAReturnedFromReturnedValues<AADereferenceable, AADereferenceableImpl,
                                     DerefState>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute for an argument
struct AADereferenceableArgument final
    : AAArgumentFromCallSiteArguments<AADereferenceable, AADereferenceableImpl,
                                      DerefState> {
  AADereferenceableArgument(const IRPosition &IRP)
      : AAArgumentFromCallSiteArguments<AADereferenceable,
                                        AADereferenceableImpl, DerefState>(
            IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute for a call site argument.
struct AADereferenceableCallSiteArgument final : AADereferenceableFloating {
  AADereferenceableCallSiteArgument(const IRPosition &IRP)
      : AADereferenceableFloating(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(dereferenceable)
  }
};

/// Dereferenceable attribute deduction for a call site return value.
struct AADereferenceableCallSiteReturned final : AADereferenceableImpl {
  AADereferenceableCallSiteReturned(const IRPosition &IRP)
      : AADereferenceableImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AADereferenceableImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AADereferenceable>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const DerefState &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(dereferenceable);
  }
};

// ------------------------ Align Argument Attribute ------------------------

struct AAAlignImpl : AAAlign {
  AAAlignImpl(const IRPosition &IRP) : AAAlign(IRP) {}

  // Max alignemnt value allowed in IR
  static const unsigned MAX_ALIGN = 1U << 29;

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    takeAssumedMinimum(MAX_ALIGN);

    SmallVector<Attribute, 4> Attrs;
    getAttrs({Attribute::Alignment}, Attrs);
    for (const Attribute &Attr : Attrs)
      takeKnownMaximum(Attr.getValueAsInt());

    if (getIRPosition().isFnInterfaceKind() &&
        (!getAssociatedFunction() ||
         !getAssociatedFunction()->hasExactDefinition()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    // Check for users that allow alignment annotations.
    Value &AnchorVal = getIRPosition().getAnchorValue();
    for (const Use &U : AnchorVal.uses()) {
      if (auto *SI = dyn_cast<StoreInst>(U.getUser())) {
        if (SI->getPointerOperand() == &AnchorVal)
          if (SI->getAlignment() < getAssumedAlign()) {
            STATS_DECLTRACK(AAAlign, Store,
                            "Number of times alignemnt added to a store");
            SI->setAlignment(Align(getAssumedAlign()));
            Changed = ChangeStatus::CHANGED;
          }
      } else if (auto *LI = dyn_cast<LoadInst>(U.getUser())) {
        if (LI->getPointerOperand() == &AnchorVal)
          if (LI->getAlignment() < getAssumedAlign()) {
            LI->setAlignment(Align(getAssumedAlign()));
            STATS_DECLTRACK(AAAlign, Load,
                            "Number of times alignemnt added to a load");
            Changed = ChangeStatus::CHANGED;
          }
      }
    }

    return AAAlign::manifest(A) | Changed;
  }

  // TODO: Provide a helper to determine the implied ABI alignment and check in
  //       the existing manifest method and a new one for AAAlignImpl that value
  //       to avoid making the alignment explicit if it did not improve.

  /// See AbstractAttribute::getDeducedAttributes
  virtual void
  getDeducedAttributes(LLVMContext &Ctx,
                       SmallVectorImpl<Attribute> &Attrs) const override {
    if (getAssumedAlign() > 1)
      Attrs.emplace_back(Attribute::getWithAlignment(Ctx, getAssumedAlign()));
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
  AAAlignFloating(const IRPosition &IRP) : AAAlignImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    auto VisitValueCB = [&](Value &V, AAAlign::StateType &T,
                            bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AAAlign>(*this, IRPosition::value(V));
      if (!Stripped && this == &AA) {
        // Use only IR information if we did not strip anything.
        T.takeKnownMaximum(V.getPointerAlignment(DL));
        T.indicatePessimisticFixpoint();
      } else {
        // Use abstract attribute information.
        const AAAlign::StateType &DS =
            static_cast<const AAAlign::StateType &>(AA.getState());
        T ^= DS;
      }
      return T.isValidState();
    };

    StateType T;
    if (!genericValueTraversal<AAAlign, StateType>(A, getIRPosition(), *this, T,
                                                   VisitValueCB))
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
  AAAlignReturned(const IRPosition &IRP)
      : AAReturnedFromReturnedValues<AAAlign, AAAlignImpl>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(aligned) }
};

/// Align attribute for function argument.
struct AAAlignArgument final
    : AAArgumentFromCallSiteArguments<AAAlign, AAAlignImpl> {
  AAAlignArgument(const IRPosition &IRP)
      : AAArgumentFromCallSiteArguments<AAAlign, AAAlignImpl>(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(aligned) }
};

struct AAAlignCallSiteArgument final : AAAlignFloating {
  AAAlignCallSiteArgument(const IRPosition &IRP) : AAAlignFloating(IRP) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    return AAAlignImpl::manifest(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(aligned) }
};

/// Align attribute deduction for a call site return value.
struct AAAlignCallSiteReturned final : AAAlignImpl {
  AAAlignCallSiteReturned(const IRPosition &IRP) : AAAlignImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAAlignImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AAAlign>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const AAAlign::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(align); }
};

/// ------------------ Function No-Return Attribute ----------------------------
struct AANoReturnImpl : public AANoReturn {
  AANoReturnImpl(const IRPosition &IRP) : AANoReturn(IRP) {}

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
  AANoReturnFunction(const IRPosition &IRP) : AANoReturnImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(noreturn) }
};

/// NoReturn attribute deduction for a call sites.
struct AANoReturnCallSite final : AANoReturnImpl {
  AANoReturnCallSite(const IRPosition &IRP) : AANoReturnImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoReturnImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
    auto &FnAA = A.getAAFor<AANoReturn>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AANoReturn::StateType &>(FnAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(noreturn); }
};

/// ----------------------- Variable Capturing ---------------------------------

/// A class to hold the state of for no-capture attributes.
struct AANoCaptureImpl : public AANoCapture {
  AANoCaptureImpl(const IRPosition &IRP) : AANoCapture(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoCapture::initialize(A);

    const IRPosition &IRP = getIRPosition();
    const Function *F =
        getArgNo() >= 0 ? IRP.getAssociatedFunction() : IRP.getAnchorScope();

    // Check what state the associated function can actually capture.
    if (F)
      determineFunctionCaptureCapabilities(*F, *this);
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

    if (getArgNo() >= 0) {
      if (isAssumedNoCapture())
        Attrs.emplace_back(Attribute::get(Ctx, Attribute::NoCapture));
      else if (ManifestInternal)
        Attrs.emplace_back(Attribute::get(Ctx, "no-capture-maybe-returned"));
    }
  }

  /// Set the NOT_CAPTURED_IN_MEM and NOT_CAPTURED_IN_RET bits in \p Known
  /// depending on the ability of the function associated with \p IRP to capture
  /// state in memory and through "returning/throwing", respectively.
  static void determineFunctionCaptureCapabilities(const Function &F,
                                                   IntegerState &State) {
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
                      const AAIsDead &IsDeadAA, IntegerState &State,
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
    const auto &DerefAA =
        A.getAAFor<AADereferenceable>(NoCaptureAA, IRPosition::value(*O));
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
    CallSite CS(UInst);
    if (!CS || !CS.isArgOperand(U))
      return isCapturedIn(/* Memory */ true, /* Integer */ true,
                          /* Return */ true);

    unsigned ArgNo = CS.getArgumentNo(U);
    const IRPosition &CSArgPos = IRPosition::callsite_argument(CS, ArgNo);
    // If we have a abstract no-capture attribute for the argument we can use
    // it to justify a non-capture attribute here. This allows recursion!
    auto &ArgNoCaptureAA = A.getAAFor<AANoCapture>(NoCaptureAA, CSArgPos);
    if (ArgNoCaptureAA.isAssumedNoCapture())
      return isCapturedIn(/* Memory */ false, /* Integer */ false,
                          /* Return */ false);
    if (ArgNoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
      addPotentialCopy(CS);
      return isCapturedIn(/* Memory */ false, /* Integer */ false,
                          /* Return */ false);
    }

    // Lastly, we could not find a reason no-capture can be assumed so we don't.
    return isCapturedIn(/* Memory */ true, /* Integer */ true,
                        /* Return */ true);
  }

  /// Register \p CS as potential copy of the value we are checking.
  void addPotentialCopy(CallSite CS) {
    PotentialCopies.push_back(CS.getInstruction());
  }

  /// See CaptureTracker::shouldExplore(...).
  bool shouldExplore(const Use *U) override {
    // Check liveness.
    return !IsDeadAA.isAssumedDead(cast<Instruction>(U->getUser()));
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
  IntegerState &State;

  /// Set of potential copies of the tracked value.
  SmallVectorImpl<const Value *> &PotentialCopies;

  /// Global counter to limit the number of explored uses.
  unsigned &RemainingUsesToExplore;
};

ChangeStatus AANoCaptureImpl::updateImpl(Attributor &A) {
  const IRPosition &IRP = getIRPosition();
  const Value *V =
      getArgNo() >= 0 ? IRP.getAssociatedArgument() : &IRP.getAssociatedValue();
  if (!V)
    return indicatePessimisticFixpoint();

  const Function *F =
      getArgNo() >= 0 ? IRP.getAssociatedFunction() : IRP.getAnchorScope();
  assert(F && "Expected a function!");
  const auto &IsDeadAA = A.getAAFor<AAIsDead>(*this, IRPosition::function(*F));

  AANoCapture::StateType T;
  // TODO: Once we have memory behavior attributes we should use them here
  // similar to the reasoning in
  // AANoCaptureImpl::determineFunctionCaptureCapabilities(...).

  // TODO: Use the AAReturnedValues to learn if the argument can return or
  // not.

  // Use the CaptureTracker interface and logic with the specialized tracker,
  // defined in AACaptureUseTracker, that can look at in-flight abstract
  // attributes and directly updates the assumed state.
  SmallVector<const Value *, 4> PotentialCopies;
  unsigned RemainingUsesToExplore = DefaultMaxUsesToExplore;
  AACaptureUseTracker Tracker(A, *this, IsDeadAA, T, PotentialCopies,
                              RemainingUsesToExplore);

  // Check all potential copies of the associated value until we can assume
  // none will be captured or we have to assume at least one might be.
  unsigned Idx = 0;
  PotentialCopies.push_back(V);
  while (T.isAssumed(NO_CAPTURE_MAYBE_RETURNED) && Idx < PotentialCopies.size())
    Tracker.valueMayBeCaptured(PotentialCopies[Idx++]);

  AAAlign::StateType &S = getState();
  auto Assumed = S.getAssumed();
  S.intersectAssumedBits(T.getAssumed());
  return Assumed == S.getAssumed() ? ChangeStatus::UNCHANGED
                                   : ChangeStatus::CHANGED;
}

/// NoCapture attribute for function arguments.
struct AANoCaptureArgument final : AANoCaptureImpl {
  AANoCaptureArgument(const IRPosition &IRP) : AANoCaptureImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nocapture) }
};

/// NoCapture attribute for call site arguments.
struct AANoCaptureCallSiteArgument final : AANoCaptureImpl {
  AANoCaptureCallSiteArgument(const IRPosition &IRP) : AANoCaptureImpl(IRP) {}

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
    auto &ArgAA = A.getAAFor<AANoCapture>(*this, ArgPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AANoCapture::StateType &>(ArgAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override{STATS_DECLTRACK_CSARG_ATTR(nocapture)};
};

/// NoCapture attribute for floating values.
struct AANoCaptureFloating final : AANoCaptureImpl {
  AANoCaptureFloating(const IRPosition &IRP) : AANoCaptureImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(nocapture)
  }
};

/// NoCapture attribute for function return value.
struct AANoCaptureReturned final : AANoCaptureImpl {
  AANoCaptureReturned(const IRPosition &IRP) : AANoCaptureImpl(IRP) {
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
  AANoCaptureCallSiteReturned(const IRPosition &IRP) : AANoCaptureImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(nocapture)
  }
};

/// ------------------ Value Simplify Attribute ----------------------------
struct AAValueSimplifyImpl : AAValueSimplify {
  AAValueSimplifyImpl(const IRPosition &IRP) : AAValueSimplify(IRP) {}

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
  void initialize(Attributor &A) override {}

  /// Helper function for querying AAValueSimplify and updating candicate.
  /// \param QueryingValue Value trying to unify with SimplifiedValue
  /// \param AccumulatedSimplifiedValue Current simplification result.
  static bool checkAndUpdate(Attributor &A, const AbstractAttribute &QueryingAA,
                             Value &QueryingValue,
                             Optional<Value *> &AccumulatedSimplifiedValue) {
    // FIXME: Add a typecast support.

    auto &ValueSimpifyAA = A.getAAFor<AAValueSimplify>(
        QueryingAA, IRPosition::value(QueryingValue));

    Optional<Value *> QueryingValueSimplified =
        ValueSimpifyAA.getAssumedSimplifiedValue(A);

    if (!QueryingValueSimplified.hasValue())
      return true;

    if (!QueryingValueSimplified.getValue())
      return false;

    Value &QueryingValueSimplifiedUnwrapped =
        *QueryingValueSimplified.getValue();

    if (isa<UndefValue>(QueryingValueSimplifiedUnwrapped))
      return true;

    if (AccumulatedSimplifiedValue.hasValue())
      return AccumulatedSimplifiedValue == QueryingValueSimplified;

    LLVM_DEBUG(dbgs() << "[Attributor][ValueSimplify] " << QueryingValue
                      << " is assumed to be "
                      << QueryingValueSimplifiedUnwrapped << "\n");

    AccumulatedSimplifiedValue = QueryingValueSimplified;
    return true;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    if (!SimplifiedAssociatedValue.hasValue() ||
        !SimplifiedAssociatedValue.getValue())
      return Changed;

    if (auto *C = dyn_cast<Constant>(SimplifiedAssociatedValue.getValue())) {
      // We can replace the AssociatedValue with the constant.
      Value &V = getAssociatedValue();
      if (!V.user_empty() && &V != C && V.getType() == C->getType()) {
        LLVM_DEBUG(dbgs() << "[Attributor][ValueSimplify] " << V << " -> " << *C
                          << "\n");
        V.replaceAllUsesWith(C);
        Changed = ChangeStatus::CHANGED;
      }
    }

    return Changed | AAValueSimplify::manifest(A);
  }

protected:
  // An assumed simplified value. Initially, it is set to Optional::None, which
  // means that the value is not clear under current assumption. If in the
  // pessimistic state, getAssumedSimplifiedValue doesn't return this value but
  // returns orignal associated value.
  Optional<Value *> SimplifiedAssociatedValue;
};

struct AAValueSimplifyArgument final : AAValueSimplifyImpl {
  AAValueSimplifyArgument(const IRPosition &IRP) : AAValueSimplifyImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    auto PredForCallSite = [&](CallSite CS) {
      return checkAndUpdate(A, *this, *CS.getArgOperand(getArgNo()),
                            SimplifiedAssociatedValue);
    };

    if (!A.checkForAllCallSites(PredForCallSite, *this, true))
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
  AAValueSimplifyReturned(const IRPosition &IRP) : AAValueSimplifyImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    auto PredForReturned = [&](Value &V) {
      return checkAndUpdate(A, *this, V, SimplifiedAssociatedValue);
    };

    if (!A.checkForAllReturnedValues(PredForReturned, *this))
      return indicatePessimisticFixpoint();

    // If a candicate was found in this update, return CHANGED.
    return HasValueBefore == SimplifiedAssociatedValue.hasValue()
               ? ChangeStatus::UNCHANGED
               : ChangeStatus ::CHANGED;
  }
  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(value_simplify)
  }
};

struct AAValueSimplifyFloating : AAValueSimplifyImpl {
  AAValueSimplifyFloating(const IRPosition &IRP) : AAValueSimplifyImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Value &V = getAnchorValue();

    // TODO: add other stuffs
    if (isa<Constant>(V) || isa<UndefValue>(V))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    auto VisitValueCB = [&](Value &V, BooleanState, bool Stripped) -> bool {
      auto &AA = A.getAAFor<AAValueSimplify>(*this, IRPosition::value(V));
      if (!Stripped && this == &AA) {
        // TODO: Look the instruction and check recursively.
        LLVM_DEBUG(
            dbgs() << "[Attributor][ValueSimplify] Can't be stripped more : "
                   << V << "\n");
        indicatePessimisticFixpoint();
        return false;
      }
      return checkAndUpdate(A, *this, V, SimplifiedAssociatedValue);
    };

    if (!genericValueTraversal<AAValueSimplify, BooleanState>(
            A, getIRPosition(), *this, static_cast<BooleanState &>(*this),
            VisitValueCB))
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
  AAValueSimplifyFunction(const IRPosition &IRP) : AAValueSimplifyImpl(IRP) {}

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
  AAValueSimplifyCallSite(const IRPosition &IRP)
      : AAValueSimplifyFunction(IRP) {}
  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(value_simplify)
  }
};

struct AAValueSimplifyCallSiteReturned : AAValueSimplifyReturned {
  AAValueSimplifyCallSiteReturned(const IRPosition &IRP)
      : AAValueSimplifyReturned(IRP) {}

  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(value_simplify)
  }
};
struct AAValueSimplifyCallSiteArgument : AAValueSimplifyFloating {
  AAValueSimplifyCallSiteArgument(const IRPosition &IRP)
      : AAValueSimplifyFloating(IRP) {}

  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(value_simplify)
  }
};

/// ----------------------- Heap-To-Stack Conversion ---------------------------
struct AAHeapToStackImpl : public AAHeapToStack {
  AAHeapToStackImpl(const IRPosition &IRP) : AAHeapToStack(IRP) {}

  const std::string getAsStr() const override {
    return "[H2S] Mallocs: " + std::to_string(MallocCalls.size());
  }

  ChangeStatus manifest(Attributor &A) override {
    assert(getState().isValidState() &&
           "Attempted to manifest an invalid state!");

    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
    Function *F = getAssociatedFunction();
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

      Constant *Size;
      if (isCallocLikeFn(MallocCall, TLI)) {
        auto *Num = cast<ConstantInt>(MallocCall->getOperand(0));
        auto *SizeT = dyn_cast<ConstantInt>(MallocCall->getOperand(1));
        APInt TotalSize = SizeT->getValue() * Num->getValue();
        Size =
            ConstantInt::get(MallocCall->getOperand(0)->getType(), TotalSize);
      } else {
        Size = cast<ConstantInt>(MallocCall->getOperand(0));
      }

      unsigned AS = cast<PointerType>(MallocCall->getType())->getAddressSpace();
      Instruction *AI = new AllocaInst(Type::getInt8Ty(F->getContext()), AS,
                                       Size, "", MallocCall->getNextNode());

      if (AI->getType() != MallocCall->getType())
        AI = new BitCastInst(AI, MallocCall->getType(), "malloc_bc",
                             AI->getNextNode());

      MallocCall->replaceAllUsesWith(AI);

      if (auto *II = dyn_cast<InvokeInst>(MallocCall)) {
        auto *NBB = II->getNormalDest();
        BranchInst::Create(NBB, MallocCall->getParent());
        A.deleteAfterManifest(*MallocCall);
      } else {
        A.deleteAfterManifest(*MallocCall);
      }

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
  const Function *F = getAssociatedFunction();
  const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);

  auto UsesCheck = [&](Instruction &I) {
    SmallPtrSet<const Use *, 8> Visited;
    SmallVector<const Use *, 8> Worklist;

    for (Use &U : I.uses())
      Worklist.push_back(&U);

    while (!Worklist.empty()) {
      const Use *U = Worklist.pop_back_val();
      if (!Visited.insert(U).second)
        continue;

      auto *UserI = U->getUser();

      if (isa<LoadInst>(UserI) || isa<StoreInst>(UserI))
        continue;

      // NOTE: Right now, if a function that has malloc pointer as an argument
      // frees memory, we assume that the malloc pointer is freed.

      // TODO: Add nofree callsite argument attribute to indicate that pointer
      // argument is not freed.
      if (auto *CB = dyn_cast<CallBase>(UserI)) {
        if (!CB->isArgOperand(U))
          continue;

        if (CB->isLifetimeStartOrEnd())
          continue;

        // Record malloc.
        if (isFreeCall(UserI, TLI)) {
          FreesForMalloc[&I].insert(
              cast<Instruction>(const_cast<User *>(UserI)));
          continue;
        }

        // If a function does not free memory we are fine
        const auto &NoFreeAA =
            A.getAAFor<AANoFree>(*this, IRPosition::callsite_function(*CB));

        unsigned ArgNo = U - CB->arg_begin();
        const auto &NoCaptureAA = A.getAAFor<AANoCapture>(
            *this, IRPosition::callsite_argument(*CB, ArgNo));

        if (!NoCaptureAA.isAssumedNoCapture() || !NoFreeAA.isAssumedNoFree()) {
          LLVM_DEBUG(dbgs() << "[H2S] Bad user: " << *UserI << "\n");
          return false;
        }
        continue;
      }

      if (isa<GetElementPtrInst>(UserI) || isa<BitCastInst>(UserI)) {
        for (Use &U : UserI->uses())
          Worklist.push_back(&U);
        continue;
      }

      // Unknown user.
      LLVM_DEBUG(dbgs() << "[H2S] Unknown user: " << *UserI << "\n");
      return false;
    }
    return true;
  };

  auto MallocCallocCheck = [&](Instruction &I) {
    if (isMallocLikeFn(&I, TLI)) {
      if (auto *Size = dyn_cast<ConstantInt>(I.getOperand(0)))
        if (!Size->getValue().sle(MaxHeapToStackSize))
          return true;
    } else if (isCallocLikeFn(&I, TLI)) {
      bool Overflow = false;
      if (auto *Num = dyn_cast<ConstantInt>(I.getOperand(0)))
        if (auto *Size = dyn_cast<ConstantInt>(I.getOperand(1)))
          if (!(Size->getValue().umul_ov(Num->getValue(), Overflow))
                   .sle(MaxHeapToStackSize))
            if (!Overflow)
              return true;
    } else {
      BadMallocCalls.insert(&I);
      return true;
    }

    if (BadMallocCalls.count(&I))
      return true;

    if (UsesCheck(I))
      MallocCalls.insert(&I);
    else
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
  AAHeapToStackFunction(const IRPosition &IRP) : AAHeapToStackImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECL(MallocCalls, Function,
               "Number of MallocCalls converted to allocas");
    BUILD_STAT_NAME(MallocCalls, Function) += MallocCalls.size();
  }
};
} // namespace

/// ----------------------------------------------------------------------------
///                               Attributor
/// ----------------------------------------------------------------------------

bool Attributor::isAssumedDead(const AbstractAttribute &AA,
                               const AAIsDead *LivenessAA) {
  const Instruction *CtxI = AA.getIRPosition().getCtxI();
  if (!CtxI)
    return false;

  if (!LivenessAA)
    LivenessAA =
        &getAAFor<AAIsDead>(AA, IRPosition::function(*CtxI->getFunction()),
                            /* TrackDependence */ false);

  // Don't check liveness for AAIsDead.
  if (&AA == LivenessAA)
    return false;

  if (!LivenessAA->isAssumedDead(CtxI))
    return false;

  // We actually used liveness information so we have to record a dependence.
  recordDependence(*LivenessAA, AA);

  return true;
}

bool Attributor::checkForAllCallSites(const function_ref<bool(CallSite)> &Pred,
                                      const AbstractAttribute &QueryingAA,
                                      bool RequireAllCallSites) {
  // We can try to determine information from
  // the call sites. However, this is only possible all call sites are known,
  // hence the function has internal linkage.
  const IRPosition &IRP = QueryingAA.getIRPosition();
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  if (RequireAllCallSites && !AssociatedFunction->hasInternalLinkage()) {
    LLVM_DEBUG(
        dbgs()
        << "[Attributor] Function " << AssociatedFunction->getName()
        << " has no internal linkage, hence not all call sites are known\n");
    return false;
  }

  for (const Use &U : AssociatedFunction->uses()) {
    Instruction *I = dyn_cast<Instruction>(U.getUser());
    // TODO: Deal with abstract call sites here.
    if (!I)
      return false;

    Function *Caller = I->getFunction();

    const auto &LivenessAA = getAAFor<AAIsDead>(
        QueryingAA, IRPosition::function(*Caller), /* TrackDependence */ false);

    // Skip dead calls.
    if (LivenessAA.isAssumedDead(I)) {
      // We actually used liveness information so we have to record a
      // dependence.
      recordDependence(LivenessAA, QueryingAA);
      continue;
    }

    CallSite CS(U.getUser());
    if (!CS || !CS.isCallee(&U)) {
      if (!RequireAllCallSites)
        continue;

      LLVM_DEBUG(dbgs() << "[Attributor] User " << *U.getUser()
                        << " is an invalid use of "
                        << AssociatedFunction->getName() << "\n");
      return false;
    }

    if (Pred(CS))
      continue;

    LLVM_DEBUG(dbgs() << "[Attributor] Call site callback failed for "
                      << *CS.getInstruction() << "\n");
    return false;
  }

  return true;
}

bool Attributor::checkForAllReturnedValuesAndReturnInsts(
    const function_ref<bool(Value &, const SmallSetVector<ReturnInst *, 4> &)>
        &Pred,
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
  const auto &AARetVal = getAAFor<AAReturnedValues>(QueryingAA, QueryIRP);
  if (!AARetVal.getState().isValidState())
    return false;

  return AARetVal.checkForAllReturnedValuesAndReturnInsts(Pred);
}

bool Attributor::checkForAllReturnedValues(
    const function_ref<bool(Value &)> &Pred,
    const AbstractAttribute &QueryingAA) {

  const IRPosition &IRP = QueryingAA.getIRPosition();
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(*AssociatedFunction);
  const auto &AARetVal = getAAFor<AAReturnedValues>(QueryingAA, QueryIRP);
  if (!AARetVal.getState().isValidState())
    return false;

  return AARetVal.checkForAllReturnedValuesAndReturnInsts(
      [&](Value &RV, const SmallSetVector<ReturnInst *, 4> &) {
        return Pred(RV);
      });
}

static bool
checkForAllInstructionsImpl(InformationCache::OpcodeInstMapTy &OpcodeInstMap,
                            const function_ref<bool(Instruction &)> &Pred,
                            const AAIsDead *LivenessAA, bool &AnyDead,
                            const ArrayRef<unsigned> &Opcodes) {
  for (unsigned Opcode : Opcodes) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      // Skip dead instructions.
      if (LivenessAA && LivenessAA->isAssumedDead(I)) {
        AnyDead = true;
        continue;
      }

      if (!Pred(*I))
        return false;
    }
  }
  return true;
}

bool Attributor::checkForAllInstructions(
    const llvm::function_ref<bool(Instruction &)> &Pred,
    const AbstractAttribute &QueryingAA, const ArrayRef<unsigned> &Opcodes) {

  const IRPosition &IRP = QueryingAA.getIRPosition();
  // Since we need to provide instructions we have to have an exact definition.
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(*AssociatedFunction);
  const auto &LivenessAA =
      getAAFor<AAIsDead>(QueryingAA, QueryIRP, /* TrackDependence */ false);
  bool AnyDead = false;

  auto &OpcodeInstMap =
      InfoCache.getOpcodeInstMapForFunction(*AssociatedFunction);
  if (!checkForAllInstructionsImpl(OpcodeInstMap, Pred, &LivenessAA, AnyDead, Opcodes))
    return false;

  // If we actually used liveness information so we have to record a dependence.
  if (AnyDead)
    recordDependence(LivenessAA, QueryingAA);

  return true;
}

bool Attributor::checkForAllReadWriteInstructions(
    const llvm::function_ref<bool(Instruction &)> &Pred,
    AbstractAttribute &QueryingAA) {

  const Function *AssociatedFunction =
      QueryingAA.getIRPosition().getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  // TODO: use the function scope once we have call site AAReturnedValues.
  const IRPosition &QueryIRP = IRPosition::function(*AssociatedFunction);
  const auto &LivenessAA =
      getAAFor<AAIsDead>(QueryingAA, QueryIRP, /* TrackDependence */ false);
  bool AnyDead = false;

  for (Instruction *I :
       InfoCache.getReadOrWriteInstsForFunction(*AssociatedFunction)) {
    // Skip dead instructions.
    if (LivenessAA.isAssumedDead(I)) {
      AnyDead = true;
      continue;
    }

    if (!Pred(*I))
      return false;
  }

  // If we actually used liveness information so we have to record a dependence.
  if (AnyDead)
    recordDependence(LivenessAA, QueryingAA);

  return true;
}

ChangeStatus Attributor::run(Module &M) {
  LLVM_DEBUG(dbgs() << "[Attributor] Identified and initialized "
                    << AllAbstractAttributes.size()
                    << " abstract attributes.\n");

  // Now that all abstract attributes are collected and initialized we start
  // the abstract analysis.

  unsigned IterationCounter = 1;

  SmallVector<AbstractAttribute *, 64> ChangedAAs;
  SetVector<AbstractAttribute *> Worklist;
  Worklist.insert(AllAbstractAttributes.begin(), AllAbstractAttributes.end());

  bool RecomputeDependences = false;

  do {
    // Remember the size to determine new attributes.
    size_t NumAAs = AllAbstractAttributes.size();
    LLVM_DEBUG(dbgs() << "\n\n[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist size: " << Worklist.size() << "\n");

    // If dependences (=QueryMap) are recomputed we have to look at all abstract
    // attributes again, regardless of what changed in the last iteration.
    if (RecomputeDependences) {
      LLVM_DEBUG(
          dbgs() << "[Attributor] Run all AAs to recompute dependences\n");
      QueryMap.clear();
      ChangedAAs.clear();
      Worklist.insert(AllAbstractAttributes.begin(),
                      AllAbstractAttributes.end());
    }

    // Add all abstract attributes that are potentially dependent on one that
    // changed to the work list.
    for (AbstractAttribute *ChangedAA : ChangedAAs) {
      auto &QuerriedAAs = QueryMap[ChangedAA];
      Worklist.insert(QuerriedAAs.begin(), QuerriedAAs.end());
    }

    LLVM_DEBUG(dbgs() << "[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist+Dependent size: " << Worklist.size()
                      << "\n");

    // Reset the changed set.
    ChangedAAs.clear();

    // Update all abstract attribute in the work list and record the ones that
    // changed.
    for (AbstractAttribute *AA : Worklist)
      if (!isAssumedDead(*AA, nullptr))
        if (AA->update(*this) == ChangeStatus::CHANGED)
          ChangedAAs.push_back(AA);

    // Check if we recompute the dependences in the next iteration.
    RecomputeDependences = (DepRecomputeInterval > 0 &&
                            IterationCounter % DepRecomputeInterval == 0);

    // Add attributes to the changed set if they have been created in the last
    // iteration.
    ChangedAAs.append(AllAbstractAttributes.begin() + NumAAs,
                      AllAbstractAttributes.end());

    // Reset the work list and repopulate with the changed abstract attributes.
    // Note that dependent ones are added above.
    Worklist.clear();
    Worklist.insert(ChangedAAs.begin(), ChangedAAs.end());

  } while (!Worklist.empty() && (IterationCounter++ < MaxFixpointIterations ||
                                 VerifyMaxFixpointIterations));

  LLVM_DEBUG(dbgs() << "\n[Attributor] Fixpoint iteration done after: "
                    << IterationCounter << "/" << MaxFixpointIterations
                    << " iterations\n");

  size_t NumFinalAAs = AllAbstractAttributes.size();

  bool FinishedAtFixpoint = Worklist.empty();

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

    auto &QuerriedAAs = QueryMap[ChangedAA];
    ChangedAAs.append(QuerriedAAs.begin(), QuerriedAAs.end());
  }

  LLVM_DEBUG({
    if (!Visited.empty())
      dbgs() << "\n[Attributor] Finalized " << Visited.size()
             << " abstract attributes.\n";
  });

  unsigned NumManifested = 0;
  unsigned NumAtFixpoint = 0;
  ChangeStatus ManifestChange = ChangeStatus::UNCHANGED;
  for (AbstractAttribute *AA : AllAbstractAttributes) {
    AbstractState &State = AA->getState();

    // If there is not already a fixpoint reached, we can now take the
    // optimistic state. This is correct because we enforced a pessimistic one
    // on abstract attributes that were transitively dependent on a changed one
    // already above.
    if (!State.isAtFixpoint())
      State.indicateOptimisticFixpoint();

    // If the state is invalid, we do not try to manifest it.
    if (!State.isValidState())
      continue;

    // Skip dead code.
    if (isAssumedDead(*AA, nullptr))
      continue;
    // Manifest the state and record if we changed the IR.
    ChangeStatus LocalChange = AA->manifest(*this);
    if (LocalChange == ChangeStatus::CHANGED && AreStatisticsEnabled())
      AA->trackStatistics();

    ManifestChange = ManifestChange | LocalChange;

    NumAtFixpoint++;
    NumManifested += (LocalChange == ChangeStatus::CHANGED);
  }

  (void)NumManifested;
  (void)NumAtFixpoint;
  LLVM_DEBUG(dbgs() << "\n[Attributor] Manifested " << NumManifested
                    << " arguments while " << NumAtFixpoint
                    << " were in a valid fixpoint state\n");

  // If verification is requested, we finished this run at a fixpoint, and the
  // IR was changed, we re-run the whole fixpoint analysis, starting at
  // re-initialization of the arguments. This re-run should not result in an IR
  // change. Though, the (virtual) state of attributes at the end of the re-run
  // might be more optimistic than the known state or the IR state if the better
  // state cannot be manifested.
  if (VerifyAttributor && FinishedAtFixpoint &&
      ManifestChange == ChangeStatus::CHANGED) {
    VerifyAttributor = false;
    ChangeStatus VerifyStatus = run(M);
    if (VerifyStatus != ChangeStatus::UNCHANGED)
      llvm_unreachable(
          "Attributor verification failed, re-run did result in an IR change "
          "even after a fixpoint was reached in the original run. (False "
          "positives possible!)");
    VerifyAttributor = true;
  }

  NumAttributesManifested += NumManifested;
  NumAttributesValidFixpoint += NumAtFixpoint;

  (void)NumFinalAAs;
  assert(
      NumFinalAAs == AllAbstractAttributes.size() &&
      "Expected the final number of abstract attributes to remain unchanged!");

  // Delete stuff at the end to avoid invalid references and a nice order.
  {
    LLVM_DEBUG(dbgs() << "\n[Attributor] Delete at least "
                      << ToBeDeletedFunctions.size() << " functions and "
                      << ToBeDeletedBlocks.size() << " blocks and "
                      << ToBeDeletedInsts.size() << " instructions\n");
    for (Instruction *I : ToBeDeletedInsts) {
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }

    if (unsigned NumDeadBlocks = ToBeDeletedBlocks.size()) {
      SmallVector<BasicBlock *, 8> ToBeDeletedBBs;
      ToBeDeletedBBs.reserve(NumDeadBlocks);
      ToBeDeletedBBs.append(ToBeDeletedBlocks.begin(), ToBeDeletedBlocks.end());
      DeleteDeadBlocks(ToBeDeletedBBs);
      STATS_DECLTRACK(AAIsDead, BasicBlock,
                      "Number of dead basic blocks deleted.");
    }

    STATS_DECL(AAIsDead, Function, "Number of dead functions deleted.");
    for (Function *Fn : ToBeDeletedFunctions) {
      Fn->replaceAllUsesWith(UndefValue::get(Fn->getType()));
      Fn->eraseFromParent();
      STATS_TRACK(AAIsDead, Function);
    }

    // Identify dead internal functions and delete them. This happens outside
    // the other fixpoint analysis as we might treat potentially dead functions
    // as live to lower the number of iterations. If they happen to be dead, the
    // below fixpoint loop will identify and eliminate them.
    SmallVector<Function *, 8> InternalFns;
    for (Function &F : M)
      if (F.hasInternalLinkage())
        InternalFns.push_back(&F);

    bool FoundDeadFn = true;
    while (FoundDeadFn) {
      FoundDeadFn = false;
      for (unsigned u = 0, e = InternalFns.size(); u < e; ++u) {
        Function *F = InternalFns[u];
        if (!F)
          continue;

        const auto *LivenessAA =
            lookupAAFor<AAIsDead>(IRPosition::function(*F));
        if (LivenessAA &&
            !checkForAllCallSites([](CallSite CS) { return false; },
                                  *LivenessAA, true))
          continue;

        STATS_TRACK(AAIsDead, Function);
        F->replaceAllUsesWith(UndefValue::get(F->getType()));
        F->eraseFromParent();
        InternalFns[u] = nullptr;
        FoundDeadFn = true;
      }
    }
  }

  if (VerifyMaxFixpointIterations &&
      IterationCounter != MaxFixpointIterations) {
    errs() << "\n[Attributor] Fixpoint iteration done after: "
           << IterationCounter << "/" << MaxFixpointIterations
           << " iterations\n";
    llvm_unreachable("The fixpoint was not reached with exactly the number of "
                     "specified iterations!");
  }

  return ManifestChange;
}

void Attributor::initializeInformationCache(Function &F) {

  // Walk all instructions to find interesting instructions that might be
  // queried by abstract attributes during their initialization or update.
  // This has to happen before we create attributes.
  auto &ReadOrWriteInsts = InfoCache.FuncRWInstsMap[&F];
  auto &InstOpcodeMap = InfoCache.FuncInstOpcodeMap[&F];

  for (Instruction &I : instructions(&F)) {
    bool IsInterestingOpcode = false;

    // To allow easy access to all instructions in a function with a given
    // opcode we store them in the InfoCache. As not all opcodes are interesting
    // to concrete attributes we only cache the ones that are as identified in
    // the following switch.
    // Note: There are no concrete attributes now so this is initially empty.
    switch (I.getOpcode()) {
    default:
      assert((!ImmutableCallSite(&I)) && (!isa<CallBase>(&I)) &&
             "New call site/base instruction type needs to be known int the "
             "Attributor.");
      break;
    case Instruction::Load:
      // The alignment of a pointer is interesting for loads.
    case Instruction::Store:
      // The alignment of a pointer is interesting for stores.
    case Instruction::Call:
    case Instruction::CallBr:
    case Instruction::Invoke:
    case Instruction::CleanupRet:
    case Instruction::CatchSwitch:
    case Instruction::Resume:
    case Instruction::Ret:
      IsInterestingOpcode = true;
    }
    if (IsInterestingOpcode)
      InstOpcodeMap[I.getOpcode()].push_back(&I);
    if (I.mayReadOrWriteMemory())
      ReadOrWriteInsts.push_back(&I);
  }
}

void Attributor::identifyDefaultAbstractAttributes(Function &F) {
  if (!VisitedFunctions.insert(&F).second)
    return;

  IRPosition FPos = IRPosition::function(F);

  // Check for dead BasicBlocks in every function.
  // We need dead instruction detection because we do not want to deal with
  // broken IR in which SSA rules do not apply.
  getOrCreateAAFor<AAIsDead>(FPos);

  // Every function might be "will-return".
  getOrCreateAAFor<AAWillReturn>(FPos);

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

    // Every function might be simplified.
    getOrCreateAAFor<AAValueSimplify>(RetPos);

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

    // Every argument might be simplified.
    getOrCreateAAFor<AAValueSimplify>(ArgPos);

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
    }
  }

  auto CallSitePred = [&](Instruction &I) -> bool {
    CallSite CS(&I);
    if (CS.getCalledFunction()) {
      for (int i = 0, e = CS.getCalledFunction()->arg_size(); i < e; i++) {

        IRPosition CSArgPos = IRPosition::callsite_argument(CS, i);

        // Call site argument might be simplified.
        getOrCreateAAFor<AAValueSimplify>(CSArgPos);

        if (!CS.getArgument(i)->getType()->isPointerTy())
          continue;

        // Call site argument attribute "non-null".
        getOrCreateAAFor<AANonNull>(CSArgPos);

        // Call site argument attribute "no-alias".
        getOrCreateAAFor<AANoAlias>(CSArgPos);

        // Call site argument attribute "dereferenceable".
        getOrCreateAAFor<AADereferenceable>(CSArgPos);

        // Call site argument attribute "align".
        getOrCreateAAFor<AAAlign>(CSArgPos);
      }
    }
    return true;
  };

  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);
  bool Success, AnyDead = false;
  Success = checkForAllInstructionsImpl(
      OpcodeInstMap, CallSitePred, nullptr, AnyDead,
      {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
       (unsigned)Instruction::Call});
  (void)Success;
  assert(Success && !AnyDead && "Expected the check call to be successful!");

  auto LoadStorePred = [&](Instruction &I) -> bool {
    if (isa<LoadInst>(I))
      getOrCreateAAFor<AAAlign>(
          IRPosition::value(*cast<LoadInst>(I).getPointerOperand()));
    else
      getOrCreateAAFor<AAAlign>(
          IRPosition::value(*cast<StoreInst>(I).getPointerOperand()));
    return true;
  };
  Success = checkForAllInstructionsImpl(
      OpcodeInstMap, LoadStorePred, nullptr, AnyDead,
      {(unsigned)Instruction::Load, (unsigned)Instruction::Store});
  (void)Success;
  assert(Success && !AnyDead && "Expected the check call to be successful!");
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
  return OS << "{" << Pos.getPositionKind() << ":" << AV.getName() << " ["
            << Pos.getAnchorValue().getName() << "@" << Pos.getArgNo() << "]}";
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const IntegerState &S) {
  return OS << "(" << S.getKnown() << "-" << S.getAssumed() << ")"
            << static_cast<const AbstractState &>(S);
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractState &S) {
  return OS << (!S.isValidState() ? "top" : (S.isAtFixpoint() ? "fix" : ""));
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractAttribute &AA) {
  AA.print(OS);
  return OS;
}

void AbstractAttribute::print(raw_ostream &OS) const {
  OS << "[P: " << getIRPosition() << "][" << getAsStr() << "][S: " << getState()
     << "]";
}
///}

/// ----------------------------------------------------------------------------
///                       Pass (Manager) Boilerplate
/// ----------------------------------------------------------------------------

static bool runAttributorOnModule(Module &M, AnalysisGetter &AG) {
  if (DisableAttributor)
    return false;

  LLVM_DEBUG(dbgs() << "[Attributor] Run on module with " << M.size()
                    << " functions.\n");

  // Create an Attributor and initially empty information cache that is filled
  // while we identify default attribute opportunities.
  InformationCache InfoCache(M, AG);
  Attributor A(InfoCache, DepRecInterval);

  for (Function &F : M)
    A.initializeInformationCache(F);

  for (Function &F : M) {
    if (F.hasExactDefinition())
      NumFnWithExactDefinition++;
    else
      NumFnWithoutExactDefinition++;

    // For now we ignore naked and optnone functions.
    if (F.hasFnAttribute(Attribute::Naked) ||
        F.hasFnAttribute(Attribute::OptimizeNone))
      continue;

    // We look at internal functions only on-demand but if any use is not a
    // direct call, we have to do it eagerly.
    if (F.hasInternalLinkage()) {
      if (llvm::all_of(F.uses(), [](const Use &U) {
            return ImmutableCallSite(U.getUser()) &&
                   ImmutableCallSite(U.getUser()).isCallee(&U);
          }))
        continue;
    }

    // Populate the Attributor with abstract attribute opportunities in the
    // function and the information cache with IR information.
    A.identifyDefaultAbstractAttributes(F);
  }

  return A.run(M) == ChangeStatus::CHANGED;
}

PreservedAnalyses AttributorPass::run(Module &M, ModuleAnalysisManager &AM) {
  AnalysisGetter AG(AM);
  if (runAttributorOnModule(M, AG)) {
    // FIXME: Think about passes we will preserve and add them here.
    return PreservedAnalyses::none();
  }
  return PreservedAnalyses::all();
}

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
    return runAttributorOnModule(M, AG);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // FIXME: Think about passes we will preserve and add them here.
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }
};

} // end anonymous namespace

Pass *llvm::createAttributorLegacyPass() { return new AttributorLegacyPass(); }

char AttributorLegacyPass::ID = 0;

const char AAReturnedValues::ID = 0;
const char AANoUnwind::ID = 0;
const char AANoSync::ID = 0;
const char AANoFree::ID = 0;
const char AANonNull::ID = 0;
const char AANoRecurse::ID = 0;
const char AAWillReturn::ID = 0;
const char AANoAlias::ID = 0;
const char AANoReturn::ID = 0;
const char AAIsDead::ID = 0;
const char AADereferenceable::ID = 0;
const char AAAlign::ID = 0;
const char AANoCapture::ID = 0;
const char AAValueSimplify::ID = 0;
const char AAHeapToStack::ID = 0;

// Macro magic to create the static generator function for attributes that
// follow the naming scheme.

#define SWITCH_PK_INV(CLASS, PK, POS_NAME)                                     \
  case IRPosition::PK:                                                         \
    llvm_unreachable("Cannot create " #CLASS " for a " POS_NAME " position!");

#define SWITCH_PK_CREATE(CLASS, IRP, PK, SUFFIX)                               \
  case IRPosition::PK:                                                         \
    AA = new CLASS##SUFFIX(IRP);                                               \
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
    AA->initialize(A);                                                         \
    return *AA;                                                                \
  }

CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoUnwind)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoSync)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoFree)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoRecurse)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAWillReturn)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoReturn)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAIsDead)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAReturnedValues)

CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANonNull)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoAlias)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AADereferenceable)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAAlign)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoCapture)

CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAValueSimplify)

CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAHeapToStack)

#undef CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef SWITCH_PK_CREATE
#undef SWITCH_PK_INV

INITIALIZE_PASS_BEGIN(AttributorLegacyPass, "attributor",
                      "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AttributorLegacyPass, "attributor",
                    "Deduce and propagate attributes", false, false)
