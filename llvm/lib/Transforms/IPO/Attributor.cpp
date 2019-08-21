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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/EHPersonalities.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/Loads.h"
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
#define STATS_DECL(NAME, TYPE, MSG) STATISTIC(BUILD_STAT_NAME(NAME, TYPE), MSG);
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
                  BUILD_STAT_MSG_IR_ATTR(function returns, NAME));
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

static cl::opt<bool> DisableAttributor(
    "attributor-disable", cl::Hidden,
    cl::desc("Disable the attributor inter-procedural deduction pass."),
    cl::init(true));

static cl::opt<bool> VerifyAttributor(
    "attributor-verify", cl::Hidden,
    cl::desc("Verify the Attributor deduction and "
             "manifestation of attributes -- may issue false-positive errors"),
    cl::init(false));

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
bool genericValueTraversal(
    Attributor &A, IRPosition IRP, const AAType &QueryingAA, StateTy &State,
    const function_ref<bool(Value &, StateTy &, bool)> &VisitValueCB,
    int MaxValues = 8) {

  const AAIsDead *LivenessAA = nullptr;
  if (IRP.getAnchorScope())
    LivenessAA = &A.getAAFor<AAIsDead>(
        QueryingAA, IRPosition::function(*IRP.getAnchorScope()));

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
        if (!LivenessAA->isAssumedDead(IncomingBB->getTerminator()))
          Worklist.push_back(PHI->getIncomingValue(u));
      }
      continue;
    }

    // Once a leaf is reached we inform the user through the callback.
    if (!VisitValueCB(*V, State, Iteration > 1))
      return false;
  } while (!Worklist.empty());

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
    if (auto *Arg = dyn_cast<Argument>(AnchorVal)) {
      assert(Arg->getArgNo() == unsigned(getArgNo()) &&
             "Argument number mismatch!");
      assert(Arg == &getAssociatedValue() && "Associated value mismatch!");
    } else {
      auto &CB = cast<CallBase>(*AnchorVal);
      (void)CB;
      assert(CB.arg_size() > unsigned(getArgNo()) &&
             "Call site argument number mismatch!");
      assert(CB.getArgOperand(getArgNo()) == &getAssociatedValue() &&
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

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({Attribute::NoUnwind}))
      indicateOptimisticFixpoint();
  }

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
using AANoUnwindCallSite = AANoUnwindFunction;

/// --------------------- Function Return Values -------------------------------

/// "Attribute" that collects all potential returned values and the return
/// instructions that they arise from.
///
/// If there is a unique returned value R, the manifest method will:
///   - mark R with the "returned" attribute, if R is an argument.
class AAReturnedValuesImpl : public AAReturnedValues, public AbstractState {

  /// Mapping of values potentially returned by the associated function to the
  /// return instructions that might return them.
  DenseMap<Value *, SmallPtrSet<ReturnInst *, 2>> ReturnedValues;

  SmallPtrSet<CallBase *, 8> UnresolvedCalls;

  /// State flags
  ///
  ///{
  bool IsFixed;
  bool IsValidState;
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
    if (!F || !F->hasExactDefinition()) {
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

  const SmallPtrSetImpl<CallBase *> &getUnresolvedCalls() const override {
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
      const function_ref<bool(Value &, const SmallPtrSetImpl<ReturnInst *> &)>
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
    IsValidState &= true;
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

  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.getValue())) {
    getIRPosition() = IRPosition::argument(*UniqueRVArg);
    Changed = IRAttribute::manifest(A) | Changed;
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
    const function_ref<bool(Value &, const SmallPtrSetImpl<ReturnInst *> &)>
        &Pred) const {
  if (!isValidState())
    return false;

  // Check all returned values but ignore call sites as long as we have not
  // encountered an overdefined one during an update.
  for (auto &It : ReturnedValues) {
    Value *RV = It.first;
    const SmallPtrSetImpl<ReturnInst *> &RetInsts = It.second;

    CallBase *CB = dyn_cast<CallBase>(RV);
    if (CB && !UnresolvedCalls.count(CB))
      continue;

    if (!Pred(*RV, RetInsts))
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
    SmallPtrSet<ReturnInst *, 2> RetInsts;
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

    const auto &RetValAA =
        A.getAAFor<AAReturnedValues>(*this, IRPosition::callsite_function(*CB));
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
      if (ReturnInsts.insert(RI).second) {
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
using AAReturnedValuesCallSite = AAReturnedValuesFunction;

/// ------------------------ NoSync Function Attribute -------------------------

struct AANoSyncImpl : AANoSync {
  AANoSyncImpl(const IRPosition &IRP) : AANoSync(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({Attribute::NoSync}))
      indicateOptimisticFixpoint();
  }

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
using AANoSyncCallSite = AANoSyncFunction;

/// ------------------------ No-Free Attributes ----------------------------

struct AANoFreeImpl : public AANoFree {
  AANoFreeImpl(const IRPosition &IRP) : AANoFree(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({Attribute::NoFree}))
      indicateOptimisticFixpoint();
  }

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
using AANoFreeCallSite = AANoFreeFunction;

/// ------------------------ NonNull Argument Attribute ------------------------
struct AANonNullImpl : AANonNull {
  AANonNullImpl(const IRPosition &IRP) : AANonNull(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({Attribute::NonNull, Attribute::Dereferenceable}))
      indicateOptimisticFixpoint();
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
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(aligned) }
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

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({getAttrKind()})) {
      indicateOptimisticFixpoint();
      return;
    }
  }

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "norecurse" : "may-recurse";
  }
};

struct AANoRecurseFunction final : AANoRecurseImpl {
  AANoRecurseFunction(const IRPosition &IRP) : AANoRecurseImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Implement this.
    return indicatePessimisticFixpoint();
  }

  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(norecurse) }
};

using AANoRecurseCallSite = AANoRecurseFunction;

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
    if (hasAttr({Attribute::WillReturn})) {
      indicateOptimisticFixpoint();
      return;
    }

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
using AAWillReturnCallSite = AAWillReturnFunction;

/// ------------------------ NoAlias Argument Attribute ------------------------

struct AANoAliasImpl : AANoAlias {
  AANoAliasImpl(const IRPosition &IRP) : AANoAlias(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({Attribute::NoAlias}))
      indicateOptimisticFixpoint();
  }

  const std::string getAsStr() const override {
    return getAssumed() ? "noalias" : "may-alias";
  }
};

/// NoAlias attribute for a floating value.
struct AANoAliasFloating final : AANoAliasImpl {
  AANoAliasFloating(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

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
struct AANoAliasArgument final : AANoAliasImpl {
  AANoAliasArgument(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Implement this.
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(noalias) }
};

struct AANoAliasCallSiteArgument final : AANoAliasImpl {
  AANoAliasCallSiteArgument(const IRPosition &IRP) : AANoAliasImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Implement this.
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(noalias) }
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

      const auto &NoAliasAA =
          A.getAAFor<AANoAlias>(*this, IRPosition::callsite_returned(ICS));
      if (!NoAliasAA.isAssumedNoAlias())
        return false;

      /// FIXME: We can improve capture check in two ways:
      /// 1. Use the AANoCapture facilities.
      /// 2. Use the location of return insts for escape queries.
      if (PointerMayBeCaptured(&RV, /* ReturnCaptures */ false,
                               /* StoreCaptures */ true))
        return false;

      return true;
    };

    if (!A.checkForAllReturnedValues(CheckReturnValue, *this))
      return indicatePessimisticFixpoint();

    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(noalias) }
};

/// NoAlias attribute deduction for a call site return value.
using AANoAliasCallSiteReturned = AANoAliasReturned;

/// -------------------AAIsDead Function Attribute-----------------------

struct AAIsDeadImpl : public AAIsDead {
  AAIsDeadImpl(const IRPosition &IRP) : AAIsDead(IRP) {}

  void initialize(Attributor &A) override {
    const Function *F = getAssociatedFunction();

    if (F->hasInternalLinkage())
      return;

    if (!F || !F->hasExactDefinition()) {
      indicatePessimisticFixpoint();
      return;
    }

    exploreFromEntry(A, F);
  }

  void exploreFromEntry(Attributor &A, const Function *F) {
    ToBeExploredPaths.insert(&(F->getEntryBlock().front()));
    AssumedLiveBlocks.insert(&(F->getEntryBlock()));

    for (size_t i = 0; i < ToBeExploredPaths.size(); ++i)
      if (const Instruction *NextNoReturnI =
              findNextNoReturn(A, ToBeExploredPaths[i]))
        NoReturnCalls.insert(NextNoReturnI);
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
      F.replaceAllUsesWith(UndefValue::get(F.getType()));
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
      }

      BB = SplitPos->getParent();
      SplitBlock(BB, SplitPos);
      changeToUnreachable(BB->getTerminator(), /* UseLLVMTrap */ false);
      HasChanged = ChangeStatus::CHANGED;
    }

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
    STATS_DECL(DeadInternalFunction, Function,
               "Number of internal functions classified as dead (no live callsite)");
    BUILD_STAT_NAME(DeadInternalFunction, Function) +=
        (getAssociatedFunction()->hasInternalLinkage() &&
         AssumedLiveBlocks.empty())
            ? 1
            : 0;
    STATS_DECL(DeadBlocks, Function,
               "Number of basic blocks classified as dead");
    BUILD_STAT_NAME(DeadBlocks, Function) +=
        getAssociatedFunction()->size() - AssumedLiveBlocks.size();
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
          AssumedLiveBlocks.insert(Invoke->getUnwindDest());
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
    AssumedLiveBlocks.insert(SuccBB);
    ToBeExploredPaths.insert(&SuccBB->front());
  }

  // No noreturn instruction found.
  return nullptr;
}

ChangeStatus AAIsDeadImpl::updateImpl(Attributor &A) {
  const Function *F = getAssociatedFunction();
  ChangeStatus Status = ChangeStatus::UNCHANGED;

  if (F->hasInternalLinkage() && AssumedLiveBlocks.empty()) {
    auto CallSiteCheck = [&](CallSite) { return false; };

    // All callsites of F are dead.
    if (A.checkForAllCallSites(CallSiteCheck, *this, true))
      return ChangeStatus::UNCHANGED;

    // There exists at least one live call site, so we explore the function.
    Status = ChangeStatus::CHANGED;

    exploreFromEntry(A, F);
  }

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
  }

  return Status;
}

/// Liveness information for a call sites.
//
// TODO: Once we have call site specific value information we can provide call
//       site specific liveness liveness information and then it makes sense to
//       specialize attributes for call sites instead of redirecting requests to
//       the callee.
using AAIsDeadCallSite = AAIsDeadFunction;

/// -------------------- Dereferenceable Argument Attribute --------------------

struct DerefState : AbstractState {

  /// State representing for dereferenceable bytes.
  IntegerState DerefBytesState;

  /// State representing that whether the value is globaly dereferenceable.
  BooleanState GlobalState;

  /// See AbstractState::isValidState()
  bool isValidState() const override { return DerefBytesState.isValidState(); }

  /// See AbstractState::isAtFixpoint()
  bool isAtFixpoint() const override {
    return !isValidState() ||
           (DerefBytesState.isAtFixpoint() && GlobalState.isAtFixpoint());
  }

  /// See AbstractState::indicateOptimisticFixpoint(...)
  ChangeStatus indicateOptimisticFixpoint() override {
    DerefBytesState.indicateOptimisticFixpoint();
    GlobalState.indicateOptimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractState::indicatePessimisticFixpoint(...)
  ChangeStatus indicatePessimisticFixpoint() override {
    DerefBytesState.indicatePessimisticFixpoint();
    GlobalState.indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  /// Update known dereferenceable bytes.
  void takeKnownDerefBytesMaximum(uint64_t Bytes) {
    DerefBytesState.takeKnownMaximum(Bytes);
  }

  /// Update assumed dereferenceable bytes.
  void takeAssumedDerefBytesMinimum(uint64_t Bytes) {
    DerefBytesState.takeAssumedMinimum(Bytes);
  }

  /// Equality for DerefState.
  bool operator==(const DerefState &R) {
    return this->DerefBytesState == R.DerefBytesState &&
           this->GlobalState == R.GlobalState;
  }

  /// Inequality for IntegerState.
  bool operator!=(const DerefState &R) { return !(*this == R); }

  /// See IntegerState::operator^=
  DerefState operator^=(const DerefState &R) {
    DerefBytesState ^= R.DerefBytesState;
    GlobalState ^= R.GlobalState;
    return *this;
  }

  /// See IntegerState::operator&=
  DerefState operator&=(const DerefState &R) {
    DerefBytesState &= R.DerefBytesState;
    GlobalState &= R.GlobalState;
    return *this;
  }

  /// See IntegerState::operator|=
  DerefState operator|=(const DerefState &R) {
    DerefBytesState |= R.DerefBytesState;
    GlobalState |= R.GlobalState;
    return *this;
  }
};

template <>
ChangeStatus clampStateAndIndicateChange<DerefState>(DerefState &S,
                                                     const DerefState &R) {
  ChangeStatus CS0 = clampStateAndIndicateChange<IntegerState>(
      S.DerefBytesState, R.DerefBytesState);
  ChangeStatus CS1 =
      clampStateAndIndicateChange<IntegerState>(S.GlobalState, R.GlobalState);
  return CS0 | CS1;
}

struct AADereferenceableImpl : AADereferenceable, DerefState {
  AADereferenceableImpl(const IRPosition &IRP) : AADereferenceable(IRP) {}
  using StateType = DerefState;

  void initialize(Attributor &A) override {
    SmallVector<Attribute, 4> Attrs;
    getAttrs({Attribute::Dereferenceable, Attribute::DereferenceableOrNull},
             Attrs);
    for (const Attribute &Attr : Attrs)
      takeKnownDerefBytesMaximum(Attr.getValueAsInt());

    NonNullAA = &A.getAAFor<AANonNull>(*this, getIRPosition());
  }

  /// See AbstractAttribute::getState()
  /// {
  StateType &getState() override { return *this; }
  const StateType &getState() const override { return *this; }
  /// }

  /// See AADereferenceable::getAssumedDereferenceableBytes().
  uint32_t getAssumedDereferenceableBytes() const override {
    return DerefBytesState.getAssumed();
  }

  /// See AADereferenceable::getKnownDereferenceableBytes().
  uint32_t getKnownDereferenceableBytes() const override {
    return DerefBytesState.getKnown();
  }

  /// See AADereferenceable::isAssumedGlobal().
  bool isAssumedGlobal() const override { return GlobalState.getAssumed(); }

  /// See AADereferenceable::isKnownGlobal().
  bool isKnownGlobal() const override { return GlobalState.getKnown(); }

  bool isAssumedNonNull() const override {
    return NonNullAA && NonNullAA->isAssumedNonNull();
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

private:
  const AANonNull *NonNullAA = nullptr;
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

      T.takeAssumedDerefBytesMinimum(
          std::max(int64_t(0), DerefBytes - Offset.getSExtValue()));

      if (!Stripped && this == &AA) {
        T.takeKnownDerefBytesMaximum(
            std::max(int64_t(0), DerefBytes - Offset.getSExtValue()));
        T.indicatePessimisticFixpoint();
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
  void trackStatistics() const override{
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
using AADereferenceableCallSiteReturned = AADereferenceableReturned;

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
  }

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

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(aligned) }
};

/// Align attribute deduction for a call site return value.
using AAAlignCallSiteReturned = AAAlignReturned;

/// ------------------ Function No-Return Attribute ----------------------------
struct AANoReturnImpl : public AANoReturn {
  AANoReturnImpl(const IRPosition &IRP) : AANoReturn(IRP) {}

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "noreturn" : "may-return";
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (hasAttr({getAttrKind()}))
      indicateOptimisticFixpoint();
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
using AANoReturnCallSite = AANoReturnFunction;

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
        &getAAFor<AAIsDead>(AA, IRPosition::function(*CtxI->getFunction()));

  // Don't check liveness for AAIsDead.
  if (&AA == LivenessAA)
    return false;

  if (!LivenessAA->isAssumedDead(CtxI))
    return false;

  // TODO: Do not track dependences automatically but add it here as only a
  //       "is-assumed-dead" result causes a dependence.
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
    Instruction *I = cast<Instruction>(U.getUser());
    Function *Caller = I->getFunction();

    const auto &LivenessAA =
        getAAFor<AAIsDead>(QueryingAA, IRPosition::function(*Caller));

    // Skip dead calls.
    if (LivenessAA.isAssumedDead(I))
      continue;

    CallSite CS(U.getUser());
    if (!CS || !CS.isCallee(&U) || !CS.getCaller()->hasExactDefinition()) {
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
    const function_ref<bool(Value &, const SmallPtrSetImpl<ReturnInst *> &)>
        &Pred,
    const AbstractAttribute &QueryingAA) {

  const IRPosition &IRP = QueryingAA.getIRPosition();
  // Since we need to provide return instructions we have to have an exact
  // definition.
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction || !AssociatedFunction->hasExactDefinition())
    return false;

  // If this is a call site query we use the call site specific return values
  // and liveness information.
  const IRPosition &QueryIRP = IRPosition::function_scope(IRP);
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
  if (!AssociatedFunction || !AssociatedFunction->hasExactDefinition())
    return false;

  const IRPosition &QueryIRP = IRPosition::function_scope(IRP);
  const auto &AARetVal = getAAFor<AAReturnedValues>(QueryingAA, QueryIRP);
  if (!AARetVal.getState().isValidState())
    return false;

  return AARetVal.checkForAllReturnedValuesAndReturnInsts(
      [&](Value &RV, const SmallPtrSetImpl<ReturnInst *> &) {
        return Pred(RV);
      });
}

bool Attributor::checkForAllInstructions(
    const llvm::function_ref<bool(Instruction &)> &Pred,
    const AbstractAttribute &QueryingAA, const ArrayRef<unsigned> &Opcodes) {

  const IRPosition &IRP = QueryingAA.getIRPosition();
  // Since we need to provide instructions we have to have an exact definition.
  const Function *AssociatedFunction = IRP.getAssociatedFunction();
  if (!AssociatedFunction || !AssociatedFunction->hasExactDefinition())
    return false;

  const IRPosition &QueryIRP = IRPosition::function_scope(IRP);
  const auto &LivenessAA = getAAFor<AAIsDead>(QueryingAA, QueryIRP);

  auto &OpcodeInstMap =
      InfoCache.getOpcodeInstMapForFunction(*AssociatedFunction);
  for (unsigned Opcode : Opcodes) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      // Skip dead instructions.
      if (LivenessAA.isAssumedDead(I))
        continue;

      if (!Pred(*I))
        return false;
    }
  }

  return true;
}

bool Attributor::checkForAllReadWriteInstructions(
    const llvm::function_ref<bool(Instruction &)> &Pred,
    AbstractAttribute &QueryingAA) {

  const Function *AssociatedFunction =
      QueryingAA.getIRPosition().getAssociatedFunction();
  if (!AssociatedFunction)
    return false;

  const auto &LivenessAA =
      getAAFor<AAIsDead>(QueryingAA, QueryingAA.getIRPosition());

  for (Instruction *I :
       InfoCache.getReadOrWriteInstsForFunction(*AssociatedFunction)) {
    // Skip dead instructions.
    if (LivenessAA.isAssumedDead(I))
      continue;

    if (!Pred(*I))
      return false;
  }

  return true;
}

ChangeStatus Attributor::run() {
  // Initialize all abstract attributes, allow new ones to be created.
  for (unsigned u = 0; u < AllAbstractAttributes.size(); u++)
    AllAbstractAttributes[u]->initialize(*this);

  LLVM_DEBUG(dbgs() << "[Attributor] Identified and initialized "
                    << AllAbstractAttributes.size()
                    << " abstract attributes.\n");

  // Now that all abstract attributes are collected and initialized we start
  // the abstract analysis.

  unsigned IterationCounter = 1;

  SmallVector<AbstractAttribute *, 64> ChangedAAs;
  SetVector<AbstractAttribute *> Worklist;
  Worklist.insert(AllAbstractAttributes.begin(), AllAbstractAttributes.end());

  do {
    // Remember the size to determine new attributes.
    size_t NumAAs = AllAbstractAttributes.size();
    LLVM_DEBUG(dbgs() << "\n\n[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist size: " << Worklist.size() << "\n");

    // Add all abstract attributes that are potentially dependent on one that
    // changed to the work list.
    for (AbstractAttribute *ChangedAA : ChangedAAs) {
      auto &QuerriedAAs = QueryMap[ChangedAA];
      Worklist.insert(QuerriedAAs.begin(), QuerriedAAs.end());
    }

    // Reset the changed set.
    ChangedAAs.clear();

    // Update all abstract attribute in the work list and record the ones that
    // changed.
    for (AbstractAttribute *AA : Worklist)
      if (!isAssumedDead(*AA, nullptr))
        if (AA->update(*this) == ChangeStatus::CHANGED)
          ChangedAAs.push_back(AA);

    // Reset the work list and repopulate with the changed abstract attributes.
    // Note that dependent ones are added above.
    Worklist.clear();
    Worklist.insert(ChangedAAs.begin(), ChangedAAs.end());

    // Add attributes to the worklist that have been created in the last
    // iteration.
    Worklist.insert(AllAbstractAttributes.begin() + NumAAs,
                    AllAbstractAttributes.end());

  } while (!Worklist.empty() && ++IterationCounter < MaxFixpointIterations);

  size_t NumFinalAAs = AllAbstractAttributes.size();

  LLVM_DEBUG(dbgs() << "\n[Attributor] Fixpoint iteration done after: "
                    << IterationCounter << "/" << MaxFixpointIterations
                    << " iterations\n");

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
    ChangeStatus VerifyStatus = run();
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
  return ManifestChange;
}

/// Helper function that checks if an abstract attribute of type \p AAType
/// should be created for IR position \p IRP and if so creates and registers it
/// with the Attributor \p A.
///
/// This method will look at the provided whitelist. If one is given and the
/// kind \p AAType::ID is not contained, no abstract attribute is created.
///
/// \returns The created abstract argument, or nullptr if none was created.
template <typename AAType>
static const AAType *checkAndRegisterAA(const IRPosition &IRP, Attributor &A,
                                        DenseSet<const char *> *Whitelist) {
  if (Whitelist && !Whitelist->count(&AAType::ID))
    return nullptr;

  return &A.registerAA<AAType>(*new AAType(IRP));
}

void Attributor::identifyDefaultAbstractAttributes(
    Function &F, DenseSet<const char *> *Whitelist) {

  IRPosition FPos = IRPosition::function(F);

  // Check for dead BasicBlocks in every function.
  // We need dead instruction detection because we do not want to deal with
  // broken IR in which SSA rules do not apply.
  checkAndRegisterAA<AAIsDeadFunction>(FPos, *this, /* Whitelist */ nullptr);

  // Every function might be "will-return".
  checkAndRegisterAA<AAWillReturnFunction>(FPos, *this, Whitelist);

  // Every function can be nounwind.
  checkAndRegisterAA<AANoUnwindFunction>(FPos, *this, Whitelist);

  // Every function might be marked "nosync"
  checkAndRegisterAA<AANoSyncFunction>(FPos, *this, Whitelist);

  // Every function might be "no-free".
  checkAndRegisterAA<AANoFreeFunction>(FPos, *this, Whitelist);

  // Every function might be "no-return".
  checkAndRegisterAA<AANoReturnFunction>(FPos, *this, Whitelist);

  // Return attributes are only appropriate if the return type is non void.
  Type *ReturnType = F.getReturnType();
  if (!ReturnType->isVoidTy()) {
    // Argument attribute "returned" --- Create only one per function even
    // though it is an argument attribute.
    checkAndRegisterAA<AAReturnedValuesFunction>(FPos, *this, Whitelist);

    if (ReturnType->isPointerTy()) {
      IRPosition RetPos = IRPosition::returned(F);

      // Every function with pointer return type might be marked align.
      checkAndRegisterAA<AAAlignReturned>(RetPos, *this, Whitelist);

      // Every function with pointer return type might be marked nonnull.
      checkAndRegisterAA<AANonNullReturned>(RetPos, *this, Whitelist);

      // Every function with pointer return type might be marked noalias.
      checkAndRegisterAA<AANoAliasReturned>(RetPos, *this, Whitelist);

      // Every function with pointer return type might be marked
      // dereferenceable.
      checkAndRegisterAA<AADereferenceableReturned>(RetPos, *this, Whitelist);
    }
  }

  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      IRPosition ArgPos = IRPosition::argument(Arg);
      // Every argument with pointer type might be marked nonnull.
      checkAndRegisterAA<AANonNullArgument>(ArgPos, *this, Whitelist);

      // Every argument with pointer type might be marked dereferenceable.
      checkAndRegisterAA<AADereferenceableArgument>(ArgPos, *this, Whitelist);

      // Every argument with pointer type might be marked align.
      checkAndRegisterAA<AAAlignArgument>(ArgPos, *this, Whitelist);
    }
  }

  // Walk all instructions to find more attribute opportunities and also
  // interesting instructions that might be queried by abstract attributes
  // during their initialization or update.
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
             "attributor.");
      break;
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

    CallSite CS(&I);
    if (CS && CS.getCalledFunction()) {
      for (int i = 0, e = CS.getCalledFunction()->arg_size(); i < e; i++) {
        if (!CS.getArgument(i)->getType()->isPointerTy())
          continue;
        IRPosition CSArgPos = IRPosition::callsite_argument(CS, i);

        // Call site argument attribute "non-null".
        checkAndRegisterAA<AANonNullCallSiteArgument>(CSArgPos, *this,
                                                      Whitelist);

        // Call site argument attribute "dereferenceable".
        checkAndRegisterAA<AADereferenceableCallSiteArgument>(CSArgPos, *this,
                                                              Whitelist);

        // Call site argument attribute "align".
        checkAndRegisterAA<AAAlignCallSiteArgument>(CSArgPos, *this, Whitelist);
      }
    }
  }
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

static bool runAttributorOnModule(Module &M) {
  if (DisableAttributor)
    return false;

  LLVM_DEBUG(dbgs() << "[Attributor] Run on module with " << M.size()
                    << " functions.\n");

  // Create an Attributor and initially empty information cache that is filled
  // while we identify default attribute opportunities.
  InformationCache InfoCache(M.getDataLayout());
  Attributor A(InfoCache);

  for (Function &F : M) {
    // TODO: Not all attributes require an exact definition. Find a way to
    //       enable deduction for some but not all attributes in case the
    //       definition might be changed at runtime, see also
    //       http://lists.llvm.org/pipermail/llvm-dev/2018-February/121275.html.
    // TODO: We could always determine abstract attributes and if sufficient
    //       information was found we could duplicate the functions that do not
    //       have an exact definition.
    if (!F.hasExactDefinition()) {
      NumFnWithoutExactDefinition++;
      continue;
    }

    // For now we ignore naked and optnone functions.
    if (F.hasFnAttribute(Attribute::Naked) ||
        F.hasFnAttribute(Attribute::OptimizeNone))
      continue;

    NumFnWithExactDefinition++;

    // Populate the Attributor with abstract attribute opportunities in the
    // function and the information cache with IR information.
    A.identifyDefaultAbstractAttributes(F);
  }

  return A.run() == ChangeStatus::CHANGED;
}

PreservedAnalyses AttributorPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (runAttributorOnModule(M)) {
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
    return runAttributorOnModule(M);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // FIXME: Think about passes we will preserve and add them here.
    AU.setPreservesCFG();
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
    AA->initialize(A);                                                         \
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

#undef CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef SWITCH_PK_CREATE
#undef SWITCH_PK_INV

INITIALIZE_PASS_BEGIN(AttributorLegacyPass, "attributor",
                      "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_END(AttributorLegacyPass, "attributor",
                    "Deduce and propagate attributes", false, false)
