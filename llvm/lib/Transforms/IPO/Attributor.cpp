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
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
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
STATISTIC(NumAttributesFixedDueToRequiredDependences,
          "Number of abstract attributes fixed due to required dependences");

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
PIPE_OPERATOR(AAValueConstantRange)

#undef PIPE_OPERATOR
} // namespace llvm

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

static cl::opt<bool> AnnotateDeclarationCallSites(
    "attributor-annotate-decl-cs", cl::Hidden,
    cl::desc("Annotate call sites of function declarations."), cl::init(false));

static cl::opt<bool> ManifestInternal(
    "attributor-manifest-internal", cl::Hidden,
    cl::desc("Manifest Attributor internal string attributes."),
    cl::init(false));

static cl::opt<unsigned> DepRecInterval(
    "attributor-dependence-recompute-interval", cl::Hidden,
    cl::desc("Number of iterations until dependences are recomputed."),
    cl::init(4));

static cl::opt<bool> EnableHeapToStack("enable-heap-to-stack-conversion",
                                       cl::init(true), cl::Hidden);

static cl::opt<int> MaxHeapToStackSize("max-heap-to-stack-size", cl::init(128),
                                       cl::Hidden);

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

Argument *IRPosition::getAssociatedArgument() const {
  if (getPositionKind() == IRP_ARGUMENT)
    return cast<Argument>(&getAnchorValue());

  // Not an Argument and no argument number means this is not a call site
  // argument, thus we cannot find a callback argument to return.
  int ArgNo = getArgNo();
  if (ArgNo < 0)
    return nullptr;

  // Use abstract call sites to make the connection between the call site
  // values and the ones in callbacks. If a callback was found that makes use
  // of the underlying call site operand, we want the corresponding callback
  // callee argument and not the direct callee argument.
  Optional<Argument *> CBCandidateArg;
  SmallVector<const Use *, 4> CBUses;
  ImmutableCallSite ICS(&getAnchorValue());
  AbstractCallSite::getCallbackUses(ICS, CBUses);
  for (const Use *U : CBUses) {
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
      if (CBCandidateArg.hasValue()) {
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
  const Function *Callee = ICS.getCalledFunction();
  if (Callee && Callee->arg_size() > unsigned(ArgNo))
    return Callee->getArg(ArgNo);

  return nullptr;
}

/// For calls (and invokes) we will only replace instruction uses to not disturb
/// the old style call graph.
/// TODO: Remove this once we get rid of the old PM.
static void replaceAllInstructionUsesWith(Value &Old, Value &New) {
  if (!isa<CallBase>(Old))
    return Old.replaceAllUsesWith(&New);
  SmallVector<Use *, 8> Uses;
  for (Use &U : Old.uses())
    if (isa<Instruction>(U.getUser()))
      Uses.push_back(&U);
  for (Use *U : Uses)
    U->set(&New);
}

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
    A.recordDependence(*LivenessAA, QueryingAA, DepClassTy::OPTIONAL);

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

static const Value *
getBasePointerOfAccessPointerOperand(const Instruction *I, int64_t &BytesOffset,
                                     const DataLayout &DL,
                                     bool AllowNonInbounds = false) {
  const Value *Ptr =
      Attributor::getPointerOperand(I, /* AllowVolatile */ false);
  if (!Ptr)
    return nullptr;

  return GetPointerBaseWithConstantOffset(Ptr, BytesOffset, DL,
                                          AllowNonInbounds);
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

bool IRPosition::hasAttr(ArrayRef<Attribute::AttrKind> AKs,
                         bool IgnoreSubsumingPositions) const {
  for (const IRPosition &EquivIRP : SubsumingPositionIterator(*this)) {
    for (Attribute::AttrKind AK : AKs)
      if (EquivIRP.getAttr(AK).getKindAsEnum() == AK)
        return true;
    // The first position returned by the SubsumingPositionIterator is
    // always the position itself. If we ignore subsuming positions we
    // are done after the first iteration.
    if (IgnoreSubsumingPositions)
      break;
  }
  return false;
}

void IRPosition::getAttrs(ArrayRef<Attribute::AttrKind> AKs,
                          SmallVectorImpl<Attribute> &Attrs,
                          bool IgnoreSubsumingPositions) const {
  for (const IRPosition &EquivIRP : SubsumingPositionIterator(*this)) {
    for (Attribute::AttrKind AK : AKs) {
      const Attribute &Attr = EquivIRP.getAttr(AK);
      if (Attr.getKindAsEnum() == AK)
        Attrs.push_back(Attr);
    }
    // The first position returned by the SubsumingPositionIterator is
    // always the position itself. If we ignore subsuming positions we
    // are done after the first iteration.
    if (IgnoreSubsumingPositions)
      break;
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
static void clampReturnedValueStates(Attributor &A, const AAType &QueryingAA,
                                     StateType &S) {
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

/// Helper class to compose two generic deduction
template <typename AAType, typename Base, typename StateType,
          template <typename...> class F, template <typename...> class G>
struct AAComposeTwoGenericDeduction
    : public F<AAType, G<AAType, Base, StateType>, StateType> {
  AAComposeTwoGenericDeduction(const IRPosition &IRP)
      : F<AAType, G<AAType, Base, StateType>, StateType>(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus ChangedF =
        F<AAType, G<AAType, Base, StateType>, StateType>::updateImpl(A);
    ChangeStatus ChangedG = G<AAType, Base, StateType>::updateImpl(A);
    return ChangedF | ChangedG;
  }
};

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
                    << QueryingAA << " into " << S << "\n");

  assert(QueryingAA.getIRPosition().getPositionKind() ==
             IRPosition::IRP_ARGUMENT &&
         "Can only clamp call site argument states for an argument position!");

  // Use an optional state as there might not be any return values and we want
  // to join (IntegerState::operator&) the state of all there are.
  Optional<StateType> T;

  // The argument number which is also the call site argument number.
  unsigned ArgNo = QueryingAA.getIRPosition().getArgNo();

  auto CallSiteCheck = [&](AbstractCallSite ACS) {
    const IRPosition &ACSArgPos = IRPosition::callsite_argument(ACS, ArgNo);
    // Check if a coresponding argument was found or if it is on not associated
    // (which can happen for callback calls).
    if (ACSArgPos.getPositionKind() == IRPosition::IRP_INVALID)
      return false;

    const AAType &AA = A.getAAFor<AAType>(QueryingAA, ACSArgPos);
    LLVM_DEBUG(dbgs() << "[Attributor] ACS: " << *ACS.getInstruction()
                      << " AA: " << AA.getAsStr() << " @" << ACSArgPos << "\n");
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
template <typename AAType, typename Base,
          typename StateType = typename AAType::StateType>
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

/// Helper class for generic deduction using must-be-executed-context
/// Base class is required to have `followUse` method.

/// bool followUse(Attributor &A, const Use *U, const Instruction *I)
/// U - Underlying use.
/// I - The user of the \p U.
/// `followUse` returns true if the value should be tracked transitively.

template <typename AAType, typename Base,
          typename StateType = typename AAType::StateType>
struct AAFromMustBeExecutedContext : public Base {
  AAFromMustBeExecutedContext(const IRPosition &IRP) : Base(IRP) {}

  void initialize(Attributor &A) override {
    Base::initialize(A);
    const IRPosition &IRP = this->getIRPosition();
    Instruction *CtxI = IRP.getCtxI();

    if (!CtxI)
      return;

    for (const Use &U : IRP.getAssociatedValue().uses())
      Uses.insert(&U);
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto BeforeState = this->getState();
    auto &S = this->getState();
    Instruction *CtxI = this->getIRPosition().getCtxI();
    if (!CtxI)
      return ChangeStatus::UNCHANGED;

    MustBeExecutedContextExplorer &Explorer =
        A.getInfoCache().getMustBeExecutedContextExplorer();

    auto EIt = Explorer.begin(CtxI), EEnd = Explorer.end(CtxI);
    for (unsigned u = 0; u < Uses.size(); ++u) {
      const Use *U = Uses[u];
      if (const Instruction *UserI = dyn_cast<Instruction>(U->getUser())) {
        bool Found = Explorer.findInContextOf(UserI, EIt, EEnd);
        if (Found && Base::followUse(A, U, UserI))
          for (const Use &Us : UserI->uses())
            Uses.insert(&Us);
      }
    }

    return BeforeState == S ? ChangeStatus::UNCHANGED : ChangeStatus::CHANGED;
  }

private:
  /// Container for (transitive) uses of the associated value.
  SetVector<const Use *> Uses;
};

template <typename AAType, typename Base,
          typename StateType = typename AAType::StateType>
using AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext =
    AAComposeTwoGenericDeduction<AAType, Base, StateType,
                                 AAFromMustBeExecutedContext,
                                 AAArgumentFromCallSiteArguments>;

template <typename AAType, typename Base,
          typename StateType = typename AAType::StateType>
using AACallSiteReturnedFromReturnedAndMustBeExecutedContext =
    AAComposeTwoGenericDeduction<AAType, Base, StateType,
                                 AAFromMustBeExecutedContext,
                                 AACallSiteReturnedFromReturned>;

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
    if (CB.getNumUses() == 0 || CB.isMustTailCall())
      return ChangeStatus::UNCHANGED;
    replaceAllInstructionUsesWith(CB, C);
    return ChangeStatus::CHANGED;
  };

  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.getValue())) {
    // TODO: This should be handled differently!
    this->AnchorVal = UniqueRVArg;
    this->KindOrArgNo = UniqueRVArg->getArgNo();
    Changed = IRAttribute::manifest(A);
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
                      << RetValAA << "\n");

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
    /// FIXME: We should improve the handling of intrinsics.

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

/// NoFree attribute for floating values.
struct AANoFreeFloating : AANoFreeImpl {
  AANoFreeFloating(const IRPosition &IRP) : AANoFreeImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override{STATS_DECLTRACK_FLOATING_ATTR(nofree)}

  /// See Abstract Attribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const IRPosition &IRP = getIRPosition();

    const auto &NoFreeAA =
        A.getAAFor<AANoFree>(*this, IRPosition::function_scope(IRP));
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
            *this, IRPosition::callsite_argument(*CB, ArgNo));
        return NoFreeArg.isAssumedNoFree();
      }

      if (isa<GetElementPtrInst>(UserI) || isa<BitCastInst>(UserI) ||
          isa<PHINode>(UserI) || isa<SelectInst>(UserI)) {
        Follow = true;
        return true;
      }

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
  AANoFreeArgument(const IRPosition &IRP) : AANoFreeFloating(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nofree) }
};

/// NoFree attribute for call site arguments.
struct AANoFreeCallSiteArgument final : AANoFreeFloating {
  AANoFreeCallSiteArgument(const IRPosition &IRP) : AANoFreeFloating(IRP) {}

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
    auto &ArgAA = A.getAAFor<AANoFree>(*this, ArgPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const AANoFree::StateType &>(ArgAA.getState()));
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override{STATS_DECLTRACK_CSARG_ATTR(nofree)};
};

/// NoFree attribute for function return value.
struct AANoFreeReturned final : AANoFreeFloating {
  AANoFreeReturned(const IRPosition &IRP) : AANoFreeFloating(IRP) {
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
  AANoFreeCallSiteReturned(const IRPosition &IRP) : AANoFreeFloating(IRP) {}

  ChangeStatus manifest(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }
  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(nofree) }
};

/// ------------------------ NonNull Argument Attribute ------------------------
static int64_t getKnownNonNullAndDerefBytesForUse(
    Attributor &A, AbstractAttribute &QueryingAA, Value &AssociatedValue,
    const Use *U, const Instruction *I, bool &IsNonNull, bool &TrackUse) {
  TrackUse = false;

  const Value *UseV = U->get();
  if (!UseV->getType()->isPointerTy())
    return 0;

  Type *PtrTy = UseV->getType();
  const Function *F = I->getFunction();
  bool NullPointerIsDefined =
      F ? llvm::NullPointerIsDefined(F, PtrTy->getPointerAddressSpace()) : true;
  const DataLayout &DL = A.getInfoCache().getDL();
  if (ImmutableCallSite ICS = ImmutableCallSite(I)) {
    if (ICS.isBundleOperand(U))
      return 0;

    if (ICS.isCallee(U)) {
      IsNonNull |= !NullPointerIsDefined;
      return 0;
    }

    unsigned ArgNo = ICS.getArgumentNo(U);
    IRPosition IRP = IRPosition::callsite_argument(ICS, ArgNo);
    // As long as we only use known information there is no need to track
    // dependences here.
    auto &DerefAA = A.getAAFor<AADereferenceable>(QueryingAA, IRP,
                                                  /* TrackDependence */ false);
    IsNonNull |= DerefAA.isKnownNonNull();
    return DerefAA.getKnownDereferenceableBytes();
  }

  // We need to follow common pointer manipulation uses to the accesses they
  // feed into. We can try to be smart to avoid looking through things we do not
  // like for now, e.g., non-inbounds GEPs.
  if (isa<CastInst>(I)) {
    TrackUse = true;
    return 0;
  }
  if (auto *GEP = dyn_cast<GetElementPtrInst>(I))
    if (GEP->hasAllConstantIndices()) {
      TrackUse = true;
      return 0;
    }

  int64_t Offset;
  if (const Value *Base = getBasePointerOfAccessPointerOperand(I, Offset, DL)) {
    if (Base == &AssociatedValue &&
        Attributor::getPointerOperand(I, /* AllowVolatile */ false) == UseV) {
      int64_t DerefBytes =
          (int64_t)DL.getTypeStoreSize(PtrTy->getPointerElementType()) + Offset;

      IsNonNull |= !NullPointerIsDefined;
      return std::max(int64_t(0), DerefBytes);
    }
  }

  /// Corner case when an offset is 0.
  if (const Value *Base = getBasePointerOfAccessPointerOperand(
          I, Offset, DL, /*AllowNonInbounds*/ true)) {
    if (Offset == 0 && Base == &AssociatedValue &&
        Attributor::getPointerOperand(I, /* AllowVolatile */ false) == UseV) {
      int64_t DerefBytes =
          (int64_t)DL.getTypeStoreSize(PtrTy->getPointerElementType());
      IsNonNull |= !NullPointerIsDefined;
      return std::max(int64_t(0), DerefBytes);
    }
  }

  return 0;
}

struct AANonNullImpl : AANonNull {
  AANonNullImpl(const IRPosition &IRP)
      : AANonNull(IRP),
        NullIsDefined(NullPointerIsDefined(
            getAnchorScope(),
            getAssociatedValue().getType()->getPointerAddressSpace())) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (!NullIsDefined &&
        hasAttr({Attribute::NonNull, Attribute::Dereferenceable}))
      indicateOptimisticFixpoint();
    else if (isa<ConstantPointerNull>(getAssociatedValue()))
      indicatePessimisticFixpoint();
    else
      AANonNull::initialize(A);
  }

  /// See AAFromMustBeExecutedContext
  bool followUse(Attributor &A, const Use *U, const Instruction *I) {
    bool IsNonNull = false;
    bool TrackUse = false;
    getKnownNonNullAndDerefBytesForUse(A, *this, getAssociatedValue(), U, I,
                                       IsNonNull, TrackUse);
    setKnown(IsNonNull);
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
struct AANonNullFloating
    : AAFromMustBeExecutedContext<AANonNull, AANonNullImpl> {
  using Base = AAFromMustBeExecutedContext<AANonNull, AANonNullImpl>;
  AANonNullFloating(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = Base::updateImpl(A);
    if (isKnownNonNull())
      return Change;

    if (!NullIsDefined) {
      const auto &DerefAA =
          A.getAAFor<AADereferenceable>(*this, getIRPosition());
      if (DerefAA.getAssumedDereferenceableBytes())
        return Change;
    }

    const DataLayout &DL = A.getDataLayout();

    DominatorTree *DT = nullptr;
    InformationCache &InfoCache = A.getInfoCache();
    if (const Function *Fn = getAnchorScope())
      DT = InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(*Fn);

    auto VisitValueCB = [&](Value &V, AANonNull::StateType &T,
                            bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AANonNull>(*this, IRPosition::value(V));
      if (!Stripped && this == &AA) {
        if (!isKnownNonZero(&V, DL, 0, /* TODO: AC */ nullptr, getCtxI(), DT))
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
    : AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext<AANonNull,
                                                              AANonNullImpl> {
  AANonNullArgument(const IRPosition &IRP)
      : AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext<AANonNull,
                                                                AANonNullImpl>(
            IRP) {}

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
    : AACallSiteReturnedFromReturnedAndMustBeExecutedContext<AANonNull,
                                                             AANonNullImpl> {
  AANonNullCallSiteReturned(const IRPosition &IRP)
      : AACallSiteReturnedFromReturnedAndMustBeExecutedContext<AANonNull,
                                                               AANonNullImpl>(
            IRP) {}

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

/// -------------------- Undefined-Behavior Attributes ------------------------

struct AAUndefinedBehaviorImpl : public AAUndefinedBehavior {
  AAUndefinedBehaviorImpl(const IRPosition &IRP) : AAUndefinedBehavior(IRP) {}

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
      const Value *PtrOp =
          Attributor::getPointerOperand(&I, /* AllowVolatile */ true);
      assert(PtrOp &&
             "Expected pointer operand of memory accessing instruction");

      // A memory access through a pointer is considered UB
      // only if the pointer has constant null value.
      // TODO: Expand it to not only check constant values.
      if (!isa<ConstantPointerNull>(PtrOp)) {
        AssumedNoUBInsts.insert(&I);
        return true;
      }
      const Type *PtrTy = PtrOp->getType();

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

    A.checkForAllInstructions(InspectMemAccessInstForUB, *this,
                              {Instruction::Load, Instruction::Store,
                               Instruction::AtomicCmpXchg,
                               Instruction::AtomicRMW});
    A.checkForAllInstructions(InspectBrInstForUB, *this, {Instruction::Br});
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
    const auto &ValueSimplifyAA =
        A.getAAFor<AAValueSimplify>(*this, IRPosition::value(*V));
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
  AAUndefinedBehaviorFunction(const IRPosition &IRP)
      : AAUndefinedBehaviorImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECL(UndefinedBehaviorInstruction, Instruction,
               "Number of instructions known to have UB");
    BUILD_STAT_NAME(UndefinedBehaviorInstruction, Instruction) +=
        KnownUBInsts.size();
  }
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

/// -------------------AAReachability Attribute--------------------------

struct AAReachabilityImpl : AAReachability {
  AAReachabilityImpl(const IRPosition &IRP) : AAReachability(IRP) {}

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
  AAReachabilityFunction(const IRPosition &IRP) : AAReachabilityImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(reachable); }
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
    Value &Val = getAssociatedValue();
    if (isa<AllocaInst>(Val))
      indicateOptimisticFixpoint();
    if (isa<ConstantPointerNull>(Val) &&
        Val.getType()->getPointerAddressSpace() == 0)
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
  using Base = AAArgumentFromCallSiteArguments<AANoAlias, AANoAliasImpl>;
  AANoAliasArgument(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::update(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // We have to make sure no-alias on the argument does not break
    // synchronization when this is a callback argument, see also [1] below.
    // If synchronization cannot be affected, we delegate to the base updateImpl
    // function, otherwise we give up for now.

    // If the function is no-sync, no-alias cannot break synchronization.
    const auto &NoSyncAA = A.getAAFor<AANoSync>(
        *this, IRPosition::function_scope(getIRPosition()));
    if (NoSyncAA.isAssumedNoSync())
      return Base::updateImpl(A);

    // If the argument is read-only, no-alias cannot break synchronization.
    const auto &MemBehaviorAA =
        A.getAAFor<AAMemoryBehavior>(*this, getIRPosition());
    if (MemBehaviorAA.isAssumedReadOnly())
      return Base::updateImpl(A);

    // If the argument is never passed through callbacks, no-alias cannot break
    // synchronization.
    if (A.checkForAllCallSites(
            [](AbstractCallSite ACS) { return !ACS.isCallbackCall(); }, *this,
            true))
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
    LLVM_DEBUG(dbgs() << "[Attributor][AANoAliasCSArg] check definition: " << V
                      << " :: " << NoAliasAA << "\n");

    if (!NoAliasAA.isAssumedNoAlias())
      return indicatePessimisticFixpoint();

    LLVM_DEBUG(dbgs() << "[Attributor][AANoAliasCSArg] " << V
                      << " is assumed NoAlias in the definition\n");

    // (ii) Check whether the value is captured in the scope using AANoCapture.
    //      FIXME: This is conservative though, it is better to look at CFG and
    //             check only uses possibly executed before this callsite.

    auto &NoCaptureAA = A.getAAFor<AANoCapture>(*this, IRP);
    if (!NoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
      LLVM_DEBUG(
          dbgs() << "[Attributor][AANoAliasCSArg] " << V
                 << " cannot be noalias as it is potentially captured\n");
      return indicatePessimisticFixpoint();
    }

    // (iii) Check there is no other pointer argument which could alias with the
    // value.
    // TODO: AbstractCallSite
    ImmutableCallSite ICS(&getAnchorValue());
    for (unsigned i = 0; i < ICS.getNumArgOperands(); i++) {
      if (getArgNo() == (int)i)
        continue;
      const Value *ArgOp = ICS.getArgOperand(i);
      if (!ArgOp->getType()->isPointerTy())
        continue;

      if (const Function *F = getAnchorScope()) {
        if (AAResults *AAR = A.getInfoCache().getAAResultsForFunction(*F)) {
          bool IsAliasing = !AAR->isNoAlias(&getAssociatedValue(), ArgOp);
          LLVM_DEBUG(dbgs()
                     << "[Attributor][NoAliasCSArg] Check alias between "
                        "callsite arguments "
                     << AAR->isNoAlias(&getAssociatedValue(), ArgOp) << " "
                     << getAssociatedValue() << " " << *ArgOp << " => "
                     << (IsAliasing ? "" : "no-") << "alias \n");

          if (!IsAliasing)
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

struct AAIsDeadValueImpl : public AAIsDead {
  AAIsDeadValueImpl(const IRPosition &IRP) : AAIsDead(IRP) {}

  /// See AAIsDead::isAssumedDead().
  bool isAssumedDead() const override { return getAssumed(); }

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
    return I == getCtxI() && getKnown();
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return isAssumedDead() ? "assumed-dead" : "assumed-live";
  }
};

struct AAIsDeadFloating : public AAIsDeadValueImpl {
  AAIsDeadFloating(const IRPosition &IRP) : AAIsDeadValueImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (Instruction *I = dyn_cast<Instruction>(&getAssociatedValue()))
      if (!wouldInstructionBeTriviallyDead(I))
        indicatePessimisticFixpoint();
    if (isa<UndefValue>(getAssociatedValue()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto UsePred = [&](const Use &U, bool &Follow) {
      Instruction *UserI = cast<Instruction>(U.getUser());
      if (CallSite CS = CallSite(UserI)) {
        if (!CS.isArgOperand(&U))
          return false;
        const IRPosition &CSArgPos =
            IRPosition::callsite_argument(CS, CS.getArgumentNo(&U));
        const auto &CSArgIsDead = A.getAAFor<AAIsDead>(*this, CSArgPos);
        return CSArgIsDead.isAssumedDead();
      }
      if (ReturnInst *RI = dyn_cast<ReturnInst>(UserI)) {
        const IRPosition &RetPos = IRPosition::returned(*RI->getFunction());
        const auto &RetIsDeadAA = A.getAAFor<AAIsDead>(*this, RetPos);
        return RetIsDeadAA.isAssumedDead();
      }
      Follow = true;
      return wouldInstructionBeTriviallyDead(UserI);
    };

    if (!A.checkForAllUses(UsePred, *this, getAssociatedValue()))
      return indicatePessimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    Value &V = getAssociatedValue();
    if (auto *I = dyn_cast<Instruction>(&V))
      if (wouldInstructionBeTriviallyDead(I)) {
        A.deleteAfterManifest(*I);
        return ChangeStatus::CHANGED;
      }

    if (V.use_empty())
      return ChangeStatus::UNCHANGED;

    UndefValue &UV = *UndefValue::get(V.getType());
    bool AnyChange = A.changeValueAfterManifest(V, UV);
    return AnyChange ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(IsDead)
  }
};

struct AAIsDeadArgument : public AAIsDeadFloating {
  AAIsDeadArgument(const IRPosition &IRP) : AAIsDeadFloating(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (!getAssociatedFunction()->hasExactDefinition())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = AAIsDeadFloating::manifest(A);
    Argument &Arg = *getAssociatedArgument();
    if (Arg.getParent()->hasLocalLinkage())
      if (A.registerFunctionSignatureRewrite(
              Arg, /* ReplacementTypes */ {},
              Attributor::ArgumentReplacementInfo::CalleeRepairCBTy{},
              Attributor::ArgumentReplacementInfo::ACSRepairCBTy{}))
        return ChangeStatus::CHANGED;
    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(IsDead) }
};

struct AAIsDeadCallSiteArgument : public AAIsDeadValueImpl {
  AAIsDeadCallSiteArgument(const IRPosition &IRP) : AAIsDeadValueImpl(IRP) {}

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
    auto &ArgAA = A.getAAFor<AAIsDead>(*this, ArgPos);
    return clampStateAndIndicateChange(
        getState(), static_cast<const AAIsDead::StateType &>(ArgAA.getState()));
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    CallBase &CB = cast<CallBase>(getAnchorValue());
    Use &U = CB.getArgOperandUse(getArgNo());
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

struct AAIsDeadReturned : public AAIsDeadValueImpl {
  AAIsDeadReturned(const IRPosition &IRP) : AAIsDeadValueImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {

    auto PredForCallSite = [&](AbstractCallSite ACS) {
      if (ACS.isCallbackCall())
        return false;
      const IRPosition &CSRetPos =
          IRPosition::callsite_returned(ACS.getCallSite());
      const auto &RetIsDeadAA = A.getAAFor<AAIsDead>(*this, CSRetPos);
      return RetIsDeadAA.isAssumedDead();
    };

    if (!A.checkForAllCallSites(PredForCallSite, *this, true))
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

struct AAIsDeadCallSiteReturned : public AAIsDeadFloating {
  AAIsDeadCallSiteReturned(const IRPosition &IRP) : AAIsDeadFloating(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSRET_ATTR(IsDead) }
};

struct AAIsDeadFunction : public AAIsDead {
  AAIsDeadFunction(const IRPosition &IRP) : AAIsDead(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    const Function *F = getAssociatedFunction();
    if (F && !F->isDeclaration()) {
      ToBeExploredFrom.insert(&F->getEntryBlock().front());
      assumeLive(A, F->getEntryBlock());
    }
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return "Live[#BB " + std::to_string(AssumedLiveBlocks.size()) + "/" +
           std::to_string(getAssociatedFunction()->size()) + "][#TBEP " +
           std::to_string(ToBeExploredFrom.size()) + "][#KDE " +
           std::to_string(KnownDeadEnds.size()) + "]";
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

    KnownDeadEnds.set_union(ToBeExploredFrom);
    for (const Instruction *DeadEndI : KnownDeadEnds) {
      auto *CB = dyn_cast<CallBase>(DeadEndI);
      if (!CB)
        continue;
      const auto &NoReturnAA =
          A.getAAFor<AANoReturn>(*this, IRPosition::callsite_function(*CB));
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

    for (BasicBlock &BB : F)
      if (!AssumedLiveBlocks.count(&BB))
        A.deleteAfterManifest(BB);

    return HasChanged;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

  /// Returns true if the function is assumed dead.
  bool isAssumedDead() const override { return false; }

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

  /// Determine if \p F might catch asynchronous exceptions.
  static bool mayCatchAsynchronousExceptions(const Function &F) {
    return F.hasPersonalityFn() && !canSimplifyInvokeNoUnwind(&F);
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
      if (ImmutableCallSite ICS = ImmutableCallSite(&I))
        if (const Function *F = ICS.getCalledFunction())
          if (F->hasLocalLinkage())
            A.markLiveInternalFunction(*F);
    return true;
  }

  /// Collection of instructions that need to be explored again, e.g., we
  /// did assume they do not transfer control to (one of their) successors.
  SmallSetVector<const Instruction *, 8> ToBeExploredFrom;

  /// Collection of instructions that are known to not transfer control.
  SmallSetVector<const Instruction *, 8> KnownDeadEnds;

  /// Collection of all assumed live BasicBlocks.
  DenseSet<const BasicBlock *> AssumedLiveBlocks;
};

static bool
identifyAliveSuccessors(Attributor &A, const CallBase &CB,
                        AbstractAttribute &AA,
                        SmallVectorImpl<const Instruction *> &AliveSuccessors) {
  const IRPosition &IPos = IRPosition::callsite_function(CB);

  const auto &NoReturnAA = A.getAAFor<AANoReturn>(AA, IPos);
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
    const auto &AANoUnw = A.getAAFor<AANoUnwind>(AA, IPos);
    if (AANoUnw.isAssumedNoUnwind()) {
      UsedAssumedInformation |= !AANoUnw.isKnownNoUnwind();
    } else {
      AliveSuccessors.push_back(&II.getUnwindDest()->front());
    }
  }
  return UsedAssumedInformation;
}

static Optional<ConstantInt *>
getAssumedConstant(Attributor &A, const Value &V, AbstractAttribute &AA,
                   bool &UsedAssumedInformation) {
  const auto &ValueSimplifyAA =
      A.getAAFor<AAValueSimplify>(AA, IRPosition::value(V));
  Optional<Value *> SimplifiedV = ValueSimplifyAA.getAssumedSimplifiedValue(A);
  UsedAssumedInformation |= !ValueSimplifyAA.isKnown();
  if (!SimplifiedV.hasValue())
    return llvm::None;
  if (isa_and_nonnull<UndefValue>(SimplifiedV.getValue()))
    return llvm::None;
  return dyn_cast_or_null<ConstantInt>(SimplifiedV.getValue());
}

static bool
identifyAliveSuccessors(Attributor &A, const BranchInst &BI,
                        AbstractAttribute &AA,
                        SmallVectorImpl<const Instruction *> &AliveSuccessors) {
  bool UsedAssumedInformation = false;
  if (BI.getNumSuccessors() == 1) {
    AliveSuccessors.push_back(&BI.getSuccessor(0)->front());
  } else {
    Optional<ConstantInt *> CI =
        getAssumedConstant(A, *BI.getCondition(), AA, UsedAssumedInformation);
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
      getAssumedConstant(A, *SI.getCondition(), AA, UsedAssumedInformation);
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
                    << getAssociatedFunction()->size() << "] BBs and "
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

    AliveSuccessors.clear();

    bool UsedAssumedInformation = false;
    switch (I->getOpcode()) {
    // TODO: look for (assumed) UB to backwards propagate "deadness".
    default:
      if (I->isTerminator()) {
        for (const BasicBlock *SuccBB : successors(I->getParent()))
          AliveSuccessors.push_back(&SuccBB->front());
      } else {
        AliveSuccessors.push_back(I->getNextNode());
      }
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
      getAssociatedFunction()->size() == AssumedLiveBlocks.size() &&
      llvm::all_of(KnownDeadEnds, [](const Instruction *DeadEndI) {
        return DeadEndI->isTerminator() && DeadEndI->getNumSuccessors() == 0;
      }))
    return indicatePessimisticFixpoint();
  return Change;
}

/// Liveness information for a call sites.
struct AAIsDeadCallSite final : AAIsDeadFunction {
  AAIsDeadCallSite(const IRPosition &IRP) : AAIsDeadFunction(IRP) {}

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

  /// Helper function for collecting accessed bytes in must-be-executed-context
  void addAccessedBytesForUse(Attributor &A, const Use *U,
                              const Instruction *I) {
    const Value *UseV = U->get();
    if (!UseV->getType()->isPointerTy())
      return;

    Type *PtrTy = UseV->getType();
    const DataLayout &DL = A.getDataLayout();
    int64_t Offset;
    if (const Value *Base = getBasePointerOfAccessPointerOperand(
            I, Offset, DL, /*AllowNonInbounds*/ true)) {
      if (Base == &getAssociatedValue() &&
          Attributor::getPointerOperand(I, /* AllowVolatile */ false) == UseV) {
        uint64_t Size = DL.getTypeStoreSize(PtrTy->getPointerElementType());
        addAccessedBytes(Offset, Size);
      }
    }
    return;
  }

  /// See AAFromMustBeExecutedContext
  bool followUse(Attributor &A, const Use *U, const Instruction *I) {
    bool IsNonNull = false;
    bool TrackUse = false;
    int64_t DerefBytes = getKnownNonNullAndDerefBytesForUse(
        A, *this, getAssociatedValue(), U, I, IsNonNull, TrackUse);

    addAccessedBytesForUse(A, U, I);
    takeKnownDerefBytesMaximum(DerefBytes);
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
struct AADereferenceableFloating
    : AAFromMustBeExecutedContext<AADereferenceable, AADereferenceableImpl> {
  using Base =
      AAFromMustBeExecutedContext<AADereferenceable, AADereferenceableImpl>;
  AADereferenceableFloating(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = Base::updateImpl(A);

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

      // TODO: Use `AAConstantRange` to infer dereferenceable bytes.

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
            A, getIRPosition(), *this, T, VisitValueCB))
      return indicatePessimisticFixpoint();

    return Change | clampStateAndIndicateChange(getState(), T);
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
    : AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext<
          AADereferenceable, AADereferenceableImpl, DerefState> {
  using Base = AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext<
      AADereferenceable, AADereferenceableImpl, DerefState>;
  AADereferenceableArgument(const IRPosition &IRP) : Base(IRP) {}

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
struct AADereferenceableCallSiteReturned final
    : AACallSiteReturnedFromReturnedAndMustBeExecutedContext<
          AADereferenceable, AADereferenceableImpl> {
  using Base = AACallSiteReturnedFromReturnedAndMustBeExecutedContext<
      AADereferenceable, AADereferenceableImpl>;
  AADereferenceableCallSiteReturned(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(dereferenceable);
  }
};

// ------------------------ Align Argument Attribute ------------------------

static unsigned int getKnownAlignForUse(Attributor &A,
                                        AbstractAttribute &QueryingAA,
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
    if (GEP->hasAllConstantIndices()) {
      TrackUse = true;
      return 0;
    }
  }

  unsigned Alignment = 0;
  if (ImmutableCallSite ICS = ImmutableCallSite(I)) {
    if (ICS.isBundleOperand(U) || ICS.isCallee(U))
      return 0;

    unsigned ArgNo = ICS.getArgumentNo(U);
    IRPosition IRP = IRPosition::callsite_argument(ICS, ArgNo);
    // As long as we only use known information there is no need to track
    // dependences here.
    auto &AlignAA = A.getAAFor<AAAlign>(QueryingAA, IRP,
                                        /* TrackDependence */ false);
    Alignment = AlignAA.getKnownAlign();
  }

  const Value *UseV = U->get();
  if (auto *SI = dyn_cast<StoreInst>(I))
    Alignment = SI->getAlignment();
  else if (auto *LI = dyn_cast<LoadInst>(I))
    Alignment = LI->getAlignment();

  if (Alignment <= 1)
    return 0;

  auto &DL = A.getDataLayout();
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
  AAAlignImpl(const IRPosition &IRP) : AAAlign(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
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
                            "Number of times alignment added to a store");
            SI->setAlignment(Align(getAssumedAlign()));
            Changed = ChangeStatus::CHANGED;
          }
      } else if (auto *LI = dyn_cast<LoadInst>(U.getUser())) {
        if (LI->getPointerOperand() == &AnchorVal)
          if (LI->getAlignment() < getAssumedAlign()) {
            LI->setAlignment(Align(getAssumedAlign()));
            STATS_DECLTRACK(AAAlign, Load,
                            "Number of times alignment added to a load");
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
      Attrs.emplace_back(
          Attribute::getWithAlignment(Ctx, Align(getAssumedAlign())));
  }
  /// See AAFromMustBeExecutedContext
  bool followUse(Attributor &A, const Use *U, const Instruction *I) {
    bool TrackUse = false;

    unsigned int KnownAlign =
        getKnownAlignForUse(A, *this, getAssociatedValue(), U, I, TrackUse);
    takeKnownMaximum(KnownAlign);

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
struct AAAlignFloating : AAFromMustBeExecutedContext<AAAlign, AAAlignImpl> {
  using Base = AAFromMustBeExecutedContext<AAAlign, AAAlignImpl>;
  AAAlignFloating(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    Base::updateImpl(A);

    const DataLayout &DL = A.getDataLayout();

    auto VisitValueCB = [&](Value &V, AAAlign::StateType &T,
                            bool Stripped) -> bool {
      const auto &AA = A.getAAFor<AAAlign>(*this, IRPosition::value(V));
      if (!Stripped && this == &AA) {
        // Use only IR information if we did not strip anything.
        const MaybeAlign PA = V.getPointerAlignment(DL);
        T.takeKnownMaximum(PA ? PA->value() : 0);
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
    : AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext<AAAlign,
                                                              AAAlignImpl> {
  AAAlignArgument(const IRPosition &IRP)
      : AAArgumentFromCallSiteArgumentsAndMustBeExecutedContext<AAAlign,
                                                                AAAlignImpl>(
            IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(aligned) }
};

struct AAAlignCallSiteArgument final : AAAlignFloating {
  AAAlignCallSiteArgument(const IRPosition &IRP) : AAAlignFloating(IRP) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    return AAAlignImpl::manifest(A);
  }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = AAAlignFloating::updateImpl(A);
    if (Argument *Arg = getAssociatedArgument()) {
      const auto &ArgAlignAA = A.getAAFor<AAAlign>(
          *this, IRPosition::argument(*Arg), /* TrackDependence */ false,
          DepClassTy::OPTIONAL);
      takeKnownMaximum(ArgAlignAA.getKnownAlign());
    }
    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CSARG_ATTR(aligned) }
};

/// Align attribute deduction for a call site return value.
struct AAAlignCallSiteReturned final
    : AACallSiteReturnedFromReturnedAndMustBeExecutedContext<AAAlign,
                                                             AAAlignImpl> {
  using Base =
      AACallSiteReturnedFromReturnedAndMustBeExecutedContext<AAAlign,
                                                             AAAlignImpl>;
  AAAlignCallSiteReturned(const IRPosition &IRP) : Base(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Base::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(align); }
};

/// ------------------ Function No-Return Attribute ----------------------------
struct AANoReturnImpl : public AANoReturn {
  AANoReturnImpl(const IRPosition &IRP) : AANoReturn(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AANoReturn::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F)
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
  AANoReturnFunction(const IRPosition &IRP) : AANoReturnImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(noreturn) }
};

/// NoReturn attribute deduction for a call sites.
struct AANoReturnCallSite final : AANoReturnImpl {
  AANoReturnCallSite(const IRPosition &IRP) : AANoReturnImpl(IRP) {}

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
    if (hasAttr(getAttrKind(), /* IgnoreSubsumingPositions */ true)) {
      indicateOptimisticFixpoint();
      return;
    }
    Function *AnchorScope = getAnchorScope();
    if (isFnInterfaceKind() &&
        (!AnchorScope || !AnchorScope->hasExactDefinition())) {
      indicatePessimisticFixpoint();
      return;
    }

    // You cannot "capture" null in the default address space.
    if (isa<ConstantPointerNull>(getAssociatedValue()) &&
        getAssociatedValue().getType()->getPointerAddressSpace() == 0) {
      indicateOptimisticFixpoint();
      return;
    }

    const Function *F = getArgNo() >= 0 ? getAssociatedFunction() : AnchorScope;

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
    int ArgNo = IRP.getArgNo();
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
  AANoCapture::StateType &State;

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
  const IRPosition &FnPos = IRPosition::function(*F);
  const auto &IsDeadAA = A.getAAFor<AAIsDead>(*this, FnPos);

  AANoCapture::StateType T;

  // Readonly means we cannot capture through memory.
  const auto &FnMemAA = A.getAAFor<AAMemoryBehavior>(*this, FnPos);
  if (FnMemAA.isAssumedReadOnly()) {
    T.addKnownBits(NOT_CAPTURED_IN_MEM);
    if (FnMemAA.isKnownReadOnly())
      addKnownBits(NOT_CAPTURED_IN_MEM);
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

  const auto &NoUnwindAA = A.getAAFor<AANoUnwind>(*this, FnPos);
  if (NoUnwindAA.isAssumedNoUnwind()) {
    bool IsVoidTy = F->getReturnType()->isVoidTy();
    const AAReturnedValues *RVAA =
        IsVoidTy ? nullptr : &A.getAAFor<AAReturnedValues>(*this, FnPos);
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
  unsigned RemainingUsesToExplore = DefaultMaxUsesToExplore;
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
  AANoCaptureArgument(const IRPosition &IRP) : AANoCaptureImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(nocapture) }
};

/// NoCapture attribute for call site arguments.
struct AANoCaptureCallSiteArgument final : AANoCaptureImpl {
  AANoCaptureCallSiteArgument(const IRPosition &IRP) : AANoCaptureImpl(IRP) {}

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

  bool askSimplifiedValueForAAValueConstantRange(Attributor &A) {
    if (!getAssociatedValue().getType()->isIntegerTy())
      return false;

    const auto &ValueConstantRangeAA =
        A.getAAFor<AAValueConstantRange>(*this, getIRPosition());

    Optional<ConstantInt *> COpt =
        ValueConstantRangeAA.getAssumedConstantInt(A);
    if (COpt.hasValue()) {
      if (auto *C = COpt.getValue())
        SimplifiedAssociatedValue = C;
      else
        return false;
    } else {
      // FIXME: It should be llvm::None but if you set llvm::None,
      //        values are mistakenly infered as `undef` now.
      SimplifiedAssociatedValue = &getAssociatedValue();
    }
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
        A.changeValueAfterManifest(V, *C);
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
  AAValueSimplifyArgument(const IRPosition &IRP) : AAValueSimplifyImpl(IRP) {}

  void initialize(Attributor &A) override {
    AAValueSimplifyImpl::initialize(A);
    if (!getAssociatedFunction() || getAssociatedFunction()->isDeclaration())
      indicatePessimisticFixpoint();
    if (hasAttr({Attribute::InAlloca, Attribute::StructRet, Attribute::Nest},
                /* IgnoreSubsumingPositions */ true))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // Byval is only replacable if it is readonly otherwise we would write into
    // the replaced value and not the copy that byval creates implicitly.
    Argument *Arg = getAssociatedArgument();
    if (Arg->hasByValAttr()) {
      const auto &MemAA = A.getAAFor<AAMemoryBehavior>(*this, getIRPosition());
      if (!MemAA.isAssumedReadOnly())
        return indicatePessimisticFixpoint();
    }

    bool HasValueBefore = SimplifiedAssociatedValue.hasValue();

    auto PredForCallSite = [&](AbstractCallSite ACS) {
      // Check if we have an associated argument or not (which can happen for
      // callback calls).
      Value *ArgOp = ACS.getCallArgOperand(getArgNo());
      if (!ArgOp)
        return false;
      // We can only propagate thread independent values through callbacks.
      // This is different to direct/indirect call sites because for them we
      // know the thread executing the caller and callee is the same. For
      // callbacks this is not guaranteed, thus a thread dependent value could
      // be different for the caller and callee, making it invalid to propagate.
      if (ACS.isCallbackCall())
        if (auto *C = dyn_cast<Constant>(ArgOp))
          if (C->isThreadDependent())
            return false;
      return checkAndUpdate(A, *this, *ArgOp, SimplifiedAssociatedValue);
    };

    if (!A.checkForAllCallSites(PredForCallSite, *this, true))
      if (!askSimplifiedValueForAAValueConstantRange(A))
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
      if (!askSimplifiedValueForAAValueConstantRange(A))
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
    if (isa<Constant>(V))
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
        return false;
      }
      return checkAndUpdate(A, *this, V, SimplifiedAssociatedValue);
    };

    if (!genericValueTraversal<AAValueSimplify, BooleanState>(
            A, getIRPosition(), *this, static_cast<BooleanState &>(*this),
            VisitValueCB))
      if (!askSimplifiedValueForAAValueConstantRange(A))
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

      replaceAllInstructionUsesWith(*MallocCall, *AI);

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
            *this, IRPosition::callsite_argument(*CB, ArgNo));

        // If a callsite argument use is nofree, we are fine.
        const auto &ArgNoFreeAA = A.getAAFor<AANoFree>(
            *this, IRPosition::callsite_argument(*CB, ArgNo));

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
    bool IsCalloc = !IsMalloc && isCallocLikeFn(&I, TLI);
    if (!IsMalloc && !IsCalloc) {
      BadMallocCalls.insert(&I);
      return true;
    }

    if (IsMalloc) {
      if (auto *Size = dyn_cast<ConstantInt>(I.getOperand(0)))
        if (Size->getValue().ule(MaxHeapToStackSize))
          if (UsesCheck(I) || FreeCheck(I)) {
            MallocCalls.insert(&I);
            return true;
          }
    } else if (IsCalloc) {
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
  AAHeapToStackFunction(const IRPosition &IRP) : AAHeapToStackImpl(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECL(MallocCalls, Function,
               "Number of malloc calls converted to allocas");
    for (auto *C : MallocCalls)
      if (!BadMallocCalls.count(C))
        ++BUILD_STAT_NAME(MallocCalls, Function);
  }
};

/// -------------------- Memory Behavior Attributes ----------------------------
/// Includes read-none, read-only, and write-only.
/// ----------------------------------------------------------------------------
struct AAMemoryBehaviorImpl : public AAMemoryBehavior {
  AAMemoryBehaviorImpl(const IRPosition &IRP) : AAMemoryBehavior(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    intersectAssumedBits(BEST_STATE);
    getKnownStateFromValue(getIRPosition(), getState());
    IRAttribute::initialize(A);
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
        llvm_unreachable("Unexpcted attribute!");
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
  AAMemoryBehaviorFloating(const IRPosition &IRP) : AAMemoryBehaviorImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAMemoryBehaviorImpl::initialize(A);
    // Initialize the use vector with all direct uses of the associated value.
    for (const Use &U : getAssociatedValue().uses())
      Uses.insert(&U);
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
  /// Container for (transitive) uses of the associated argument.
  SetVector<const Use *> Uses;
};

/// Memory behavior attribute for function argument.
struct AAMemoryBehaviorArgument : AAMemoryBehaviorFloating {
  AAMemoryBehaviorArgument(const IRPosition &IRP)
      : AAMemoryBehaviorFloating(IRP) {}

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
    if (!Arg || !Arg->getParent()->hasExactDefinition()) {
      indicatePessimisticFixpoint();
    } else {
      // Initialize the use vector with all direct uses of the associated value.
      for (const Use &U : Arg->uses())
        Uses.insert(&U);
    }
  }

  ChangeStatus manifest(Attributor &A) override {
    // TODO: From readattrs.ll: "inalloca parameters are always
    //                           considered written"
    if (hasAttr({Attribute::InAlloca})) {
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
  AAMemoryBehaviorCallSiteArgument(const IRPosition &IRP)
      : AAMemoryBehaviorArgument(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (Argument *Arg = getAssociatedArgument()) {
      if (Arg->hasByValAttr()) {
        addKnownBits(NO_WRITES);
        removeKnownBits(NO_READS);
        removeAssumedBits(NO_READS);
      }
    } else {
    }
    AAMemoryBehaviorArgument::initialize(A);
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Argument *Arg = getAssociatedArgument();
    const IRPosition &ArgPos = IRPosition::argument(*Arg);
    auto &ArgAA = A.getAAFor<AAMemoryBehavior>(*this, ArgPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AAMemoryBehavior::StateType &>(ArgAA.getState()));
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
  AAMemoryBehaviorCallSiteReturned(const IRPosition &IRP)
      : AAMemoryBehaviorFloating(IRP) {}

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
  AAMemoryBehaviorFunction(const IRPosition &IRP) : AAMemoryBehaviorImpl(IRP) {}

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
  AAMemoryBehaviorCallSite(const IRPosition &IRP) : AAMemoryBehaviorImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAMemoryBehaviorImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || !F->hasExactDefinition())
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
    auto &FnAA = A.getAAFor<AAMemoryBehavior>(*this, FnPos);
    return clampStateAndIndicateChange(
        getState(),
        static_cast<const AAMemoryBehavior::StateType &>(FnAA.getState()));
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
} // namespace

ChangeStatus AAMemoryBehaviorFunction::updateImpl(Attributor &A) {

  // The current assumed state used to determine a change.
  auto AssumedState = getAssumed();

  auto CheckRWInst = [&](Instruction &I) {
    // If the instruction has an own memory behavior state, use it to restrict
    // the local state. No further analysis is required as the other memory
    // state is as optimistic as it gets.
    if (ImmutableCallSite ICS = ImmutableCallSite(&I)) {
      const auto &MemBehaviorAA = A.getAAFor<AAMemoryBehavior>(
          *this, IRPosition::callsite_function(ICS));
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
    const auto &FnMemAA = A.getAAFor<AAMemoryBehavior>(*this, FnPos);
    FnMemAssumedState = FnMemAA.getAssumed();
    S.addKnownBits(FnMemAA.getKnown());
    if ((S.getAssumed() & FnMemAA.getAssumed()) == S.getAssumed())
      return ChangeStatus::UNCHANGED;
  }

  // Make sure the value is not captured (except through "return"), if
  // it is, any information derived would be irrelevant anyway as we cannot
  // check the potential aliases introduced by the capture. However, no need
  // to fall back to anythign less optimistic than the function state.
  const auto &ArgNoCaptureAA = A.getAAFor<AANoCapture>(
      *this, IRP, /* TrackDependence */ true, DepClassTy::OPTIONAL);
  if (!ArgNoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
    S.intersectAssumedBits(FnMemAssumedState);
    return ChangeStatus::CHANGED;
  }

  // The current assumed state used to determine a change.
  auto AssumedState = S.getAssumed();

  // Liveness information to exclude dead users.
  // TODO: Take the FnPos once we have call site specific liveness information.
  const auto &LivenessAA = A.getAAFor<AAIsDead>(
      *this, IRPosition::function(*IRP.getAssociatedFunction()));

  // Visit and expand uses until all are analyzed or a fixpoint is reached.
  for (unsigned i = 0; i < Uses.size() && !isAtFixpoint(); i++) {
    const Use *U = Uses[i];
    Instruction *UserI = cast<Instruction>(U->getUser());
    LLVM_DEBUG(dbgs() << "[AAMemoryBehavior] Use: " << **U << " in " << *UserI
                      << " [Dead: " << (LivenessAA.isAssumedDead(UserI))
                      << "]\n");
    if (LivenessAA.isAssumedDead(UserI))
      continue;

    // Check if the users of UserI should also be visited.
    if (followUsersOfUseIn(A, U, UserI))
      for (const Use &UserIUse : UserI->uses())
        Uses.insert(&UserIUse);

    // If UserI might touch memory we analyze the use in detail.
    if (UserI->mayReadOrWriteMemory())
      analyzeUseIn(A, U, UserI);
  }

  return (AssumedState != getAssumed()) ? ChangeStatus::CHANGED
                                        : ChangeStatus::UNCHANGED;
}

bool AAMemoryBehaviorFloating::followUsersOfUseIn(Attributor &A, const Use *U,
                                                  const Instruction *UserI) {
  // The loaded value is unrelated to the pointer argument, no need to
  // follow the users of the load.
  if (isa<LoadInst>(UserI))
    return false;

  // By default we follow all uses assuming UserI might leak information on U,
  // we have special handling for call sites operands though.
  ImmutableCallSite ICS(UserI);
  if (!ICS || !ICS.isArgOperand(U))
    return true;

  // If the use is a call argument known not to be captured, the users of
  // the call do not need to be visited because they have to be unrelated to
  // the input. Note that this check is not trivial even though we disallow
  // general capturing of the underlying argument. The reason is that the
  // call might the argument "through return", which we allow and for which we
  // need to check call users.
  unsigned ArgNo = ICS.getArgumentNo(U);
  const auto &ArgNoCaptureAA =
      A.getAAFor<AANoCapture>(*this, IRPosition::callsite_argument(ICS, ArgNo));
  return !ArgNoCaptureAA.isAssumedNoCapture();
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
    ImmutableCallSite ICS(UserI);

    // Give up on operand bundles.
    if (ICS.isBundleOperand(U)) {
      indicatePessimisticFixpoint();
      return;
    }

    // Calling a function does read the function pointer, maybe write it if the
    // function is self-modifying.
    if (ICS.isCallee(U)) {
      removeAssumedBits(NO_READS);
      break;
    }

    // Adjust the possible access behavior based on the information on the
    // argument.
    unsigned ArgNo = ICS.getArgumentNo(U);
    const IRPosition &ArgPos = IRPosition::callsite_argument(ICS, ArgNo);
    const auto &MemBehaviorAA = A.getAAFor<AAMemoryBehavior>(*this, ArgPos);
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
/// ------------------ Value Constant Range Attribute -------------------------

struct AAValueConstantRangeImpl : AAValueConstantRange {
  using StateType = IntegerRangeState;
  AAValueConstantRangeImpl(const IRPosition &IRP) : AAValueConstantRange(IRP) {}

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
                                 const_cast<BasicBlock *>(CtxI->getParent()),
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
      if (Instruction *I = dyn_cast<Instruction>(&V))
        if (isa<CallInst>(I) || isa<LoadInst>(I))
          if (setRangeMetadataIfisBetterRange(I, AssumedConstantRange))
            Changed = ChangeStatus::CHANGED;
    }

    return Changed;
  }
};

struct AAValueConstantRangeArgument final : public AAValueConstantRangeImpl {

  AAValueConstantRangeArgument(const IRPosition &IRP)
      : AAValueConstantRangeImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Use AAArgumentFromCallSiteArguments

    IntegerRangeState S(getBitWidth());
    clampCallSiteArgumentStates<AAValueConstantRange, IntegerRangeState>(
        A, *this, S);

    // TODO: If we know we visited all incoming values, thus no are assumed
    // dead, we can take the known information from the state T.
    return clampStateAndIndicateChange<IntegerRangeState>(this->getState(), S);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_ARG_ATTR(value_range)
  }
};

struct AAValueConstantRangeReturned : AAValueConstantRangeImpl {
  AAValueConstantRangeReturned(const IRPosition &IRP)
      : AAValueConstantRangeImpl(IRP) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    // TODO: Use AAReturnedFromReturnedValues

    // TODO: If we know we visited all returned values, thus no are assumed
    // dead, we can take the known information from the state T.

    IntegerRangeState S(getBitWidth());

    clampReturnedValueStates<AAValueConstantRange, IntegerRangeState>(A, *this,
                                                                      S);
    return clampStateAndIndicateChange<StateType>(this->getState(), S);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(value_range)
  }
};

struct AAValueConstantRangeFloating : AAValueConstantRangeImpl {
  AAValueConstantRangeFloating(const IRPosition &IRP)
      : AAValueConstantRangeImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAValueConstantRange::initialize(A);
    Value &V = getAssociatedValue();

    if (auto *C = dyn_cast<ConstantInt>(&V)) {
      unionAssumed(ConstantRange(C->getValue()));
      indicateOptimisticFixpoint();
      return;
    }

    if (isa<UndefValue>(&V)) {
      indicateOptimisticFixpoint();
      return;
    }

    if (auto *I = dyn_cast<Instruction>(&V))
      if (isa<BinaryOperator>(I) || isa<CmpInst>(I)) {
        Value *LHS = I->getOperand(0);
        Value *RHS = I->getOperand(1);

        if (LHS->getType()->isIntegerTy() && RHS->getType()->isIntegerTy())
          return;
      }

    // If it is a load instruction with range metadata, use it.
    if (LoadInst *LI = dyn_cast<LoadInst>(&V))
      if (auto *RangeMD = LI->getMetadata(LLVMContext::MD_range)) {
        intersectKnown(getConstantRangeFromMetadata(*RangeMD));
        return;
      }

    // Otherwise we give up.
    indicatePessimisticFixpoint();

    LLVM_DEBUG(dbgs() << "[Attributor][AAValueConstantRange] We give up: "
                      << getAssociatedValue());
  }

  bool calculateBinaryOperator(Attributor &A, BinaryOperator *BinOp,
                               IntegerRangeState &T, Instruction *CtxI) {
    Value *LHS = BinOp->getOperand(0);
    Value *RHS = BinOp->getOperand(1);

    auto &LHSAA =
        A.getAAFor<AAValueConstantRange>(*this, IRPosition::value(*LHS));
    auto LHSAARange = LHSAA.getAssumedConstantRange(A, CtxI);

    auto &RHSAA =
        A.getAAFor<AAValueConstantRange>(*this, IRPosition::value(*RHS));
    auto RHSAARange = RHSAA.getAssumedConstantRange(A, CtxI);

    auto AssumedRange = LHSAARange.binaryOp(BinOp->getOpcode(), RHSAARange);

    T.unionAssumed(AssumedRange);

    // TODO: Track a known state too.

    return T.isValidState();
  }

  bool calculateCmpInst(Attributor &A, CmpInst *CmpI, IntegerRangeState &T,
                        Instruction *CtxI) {
    Value *LHS = CmpI->getOperand(0);
    Value *RHS = CmpI->getOperand(1);

    auto &LHSAA =
        A.getAAFor<AAValueConstantRange>(*this, IRPosition::value(*LHS));
    auto &RHSAA =
        A.getAAFor<AAValueConstantRange>(*this, IRPosition::value(*RHS));

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
    Instruction *CtxI = getCtxI();
    auto VisitValueCB = [&](Value &V, IntegerRangeState &T,
                            bool Stripped) -> bool {
      Instruction *I = dyn_cast<Instruction>(&V);
      if (!I) {

        // If the value is not instruction, we query AA to Attributor.
        const auto &AA =
            A.getAAFor<AAValueConstantRange>(*this, IRPosition::value(V));

        // Clamp operator is not used to utilize a program point CtxI.
        T.unionAssumed(AA.getAssumedConstantRange(A, CtxI));

        return T.isValidState();
      }

      if (auto *BinOp = dyn_cast<BinaryOperator>(I))
        return calculateBinaryOperator(A, BinOp, T, CtxI);
      else if (auto *CmpI = dyn_cast<CmpInst>(I))
        return calculateCmpInst(A, CmpI, T, CtxI);
      else {
        // Give up with other instructions.
        // TODO: Add other instructions

        T.indicatePessimisticFixpoint();
        return false;
      }
    };

    IntegerRangeState T(getBitWidth());

    if (!genericValueTraversal<AAValueConstantRange, IntegerRangeState>(
            A, getIRPosition(), *this, T, VisitValueCB))
      return indicatePessimisticFixpoint();

    return clampStateAndIndicateChange(getState(), T);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(value_range)
  }
};

struct AAValueConstantRangeFunction : AAValueConstantRangeImpl {
  AAValueConstantRangeFunction(const IRPosition &IRP)
      : AAValueConstantRangeImpl(IRP) {}

  /// See AbstractAttribute::initialize(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("AAValueConstantRange(Function|CallSite)::updateImpl will "
                     "not be called");
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(value_range) }
};

struct AAValueConstantRangeCallSite : AAValueConstantRangeFunction {
  AAValueConstantRangeCallSite(const IRPosition &IRP)
      : AAValueConstantRangeFunction(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_CS_ATTR(value_range) }
};

struct AAValueConstantRangeCallSiteReturned : AAValueConstantRangeReturned {
  AAValueConstantRangeCallSiteReturned(const IRPosition &IRP)
      : AAValueConstantRangeReturned(IRP) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // If it is a load instruction with range metadata, use the metadata.
    if (CallInst *CI = dyn_cast<CallInst>(&getAssociatedValue()))
      if (auto *RangeMD = CI->getMetadata(LLVMContext::MD_range))
        intersectKnown(getConstantRangeFromMetadata(*RangeMD));

    AAValueConstantRangeReturned::initialize(A);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(value_range)
  }
};
struct AAValueConstantRangeCallSiteArgument : AAValueConstantRangeFloating {
  AAValueConstantRangeCallSiteArgument(const IRPosition &IRP)
      : AAValueConstantRangeFloating(IRP) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(value_range)
  }
};
/// ----------------------------------------------------------------------------
///                               Attributor
/// ----------------------------------------------------------------------------

bool Attributor::isAssumedDead(const AbstractAttribute &AA,
                               const AAIsDead *LivenessAA) {
  const Instruction *CtxI = AA.getIRPosition().getCtxI();
  if (!CtxI)
    return false;

  // TODO: Find a good way to utilize fine and coarse grained liveness
  // information.
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
  recordDependence(*LivenessAA, AA, DepClassTy::OPTIONAL);

  return true;
}

bool Attributor::checkForAllUses(
    const function_ref<bool(const Use &, bool &)> &Pred,
    const AbstractAttribute &QueryingAA, const Value &V) {
  const IRPosition &IRP = QueryingAA.getIRPosition();
  SmallVector<const Use *, 16> Worklist;
  SmallPtrSet<const Use *, 16> Visited;

  for (const Use &U : V.uses())
    Worklist.push_back(&U);

  LLVM_DEBUG(dbgs() << "[Attributor] Got " << Worklist.size()
                    << " initial uses to check\n");

  if (Worklist.empty())
    return true;

  bool AnyDead = false;
  const Function *ScopeFn = IRP.getAnchorScope();
  const auto *LivenessAA =
      ScopeFn ? &getAAFor<AAIsDead>(QueryingAA, IRPosition::function(*ScopeFn),
                                    /* TrackDependence */ false)
              : nullptr;

  while (!Worklist.empty()) {
    const Use *U = Worklist.pop_back_val();
    if (!Visited.insert(U).second)
      continue;
    LLVM_DEBUG(dbgs() << "[Attributor] Check use: " << **U << "\n");
    if (Instruction *UserI = dyn_cast<Instruction>(U->getUser()))
      if (LivenessAA && LivenessAA->isAssumedDead(UserI)) {
        LLVM_DEBUG(dbgs() << "[Attributor] Dead user: " << *UserI << ": "
                          << *LivenessAA << "\n");
        AnyDead = true;
        continue;
      }

    bool Follow = false;
    if (!Pred(*U, Follow))
      return false;
    if (!Follow)
      continue;
    for (const Use &UU : U->getUser()->uses())
      Worklist.push_back(&UU);
  }

  if (AnyDead)
    recordDependence(*LivenessAA, QueryingAA, DepClassTy::OPTIONAL);

  return true;
}

bool Attributor::checkForAllCallSites(
    const function_ref<bool(AbstractCallSite)> &Pred,
    const AbstractAttribute &QueryingAA, bool RequireAllCallSites) {
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
                              &QueryingAA);
}

bool Attributor::checkForAllCallSites(
    const function_ref<bool(AbstractCallSite)> &Pred, const Function &Fn,
    bool RequireAllCallSites, const AbstractAttribute *QueryingAA) {
  if (RequireAllCallSites && !Fn.hasLocalLinkage()) {
    LLVM_DEBUG(
        dbgs()
        << "[Attributor] Function " << Fn.getName()
        << " has no internal linkage, hence not all call sites are known\n");
    return false;
  }

  for (const Use &U : Fn.uses()) {
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

    Instruction *I = ACS.getInstruction();
    Function *Caller = I->getFunction();

    const auto *LivenessAA =
        lookupAAFor<AAIsDead>(IRPosition::function(*Caller), QueryingAA,
                              /* TrackDependence */ false);

    // Skip dead calls.
    if (LivenessAA && LivenessAA->isAssumedDead(I)) {
      // We actually used liveness information so we have to record a
      // dependence.
      if (QueryingAA)
        recordDependence(*LivenessAA, *QueryingAA, DepClassTy::OPTIONAL);
      continue;
    }

    const Use *EffectiveUse =
        ACS.isCallbackCall() ? &ACS.getCalleeUseForCallback() : &U;
    if (!ACS.isCallee(EffectiveUse)) {
      if (!RequireAllCallSites)
        continue;
      LLVM_DEBUG(dbgs() << "[Attributor] User " << EffectiveUse->getUser()
                        << " is an invalid use of " << Fn.getName() << "\n");
      return false;
    }

    if (Pred(ACS))
      continue;

    LLVM_DEBUG(dbgs() << "[Attributor] Call site callback failed for "
                      << *ACS.getInstruction() << "\n");
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
  if (!checkForAllInstructionsImpl(OpcodeInstMap, Pred, &LivenessAA, AnyDead,
                                   Opcodes))
    return false;

  // If we actually used liveness information so we have to record a dependence.
  if (AnyDead)
    recordDependence(LivenessAA, QueryingAA, DepClassTy::OPTIONAL);

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
    recordDependence(LivenessAA, QueryingAA, DepClassTy::OPTIONAL);

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
  SetVector<AbstractAttribute *> Worklist, InvalidAAs;
  Worklist.insert(AllAbstractAttributes.begin(), AllAbstractAttributes.end());

  bool RecomputeDependences = false;

  do {
    // Remember the size to determine new attributes.
    size_t NumAAs = AllAbstractAttributes.size();
    LLVM_DEBUG(dbgs() << "\n\n[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist size: " << Worklist.size() << "\n");

    // For invalid AAs we can fix dependent AAs that have a required dependence,
    // thereby folding long dependence chains in a single step without the need
    // to run updates.
    for (unsigned u = 0; u < InvalidAAs.size(); ++u) {
      AbstractAttribute *InvalidAA = InvalidAAs[u];
      auto &QuerriedAAs = QueryMap[InvalidAA];
      LLVM_DEBUG(dbgs() << "[Attributor] InvalidAA: " << *InvalidAA << " has "
                        << QuerriedAAs.RequiredAAs.size() << "/"
                        << QuerriedAAs.OptionalAAs.size()
                        << " required/optional dependences\n");
      for (AbstractAttribute *DepOnInvalidAA : QuerriedAAs.RequiredAAs) {
        AbstractState &DOIAAState = DepOnInvalidAA->getState();
        DOIAAState.indicatePessimisticFixpoint();
        ++NumAttributesFixedDueToRequiredDependences;
        assert(DOIAAState.isAtFixpoint() && "Expected fixpoint state!");
        if (!DOIAAState.isValidState())
          InvalidAAs.insert(DepOnInvalidAA);
      }
      if (!RecomputeDependences)
        Worklist.insert(QuerriedAAs.OptionalAAs.begin(),
                        QuerriedAAs.OptionalAAs.end());
    }

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
      Worklist.insert(QuerriedAAs.OptionalAAs.begin(),
                      QuerriedAAs.OptionalAAs.end());
      Worklist.insert(QuerriedAAs.RequiredAAs.begin(),
                      QuerriedAAs.RequiredAAs.end());
    }

    LLVM_DEBUG(dbgs() << "[Attributor] #Iteration: " << IterationCounter
                      << ", Worklist+Dependent size: " << Worklist.size()
                      << "\n");

    // Reset the changed and invalid set.
    ChangedAAs.clear();
    InvalidAAs.clear();

    // Update all abstract attribute in the work list and record the ones that
    // changed.
    for (AbstractAttribute *AA : Worklist)
      if (!AA->getState().isAtFixpoint() && !isAssumedDead(*AA, nullptr)) {
        QueriedNonFixAA = false;
        if (AA->update(*this) == ChangeStatus::CHANGED) {
          ChangedAAs.push_back(AA);
          if (!AA->getState().isValidState())
            InvalidAAs.insert(AA);
        } else if (!QueriedNonFixAA) {
          // If the attribute did not query any non-fix information, the state
          // will not change and we can indicate that right away.
          AA->getState().indicateOptimisticFixpoint();
        }
      }

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
    ChangedAAs.append(QuerriedAAs.OptionalAAs.begin(),
                      QuerriedAAs.OptionalAAs.end());
    ChangedAAs.append(QuerriedAAs.RequiredAAs.begin(),
                      QuerriedAAs.RequiredAAs.end());
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
                      << ToBeDeletedInsts.size() << " instructions and "
                      << ToBeChangedUses.size() << " uses\n");

    SmallVector<Instruction *, 32> DeadInsts;
    SmallVector<Instruction *, 32> TerminatorsToFold;

    for (auto &It : ToBeChangedUses) {
      Use *U = It.first;
      Value *NewV = It.second;
      Value *OldV = U->get();
      LLVM_DEBUG(dbgs() << "Use " << *NewV << " in " << *U->getUser()
                        << " instead of " << *OldV << "\n");
      U->set(NewV);
      if (Instruction *I = dyn_cast<Instruction>(OldV))
        if (!isa<PHINode>(I) && !ToBeDeletedInsts.count(I) &&
            isInstructionTriviallyDead(I)) {
          DeadInsts.push_back(I);
        }
      if (isa<Constant>(NewV) && isa<BranchInst>(U->getUser())) {
        Instruction *UserI = cast<Instruction>(U->getUser());
        if (isa<UndefValue>(NewV)) {
          ToBeChangedToUnreachableInsts.insert(UserI);
        } else {
          TerminatorsToFold.push_back(UserI);
        }
      }
    }
    for (auto &V : InvokeWithDeadSuccessor)
      if (InvokeInst *II = dyn_cast_or_null<InvokeInst>(V)) {
        bool UnwindBBIsDead = II->hasFnAttr(Attribute::NoUnwind);
        bool NormalBBIsDead = II->hasFnAttr(Attribute::NoReturn);
        bool Invoke2CallAllowed =
            !AAIsDeadFunction::mayCatchAsynchronousExceptions(
                *II->getFunction());
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
    for (auto &V : ToBeChangedToUnreachableInsts)
      if (Instruction *I = dyn_cast_or_null<Instruction>(V))
        changeToUnreachable(I, /* UseLLVMTrap */ false);
    for (Instruction *I : TerminatorsToFold)
      ConstantFoldTerminator(I->getParent());

    for (Instruction *I : ToBeDeletedInsts) {
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
      if (!isa<PHINode>(I) && isInstructionTriviallyDead(I))
        DeadInsts.push_back(I);
      else
        I->eraseFromParent();
    }

    RecursivelyDeleteTriviallyDeadInstructions(DeadInsts);

    if (unsigned NumDeadBlocks = ToBeDeletedBlocks.size()) {
      SmallVector<BasicBlock *, 8> ToBeDeletedBBs;
      ToBeDeletedBBs.reserve(NumDeadBlocks);
      ToBeDeletedBBs.append(ToBeDeletedBlocks.begin(), ToBeDeletedBlocks.end());
      // Actually we do not delete the blocks but squash them into a single
      // unreachable but untangling branches that jump here is something we need
      // to do in a more generic way.
      DetatchDeadBlocks(ToBeDeletedBBs, nullptr);
      STATS_DECL(AAIsDead, BasicBlock, "Number of dead basic blocks deleted.");
      BUILD_STAT_NAME(AAIsDead, BasicBlock) += ToBeDeletedBlocks.size();
    }

    // Identify dead internal functions and delete them. This happens outside
    // the other fixpoint analysis as we might treat potentially dead functions
    // as live to lower the number of iterations. If they happen to be dead, the
    // below fixpoint loop will identify and eliminate them.
    SmallVector<Function *, 8> InternalFns;
    for (Function &F : M)
      if (F.hasLocalLinkage())
        InternalFns.push_back(&F);

    bool FoundDeadFn = true;
    while (FoundDeadFn) {
      FoundDeadFn = false;
      for (unsigned u = 0, e = InternalFns.size(); u < e; ++u) {
        Function *F = InternalFns[u];
        if (!F)
          continue;

        if (!checkForAllCallSites(
                [this](AbstractCallSite ACS) {
                  return ToBeDeletedFunctions.count(
                      ACS.getInstruction()->getFunction());
                },
                *F, true, nullptr))
          continue;

        ToBeDeletedFunctions.insert(F);
        InternalFns[u] = nullptr;
        FoundDeadFn = true;
      }
    }
  }

  STATS_DECL(AAIsDead, Function, "Number of dead functions deleted.");
  BUILD_STAT_NAME(AAIsDead, Function) += ToBeDeletedFunctions.size();

  // Rewrite the functions as requested during manifest.
  ManifestChange = ManifestChange | rewriteFunctionSignatures();

  for (Function *Fn : ToBeDeletedFunctions) {
    Fn->deleteBody();
    Fn->replaceAllUsesWith(UndefValue::get(Fn->getType()));
    Fn->eraseFromParent();
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

bool Attributor::registerFunctionSignatureRewrite(
    Argument &Arg, ArrayRef<Type *> ReplacementTypes,
    ArgumentReplacementInfo::CalleeRepairCBTy &&CalleeRepairCB,
    ArgumentReplacementInfo::ACSRepairCBTy &&ACSRepairCB) {

  auto CallSiteCanBeChanged = [](AbstractCallSite ACS) {
    // Forbid must-tail calls for now.
    return !ACS.isCallbackCall() && !ACS.getCallSite().isMustTailCall();
  };

  Function *Fn = Arg.getParent();
  // Avoid var-arg functions for now.
  if (Fn->isVarArg()) {
    LLVM_DEBUG(dbgs() << "[Attributor] Cannot rewrite var-args functions\n");
    return false;
  }

  // Avoid functions with complicated argument passing semantics.
  AttributeList FnAttributeList = Fn->getAttributes();
  if (FnAttributeList.hasAttrSomewhere(Attribute::Nest) ||
      FnAttributeList.hasAttrSomewhere(Attribute::StructRet) ||
      FnAttributeList.hasAttrSomewhere(Attribute::InAlloca)) {
    LLVM_DEBUG(
        dbgs() << "[Attributor] Cannot rewrite due to complex attribute\n");
    return false;
  }

  // Avoid callbacks for now.
  if (!checkForAllCallSites(CallSiteCanBeChanged, *Fn, true, nullptr)) {
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
  bool AnyDead;
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(*Fn);
  if (!checkForAllInstructionsImpl(OpcodeInstMap, InstPred, nullptr, AnyDead,
                                   {Instruction::Call})) {
    LLVM_DEBUG(dbgs() << "[Attributor] Cannot rewrite due to instructions\n");
    return false;
  }

  SmallVectorImpl<ArgumentReplacementInfo *> &ARIs = ArgumentReplacementMap[Fn];
  if (ARIs.size() == 0)
    ARIs.resize(Fn->arg_size());

  // If we have a replacement already with less than or equal new arguments,
  // ignore this request.
  ArgumentReplacementInfo *&ARI = ARIs[Arg.getArgNo()];
  if (ARI && ARI->getNumReplacementArgs() <= ReplacementTypes.size()) {
    LLVM_DEBUG(dbgs() << "[Attributor] Existing rewrite is preferred\n");
    return false;
  }

  // If we have a replacement already but we like the new one better, delete
  // the old.
  if (ARI)
    delete ARI;

  // Remember the replacement.
  ARI = new ArgumentReplacementInfo(*this, Arg, ReplacementTypes,
                                    std::move(CalleeRepairCB),
                                    std::move(ACSRepairCB));

  return true;
}

ChangeStatus Attributor::rewriteFunctionSignatures() {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  for (auto &It : ArgumentReplacementMap) {
    Function *OldFn = It.getFirst();

    // Deleted functions do not require rewrites.
    if (ToBeDeletedFunctions.count(OldFn))
      continue;

    const SmallVectorImpl<ArgumentReplacementInfo *> &ARIs = It.getSecond();
    assert(ARIs.size() == OldFn->arg_size() && "Inconsistent state!");

    SmallVector<Type *, 16> NewArgumentTypes;
    SmallVector<AttributeSet, 16> NewArgumentAttributes;

    // Collect replacement argument types and copy over existing attributes.
    AttributeList OldFnAttributeList = OldFn->getAttributes();
    for (Argument &Arg : OldFn->args()) {
      if (ArgumentReplacementInfo *ARI = ARIs[Arg.getArgNo()]) {
        NewArgumentTypes.append(ARI->ReplacementTypes.begin(),
                                ARI->ReplacementTypes.end());
        NewArgumentAttributes.append(ARI->getNumReplacementArgs(),
                                     AttributeSet());
      } else {
        NewArgumentTypes.push_back(Arg.getType());
        NewArgumentAttributes.push_back(
            OldFnAttributeList.getParamAttributes(Arg.getArgNo()));
      }
    }

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
        Ctx, OldFnAttributeList.getFnAttributes(),
        OldFnAttributeList.getRetAttributes(), NewArgumentAttributes));

    // Since we have now created the new function, splice the body of the old
    // function right into the new function, leaving the old rotting hulk of the
    // function empty.
    NewFn->getBasicBlockList().splice(NewFn->begin(),
                                      OldFn->getBasicBlockList());

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
        if (ArgumentReplacementInfo *ARI = ARIs[OldArgNum]) {
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
              OldCallAttributeList.getParamAttributes(OldArgNum));
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
      uint64_t W;
      if (OldCB->extractProfTotalWeight(W))
        NewCB->setProfWeight(W);
      NewCB->setCallingConv(OldCB->getCallingConv());
      NewCB->setDebugLoc(OldCB->getDebugLoc());
      NewCB->takeName(OldCB);
      NewCB->setAttributes(AttributeList::get(
          Ctx, OldCallAttributeList.getFnAttributes(),
          OldCallAttributeList.getRetAttributes(), NewArgOperandAttributes));

      CallSitePairs.push_back({OldCB, NewCB});
      return true;
    };

    // Use the CallSiteReplacementCreator to create replacement call sites.
    bool Success =
        checkForAllCallSites(CallSiteReplacementCreator, *OldFn, true, nullptr);
    (void)Success;
    assert(Success && "Assumed call site replacement to succeed!");

    // Rewire the arguments.
    auto OldFnArgIt = OldFn->arg_begin();
    auto NewFnArgIt = NewFn->arg_begin();
    for (unsigned OldArgNum = 0; OldArgNum < ARIs.size();
         ++OldArgNum, ++OldFnArgIt) {
      if (ArgumentReplacementInfo *ARI = ARIs[OldArgNum]) {
        if (ARI->CalleeRepairCB)
          ARI->CalleeRepairCB(*ARI, *NewFn, NewFnArgIt);
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
      OldCB.replaceAllUsesWith(&NewCB);
      OldCB.eraseFromParent();
    }

    ToBeDeletedFunctions.insert(OldFn);

    Changed = ChangeStatus::CHANGED;
  }

  return Changed;
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
    case Instruction::AtomicRMW:
    case Instruction::AtomicCmpXchg:
    case Instruction::Br:
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

void Attributor::recordDependence(const AbstractAttribute &FromAA,
                                  const AbstractAttribute &ToAA,
                                  DepClassTy DepClass) {
  if (FromAA.getState().isAtFixpoint())
    return;

  if (DepClass == DepClassTy::REQUIRED)
    QueryMap[&FromAA].RequiredAAs.insert(
        const_cast<AbstractAttribute *>(&ToAA));
  else
    QueryMap[&FromAA].OptionalAAs.insert(
        const_cast<AbstractAttribute *>(&ToAA));
  QueriedNonFixAA = true;
}

void Attributor::identifyDefaultAbstractAttributes(Function &F) {
  if (!VisitedFunctions.insert(&F).second)
    return;
  if (F.isDeclaration())
    return;

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

      // Every argument with pointer type might be marked
      // "readnone/readonly/writeonly/..."
      getOrCreateAAFor<AAMemoryBehavior>(ArgPos);

      // Every argument with pointer type might be marked nofree.
      getOrCreateAAFor<AANoFree>(ArgPos);
    }
  }

  auto CallSitePred = [&](Instruction &I) -> bool {
    CallSite CS(&I);
    if (Function *Callee = CS.getCalledFunction()) {
      // Skip declerations except if annotations on their call sites were
      // explicitly requested.
      if (!AnnotateDeclarationCallSites && Callee->isDeclaration() &&
          !Callee->hasMetadata(LLVMContext::MD_callback))
        return true;

      if (!Callee->getReturnType()->isVoidTy() && !CS->use_empty()) {

        IRPosition CSRetPos = IRPosition::callsite_returned(CS);

        // Call site return values might be dead.
        getOrCreateAAFor<AAIsDead>(CSRetPos);

        // Call site return integer values might be limited by a constant range.
        if (Callee->getReturnType()->isIntegerTy()) {
          getOrCreateAAFor<AAValueConstantRange>(CSRetPos);
        }
      }

      for (int i = 0, e = CS.getNumArgOperands(); i < e; i++) {

        IRPosition CSArgPos = IRPosition::callsite_argument(CS, i);

        // Every call site argument might be dead.
        getOrCreateAAFor<AAIsDead>(CSArgPos);

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

        // Call site argument attribute
        // "readnone/readonly/writeonly/..."
        getOrCreateAAFor<AAMemoryBehavior>(CSArgPos);

        // Call site argument attribute "nofree".
        getOrCreateAAFor<AANoFree>(CSArgPos);
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

template <typename base_ty, base_ty BestState, base_ty WorstState>
raw_ostream &
llvm::operator<<(raw_ostream &OS,
                 const IntegerStateBase<base_ty, BestState, WorstState> &S) {
  return OS << "(" << S.getKnown() << "-" << S.getAssumed() << ")"
            << static_cast<const AbstractState &>(S);
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

    // We look at internal functions only on-demand but if any use is not a
    // direct call, we have to do it eagerly.
    if (F.hasLocalLinkage()) {
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

  bool Changed = A.run(M) == ChangeStatus::CHANGED;
  assert(!verifyModule(M, &errs()) && "Module verification failed!");
  return Changed;
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
const char AAMemoryBehavior::ID = 0;
const char AAValueConstantRange::ID = 0;

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

CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANonNull)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoAlias)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AADereferenceable)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAAlign)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoCapture)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAValueConstantRange)

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

INITIALIZE_PASS_BEGIN(AttributorLegacyPass, "attributor",
                      "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AttributorLegacyPass, "attributor",
                    "Deduce and propagate attributes", false, false)
