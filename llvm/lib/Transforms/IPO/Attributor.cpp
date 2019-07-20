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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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
STATISTIC(NumFnNoUnwind, "Number of functions marked nounwind");

STATISTIC(NumFnUniqueReturned, "Number of function with unique return");
STATISTIC(NumFnKnownReturns, "Number of function with known return values");
STATISTIC(NumFnArgumentReturned,
          "Number of function arguments marked returned");
STATISTIC(NumFnNoSync, "Number of functions marked nosync");
STATISTIC(NumFnNoFree, "Number of functions marked nofree");
STATISTIC(NumFnReturnedNonNull,
          "Number of function return values marked nonnull");
STATISTIC(NumFnArgumentNonNull, "Number of function arguments marked nonnull");
STATISTIC(NumCSArgumentNonNull, "Number of call site arguments marked nonnull");
STATISTIC(NumFnWillReturn, "Number of functions marked willreturn");

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

/// Helper to adjust the statistics.
static void bookkeeping(AbstractAttribute::ManifestPosition MP,
                        const Attribute &Attr) {
  if (!AreStatisticsEnabled())
    return;

  if (!Attr.isEnumAttribute())
    return;
  switch (Attr.getKindAsEnum()) {
  case Attribute::NoUnwind:
    NumFnNoUnwind++;
    return;
  case Attribute::Returned:
    NumFnArgumentReturned++;
    return;
  case Attribute::NoSync:
    NumFnNoSync++;
    break;
  case Attribute::NoFree:
    NumFnNoFree++;
    break;
  case Attribute::NonNull:
    switch (MP) {
    case AbstractAttribute::MP_RETURNED:
      NumFnReturnedNonNull++;
      break;
    case AbstractAttribute::MP_ARGUMENT:
      NumFnArgumentNonNull++;
      break;
    case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
      NumCSArgumentNonNull++;
      break;
    default:
      break;
    }
    break;
  case Attribute::WillReturn:
    NumFnWillReturn++;
    break;
  default:
    return;
  }
}

template <typename StateTy>
using followValueCB_t = std::function<bool(Value *, StateTy &State)>;
template <typename StateTy>
using visitValueCB_t = std::function<void(Value *, StateTy &State)>;

/// Recursively visit all values that might become \p InitV at some point. This
/// will be done by looking through cast instructions, selects, phis, and calls
/// with the "returned" attribute. The callback \p FollowValueCB is asked before
/// a potential origin value is looked at. If no \p FollowValueCB is passed, a
/// default one is used that will make sure we visit every value only once. Once
/// we cannot look through the value any further, the callback \p VisitValueCB
/// is invoked and passed the current value and the \p State. To limit how much
/// effort is invested, we will never visit more than \p MaxValues values.
template <typename StateTy>
static bool genericValueTraversal(
    Value *InitV, StateTy &State, visitValueCB_t<StateTy> &VisitValueCB,
    followValueCB_t<StateTy> *FollowValueCB = nullptr, int MaxValues = 8) {

  SmallPtrSet<Value *, 16> Visited;
  followValueCB_t<bool> DefaultFollowValueCB = [&](Value *Val, bool &) {
    return Visited.insert(Val).second;
  };

  if (!FollowValueCB)
    FollowValueCB = &DefaultFollowValueCB;

  SmallVector<Value *, 16> Worklist;
  Worklist.push_back(InitV);

  int Iteration = 0;
  do {
    Value *V = Worklist.pop_back_val();

    // Check if we should process the current value. To prevent endless
    // recursion keep a record of the values we followed!
    if (!(*FollowValueCB)(V, State))
      continue;

    // Make sure we limit the compile time for complex expressions.
    if (Iteration++ >= MaxValues)
      return false;

    // Explicitly look through calls with a "returned" attribute if we do
    // not have a pointer as stripPointerCasts only works on them.
    if (V->getType()->isPointerTy()) {
      V = V->stripPointerCasts();
    } else {
      CallSite CS(V);
      if (CS && CS.getCalledFunction()) {
        Value *NewV = nullptr;
        for (Argument &Arg : CS.getCalledFunction()->args())
          if (Arg.hasReturnedAttr()) {
            NewV = CS.getArgOperand(Arg.getArgNo());
            break;
          }
        if (NewV) {
          Worklist.push_back(NewV);
          continue;
        }
      }
    }

    // Look through select instructions, visit both potential values.
    if (auto *SI = dyn_cast<SelectInst>(V)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    // Look through phi nodes, visit all operands.
    if (auto *PHI = dyn_cast<PHINode>(V)) {
      Worklist.append(PHI->op_begin(), PHI->op_end());
      continue;
    }

    // Once a leaf is reached we inform the user through the callback.
    VisitValueCB(V, State);
  } while (!Worklist.empty());

  // All values have been visited.
  return true;
}

/// Helper to identify the correct offset into an attribute list.
static unsigned getAttrIndex(AbstractAttribute::ManifestPosition MP,
                             unsigned ArgNo = 0) {
  switch (MP) {
  case AbstractAttribute::MP_ARGUMENT:
  case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
    return ArgNo + AttributeList::FirstArgIndex;
  case AbstractAttribute::MP_FUNCTION:
    return AttributeList::FunctionIndex;
  case AbstractAttribute::MP_RETURNED:
    return AttributeList::ReturnIndex;
  }
  llvm_unreachable("Unknown manifest position!");
}

/// Return true if \p New is equal or worse than \p Old.
static bool isEqualOrWorse(const Attribute &New, const Attribute &Old) {
  if (!Old.isIntAttribute())
    return true;

  return Old.getValueAsInt() >= New.getValueAsInt();
}

/// Return true if the information provided by \p Attr was added to the
/// attribute list \p Attrs. This is only the case if it was not already present
/// in \p Attrs at the position describe by \p MP and \p ArgNo.
static bool addIfNotExistent(LLVMContext &Ctx, const Attribute &Attr,
                             AttributeList &Attrs,
                             AbstractAttribute::ManifestPosition MP,
                             unsigned ArgNo = 0) {
  unsigned AttrIdx = getAttrIndex(MP, ArgNo);

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

ChangeStatus AbstractAttribute::manifest(Attributor &A) {
  assert(getState().isValidState() &&
         "Attempted to manifest an invalid state!");
  assert(getAssociatedValue() &&
         "Attempted to manifest an attribute without associated value!");

  ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
  SmallVector<Attribute, 4> DeducedAttrs;
  getDeducedAttributes(DeducedAttrs);

  Function &ScopeFn = getAnchorScope();
  LLVMContext &Ctx = ScopeFn.getContext();
  ManifestPosition MP = getManifestPosition();

  AttributeList Attrs;
  SmallVector<unsigned, 4> ArgNos;

  // In the following some generic code that will manifest attributes in
  // DeducedAttrs if they improve the current IR. Due to the different
  // annotation positions we use the underlying AttributeList interface.
  // Note that MP_CALL_SITE_ARGUMENT can annotate multiple locations.

  switch (MP) {
  case MP_ARGUMENT:
    ArgNos.push_back(cast<Argument>(getAssociatedValue())->getArgNo());
    Attrs = ScopeFn.getAttributes();
    break;
  case MP_FUNCTION:
  case MP_RETURNED:
    ArgNos.push_back(0);
    Attrs = ScopeFn.getAttributes();
    break;
  case MP_CALL_SITE_ARGUMENT: {
    CallSite CS(&getAnchoredValue());
    for (unsigned u = 0, e = CS.getNumArgOperands(); u != e; u++)
      if (CS.getArgOperand(u) == getAssociatedValue())
        ArgNos.push_back(u);
    Attrs = CS.getAttributes();
  }
  }

  for (const Attribute &Attr : DeducedAttrs) {
    for (unsigned ArgNo : ArgNos) {
      if (!addIfNotExistent(Ctx, Attr, Attrs, MP, ArgNo))
        continue;

      HasChanged = ChangeStatus::CHANGED;
      bookkeeping(MP, Attr);
    }
  }

  if (HasChanged == ChangeStatus::UNCHANGED)
    return HasChanged;

  switch (MP) {
  case MP_ARGUMENT:
  case MP_FUNCTION:
  case MP_RETURNED:
    ScopeFn.setAttributes(Attrs);
    break;
  case MP_CALL_SITE_ARGUMENT:
    CallSite(&getAnchoredValue()).setAttributes(Attrs);
  }

  return HasChanged;
}

Function &AbstractAttribute::getAnchorScope() {
  Value &V = getAnchoredValue();
  if (isa<Function>(V))
    return cast<Function>(V);
  if (isa<Argument>(V))
    return *cast<Argument>(V).getParent();
  if (isa<Instruction>(V))
    return *cast<Instruction>(V).getFunction();
  llvm_unreachable("No scope for anchored value found!");
}

const Function &AbstractAttribute::getAnchorScope() const {
  return const_cast<AbstractAttribute *>(this)->getAnchorScope();
}

/// -----------------------NoUnwind Function Attribute--------------------------

struct AANoUnwindFunction : AANoUnwind, BooleanState {

  AANoUnwindFunction(Function &F, InformationCache &InfoCache)
      : AANoUnwind(F, InfoCache) {}

  /// See AbstractAttribute::getState()
  /// {
  AbstractState &getState() override { return *this; }
  const AbstractState &getState() const override { return *this; }
  /// }

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override { return MP_FUNCTION; }

  const std::string getAsStr() const override {
    return getAssumed() ? "nounwind" : "may-unwind";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AANoUnwind::isAssumedNoUnwind().
  bool isAssumedNoUnwind() const override { return getAssumed(); }

  /// See AANoUnwind::isKnownNoUnwind().
  bool isKnownNoUnwind() const override { return getKnown(); }
};

ChangeStatus AANoUnwindFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  // The map from instruction opcodes to those instructions in the function.
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);
  auto Opcodes = {
      (unsigned)Instruction::Invoke,      (unsigned)Instruction::CallBr,
      (unsigned)Instruction::Call,        (unsigned)Instruction::CleanupRet,
      (unsigned)Instruction::CatchSwitch, (unsigned)Instruction::Resume};

  for (unsigned Opcode : Opcodes) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      if (!I->mayThrow())
        continue;

      auto *NoUnwindAA = A.getAAFor<AANoUnwind>(*this, *I);

      if (!NoUnwindAA || !NoUnwindAA->isAssumedNoUnwind()) {
        indicatePessimisticFixpoint();
        return ChangeStatus::CHANGED;
      }
    }
  }
  return ChangeStatus::UNCHANGED;
}

/// --------------------- Function Return Values -------------------------------

/// "Attribute" that collects all potential returned values and the return
/// instructions that they arise from.
///
/// If there is a unique returned value R, the manifest method will:
///   - mark R with the "returned" attribute, if R is an argument.
class AAReturnedValuesImpl final : public AAReturnedValues, AbstractState {

  /// Mapping of values potentially returned by the associated function to the
  /// return instructions that might return them.
  DenseMap<Value *, SmallPtrSet<ReturnInst *, 2>> ReturnedValues;

  /// State flags
  ///
  ///{
  bool IsFixed;
  bool IsValidState;
  bool HasOverdefinedReturnedCalls;
  ///}

  /// Collect values that could become \p V in the set \p Values, each mapped to
  /// \p ReturnInsts.
  void collectValuesRecursively(
      Attributor &A, Value *V, SmallPtrSetImpl<ReturnInst *> &ReturnInsts,
      DenseMap<Value *, SmallPtrSet<ReturnInst *, 2>> &Values) {

    visitValueCB_t<bool> VisitValueCB = [&](Value *Val, bool &) {
      assert(!isa<Instruction>(Val) ||
             &getAnchorScope() == cast<Instruction>(Val)->getFunction());
      Values[Val].insert(ReturnInsts.begin(), ReturnInsts.end());
    };

    bool UnusedBool;
    bool Success = genericValueTraversal(V, UnusedBool, VisitValueCB);

    // If we did abort the above traversal we haven't see all the values.
    // Consequently, we cannot know if the information we would derive is
    // accurate so we give up early.
    if (!Success)
      indicatePessimisticFixpoint();
  }

public:
  /// See AbstractAttribute::AbstractAttribute(...).
  AAReturnedValuesImpl(Function &F, InformationCache &InfoCache)
      : AAReturnedValues(F, InfoCache) {
    // We do not have an associated argument yet.
    AssociatedVal = nullptr;
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    // Reset the state.
    AssociatedVal = nullptr;
    IsFixed = false;
    IsValidState = true;
    HasOverdefinedReturnedCalls = false;
    ReturnedValues.clear();

    Function &F = cast<Function>(getAnchoredValue());

    // The map from instruction opcodes to those instructions in the function.
    auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

    // Look through all arguments, if one is marked as returned we are done.
    for (Argument &Arg : F.args()) {
      if (Arg.hasReturnedAttr()) {

        auto &ReturnInstSet = ReturnedValues[&Arg];
        for (Instruction *RI : OpcodeInstMap[Instruction::Ret])
          ReturnInstSet.insert(cast<ReturnInst>(RI));

        indicateOptimisticFixpoint();
        return;
      }
    }

    // If no argument was marked as returned we look at all return instructions
    // and collect potentially returned values.
    for (Instruction *RI : OpcodeInstMap[Instruction::Ret]) {
      SmallPtrSet<ReturnInst *, 1> RISet({cast<ReturnInst>(RI)});
      collectValuesRecursively(A, cast<ReturnInst>(RI)->getReturnValue(), RISet,
                               ReturnedValues);
    }
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override;

  /// See AbstractAttribute::getState(...).
  AbstractState &getState() override { return *this; }

  /// See AbstractAttribute::getState(...).
  const AbstractState &getState() const override { return *this; }

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override { return MP_ARGUMENT; }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;

  /// Return the number of potential return values, -1 if unknown.
  size_t getNumReturnValues() const {
    return isValidState() ? ReturnedValues.size() : -1;
  }

  /// Return an assumed unique return value if a single candidate is found. If
  /// there cannot be one, return a nullptr. If it is not clear yet, return the
  /// Optional::NoneType.
  Optional<Value *> getAssumedUniqueReturnValue() const;

  /// See AbstractState::checkForallReturnedValues(...).
  bool
  checkForallReturnedValues(std::function<bool(Value &)> &Pred) const override;

  /// Pretty print the attribute similar to the IR representation.
  const std::string getAsStr() const override;

  /// See AbstractState::isAtFixpoint().
  bool isAtFixpoint() const override { return IsFixed; }

  /// See AbstractState::isValidState().
  bool isValidState() const override { return IsValidState; }

  /// See AbstractState::indicateOptimisticFixpoint(...).
  void indicateOptimisticFixpoint() override {
    IsFixed = true;
    IsValidState &= true;
  }
  void indicatePessimisticFixpoint() override {
    IsFixed = true;
    IsValidState = false;
  }
};

ChangeStatus AAReturnedValuesImpl::manifest(Attributor &A) {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  // Bookkeeping.
  assert(isValidState());
  NumFnKnownReturns++;

  // Check if we have an assumed unique return value that we could manifest.
  Optional<Value *> UniqueRV = getAssumedUniqueReturnValue();

  if (!UniqueRV.hasValue() || !UniqueRV.getValue())
    return Changed;

  // Bookkeeping.
  NumFnUniqueReturned++;

  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.getValue())) {
    AssociatedVal = UniqueRVArg;
    Changed = AbstractAttribute::manifest(A) | Changed;
  }

  return Changed;
}

const std::string AAReturnedValuesImpl::getAsStr() const {
  return (isAtFixpoint() ? "returns(#" : "may-return(#") +
         (isValidState() ? std::to_string(getNumReturnValues()) : "?") + ")";
}

Optional<Value *> AAReturnedValuesImpl::getAssumedUniqueReturnValue() const {
  // If checkForallReturnedValues provides a unique value, ignoring potential
  // undef values that can also be present, it is assumed to be the actual
  // return value and forwarded to the caller of this method. If there are
  // multiple, a nullptr is returned indicating there cannot be a unique
  // returned value.
  Optional<Value *> UniqueRV;

  std::function<bool(Value &)> Pred = [&](Value &RV) -> bool {
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

  if (!checkForallReturnedValues(Pred))
    UniqueRV = nullptr;

  return UniqueRV;
}

bool AAReturnedValuesImpl::checkForallReturnedValues(
    std::function<bool(Value &)> &Pred) const {
  if (!isValidState())
    return false;

  // Check all returned values but ignore call sites as long as we have not
  // encountered an overdefined one during an update.
  for (auto &It : ReturnedValues) {
    Value *RV = It.first;

    ImmutableCallSite ICS(RV);
    if (ICS && !HasOverdefinedReturnedCalls)
      continue;

    if (!Pred(*RV))
      return false;
  }

  return true;
}

ChangeStatus AAReturnedValuesImpl::updateImpl(Attributor &A) {

  // Check if we know of any values returned by the associated function,
  // if not, we are done.
  if (getNumReturnValues() == 0) {
    indicateOptimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  // Check if any of the returned values is a call site we can refine.
  decltype(ReturnedValues) AddRVs;
  bool HasCallSite = false;

  // Look at all returned call sites.
  for (auto &It : ReturnedValues) {
    SmallPtrSet<ReturnInst *, 2> &ReturnInsts = It.second;
    Value *RV = It.first;
    LLVM_DEBUG(dbgs() << "[AAReturnedValues] Potentially returned value " << *RV
                      << "\n");

    // Only call sites can change during an update, ignore the rest.
    CallSite RetCS(RV);
    if (!RetCS)
      continue;

    // For now, any call site we see will prevent us from directly fixing the
    // state. However, if the information on the callees is fixed, the call
    // sites will be removed and we will fix the information for this state.
    HasCallSite = true;

    // Try to find a assumed unique return value for the called function.
    auto *RetCSAA = A.getAAFor<AAReturnedValuesImpl>(*this, *RV);
    if (!RetCSAA) {
      HasOverdefinedReturnedCalls = true;
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned call site (" << *RV
                        << ") with " << (RetCSAA ? "invalid" : "no")
                        << " associated state\n");
      continue;
    }

    // Try to find a assumed unique return value for the called function.
    Optional<Value *> AssumedUniqueRV = RetCSAA->getAssumedUniqueReturnValue();

    // If no assumed unique return value was found due to the lack of
    // candidates, we may need to resolve more calls (through more update
    // iterations) or the called function will not return. Either way, we simply
    // stick with the call sites as return values. Because there were not
    // multiple possibilities, we do not treat it as overdefined.
    if (!AssumedUniqueRV.hasValue())
      continue;

    // If multiple, non-refinable values were found, there cannot be a unique
    // return value for the called function. The returned call is overdefined!
    if (!AssumedUniqueRV.getValue()) {
      HasOverdefinedReturnedCalls = true;
      LLVM_DEBUG(dbgs() << "[AAReturnedValues] Returned call site has multiple "
                           "potentially returned values\n");
      continue;
    }

    LLVM_DEBUG({
      bool UniqueRVIsKnown = RetCSAA->isAtFixpoint();
      dbgs() << "[AAReturnedValues] Returned call site "
             << (UniqueRVIsKnown ? "known" : "assumed")
             << " unique return value: " << *AssumedUniqueRV << "\n";
    });

    // The assumed unique return value.
    Value *AssumedRetVal = AssumedUniqueRV.getValue();

    // If the assumed unique return value is an argument, lookup the matching
    // call site operand and recursively collect new returned values.
    // If it is not an argument, it is just put into the set of returned values
    // as we would have already looked through casts, phis, and similar values.
    if (Argument *AssumedRetArg = dyn_cast<Argument>(AssumedRetVal))
      collectValuesRecursively(A,
                               RetCS.getArgOperand(AssumedRetArg->getArgNo()),
                               ReturnInsts, AddRVs);
    else
      AddRVs[AssumedRetVal].insert(ReturnInsts.begin(), ReturnInsts.end());
  }

  // Keep track of any change to trigger updates on dependent attributes.
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  for (auto &It : AddRVs) {
    assert(!It.second.empty() && "Entry does not add anything.");
    auto &ReturnInsts = ReturnedValues[It.first];
    for (ReturnInst *RI : It.second)
      if (ReturnInsts.insert(RI).second) {
        LLVM_DEBUG(dbgs() << "[AAReturnedValues] Add new returned value "
                          << *It.first << " => " << *RI << "\n");
        Changed = ChangeStatus::CHANGED;
      }
  }

  // If there is no call site in the returned values we are done.
  if (!HasCallSite) {
    indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  return Changed;
}

/// ------------------------ NoSync Function Attribute -------------------------

struct AANoSyncFunction : AANoSync, BooleanState {

  AANoSyncFunction(Function &F, InformationCache &InfoCache)
      : AANoSync(F, InfoCache) {}

  /// See AbstractAttribute::getState()
  /// {
  AbstractState &getState() override { return *this; }
  const AbstractState &getState() const override { return *this; }
  /// }

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override { return MP_FUNCTION; }

  const std::string getAsStr() const override {
    return getAssumed() ? "nosync" : "may-sync";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AANoSync::isAssumedNoSync()
  bool isAssumedNoSync() const override { return getAssumed(); }

  /// See AANoSync::isKnownNoSync()
  bool isKnownNoSync() const override { return getKnown(); }

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

bool AANoSyncFunction::isNonRelaxedAtomic(Instruction *I) {
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
bool AANoSyncFunction::isNoSyncIntrinsic(Instruction *I) {
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

bool AANoSyncFunction::isVolatile(Instruction *I) {
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

ChangeStatus AANoSyncFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  /// We are looking for volatile instructions or Non-Relaxed atomics.
  /// FIXME: We should ipmrove the handling of intrinsics.
  for (Instruction *I : InfoCache.getReadOrWriteInstsForFunction(F)) {
    ImmutableCallSite ICS(I);
    auto *NoSyncAA = A.getAAFor<AANoSyncFunction>(*this, *I);

    if (isa<IntrinsicInst>(I) && isNoSyncIntrinsic(I))
      continue;

    if (ICS && (!NoSyncAA || !NoSyncAA->isAssumedNoSync()) &&
        !ICS.hasFnAttr(Attribute::NoSync)) {
      indicatePessimisticFixpoint();
      return ChangeStatus::CHANGED;
    }

    if (ICS)
      continue;

    if (!isVolatile(I) && !isNonRelaxedAtomic(I))
      continue;

    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);
  auto Opcodes = {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
                  (unsigned)Instruction::Call};

  for (unsigned Opcode : Opcodes) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      // At this point we handled all read/write effects and they are all
      // nosync, so they can be skipped.
      if (I->mayReadOrWriteMemory())
        continue;

      ImmutableCallSite ICS(I);

      // non-convergent and readnone imply nosync.
      if (!ICS.isConvergent())
        continue;

      indicatePessimisticFixpoint();
      return ChangeStatus::CHANGED;
    }
  }

  return ChangeStatus::UNCHANGED;
}

/// ------------------------ No-Free Attributes ----------------------------

struct AANoFreeFunction : AbstractAttribute, BooleanState {

  /// See AbstractAttribute::AbstractAttribute(...).
  AANoFreeFunction(Function &F, InformationCache &InfoCache)
      : AbstractAttribute(F, InfoCache) {}

  /// See AbstractAttribute::getState()
  ///{
  AbstractState &getState() override { return *this; }
  const AbstractState &getState() const override { return *this; }
  ///}

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override { return MP_FUNCTION; }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nofree" : "may-free";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::getAttrKind().
  Attribute::AttrKind getAttrKind() const override { return ID; }

  /// Return true if "nofree" is assumed.
  bool isAssumedNoFree() const { return getAssumed(); }

  /// Return true if "nofree" is known.
  bool isKnownNoFree() const { return getKnown(); }

  /// The identifier used by the Attributor for this class of attributes.
  static constexpr Attribute::AttrKind ID = Attribute::NoFree;
};

ChangeStatus AANoFreeFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  // The map from instruction opcodes to those instructions in the function.
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

  for (unsigned Opcode :
       {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
        (unsigned)Instruction::Call}) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {

      auto ICS = ImmutableCallSite(I);
      auto *NoFreeAA = A.getAAFor<AANoFreeFunction>(*this, *I);

      if ((!NoFreeAA || !NoFreeAA->isAssumedNoFree()) &&
          !ICS.hasFnAttr(Attribute::NoFree)) {
        indicatePessimisticFixpoint();
        return ChangeStatus::CHANGED;
      }
    }
  }
  return ChangeStatus::UNCHANGED;
}

/// ------------------------ NonNull Argument Attribute ------------------------
struct AANonNullImpl : AANonNull, BooleanState {

  AANonNullImpl(Value &V, InformationCache &InfoCache)
      : AANonNull(V, InfoCache) {}

  AANonNullImpl(Value *AssociatedVal, Value &AnchoredValue,
                InformationCache &InfoCache)
      : AANonNull(AssociatedVal, AnchoredValue, InfoCache) {}

  /// See AbstractAttribute::getState()
  /// {
  AbstractState &getState() override { return *this; }
  const AbstractState &getState() const override { return *this; }
  /// }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return getAssumed() ? "nonnull" : "may-null";
  }

  /// See AANonNull::isAssumedNonNull().
  bool isAssumedNonNull() const override { return getAssumed(); }

  /// See AANonNull::isKnownNonNull().
  bool isKnownNonNull() const override { return getKnown(); }

  /// Generate a predicate that checks if a given value is assumed nonnull.
  /// The generated function returns true if a value satisfies any of
  /// following conditions.
  /// (i) A value is known nonZero(=nonnull).
  /// (ii) A value is associated with AANonNull and its isAssumedNonNull() is
  /// true.
  std::function<bool(Value &)> generatePredicate(Attributor &);
};

std::function<bool(Value &)> AANonNullImpl::generatePredicate(Attributor &A) {
  // FIXME: The `AAReturnedValues` should provide the predicate with the
  // `ReturnInst` vector as well such that we can use the control flow sensitive
  // version of `isKnownNonZero`. This should fix `test11` in
  // `test/Transforms/FunctionAttrs/nonnull.ll`

  std::function<bool(Value &)> Pred = [&](Value &RV) -> bool {
    if (isKnownNonZero(&RV, getAnchorScope().getParent()->getDataLayout()))
      return true;

    auto *NonNullAA = A.getAAFor<AANonNull>(*this, RV);

    ImmutableCallSite ICS(&RV);

    if ((!NonNullAA || !NonNullAA->isAssumedNonNull()) &&
        (!ICS || !ICS.hasRetAttr(Attribute::NonNull)))
      return false;

    return true;
  };

  return Pred;
}

/// NonNull attribute for function return value.
struct AANonNullReturned : AANonNullImpl {

  AANonNullReturned(Function &F, InformationCache &InfoCache)
      : AANonNullImpl(F, InfoCache) {}

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override { return MP_RETURNED; }

  /// See AbstractAttriubute::initialize(...).
  void initialize(Attributor &A) override {
    Function &F = getAnchorScope();

    // Already nonnull.
    if (F.getAttributes().hasAttribute(AttributeList::ReturnIndex,
                                       Attribute::NonNull))
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AANonNullReturned::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  auto *AARetVal = A.getAAFor<AAReturnedValues>(*this, F);
  if (!AARetVal) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  std::function<bool(Value &)> Pred = this->generatePredicate(A);
  if (!AARetVal->checkForallReturnedValues(Pred)) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }
  return ChangeStatus::UNCHANGED;
}

/// NonNull attribute for function argument.
struct AANonNullArgument : AANonNullImpl {

  AANonNullArgument(Argument &A, InformationCache &InfoCache)
      : AANonNullImpl(A, InfoCache) {}

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override { return MP_ARGUMENT; }

  /// See AbstractAttriubute::initialize(...).
  void initialize(Attributor &A) override {
    Argument *Arg = cast<Argument>(getAssociatedValue());
    if (Arg->hasNonNullAttr())
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

/// NonNull attribute for a call site argument.
struct AANonNullCallSiteArgument : AANonNullImpl {

  /// See AANonNullImpl::AANonNullImpl(...).
  AANonNullCallSiteArgument(CallSite CS, unsigned ArgNo,
                            InformationCache &InfoCache)
      : AANonNullImpl(CS.getArgOperand(ArgNo), *CS.getInstruction(), InfoCache),
        ArgNo(ArgNo) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    CallSite CS(&getAnchoredValue());
    if (isKnownNonZero(getAssociatedValue(),
                       getAnchorScope().getParent()->getDataLayout()) ||
        CS.paramHasAttr(ArgNo, getAttrKind()))
      indicateOptimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(Attributor &A).
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override {
    return MP_CALL_SITE_ARGUMENT;
  };

  // Return argument index of associated value.
  int getArgNo() const { return ArgNo; }

private:
  unsigned ArgNo;
};
ChangeStatus AANonNullArgument::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();
  Argument &Arg = cast<Argument>(getAnchoredValue());

  unsigned ArgNo = Arg.getArgNo();

  // Callback function
  std::function<bool(CallSite)> CallSiteCheck = [&](CallSite CS) {
    assert(CS && "Sanity check: Call site was not initialized properly!");

    auto *NonNullAA = A.getAAFor<AANonNull>(*this, *CS.getInstruction(), ArgNo);

    // Check that NonNullAA is AANonNullCallSiteArgument.
    if (NonNullAA) {
      ImmutableCallSite ICS(&NonNullAA->getAnchoredValue());
      if (ICS && CS.getInstruction() == ICS.getInstruction())
        return NonNullAA->isAssumedNonNull();
      return false;
    }

    if (CS.paramHasAttr(ArgNo, Attribute::NonNull))
      return true;

    Value *V = CS.getArgOperand(ArgNo);
    if (isKnownNonZero(V, getAnchorScope().getParent()->getDataLayout()))
      return true;

    return false;
  };
  if (!A.checkForAllCallSites(F, CallSiteCheck, true)) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }
  return ChangeStatus::UNCHANGED;
}

ChangeStatus AANonNullCallSiteArgument::updateImpl(Attributor &A) {
  // NOTE: Never look at the argument of the callee in this method.
  //       If we do this, "nonnull" is always deduced because of the assumption.

  Value &V = *getAssociatedValue();

  auto *NonNullAA = A.getAAFor<AANonNull>(*this, V);

  if (!NonNullAA || !NonNullAA->isAssumedNonNull()) {
    indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  return ChangeStatus::UNCHANGED;
}

/// ------------------------ Will-Return Attributes ----------------------------

struct AAWillReturnImpl : public AAWillReturn, BooleanState {

  /// See AbstractAttribute::AbstractAttribute(...).
  AAWillReturnImpl(Function &F, InformationCache &InfoCache)
      : AAWillReturn(F, InfoCache) {}

  /// See AAWillReturn::isKnownWillReturn().
  bool isKnownWillReturn() const override { return getKnown(); }

  /// See AAWillReturn::isAssumedWillReturn().
  bool isAssumedWillReturn() const override { return getAssumed(); }

  /// See AbstractAttribute::getState(...).
  AbstractState &getState() override { return *this; }

  /// See AbstractAttribute::getState(...).
  const AbstractState &getState() const override { return *this; }

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    return getAssumed() ? "willreturn" : "may-noreturn";
  }
};

struct AAWillReturnFunction final : AAWillReturnImpl {

  /// See AbstractAttribute::AbstractAttribute(...).
  AAWillReturnFunction(Function &F, InformationCache &InfoCache)
      : AAWillReturnImpl(F, InfoCache) {}

  /// See AbstractAttribute::getManifestPosition().
  ManifestPosition getManifestPosition() const override {
    return MP_FUNCTION;
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override;

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

// Helper function that checks whether a function has any cycle.
// TODO: Replace with more efficent code
bool containsCycle(Function &F) {
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
bool containsPossiblyEndlessLoop(Function &F) { return containsCycle(F); }

void AAWillReturnFunction::initialize(Attributor &A) {
  Function &F = getAnchorScope();

  if (containsPossiblyEndlessLoop(F))
    indicatePessimisticFixpoint();
}

ChangeStatus AAWillReturnFunction::updateImpl(Attributor &A) {
  Function &F = getAnchorScope();

  // The map from instruction opcodes to those instructions in the function.
  auto &OpcodeInstMap = InfoCache.getOpcodeInstMapForFunction(F);

  for (unsigned Opcode :
       {(unsigned)Instruction::Invoke, (unsigned)Instruction::CallBr,
        (unsigned)Instruction::Call}) {
    for (Instruction *I : OpcodeInstMap[Opcode]) {
      auto ICS = ImmutableCallSite(I);

      if (ICS.hasFnAttr(Attribute::WillReturn))
        continue;

      auto *WillReturnAA = A.getAAFor<AAWillReturn>(*this, *I);
      if (!WillReturnAA || !WillReturnAA->isAssumedWillReturn()) {
        indicatePessimisticFixpoint();
        return ChangeStatus::CHANGED;
      }

      auto *NoRecurseAA = A.getAAFor<AANoRecurse>(*this, *I);

      // FIXME: (i) Prohibit any recursion for now.
      //        (ii) AANoRecurse isn't implemented yet so currently any call is
      //        regarded as having recursion.
      //       Code below should be
      //       if ((!NoRecurseAA || !NoRecurseAA->isAssumedNoRecurse()) &&
      if (!NoRecurseAA && !ICS.hasFnAttr(Attribute::NoRecurse)) {
        indicatePessimisticFixpoint();
        return ChangeStatus::CHANGED;
      }
    }
  }

  return ChangeStatus::UNCHANGED;
}

/// ----------------------------------------------------------------------------
///                               Attributor
/// ----------------------------------------------------------------------------

bool Attributor::checkForAllCallSites(Function &F,
                                      std::function<bool(CallSite)> &Pred,
                                      bool RequireAllCallSites) {
  // We can try to determine information from
  // the call sites. However, this is only possible all call sites are known,
  // hence the function has internal linkage.
  if (RequireAllCallSites && !F.hasInternalLinkage()) {
    LLVM_DEBUG(
        dbgs()
        << "Attributor: Function " << F.getName()
        << " has no internal linkage, hence not all call sites are known\n");
    return false;
  }

  for (const Use &U : F.uses()) {

    CallSite CS(U.getUser());
    if (!CS || !CS.isCallee(&U) || !CS.getCaller()->hasExactDefinition()) {
      if (!RequireAllCallSites)
        continue;

      LLVM_DEBUG(dbgs() << "Attributor: User " << *U.getUser()
                        << " is an invalid use of " << F.getName() << "\n");
      return false;
    }

    if (Pred(CS))
      continue;

    LLVM_DEBUG(dbgs() << "Attributor: Call site callback failed for "
                      << *CS.getInstruction() << "\n");
    return false;
  }

  return true;
}

ChangeStatus Attributor::run() {
  // Initialize all abstract attributes.
  for (AbstractAttribute *AA : AllAbstractAttributes)
    AA->initialize(*this);

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
      if (AA->update(*this) == ChangeStatus::CHANGED)
        ChangedAAs.push_back(AA);

    // Reset the work list and repopulate with the changed abstract attributes.
    // Note that dependent ones are added above.
    Worklist.clear();
    Worklist.insert(ChangedAAs.begin(), ChangedAAs.end());

  } while (!Worklist.empty() && ++IterationCounter < MaxFixpointIterations);

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

    // Manifest the state and record if we changed the IR.
    ChangeStatus LocalChange = AA->manifest(*this);
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

  return ManifestChange;
}

void Attributor::identifyDefaultAbstractAttributes(
    Function &F, InformationCache &InfoCache,
    DenseSet</* Attribute::AttrKind */ unsigned> *Whitelist) {

  // Every function can be nounwind.
  registerAA(*new AANoUnwindFunction(F, InfoCache));

  // Every function might be marked "nosync"
  registerAA(*new AANoSyncFunction(F, InfoCache));

  // Every function might be "no-free".
  registerAA(*new AANoFreeFunction(F, InfoCache));

  // Return attributes are only appropriate if the return type is non void.
  Type *ReturnType = F.getReturnType();
  if (!ReturnType->isVoidTy()) {
    // Argument attribute "returned" --- Create only one per function even
    // though it is an argument attribute.
    if (!Whitelist || Whitelist->count(AAReturnedValues::ID))
      registerAA(*new AAReturnedValuesImpl(F, InfoCache));

    // Every function with pointer return type might be marked nonnull.
    if (ReturnType->isPointerTy() &&
        (!Whitelist || Whitelist->count(AANonNullReturned::ID)))
      registerAA(*new AANonNullReturned(F, InfoCache));
  }

  // Every argument with pointer type might be marked nonnull.
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy())
      registerAA(*new AANonNullArgument(Arg, InfoCache));
  }

  // Every function might be "will-return".
  registerAA(*new AAWillReturnFunction(F, InfoCache));

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

        // Call site argument attribute "non-null".
        registerAA(*new AANonNullCallSiteArgument(CS, i, InfoCache), i);
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

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              AbstractAttribute::ManifestPosition AP) {
  switch (AP) {
  case AbstractAttribute::MP_ARGUMENT:
    return OS << "arg";
  case AbstractAttribute::MP_CALL_SITE_ARGUMENT:
    return OS << "cs_arg";
  case AbstractAttribute::MP_FUNCTION:
    return OS << "fn";
  case AbstractAttribute::MP_RETURNED:
    return OS << "fn_ret";
  }
  llvm_unreachable("Unknown attribute position!");
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractState &S) {
  return OS << (!S.isValidState() ? "top" : (S.isAtFixpoint() ? "fix" : ""));
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const AbstractAttribute &AA) {
  AA.print(OS);
  return OS;
}

void AbstractAttribute::print(raw_ostream &OS) const {
  OS << "[" << getManifestPosition() << "][" << getAsStr() << "]["
     << AnchoredVal.getName() << "]";
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
  Attributor A;
  InformationCache InfoCache;

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
    A.identifyDefaultAbstractAttributes(F, InfoCache);
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
INITIALIZE_PASS_BEGIN(AttributorLegacyPass, "attributor",
                      "Deduce and propagate attributes", false, false)
INITIALIZE_PASS_END(AttributorLegacyPass, "attributor",
                    "Deduce and propagate attributes", false, false)
