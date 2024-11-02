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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Assumptions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <numeric>

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

template <> unsigned llvm::PotentialLLVMValuesState::MaxPotentialValues = -1;

static cl::opt<unsigned, true> MaxPotentialValues(
    "attributor-max-potential-values", cl::Hidden,
    cl::desc("Maximum number of potential values to be "
             "tracked for each position."),
    cl::location(llvm::PotentialConstantIntValuesState::MaxPotentialValues),
    cl::init(7));

static cl::opt<int> MaxPotentialValuesIterations(
    "attributor-max-potential-values-iterations", cl::Hidden,
    cl::desc(
        "Maximum number of iterations we keep dismantling potential values."),
    cl::init(64));

static cl::opt<unsigned> MaxInterferingAccesses(
    "attributor-max-interfering-accesses", cl::Hidden,
    cl::desc("Maximum number of interfering accesses to "
             "check before assuming all might interfere."),
    cl::init(6));

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
PIPE_OPERATOR(AAInstanceInfo)
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
PIPE_OPERATOR(AAPotentialConstantValues)
PIPE_OPERATOR(AAPotentialValues)
PIPE_OPERATOR(AANoUndef)
PIPE_OPERATOR(AACallEdges)
PIPE_OPERATOR(AAFunctionReachability)
PIPE_OPERATOR(AAPointerInfo)
PIPE_OPERATOR(AAAssumptionInfo)

#undef PIPE_OPERATOR

template <>
ChangeStatus clampStateAndIndicateChange<DerefState>(DerefState &S,
                                                     const DerefState &R) {
  ChangeStatus CS0 =
      clampStateAndIndicateChange(S.DerefBytesState, R.DerefBytesState);
  ChangeStatus CS1 = clampStateAndIndicateChange(S.GlobalState, R.GlobalState);
  return CS0 | CS1;
}

} // namespace llvm

/// Checks if a type could have padding bytes.
static bool isDenselyPacked(Type *Ty, const DataLayout &DL) {
  // There is no size information, so be conservative.
  if (!Ty->isSized())
    return false;

  // If the alloc size is not equal to the storage size, then there are padding
  // bytes. For x86_fp80 on x86-64, size: 80 alloc size: 128.
  if (DL.getTypeSizeInBits(Ty) != DL.getTypeAllocSizeInBits(Ty))
    return false;

  // FIXME: This isn't the right way to check for padding in vectors with
  // non-byte-size elements.
  if (VectorType *SeqTy = dyn_cast<VectorType>(Ty))
    return isDenselyPacked(SeqTy->getElementType(), DL);

  // For array types, check for padding within members.
  if (ArrayType *SeqTy = dyn_cast<ArrayType>(Ty))
    return isDenselyPacked(SeqTy->getElementType(), DL);

  if (!isa<StructType>(Ty))
    return true;

  // Check for padding within and between elements of a struct.
  StructType *StructTy = cast<StructType>(Ty);
  const StructLayout *Layout = DL.getStructLayout(StructTy);
  uint64_t StartPos = 0;
  for (unsigned I = 0, E = StructTy->getNumElements(); I < E; ++I) {
    Type *ElTy = StructTy->getElementType(I);
    if (!isDenselyPacked(ElTy, DL))
      return false;
    if (StartPos != Layout->getElementOffsetInBits(I))
      return false;
    StartPos += DL.getTypeAllocSizeInBits(ElTy);
  }

  return true;
}

/// Get pointer operand of memory accessing instruction. If \p I is
/// not a memory accessing instruction, return nullptr. If \p AllowVolatile,
/// is set to false and the instruction is volatile, return nullptr.
static const Value *getPointerOperand(const Instruction *I,
                                      bool AllowVolatile) {
  if (!AllowVolatile && I->isVolatile())
    return nullptr;

  if (auto *LI = dyn_cast<LoadInst>(I)) {
    return LI->getPointerOperand();
  }

  if (auto *SI = dyn_cast<StoreInst>(I)) {
    return SI->getPointerOperand();
  }

  if (auto *CXI = dyn_cast<AtomicCmpXchgInst>(I)) {
    return CXI->getPointerOperand();
  }

  if (auto *RMWI = dyn_cast<AtomicRMWInst>(I)) {
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
static Value *constructPointer(Type *ResTy, Type *PtrElemTy, Value *Ptr,
                               int64_t Offset, IRBuilder<NoFolder> &IRB,
                               const DataLayout &DL) {
  assert(Offset >= 0 && "Negative offset not supported yet!");
  LLVM_DEBUG(dbgs() << "Construct pointer: " << *Ptr << " + " << Offset
                    << "-bytes as " << *ResTy << "\n");

  if (Offset) {
    Type *Ty = PtrElemTy;
    APInt IntOffset(DL.getIndexTypeSizeInBits(Ptr->getType()), Offset);
    SmallVector<APInt> IntIndices = DL.getGEPIndicesForOffset(Ty, IntOffset);

    SmallVector<Value *, 4> ValIndices;
    std::string GEPName = Ptr->getName().str();
    for (const APInt &Index : IntIndices) {
      ValIndices.push_back(IRB.getInt(Index));
      GEPName += "." + std::to_string(Index.getZExtValue());
    }

    // Create a GEP for the indices collected above.
    Ptr = IRB.CreateGEP(PtrElemTy, Ptr, ValIndices, GEPName);

    // If an offset is left we use byte-wise adjustment.
    if (IntOffset != 0) {
      Ptr = IRB.CreateBitCast(Ptr, IRB.getInt8PtrTy());
      Ptr = IRB.CreateGEP(IRB.getInt8Ty(), Ptr, IRB.getInt(IntOffset),
                          GEPName + ".b" + Twine(IntOffset.getZExtValue()));
    }
  }

  // Ensure the result has the requested type.
  Ptr = IRB.CreatePointerBitCastOrAddrSpaceCast(Ptr, ResTy,
                                                Ptr->getName() + ".cast");

  LLVM_DEBUG(dbgs() << "Constructed pointer: " << *Ptr << "\n");
  return Ptr;
}

bool AA::getAssumedUnderlyingObjects(Attributor &A, const Value &Ptr,
                                     SmallSetVector<Value *, 8> &Objects,
                                     const AbstractAttribute &QueryingAA,
                                     const Instruction *CtxI,
                                     bool &UsedAssumedInformation,
                                     AA::ValueScope S,
                                     SmallPtrSetImpl<Value *> *SeenObjects) {
  SmallPtrSet<Value *, 8> LocalSeenObjects;
  if (!SeenObjects)
    SeenObjects = &LocalSeenObjects;

  SmallVector<AA::ValueAndContext> Values;
  if (!A.getAssumedSimplifiedValues(IRPosition::value(Ptr), &QueryingAA, Values,
                                    S, UsedAssumedInformation)) {
    Objects.insert(const_cast<Value *>(&Ptr));
    return true;
  }

  for (auto &VAC : Values) {
    Value *UO = getUnderlyingObject(VAC.getValue());
    if (UO && UO != VAC.getValue() && SeenObjects->insert(UO).second) {
      if (!getAssumedUnderlyingObjects(A, *UO, Objects, QueryingAA,
                                       VAC.getCtxI(), UsedAssumedInformation, S,
                                       SeenObjects))
        return false;
      continue;
    }
    Objects.insert(VAC.getValue());
  }
  return true;
}

static const Value *
stripAndAccumulateOffsets(Attributor &A, const AbstractAttribute &QueryingAA,
                          const Value *Val, const DataLayout &DL, APInt &Offset,
                          bool GetMinOffset, bool AllowNonInbounds,
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
    if (Range.isFullSet())
      return false;

    // We can only use the lower part of the range because the upper part can
    // be higher than what the value can really be.
    if (GetMinOffset)
      ROffset = Range.getSignedMin();
    else
      ROffset = Range.getSignedMax();
    return true;
  };

  return Val->stripAndAccumulateConstantOffsets(DL, Offset, AllowNonInbounds,
                                                /* AllowInvariant */ true,
                                                AttributorAnalysis);
}

static const Value *
getMinimalBaseOfPointer(Attributor &A, const AbstractAttribute &QueryingAA,
                        const Value *Ptr, int64_t &BytesOffset,
                        const DataLayout &DL, bool AllowNonInbounds = false) {
  APInt OffsetAPInt(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
  const Value *Base =
      stripAndAccumulateOffsets(A, QueryingAA, Ptr, DL, OffsetAPInt,
                                /* GetMinOffset */ true, AllowNonInbounds);

  BytesOffset = OffsetAPInt.getSExtValue();
  return Base;
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
    if (!T)
      T = StateType::getBestState(AAS);
    *T &= AAS;
    LLVM_DEBUG(dbgs() << "[Attributor] AA State: " << AAS << " RV State: " << T
                      << "\n");
    return T->isValidState();
  };

  if (!A.checkForAllReturnedValues(CheckReturnValue, QueryingAA))
    S.indicatePessimisticFixpoint();
  else if (T)
    S ^= *T;
}

namespace {
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
    if (!T)
      T = StateType::getBestState(AAS);
    *T &= AAS;
    LLVM_DEBUG(dbgs() << "[Attributor] AA State: " << AAS << " CSA State: " << T
                      << "\n");
    return T->isValidState();
  };

  bool UsedAssumedInformation = false;
  if (!A.checkForAllCallSites(CallSiteCheck, QueryingAA, true,
                              UsedAssumedInformation))
    S.indicatePessimisticFixpoint();
  else if (T)
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

    CallBase &CBContext = cast<CallBase>(this->getAnchorValue());
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
} // namespace

/// ------------------------ PointerInfo ---------------------------------------

namespace llvm {
namespace AA {
namespace PointerInfo {

struct State;

} // namespace PointerInfo
} // namespace AA

/// Helper for AA::PointerInfo::Access DenseMap/Set usage.
template <>
struct DenseMapInfo<AAPointerInfo::Access> : DenseMapInfo<Instruction *> {
  using Access = AAPointerInfo::Access;
  static inline Access getEmptyKey();
  static inline Access getTombstoneKey();
  static unsigned getHashValue(const Access &A);
  static bool isEqual(const Access &LHS, const Access &RHS);
};

/// Helper that allows OffsetAndSize as a key in a DenseMap.
template <> struct DenseMapInfo<AA::OffsetAndSize> {
  static inline AA::OffsetAndSize getEmptyKey() {
    auto EmptyKey = DenseMapInfo<int64_t>::getEmptyKey();
    return AA::OffsetAndSize{EmptyKey, EmptyKey};
  }

  static inline AA::OffsetAndSize getTombstoneKey() {
    auto TombstoneKey = DenseMapInfo<int64_t>::getTombstoneKey();
    return AA::OffsetAndSize{TombstoneKey, TombstoneKey};
  }

  static unsigned getHashValue(const AA::OffsetAndSize &OAS) {
    return detail::combineHashValue(
        DenseMapInfo<int64_t>::getHashValue(OAS.Offset),
        DenseMapInfo<int64_t>::getHashValue(OAS.Size));
  }

  static bool isEqual(const AA::OffsetAndSize &A, const AA::OffsetAndSize B) {
    return A == B;
  }
};

/// Helper for AA::PointerInfo::Access DenseMap/Set usage ignoring everythign
/// but the instruction
struct AccessAsInstructionInfo : DenseMapInfo<Instruction *> {
  using Base = DenseMapInfo<Instruction *>;
  using Access = AAPointerInfo::Access;
  static inline Access getEmptyKey();
  static inline Access getTombstoneKey();
  static unsigned getHashValue(const Access &A);
  static bool isEqual(const Access &LHS, const Access &RHS);
};

} // namespace llvm

/// A type to track pointer/struct usage and accesses for AAPointerInfo.
struct AA::PointerInfo::State : public AbstractState {

  ~State() {
    // We do not delete the Accesses objects but need to destroy them still.
    for (auto &It : AccessBins)
      It.second->~Accesses();
  }

  /// Return the best possible representable state.
  static State getBestState(const State &SIS) { return State(); }

  /// Return the worst possible representable state.
  static State getWorstState(const State &SIS) {
    State R;
    R.indicatePessimisticFixpoint();
    return R;
  }

  State() = default;
  State(State &&SIS) : AccessBins(std::move(SIS.AccessBins)) {
    SIS.AccessBins.clear();
  }

  const State &getAssumed() const { return *this; }

  /// See AbstractState::isValidState().
  bool isValidState() const override { return BS.isValidState(); }

  /// See AbstractState::isAtFixpoint().
  bool isAtFixpoint() const override { return BS.isAtFixpoint(); }

  /// See AbstractState::indicateOptimisticFixpoint().
  ChangeStatus indicateOptimisticFixpoint() override {
    BS.indicateOptimisticFixpoint();
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractState::indicatePessimisticFixpoint().
  ChangeStatus indicatePessimisticFixpoint() override {
    BS.indicatePessimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  State &operator=(const State &R) {
    if (this == &R)
      return *this;
    BS = R.BS;
    AccessBins = R.AccessBins;
    return *this;
  }

  State &operator=(State &&R) {
    if (this == &R)
      return *this;
    std::swap(BS, R.BS);
    std::swap(AccessBins, R.AccessBins);
    return *this;
  }

  bool operator==(const State &R) const {
    if (BS != R.BS)
      return false;
    if (AccessBins.size() != R.AccessBins.size())
      return false;
    auto It = begin(), RIt = R.begin(), E = end();
    while (It != E) {
      if (It->getFirst() != RIt->getFirst())
        return false;
      auto &Accs = It->getSecond();
      auto &RAccs = RIt->getSecond();
      if (Accs->size() != RAccs->size())
        return false;
      for (const auto &ZipIt : llvm::zip(*Accs, *RAccs))
        if (std::get<0>(ZipIt) != std::get<1>(ZipIt))
          return false;
      ++It;
      ++RIt;
    }
    return true;
  }
  bool operator!=(const State &R) const { return !(*this == R); }

  /// We store accesses in a set with the instruction as key.
  struct Accesses {
    SmallVector<AAPointerInfo::Access, 4> Accesses;
    DenseMap<const Instruction *, unsigned> Map;

    unsigned size() const { return Accesses.size(); }

    using vec_iterator = decltype(Accesses)::iterator;
    vec_iterator begin() { return Accesses.begin(); }
    vec_iterator end() { return Accesses.end(); }

    using iterator = decltype(Map)::const_iterator;
    iterator find(AAPointerInfo::Access &Acc) {
      return Map.find(Acc.getRemoteInst());
    }
    iterator find_end() { return Map.end(); }

    AAPointerInfo::Access &get(iterator &It) {
      return Accesses[It->getSecond()];
    }

    void insert(AAPointerInfo::Access &Acc) {
      Map[Acc.getRemoteInst()] = Accesses.size();
      Accesses.push_back(Acc);
    }
  };

  /// We store all accesses in bins denoted by their offset and size.
  using AccessBinsTy = DenseMap<AA::OffsetAndSize, Accesses *>;

  AccessBinsTy::const_iterator begin() const { return AccessBins.begin(); }
  AccessBinsTy::const_iterator end() const { return AccessBins.end(); }

protected:
  /// The bins with all the accesses for the associated pointer.
  AccessBinsTy AccessBins;

  /// Add a new access to the state at offset \p Offset and with size \p Size.
  /// The access is associated with \p I, writes \p Content (if anything), and
  /// is of kind \p Kind.
  /// \Returns CHANGED, if the state changed, UNCHANGED otherwise.
  ChangeStatus addAccess(Attributor &A, int64_t Offset, int64_t Size,
                         Instruction &I, Optional<Value *> Content,
                         AAPointerInfo::AccessKind Kind, Type *Ty,
                         Instruction *RemoteI = nullptr,
                         Accesses *BinPtr = nullptr) {
    AA::OffsetAndSize Key{Offset, Size};
    Accesses *&Bin = BinPtr ? BinPtr : AccessBins[Key];
    if (!Bin)
      Bin = new (A.Allocator) Accesses;
    AAPointerInfo::Access Acc(&I, RemoteI ? RemoteI : &I, Content, Kind, Ty);
    // Check if we have an access for this instruction in this bin, if not,
    // simply add it.
    auto It = Bin->find(Acc);
    if (It == Bin->find_end()) {
      Bin->insert(Acc);
      return ChangeStatus::CHANGED;
    }
    // If the existing access is the same as then new one, nothing changed.
    AAPointerInfo::Access &Current = Bin->get(It);
    AAPointerInfo::Access Before = Current;
    // The new one will be combined with the existing one.
    Current &= Acc;
    return Current == Before ? ChangeStatus::UNCHANGED : ChangeStatus::CHANGED;
  }

  /// See AAPointerInfo::forallInterferingAccesses.
  bool forallInterferingAccesses(
      AA::OffsetAndSize OAS,
      function_ref<bool(const AAPointerInfo::Access &, bool)> CB) const {
    if (!isValidState())
      return false;

    for (const auto &It : AccessBins) {
      AA::OffsetAndSize ItOAS = It.getFirst();
      if (!OAS.mayOverlap(ItOAS))
        continue;
      bool IsExact = OAS == ItOAS && !OAS.offsetOrSizeAreUnknown();
      for (auto &Access : *It.getSecond())
        if (!CB(Access, IsExact))
          return false;
    }
    return true;
  }

  /// See AAPointerInfo::forallInterferingAccesses.
  bool forallInterferingAccesses(
      Instruction &I,
      function_ref<bool(const AAPointerInfo::Access &, bool)> CB,
      AA::OffsetAndSize *OASPtr) const {
    if (!isValidState())
      return false;

    // First find the offset and size of I.
    AA::OffsetAndSize OAS;
    for (const auto &It : AccessBins) {
      for (auto &Access : *It.getSecond()) {
        if (Access.getRemoteInst() == &I) {
          OAS = It.getFirst();
          break;
        }
      }
      if (OAS.Size != AA::OffsetAndSize::Unassigned)
        break;
    }

    if (OASPtr)
      *OASPtr = OAS;

    // No access for I was found, we are done.
    if (OAS.Size == AA::OffsetAndSize::Unassigned)
      return true;

    // Now that we have an offset and size, find all overlapping ones and use
    // the callback on the accesses.
    return forallInterferingAccesses(OAS, CB);
  }

private:
  /// State to track fixpoint and validity.
  BooleanState BS;
};

namespace {
struct AAPointerInfoImpl
    : public StateWrapper<AA::PointerInfo::State, AAPointerInfo> {
  using BaseTy = StateWrapper<AA::PointerInfo::State, AAPointerInfo>;
  AAPointerInfoImpl(const IRPosition &IRP, Attributor &A) : BaseTy(IRP) {}

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return std::string("PointerInfo ") +
           (isValidState() ? (std::string("#") +
                              std::to_string(AccessBins.size()) + " bins")
                           : "<invalid>");
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    return AAPointerInfo::manifest(A);
  }

  bool forallInterferingAccesses(
      AA::OffsetAndSize OAS,
      function_ref<bool(const AAPointerInfo::Access &, bool)> CB)
      const override {
    return State::forallInterferingAccesses(OAS, CB);
  }

  bool forallInterferingAccesses(
      Attributor &A, const AbstractAttribute &QueryingAA, Instruction &I,
      function_ref<bool(const Access &, bool)> UserCB, bool &HasBeenWrittenTo,
      AA::OffsetAndSize *OASPtr = nullptr) const override {
    HasBeenWrittenTo = false;

    SmallPtrSet<const Access *, 8> DominatingWrites;
    SmallVector<std::pair<const Access *, bool>, 8> InterferingAccesses;

    Function &Scope = *I.getFunction();
    const auto &NoSyncAA = A.getAAFor<AANoSync>(
        QueryingAA, IRPosition::function(Scope), DepClassTy::OPTIONAL);
    const auto *ExecDomainAA = A.lookupAAFor<AAExecutionDomain>(
        IRPosition::function(Scope), &QueryingAA, DepClassTy::OPTIONAL);
    const bool NoSync = NoSyncAA.isAssumedNoSync();

    // Helper to determine if we need to consider threading, which we cannot
    // right now. However, if the function is (assumed) nosync or the thread
    // executing all instructions is the main thread only we can ignore
    // threading.
    auto CanIgnoreThreading = [&](const Instruction &I) -> bool {
      if (NoSync)
        return true;
      if (ExecDomainAA && ExecDomainAA->isExecutedByInitialThreadOnly(I))
        return true;
      return false;
    };

    // Helper to determine if the access is executed by the same thread as the
    // load, for now it is sufficient to avoid any potential threading effects
    // as we cannot deal with them anyway.
    auto IsSameThreadAsLoad = [&](const Access &Acc) -> bool {
      return CanIgnoreThreading(*Acc.getLocalInst());
    };

    // TODO: Use inter-procedural reachability and dominance.
    const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
        QueryingAA, IRPosition::function(Scope), DepClassTy::OPTIONAL);

    const bool FindInterferingWrites = I.mayReadFromMemory();
    const bool FindInterferingReads = I.mayWriteToMemory();
    const bool UseDominanceReasoning =
        FindInterferingWrites && NoRecurseAA.isKnownNoRecurse();
    const bool CanUseCFGResoning = CanIgnoreThreading(I);
    InformationCache &InfoCache = A.getInfoCache();
    const DominatorTree *DT =
        InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(Scope);

    enum class GPUAddressSpace : unsigned {
      Generic = 0,
      Global = 1,
      Shared = 3,
      Constant = 4,
      Local = 5,
    };

    // Helper to check if a value has "kernel lifetime", that is it will not
    // outlive a GPU kernel. This is true for shared, constant, and local
    // globals on AMD and NVIDIA GPUs.
    auto HasKernelLifetime = [&](Value *V, Module &M) {
      Triple T(M.getTargetTriple());
      if (!(T.isAMDGPU() || T.isNVPTX()))
        return false;
      switch (GPUAddressSpace(V->getType()->getPointerAddressSpace())) {
      case GPUAddressSpace::Shared:
      case GPUAddressSpace::Constant:
      case GPUAddressSpace::Local:
        return true;
      default:
        return false;
      };
    };

    // The IsLiveInCalleeCB will be used by the AA::isPotentiallyReachable query
    // to determine if we should look at reachability from the callee. For
    // certain pointers we know the lifetime and we do not have to step into the
    // callee to determine reachability as the pointer would be dead in the
    // callee. See the conditional initialization below.
    std::function<bool(const Function &)> IsLiveInCalleeCB;

    if (auto *AI = dyn_cast<AllocaInst>(&getAssociatedValue())) {
      // If the alloca containing function is not recursive the alloca
      // must be dead in the callee.
      const Function *AIFn = AI->getFunction();
      const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
          *this, IRPosition::function(*AIFn), DepClassTy::OPTIONAL);
      if (NoRecurseAA.isAssumedNoRecurse()) {
        IsLiveInCalleeCB = [AIFn](const Function &Fn) { return AIFn != &Fn; };
      }
    } else if (auto *GV = dyn_cast<GlobalValue>(&getAssociatedValue())) {
      // If the global has kernel lifetime we can stop if we reach a kernel
      // as it is "dead" in the (unknown) callees.
      if (HasKernelLifetime(GV, *GV->getParent()))
        IsLiveInCalleeCB = [](const Function &Fn) {
          return !Fn.hasFnAttribute("kernel");
        };
    }

    auto AccessCB = [&](const Access &Acc, bool Exact) {
      if ((!FindInterferingWrites || !Acc.isWrite()) &&
          (!FindInterferingReads || !Acc.isRead()))
        return true;

      bool Dominates = DT && Exact && Acc.isMustAccess() &&
                       (Acc.getLocalInst()->getFunction() == &Scope) &&
                       DT->dominates(Acc.getRemoteInst(), &I);
      if (FindInterferingWrites && Dominates)
        HasBeenWrittenTo = true;

      // For now we only filter accesses based on CFG reasoning which does not
      // work yet if we have threading effects, or the access is complicated.
      if (CanUseCFGResoning && Dominates && UseDominanceReasoning &&
          IsSameThreadAsLoad(Acc))
        DominatingWrites.insert(&Acc);

      InterferingAccesses.push_back({&Acc, Exact});
      return true;
    };
    if (!State::forallInterferingAccesses(I, AccessCB, OASPtr))
      return false;

    if (HasBeenWrittenTo) {
      const Function *ScopePtr = &Scope;
      IsLiveInCalleeCB = [ScopePtr](const Function &Fn) {
        return ScopePtr != &Fn;
      };
    }

    // Helper to determine if we can skip a specific write access. This is in
    // the worst case quadratic as we are looking for another write that will
    // hide the effect of this one.
    auto CanSkipAccess = [&](const Access &Acc, bool Exact) {
      if ((!Acc.isWrite() ||
           !AA::isPotentiallyReachable(A, *Acc.getLocalInst(), I, QueryingAA,
                                       IsLiveInCalleeCB)) &&
          (!Acc.isRead() ||
           !AA::isPotentiallyReachable(A, I, *Acc.getLocalInst(), QueryingAA,
                                       IsLiveInCalleeCB)))
        return true;

      if (!DT || !UseDominanceReasoning)
        return false;
      if (!IsSameThreadAsLoad(Acc))
        return false;
      if (!DominatingWrites.count(&Acc))
        return false;
      for (const Access *DomAcc : DominatingWrites) {
        assert(Acc.getLocalInst()->getFunction() ==
                   DomAcc->getLocalInst()->getFunction() &&
               "Expected dominating writes to be in the same function!");

        if (DomAcc != &Acc &&
            DT->dominates(Acc.getLocalInst(), DomAcc->getLocalInst())) {
          return true;
        }
      }
      return false;
    };

    // Run the user callback on all accesses we cannot skip and return if that
    // succeeded for all or not.
    unsigned NumInterferingAccesses = InterferingAccesses.size();
    for (auto &It : InterferingAccesses) {
      if (NumInterferingAccesses > MaxInterferingAccesses ||
          !CanSkipAccess(*It.first, It.second)) {
        if (!UserCB(*It.first, It.second))
          return false;
      }
    }
    return true;
  }

  ChangeStatus translateAndAddState(Attributor &A, const AAPointerInfo &OtherAA,
                                    int64_t Offset, CallBase &CB,
                                    bool FromCallee = false) {
    using namespace AA::PointerInfo;
    if (!OtherAA.getState().isValidState() || !isValidState())
      return indicatePessimisticFixpoint();

    const auto &OtherAAImpl = static_cast<const AAPointerInfoImpl &>(OtherAA);
    bool IsByval =
        FromCallee && OtherAAImpl.getAssociatedArgument()->hasByValAttr();

    // Combine the accesses bin by bin.
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    for (const auto &It : OtherAAImpl.getState()) {
      AA::OffsetAndSize OAS = AA::OffsetAndSize::getUnknown();
      if (Offset != AA::OffsetAndSize::Unknown &&
          !It.first.offsetOrSizeAreUnknown()) {
        OAS = AA::OffsetAndSize(It.first.Offset + Offset, It.first.Size);
      }
      Accesses *Bin = AccessBins.lookup(OAS);
      for (const AAPointerInfo::Access &RAcc : *It.second) {
        if (IsByval && !RAcc.isRead())
          continue;
        bool UsedAssumedInformation = false;
        AccessKind AK = RAcc.getKind();
        Optional<Value *> Content = RAcc.getContent();
        if (FromCallee) {
          Content = A.translateArgumentToCallSiteContent(
              RAcc.getContent(), CB, *this, UsedAssumedInformation);
          AK =
              AccessKind(AK & (IsByval ? AccessKind::AK_R : AccessKind::AK_RW));
          AK = AccessKind(AK | (RAcc.isMayAccess() ? AK_MAY : AK_MUST));
        }
        Changed =
            Changed | addAccess(A, OAS.Offset, OAS.Size, CB, Content, AK,
                                RAcc.getType(), RAcc.getRemoteInst(), Bin);
      }
    }
    return Changed;
  }

  /// Statistic tracking for all AAPointerInfo implementations.
  /// See AbstractAttribute::trackStatistics().
  void trackPointerInfoStatistics(const IRPosition &IRP) const {}

  /// Dump the state into \p O.
  void dumpState(raw_ostream &O) {
    for (auto &It : AccessBins) {
      O << "[" << It.first.Offset << "-" << It.first.Offset + It.first.Size
        << "] : " << It.getSecond()->size() << "\n";
      for (auto &Acc : *It.getSecond()) {
        O << "     - " << Acc.getKind() << " - " << *Acc.getLocalInst() << "\n";
        if (Acc.getLocalInst() != Acc.getRemoteInst())
          O << "     -->                         " << *Acc.getRemoteInst()
            << "\n";
        if (!Acc.isWrittenValueYetUndetermined()) {
          if (Acc.getWrittenValue())
            O << "       - c: " << *Acc.getWrittenValue() << "\n";
          else
            O << "       - c: <unknown>\n";
        }
      }
    }
  }
};

struct AAPointerInfoFloating : public AAPointerInfoImpl {
  using AccessKind = AAPointerInfo::AccessKind;
  AAPointerInfoFloating(const IRPosition &IRP, Attributor &A)
      : AAPointerInfoImpl(IRP, A) {}

  /// Deal with an access and signal if it was handled successfully.
  bool handleAccess(Attributor &A, Instruction &I, Value &Ptr,
                    Optional<Value *> Content, AccessKind Kind, int64_t Offset,
                    ChangeStatus &Changed, Type *Ty,
                    int64_t Size = AA::OffsetAndSize::Unknown) {
    using namespace AA::PointerInfo;
    // No need to find a size if one is given.
    if (Size == AA::OffsetAndSize::Unknown && Ty) {
      const DataLayout &DL = A.getDataLayout();
      TypeSize AccessSize = DL.getTypeStoreSize(Ty);
      if (!AccessSize.isScalable())
        Size = AccessSize.getFixedSize();
    }
    Changed = Changed | addAccess(A, Offset, Size, I, Content, Kind, Ty);
    return true;
  };

  /// Helper struct, will support ranges eventually.
  struct OffsetInfo {
    int64_t Offset = AA::OffsetAndSize::Unassigned;

    bool operator==(const OffsetInfo &OI) const { return Offset == OI.Offset; }
  };

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    using namespace AA::PointerInfo;
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    Value &AssociatedValue = getAssociatedValue();

    const DataLayout &DL = A.getDataLayout();
    DenseMap<Value *, OffsetInfo> OffsetInfoMap;
    OffsetInfoMap[&AssociatedValue] = OffsetInfo{0};

    auto HandlePassthroughUser = [&](Value *Usr, OffsetInfo PtrOI,
                                     bool &Follow) {
      assert(PtrOI.Offset != AA::OffsetAndSize::Unassigned &&
             "Cannot pass through if the input Ptr was not visited!");
      OffsetInfo &UsrOI = OffsetInfoMap[Usr];
      UsrOI = PtrOI;
      Follow = true;
      return true;
    };

    const auto *TLI = getAnchorScope()
                          ? A.getInfoCache().getTargetLibraryInfoForFunction(
                                *getAnchorScope())
                          : nullptr;
    auto UsePred = [&](const Use &U, bool &Follow) -> bool {
      Value *CurPtr = U.get();
      User *Usr = U.getUser();
      LLVM_DEBUG(dbgs() << "[AAPointerInfo] Analyze " << *CurPtr << " in "
                        << *Usr << "\n");
      assert(OffsetInfoMap.count(CurPtr) &&
             "The current pointer offset should have been seeded!");

      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Usr)) {
        if (CE->isCast())
          return HandlePassthroughUser(Usr, OffsetInfoMap[CurPtr], Follow);
        if (CE->isCompare())
          return true;
        if (!isa<GEPOperator>(CE)) {
          LLVM_DEBUG(dbgs() << "[AAPointerInfo] Unhandled constant user " << *CE
                            << "\n");
          return false;
        }
      }
      if (auto *GEP = dyn_cast<GEPOperator>(Usr)) {
        // Note the order here, the Usr access might change the map, CurPtr is
        // already in it though.
        OffsetInfo &UsrOI = OffsetInfoMap[Usr];
        OffsetInfo &PtrOI = OffsetInfoMap[CurPtr];
        UsrOI = PtrOI;

        // TODO: Use range information.
        APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (PtrOI.Offset == AA::OffsetAndSize::Unknown ||
            !GEP->accumulateConstantOffset(DL, GEPOffset)) {
          LLVM_DEBUG(dbgs() << "[AAPointerInfo] GEP offset not constant "
                            << *GEP << "\n");
          UsrOI.Offset = AA::OffsetAndSize::Unknown;
          Follow = true;
          return true;
        }

        LLVM_DEBUG(dbgs() << "[AAPointerInfo] GEP offset is constant " << *GEP
                          << "\n");
        UsrOI.Offset = PtrOI.Offset + GEPOffset.getZExtValue();
        Follow = true;
        return true;
      }
      if (isa<PtrToIntInst>(Usr))
        return false;
      if (isa<CastInst>(Usr) || isa<SelectInst>(Usr) || isa<ReturnInst>(Usr))
        return HandlePassthroughUser(Usr, OffsetInfoMap[CurPtr], Follow);

      // For PHIs we need to take care of the recurrence explicitly as the value
      // might change while we iterate through a loop. For now, we give up if
      // the PHI is not invariant.
      if (isa<PHINode>(Usr)) {
        // Note the order here, the Usr access might change the map, CurPtr is
        // already in it though.
        bool IsFirstPHIUser = !OffsetInfoMap.count(Usr);
        OffsetInfo &UsrOI = OffsetInfoMap[Usr];
        OffsetInfo &PtrOI = OffsetInfoMap[CurPtr];

        // Check if the PHI operand has already an unknown offset as we can't
        // improve on that anymore.
        if (PtrOI.Offset == AA::OffsetAndSize::Unknown) {
          LLVM_DEBUG(dbgs() << "[AAPointerInfo] PHI operand offset unknown "
                            << *CurPtr << " in " << *Usr << "\n");
          Follow = UsrOI.Offset != AA::OffsetAndSize::Unknown;
          UsrOI = PtrOI;
          return true;
        }

        // Check if the PHI is invariant (so far).
        if (UsrOI == PtrOI) {
          assert(PtrOI.Offset != AA::OffsetAndSize::Unassigned &&
                 "Cannot assign if the current Ptr was not visited!");
          LLVM_DEBUG(dbgs() << "[AAPointerInfo] PHI is invariant (so far)");
          return true;
        }

        // Check if the PHI operand is not dependent on the PHI itself.
        APInt Offset(
            DL.getIndexSizeInBits(CurPtr->getType()->getPointerAddressSpace()),
            0);
        Value *CurPtrBase = CurPtr->stripAndAccumulateConstantOffsets(
            DL, Offset, /* AllowNonInbounds */ true);
        auto It = OffsetInfoMap.find(CurPtrBase);
        if (It != OffsetInfoMap.end()) {
          Offset += It->getSecond().Offset;
          if (IsFirstPHIUser || Offset == UsrOI.Offset)
            return HandlePassthroughUser(Usr, PtrOI, Follow);
          LLVM_DEBUG(dbgs()
                     << "[AAPointerInfo] PHI operand pointer offset mismatch "
                     << *CurPtr << " in " << *Usr << "\n");
        } else {
          LLVM_DEBUG(dbgs() << "[AAPointerInfo] PHI operand is too complex "
                            << *CurPtr << " in " << *Usr << "\n");
        }

        // TODO: Approximate in case we know the direction of the recurrence.
        UsrOI = PtrOI;
        UsrOI.Offset = AA::OffsetAndSize::Unknown;
        Follow = true;
        return true;
      }

      if (auto *LoadI = dyn_cast<LoadInst>(Usr)) {
        // If the access is to a pointer that may or may not be the associated
        // value, e.g. due to a PHI, we cannot assume it will be read.
        AccessKind AK = AccessKind::AK_R;
        if (getUnderlyingObject(CurPtr) == &AssociatedValue)
          AK = AccessKind(AK | AccessKind::AK_MUST);
        else
          AK = AccessKind(AK | AccessKind::AK_MAY);
        return handleAccess(A, *LoadI, *CurPtr, /* Content */ nullptr, AK,
                            OffsetInfoMap[CurPtr].Offset, Changed,
                            LoadI->getType());
      }

      auto HandleStoreLike = [&](Instruction &I, Value *ValueOp, Type &ValueTy,
                                 ArrayRef<Value *> OtherOps, AccessKind AK) {
        for (auto *OtherOp : OtherOps) {
          if (OtherOp == CurPtr) {
            LLVM_DEBUG(
                dbgs()
                << "[AAPointerInfo] Escaping use in store like instruction "
                << I << "\n");
            return false;
          }
        }

        // If the access is to a pointer that may or may not be the associated
        // value, e.g. due to a PHI, we cannot assume it will be written.
        if (getUnderlyingObject(CurPtr) == &AssociatedValue)
          AK = AccessKind(AK | AccessKind::AK_MUST);
        else
          AK = AccessKind(AK | AccessKind::AK_MAY);
        bool UsedAssumedInformation = false;
        Optional<Value *> Content = nullptr;
        if (ValueOp)
          Content = A.getAssumedSimplified(
              *ValueOp, *this, UsedAssumedInformation, AA::Interprocedural);
        return handleAccess(A, I, *CurPtr, Content, AK,
                            OffsetInfoMap[CurPtr].Offset, Changed, &ValueTy);
      };

      if (auto *StoreI = dyn_cast<StoreInst>(Usr))
        return HandleStoreLike(*StoreI, StoreI->getValueOperand(),
                               *StoreI->getValueOperand()->getType(),
                               {StoreI->getValueOperand()}, AccessKind::AK_W);
      if (auto *RMWI = dyn_cast<AtomicRMWInst>(Usr))
        return HandleStoreLike(*RMWI, nullptr,
                               *RMWI->getValOperand()->getType(),
                               {RMWI->getValOperand()}, AccessKind::AK_RW);
      if (auto *CXI = dyn_cast<AtomicCmpXchgInst>(Usr))
        return HandleStoreLike(
            *CXI, nullptr, *CXI->getNewValOperand()->getType(),
            {CXI->getCompareOperand(), CXI->getNewValOperand()},
            AccessKind::AK_RW);

      if (auto *CB = dyn_cast<CallBase>(Usr)) {
        if (CB->isLifetimeStartOrEnd())
          return true;
        if (getFreedOperand(CB, TLI) == U)
          return true;
        if (CB->isArgOperand(&U)) {
          unsigned ArgNo = CB->getArgOperandNo(&U);
          const auto &CSArgPI = A.getAAFor<AAPointerInfo>(
              *this, IRPosition::callsite_argument(*CB, ArgNo),
              DepClassTy::REQUIRED);
          Changed = translateAndAddState(A, CSArgPI,
                                         OffsetInfoMap[CurPtr].Offset, *CB) |
                    Changed;
          return isValidState();
        }
        LLVM_DEBUG(dbgs() << "[AAPointerInfo] Call user not handled " << *CB
                          << "\n");
        // TODO: Allow some call uses
        return false;
      }

      LLVM_DEBUG(dbgs() << "[AAPointerInfo] User not handled " << *Usr << "\n");
      return false;
    };
    auto EquivalentUseCB = [&](const Use &OldU, const Use &NewU) {
      if (OffsetInfoMap.count(NewU)) {
        LLVM_DEBUG({
          if (!(OffsetInfoMap[NewU] == OffsetInfoMap[OldU])) {
            dbgs() << "[AAPointerInfo] Equivalent use callback failed: "
                   << OffsetInfoMap[NewU].Offset << " vs "
                   << OffsetInfoMap[OldU].Offset << "\n";
          }
        });
        return OffsetInfoMap[NewU] == OffsetInfoMap[OldU];
      }
      OffsetInfoMap[NewU] = OffsetInfoMap[OldU];
      return true;
    };
    if (!A.checkForAllUses(UsePred, *this, AssociatedValue,
                           /* CheckBBLivenessOnly */ true, DepClassTy::OPTIONAL,
                           /* IgnoreDroppableUses */ true, EquivalentUseCB)) {
      LLVM_DEBUG(
          dbgs() << "[AAPointerInfo] Check for all uses failed, abort!\n");
      return indicatePessimisticFixpoint();
    }

    LLVM_DEBUG({
      dbgs() << "Accesses by bin after update:\n";
      dumpState(dbgs());
    });

    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    AAPointerInfoImpl::trackPointerInfoStatistics(getIRPosition());
  }
};

struct AAPointerInfoReturned final : AAPointerInfoImpl {
  AAPointerInfoReturned(const IRPosition &IRP, Attributor &A)
      : AAPointerInfoImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    AAPointerInfoImpl::trackPointerInfoStatistics(getIRPosition());
  }
};

struct AAPointerInfoArgument final : AAPointerInfoFloating {
  AAPointerInfoArgument(const IRPosition &IRP, Attributor &A)
      : AAPointerInfoFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAPointerInfoFloating::initialize(A);
    if (getAnchorScope()->isDeclaration())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    AAPointerInfoImpl::trackPointerInfoStatistics(getIRPosition());
  }
};

struct AAPointerInfoCallSiteArgument final : AAPointerInfoFloating {
  AAPointerInfoCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAPointerInfoFloating(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    using namespace AA::PointerInfo;
    // We handle memory intrinsics explicitly, at least the first (=
    // destination) and second (=source) arguments as we know how they are
    // accessed.
    if (auto *MI = dyn_cast_or_null<MemIntrinsic>(getCtxI())) {
      ConstantInt *Length = dyn_cast<ConstantInt>(MI->getLength());
      int64_t LengthVal = AA::OffsetAndSize::Unknown;
      if (Length)
        LengthVal = Length->getSExtValue();
      Value &Ptr = getAssociatedValue();
      unsigned ArgNo = getIRPosition().getCallSiteArgNo();
      ChangeStatus Changed = ChangeStatus::UNCHANGED;
      if (ArgNo == 0) {
        handleAccess(A, *MI, Ptr, nullptr, AccessKind::AK_MUST_WRITE, 0,
                     Changed, nullptr, LengthVal);
      } else if (ArgNo == 1) {
        handleAccess(A, *MI, Ptr, nullptr, AccessKind::AK_MUST_READ, 0, Changed,
                     nullptr, LengthVal);
      } else {
        LLVM_DEBUG(dbgs() << "[AAPointerInfo] Unhandled memory intrinsic "
                          << *MI << "\n");
        return indicatePessimisticFixpoint();
      }

      LLVM_DEBUG({
        dbgs() << "Accesses by bin after update:\n";
        dumpState(dbgs());
      });

      return Changed;
    }

    // TODO: Once we have call site specific value information we can provide
    //       call site specific liveness information and then it makes
    //       sense to specialize attributes for call sites arguments instead of
    //       redirecting requests to the callee argument.
    Argument *Arg = getAssociatedArgument();
    if (Arg) {
      const IRPosition &ArgPos = IRPosition::argument(*Arg);
      auto &ArgAA =
          A.getAAFor<AAPointerInfo>(*this, ArgPos, DepClassTy::REQUIRED);
      if (ArgAA.getState().isValidState())
        return translateAndAddState(A, ArgAA, 0, *cast<CallBase>(getCtxI()),
                                    /* FromCallee */ true);
      if (!Arg->getParent()->isDeclaration())
        return indicatePessimisticFixpoint();
    }

    const auto &NoCaptureAA =
        A.getAAFor<AANoCapture>(*this, getIRPosition(), DepClassTy::OPTIONAL);

    if (!NoCaptureAA.isAssumedNoCapture())
      return indicatePessimisticFixpoint();

    bool IsKnown = false;
    if (AA::isAssumedReadNone(A, getIRPosition(), *this, IsKnown))
      return ChangeStatus::UNCHANGED;
    bool ReadOnly = AA::isAssumedReadOnly(A, getIRPosition(), *this, IsKnown);

    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    handleAccess(A, *getCtxI(), getAssociatedValue(), nullptr,
                 ReadOnly ? AccessKind::AK_MAY_READ
                          : AccessKind::AK_MAY_READ_WRITE,
                 AA::OffsetAndSize::Unknown, Changed, nullptr,
                 AA::OffsetAndSize::Unknown);
    return Changed;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    AAPointerInfoImpl::trackPointerInfoStatistics(getIRPosition());
  }
};

struct AAPointerInfoCallSiteReturned final : AAPointerInfoFloating {
  AAPointerInfoCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAPointerInfoFloating(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    AAPointerInfoImpl::trackPointerInfoStatistics(getIRPosition());
  }
};
} // namespace

/// -----------------------NoUnwind Function Attribute--------------------------

namespace {
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

    bool UsedAssumedInformation = false;
    if (!A.checkForAllInstructions(CheckForNoUnwind, *this, Opcodes,
                                   UsedAssumedInformation))
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
} // namespace

/// --------------------- Function Return Values -------------------------------

namespace {
/// "Attribute" that collects all potential returned values and the return
/// instructions that they arise from.
///
/// If there is a unique returned value R, the manifest method will:
///   - mark R with the "returned" attribute, if R is an argument.
class AAReturnedValuesImpl : public AAReturnedValues, public AbstractState {

  /// Mapping of values potentially returned by the associated function to the
  /// return instructions that might return them.
  MapVector<Value *, SmallSetVector<ReturnInst *, 4>> ReturnedValues;

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

  if (!UniqueRV || !UniqueRV.value())
    return Changed;

  // Bookkeeping.
  STATS_DECLTRACK(UniqueReturnValue, FunctionReturn,
                  "Number of function with unique return");
  // If the assumed unique return value is an argument, annotate it.
  if (auto *UniqueRVArg = dyn_cast<Argument>(UniqueRV.value())) {
    if (UniqueRVArg->getType()->canLosslesslyBitCastTo(
            getAssociatedFunction()->getReturnType())) {
      getIRPosition() = IRPosition::argument(*UniqueRVArg);
      Changed = IRAttribute::manifest(A);
    }
  }
  return Changed;
}

const std::string AAReturnedValuesImpl::getAsStr() const {
  return (isAtFixpoint() ? "returns(#" : "may-return(#") +
         (isValidState() ? std::to_string(getNumReturnValues()) : "?") + ")";
}

Optional<Value *>
AAReturnedValuesImpl::getAssumedUniqueReturnValue(Attributor &A) const {
  // If checkForAllReturnedValues provides a unique value, ignoring potential
  // undef values that can also be present, it is assumed to be the actual
  // return value and forwarded to the caller of this method. If there are
  // multiple, a nullptr is returned indicating there cannot be a unique
  // returned value.
  Optional<Value *> UniqueRV;
  Type *Ty = getAssociatedFunction()->getReturnType();

  auto Pred = [&](Value &RV) -> bool {
    UniqueRV = AA::combineOptionalValuesInAAValueLatice(UniqueRV, &RV, Ty);
    return UniqueRV != Optional<Value *>(nullptr);
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
  for (const auto &It : ReturnedValues) {
    Value *RV = It.first;
    if (!Pred(*RV, It.second))
      return false;
  }

  return true;
}

ChangeStatus AAReturnedValuesImpl::updateImpl(Attributor &A) {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;

  SmallVector<AA::ValueAndContext> Values;
  bool UsedAssumedInformation = false;
  auto ReturnInstCB = [&](Instruction &I) {
    ReturnInst &Ret = cast<ReturnInst>(I);
    Values.clear();
    if (!A.getAssumedSimplifiedValues(IRPosition::value(*Ret.getReturnValue()),
                                      *this, Values, AA::Intraprocedural,
                                      UsedAssumedInformation))
      Values.push_back({*Ret.getReturnValue(), Ret});

    for (auto &VAC : Values) {
      assert(AA::isValidInScope(*VAC.getValue(), Ret.getFunction()) &&
             "Assumed returned value should be valid in function scope!");
      if (ReturnedValues[VAC.getValue()].insert(&Ret))
        Changed = ChangeStatus::CHANGED;
    }
    return true;
  };

  // Discover returned values from all live returned instructions in the
  // associated function.
  if (!A.checkForAllInstructions(ReturnInstCB, *this, {Instruction::Ret},
                                 UsedAssumedInformation))
    return indicatePessimisticFixpoint();
  return Changed;
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
} // namespace

/// ------------------------ NoSync Function Attribute -------------------------

bool AANoSync::isNonRelaxedAtomic(const Instruction *I) {
  if (!I->isAtomic())
    return false;

  if (auto *FI = dyn_cast<FenceInst>(I))
    // All legal orderings for fence are stronger than monotonic.
    return FI->getSyncScopeID() != SyncScope::SingleThread;
  if (auto *AI = dyn_cast<AtomicCmpXchgInst>(I)) {
    // Unordered is not a legal ordering for cmpxchg.
    return (AI->getSuccessOrdering() != AtomicOrdering::Monotonic ||
            AI->getFailureOrdering() != AtomicOrdering::Monotonic);
  }

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
  default:
    llvm_unreachable(
        "New atomic operations need to be known in the attributor.");
  }

  return (Ordering != AtomicOrdering::Unordered &&
          Ordering != AtomicOrdering::Monotonic);
}

/// Return true if this intrinsic is nosync.  This is only used for intrinsics
/// which would be nosync except that they have a volatile flag.  All other
/// intrinsics are simply annotated with the nosync attribute in Intrinsics.td.
bool AANoSync::isNoSyncIntrinsic(const Instruction *I) {
  if (auto *MI = dyn_cast<MemIntrinsic>(I))
    return !MI->isVolatile();
  return false;
}

namespace {
struct AANoSyncImpl : AANoSync {
  AANoSyncImpl(const IRPosition &IRP, Attributor &A) : AANoSync(IRP, A) {}

  const std::string getAsStr() const override {
    return getAssumed() ? "nosync" : "may-sync";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AANoSyncImpl::updateImpl(Attributor &A) {

  auto CheckRWInstForNoSync = [&](Instruction &I) {
    return AA::isNoSyncInst(A, I, *this);
  };

  auto CheckForNoSync = [&](Instruction &I) {
    // At this point we handled all read/write effects and they are all
    // nosync, so they can be skipped.
    if (I.mayReadOrWriteMemory())
      return true;

    // non-convergent and readnone imply nosync.
    return !cast<CallBase>(I).isConvergent();
  };

  bool UsedAssumedInformation = false;
  if (!A.checkForAllReadWriteInstructions(CheckRWInstForNoSync, *this,
                                          UsedAssumedInformation) ||
      !A.checkForAllCallLikeInstructions(CheckForNoSync, *this,
                                         UsedAssumedInformation))
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
} // namespace

/// ------------------------ No-Free Attributes ----------------------------

namespace {
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

    bool UsedAssumedInformation = false;
    if (!A.checkForAllCallLikeInstructions(CheckForNoFree, *this,
                                           UsedAssumedInformation))
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
      if (isa<StoreInst>(UserI) || isa<LoadInst>(UserI) ||
          isa<ReturnInst>(UserI))
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
} // namespace

/// ------------------------ NonNull Argument Attribute ------------------------
namespace {
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

  Optional<MemoryLocation> Loc = MemoryLocation::getOrNone(I);
  if (!Loc || Loc->Ptr != UseV || !Loc->Size.isPrecise() || I->isVolatile())
    return 0;

  int64_t Offset;
  const Value *Base =
      getMinimalBaseOfPointer(A, QueryingAA, Loc->Ptr, Offset, DL);
  if (Base && Base == &AssociatedValue) {
    int64_t DerefBytes = Loc->Size.getValue() + Offset;
    IsNonNull |= !NullPointerIsDefined;
    return std::max(int64_t(0), DerefBytes);
  }

  /// Corner case when an offset is 0.
  Base = GetPointerBaseWithConstantOffset(Loc->Ptr, Offset, DL,
                                          /*AllowNonInbounds*/ true);
  if (Base && Base == &AssociatedValue && Offset == 0) {
    int64_t DerefBytes = Loc->Size.getValue();
    IsNonNull |= !NullPointerIsDefined;
    return std::max(int64_t(0), DerefBytes);
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
    Value &V = *getAssociatedValue().stripPointerCasts();
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

    bool CanBeNull, CanBeFreed;
    if (V.getPointerDereferenceableBytes(A.getDataLayout(), CanBeNull,
                                         CanBeFreed)) {
      if (!CanBeNull) {
        indicateOptimisticFixpoint();
        return;
      }
    }

    if (isa<GlobalValue>(V)) {
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

    bool Stripped;
    bool UsedAssumedInformation = false;
    SmallVector<AA::ValueAndContext> Values;
    if (!A.getAssumedSimplifiedValues(getIRPosition(), *this, Values,
                                      AA::AnyScope, UsedAssumedInformation)) {
      Values.push_back({getAssociatedValue(), getCtxI()});
      Stripped = false;
    } else {
      Stripped = Values.size() != 1 ||
                 Values.front().getValue() != &getAssociatedValue();
    }

    DominatorTree *DT = nullptr;
    AssumptionCache *AC = nullptr;
    InformationCache &InfoCache = A.getInfoCache();
    if (const Function *Fn = getAnchorScope()) {
      DT = InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(*Fn);
      AC = InfoCache.getAnalysisResultForFunction<AssumptionAnalysis>(*Fn);
    }

    AANonNull::StateType T;
    auto VisitValueCB = [&](Value &V, const Instruction *CtxI) -> bool {
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

    for (const auto &VAC : Values)
      if (!VisitValueCB(*VAC.getValue(), VAC.getCtxI()))
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
} // namespace

/// ------------------------ No-Recurse Attributes ----------------------------

namespace {
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

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {

    // If all live call sites are known to be no-recurse, we are as well.
    auto CallSitePred = [&](AbstractCallSite ACS) {
      const auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
          *this, IRPosition::function(*ACS.getInstruction()->getFunction()),
          DepClassTy::NONE);
      return NoRecurseAA.isKnownNoRecurse();
    };
    bool UsedAssumedInformation = false;
    if (A.checkForAllCallSites(CallSitePred, *this, true,
                               UsedAssumedInformation)) {
      // If we know all call sites and all are known no-recurse, we are done.
      // If all known call sites, which might not be all that exist, are known
      // to be no-recurse, we are not done but we can continue to assume
      // no-recurse. If one of the call sites we have not visited will become
      // live, another update is triggered.
      if (!UsedAssumedInformation)
        indicateOptimisticFixpoint();
      return ChangeStatus::UNCHANGED;
    }

    const AAFunctionReachability &EdgeReachability =
        A.getAAFor<AAFunctionReachability>(*this, getIRPosition(),
                                           DepClassTy::REQUIRED);
    if (EdgeReachability.canReach(A, *getAnchorScope()))
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
} // namespace

/// -------------------- Undefined-Behavior Attributes ------------------------

namespace {
struct AAUndefinedBehaviorImpl : public AAUndefinedBehavior {
  AAUndefinedBehaviorImpl(const IRPosition &IRP, Attributor &A)
      : AAUndefinedBehavior(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  // through a pointer (i.e. also branches etc.)
  ChangeStatus updateImpl(Attributor &A) override {
    const size_t UBPrevSize = KnownUBInsts.size();
    const size_t NoUBPrevSize = AssumedNoUBInsts.size();

    auto InspectMemAccessInstForUB = [&](Instruction &I) {
      // Lang ref now states volatile store is not UB, let's skip them.
      if (I.isVolatile() && I.mayWriteToMemory())
        return true;

      // Skip instructions that are already saved.
      if (AssumedNoUBInsts.count(&I) || KnownUBInsts.count(&I))
        return true;

      // If we reach here, we know we have an instruction
      // that accesses memory through a pointer operand,
      // for which getPointerOperand() should give it to us.
      Value *PtrOp =
          const_cast<Value *>(getPointerOperand(&I, /* AllowVolatile */ true));
      assert(PtrOp &&
             "Expected pointer operand of memory accessing instruction");

      // Either we stopped and the appropriate action was taken,
      // or we got back a simplified value to continue.
      Optional<Value *> SimplifiedPtrOp = stopOnUndefOrAssumed(A, PtrOp, &I);
      if (!SimplifiedPtrOp || !SimplifiedPtrOp.value())
        return true;
      const Value *PtrOpVal = SimplifiedPtrOp.value();

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
      auto *BrInst = cast<BranchInst>(&I);

      // Unconditional branches are never considered UB.
      if (BrInst->isUnconditional())
        return true;

      // Either we stopped and the appropriate action was taken,
      // or we got back a simplified value to continue.
      Optional<Value *> SimplifiedCond =
          stopOnUndefOrAssumed(A, BrInst->getCondition(), BrInst);
      if (!SimplifiedCond || !*SimplifiedCond)
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
      for (unsigned idx = 0; idx < CB.arg_size(); idx++) {
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
        bool UsedAssumedInformation = false;
        Optional<Value *> SimplifiedVal =
            A.getAssumedSimplified(IRPosition::value(*ArgVal), *this,
                                   UsedAssumedInformation, AA::Interprocedural);
        if (UsedAssumedInformation)
          continue;
        if (SimplifiedVal && !SimplifiedVal.value())
          return true;
        if (!SimplifiedVal || isa<UndefValue>(*SimplifiedVal.value())) {
          KnownUBInsts.insert(&I);
          continue;
        }
        if (!ArgVal->getType()->isPointerTy() ||
            !isa<ConstantPointerNull>(*SimplifiedVal.value()))
          continue;
        auto &NonNullAA =
            A.getAAFor<AANonNull>(*this, CalleeArgumentIRP, DepClassTy::NONE);
        if (NonNullAA.isKnownNonNull())
          KnownUBInsts.insert(&I);
      }
      return true;
    };

    auto InspectReturnInstForUB = [&](Instruction &I) {
      auto &RI = cast<ReturnInst>(I);
      // Either we stopped and the appropriate action was taken,
      // or we got back a simplified return value to continue.
      Optional<Value *> SimplifiedRetValue =
          stopOnUndefOrAssumed(A, RI.getReturnValue(), &I);
      if (!SimplifiedRetValue || !*SimplifiedRetValue)
        return true;

      // Check if a return instruction always cause UB or not
      // Note: It is guaranteed that the returned position of the anchor
      //       scope has noundef attribute when this is called.
      //       We also ensure the return position is not "assumed dead"
      //       because the returned value was then potentially simplified to
      //       `undef` in AAReturnedValues without removing the `noundef`
      //       attribute yet.

      // When the returned position has noundef attriubte, UB occurs in the
      // following cases.
      //   (1) Returned value is known to be undef.
      //   (2) The value is known to be a null pointer and the returned
      //       position has nonnull attribute (because the returned value is
      //       poison).
      if (isa<ConstantPointerNull>(*SimplifiedRetValue)) {
        auto &NonNullAA = A.getAAFor<AANonNull>(
            *this, IRPosition::returned(*getAnchorScope()), DepClassTy::NONE);
        if (NonNullAA.isKnownNonNull())
          KnownUBInsts.insert(&I);
      }

      return true;
    };

    bool UsedAssumedInformation = false;
    A.checkForAllInstructions(InspectMemAccessInstForUB, *this,
                              {Instruction::Load, Instruction::Store,
                               Instruction::AtomicCmpXchg,
                               Instruction::AtomicRMW},
                              UsedAssumedInformation,
                              /* CheckBBLivenessOnly */ true);
    A.checkForAllInstructions(InspectBrInstForUB, *this, {Instruction::Br},
                              UsedAssumedInformation,
                              /* CheckBBLivenessOnly */ true);
    A.checkForAllCallLikeInstructions(InspectCallSiteForUB, *this,
                                      UsedAssumedInformation);

    // If the returned position of the anchor scope has noundef attriubte, check
    // all returned instructions.
    if (!getAnchorScope()->getReturnType()->isVoidTy()) {
      const IRPosition &ReturnIRP = IRPosition::returned(*getAnchorScope());
      if (!A.isAssumedDead(ReturnIRP, this, nullptr, UsedAssumedInformation)) {
        auto &RetPosNoUndefAA =
            A.getAAFor<AANoUndef>(*this, ReturnIRP, DepClassTy::NONE);
        if (RetPosNoUndefAA.isKnownNoUndef())
          A.checkForAllInstructions(InspectReturnInstForUB, *this,
                                    {Instruction::Ret}, UsedAssumedInformation,
                                    /* CheckBBLivenessOnly */ true);
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
      auto *BrInst = cast<BranchInst>(I);
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
  Optional<Value *> stopOnUndefOrAssumed(Attributor &A, Value *V,
                                         Instruction *I) {
    bool UsedAssumedInformation = false;
    Optional<Value *> SimplifiedV =
        A.getAssumedSimplified(IRPosition::value(*V), *this,
                               UsedAssumedInformation, AA::Interprocedural);
    if (!UsedAssumedInformation) {
      // Don't depend on assumed values.
      if (!SimplifiedV) {
        // If it is known (which we tested above) but it doesn't have a value,
        // then we can assume `undef` and hence the instruction is UB.
        KnownUBInsts.insert(I);
        return llvm::None;
      }
      if (!*SimplifiedV)
        return nullptr;
      V = *SimplifiedV;
    }
    if (isa<UndefValue>(V)) {
      KnownUBInsts.insert(I);
      return llvm::None;
    }
    return V;
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
} // namespace

/// ------------------------ Will-Return Attributes ----------------------------

namespace {
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

    if (isImpliedByMustprogressAndReadonly(A, /* KnownOnly */ true)) {
      indicateOptimisticFixpoint();
      return;
    }
  }

  /// Check for `mustprogress` and `readonly` as they imply `willreturn`.
  bool isImpliedByMustprogressAndReadonly(Attributor &A, bool KnownOnly) {
    // Check for `mustprogress` in the scope and the associated function which
    // might be different if this is a call site.
    if ((!getAnchorScope() || !getAnchorScope()->mustProgress()) &&
        (!getAssociatedFunction() || !getAssociatedFunction()->mustProgress()))
      return false;

    bool IsKnown;
    if (AA::isAssumedReadOnly(A, getIRPosition(), *this, IsKnown))
      return IsKnown || !KnownOnly;
    return false;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    if (isImpliedByMustprogressAndReadonly(A, /* KnownOnly */ false))
      return ChangeStatus::UNCHANGED;

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

    bool UsedAssumedInformation = false;
    if (!A.checkForAllCallLikeInstructions(CheckForWillReturn, *this,
                                           UsedAssumedInformation))
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

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAWillReturnImpl::initialize(A);

    Function *F = getAnchorScope();
    if (!F || F->isDeclaration() || mayContainUnboundedCycle(*F, A))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(willreturn) }
};

/// WillReturn attribute deduction for a call sites.
struct AAWillReturnCallSite final : AAWillReturnImpl {
  AAWillReturnCallSite(const IRPosition &IRP, Attributor &A)
      : AAWillReturnImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAWillReturnImpl::initialize(A);
    Function *F = getAssociatedFunction();
    if (!F || !A.isFunctionIPOAmendable(*F))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    if (isImpliedByMustprogressAndReadonly(A, /* KnownOnly */ false))
      return ChangeStatus::UNCHANGED;

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
} // namespace

/// -------------------AAReachability Attribute--------------------------

namespace {
struct AAReachabilityImpl : AAReachability {
  AAReachabilityImpl(const IRPosition &IRP, Attributor &A)
      : AAReachability(IRP, A) {}

  const std::string getAsStr() const override {
    // TODO: Return the number of reachable queries.
    return "reachable";
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return ChangeStatus::UNCHANGED;
  }
};

struct AAReachabilityFunction final : public AAReachabilityImpl {
  AAReachabilityFunction(const IRPosition &IRP, Attributor &A)
      : AAReachabilityImpl(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FN_ATTR(reachable); }
};
} // namespace

/// ------------------------ NoAlias Argument Attribute ------------------------

namespace {
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
    bool IsKnown;
    if (AA::isAssumedReadOnly(A, getIRPosition(), *this, IsKnown))
      return Base::updateImpl(A);

    // If the argument is never passed through callbacks, no-alias cannot break
    // synchronization.
    bool UsedAssumedInformation = false;
    if (A.checkForAllCallSites(
            [](AbstractCallSite ACS) { return !ACS.isCallbackCall(); }, *this,
            true, UsedAssumedInformation))
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

    auto IsDereferenceableOrNull = [&](Value *O, const DataLayout &DL) {
      const auto &DerefAA = A.getAAFor<AADereferenceable>(
          *this, IRPosition::value(*O), DepClassTy::OPTIONAL);
      return DerefAA.getAssumedDereferenceableBytes();
    };

    A.recordDependence(NoAliasAA, *this, DepClassTy::OPTIONAL);

    const IRPosition &VIRP = IRPosition::value(getAssociatedValue());
    const Function *ScopeFn = VIRP.getAnchorScope();
    auto &NoCaptureAA = A.getAAFor<AANoCapture>(*this, VIRP, DepClassTy::NONE);
    // Check whether the value is captured in the scope using AANoCapture.
    // Look at CFG and check only uses possibly executed before this
    // callsite.
    auto UsePred = [&](const Use &U, bool &Follow) -> bool {
      Instruction *UserI = cast<Instruction>(U.getUser());

      // If UserI is the curr instruction and there is a single potential use of
      // the value in UserI we allow the use.
      // TODO: We should inspect the operands and allow those that cannot alias
      //       with the value.
      if (UserI == getCtxI() && UserI->getNumOperands() == 1)
        return true;

      if (ScopeFn) {
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

        if (!AA::isPotentiallyReachable(
                A, *UserI, *getCtxI(), *this,
                [ScopeFn](const Function &Fn) { return &Fn != ScopeFn; }))
          return true;
      }

      // TODO: We should track the capturing uses in AANoCapture but the problem
      //       is CGSCC runs. For those we would need to "allow" AANoCapture for
      //       a value in the module slice.
      switch (DetermineUseCaptureKind(U, IsDereferenceableOrNull)) {
      case UseCaptureKind::NO_CAPTURE:
        return true;
      case UseCaptureKind::MAY_CAPTURE:
        LLVM_DEBUG(dbgs() << "[AANoAliasCSArg] Unknown user: " << *UserI
                          << "\n");
        return false;
      case UseCaptureKind::PASSTHROUGH:
        Follow = true;
        return true;
      }
      llvm_unreachable("unknown UseCaptureKind");
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
    for (unsigned OtherArgNo = 0; OtherArgNo < CB.arg_size(); OtherArgNo++)
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
  ChangeStatus updateImpl(Attributor &A) override {

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
} // namespace

/// -------------------AAIsDead Function Attribute-----------------------

namespace {
struct AAIsDeadValueImpl : public AAIsDead {
  AAIsDeadValueImpl(const IRPosition &IRP, Attributor &A) : AAIsDead(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (auto *Scope = getAnchorScope())
      if (!A.isRunOn(*Scope))
        indicatePessimisticFixpoint();
  }

  /// See AAIsDead::isAssumedDead().
  bool isAssumedDead() const override { return isAssumed(IS_DEAD); }

  /// See AAIsDead::isKnownDead().
  bool isKnownDead() const override { return isKnown(IS_DEAD); }

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
    return isAssumedDead(I) && isKnownDead();
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    return isAssumedDead() ? "assumed-dead" : "assumed-live";
  }

  /// Check if all uses are assumed dead.
  bool areAllUsesAssumedDead(Attributor &A, Value &V) {
    // Callers might not check the type, void has no uses.
    if (V.getType()->isVoidTy() || V.use_empty())
      return true;

    // If we replace a value with a constant there are no uses left afterwards.
    if (!isa<Constant>(V)) {
      if (auto *I = dyn_cast<Instruction>(&V))
        if (!A.isRunOn(*I->getFunction()))
          return false;
      bool UsedAssumedInformation = false;
      Optional<Constant *> C =
          A.getAssumedConstant(V, *this, UsedAssumedInformation);
      if (!C || *C)
        return true;
    }

    auto UsePred = [&](const Use &U, bool &Follow) { return false; };
    // Explicitly set the dependence class to required because we want a long
    // chain of N dependent instructions to be considered live as soon as one is
    // without going through N update cycles. This is not required for
    // correctness.
    return A.checkForAllUses(UsePred, *this, V, /* CheckBBLivenessOnly */ false,
                             DepClassTy::REQUIRED,
                             /* IgnoreDroppableUses */ false);
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

    bool IsKnown;
    return AA::isAssumedReadOnly(A, CallIRP, *this, IsKnown);
  }
};

struct AAIsDeadFloating : public AAIsDeadValueImpl {
  AAIsDeadFloating(const IRPosition &IRP, Attributor &A)
      : AAIsDeadValueImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAIsDeadValueImpl::initialize(A);

    if (isa<UndefValue>(getAssociatedValue())) {
      indicatePessimisticFixpoint();
      return;
    }

    Instruction *I = dyn_cast<Instruction>(&getAssociatedValue());
    if (!isAssumedSideEffectFree(A, I)) {
      if (!isa_and_nonnull<StoreInst>(I))
        indicatePessimisticFixpoint();
      else
        removeAssumedBits(HAS_NO_EFFECT);
    }
  }

  bool isDeadStore(Attributor &A, StoreInst &SI) {
    // Lang ref now states volatile store is not UB/dead, let's skip them.
    if (SI.isVolatile())
      return false;

    bool UsedAssumedInformation = false;
    SmallSetVector<Value *, 4> PotentialCopies;
    if (!AA::getPotentialCopiesOfStoredValue(A, SI, PotentialCopies, *this,
                                             UsedAssumedInformation)) {
      LLVM_DEBUG(
          dbgs()
          << "[AAIsDead] Could not determine potential copies of store!\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << "[AAIsDead] Store has " << PotentialCopies.size()
                      << " potential copies.\n");
    return llvm::all_of(PotentialCopies, [&](Value *V) {
      if (A.isAssumedDead(IRPosition::value(*V), this, nullptr,
                          UsedAssumedInformation))
        return true;
      LLVM_DEBUG(dbgs() << "[AAIsDead] Potential copy " << *V
                        << " is assumed live!\n");
      return false;
    });
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    Instruction *I = dyn_cast<Instruction>(&getAssociatedValue());
    if (isa_and_nonnull<StoreInst>(I))
      if (isValidState())
        return "assumed-dead-store";
    return AAIsDeadValueImpl::getAsStr();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    Instruction *I = dyn_cast<Instruction>(&getAssociatedValue());
    if (auto *SI = dyn_cast_or_null<StoreInst>(I)) {
      if (!isDeadStore(A, *SI))
        return indicatePessimisticFixpoint();
    } else {
      if (!isAssumedSideEffectFree(A, I))
        return indicatePessimisticFixpoint();
      if (!areAllUsesAssumedDead(A, getAssociatedValue()))
        return indicatePessimisticFixpoint();
    }
    return ChangeStatus::UNCHANGED;
  }

  bool isRemovableStore() const override {
    return isAssumed(IS_REMOVABLE) && isa<StoreInst>(&getAssociatedValue());
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    Value &V = getAssociatedValue();
    if (auto *I = dyn_cast<Instruction>(&V)) {
      // If we get here we basically know the users are all dead. We check if
      // isAssumedSideEffectFree returns true here again because it might not be
      // the case and only the users are dead but the instruction (=call) is
      // still needed.
      if (isa<StoreInst>(I) ||
          (isAssumedSideEffectFree(A, I) && !isa<InvokeInst>(I))) {
        A.deleteAfterManifest(*I);
        return ChangeStatus::CHANGED;
      }
    }
    return ChangeStatus::UNCHANGED;
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
    AAIsDeadFloating::initialize(A);
    if (!A.isFunctionIPOAmendable(*getAnchorScope()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    Argument &Arg = *getAssociatedArgument();
    if (A.isValidFunctionSignatureRewrite(Arg, /* ReplacementTypes */ {}))
      if (A.registerFunctionSignatureRewrite(
              Arg, /* ReplacementTypes */ {},
              Attributor::ArgumentReplacementInfo::CalleeRepairCBTy{},
              Attributor::ArgumentReplacementInfo::ACSRepairCBTy{})) {
        return ChangeStatus::CHANGED;
      }
    return ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_ARG_ATTR(IsDead) }
};

struct AAIsDeadCallSiteArgument : public AAIsDeadValueImpl {
  AAIsDeadCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAIsDeadValueImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAIsDeadValueImpl::initialize(A);
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
      : AAIsDeadFloating(IRP, A) {}

  /// See AAIsDead::isAssumedDead().
  bool isAssumedDead() const override {
    return AAIsDeadFloating::isAssumedDead() && IsAssumedSideEffectFree;
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    AAIsDeadFloating::initialize(A);
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
  bool IsAssumedSideEffectFree = true;
};

struct AAIsDeadReturned : public AAIsDeadValueImpl {
  AAIsDeadReturned(const IRPosition &IRP, Attributor &A)
      : AAIsDeadValueImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {

    bool UsedAssumedInformation = false;
    A.checkForAllInstructions([](Instruction &) { return true; }, *this,
                              {Instruction::Ret}, UsedAssumedInformation);

    auto PredForCallSite = [&](AbstractCallSite ACS) {
      if (ACS.isCallbackCall() || !ACS.getInstruction())
        return false;
      return areAllUsesAssumedDead(A, *ACS.getInstruction());
    };

    if (!A.checkForAllCallSites(PredForCallSite, *this, true,
                                UsedAssumedInformation))
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
    bool UsedAssumedInformation = false;
    A.checkForAllInstructions(RetInstPred, *this, {Instruction::Ret},
                              UsedAssumedInformation);
    return AnyChange ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override { STATS_DECLTRACK_FNRET_ATTR(IsDead) }
};

struct AAIsDeadFunction : public AAIsDead {
  AAIsDeadFunction(const IRPosition &IRP, Attributor &A) : AAIsDead(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Function *F = getAnchorScope();
    if (!F || F->isDeclaration() || !A.isRunOn(*F)) {
      indicatePessimisticFixpoint();
      return;
    }
    ToBeExploredFrom.insert(&F->getEntryBlock().front());
    assumeLive(A, F->getEntryBlock());
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
        HasChanged = ChangeStatus::CHANGED;
      }

    return HasChanged;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override;

  bool isEdgeDead(const BasicBlock *From, const BasicBlock *To) const override {
    assert(From->getParent() == getAnchorScope() &&
           To->getParent() == getAnchorScope() &&
           "Used AAIsDead of the wrong function");
    return isValidState() && !AssumedLiveEdges.count(std::make_pair(From, To));
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
    Optional<Constant *> C =
        A.getAssumedConstant(*BI.getCondition(), AA, UsedAssumedInformation);
    if (!C || isa_and_nonnull<UndefValue>(*C)) {
      // No value yet, assume both edges are dead.
    } else if (isa_and_nonnull<ConstantInt>(*C)) {
      const BasicBlock *SuccBB =
          BI.getSuccessor(1 - cast<ConstantInt>(*C)->getValue().getZExtValue());
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
  Optional<Constant *> C =
      A.getAssumedConstant(*SI.getCondition(), AA, UsedAssumedInformation);
  if (!C || isa_and_nonnull<UndefValue>(C.value())) {
    // No value yet, assume all edges are dead.
  } else if (isa_and_nonnull<ConstantInt>(C.value())) {
    for (const auto &CaseIt : SI.cases()) {
      if (CaseIt.getCaseValue() == C.value()) {
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
    while (!I->isTerminator() && !isa<CallBase>(I))
      I = I->getNextNode();

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
    } else if (AliveSuccessors.empty() ||
               (I->isTerminator() &&
                AliveSuccessors.size() < I->getNumSuccessors())) {
      if (KnownDeadEnds.insert(I))
        Change = ChangeStatus::CHANGED;
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
        auto Edge = std::make_pair(I->getParent(), AliveSuccessor->getParent());
        if (AssumedLiveEdges.insert(Edge).second)
          Change = ChangeStatus::CHANGED;
        if (assumeLive(A, *AliveSuccessor->getParent()))
          Worklist.push_back(AliveSuccessor);
      }
    }
  }

  // Check if the content of ToBeExploredFrom changed, ignore the order.
  if (NewToBeExploredFrom.size() != ToBeExploredFrom.size() ||
      llvm::any_of(NewToBeExploredFrom, [&](const Instruction *I) {
        return !ToBeExploredFrom.count(I);
      })) {
    Change = ChangeStatus::CHANGED;
    ToBeExploredFrom = std::move(NewToBeExploredFrom);
  }

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
} // namespace

/// -------------------- Dereferenceable Argument Attribute --------------------

namespace {
struct AADereferenceableImpl : AADereferenceable {
  AADereferenceableImpl(const IRPosition &IRP, Attributor &A)
      : AADereferenceable(IRP, A) {}
  using StateType = DerefState;

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Value &V = *getAssociatedValue().stripPointerCasts();
    SmallVector<Attribute, 4> Attrs;
    getAttrs({Attribute::Dereferenceable, Attribute::DereferenceableOrNull},
             Attrs, /* IgnoreSubsumingPositions */ false, &A);
    for (const Attribute &Attr : Attrs)
      takeKnownDerefBytesMaximum(Attr.getValueAsInt());

    const IRPosition &IRP = this->getIRPosition();
    NonNullAA = &A.getAAFor<AANonNull>(*this, IRP, DepClassTy::NONE);

    bool CanBeNull, CanBeFreed;
    takeKnownDerefBytesMaximum(V.getPointerDereferenceableBytes(
        A.getDataLayout(), CanBeNull, CanBeFreed));

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

    Optional<MemoryLocation> Loc = MemoryLocation::getOrNone(I);
    if (!Loc || Loc->Ptr != UseV || !Loc->Size.isPrecise() || I->isVolatile())
      return;

    int64_t Offset;
    const Value *Base = GetPointerBaseWithConstantOffset(
        Loc->Ptr, Offset, A.getDataLayout(), /*AllowNonInbounds*/ true);
    if (Base && Base == &getAssociatedValue())
      State.addAccessedBytes(Offset, Loc->Size.getValue());
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

    bool Stripped;
    bool UsedAssumedInformation = false;
    SmallVector<AA::ValueAndContext> Values;
    if (!A.getAssumedSimplifiedValues(getIRPosition(), *this, Values,
                                      AA::AnyScope, UsedAssumedInformation)) {
      Values.push_back({getAssociatedValue(), getCtxI()});
      Stripped = false;
    } else {
      Stripped = Values.size() != 1 ||
                 Values.front().getValue() != &getAssociatedValue();
    }

    const DataLayout &DL = A.getDataLayout();
    DerefState T;

    auto VisitValueCB = [&](const Value &V) -> bool {
      unsigned IdxWidth =
          DL.getIndexSizeInBits(V.getType()->getPointerAddressSpace());
      APInt Offset(IdxWidth, 0);
      const Value *Base = stripAndAccumulateOffsets(
          A, *this, &V, DL, Offset, /* GetMinOffset */ false,
          /* AllowNonInbounds */ true);

      const auto &AA = A.getAAFor<AADereferenceable>(
          *this, IRPosition::value(*Base), DepClassTy::REQUIRED);
      int64_t DerefBytes = 0;
      if (!Stripped && this == &AA) {
        // Use IR information if we did not strip anything.
        // TODO: track globally.
        bool CanBeNull, CanBeFreed;
        DerefBytes =
            Base->getPointerDereferenceableBytes(DL, CanBeNull, CanBeFreed);
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
          // dereferenceable bytes in a circular loop now, which will simply
          // drive them down to the known value in a very slow way which we
          // can accelerate.
          T.indicatePessimisticFixpoint();
        }
      }

      return T.isValidState();
    };

    for (const auto &VAC : Values)
      if (!VisitValueCB(*VAC.getValue()))
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
} // namespace

// ------------------------ Align Argument Attribute ------------------------

namespace {
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

      uint32_t gcd = std::gcd(uint32_t(abs((int32_t)Offset)), Alignment);
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

    Value &V = *getAssociatedValue().stripPointerCasts();
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
          if (SI->getAlign() < getAssumedAlign()) {
            STATS_DECLTRACK(AAAlign, Store,
                            "Number of times alignment added to a store");
            SI->setAlignment(getAssumedAlign());
            LoadStoreChanged = ChangeStatus::CHANGED;
          }
      } else if (auto *LI = dyn_cast<LoadInst>(U.getUser())) {
        if (LI->getPointerOperand() == &AssociatedValue)
          if (LI->getAlign() < getAssumedAlign()) {
            LI->setAlignment(getAssumedAlign());
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
  void getDeducedAttributes(LLVMContext &Ctx,
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
    return "align<" + std::to_string(getKnownAlign().value()) + "-" +
           std::to_string(getAssumedAlign().value()) + ">";
  }
};

/// Align attribute for a floating value.
struct AAAlignFloating : AAAlignImpl {
  AAAlignFloating(const IRPosition &IRP, Attributor &A) : AAAlignImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const DataLayout &DL = A.getDataLayout();

    bool Stripped;
    bool UsedAssumedInformation = false;
    SmallVector<AA::ValueAndContext> Values;
    if (!A.getAssumedSimplifiedValues(getIRPosition(), *this, Values,
                                      AA::AnyScope, UsedAssumedInformation)) {
      Values.push_back({getAssociatedValue(), getCtxI()});
      Stripped = false;
    } else {
      Stripped = Values.size() != 1 ||
                 Values.front().getValue() != &getAssociatedValue();
    }

    StateType T;
    auto VisitValueCB = [&](Value &V) -> bool {
      if (isa<UndefValue>(V) || isa<ConstantPointerNull>(V))
        return true;
      const auto &AA = A.getAAFor<AAAlign>(*this, IRPosition::value(V),
                                           DepClassTy::REQUIRED);
      if (!Stripped && this == &AA) {
        int64_t Offset;
        unsigned Alignment = 1;
        if (const Value *Base =
                GetPointerBaseWithConstantOffset(&V, Offset, DL)) {
          // TODO: Use AAAlign for the base too.
          Align PA = Base->getPointerAlignment(DL);
          // BasePointerAddr + Offset = Alignment * Q for some integer Q.
          // So we can say that the maximum power of two which is a divisor of
          // gcd(Offset, Alignment) is an alignment.

          uint32_t gcd =
              std::gcd(uint32_t(abs((int32_t)Offset)), uint32_t(PA.value()));
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

    for (const auto &VAC : Values) {
      if (!VisitValueCB(*VAC.getValue()))
        return indicatePessimisticFixpoint();
    }

    //  TODO: If we know we visited all incoming values, thus no are assumed
    //  dead, we can take the known information from the state T.
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
      takeKnownMaximum(ArgAlignAA.getKnownAlign().value());
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
} // namespace

/// ------------------ Function No-Return Attribute ----------------------------
namespace {
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
  ChangeStatus updateImpl(Attributor &A) override {
    auto CheckForNoReturn = [](Instruction &) { return false; };
    bool UsedAssumedInformation = false;
    if (!A.checkForAllInstructions(CheckForNoReturn, *this,
                                   {(unsigned)Instruction::Ret},
                                   UsedAssumedInformation))
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
} // namespace

/// ----------------------- Instance Info ---------------------------------

namespace {
/// A class to hold the state of for no-capture attributes.
struct AAInstanceInfoImpl : public AAInstanceInfo {
  AAInstanceInfoImpl(const IRPosition &IRP, Attributor &A)
      : AAInstanceInfo(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    Value &V = getAssociatedValue();
    if (auto *C = dyn_cast<Constant>(&V)) {
      if (C->isThreadDependent())
        indicatePessimisticFixpoint();
      else
        indicateOptimisticFixpoint();
      return;
    }
    if (auto *CB = dyn_cast<CallBase>(&V))
      if (CB->arg_size() == 0 && !CB->mayHaveSideEffects() &&
          !CB->mayReadFromMemory()) {
        indicateOptimisticFixpoint();
        return;
      }
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    Value &V = getAssociatedValue();
    const Function *Scope = nullptr;
    if (auto *I = dyn_cast<Instruction>(&V))
      Scope = I->getFunction();
    if (auto *A = dyn_cast<Argument>(&V)) {
      Scope = A->getParent();
      if (!Scope->hasLocalLinkage())
        return Changed;
    }
    if (!Scope)
      return indicateOptimisticFixpoint();

    auto &NoRecurseAA = A.getAAFor<AANoRecurse>(
        *this, IRPosition::function(*Scope), DepClassTy::OPTIONAL);
    if (NoRecurseAA.isAssumedNoRecurse())
      return Changed;

    auto UsePred = [&](const Use &U, bool &Follow) {
      const Instruction *UserI = dyn_cast<Instruction>(U.getUser());
      if (!UserI || isa<GetElementPtrInst>(UserI) || isa<CastInst>(UserI) ||
          isa<PHINode>(UserI) || isa<SelectInst>(UserI)) {
        Follow = true;
        return true;
      }
      if (isa<LoadInst>(UserI) || isa<CmpInst>(UserI) ||
          (isa<StoreInst>(UserI) &&
           cast<StoreInst>(UserI)->getValueOperand() != U.get()))
        return true;
      if (auto *CB = dyn_cast<CallBase>(UserI)) {
        // This check is not guaranteeing uniqueness but for now that we cannot
        // end up with two versions of \p U thinking it was one.
        if (!CB->getCalledFunction() ||
            !CB->getCalledFunction()->hasLocalLinkage())
          return true;
        if (!CB->isArgOperand(&U))
          return false;
        const auto &ArgInstanceInfoAA = A.getAAFor<AAInstanceInfo>(
            *this, IRPosition::callsite_argument(*CB, CB->getArgOperandNo(&U)),
            DepClassTy::OPTIONAL);
        if (!ArgInstanceInfoAA.isAssumedUniqueForAnalysis())
          return false;
        // If this call base might reach the scope again we might forward the
        // argument back here. This is very conservative.
        if (AA::isPotentiallyReachable(
                A, *CB, *Scope, *this,
                [Scope](const Function &Fn) { return &Fn != Scope; }))
          return false;
        return true;
      }
      return false;
    };

    auto EquivalentUseCB = [&](const Use &OldU, const Use &NewU) {
      if (auto *SI = dyn_cast<StoreInst>(OldU.getUser())) {
        auto *Ptr = SI->getPointerOperand()->stripPointerCasts();
        if ((isa<AllocaInst>(Ptr) || isNoAliasCall(Ptr)) &&
            AA::isDynamicallyUnique(A, *this, *Ptr))
          return true;
      }
      return false;
    };

    if (!A.checkForAllUses(UsePred, *this, V, /* CheckBBLivenessOnly */ true,
                           DepClassTy::OPTIONAL,
                           /* IgnoreDroppableUses */ true, EquivalentUseCB))
      return indicatePessimisticFixpoint();

    return Changed;
  }

  /// See AbstractState::getAsStr().
  const std::string getAsStr() const override {
    return isAssumedUniqueForAnalysis() ? "<unique [fAa]>" : "<unknown>";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}
};

/// InstanceInfo attribute for floating values.
struct AAInstanceInfoFloating : AAInstanceInfoImpl {
  AAInstanceInfoFloating(const IRPosition &IRP, Attributor &A)
      : AAInstanceInfoImpl(IRP, A) {}
};

/// NoCapture attribute for function arguments.
struct AAInstanceInfoArgument final : AAInstanceInfoFloating {
  AAInstanceInfoArgument(const IRPosition &IRP, Attributor &A)
      : AAInstanceInfoFloating(IRP, A) {}
};

/// InstanceInfo attribute for call site arguments.
struct AAInstanceInfoCallSiteArgument final : AAInstanceInfoImpl {
  AAInstanceInfoCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAInstanceInfoImpl(IRP, A) {}

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
    auto &ArgAA =
        A.getAAFor<AAInstanceInfo>(*this, ArgPos, DepClassTy::REQUIRED);
    return clampStateAndIndicateChange(getState(), ArgAA.getState());
  }
};

/// InstanceInfo attribute for function return value.
struct AAInstanceInfoReturned final : AAInstanceInfoImpl {
  AAInstanceInfoReturned(const IRPosition &IRP, Attributor &A)
      : AAInstanceInfoImpl(IRP, A) {
    llvm_unreachable("InstanceInfo is not applicable to function returns!");
  }

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    llvm_unreachable("InstanceInfo is not applicable to function returns!");
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable("InstanceInfo is not applicable to function returns!");
  }
};

/// InstanceInfo attribute deduction for a call site return value.
struct AAInstanceInfoCallSiteReturned final : AAInstanceInfoFloating {
  AAInstanceInfoCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAInstanceInfoFloating(IRP, A) {}
};
} // namespace

/// ----------------------- Variable Capturing ---------------------------------

namespace {
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
  void getDeducedAttributes(LLVMContext &Ctx,
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

  /// Check the use \p U and update \p State accordingly. Return true if we
  /// should continue to update the state.
  bool checkUse(Attributor &A, AANoCapture::StateType &State, const Use &U,
                bool &Follow) {
    Instruction *UInst = cast<Instruction>(U.getUser());
    LLVM_DEBUG(dbgs() << "[AANoCapture] Check use: " << *U.get() << " in "
                      << *UInst << "\n");

    // Deal with ptr2int by following uses.
    if (isa<PtrToIntInst>(UInst)) {
      LLVM_DEBUG(dbgs() << " - ptr2int assume the worst!\n");
      return isCapturedIn(State, /* Memory */ true, /* Integer */ true,
                          /* Return */ true);
    }

    // For stores we already checked if we can follow them, if they make it
    // here we give up.
    if (isa<StoreInst>(UInst))
      return isCapturedIn(State, /* Memory */ true, /* Integer */ false,
                          /* Return */ false);

    // Explicitly catch return instructions.
    if (isa<ReturnInst>(UInst)) {
      if (UInst->getFunction() == getAnchorScope())
        return isCapturedIn(State, /* Memory */ false, /* Integer */ false,
                            /* Return */ true);
      return isCapturedIn(State, /* Memory */ true, /* Integer */ true,
                          /* Return */ true);
    }

    // For now we only use special logic for call sites. However, the tracker
    // itself knows about a lot of other non-capturing cases already.
    auto *CB = dyn_cast<CallBase>(UInst);
    if (!CB || !CB->isArgOperand(&U))
      return isCapturedIn(State, /* Memory */ true, /* Integer */ true,
                          /* Return */ true);

    unsigned ArgNo = CB->getArgOperandNo(&U);
    const IRPosition &CSArgPos = IRPosition::callsite_argument(*CB, ArgNo);
    // If we have a abstract no-capture attribute for the argument we can use
    // it to justify a non-capture attribute here. This allows recursion!
    auto &ArgNoCaptureAA =
        A.getAAFor<AANoCapture>(*this, CSArgPos, DepClassTy::REQUIRED);
    if (ArgNoCaptureAA.isAssumedNoCapture())
      return isCapturedIn(State, /* Memory */ false, /* Integer */ false,
                          /* Return */ false);
    if (ArgNoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
      Follow = true;
      return isCapturedIn(State, /* Memory */ false, /* Integer */ false,
                          /* Return */ false);
    }

    // Lastly, we could not find a reason no-capture can be assumed so we don't.
    return isCapturedIn(State, /* Memory */ true, /* Integer */ true,
                        /* Return */ true);
  }

  /// Update \p State according to \p CapturedInMem, \p CapturedInInt, and
  /// \p CapturedInRet, then return true if we should continue updating the
  /// state.
  static bool isCapturedIn(AANoCapture::StateType &State, bool CapturedInMem,
                           bool CapturedInInt, bool CapturedInRet) {
    LLVM_DEBUG(dbgs() << " - captures [Mem " << CapturedInMem << "|Int "
                      << CapturedInInt << "|Ret " << CapturedInRet << "]\n");
    if (CapturedInMem)
      State.removeAssumedBits(AANoCapture::NOT_CAPTURED_IN_MEM);
    if (CapturedInInt)
      State.removeAssumedBits(AANoCapture::NOT_CAPTURED_IN_INT);
    if (CapturedInRet)
      State.removeAssumedBits(AANoCapture::NOT_CAPTURED_IN_RET);
    return State.isAssumed(AANoCapture::NO_CAPTURE_MAYBE_RETURNED);
  }
};

ChangeStatus AANoCaptureImpl::updateImpl(Attributor &A) {
  const IRPosition &IRP = getIRPosition();
  Value *V = isArgumentPosition() ? IRP.getAssociatedArgument()
                                  : &IRP.getAssociatedValue();
  if (!V)
    return indicatePessimisticFixpoint();

  const Function *F =
      isArgumentPosition() ? IRP.getAssociatedFunction() : IRP.getAnchorScope();
  assert(F && "Expected a function!");
  const IRPosition &FnPos = IRPosition::function(*F);

  AANoCapture::StateType T;

  // Readonly means we cannot capture through memory.
  bool IsKnown;
  if (AA::isAssumedReadOnly(A, FnPos, *this, IsKnown)) {
    T.addKnownBits(NOT_CAPTURED_IN_MEM);
    if (IsKnown)
      addKnownBits(NOT_CAPTURED_IN_MEM);
  }

  // Make sure all returned values are different than the underlying value.
  // TODO: we could do this in a more sophisticated way inside
  //       AAReturnedValues, e.g., track all values that escape through returns
  //       directly somehow.
  auto CheckReturnedArgs = [&](const AAReturnedValues &RVAA) {
    if (!RVAA.getState().isValidState())
      return false;
    bool SeenConstant = false;
    for (const auto &It : RVAA.returned_values()) {
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

  auto IsDereferenceableOrNull = [&](Value *O, const DataLayout &DL) {
    const auto &DerefAA = A.getAAFor<AADereferenceable>(
        *this, IRPosition::value(*O), DepClassTy::OPTIONAL);
    return DerefAA.getAssumedDereferenceableBytes();
  };

  auto UseCheck = [&](const Use &U, bool &Follow) -> bool {
    switch (DetermineUseCaptureKind(U, IsDereferenceableOrNull)) {
    case UseCaptureKind::NO_CAPTURE:
      return true;
    case UseCaptureKind::MAY_CAPTURE:
      return checkUse(A, T, U, Follow);
    case UseCaptureKind::PASSTHROUGH:
      Follow = true;
      return true;
    }
    llvm_unreachable("Unexpected use capture kind!");
  };

  if (!A.checkForAllUses(UseCheck, *this, *V))
    return indicatePessimisticFixpoint();

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
} // namespace

/// ------------------ Value Simplify Attribute ----------------------------

bool ValueSimplifyStateType::unionAssumed(Optional<Value *> Other) {
  // FIXME: Add a typecast support.
  SimplifiedAssociatedValue = AA::combineOptionalValuesInAAValueLatice(
      SimplifiedAssociatedValue, Other, Ty);
  if (SimplifiedAssociatedValue == Optional<Value *>(nullptr))
    return false;

  LLVM_DEBUG({
    if (SimplifiedAssociatedValue)
      dbgs() << "[ValueSimplify] is assumed to be "
             << **SimplifiedAssociatedValue << "\n";
    else
      dbgs() << "[ValueSimplify] is assumed to be <none>\n";
  });
  return true;
}

namespace {
struct AAValueSimplifyImpl : AAValueSimplify {
  AAValueSimplifyImpl(const IRPosition &IRP, Attributor &A)
      : AAValueSimplify(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    if (getAssociatedValue().getType()->isVoidTy())
      indicatePessimisticFixpoint();
    if (A.hasSimplificationCallback(getIRPosition()))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    LLVM_DEBUG({
      dbgs() << "SAV: " << (bool)SimplifiedAssociatedValue << " ";
      if (SimplifiedAssociatedValue && *SimplifiedAssociatedValue)
        dbgs() << "SAV: " << **SimplifiedAssociatedValue << " ";
    });
    return isValidState() ? (isAtFixpoint() ? "simplified" : "maybe-simple")
                          : "not-simple";
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

  /// See AAValueSimplify::getAssumedSimplifiedValue()
  Optional<Value *> getAssumedSimplifiedValue(Attributor &A) const override {
    return SimplifiedAssociatedValue;
  }

  /// Ensure the return value is \p V with type \p Ty, if not possible return
  /// nullptr. If \p Check is true we will only verify such an operation would
  /// suceed and return a non-nullptr value if that is the case. No IR is
  /// generated or modified.
  static Value *ensureType(Attributor &A, Value &V, Type &Ty, Instruction *CtxI,
                           bool Check) {
    if (auto *TypedV = AA::getWithType(V, Ty))
      return TypedV;
    if (CtxI && V.getType()->canLosslesslyBitCastTo(&Ty))
      return Check ? &V
                   : BitCastInst::CreatePointerBitCastOrAddrSpaceCast(&V, &Ty,
                                                                      "", CtxI);
    return nullptr;
  }

  /// Reproduce \p I with type \p Ty or return nullptr if that is not posisble.
  /// If \p Check is true we will only verify such an operation would suceed and
  /// return a non-nullptr value if that is the case. No IR is generated or
  /// modified.
  static Value *reproduceInst(Attributor &A,
                              const AbstractAttribute &QueryingAA,
                              Instruction &I, Type &Ty, Instruction *CtxI,
                              bool Check, ValueToValueMapTy &VMap) {
    assert(CtxI && "Cannot reproduce an instruction without context!");
    if (Check && (I.mayReadFromMemory() ||
                  !isSafeToSpeculativelyExecute(&I, CtxI, /* DT */ nullptr,
                                                /* TLI */ nullptr)))
      return nullptr;
    for (Value *Op : I.operands()) {
      Value *NewOp = reproduceValue(A, QueryingAA, *Op, Ty, CtxI, Check, VMap);
      if (!NewOp) {
        assert(Check && "Manifest of new value unexpectedly failed!");
        return nullptr;
      }
      if (!Check)
        VMap[Op] = NewOp;
    }
    if (Check)
      return &I;

    Instruction *CloneI = I.clone();
    // TODO: Try to salvage debug information here.
    CloneI->setDebugLoc(DebugLoc());
    VMap[&I] = CloneI;
    CloneI->insertBefore(CtxI);
    RemapInstruction(CloneI, VMap);
    return CloneI;
  }

  /// Reproduce \p V with type \p Ty or return nullptr if that is not posisble.
  /// If \p Check is true we will only verify such an operation would suceed and
  /// return a non-nullptr value if that is the case. No IR is generated or
  /// modified.
  static Value *reproduceValue(Attributor &A,
                               const AbstractAttribute &QueryingAA, Value &V,
                               Type &Ty, Instruction *CtxI, bool Check,
                               ValueToValueMapTy &VMap) {
    if (const auto &NewV = VMap.lookup(&V))
      return NewV;
    bool UsedAssumedInformation = false;
    Optional<Value *> SimpleV = A.getAssumedSimplified(
        V, QueryingAA, UsedAssumedInformation, AA::Interprocedural);
    if (!SimpleV.has_value())
      return PoisonValue::get(&Ty);
    Value *EffectiveV = &V;
    if (SimpleV.value())
      EffectiveV = SimpleV.value();
    if (auto *C = dyn_cast<Constant>(EffectiveV))
      return C;
    if (CtxI && AA::isValidAtPosition(AA::ValueAndContext(*EffectiveV, *CtxI),
                                      A.getInfoCache()))
      return ensureType(A, *EffectiveV, Ty, CtxI, Check);
    if (auto *I = dyn_cast<Instruction>(EffectiveV))
      if (Value *NewV = reproduceInst(A, QueryingAA, *I, Ty, CtxI, Check, VMap))
        return ensureType(A, *NewV, Ty, CtxI, Check);
    return nullptr;
  }

  /// Return a value we can use as replacement for the associated one, or
  /// nullptr if we don't have one that makes sense.
  Value *manifestReplacementValue(Attributor &A, Instruction *CtxI) const {
    Value *NewV = SimplifiedAssociatedValue
                      ? SimplifiedAssociatedValue.value()
                      : UndefValue::get(getAssociatedType());
    if (NewV && NewV != &getAssociatedValue()) {
      ValueToValueMapTy VMap;
      // First verify we can reprduce the value with the required type at the
      // context location before we actually start modifying the IR.
      if (reproduceValue(A, *this, *NewV, *getAssociatedType(), CtxI,
                         /* CheckOnly */ true, VMap))
        return reproduceValue(A, *this, *NewV, *getAssociatedType(), CtxI,
                              /* CheckOnly */ false, VMap);
    }
    return nullptr;
  }

  /// Helper function for querying AAValueSimplify and updating candidate.
  /// \param IRP The value position we are trying to unify with SimplifiedValue
  bool checkAndUpdate(Attributor &A, const AbstractAttribute &QueryingAA,
                      const IRPosition &IRP, bool Simplify = true) {
    bool UsedAssumedInformation = false;
    Optional<Value *> QueryingValueSimplified = &IRP.getAssociatedValue();
    if (Simplify)
      QueryingValueSimplified = A.getAssumedSimplified(
          IRP, QueryingAA, UsedAssumedInformation, AA::Interprocedural);
    return unionAssumed(QueryingValueSimplified);
  }

  /// Returns a candidate is found or not
  template <typename AAType> bool askSimplifiedValueFor(Attributor &A) {
    if (!getAssociatedValue().getType()->isIntegerTy())
      return false;

    // This will also pass the call base context.
    const auto &AA =
        A.getAAFor<AAType>(*this, getIRPosition(), DepClassTy::NONE);

    Optional<Constant *> COpt = AA.getAssumedConstant(A);

    if (!COpt) {
      SimplifiedAssociatedValue = llvm::None;
      A.recordDependence(AA, *this, DepClassTy::OPTIONAL);
      return true;
    }
    if (auto *C = *COpt) {
      SimplifiedAssociatedValue = C;
      A.recordDependence(AA, *this, DepClassTy::OPTIONAL);
      return true;
    }
    return false;
  }

  bool askSimplifiedValueForOtherAAs(Attributor &A) {
    if (askSimplifiedValueFor<AAValueConstantRange>(A))
      return true;
    if (askSimplifiedValueFor<AAPotentialConstantValues>(A))
      return true;
    return false;
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;
    for (auto &U : getAssociatedValue().uses()) {
      // Check if we need to adjust the insertion point to make sure the IR is
      // valid.
      Instruction *IP = dyn_cast<Instruction>(U.getUser());
      if (auto *PHI = dyn_cast_or_null<PHINode>(IP))
        IP = PHI->getIncomingBlock(U)->getTerminator();
      if (auto *NewV = manifestReplacementValue(A, IP)) {
        LLVM_DEBUG(dbgs() << "[ValueSimplify] " << getAssociatedValue()
                          << " -> " << *NewV << " :: " << *this << "\n");
        if (A.changeUseAfterManifest(U, *NewV))
          Changed = ChangeStatus::CHANGED;
      }
    }

    return Changed | AAValueSimplify::manifest(A);
  }

  /// See AbstractState::indicatePessimisticFixpoint(...).
  ChangeStatus indicatePessimisticFixpoint() override {
    SimplifiedAssociatedValue = &getAssociatedValue();
    return AAValueSimplify::indicatePessimisticFixpoint();
  }
};

struct AAValueSimplifyArgument final : AAValueSimplifyImpl {
  AAValueSimplifyArgument(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyImpl(IRP, A) {}

  void initialize(Attributor &A) override {
    AAValueSimplifyImpl::initialize(A);
    if (!getAnchorScope() || getAnchorScope()->isDeclaration())
      indicatePessimisticFixpoint();
    if (hasAttr({Attribute::InAlloca, Attribute::Preallocated,
                 Attribute::StructRet, Attribute::Nest, Attribute::ByVal},
                /* IgnoreSubsumingPositions */ true))
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
      bool IsKnown;
      if (!AA::isAssumedReadOnly(A, getIRPosition(), *this, IsKnown))
        return indicatePessimisticFixpoint();
    }

    auto Before = SimplifiedAssociatedValue;

    auto PredForCallSite = [&](AbstractCallSite ACS) {
      const IRPosition &ACSArgPos =
          IRPosition::callsite_argument(ACS, getCallSiteArgNo());
      // Check if a coresponding argument was found or if it is on not
      // associated (which can happen for callback calls).
      if (ACSArgPos.getPositionKind() == IRPosition::IRP_INVALID)
        return false;

      // Simplify the argument operand explicitly and check if the result is
      // valid in the current scope. This avoids refering to simplified values
      // in other functions, e.g., we don't want to say a an argument in a
      // static function is actually an argument in a different function.
      bool UsedAssumedInformation = false;
      Optional<Constant *> SimpleArgOp =
          A.getAssumedConstant(ACSArgPos, *this, UsedAssumedInformation);
      if (!SimpleArgOp)
        return true;
      if (!SimpleArgOp.value())
        return false;
      if (!AA::isDynamicallyUnique(A, *this, **SimpleArgOp))
        return false;
      return unionAssumed(*SimpleArgOp);
    };

    // Generate a answer specific to a call site context.
    bool Success;
    bool UsedAssumedInformation = false;
    if (hasCallBaseContext() &&
        getCallBaseContext()->getCalledFunction() == Arg->getParent())
      Success = PredForCallSite(
          AbstractCallSite(&getCallBaseContext()->getCalledOperandUse()));
    else
      Success = A.checkForAllCallSites(PredForCallSite, *this, true,
                                       UsedAssumedInformation);

    if (!Success)
      if (!askSimplifiedValueForOtherAAs(A))
        return indicatePessimisticFixpoint();

    // If a candidate was found in this update, return CHANGED.
    return Before == SimplifiedAssociatedValue ? ChangeStatus::UNCHANGED
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

  /// See AAValueSimplify::getAssumedSimplifiedValue()
  Optional<Value *> getAssumedSimplifiedValue(Attributor &A) const override {
    if (!isValidState())
      return nullptr;
    return SimplifiedAssociatedValue;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto Before = SimplifiedAssociatedValue;

    auto ReturnInstCB = [&](Instruction &I) {
      auto &RI = cast<ReturnInst>(I);
      return checkAndUpdate(
          A, *this,
          IRPosition::value(*RI.getReturnValue(), getCallBaseContext()));
    };

    bool UsedAssumedInformation = false;
    if (!A.checkForAllInstructions(ReturnInstCB, *this, {Instruction::Ret},
                                   UsedAssumedInformation))
      if (!askSimplifiedValueForOtherAAs(A))
        return indicatePessimisticFixpoint();

    // If a candidate was found in this update, return CHANGED.
    return Before == SimplifiedAssociatedValue ? ChangeStatus::UNCHANGED
                                               : ChangeStatus ::CHANGED;
  }

  ChangeStatus manifest(Attributor &A) override {
    // We queried AAValueSimplify for the returned values so they will be
    // replaced if a simplified form was found. Nothing to do here.
    return ChangeStatus::UNCHANGED;
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
    AAValueSimplifyImpl::initialize(A);
    Value &V = getAnchorValue();

    // TODO: add other stuffs
    if (isa<Constant>(V))
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto Before = SimplifiedAssociatedValue;
    if (!askSimplifiedValueForOtherAAs(A))
      return indicatePessimisticFixpoint();

    // If a candidate was found in this update, return CHANGED.
    return Before == SimplifiedAssociatedValue ? ChangeStatus::UNCHANGED
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
    SimplifiedAssociatedValue = nullptr;
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

struct AAValueSimplifyCallSiteReturned : AAValueSimplifyImpl {
  AAValueSimplifyCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAValueSimplifyImpl(IRP, A) {}

  void initialize(Attributor &A) override {
    AAValueSimplifyImpl::initialize(A);
    Function *Fn = getAssociatedFunction();
    if (!Fn) {
      indicatePessimisticFixpoint();
      return;
    }
    for (Argument &Arg : Fn->args()) {
      if (Arg.hasReturnedAttr()) {
        auto IRP = IRPosition::callsite_argument(*cast<CallBase>(getCtxI()),
                                                 Arg.getArgNo());
        if (IRP.getPositionKind() == IRPosition::IRP_CALL_SITE_ARGUMENT &&
            checkAndUpdate(A, *this, IRP))
          indicateOptimisticFixpoint();
        else
          indicatePessimisticFixpoint();
        return;
      }
    }
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto Before = SimplifiedAssociatedValue;
    auto &RetAA = A.getAAFor<AAReturnedValues>(
        *this, IRPosition::function(*getAssociatedFunction()),
        DepClassTy::REQUIRED);
    auto PredForReturned =
        [&](Value &RetVal, const SmallSetVector<ReturnInst *, 4> &RetInsts) {
          bool UsedAssumedInformation = false;
          Optional<Value *> CSRetVal = A.translateArgumentToCallSiteContent(
              &RetVal, *cast<CallBase>(getCtxI()), *this,
              UsedAssumedInformation);
          SimplifiedAssociatedValue = AA::combineOptionalValuesInAAValueLatice(
              SimplifiedAssociatedValue, CSRetVal, getAssociatedType());
          return SimplifiedAssociatedValue != Optional<Value *>(nullptr);
        };
    if (!RetAA.checkForAllReturnedValuesAndReturnInsts(PredForReturned))
      if (!askSimplifiedValueForOtherAAs(A))
        return indicatePessimisticFixpoint();
    return Before == SimplifiedAssociatedValue ? ChangeStatus::UNCHANGED
                                               : ChangeStatus ::CHANGED;
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
    // TODO: We should avoid simplification duplication to begin with.
    auto *FloatAA = A.lookupAAFor<AAValueSimplify>(
        IRPosition::value(getAssociatedValue()), this, DepClassTy::NONE);
    if (FloatAA && FloatAA->getState().isValidState())
      return Changed;

    if (auto *NewV = manifestReplacementValue(A, getCtxI())) {
      Use &U = cast<CallBase>(&getAnchorValue())
                   ->getArgOperandUse(getCallSiteArgNo());
      if (A.changeUseAfterManifest(U, *NewV))
        Changed = ChangeStatus::CHANGED;
    }

    return Changed | AAValueSimplify::manifest(A);
  }

  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(value_simplify)
  }
};
} // namespace

/// ----------------------- Heap-To-Stack Conversion ---------------------------
namespace {
struct AAHeapToStackFunction final : public AAHeapToStack {

  struct AllocationInfo {
    /// The call that allocates the memory.
    CallBase *const CB;

    /// The library function id for the allocation.
    LibFunc LibraryFunctionId = NotLibFunc;

    /// The status wrt. a rewrite.
    enum {
      STACK_DUE_TO_USE,
      STACK_DUE_TO_FREE,
      INVALID,
    } Status = STACK_DUE_TO_USE;

    /// Flag to indicate if we encountered a use that might free this allocation
    /// but which is not in the deallocation infos.
    bool HasPotentiallyFreeingUnknownUses = false;

    /// Flag to indicate that we should place the new alloca in the function
    /// entry block rather than where the call site (CB) is.
    bool MoveAllocaIntoEntry = true;

    /// The set of free calls that use this allocation.
    SmallSetVector<CallBase *, 1> PotentialFreeCalls{};
  };

  struct DeallocationInfo {
    /// The call that deallocates the memory.
    CallBase *const CB;
    /// The value freed by the call.
    Value *FreedOp;

    /// Flag to indicate if we don't know all objects this deallocation might
    /// free.
    bool MightFreeUnknownObjects = false;

    /// The set of allocation calls that are potentially freed.
    SmallSetVector<CallBase *, 1> PotentialAllocationCalls{};
  };

  AAHeapToStackFunction(const IRPosition &IRP, Attributor &A)
      : AAHeapToStack(IRP, A) {}

  ~AAHeapToStackFunction() {
    // Ensure we call the destructor so we release any memory allocated in the
    // sets.
    for (auto &It : AllocationInfos)
      It.second->~AllocationInfo();
    for (auto &It : DeallocationInfos)
      It.second->~DeallocationInfo();
  }

  void initialize(Attributor &A) override {
    AAHeapToStack::initialize(A);

    const Function *F = getAnchorScope();
    const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);

    auto AllocationIdentifierCB = [&](Instruction &I) {
      CallBase *CB = dyn_cast<CallBase>(&I);
      if (!CB)
        return true;
      if (Value *FreedOp = getFreedOperand(CB, TLI)) {
        DeallocationInfos[CB] = new (A.Allocator) DeallocationInfo{CB, FreedOp};
        return true;
      }
      // To do heap to stack, we need to know that the allocation itself is
      // removable once uses are rewritten, and that we can initialize the
      // alloca to the same pattern as the original allocation result.
      if (isRemovableAlloc(CB, TLI)) {
        auto *I8Ty = Type::getInt8Ty(CB->getParent()->getContext());
        if (nullptr != getInitialValueOfAllocation(CB, TLI, I8Ty)) {
          AllocationInfo *AI = new (A.Allocator) AllocationInfo{CB};
          AllocationInfos[CB] = AI;
          if (TLI)
            TLI->getLibFunc(*CB, AI->LibraryFunctionId);
        }
      }
      return true;
    };

    bool UsedAssumedInformation = false;
    bool Success = A.checkForAllCallLikeInstructions(
        AllocationIdentifierCB, *this, UsedAssumedInformation,
        /* CheckBBLivenessOnly */ false,
        /* CheckPotentiallyDead */ true);
    (void)Success;
    assert(Success && "Did not expect the call base visit callback to fail!");

    Attributor::SimplifictionCallbackTy SCB =
        [](const IRPosition &, const AbstractAttribute *,
           bool &) -> Optional<Value *> { return nullptr; };
    for (const auto &It : AllocationInfos)
      A.registerSimplificationCallback(IRPosition::callsite_returned(*It.first),
                                       SCB);
    for (const auto &It : DeallocationInfos)
      A.registerSimplificationCallback(IRPosition::callsite_returned(*It.first),
                                       SCB);
  }

  const std::string getAsStr() const override {
    unsigned NumH2SMallocs = 0, NumInvalidMallocs = 0;
    for (const auto &It : AllocationInfos) {
      if (It.second->Status == AllocationInfo::INVALID)
        ++NumInvalidMallocs;
      else
        ++NumH2SMallocs;
    }
    return "[H2S] Mallocs Good/Bad: " + std::to_string(NumH2SMallocs) + "/" +
           std::to_string(NumInvalidMallocs);
  }

  /// See AbstractAttribute::trackStatistics().
  void trackStatistics() const override {
    STATS_DECL(
        MallocCalls, Function,
        "Number of malloc/calloc/aligned_alloc calls converted to allocas");
    for (const auto &It : AllocationInfos)
      if (It.second->Status != AllocationInfo::INVALID)
        ++BUILD_STAT_NAME(MallocCalls, Function);
  }

  bool isAssumedHeapToStack(const CallBase &CB) const override {
    if (isValidState())
      if (AllocationInfo *AI =
              AllocationInfos.lookup(const_cast<CallBase *>(&CB)))
        return AI->Status != AllocationInfo::INVALID;
    return false;
  }

  bool isAssumedHeapToStackRemovedFree(CallBase &CB) const override {
    if (!isValidState())
      return false;

    for (const auto &It : AllocationInfos) {
      AllocationInfo &AI = *It.second;
      if (AI.Status == AllocationInfo::INVALID)
        continue;

      if (AI.PotentialFreeCalls.count(&CB))
        return true;
    }

    return false;
  }

  ChangeStatus manifest(Attributor &A) override {
    assert(getState().isValidState() &&
           "Attempted to manifest an invalid state!");

    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;
    Function *F = getAnchorScope();
    const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);

    for (auto &It : AllocationInfos) {
      AllocationInfo &AI = *It.second;
      if (AI.Status == AllocationInfo::INVALID)
        continue;

      for (CallBase *FreeCall : AI.PotentialFreeCalls) {
        LLVM_DEBUG(dbgs() << "H2S: Removing free call: " << *FreeCall << "\n");
        A.deleteAfterManifest(*FreeCall);
        HasChanged = ChangeStatus::CHANGED;
      }

      LLVM_DEBUG(dbgs() << "H2S: Removing malloc-like call: " << *AI.CB
                        << "\n");

      auto Remark = [&](OptimizationRemark OR) {
        LibFunc IsAllocShared;
        if (TLI->getLibFunc(*AI.CB, IsAllocShared))
          if (IsAllocShared == LibFunc___kmpc_alloc_shared)
            return OR << "Moving globalized variable to the stack.";
        return OR << "Moving memory allocation from the heap to the stack.";
      };
      if (AI.LibraryFunctionId == LibFunc___kmpc_alloc_shared)
        A.emitRemark<OptimizationRemark>(AI.CB, "OMP110", Remark);
      else
        A.emitRemark<OptimizationRemark>(AI.CB, "HeapToStack", Remark);

      const DataLayout &DL = A.getInfoCache().getDL();
      Value *Size;
      Optional<APInt> SizeAPI = getSize(A, *this, AI);
      if (SizeAPI) {
        Size = ConstantInt::get(AI.CB->getContext(), *SizeAPI);
      } else {
        LLVMContext &Ctx = AI.CB->getContext();
        ObjectSizeOpts Opts;
        ObjectSizeOffsetEvaluator Eval(DL, TLI, Ctx, Opts);
        SizeOffsetEvalType SizeOffsetPair = Eval.compute(AI.CB);
        assert(SizeOffsetPair != ObjectSizeOffsetEvaluator::unknown() &&
               cast<ConstantInt>(SizeOffsetPair.second)->isZero());
        Size = SizeOffsetPair.first;
      }

      Instruction *IP =
          AI.MoveAllocaIntoEntry ? &F->getEntryBlock().front() : AI.CB;

      Align Alignment(1);
      if (MaybeAlign RetAlign = AI.CB->getRetAlign())
        Alignment = std::max(Alignment, *RetAlign);
      if (Value *Align = getAllocAlignment(AI.CB, TLI)) {
        Optional<APInt> AlignmentAPI = getAPInt(A, *this, *Align);
        assert(AlignmentAPI && AlignmentAPI.value().getZExtValue() > 0 &&
               "Expected an alignment during manifest!");
        Alignment = std::max(
            Alignment, assumeAligned(AlignmentAPI.value().getZExtValue()));
      }

      // TODO: Hoist the alloca towards the function entry.
      unsigned AS = DL.getAllocaAddrSpace();
      Instruction *Alloca =
          new AllocaInst(Type::getInt8Ty(F->getContext()), AS, Size, Alignment,
                         AI.CB->getName() + ".h2s", IP);

      if (Alloca->getType() != AI.CB->getType())
        Alloca = BitCastInst::CreatePointerBitCastOrAddrSpaceCast(
            Alloca, AI.CB->getType(), "malloc_cast", AI.CB);

      auto *I8Ty = Type::getInt8Ty(F->getContext());
      auto *InitVal = getInitialValueOfAllocation(AI.CB, TLI, I8Ty);
      assert(InitVal &&
             "Must be able to materialize initial memory state of allocation");

      A.changeAfterManifest(IRPosition::inst(*AI.CB), *Alloca);

      if (auto *II = dyn_cast<InvokeInst>(AI.CB)) {
        auto *NBB = II->getNormalDest();
        BranchInst::Create(NBB, AI.CB->getParent());
        A.deleteAfterManifest(*AI.CB);
      } else {
        A.deleteAfterManifest(*AI.CB);
      }

      // Initialize the alloca with the same value as used by the allocation
      // function.  We can skip undef as the initial value of an alloc is
      // undef, and the memset would simply end up being DSEd.
      if (!isa<UndefValue>(InitVal)) {
        IRBuilder<> Builder(Alloca->getNextNode());
        // TODO: Use alignment above if align!=1
        Builder.CreateMemSet(Alloca, InitVal, Size, None);
      }
      HasChanged = ChangeStatus::CHANGED;
    }

    return HasChanged;
  }

  Optional<APInt> getAPInt(Attributor &A, const AbstractAttribute &AA,
                           Value &V) {
    bool UsedAssumedInformation = false;
    Optional<Constant *> SimpleV =
        A.getAssumedConstant(V, AA, UsedAssumedInformation);
    if (!SimpleV)
      return APInt(64, 0);
    if (auto *CI = dyn_cast_or_null<ConstantInt>(SimpleV.value()))
      return CI->getValue();
    return llvm::None;
  }

  Optional<APInt> getSize(Attributor &A, const AbstractAttribute &AA,
                          AllocationInfo &AI) {
    auto Mapper = [&](const Value *V) -> const Value * {
      bool UsedAssumedInformation = false;
      if (Optional<Constant *> SimpleV =
              A.getAssumedConstant(*V, AA, UsedAssumedInformation))
        if (*SimpleV)
          return *SimpleV;
      return V;
    };

    const Function *F = getAnchorScope();
    const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);
    return getAllocSize(AI.CB, TLI, Mapper);
  }

  /// Collection of all malloc-like calls in a function with associated
  /// information.
  MapVector<CallBase *, AllocationInfo *> AllocationInfos;

  /// Collection of all free-like calls in a function with associated
  /// information.
  MapVector<CallBase *, DeallocationInfo *> DeallocationInfos;

  ChangeStatus updateImpl(Attributor &A) override;
};

ChangeStatus AAHeapToStackFunction::updateImpl(Attributor &A) {
  ChangeStatus Changed = ChangeStatus::UNCHANGED;
  const Function *F = getAnchorScope();
  const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);

  const auto &LivenessAA =
      A.getAAFor<AAIsDead>(*this, IRPosition::function(*F), DepClassTy::NONE);

  MustBeExecutedContextExplorer &Explorer =
      A.getInfoCache().getMustBeExecutedContextExplorer();

  bool StackIsAccessibleByOtherThreads =
      A.getInfoCache().stackIsAccessibleByOtherThreads();

  LoopInfo *LI =
      A.getInfoCache().getAnalysisResultForFunction<LoopAnalysis>(*F);
  Optional<bool> MayContainIrreducibleControl;
  auto IsInLoop = [&](BasicBlock &BB) {
    if (&F->getEntryBlock() == &BB)
      return false;
    if (!MayContainIrreducibleControl.has_value())
      MayContainIrreducibleControl = mayContainIrreducibleControl(*F, LI);
    if (MayContainIrreducibleControl.value())
      return true;
    if (!LI)
      return true;
    return LI->getLoopFor(&BB) != nullptr;
  };

  // Flag to ensure we update our deallocation information at most once per
  // updateImpl call and only if we use the free check reasoning.
  bool HasUpdatedFrees = false;

  auto UpdateFrees = [&]() {
    HasUpdatedFrees = true;

    for (auto &It : DeallocationInfos) {
      DeallocationInfo &DI = *It.second;
      // For now we cannot use deallocations that have unknown inputs, skip
      // them.
      if (DI.MightFreeUnknownObjects)
        continue;

      // No need to analyze dead calls, ignore them instead.
      bool UsedAssumedInformation = false;
      if (A.isAssumedDead(*DI.CB, this, &LivenessAA, UsedAssumedInformation,
                          /* CheckBBLivenessOnly */ true))
        continue;

      // Use the non-optimistic version to get the freed object.
      Value *Obj = getUnderlyingObject(DI.FreedOp);
      if (!Obj) {
        LLVM_DEBUG(dbgs() << "[H2S] Unknown underlying object for free!\n");
        DI.MightFreeUnknownObjects = true;
        continue;
      }

      // Free of null and undef can be ignored as no-ops (or UB in the latter
      // case).
      if (isa<ConstantPointerNull>(Obj) || isa<UndefValue>(Obj))
        continue;

      CallBase *ObjCB = dyn_cast<CallBase>(Obj);
      if (!ObjCB) {
        LLVM_DEBUG(dbgs() << "[H2S] Free of a non-call object: " << *Obj
                          << "\n");
        DI.MightFreeUnknownObjects = true;
        continue;
      }

      AllocationInfo *AI = AllocationInfos.lookup(ObjCB);
      if (!AI) {
        LLVM_DEBUG(dbgs() << "[H2S] Free of a non-allocation object: " << *Obj
                          << "\n");
        DI.MightFreeUnknownObjects = true;
        continue;
      }

      DI.PotentialAllocationCalls.insert(ObjCB);
    }
  };

  auto FreeCheck = [&](AllocationInfo &AI) {
    // If the stack is not accessible by other threads, the "must-free" logic
    // doesn't apply as the pointer could be shared and needs to be places in
    // "shareable" memory.
    if (!StackIsAccessibleByOtherThreads) {
      auto &NoSyncAA =
          A.getAAFor<AANoSync>(*this, getIRPosition(), DepClassTy::OPTIONAL);
      if (!NoSyncAA.isAssumedNoSync()) {
        LLVM_DEBUG(
            dbgs() << "[H2S] found an escaping use, stack is not accessible by "
                      "other threads and function is not nosync:\n");
        return false;
      }
    }
    if (!HasUpdatedFrees)
      UpdateFrees();

    // TODO: Allow multi exit functions that have different free calls.
    if (AI.PotentialFreeCalls.size() != 1) {
      LLVM_DEBUG(dbgs() << "[H2S] did not find one free call but "
                        << AI.PotentialFreeCalls.size() << "\n");
      return false;
    }
    CallBase *UniqueFree = *AI.PotentialFreeCalls.begin();
    DeallocationInfo *DI = DeallocationInfos.lookup(UniqueFree);
    if (!DI) {
      LLVM_DEBUG(
          dbgs() << "[H2S] unique free call was not known as deallocation call "
                 << *UniqueFree << "\n");
      return false;
    }
    if (DI->MightFreeUnknownObjects) {
      LLVM_DEBUG(
          dbgs() << "[H2S] unique free call might free unknown allocations\n");
      return false;
    }
    if (DI->PotentialAllocationCalls.empty())
      return true;
    if (DI->PotentialAllocationCalls.size() > 1) {
      LLVM_DEBUG(dbgs() << "[H2S] unique free call might free "
                        << DI->PotentialAllocationCalls.size()
                        << " different allocations\n");
      return false;
    }
    if (*DI->PotentialAllocationCalls.begin() != AI.CB) {
      LLVM_DEBUG(
          dbgs()
          << "[H2S] unique free call not known to free this allocation but "
          << **DI->PotentialAllocationCalls.begin() << "\n");
      return false;
    }
    Instruction *CtxI = isa<InvokeInst>(AI.CB) ? AI.CB : AI.CB->getNextNode();
    if (!Explorer.findInContextOf(UniqueFree, CtxI)) {
      LLVM_DEBUG(
          dbgs()
          << "[H2S] unique free call might not be executed with the allocation "
          << *UniqueFree << "\n");
      return false;
    }
    return true;
  };

  auto UsesCheck = [&](AllocationInfo &AI) {
    bool ValidUsesOnly = true;

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
        if (DeallocationInfos.count(CB)) {
          AI.PotentialFreeCalls.insert(CB);
          return true;
        }

        unsigned ArgNo = CB->getArgOperandNo(&U);

        const auto &NoCaptureAA = A.getAAFor<AANoCapture>(
            *this, IRPosition::callsite_argument(*CB, ArgNo),
            DepClassTy::OPTIONAL);

        // If a call site argument use is nofree, we are fine.
        const auto &ArgNoFreeAA = A.getAAFor<AANoFree>(
            *this, IRPosition::callsite_argument(*CB, ArgNo),
            DepClassTy::OPTIONAL);

        bool MaybeCaptured = !NoCaptureAA.isAssumedNoCapture();
        bool MaybeFreed = !ArgNoFreeAA.isAssumedNoFree();
        if (MaybeCaptured ||
            (AI.LibraryFunctionId != LibFunc___kmpc_alloc_shared &&
             MaybeFreed)) {
          AI.HasPotentiallyFreeingUnknownUses |= MaybeFreed;

          // Emit a missed remark if this is missed OpenMP globalization.
          auto Remark = [&](OptimizationRemarkMissed ORM) {
            return ORM
                   << "Could not move globalized variable to the stack. "
                      "Variable is potentially captured in call. Mark "
                      "parameter as `__attribute__((noescape))` to override.";
          };

          if (ValidUsesOnly &&
              AI.LibraryFunctionId == LibFunc___kmpc_alloc_shared)
            A.emitRemark<OptimizationRemarkMissed>(CB, "OMP113", Remark);

          LLVM_DEBUG(dbgs() << "[H2S] Bad user: " << *UserI << "\n");
          ValidUsesOnly = false;
        }
        return true;
      }

      if (isa<GetElementPtrInst>(UserI) || isa<BitCastInst>(UserI) ||
          isa<PHINode>(UserI) || isa<SelectInst>(UserI)) {
        Follow = true;
        return true;
      }
      // Unknown user for which we can not track uses further (in a way that
      // makes sense).
      LLVM_DEBUG(dbgs() << "[H2S] Unknown user: " << *UserI << "\n");
      ValidUsesOnly = false;
      return true;
    };
    if (!A.checkForAllUses(Pred, *this, *AI.CB))
      return false;
    return ValidUsesOnly;
  };

  // The actual update starts here. We look at all allocations and depending on
  // their status perform the appropriate check(s).
  for (auto &It : AllocationInfos) {
    AllocationInfo &AI = *It.second;
    if (AI.Status == AllocationInfo::INVALID)
      continue;

    if (Value *Align = getAllocAlignment(AI.CB, TLI)) {
      Optional<APInt> APAlign = getAPInt(A, *this, *Align);
      if (!APAlign) {
        // Can't generate an alloca which respects the required alignment
        // on the allocation.
        LLVM_DEBUG(dbgs() << "[H2S] Unknown allocation alignment: " << *AI.CB
                          << "\n");
        AI.Status = AllocationInfo::INVALID;
        Changed = ChangeStatus::CHANGED;
        continue;
      }
      if (APAlign->ugt(llvm::Value::MaximumAlignment) ||
          !APAlign->isPowerOf2()) {
        LLVM_DEBUG(dbgs() << "[H2S] Invalid allocation alignment: " << APAlign
                          << "\n");
        AI.Status = AllocationInfo::INVALID;
        Changed = ChangeStatus::CHANGED;
        continue;
      }
    }

    Optional<APInt> Size = getSize(A, *this, AI);
    if (MaxHeapToStackSize != -1) {
      if (!Size || Size.value().ugt(MaxHeapToStackSize)) {
        LLVM_DEBUG({
          if (!Size)
            dbgs() << "[H2S] Unknown allocation size: " << *AI.CB << "\n";
          else
            dbgs() << "[H2S] Allocation size too large: " << *AI.CB << " vs. "
                   << MaxHeapToStackSize << "\n";
        });

        AI.Status = AllocationInfo::INVALID;
        Changed = ChangeStatus::CHANGED;
        continue;
      }
    }

    switch (AI.Status) {
    case AllocationInfo::STACK_DUE_TO_USE:
      if (UsesCheck(AI))
        break;
      AI.Status = AllocationInfo::STACK_DUE_TO_FREE;
      [[fallthrough]];
    case AllocationInfo::STACK_DUE_TO_FREE:
      if (FreeCheck(AI))
        break;
      AI.Status = AllocationInfo::INVALID;
      Changed = ChangeStatus::CHANGED;
      break;
    case AllocationInfo::INVALID:
      llvm_unreachable("Invalid allocations should never reach this point!");
    };

    // Check if we still think we can move it into the entry block. If the
    // alloca comes from a converted __kmpc_alloc_shared then we can usually
    // ignore the potential compilations associated with loops.
    bool IsGlobalizedLocal =
        AI.LibraryFunctionId == LibFunc___kmpc_alloc_shared;
    if (AI.MoveAllocaIntoEntry &&
        (!Size.has_value() ||
         (!IsGlobalizedLocal && IsInLoop(*AI.CB->getParent()))))
      AI.MoveAllocaIntoEntry = false;
  }

  return Changed;
}
} // namespace

/// ----------------------- Privatizable Pointers ------------------------------
namespace {
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
    if (!T0)
      return T1;
    if (!T1)
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
    bool UsedAssumedInformation = false;
    SmallVector<Attribute, 1> Attrs;
    getAttrs({Attribute::ByVal}, Attrs, /* IgnoreSubsumingPositions */ true);
    if (!Attrs.empty() &&
        A.checkForAllCallSites([](AbstractCallSite ACS) { return true; }, *this,
                               true, UsedAssumedInformation))
      return Attrs[0].getValueAsType();

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
        if (CSTy && CSTy.value())
          CSTy.value()->print(dbgs());
        else if (CSTy)
          dbgs() << "<nullptr>";
        else
          dbgs() << "<none>";
      });

      Ty = combineTypes(Ty, CSTy);

      LLVM_DEBUG({
        dbgs() << " : New Type: ";
        if (Ty && Ty.value())
          Ty.value()->print(dbgs());
        else if (Ty)
          dbgs() << "<nullptr>";
        else
          dbgs() << "<none>";
        dbgs() << "\n";
      });

      return !Ty || Ty.value();
    };

    if (!A.checkForAllCallSites(CallSiteCheck, *this, true,
                                UsedAssumedInformation))
      return nullptr;
    return Ty;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    PrivatizableType = identifyPrivatizableType(A);
    if (!PrivatizableType)
      return ChangeStatus::UNCHANGED;
    if (!PrivatizableType.value())
      return indicatePessimisticFixpoint();

    // The dependence is optional so we don't give up once we give up on the
    // alignment.
    A.getAAFor<AAAlign>(*this, IRPosition::value(getAssociatedValue()),
                        DepClassTy::OPTIONAL);

    // Avoid arguments with padding for now.
    if (!getIRPosition().hasAttr(Attribute::ByVal) &&
        !isDenselyPacked(*PrivatizableType, A.getInfoCache().getDL())) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] Padding detected\n");
      return indicatePessimisticFixpoint();
    }

    // Collect the types that will replace the privatizable type in the function
    // signature.
    SmallVector<Type *, 16> ReplacementTypes;
    identifyReplacementTypes(*PrivatizableType, ReplacementTypes);

    // Verify callee and caller agree on how the promoted argument would be
    // passed.
    Function &Fn = *getIRPosition().getAnchorScope();
    const auto *TTI =
        A.getInfoCache().getAnalysisResultForFunction<TargetIRAnalysis>(Fn);
    if (!TTI) {
      LLVM_DEBUG(dbgs() << "[AAPrivatizablePtr] Missing TTI for function "
                        << Fn.getName() << "\n");
      return indicatePessimisticFixpoint();
    }

    auto CallSiteCheck = [&](AbstractCallSite ACS) {
      CallBase *CB = ACS.getInstruction();
      return TTI->areTypesABICompatible(
          CB->getCaller(), CB->getCalledFunction(), ReplacementTypes);
    };
    bool UsedAssumedInformation = false;
    if (!A.checkForAllCallSites(CallSiteCheck, *this, true,
                                UsedAssumedInformation)) {
      LLVM_DEBUG(
          dbgs() << "[AAPrivatizablePtr] ABI incompatibility detected for "
                 << Fn.getName() << "\n");
      return indicatePessimisticFixpoint();
    }

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
            if (!CBArgPrivTy)
              continue;
            if (CBArgPrivTy.value() == PrivatizableType)
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
      assert(DCArgNo >= 0 && unsigned(DCArgNo) < DC->arg_size() &&
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
          if (!DCArgPrivTy)
            return true;
          if (DCArgPrivTy.value() == PrivatizableType)
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

    if (!A.checkForAllCallSites(IsCompatiblePrivArgOfOtherCallSite, *this, true,
                                UsedAssumedInformation))
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
        Value *Ptr =
            constructPointer(PointeeTy, PrivType, &Base,
                             PrivStructLayout->getElementOffset(u), IRB, DL);
        new StoreInst(F.getArg(ArgNo + u), Ptr, &IP);
      }
    } else if (auto *PrivArrayType = dyn_cast<ArrayType>(PrivType)) {
      Type *PointeeTy = PrivArrayType->getElementType();
      Type *PointeePtrTy = PointeeTy->getPointerTo();
      uint64_t PointeeTySize = DL.getTypeStoreSize(PointeeTy);
      for (unsigned u = 0, e = PrivArrayType->getNumElements(); u < e; u++) {
        Value *Ptr = constructPointer(PointeePtrTy, PrivType, &Base,
                                      u * PointeeTySize, IRB, DL);
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

    Type *PrivPtrType = PrivType->getPointerTo();
    if (Base->getType() != PrivPtrType)
      Base = BitCastInst::CreatePointerBitCastOrAddrSpaceCast(
          Base, PrivPtrType, "", ACS.getInstruction());

    // Traverse the type, build GEPs and loads.
    if (auto *PrivStructType = dyn_cast<StructType>(PrivType)) {
      const StructLayout *PrivStructLayout = DL.getStructLayout(PrivStructType);
      for (unsigned u = 0, e = PrivStructType->getNumElements(); u < e; u++) {
        Type *PointeeTy = PrivStructType->getElementType(u);
        Value *Ptr =
            constructPointer(PointeeTy->getPointerTo(), PrivType, Base,
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
        Value *Ptr = constructPointer(PointeePtrTy, PrivType, Base,
                                      u * PointeeTySize, IRB, DL);
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
    if (!PrivatizableType)
      return ChangeStatus::UNCHANGED;
    assert(PrivatizableType.value() && "Expected privatizable type!");

    // Collect all tail calls in the function as we cannot allow new allocas to
    // escape into tail recursion.
    // TODO: Be smarter about new allocas escaping into tail calls.
    SmallVector<CallInst *, 16> TailCalls;
    bool UsedAssumedInformation = false;
    if (!A.checkForAllInstructions(
            [&](Instruction &I) {
              CallInst &CI = cast<CallInst>(I);
              if (CI.isTailCall())
                TailCalls.push_back(&CI);
              return true;
            },
            *this, {Instruction::Call}, UsedAssumedInformation))
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
          const DataLayout &DL = IP->getModule()->getDataLayout();
          unsigned AS = DL.getAllocaAddrSpace();
          Instruction *AI = new AllocaInst(PrivatizableType.value(), AS,
                                           Arg->getName() + ".priv", IP);
          createInitialization(PrivatizableType.value(), *AI, ReplacementFn,
                               ArgIt->getArgNo(), *IP);

          if (AI->getType() != Arg->getType())
            AI = BitCastInst::CreatePointerBitCastOrAddrSpaceCast(
                AI, Arg->getType(), "", IP);
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
              AlignAA.getAssumedAlign(), *PrivatizableType, ACS,
              ACS.getCallArgOperand(ARI.getReplacedArg().getArgNo()),
              NewArgOperands);
        };

    // Collect the types that will replace the privatizable type in the function
    // signature.
    SmallVector<Type *, 16> ReplacementTypes;
    identifyReplacementTypes(*PrivatizableType, ReplacementTypes);

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
  void initialize(Attributor &A) override {
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
          return AI->getAllocatedType();
    if (auto *Arg = dyn_cast<Argument>(Obj)) {
      auto &PrivArgAA = A.getAAFor<AAPrivatizablePtr>(
          *this, IRPosition::argument(*Arg), DepClassTy::REQUIRED);
      if (PrivArgAA.isAssumedPrivatizablePtr())
        return PrivArgAA.getPrivatizableType();
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
    if (!PrivatizableType)
      return ChangeStatus::UNCHANGED;
    if (!PrivatizableType.value())
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

    bool IsKnown;
    if (!AA::isAssumedReadOnly(A, IRP, *this, IsKnown)) {
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
} // namespace

/// -------------------- Memory Behavior Attributes ----------------------------
/// Includes read-none, read-only, and write-only.
/// ----------------------------------------------------------------------------
namespace {
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
  bool followUsersOfUseIn(Attributor &A, const Use &U,
                          const Instruction *UserI);

  /// Update the state according to the effect of use \p U in \p UserI.
  void analyzeUseIn(Attributor &A, const Use &U, const Instruction *UserI);
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
    if (!Arg || !A.isFunctionIPOAmendable(*(Arg->getParent())))
      indicatePessimisticFixpoint();
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
  ChangeStatus updateImpl(Attributor &A) override;

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // TODO: It would be better to merge this with AAMemoryLocation, so that
    // we could determine read/write per location. This would also have the
    // benefit of only one place trying to manifest the memory attribute.
    Function &F = cast<Function>(getAnchorValue());
    MemoryEffects ME = MemoryEffects::unknown();
    if (isAssumedReadNone())
      ME = MemoryEffects::none();
    else if (isAssumedReadOnly())
      ME = MemoryEffects::readOnly();
    else if (isAssumedWriteOnly())
      ME = MemoryEffects::writeOnly();

    // Intersect with existing memory attribute, as we currently deduce the
    // location and modref portion separately.
    MemoryEffects ExistingME = F.getMemoryEffects();
    ME &= ExistingME;
    if (ME == ExistingME)
      return ChangeStatus::UNCHANGED;

    return IRAttributeManifest::manifestAttrs(
        A, getIRPosition(), Attribute::getWithMemoryEffects(F.getContext(), ME),
        /*ForceReplace*/ true);
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

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // TODO: Deduplicate this with AAMemoryBehaviorFunction.
    CallBase &CB = cast<CallBase>(getAnchorValue());
    MemoryEffects ME = MemoryEffects::unknown();
    if (isAssumedReadNone())
      ME = MemoryEffects::none();
    else if (isAssumedReadOnly())
      ME = MemoryEffects::readOnly();
    else if (isAssumedWriteOnly())
      ME = MemoryEffects::writeOnly();

    // Intersect with existing memory attribute, as we currently deduce the
    // location and modref portion separately.
    MemoryEffects ExistingME = CB.getMemoryEffects();
    ME &= ExistingME;
    if (ME == ExistingME)
      return ChangeStatus::UNCHANGED;

    return IRAttributeManifest::manifestAttrs(
        A, getIRPosition(),
        Attribute::getWithMemoryEffects(CB.getContext(), ME),
        /*ForceReplace*/ true);
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

  bool UsedAssumedInformation = false;
  if (!A.checkForAllReadWriteInstructions(CheckRWInst, *this,
                                          UsedAssumedInformation))
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

  // The current assumed state used to determine a change.
  auto AssumedState = S.getAssumed();

  // Make sure the value is not captured (except through "return"), if
  // it is, any information derived would be irrelevant anyway as we cannot
  // check the potential aliases introduced by the capture. However, no need
  // to fall back to anythign less optimistic than the function state.
  const auto &ArgNoCaptureAA =
      A.getAAFor<AANoCapture>(*this, IRP, DepClassTy::OPTIONAL);
  if (!ArgNoCaptureAA.isAssumedNoCaptureMaybeReturned()) {
    S.intersectAssumedBits(FnMemAssumedState);
    return (AssumedState != getAssumed()) ? ChangeStatus::CHANGED
                                          : ChangeStatus::UNCHANGED;
  }

  // Visit and expand uses until all are analyzed or a fixpoint is reached.
  auto UsePred = [&](const Use &U, bool &Follow) -> bool {
    Instruction *UserI = cast<Instruction>(U.getUser());
    LLVM_DEBUG(dbgs() << "[AAMemoryBehavior] Use: " << *U << " in " << *UserI
                      << " \n");

    // Droppable users, e.g., llvm::assume does not actually perform any action.
    if (UserI->isDroppable())
      return true;

    // Check if the users of UserI should also be visited.
    Follow = followUsersOfUseIn(A, U, UserI);

    // If UserI might touch memory we analyze the use in detail.
    if (UserI->mayReadOrWriteMemory())
      analyzeUseIn(A, U, UserI);

    return !isAtFixpoint();
  };

  if (!A.checkForAllUses(UsePred, *this, getAssociatedValue()))
    return indicatePessimisticFixpoint();

  return (AssumedState != getAssumed()) ? ChangeStatus::CHANGED
                                        : ChangeStatus::UNCHANGED;
}

bool AAMemoryBehaviorFloating::followUsersOfUseIn(Attributor &A, const Use &U,
                                                  const Instruction *UserI) {
  // The loaded value is unrelated to the pointer argument, no need to
  // follow the users of the load.
  if (isa<LoadInst>(UserI) || isa<ReturnInst>(UserI))
    return false;

  // By default we follow all uses assuming UserI might leak information on U,
  // we have special handling for call sites operands though.
  const auto *CB = dyn_cast<CallBase>(UserI);
  if (!CB || !CB->isArgOperand(&U))
    return true;

  // If the use is a call argument known not to be captured, the users of
  // the call do not need to be visited because they have to be unrelated to
  // the input. Note that this check is not trivial even though we disallow
  // general capturing of the underlying argument. The reason is that the
  // call might the argument "through return", which we allow and for which we
  // need to check call users.
  if (U.get()->getType()->isPointerTy()) {
    unsigned ArgNo = CB->getArgOperandNo(&U);
    const auto &ArgNoCaptureAA = A.getAAFor<AANoCapture>(
        *this, IRPosition::callsite_argument(*CB, ArgNo), DepClassTy::OPTIONAL);
    return !ArgNoCaptureAA.isAssumedNoCapture();
  }

  return true;
}

void AAMemoryBehaviorFloating::analyzeUseIn(Attributor &A, const Use &U,
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
    // pointer operand. Note that while capturing was taken care of somewhere
    // else we need to deal with stores of the value that is not looked through.
    if (cast<StoreInst>(UserI)->getPointerOperand() == U.get())
      removeAssumedBits(NO_WRITES);
    else
      indicatePessimisticFixpoint();
    return;

  case Instruction::Call:
  case Instruction::CallBr:
  case Instruction::Invoke: {
    // For call sites we look at the argument memory behavior attribute (this
    // could be recursive!) in order to restrict our own state.
    const auto *CB = cast<CallBase>(UserI);

    // Give up on operand bundles.
    if (CB->isBundleOperand(&U)) {
      indicatePessimisticFixpoint();
      return;
    }

    // Calling a function does read the function pointer, maybe write it if the
    // function is self-modifying.
    if (CB->isCallee(&U)) {
      removeAssumedBits(NO_READS);
      break;
    }

    // Adjust the possible access behavior based on the information on the
    // argument.
    IRPosition Pos;
    if (U.get()->getType()->isPointerTy())
      Pos = IRPosition::callsite_argument(*CB, CB->getArgOperandNo(&U));
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
    AccessKind2Accesses.fill(nullptr);
  }

  ~AAMemoryLocationImpl() {
    // The AccessSets are allocated via a BumpPtrAllocator, we call
    // the destructor manually.
    for (AccessSet *AS : AccessKind2Accesses)
      if (AS)
        AS->~AccessSet();
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
    // TODO: A better way to handle this would be to add ~NO_GLOBAL_MEM /
    // MemoryEffects::Other as a possible location.
    bool UseArgMemOnly = true;
    Function *AnchorFn = IRP.getAnchorScope();
    if (AnchorFn && A.isRunOn(*AnchorFn))
      UseArgMemOnly = !AnchorFn->hasLocalLinkage();

    SmallVector<Attribute, 2> Attrs;
    IRP.getAttrs({Attribute::Memory}, Attrs, IgnoreSubsumingPositions);
    for (const Attribute &Attr : Attrs) {
      // TODO: We can map MemoryEffects to Attributor locations more precisely.
      MemoryEffects ME = Attr.getMemoryEffects();
      if (ME.doesNotAccessMemory()) {
        State.addKnownBits(NO_LOCAL_MEM | NO_CONST_MEM);
        continue;
      }
      if (ME.onlyAccessesInaccessibleMem()) {
        State.addKnownBits(inverseLocation(NO_INACCESSIBLE_MEM, true, true));
        continue;
      }
      if (ME.onlyAccessesArgPointees()) {
        if (UseArgMemOnly)
          State.addKnownBits(inverseLocation(NO_ARGUMENT_MEM, true, true));
        else {
          // Remove location information, only keep read/write info.
          ME = MemoryEffects(ME.getModRef());
          IRAttributeManifest::manifestAttrs(
              A, IRP,
              Attribute::getWithMemoryEffects(IRP.getAnchorValue().getContext(),
                                              ME),
              /*ForceReplace*/ true);
        }
        continue;
      }
      if (ME.onlyAccessesInaccessibleOrArgMem()) {
        if (UseArgMemOnly)
          State.addKnownBits(inverseLocation(
              NO_INACCESSIBLE_MEM | NO_ARGUMENT_MEM, true, true));
        else {
          // Remove location information, only keep read/write info.
          ME = MemoryEffects(ME.getModRef());
          IRAttributeManifest::manifestAttrs(
              A, IRP,
              Attribute::getWithMemoryEffects(IRP.getAnchorValue().getContext(),
                                              ME),
              /*ForceReplace*/ true);
        }
        continue;
      }
    }
  }

  /// See AbstractAttribute::getDeducedAttributes(...).
  void getDeducedAttributes(LLVMContext &Ctx,
                            SmallVectorImpl<Attribute> &Attrs) const override {
    // TODO: We can map Attributor locations to MemoryEffects more precisely.
    assert(Attrs.size() == 0);
    if (getIRPosition().getPositionKind() == IRPosition::IRP_FUNCTION) {
      if (isAssumedReadNone())
        Attrs.push_back(
            Attribute::getWithMemoryEffects(Ctx, MemoryEffects::none()));
      else if (isAssumedInaccessibleMemOnly())
        Attrs.push_back(Attribute::getWithMemoryEffects(
            Ctx, MemoryEffects::inaccessibleMemOnly()));
      else if (isAssumedArgMemOnly())
        Attrs.push_back(
            Attribute::getWithMemoryEffects(Ctx, MemoryEffects::argMemOnly()));
      else if (isAssumedInaccessibleOrArgMemOnly())
        Attrs.push_back(Attribute::getWithMemoryEffects(
            Ctx, MemoryEffects::inaccessibleOrArgMemOnly()));
    }
    assert(Attrs.size() <= 1);
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // TODO: If AAMemoryLocation and AAMemoryBehavior are merged, we could
    // provide per-location modref information here.
    const IRPosition &IRP = getIRPosition();

    SmallVector<Attribute, 1> DeducedAttrs;
    getDeducedAttributes(IRP.getAnchorValue().getContext(), DeducedAttrs);
    if (DeducedAttrs.size() != 1)
      return ChangeStatus::UNCHANGED;
    MemoryEffects ME = DeducedAttrs[0].getMemoryEffects();

    // Intersect with existing memory attribute, as we currently deduce the
    // location and modref portion separately.
    SmallVector<Attribute, 1> ExistingAttrs;
    IRP.getAttrs({Attribute::Memory}, ExistingAttrs,
                 /* IgnoreSubsumingPositions */ true);
    if (ExistingAttrs.size() == 1) {
      MemoryEffects ExistingME = ExistingAttrs[0].getMemoryEffects();
      ME &= ExistingME;
      if (ME == ExistingME)
        return ChangeStatus::UNCHANGED;
    }

    return IRAttributeManifest::manifestAttrs(
        A, IRP,
        Attribute::getWithMemoryEffects(IRP.getAnchorValue().getContext(), ME),
        /*ForceReplace*/ true);
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
  std::array<AccessSet *, llvm::CTLog2<VALID_STATE>()> AccessKind2Accesses;

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
};

void AAMemoryLocationImpl::categorizePtrValue(
    Attributor &A, const Instruction &I, const Value &Ptr,
    AAMemoryLocation::StateType &State, bool &Changed) {
  LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Categorize pointer locations for "
                    << Ptr << " ["
                    << getMemoryLocationsAsStr(State.getAssumed()) << "]\n");

  SmallSetVector<Value *, 8> Objects;
  bool UsedAssumedInformation = false;
  if (!AA::getAssumedUnderlyingObjects(A, Ptr, Objects, *this, &I,
                                       UsedAssumedInformation,
                                       AA::Intraprocedural)) {
    LLVM_DEBUG(
        dbgs() << "[AAMemoryLocation] Pointer locations not categorized\n");
    updateStateAndAccessesMap(State, NO_UNKOWN_MEM, &I, nullptr, Changed,
                              getAccessKindFromInst(&I));
    return;
  }

  for (Value *Obj : Objects) {
    // TODO: recognize the TBAA used for constant accesses.
    MemoryLocationsKind MLK = NO_LOCATIONS;
    if (isa<UndefValue>(Obj))
      continue;
    if (isa<Argument>(Obj)) {
      // TODO: For now we do not treat byval arguments as local copies performed
      // on the call edge, though, we should. To make that happen we need to
      // teach various passes, e.g., DSE, about the copy effect of a byval. That
      // would also allow us to mark functions only accessing byval arguments as
      // readnone again, arguably their accesses have no effect outside of the
      // function, like accesses to allocas.
      MLK = NO_ARGUMENT_MEM;
    } else if (auto *GV = dyn_cast<GlobalValue>(Obj)) {
      // Reading constant memory is not treated as a read "effect" by the
      // function attr pass so we won't neither. Constants defined by TBAA are
      // similar. (We know we do not write it because it is constant.)
      if (auto *GVar = dyn_cast<GlobalVariable>(GV))
        if (GVar->isConstant())
          continue;

      if (GV->hasLocalLinkage())
        MLK = NO_GLOBAL_INTERNAL_MEM;
      else
        MLK = NO_GLOBAL_EXTERNAL_MEM;
    } else if (isa<ConstantPointerNull>(Obj) &&
               !NullPointerIsDefined(getAssociatedFunction(),
                                     Ptr.getType()->getPointerAddressSpace())) {
      continue;
    } else if (isa<AllocaInst>(Obj)) {
      MLK = NO_LOCAL_MEM;
    } else if (const auto *CB = dyn_cast<CallBase>(Obj)) {
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
    LLVM_DEBUG(dbgs() << "[AAMemoryLocation] Ptr value can be categorized: "
                      << *Obj << " -> " << getMemoryLocationsAsStr(MLK)
                      << "\n");
    updateStateAndAccessesMap(getState(), MLK, &I, Obj, Changed,
                              getAccessKindFromInst(&I));
  }

  LLVM_DEBUG(
      dbgs() << "[AAMemoryLocation] Accessed locations with pointer locations: "
             << getMemoryLocationsAsStr(State.getAssumed()) << "\n");
}

void AAMemoryLocationImpl::categorizeArgumentPointerLocations(
    Attributor &A, CallBase &CB, AAMemoryLocation::StateType &AccessedLocs,
    bool &Changed) {
  for (unsigned ArgNo = 0, E = CB.arg_size(); ArgNo < E; ++ArgNo) {

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
  ChangeStatus updateImpl(Attributor &A) override {

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

    bool UsedAssumedInformation = false;
    if (!A.checkForAllReadWriteInstructions(CheckRWInst, *this,
                                            UsedAssumedInformation))
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
} // namespace

/// ------------------ Value Constant Range Attribute -------------------------

namespace {
struct AAValueConstantRangeImpl : AAValueConstantRange {
  using StateType = IntegerRangeState;
  AAValueConstantRangeImpl(const IRPosition &IRP, Attributor &A)
      : AAValueConstantRange(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    if (A.hasSimplificationCallback(getIRPosition())) {
      indicatePessimisticFixpoint();
      return;
    }

    // Intersect a range given by SCEV.
    intersectKnown(getConstantRangeFromSCEV(A, getCtxI()));

    // Intersect a range given by LVI.
    intersectKnown(getConstantRangeFromLVI(A, getCtxI()));
  }

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

  /// Return true if \p CtxI is valid for querying outside analyses.
  /// This basically makes sure we do not ask intra-procedural analysis
  /// about a context in the wrong function or a context that violates
  /// dominance assumptions they might have. The \p AllowAACtxI flag indicates
  /// if the original context of this AA is OK or should be considered invalid.
  bool isValidCtxInstructionForOutsideAnalysis(Attributor &A,
                                               const Instruction *CtxI,
                                               bool AllowAACtxI) const {
    if (!CtxI || (!AllowAACtxI && CtxI == getCtxI()))
      return false;

    // Our context might be in a different function, neither intra-procedural
    // analysis (ScalarEvolution nor LazyValueInfo) can handle that.
    if (!AA::isValidInScope(getAssociatedValue(), CtxI->getFunction()))
      return false;

    // If the context is not dominated by the value there are paths to the
    // context that do not define the value. This cannot be handled by
    // LazyValueInfo so we need to bail.
    if (auto *I = dyn_cast<Instruction>(&getAssociatedValue())) {
      InformationCache &InfoCache = A.getInfoCache();
      const DominatorTree *DT =
          InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(
              *I->getFunction());
      return DT && DT->dominates(I, CtxI);
    }

    return true;
  }

  /// See AAValueConstantRange::getKnownConstantRange(..).
  ConstantRange
  getKnownConstantRange(Attributor &A,
                        const Instruction *CtxI = nullptr) const override {
    if (!isValidCtxInstructionForOutsideAnalysis(A, CtxI,
                                                 /* AllowAACtxI */ false))
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
    if (!isValidCtxInstructionForOutsideAnalysis(A, CtxI,
                                                 /* AllowAACtxI */ false))
      return getAssumed();

    ConstantRange LVIR = getConstantRangeFromLVI(A, CtxI);
    ConstantRange SCEVR = getConstantRangeFromSCEV(A, CtxI);
    return getAssumed().intersectWith(SCEVR).intersectWith(LVIR);
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
    if (isAtFixpoint())
      return;

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

    // Simplify the operands first.
    bool UsedAssumedInformation = false;
    const auto &SimplifiedLHS = A.getAssumedSimplified(
        IRPosition::value(*LHS, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Interprocedural);
    if (!SimplifiedLHS.has_value())
      return true;
    if (!SimplifiedLHS.value())
      return false;
    LHS = *SimplifiedLHS;

    const auto &SimplifiedRHS = A.getAssumedSimplified(
        IRPosition::value(*RHS, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Interprocedural);
    if (!SimplifiedRHS.has_value())
      return true;
    if (!SimplifiedRHS.value())
      return false;
    RHS = *SimplifiedRHS;

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
    Value *OpV = CastI->getOperand(0);

    // Simplify the operand first.
    bool UsedAssumedInformation = false;
    const auto &SimplifiedOpV = A.getAssumedSimplified(
        IRPosition::value(*OpV, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Interprocedural);
    if (!SimplifiedOpV.has_value())
      return true;
    if (!SimplifiedOpV.value())
      return false;
    OpV = *SimplifiedOpV;

    if (!OpV->getType()->isIntegerTy())
      return false;

    auto &OpAA = A.getAAFor<AAValueConstantRange>(
        *this, IRPosition::value(*OpV, getCallBaseContext()),
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

    // Simplify the operands first.
    bool UsedAssumedInformation = false;
    const auto &SimplifiedLHS = A.getAssumedSimplified(
        IRPosition::value(*LHS, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Interprocedural);
    if (!SimplifiedLHS.has_value())
      return true;
    if (!SimplifiedLHS.value())
      return false;
    LHS = *SimplifiedLHS;

    const auto &SimplifiedRHS = A.getAssumedSimplified(
        IRPosition::value(*RHS, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Interprocedural);
    if (!SimplifiedRHS.has_value())
      return true;
    if (!SimplifiedRHS.value())
      return false;
    RHS = *SimplifiedRHS;

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
    QuerriedAAs.push_back(&RHSAA);
    auto LHSAARange = LHSAA.getAssumedConstantRange(A, CtxI);
    auto RHSAARange = RHSAA.getAssumedConstantRange(A, CtxI);

    // If one of them is empty set, we can't decide.
    if (LHSAARange.isEmptySet() || RHSAARange.isEmptySet())
      return true;

    bool MustTrue = false, MustFalse = false;

    auto AllowedRegion =
        ConstantRange::makeAllowedICmpRegion(CmpI->getPredicate(), RHSAARange);

    if (AllowedRegion.intersectWith(LHSAARange).isEmptySet())
      MustFalse = true;

    if (LHSAARange.icmp(CmpI->getPredicate(), RHSAARange))
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

    IntegerRangeState T(getBitWidth());
    auto VisitValueCB = [&](Value &V, const Instruction *CtxI) -> bool {
      Instruction *I = dyn_cast<Instruction>(&V);
      if (!I || isa<CallBase>(I)) {

        // Simplify the operand first.
        bool UsedAssumedInformation = false;
        const auto &SimplifiedOpV = A.getAssumedSimplified(
            IRPosition::value(V, getCallBaseContext()), *this,
            UsedAssumedInformation, AA::Interprocedural);
        if (!SimplifiedOpV.has_value())
          return true;
        if (!SimplifiedOpV.value())
          return false;
        Value *VPtr = *SimplifiedOpV;

        // If the value is not instruction, we query AA to Attributor.
        const auto &AA = A.getAAFor<AAValueConstantRange>(
            *this, IRPosition::value(*VPtr, getCallBaseContext()),
            DepClassTy::REQUIRED);

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

    if (!VisitValueCB(getAssociatedValue(), getCtxI()))
      return indicatePessimisticFixpoint();

    // Ensure that long def-use chains can't cause circular reasoning either by
    // introducing a cutoff below.
    if (clampStateAndIndicateChange(getState(), T) == ChangeStatus::UNCHANGED)
      return ChangeStatus::UNCHANGED;
    if (++NumChanges > MaxNumChanges) {
      LLVM_DEBUG(dbgs() << "[AAValueConstantRange] performed " << NumChanges
                        << " but only " << MaxNumChanges
                        << " are allowed to avoid cyclic reasoning.");
      return indicatePessimisticFixpoint();
    }
    return ChangeStatus::CHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(value_range)
  }

  /// Tracker to bail after too many widening steps of the constant range.
  int NumChanges = 0;

  /// Upper bound for the number of allowed changes (=widening steps) for the
  /// constant range before we give up.
  static constexpr int MaxNumChanges = 5;
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
} // namespace

/// ------------------ Potential Values Attribute -------------------------

namespace {
struct AAPotentialConstantValuesImpl : AAPotentialConstantValues {
  using StateType = PotentialConstantIntValuesState;

  AAPotentialConstantValuesImpl(const IRPosition &IRP, Attributor &A)
      : AAPotentialConstantValues(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    if (A.hasSimplificationCallback(getIRPosition()))
      indicatePessimisticFixpoint();
    else
      AAPotentialConstantValues::initialize(A);
  }

  bool fillSetWithConstantValues(Attributor &A, const IRPosition &IRP, SetTy &S,
                                 bool &ContainsUndef) {
    SmallVector<AA::ValueAndContext> Values;
    bool UsedAssumedInformation = false;
    if (!A.getAssumedSimplifiedValues(IRP, *this, Values, AA::Interprocedural,
                                      UsedAssumedInformation)) {
      if (!IRP.getAssociatedType()->isIntegerTy())
        return false;
      auto &PotentialValuesAA = A.getAAFor<AAPotentialConstantValues>(
          *this, IRP, DepClassTy::REQUIRED);
      if (!PotentialValuesAA.getState().isValidState())
        return false;
      ContainsUndef = PotentialValuesAA.getState().undefIsContained();
      S = PotentialValuesAA.getState().getAssumedSet();
      return true;
    }

    for (auto &It : Values) {
      if (isa<UndefValue>(It.getValue()))
        continue;
      auto *CI = dyn_cast<ConstantInt>(It.getValue());
      if (!CI)
        return false;
      S.insert(CI->getValue());
    }
    ContainsUndef = S.empty();

    return true;
  }

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

struct AAPotentialConstantValuesArgument final
    : AAArgumentFromCallSiteArguments<AAPotentialConstantValues,
                                      AAPotentialConstantValuesImpl,
                                      PotentialConstantIntValuesState> {
  using Base = AAArgumentFromCallSiteArguments<AAPotentialConstantValues,
                                               AAPotentialConstantValuesImpl,
                                               PotentialConstantIntValuesState>;
  AAPotentialConstantValuesArgument(const IRPosition &IRP, Attributor &A)
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

struct AAPotentialConstantValuesReturned
    : AAReturnedFromReturnedValues<AAPotentialConstantValues,
                                   AAPotentialConstantValuesImpl> {
  using Base = AAReturnedFromReturnedValues<AAPotentialConstantValues,
                                            AAPotentialConstantValuesImpl>;
  AAPotentialConstantValuesReturned(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(potential_values)
  }
};

struct AAPotentialConstantValuesFloating : AAPotentialConstantValuesImpl {
  AAPotentialConstantValuesFloating(const IRPosition &IRP, Attributor &A)
      : AAPotentialConstantValuesImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    AAPotentialConstantValuesImpl::initialize(A);
    if (isAtFixpoint())
      return;

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

    if (isa<SelectInst>(V) || isa<PHINode>(V) || isa<LoadInst>(V))
      return;

    indicatePessimisticFixpoint();

    LLVM_DEBUG(dbgs() << "[AAPotentialConstantValues] We give up: "
                      << getAssociatedValue() << "\n");
  }

  static bool calculateICmpInst(const ICmpInst *ICI, const APInt &LHS,
                                const APInt &RHS) {
    return ICmpInst::compare(LHS, RHS, ICI->getPredicate());
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
      if (RHS.isZero()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.udiv(RHS);
    case Instruction::SDiv:
      if (RHS.isZero()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.sdiv(RHS);
    case Instruction::URem:
      if (RHS.isZero()) {
        SkipOperation = true;
        return LHS;
      }
      return LHS.urem(RHS);
    case Instruction::SRem:
      if (RHS.isZero()) {
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

    bool LHSContainsUndef = false, RHSContainsUndef = false;
    SetTy LHSAAPVS, RHSAAPVS;
    if (!fillSetWithConstantValues(A, IRPosition::value(*LHS), LHSAAPVS,
                                   LHSContainsUndef) ||
        !fillSetWithConstantValues(A, IRPosition::value(*RHS), RHSAAPVS,
                                   RHSContainsUndef))
      return indicatePessimisticFixpoint();

    // TODO: make use of undef flag to limit potential values aggressively.
    bool MaybeTrue = false, MaybeFalse = false;
    const APInt Zero(RHS->getType()->getIntegerBitWidth(), 0);
    if (LHSContainsUndef && RHSContainsUndef) {
      // The result of any comparison between undefs can be soundly replaced
      // with undef.
      unionAssumedWithUndef();
    } else if (LHSContainsUndef) {
      for (const APInt &R : RHSAAPVS) {
        bool CmpResult = calculateICmpInst(ICI, Zero, R);
        MaybeTrue |= CmpResult;
        MaybeFalse |= !CmpResult;
        if (MaybeTrue & MaybeFalse)
          return indicatePessimisticFixpoint();
      }
    } else if (RHSContainsUndef) {
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

    bool UsedAssumedInformation = false;
    Optional<Constant *> C = A.getAssumedConstant(*SI->getCondition(), *this,
                                                  UsedAssumedInformation);

    // Check if we only need one operand.
    bool OnlyLeft = false, OnlyRight = false;
    if (C && *C && (*C)->isOneValue())
      OnlyLeft = true;
    else if (C && *C && (*C)->isZeroValue())
      OnlyRight = true;

    bool LHSContainsUndef = false, RHSContainsUndef = false;
    SetTy LHSAAPVS, RHSAAPVS;
    if (!OnlyRight && !fillSetWithConstantValues(A, IRPosition::value(*LHS),
                                                 LHSAAPVS, LHSContainsUndef))
      return indicatePessimisticFixpoint();

    if (!OnlyLeft && !fillSetWithConstantValues(A, IRPosition::value(*RHS),
                                                RHSAAPVS, RHSContainsUndef))
      return indicatePessimisticFixpoint();

    if (OnlyLeft || OnlyRight) {
      // select (true/false), lhs, rhs
      auto *OpAA = OnlyLeft ? &LHSAAPVS : &RHSAAPVS;
      auto Undef = OnlyLeft ? LHSContainsUndef : RHSContainsUndef;

      if (Undef)
        unionAssumedWithUndef();
      else {
        for (const auto &It : *OpAA)
          unionAssumed(It);
      }

    } else if (LHSContainsUndef && RHSContainsUndef) {
      // select i1 *, undef , undef => undef
      unionAssumedWithUndef();
    } else {
      for (const auto &It : LHSAAPVS)
        unionAssumed(It);
      for (const auto &It : RHSAAPVS)
        unionAssumed(It);
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

    bool SrcContainsUndef = false;
    SetTy SrcPVS;
    if (!fillSetWithConstantValues(A, IRPosition::value(*Src), SrcPVS,
                                   SrcContainsUndef))
      return indicatePessimisticFixpoint();

    if (SrcContainsUndef)
      unionAssumedWithUndef();
    else {
      for (const APInt &S : SrcPVS) {
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

    bool LHSContainsUndef = false, RHSContainsUndef = false;
    SetTy LHSAAPVS, RHSAAPVS;
    if (!fillSetWithConstantValues(A, IRPosition::value(*LHS), LHSAAPVS,
                                   LHSContainsUndef) ||
        !fillSetWithConstantValues(A, IRPosition::value(*RHS), RHSAAPVS,
                                   RHSContainsUndef))
      return indicatePessimisticFixpoint();

    const APInt Zero = APInt(LHS->getType()->getIntegerBitWidth(), 0);

    // TODO: make use of undef flag to limit potential values aggressively.
    if (LHSContainsUndef && RHSContainsUndef) {
      if (!calculateBinaryOperatorAndTakeUnion(BinOp, Zero, Zero))
        return indicatePessimisticFixpoint();
    } else if (LHSContainsUndef) {
      for (const APInt &R : RHSAAPVS) {
        if (!calculateBinaryOperatorAndTakeUnion(BinOp, Zero, R))
          return indicatePessimisticFixpoint();
      }
    } else if (RHSContainsUndef) {
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

    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(potential_values)
  }
};

struct AAPotentialConstantValuesFunction : AAPotentialConstantValuesImpl {
  AAPotentialConstantValuesFunction(const IRPosition &IRP, Attributor &A)
      : AAPotentialConstantValuesImpl(IRP, A) {}

  /// See AbstractAttribute::initialize(...).
  ChangeStatus updateImpl(Attributor &A) override {
    llvm_unreachable(
        "AAPotentialConstantValues(Function|CallSite)::updateImpl will "
        "not be called");
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FN_ATTR(potential_values)
  }
};

struct AAPotentialConstantValuesCallSite : AAPotentialConstantValuesFunction {
  AAPotentialConstantValuesCallSite(const IRPosition &IRP, Attributor &A)
      : AAPotentialConstantValuesFunction(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CS_ATTR(potential_values)
  }
};

struct AAPotentialConstantValuesCallSiteReturned
    : AACallSiteReturnedFromReturned<AAPotentialConstantValues,
                                     AAPotentialConstantValuesImpl> {
  AAPotentialConstantValuesCallSiteReturned(const IRPosition &IRP,
                                            Attributor &A)
      : AACallSiteReturnedFromReturned<AAPotentialConstantValues,
                                       AAPotentialConstantValuesImpl>(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(potential_values)
  }
};

struct AAPotentialConstantValuesCallSiteArgument
    : AAPotentialConstantValuesFloating {
  AAPotentialConstantValuesCallSiteArgument(const IRPosition &IRP,
                                            Attributor &A)
      : AAPotentialConstantValuesFloating(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    AAPotentialConstantValuesImpl::initialize(A);
    if (isAtFixpoint())
      return;

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
    auto &AA = A.getAAFor<AAPotentialConstantValues>(
        *this, IRPosition::value(V), DepClassTy::REQUIRED);
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
    bool UsedAssumedInformation = false;
    if (A.isAssumedDead(getIRPosition(), nullptr, nullptr,
                        UsedAssumedInformation))
      return ChangeStatus::UNCHANGED;
    // A position whose simplified value does not have any value is
    // considered to be dead. We don't manifest noundef in such positions for
    // the same reason above.
    if (!A.getAssumedSimplified(getIRPosition(), *this, UsedAssumedInformation,
                                AA::Interprocedural)
             .has_value())
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

    SmallVector<AA::ValueAndContext> Values;
    bool UsedAssumedInformation = false;
    if (!A.getAssumedSimplifiedValues(getIRPosition(), *this, Values,
                                      AA::AnyScope, UsedAssumedInformation)) {
      Values.push_back({getAssociatedValue(), getCtxI()});
    }

    StateType T;
    auto VisitValueCB = [&](Value &V, const Instruction *CtxI) -> bool {
      const auto &AA = A.getAAFor<AANoUndef>(*this, IRPosition::value(V),
                                             DepClassTy::REQUIRED);
      if (this == &AA) {
        T.indicatePessimisticFixpoint();
      } else {
        const AANoUndef::StateType &S =
            static_cast<const AANoUndef::StateType &>(AA.getState());
        T ^= S;
      }
      return T.isValidState();
    };

    for (const auto &VAC : Values)
      if (!VisitValueCB(*VAC.getValue(), VAC.getCtxI()))
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

struct AACallEdgesImpl : public AACallEdges {
  AACallEdgesImpl(const IRPosition &IRP, Attributor &A) : AACallEdges(IRP, A) {}

  const SetVector<Function *> &getOptimisticEdges() const override {
    return CalledFunctions;
  }

  bool hasUnknownCallee() const override { return HasUnknownCallee; }

  bool hasNonAsmUnknownCallee() const override {
    return HasUnknownCalleeNonAsm;
  }

  const std::string getAsStr() const override {
    return "CallEdges[" + std::to_string(HasUnknownCallee) + "," +
           std::to_string(CalledFunctions.size()) + "]";
  }

  void trackStatistics() const override {}

protected:
  void addCalledFunction(Function *Fn, ChangeStatus &Change) {
    if (CalledFunctions.insert(Fn)) {
      Change = ChangeStatus::CHANGED;
      LLVM_DEBUG(dbgs() << "[AACallEdges] New call edge: " << Fn->getName()
                        << "\n");
    }
  }

  void setHasUnknownCallee(bool NonAsm, ChangeStatus &Change) {
    if (!HasUnknownCallee)
      Change = ChangeStatus::CHANGED;
    if (NonAsm && !HasUnknownCalleeNonAsm)
      Change = ChangeStatus::CHANGED;
    HasUnknownCalleeNonAsm |= NonAsm;
    HasUnknownCallee = true;
  }

private:
  /// Optimistic set of functions that might be called by this position.
  SetVector<Function *> CalledFunctions;

  /// Is there any call with a unknown callee.
  bool HasUnknownCallee = false;

  /// Is there any call with a unknown callee, excluding any inline asm.
  bool HasUnknownCalleeNonAsm = false;
};

struct AACallEdgesCallSite : public AACallEdgesImpl {
  AACallEdgesCallSite(const IRPosition &IRP, Attributor &A)
      : AACallEdgesImpl(IRP, A) {}
  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto VisitValue = [&](Value &V, const Instruction *CtxI) -> bool {
      if (Function *Fn = dyn_cast<Function>(&V)) {
        addCalledFunction(Fn, Change);
      } else {
        LLVM_DEBUG(dbgs() << "[AACallEdges] Unrecognized value: " << V << "\n");
        setHasUnknownCallee(true, Change);
      }

      // Explore all values.
      return true;
    };

    SmallVector<AA::ValueAndContext> Values;
    // Process any value that we might call.
    auto ProcessCalledOperand = [&](Value *V, Instruction *CtxI) {
      bool UsedAssumedInformation = false;
      Values.clear();
      if (!A.getAssumedSimplifiedValues(IRPosition::value(*V), *this, Values,
                                        AA::AnyScope, UsedAssumedInformation)) {
        Values.push_back({*V, CtxI});
      }
      for (auto &VAC : Values)
        VisitValue(*VAC.getValue(), VAC.getCtxI());
    };

    CallBase *CB = cast<CallBase>(getCtxI());

    if (CB->isInlineAsm()) {
      if (!hasAssumption(*CB->getCaller(), "ompx_no_call_asm") &&
          !hasAssumption(*CB, "ompx_no_call_asm"))
        setHasUnknownCallee(false, Change);
      return Change;
    }

    // Process callee metadata if available.
    if (auto *MD = getCtxI()->getMetadata(LLVMContext::MD_callees)) {
      for (const auto &Op : MD->operands()) {
        Function *Callee = mdconst::dyn_extract_or_null<Function>(Op);
        if (Callee)
          addCalledFunction(Callee, Change);
      }
      return Change;
    }

    // The most simple case.
    ProcessCalledOperand(CB->getCalledOperand(), CB);

    // Process callback functions.
    SmallVector<const Use *, 4u> CallbackUses;
    AbstractCallSite::getCallbackUses(*CB, CallbackUses);
    for (const Use *U : CallbackUses)
      ProcessCalledOperand(U->get(), CB);

    return Change;
  }
};

struct AACallEdgesFunction : public AACallEdgesImpl {
  AACallEdgesFunction(const IRPosition &IRP, Attributor &A)
      : AACallEdgesImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    auto ProcessCallInst = [&](Instruction &Inst) {
      CallBase &CB = cast<CallBase>(Inst);

      auto &CBEdges = A.getAAFor<AACallEdges>(
          *this, IRPosition::callsite_function(CB), DepClassTy::REQUIRED);
      if (CBEdges.hasNonAsmUnknownCallee())
        setHasUnknownCallee(true, Change);
      if (CBEdges.hasUnknownCallee())
        setHasUnknownCallee(false, Change);

      for (Function *F : CBEdges.getOptimisticEdges())
        addCalledFunction(F, Change);

      return true;
    };

    // Visit all callable instructions.
    bool UsedAssumedInformation = false;
    if (!A.checkForAllCallLikeInstructions(ProcessCallInst, *this,
                                           UsedAssumedInformation,
                                           /* CheckBBLivenessOnly */ true)) {
      // If we haven't looked at all call like instructions, assume that there
      // are unknown callees.
      setHasUnknownCallee(true, Change);
    }

    return Change;
  }
};

struct AAFunctionReachabilityFunction : public AAFunctionReachability {
private:
  struct QuerySet {
    void markReachable(const Function &Fn) {
      Reachable.insert(&Fn);
      Unreachable.erase(&Fn);
    }

    /// If there is no information about the function None is returned.
    Optional<bool> isCachedReachable(const Function &Fn) {
      // Assume that we can reach the function.
      // TODO: Be more specific with the unknown callee.
      if (CanReachUnknownCallee)
        return true;

      if (Reachable.count(&Fn))
        return true;

      if (Unreachable.count(&Fn))
        return false;

      return llvm::None;
    }

    /// Set of functions that we know for sure is reachable.
    DenseSet<const Function *> Reachable;

    /// Set of functions that are unreachable, but might become reachable.
    DenseSet<const Function *> Unreachable;

    /// If we can reach a function with a call to a unknown function we assume
    /// that we can reach any function.
    bool CanReachUnknownCallee = false;
  };

  struct QueryResolver : public QuerySet {
    ChangeStatus update(Attributor &A, const AAFunctionReachability &AA,
                        ArrayRef<const AACallEdges *> AAEdgesList) {
      ChangeStatus Change = ChangeStatus::UNCHANGED;

      for (const auto *AAEdges : AAEdgesList) {
        if (AAEdges->hasUnknownCallee()) {
          if (!CanReachUnknownCallee) {
            LLVM_DEBUG(dbgs()
                       << "[QueryResolver] Edges include unknown callee!\n");
            Change = ChangeStatus::CHANGED;
          }
          CanReachUnknownCallee = true;
          return Change;
        }
      }

      for (const Function *Fn : make_early_inc_range(Unreachable)) {
        if (checkIfReachable(A, AA, AAEdgesList, *Fn)) {
          Change = ChangeStatus::CHANGED;
          markReachable(*Fn);
        }
      }
      return Change;
    }

    bool isReachable(Attributor &A, AAFunctionReachability &AA,
                     ArrayRef<const AACallEdges *> AAEdgesList,
                     const Function &Fn) {
      Optional<bool> Cached = isCachedReachable(Fn);
      if (Cached)
        return Cached.value();

      // The query was not cached, thus it is new. We need to request an update
      // explicitly to make sure this the information is properly run to a
      // fixpoint.
      A.registerForUpdate(AA);

      // We need to assume that this function can't reach Fn to prevent
      // an infinite loop if this function is recursive.
      Unreachable.insert(&Fn);

      bool Result = checkIfReachable(A, AA, AAEdgesList, Fn);
      if (Result)
        markReachable(Fn);
      return Result;
    }

    bool checkIfReachable(Attributor &A, const AAFunctionReachability &AA,
                          ArrayRef<const AACallEdges *> AAEdgesList,
                          const Function &Fn) const {

      // Handle the most trivial case first.
      for (const auto *AAEdges : AAEdgesList) {
        const SetVector<Function *> &Edges = AAEdges->getOptimisticEdges();

        if (Edges.count(const_cast<Function *>(&Fn)))
          return true;
      }

      SmallVector<const AAFunctionReachability *, 8> Deps;
      for (const auto &AAEdges : AAEdgesList) {
        const SetVector<Function *> &Edges = AAEdges->getOptimisticEdges();

        for (Function *Edge : Edges) {
          // Functions that do not call back into the module can be ignored.
          if (Edge->hasFnAttribute(Attribute::NoCallback))
            continue;

          // We don't need a dependency if the result is reachable.
          const AAFunctionReachability &EdgeReachability =
              A.getAAFor<AAFunctionReachability>(
                  AA, IRPosition::function(*Edge), DepClassTy::NONE);
          Deps.push_back(&EdgeReachability);

          if (EdgeReachability.canReach(A, Fn))
            return true;
        }
      }

      // The result is false for now, set dependencies and leave.
      for (const auto *Dep : Deps)
        A.recordDependence(*Dep, AA, DepClassTy::REQUIRED);

      return false;
    }
  };

  /// Get call edges that can be reached by this instruction.
  bool getReachableCallEdges(Attributor &A, const AAReachability &Reachability,
                             const Instruction &Inst,
                             SmallVector<const AACallEdges *> &Result) const {
    // Determine call like instructions that we can reach from the inst.
    auto CheckCallBase = [&](Instruction &CBInst) {
      if (!Reachability.isAssumedReachable(A, Inst, CBInst))
        return true;

      auto &CB = cast<CallBase>(CBInst);
      const AACallEdges &AAEdges = A.getAAFor<AACallEdges>(
          *this, IRPosition::callsite_function(CB), DepClassTy::REQUIRED);

      Result.push_back(&AAEdges);
      return true;
    };

    bool UsedAssumedInformation = false;
    return A.checkForAllCallLikeInstructions(CheckCallBase, *this,
                                             UsedAssumedInformation,
                                             /* CheckBBLivenessOnly */ true);
  }

public:
  AAFunctionReachabilityFunction(const IRPosition &IRP, Attributor &A)
      : AAFunctionReachability(IRP, A) {}

  bool canReach(Attributor &A, const Function &Fn) const override {
    if (!isValidState())
      return true;

    const AACallEdges &AAEdges =
        A.getAAFor<AACallEdges>(*this, getIRPosition(), DepClassTy::REQUIRED);

    // Attributor returns attributes as const, so this function has to be
    // const for users of this attribute to use it without having to do
    // a const_cast.
    // This is a hack for us to be able to cache queries.
    auto *NonConstThis = const_cast<AAFunctionReachabilityFunction *>(this);
    bool Result = NonConstThis->WholeFunction.isReachable(A, *NonConstThis,
                                                          {&AAEdges}, Fn);

    return Result;
  }

  /// Can \p CB reach \p Fn
  bool canReach(Attributor &A, CallBase &CB,
                const Function &Fn) const override {
    if (!isValidState())
      return true;

    const AACallEdges &AAEdges = A.getAAFor<AACallEdges>(
        *this, IRPosition::callsite_function(CB), DepClassTy::REQUIRED);

    // Attributor returns attributes as const, so this function has to be
    // const for users of this attribute to use it without having to do
    // a const_cast.
    // This is a hack for us to be able to cache queries.
    auto *NonConstThis = const_cast<AAFunctionReachabilityFunction *>(this);
    QueryResolver &CBQuery = NonConstThis->CBQueries[&CB];

    bool Result = CBQuery.isReachable(A, *NonConstThis, {&AAEdges}, Fn);

    return Result;
  }

  bool instructionCanReach(Attributor &A, const Instruction &Inst,
                           const Function &Fn) const override {
    if (!isValidState())
      return true;

    const auto &Reachability = A.getAAFor<AAReachability>(
        *this, IRPosition::function(*getAssociatedFunction()),
        DepClassTy::REQUIRED);

    SmallVector<const AACallEdges *> CallEdges;
    bool AllKnown = getReachableCallEdges(A, Reachability, Inst, CallEdges);
    // Attributor returns attributes as const, so this function has to be
    // const for users of this attribute to use it without having to do
    // a const_cast.
    // This is a hack for us to be able to cache queries.
    auto *NonConstThis = const_cast<AAFunctionReachabilityFunction *>(this);
    QueryResolver &InstQSet = NonConstThis->InstQueries[&Inst];
    if (!AllKnown) {
      LLVM_DEBUG(dbgs() << "[AAReachability] Not all reachable edges known, "
                           "may reach unknown callee!\n");
      InstQSet.CanReachUnknownCallee = true;
    }

    return InstQSet.isReachable(A, *NonConstThis, CallEdges, Fn);
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const AACallEdges &AAEdges =
        A.getAAFor<AACallEdges>(*this, getIRPosition(), DepClassTy::REQUIRED);
    ChangeStatus Change = ChangeStatus::UNCHANGED;

    Change |= WholeFunction.update(A, *this, {&AAEdges});

    for (auto &CBPair : CBQueries) {
      const AACallEdges &AAEdges = A.getAAFor<AACallEdges>(
          *this, IRPosition::callsite_function(*CBPair.first),
          DepClassTy::REQUIRED);

      Change |= CBPair.second.update(A, *this, {&AAEdges});
    }

    // Update the Instruction queries.
    if (!InstQueries.empty()) {
      const AAReachability *Reachability = &A.getAAFor<AAReachability>(
          *this, IRPosition::function(*getAssociatedFunction()),
          DepClassTy::REQUIRED);

      // Check for local callbases first.
      for (auto &InstPair : InstQueries) {
        SmallVector<const AACallEdges *> CallEdges;
        bool AllKnown =
            getReachableCallEdges(A, *Reachability, *InstPair.first, CallEdges);
        // Update will return change if we this effects any queries.
        if (!AllKnown) {
          LLVM_DEBUG(dbgs() << "[AAReachability] Not all reachable edges "
                               "known, may reach unknown callee!\n");
          InstPair.second.CanReachUnknownCallee = true;
        }
        Change |= InstPair.second.update(A, *this, CallEdges);
      }
    }

    return Change;
  }

  const std::string getAsStr() const override {
    size_t QueryCount =
        WholeFunction.Reachable.size() + WholeFunction.Unreachable.size();

    return "FunctionReachability [" +
           (canReachUnknownCallee()
                ? "unknown"
                : (std::to_string(WholeFunction.Reachable.size()) + "," +
                   std::to_string(QueryCount))) +
           "]";
  }

  void trackStatistics() const override {}

private:
  bool canReachUnknownCallee() const override {
    return WholeFunction.CanReachUnknownCallee;
  }

  /// Used to answer if a the whole function can reacha a specific function.
  QueryResolver WholeFunction;

  /// Used to answer if a call base inside this function can reach a specific
  /// function.
  MapVector<const CallBase *, QueryResolver> CBQueries;

  /// This is for instruction queries than scan "forward".
  MapVector<const Instruction *, QueryResolver> InstQueries;
};
} // namespace

template <typename AAType>
static Optional<Constant *>
askForAssumedConstant(Attributor &A, const AbstractAttribute &QueryingAA,
                      const IRPosition &IRP, Type &Ty) {
  if (!Ty.isIntegerTy())
    return nullptr;

  // This will also pass the call base context.
  const auto &AA = A.getAAFor<AAType>(QueryingAA, IRP, DepClassTy::NONE);

  Optional<Constant *> COpt = AA.getAssumedConstant(A);

  if (!COpt.has_value()) {
    A.recordDependence(AA, QueryingAA, DepClassTy::OPTIONAL);
    return llvm::None;
  }
  if (auto *C = COpt.value()) {
    A.recordDependence(AA, QueryingAA, DepClassTy::OPTIONAL);
    return C;
  }
  return nullptr;
}

Value *AAPotentialValues::getSingleValue(
    Attributor &A, const AbstractAttribute &AA, const IRPosition &IRP,
    SmallVectorImpl<AA::ValueAndContext> &Values) {
  Type &Ty = *IRP.getAssociatedType();
  Optional<Value *> V;
  for (auto &It : Values) {
    V = AA::combineOptionalValuesInAAValueLatice(V, It.getValue(), &Ty);
    if (V.has_value() && !V.value())
      break;
  }
  if (!V.has_value())
    return UndefValue::get(&Ty);
  return V.value();
}

namespace {
struct AAPotentialValuesImpl : AAPotentialValues {
  using StateType = PotentialLLVMValuesState;

  AAPotentialValuesImpl(const IRPosition &IRP, Attributor &A)
      : AAPotentialValues(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    if (A.hasSimplificationCallback(getIRPosition())) {
      indicatePessimisticFixpoint();
      return;
    }
    Value *Stripped = getAssociatedValue().stripPointerCasts();
    if (isa<Constant>(Stripped)) {
      addValue(A, getState(), *Stripped, getCtxI(), AA::AnyScope,
               getAnchorScope());
      indicateOptimisticFixpoint();
      return;
    }
    AAPotentialValues::initialize(A);
  }

  /// See AbstractAttribute::getAsStr().
  const std::string getAsStr() const override {
    std::string Str;
    llvm::raw_string_ostream OS(Str);
    OS << getState();
    return OS.str();
  }

  template <typename AAType>
  static Optional<Value *> askOtherAA(Attributor &A,
                                      const AbstractAttribute &AA,
                                      const IRPosition &IRP, Type &Ty) {
    if (isa<Constant>(IRP.getAssociatedValue()))
      return &IRP.getAssociatedValue();
    Optional<Constant *> C = askForAssumedConstant<AAType>(A, AA, IRP, Ty);
    if (!C)
      return llvm::None;
    if (C.value())
      if (auto *CC = AA::getWithType(**C, Ty))
        return CC;
    return nullptr;
  }

  void addValue(Attributor &A, StateType &State, Value &V,
                const Instruction *CtxI, AA::ValueScope S,
                Function *AnchorScope) const {

    IRPosition ValIRP = IRPosition::value(V);
    if (auto *CB = dyn_cast_or_null<CallBase>(CtxI)) {
      for (const auto &U : CB->args()) {
        if (U.get() != &V)
          continue;
        ValIRP = IRPosition::callsite_argument(*CB, CB->getArgOperandNo(&U));
        break;
      }
    }

    Value *VPtr = &V;
    if (ValIRP.getAssociatedType()->isIntegerTy()) {
      Type &Ty = *getAssociatedType();
      Optional<Value *> SimpleV =
          askOtherAA<AAValueConstantRange>(A, *this, ValIRP, Ty);
      if (SimpleV.has_value() && !SimpleV.value()) {
        auto &PotentialConstantsAA = A.getAAFor<AAPotentialConstantValues>(
            *this, ValIRP, DepClassTy::OPTIONAL);
        if (PotentialConstantsAA.isValidState()) {
          for (const auto &It : PotentialConstantsAA.getAssumedSet())
            State.unionAssumed({{*ConstantInt::get(&Ty, It), nullptr}, S});
          if (PotentialConstantsAA.undefIsContained())
            State.unionAssumed({{*UndefValue::get(&Ty), nullptr}, S});
          return;
        }
      }
      if (!SimpleV.has_value())
        return;

      if (SimpleV.value())
        VPtr = SimpleV.value();
    }

    if (isa<ConstantInt>(VPtr))
      CtxI = nullptr;
    if (!AA::isValidInScope(*VPtr, AnchorScope))
      S = AA::ValueScope(S | AA::Interprocedural);

    State.unionAssumed({{*VPtr, CtxI}, S});
  }

  /// Helper struct to tie a value+context pair together with the scope for
  /// which this is the simplified version.
  struct ItemInfo {
    AA::ValueAndContext I;
    AA::ValueScope S;

    bool operator==(const ItemInfo &II) const {
      return II.I == I && II.S == S;
    };
    bool operator<(const ItemInfo &II) const {
      if (I == II.I)
        return S < II.S;
      return I < II.I;
    };
  };

  bool recurseForValue(Attributor &A, const IRPosition &IRP, AA::ValueScope S) {
    SmallMapVector<AA::ValueAndContext, int, 8> ValueScopeMap;
    for (auto CS : {AA::Intraprocedural, AA::Interprocedural}) {
      if (!(CS & S))
        continue;

      bool UsedAssumedInformation = false;
      SmallVector<AA::ValueAndContext> Values;
      if (!A.getAssumedSimplifiedValues(IRP, this, Values, CS,
                                        UsedAssumedInformation))
        return false;

      for (auto &It : Values)
        ValueScopeMap[It] += CS;
    }
    for (auto &It : ValueScopeMap)
      addValue(A, getState(), *It.first.getValue(), It.first.getCtxI(),
               AA::ValueScope(It.second), getAnchorScope());

    return true;
  }

  void giveUpOnIntraprocedural(Attributor &A) {
    auto NewS = StateType::getBestState(getState());
    for (const auto &It : getAssumedSet()) {
      if (It.second == AA::Intraprocedural)
        continue;
      addValue(A, NewS, *It.first.getValue(), It.first.getCtxI(),
               AA::Interprocedural, getAnchorScope());
    }
    assert(!undefIsContained() && "Undef should be an explicit value!");
    addValue(A, NewS, getAssociatedValue(), getCtxI(), AA::Intraprocedural,
             getAnchorScope());
    getState() = NewS;
  }

  /// See AbstractState::indicatePessimisticFixpoint(...).
  ChangeStatus indicatePessimisticFixpoint() override {
    getState() = StateType::getBestState(getState());
    getState().unionAssumed({{getAssociatedValue(), getCtxI()}, AA::AnyScope});
    AAPotentialValues::indicateOptimisticFixpoint();
    return ChangeStatus::CHANGED;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    return indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    SmallVector<AA::ValueAndContext> Values;
    for (AA::ValueScope S : {AA::Interprocedural, AA::Intraprocedural}) {
      Values.clear();
      if (!getAssumedSimplifiedValues(A, Values, S))
        continue;
      Value &OldV = getAssociatedValue();
      if (isa<UndefValue>(OldV))
        continue;
      Value *NewV = getSingleValue(A, *this, getIRPosition(), Values);
      if (!NewV || NewV == &OldV)
        continue;
      if (getCtxI() &&
          !AA::isValidAtPosition({*NewV, *getCtxI()}, A.getInfoCache()))
        continue;
      if (A.changeAfterManifest(getIRPosition(), *NewV))
        return ChangeStatus::CHANGED;
    }
    return ChangeStatus::UNCHANGED;
  }

  bool getAssumedSimplifiedValues(Attributor &A,
                                  SmallVectorImpl<AA::ValueAndContext> &Values,
                                  AA::ValueScope S) const override {
    if (!isValidState())
      return false;
    for (const auto &It : getAssumedSet())
      if (It.second & S)
        Values.push_back(It.first);
    assert(!undefIsContained() && "Undef should be an explicit value!");
    return true;
  }
};

struct AAPotentialValuesFloating : AAPotentialValuesImpl {
  AAPotentialValuesFloating(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto AssumedBefore = getAssumed();

    genericValueTraversal(A);

    return (AssumedBefore == getAssumed()) ? ChangeStatus::UNCHANGED
                                           : ChangeStatus::CHANGED;
  }

  /// Helper struct to remember which AAIsDead instances we actually used.
  struct LivenessInfo {
    const AAIsDead *LivenessAA = nullptr;
    bool AnyDead = false;
  };

  /// Check if \p Cmp is a comparison we can simplify.
  ///
  /// We handle multiple cases, one in which at least one operand is an
  /// (assumed) nullptr. If so, try to simplify it using AANonNull on the other
  /// operand. Return true if successful, in that case Worklist will be updated.
  bool handleCmp(Attributor &A, CmpInst &Cmp, ItemInfo II,
                 SmallVectorImpl<ItemInfo> &Worklist) {
    Value *LHS = Cmp.getOperand(0);
    Value *RHS = Cmp.getOperand(1);

    // Simplify the operands first.
    bool UsedAssumedInformation = false;
    const auto &SimplifiedLHS = A.getAssumedSimplified(
        IRPosition::value(*LHS, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Intraprocedural);
    if (!SimplifiedLHS.has_value())
      return true;
    if (!SimplifiedLHS.value())
      return false;
    LHS = *SimplifiedLHS;

    const auto &SimplifiedRHS = A.getAssumedSimplified(
        IRPosition::value(*RHS, getCallBaseContext()), *this,
        UsedAssumedInformation, AA::Intraprocedural);
    if (!SimplifiedRHS.has_value())
      return true;
    if (!SimplifiedRHS.value())
      return false;
    RHS = *SimplifiedRHS;

    LLVMContext &Ctx = Cmp.getContext();
    // Handle the trivial case first in which we don't even need to think about
    // null or non-null.
    if (LHS == RHS && (Cmp.isTrueWhenEqual() || Cmp.isFalseWhenEqual())) {
      Constant *NewV =
          ConstantInt::get(Type::getInt1Ty(Ctx), Cmp.isTrueWhenEqual());
      addValue(A, getState(), *NewV, /* CtxI */ nullptr, II.S,
               getAnchorScope());
      return true;
    }

    // From now on we only handle equalities (==, !=).
    ICmpInst *ICmp = dyn_cast<ICmpInst>(&Cmp);
    if (!ICmp || !ICmp->isEquality())
      return false;

    bool LHSIsNull = isa<ConstantPointerNull>(LHS);
    bool RHSIsNull = isa<ConstantPointerNull>(RHS);
    if (!LHSIsNull && !RHSIsNull)
      return false;

    // Left is the nullptr ==/!= non-nullptr case. We'll use AANonNull on the
    // non-nullptr operand and if we assume it's non-null we can conclude the
    // result of the comparison.
    assert((LHSIsNull || RHSIsNull) &&
           "Expected nullptr versus non-nullptr comparison at this point");

    // The index is the operand that we assume is not null.
    unsigned PtrIdx = LHSIsNull;
    auto &PtrNonNullAA = A.getAAFor<AANonNull>(
        *this, IRPosition::value(*ICmp->getOperand(PtrIdx)),
        DepClassTy::REQUIRED);
    if (!PtrNonNullAA.isAssumedNonNull())
      return false;

    // The new value depends on the predicate, true for != and false for ==.
    Constant *NewV = ConstantInt::get(Type::getInt1Ty(Ctx),
                                      ICmp->getPredicate() == CmpInst::ICMP_NE);
    addValue(A, getState(), *NewV, /* CtxI */ nullptr, II.S, getAnchorScope());
    return true;
  }

  bool handleSelectInst(Attributor &A, SelectInst &SI, ItemInfo II,
                        SmallVectorImpl<ItemInfo> &Worklist) {
    const Instruction *CtxI = II.I.getCtxI();
    bool UsedAssumedInformation = false;

    Optional<Constant *> C =
        A.getAssumedConstant(*SI.getCondition(), *this, UsedAssumedInformation);
    bool NoValueYet = !C.has_value();
    if (NoValueYet || isa_and_nonnull<UndefValue>(*C))
      return true;
    if (auto *CI = dyn_cast_or_null<ConstantInt>(*C)) {
      if (CI->isZero())
        Worklist.push_back({{*SI.getFalseValue(), CtxI}, II.S});
      else
        Worklist.push_back({{*SI.getTrueValue(), CtxI}, II.S});
    } else {
      // We could not simplify the condition, assume both values.
      Worklist.push_back({{*SI.getTrueValue(), CtxI}, II.S});
      Worklist.push_back({{*SI.getFalseValue(), CtxI}, II.S});
    }
    return true;
  }

  bool handleLoadInst(Attributor &A, LoadInst &LI, ItemInfo II,
                      SmallVectorImpl<ItemInfo> &Worklist) {
    SmallSetVector<Value *, 4> PotentialCopies;
    SmallSetVector<Instruction *, 4> PotentialValueOrigins;
    bool UsedAssumedInformation = false;
    if (!AA::getPotentiallyLoadedValues(A, LI, PotentialCopies,
                                        PotentialValueOrigins, *this,
                                        UsedAssumedInformation,
                                        /* OnlyExact */ true)) {
      LLVM_DEBUG(dbgs() << "[AAPotentialValues] Failed to get potentially "
                           "loaded values for load instruction "
                        << LI << "\n");
      return false;
    }

    // Do not simplify loads that are only used in llvm.assume if we cannot also
    // remove all stores that may feed into the load. The reason is that the
    // assume is probably worth something as long as the stores are around.
    InformationCache &InfoCache = A.getInfoCache();
    if (InfoCache.isOnlyUsedByAssume(LI)) {
      if (!llvm::all_of(PotentialValueOrigins, [&](Instruction *I) {
            if (!I)
              return true;
            if (auto *SI = dyn_cast<StoreInst>(I))
              return A.isAssumedDead(SI->getOperandUse(0), this,
                                     /* LivenessAA */ nullptr,
                                     UsedAssumedInformation,
                                     /* CheckBBLivenessOnly */ false);
            return A.isAssumedDead(*I, this, /* LivenessAA */ nullptr,
                                   UsedAssumedInformation,
                                   /* CheckBBLivenessOnly */ false);
          })) {
        LLVM_DEBUG(dbgs() << "[AAPotentialValues] Load is onl used by assumes "
                             "and we cannot delete all the stores: "
                          << LI << "\n");
        return false;
      }
    }

    // Values have to be dynamically unique or we loose the fact that a
    // single llvm::Value might represent two runtime values (e.g.,
    // stack locations in different recursive calls).
    const Instruction *CtxI = II.I.getCtxI();
    bool ScopeIsLocal = (II.S & AA::Intraprocedural);
    bool AllLocal = ScopeIsLocal;
    bool DynamicallyUnique = llvm::all_of(PotentialCopies, [&](Value *PC) {
      AllLocal &= AA::isValidInScope(*PC, getAnchorScope());
      return AA::isDynamicallyUnique(A, *this, *PC);
    });
    if (!DynamicallyUnique) {
      LLVM_DEBUG(dbgs() << "[AAPotentialValues] Not all potentially loaded "
                           "values are dynamically unique: "
                        << LI << "\n");
      return false;
    }

    for (auto *PotentialCopy : PotentialCopies) {
      if (AllLocal) {
        Worklist.push_back({{*PotentialCopy, CtxI}, II.S});
      } else {
        Worklist.push_back({{*PotentialCopy, CtxI}, AA::Interprocedural});
      }
    }
    if (!AllLocal && ScopeIsLocal)
      addValue(A, getState(), LI, CtxI, AA::Intraprocedural, getAnchorScope());
    return true;
  }

  bool handlePHINode(
      Attributor &A, PHINode &PHI, ItemInfo II,
      SmallVectorImpl<ItemInfo> &Worklist,
      SmallMapVector<const Function *, LivenessInfo, 4> &LivenessAAs) {
    auto GetLivenessInfo = [&](const Function &F) -> LivenessInfo & {
      LivenessInfo &LI = LivenessAAs[&F];
      if (!LI.LivenessAA)
        LI.LivenessAA = &A.getAAFor<AAIsDead>(*this, IRPosition::function(F),
                                              DepClassTy::NONE);
      return LI;
    };

    LivenessInfo &LI = GetLivenessInfo(*PHI.getFunction());
    for (unsigned u = 0, e = PHI.getNumIncomingValues(); u < e; u++) {
      BasicBlock *IncomingBB = PHI.getIncomingBlock(u);
      if (LI.LivenessAA->isEdgeDead(IncomingBB, PHI.getParent())) {
        LI.AnyDead = true;
        continue;
      }
      Worklist.push_back(
          {{*PHI.getIncomingValue(u), IncomingBB->getTerminator()}, II.S});
    }
    return true;
  }

  /// Use the generic, non-optimistic InstSimplfy functionality if we managed to
  /// simplify any operand of the instruction \p I. Return true if successful,
  /// in that case Worklist will be updated.
  bool handleGenericInst(Attributor &A, Instruction &I, ItemInfo II,
                         SmallVectorImpl<ItemInfo> &Worklist) {
    bool SomeSimplified = false;
    bool UsedAssumedInformation = false;

    SmallVector<Value *, 8> NewOps(I.getNumOperands());
    int Idx = 0;
    for (Value *Op : I.operands()) {
      const auto &SimplifiedOp = A.getAssumedSimplified(
          IRPosition::value(*Op, getCallBaseContext()), *this,
          UsedAssumedInformation, AA::Intraprocedural);
      // If we are not sure about any operand we are not sure about the entire
      // instruction, we'll wait.
      if (!SimplifiedOp.has_value())
        return true;

      if (SimplifiedOp.value())
        NewOps[Idx] = SimplifiedOp.value();
      else
        NewOps[Idx] = Op;

      SomeSimplified |= (NewOps[Idx] != Op);
      ++Idx;
    }

    // We won't bother with the InstSimplify interface if we didn't simplify any
    // operand ourselves.
    if (!SomeSimplified)
      return false;

    InformationCache &InfoCache = A.getInfoCache();
    Function *F = I.getFunction();
    const auto *DT =
        InfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(*F);
    const auto *TLI = A.getInfoCache().getTargetLibraryInfoForFunction(*F);
    auto *AC = InfoCache.getAnalysisResultForFunction<AssumptionAnalysis>(*F);
    OptimizationRemarkEmitter *ORE = nullptr;

    const DataLayout &DL = I.getModule()->getDataLayout();
    SimplifyQuery Q(DL, TLI, DT, AC, &I);
    Value *NewV = simplifyInstructionWithOperands(&I, NewOps, Q, ORE);
    if (!NewV || NewV == &I)
      return false;

    LLVM_DEBUG(dbgs() << "Generic inst " << I << " assumed simplified to "
                      << *NewV << "\n");
    Worklist.push_back({{*NewV, II.I.getCtxI()}, II.S});
    return true;
  }

  bool simplifyInstruction(
      Attributor &A, Instruction &I, ItemInfo II,
      SmallVectorImpl<ItemInfo> &Worklist,
      SmallMapVector<const Function *, LivenessInfo, 4> &LivenessAAs) {
    if (auto *CI = dyn_cast<CmpInst>(&I))
      if (handleCmp(A, *CI, II, Worklist))
        return true;

    switch (I.getOpcode()) {
    case Instruction::Select:
      return handleSelectInst(A, cast<SelectInst>(I), II, Worklist);
    case Instruction::PHI:
      return handlePHINode(A, cast<PHINode>(I), II, Worklist, LivenessAAs);
    case Instruction::Load:
      return handleLoadInst(A, cast<LoadInst>(I), II, Worklist);
    default:
      return handleGenericInst(A, I, II, Worklist);
    };
    return false;
  }

  void genericValueTraversal(Attributor &A) {
    SmallMapVector<const Function *, LivenessInfo, 4> LivenessAAs;

    Value *InitialV = &getAssociatedValue();
    SmallSet<ItemInfo, 16> Visited;
    SmallVector<ItemInfo, 16> Worklist;
    Worklist.push_back({{*InitialV, getCtxI()}, AA::AnyScope});

    int Iteration = 0;
    do {
      ItemInfo II = Worklist.pop_back_val();
      Value *V = II.I.getValue();
      assert(V);
      const Instruction *CtxI = II.I.getCtxI();
      AA::ValueScope S = II.S;

      // Check if we should process the current value. To prevent endless
      // recursion keep a record of the values we followed!
      if (!Visited.insert(II).second)
        continue;

      // Make sure we limit the compile time for complex expressions.
      if (Iteration++ >= MaxPotentialValuesIterations) {
        LLVM_DEBUG(dbgs() << "Generic value traversal reached iteration limit: "
                          << Iteration << "!\n");
        addValue(A, getState(), *V, CtxI, S, getAnchorScope());
        continue;
      }

      // Explicitly look through calls with a "returned" attribute if we do
      // not have a pointer as stripPointerCasts only works on them.
      Value *NewV = nullptr;
      if (V->getType()->isPointerTy()) {
        NewV = AA::getWithType(*V->stripPointerCasts(), *V->getType());
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
        Worklist.push_back({{*NewV, CtxI}, S});
        continue;
      }

      if (auto *I = dyn_cast<Instruction>(V)) {
        if (simplifyInstruction(A, *I, II, Worklist, LivenessAAs))
          continue;
      }

      if (V != InitialV || isa<Argument>(V))
        if (recurseForValue(A, IRPosition::value(*V), II.S))
          continue;

      // If we haven't stripped anything we give up.
      if (V == InitialV && CtxI == getCtxI()) {
        indicatePessimisticFixpoint();
        return;
      }

      addValue(A, getState(), *V, CtxI, S, getAnchorScope());
    } while (!Worklist.empty());

    // If we actually used liveness information so we have to record a
    // dependence.
    for (auto &It : LivenessAAs)
      if (It.second.AnyDead)
        A.recordDependence(*It.second.LivenessAA, *this, DepClassTy::OPTIONAL);
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FLOATING_ATTR(potential_values)
  }
};

struct AAPotentialValuesArgument final : AAPotentialValuesImpl {
  using Base = AAPotentialValuesImpl;
  AAPotentialValuesArgument(const IRPosition &IRP, Attributor &A)
      : Base(IRP, A) {}

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    auto &Arg = cast<Argument>(getAssociatedValue());
    if (Arg.hasPointeeInMemoryValueAttr())
      indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto AssumedBefore = getAssumed();

    unsigned CSArgNo = getCallSiteArgNo();

    bool UsedAssumedInformation = false;
    SmallVector<AA::ValueAndContext> Values;
    auto CallSitePred = [&](AbstractCallSite ACS) {
      const auto CSArgIRP = IRPosition::callsite_argument(ACS, CSArgNo);
      if (CSArgIRP.getPositionKind() == IRP_INVALID)
        return false;

      if (!A.getAssumedSimplifiedValues(CSArgIRP, this, Values,
                                        AA::Interprocedural,
                                        UsedAssumedInformation))
        return false;

      return isValidState();
    };

    if (!A.checkForAllCallSites(CallSitePred, *this,
                                /* RequireAllCallSites */ true,
                                UsedAssumedInformation))
      return indicatePessimisticFixpoint();

    Function *Fn = getAssociatedFunction();
    bool AnyNonLocal = false;
    for (auto &It : Values) {
      if (isa<Constant>(It.getValue())) {
        addValue(A, getState(), *It.getValue(), It.getCtxI(), AA::AnyScope,
                 getAnchorScope());
        continue;
      }
      if (!AA::isDynamicallyUnique(A, *this, *It.getValue()))
        return indicatePessimisticFixpoint();

      if (auto *Arg = dyn_cast<Argument>(It.getValue()))
        if (Arg->getParent() == Fn) {
          addValue(A, getState(), *It.getValue(), It.getCtxI(), AA::AnyScope,
                   getAnchorScope());
          continue;
        }
      addValue(A, getState(), *It.getValue(), It.getCtxI(), AA::Interprocedural,
               getAnchorScope());
      AnyNonLocal = true;
    }
    assert(!undefIsContained() && "Undef should be an explicit value!");
    if (AnyNonLocal)
      giveUpOnIntraprocedural(A);

    return (AssumedBefore == getAssumed()) ? ChangeStatus::UNCHANGED
                                           : ChangeStatus::CHANGED;
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

  /// See AbstractAttribute::initialize(..).
  void initialize(Attributor &A) override {
    if (A.hasSimplificationCallback(getIRPosition()))
      indicatePessimisticFixpoint();
    else
      AAPotentialValues::initialize(A);
  }

  ChangeStatus manifest(Attributor &A) override {
    // We queried AAValueSimplify for the returned values so they will be
    // replaced if a simplified form was found. Nothing to do here.
    return ChangeStatus::UNCHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    return AAPotentialValues::indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_FNRET_ATTR(potential_values)
  }
};

struct AAPotentialValuesFunction : AAPotentialValuesImpl {
  AAPotentialValuesFunction(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
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

struct AAPotentialValuesCallSiteReturned : AAPotentialValuesImpl {
  AAPotentialValuesCallSiteReturned(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesImpl(IRP, A) {}

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    auto AssumedBefore = getAssumed();

    Function *Callee = getAssociatedFunction();
    if (!Callee)
      return indicatePessimisticFixpoint();

    bool UsedAssumedInformation = false;
    auto *CB = cast<CallBase>(getCtxI());
    if (CB->isMustTailCall() &&
        !A.isAssumedDead(IRPosition::inst(*CB), this, nullptr,
                         UsedAssumedInformation))
      return indicatePessimisticFixpoint();

    SmallVector<AA::ValueAndContext> Values;
    if (!A.getAssumedSimplifiedValues(IRPosition::returned(*Callee), this,
                                      Values, AA::Intraprocedural,
                                      UsedAssumedInformation))
      return indicatePessimisticFixpoint();

    Function *Caller = CB->getCaller();

    bool AnyNonLocal = false;
    for (auto &It : Values) {
      Value *V = It.getValue();
      Optional<Value *> CallerV = A.translateArgumentToCallSiteContent(
          V, *CB, *this, UsedAssumedInformation);
      if (!CallerV.has_value()) {
        // Nothing to do as long as no value was determined.
        continue;
      }
      V = CallerV.value() ? CallerV.value() : V;
      if (AA::isDynamicallyUnique(A, *this, *V) &&
          AA::isValidInScope(*V, Caller)) {
        if (CallerV.value()) {
          SmallVector<AA::ValueAndContext> ArgValues;
          IRPosition IRP = IRPosition::value(*V);
          if (auto *Arg = dyn_cast<Argument>(V))
            if (Arg->getParent() == CB->getCalledFunction())
              IRP = IRPosition::callsite_argument(*CB, Arg->getArgNo());
          if (recurseForValue(A, IRP, AA::AnyScope))
            continue;
        }
        addValue(A, getState(), *V, CB, AA::AnyScope, getAnchorScope());
      } else {
        AnyNonLocal = true;
        break;
      }
    }
    if (AnyNonLocal) {
      Values.clear();
      if (!A.getAssumedSimplifiedValues(IRPosition::returned(*Callee), this,
                                        Values, AA::Interprocedural,
                                        UsedAssumedInformation))
        return indicatePessimisticFixpoint();
      AnyNonLocal = false;
      getState() = PotentialLLVMValuesState::getBestState();
      for (auto &It : Values) {
        Value *V = It.getValue();
        if (!AA::isDynamicallyUnique(A, *this, *V))
          return indicatePessimisticFixpoint();
        if (AA::isValidInScope(*V, Caller)) {
          addValue(A, getState(), *V, CB, AA::AnyScope, getAnchorScope());
        } else {
          AnyNonLocal = true;
          addValue(A, getState(), *V, CB, AA::Interprocedural,
                   getAnchorScope());
        }
      }
      if (AnyNonLocal)
        giveUpOnIntraprocedural(A);
    }
    return (AssumedBefore == getAssumed()) ? ChangeStatus::UNCHANGED
                                           : ChangeStatus::CHANGED;
  }

  ChangeStatus indicatePessimisticFixpoint() override {
    return AAPotentialValues::indicatePessimisticFixpoint();
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSRET_ATTR(potential_values)
  }
};

struct AAPotentialValuesCallSiteArgument : AAPotentialValuesFloating {
  AAPotentialValuesCallSiteArgument(const IRPosition &IRP, Attributor &A)
      : AAPotentialValuesFloating(IRP, A) {}

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {
    STATS_DECLTRACK_CSARG_ATTR(potential_values)
  }
};
} // namespace

/// ---------------------- Assumption Propagation ------------------------------
namespace {
struct AAAssumptionInfoImpl : public AAAssumptionInfo {
  AAAssumptionInfoImpl(const IRPosition &IRP, Attributor &A,
                       const DenseSet<StringRef> &Known)
      : AAAssumptionInfo(IRP, A, Known) {}

  bool hasAssumption(const StringRef Assumption) const override {
    return isValidState() && setContains(Assumption);
  }

  /// See AbstractAttribute::getAsStr()
  const std::string getAsStr() const override {
    const SetContents &Known = getKnown();
    const SetContents &Assumed = getAssumed();

    const std::string KnownStr =
        llvm::join(Known.getSet().begin(), Known.getSet().end(), ",");
    const std::string AssumedStr =
        (Assumed.isUniversal())
            ? "Universal"
            : llvm::join(Assumed.getSet().begin(), Assumed.getSet().end(), ",");

    return "Known [" + KnownStr + "]," + " Assumed [" + AssumedStr + "]";
  }
};

/// Propagates assumption information from parent functions to all of their
/// successors. An assumption can be propagated if the containing function
/// dominates the called function.
///
/// We start with a "known" set of assumptions already valid for the associated
/// function and an "assumed" set that initially contains all possible
/// assumptions. The assumed set is inter-procedurally updated by narrowing its
/// contents as concrete values are known. The concrete values are seeded by the
/// first nodes that are either entries into the call graph, or contains no
/// assumptions. Each node is updated as the intersection of the assumed state
/// with all of its predecessors.
struct AAAssumptionInfoFunction final : AAAssumptionInfoImpl {
  AAAssumptionInfoFunction(const IRPosition &IRP, Attributor &A)
      : AAAssumptionInfoImpl(IRP, A,
                             getAssumptions(*IRP.getAssociatedFunction())) {}

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    const auto &Assumptions = getKnown();

    // Don't manifest a universal set if it somehow made it here.
    if (Assumptions.isUniversal())
      return ChangeStatus::UNCHANGED;

    Function *AssociatedFunction = getAssociatedFunction();

    bool Changed = addAssumptions(*AssociatedFunction, Assumptions.getSet());

    return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    bool Changed = false;

    auto CallSitePred = [&](AbstractCallSite ACS) {
      const auto &AssumptionAA = A.getAAFor<AAAssumptionInfo>(
          *this, IRPosition::callsite_function(*ACS.getInstruction()),
          DepClassTy::REQUIRED);
      // Get the set of assumptions shared by all of this function's callers.
      Changed |= getIntersection(AssumptionAA.getAssumed());
      return !getAssumed().empty() || !getKnown().empty();
    };

    bool UsedAssumedInformation = false;
    // Get the intersection of all assumptions held by this node's predecessors.
    // If we don't know all the call sites then this is either an entry into the
    // call graph or an empty node. This node is known to only contain its own
    // assumptions and can be propagated to its successors.
    if (!A.checkForAllCallSites(CallSitePred, *this, true,
                                UsedAssumedInformation))
      return indicatePessimisticFixpoint();

    return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  void trackStatistics() const override {}
};

/// Assumption Info defined for call sites.
struct AAAssumptionInfoCallSite final : AAAssumptionInfoImpl {

  AAAssumptionInfoCallSite(const IRPosition &IRP, Attributor &A)
      : AAAssumptionInfoImpl(IRP, A, getInitialAssumptions(IRP)) {}

  /// See AbstractAttribute::initialize(...).
  void initialize(Attributor &A) override {
    const IRPosition &FnPos = IRPosition::function(*getAnchorScope());
    A.getAAFor<AAAssumptionInfo>(*this, FnPos, DepClassTy::REQUIRED);
  }

  /// See AbstractAttribute::manifest(...).
  ChangeStatus manifest(Attributor &A) override {
    // Don't manifest a universal set if it somehow made it here.
    if (getKnown().isUniversal())
      return ChangeStatus::UNCHANGED;

    CallBase &AssociatedCall = cast<CallBase>(getAssociatedValue());
    bool Changed = addAssumptions(AssociatedCall, getAssumed().getSet());

    return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::updateImpl(...).
  ChangeStatus updateImpl(Attributor &A) override {
    const IRPosition &FnPos = IRPosition::function(*getAnchorScope());
    auto &AssumptionAA =
        A.getAAFor<AAAssumptionInfo>(*this, FnPos, DepClassTy::REQUIRED);
    bool Changed = getIntersection(AssumptionAA.getAssumed());
    return Changed ? ChangeStatus::CHANGED : ChangeStatus::UNCHANGED;
  }

  /// See AbstractAttribute::trackStatistics()
  void trackStatistics() const override {}

private:
  /// Helper to initialized the known set as all the assumptions this call and
  /// the callee contain.
  DenseSet<StringRef> getInitialAssumptions(const IRPosition &IRP) {
    const CallBase &CB = cast<CallBase>(IRP.getAssociatedValue());
    auto Assumptions = getAssumptions(CB);
    if (Function *F = IRP.getAssociatedFunction())
      set_union(Assumptions, getAssumptions(*F));
    if (Function *F = IRP.getAssociatedFunction())
      set_union(Assumptions, getAssumptions(*F));
    return Assumptions;
  }
};
} // namespace

AACallGraphNode *AACallEdgeIterator::operator*() const {
  return static_cast<AACallGraphNode *>(const_cast<AACallEdges *>(
      &A.getOrCreateAAFor<AACallEdges>(IRPosition::function(**I))));
}

void AttributorCallGraph::print() { llvm::WriteGraph(outs(), this); }

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
const char AAInstanceInfo::ID = 0;
const char AANoCapture::ID = 0;
const char AAValueSimplify::ID = 0;
const char AAHeapToStack::ID = 0;
const char AAPrivatizablePtr::ID = 0;
const char AAMemoryBehavior::ID = 0;
const char AAMemoryLocation::ID = 0;
const char AAValueConstantRange::ID = 0;
const char AAPotentialConstantValues::ID = 0;
const char AAPotentialValues::ID = 0;
const char AANoUndef::ID = 0;
const char AACallEdges::ID = 0;
const char AAFunctionReachability::ID = 0;
const char AAPointerInfo::ID = 0;
const char AAAssumptionInfo::ID = 0;

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
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AACallEdges)
CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAAssumptionInfo)

CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANonNull)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoAlias)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAPrivatizablePtr)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AADereferenceable)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAAlign)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAInstanceInfo)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoCapture)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAValueConstantRange)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAPotentialConstantValues)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAPotentialValues)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoUndef)
CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAPointerInfo)

CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAValueSimplify)
CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAIsDead)
CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION(AANoFree)

CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAHeapToStack)
CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAReachability)
CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAUndefinedBehavior)
CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAFunctionReachability)

CREATE_NON_RET_ABSTRACT_ATTRIBUTE_FOR_POSITION(AAMemoryBehavior)

#undef CREATE_FUNCTION_ONLY_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_FUNCTION_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_NON_RET_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_VALUE_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef CREATE_ALL_ABSTRACT_ATTRIBUTE_FOR_POSITION
#undef SWITCH_PK_CREATE
#undef SWITCH_PK_INV
