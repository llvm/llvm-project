//===- FunctionSpecialization.cpp - Function Specialization ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This specialises functions with constant parameters. Constant parameters
// like function pointers and constant globals are propagated to the callee by
// specializing the function. The main benefit of this pass at the moment is
// that indirect calls are transformed into direct calls, which provides inline
// opportunities that the inliner would not have been able to achieve. That's
// why function specialisation is run before the inliner in the optimisation
// pipeline; that is by design. Otherwise, we would only benefit from constant
// passing, which is a valid use-case too, but hasn't been explored much in
// terms of performance uplifts, cost-model and compile-time impact.
//
// Current limitations:
// - It does not yet handle integer ranges. We do support "literal constants",
//   but that's off by default under an option.
// - The cost-model could be further looked into (it mainly focuses on inlining
//   benefits),
//
// Ideas:
// - With a function specialization attribute for arguments, we could have
//   a direct way to steer function specialization, avoiding the cost-model,
//   and thus control compile-times / code-size.
//
// Todos:
// - Specializing recursive functions relies on running the transformation a
//   number of times, which is controlled by option
//   `func-specialization-max-iters`. Thus, increasing this value and the
//   number of iterations, will linearly increase the number of times recursive
//   functions get specialized, see also the discussion in
//   https://reviews.llvm.org/D106426 for details. Perhaps there is a
//   compile-time friendlier way to control/limit the number of specialisations
//   for recursive functions.
// - Don't transform the function if function specialization does not trigger;
//   the SCCPSolver may make IR changes.
//
// References:
// - 2021 LLVM Dev Mtg “Introducing function specialisation, and can we enable
//   it by default?”, https://www.youtube.com/watch?v=zJiCjeXgV5Q
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionSpecialization.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueLattice.h"
#include "llvm/Analysis/ValueLatticeUtils.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "function-specialization"

STATISTIC(NumFuncSpecialized, "Number of functions specialized");

static cl::opt<bool> ForceFunctionSpecialization(
    "force-function-specialization", cl::init(false), cl::Hidden,
    cl::desc("Force function specialization for every call site with a "
             "constant argument"));

static cl::opt<unsigned> MaxClonesThreshold(
    "func-specialization-max-clones", cl::Hidden,
    cl::desc("The maximum number of clones allowed for a single function "
             "specialization"),
    cl::init(3));

static cl::opt<unsigned> SmallFunctionThreshold(
    "func-specialization-size-threshold", cl::Hidden,
    cl::desc("Don't specialize functions that have less than this theshold "
             "number of instructions"),
    cl::init(100));

static cl::opt<unsigned>
    AvgLoopIterationCount("func-specialization-avg-iters-cost", cl::Hidden,
                          cl::desc("Average loop iteration count cost"),
                          cl::init(10));

static cl::opt<bool> SpecializeOnAddresses(
    "func-specialization-on-address", cl::init(false), cl::Hidden,
    cl::desc("Enable function specialization on the address of global values"));

// Disabled by default as it can significantly increase compilation times.
//
// https://llvm-compile-time-tracker.com
// https://github.com/nikic/llvm-compile-time-tracker
static cl::opt<bool> EnableSpecializationForLiteralConstant(
    "function-specialization-for-literal-constant", cl::init(false), cl::Hidden,
    cl::desc("Enable specialization of functions that take a literal constant "
             "as an argument."));

Constant *FunctionSpecializer::getPromotableAlloca(AllocaInst *Alloca,
                                                   CallInst *Call) {
  Value *StoreValue = nullptr;
  for (auto *User : Alloca->users()) {
    // We can't use llvm::isAllocaPromotable() as that would fail because of
    // the usage in the CallInst, which is what we check here.
    if (User == Call)
      continue;
    if (auto *Bitcast = dyn_cast<BitCastInst>(User)) {
      if (!Bitcast->hasOneUse() || *Bitcast->user_begin() != Call)
        return nullptr;
      continue;
    }

    if (auto *Store = dyn_cast<StoreInst>(User)) {
      // This is a duplicate store, bail out.
      if (StoreValue || Store->isVolatile())
        return nullptr;
      StoreValue = Store->getValueOperand();
      continue;
    }
    // Bail if there is any other unknown usage.
    return nullptr;
  }
  return getCandidateConstant(StoreValue);
}

// A constant stack value is an AllocaInst that has a single constant
// value stored to it. Return this constant if such an alloca stack value
// is a function argument.
Constant *FunctionSpecializer::getConstantStackValue(CallInst *Call,
                                                     Value *Val) {
  if (!Val)
    return nullptr;
  Val = Val->stripPointerCasts();
  if (auto *ConstVal = dyn_cast<ConstantInt>(Val))
    return ConstVal;
  auto *Alloca = dyn_cast<AllocaInst>(Val);
  if (!Alloca || !Alloca->getAllocatedType()->isIntegerTy())
    return nullptr;
  return getPromotableAlloca(Alloca, Call);
}

// To support specializing recursive functions, it is important to propagate
// constant arguments because after a first iteration of specialisation, a
// reduced example may look like this:
//
//     define internal void @RecursiveFn(i32* arg1) {
//       %temp = alloca i32, align 4
//       store i32 2 i32* %temp, align 4
//       call void @RecursiveFn.1(i32* nonnull %temp)
//       ret void
//     }
//
// Before a next iteration, we need to propagate the constant like so
// which allows further specialization in next iterations.
//
//     @funcspec.arg = internal constant i32 2
//
//     define internal void @someFunc(i32* arg1) {
//       call void @otherFunc(i32* nonnull @funcspec.arg)
//       ret void
//     }
//
void FunctionSpecializer::promoteConstantStackValues() {
  // Iterate over the argument tracked functions see if there
  // are any new constant values for the call instruction via
  // stack variables.
  for (Function &F : M) {
    if (!Solver.isArgumentTrackedFunction(&F))
      continue;

    for (auto *User : F.users()) {

      auto *Call = dyn_cast<CallInst>(User);
      if (!Call)
        continue;

      if (!Solver.isBlockExecutable(Call->getParent()))
        continue;

      bool Changed = false;
      for (const Use &U : Call->args()) {
        unsigned Idx = Call->getArgOperandNo(&U);
        Value *ArgOp = Call->getArgOperand(Idx);
        Type *ArgOpType = ArgOp->getType();

        if (!Call->onlyReadsMemory(Idx) || !ArgOpType->isPointerTy())
          continue;

        auto *ConstVal = getConstantStackValue(Call, ArgOp);
        if (!ConstVal)
          continue;

        Value *GV = new GlobalVariable(M, ConstVal->getType(), true,
                                       GlobalValue::InternalLinkage, ConstVal,
                                       "funcspec.arg");
        if (ArgOpType != ConstVal->getType())
          GV = ConstantExpr::getBitCast(cast<Constant>(GV), ArgOpType);

        Call->setArgOperand(Idx, GV);
        Changed = true;
      }

      // Add the changed CallInst to Solver Worklist
      if (Changed)
        Solver.visitCall(*Call);
    }
  }
}

// ssa_copy intrinsics are introduced by the SCCP solver. These intrinsics
// interfere with the promoteConstantStackValues() optimization.
static void removeSSACopy(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &Inst : llvm::make_early_inc_range(BB)) {
      auto *II = dyn_cast<IntrinsicInst>(&Inst);
      if (!II)
        continue;
      if (II->getIntrinsicID() != Intrinsic::ssa_copy)
        continue;
      Inst.replaceAllUsesWith(II->getOperand(0));
      Inst.eraseFromParent();
    }
  }
}

/// Remove any ssa_copy intrinsics that may have been introduced.
void FunctionSpecializer::cleanUpSSA() {
  for (Function *F : SpecializedFuncs)
    removeSSACopy(*F);
}

/// Attempt to specialize functions in the module to enable constant
/// propagation across function boundaries.
///
/// \returns true if at least one function is specialized.
bool FunctionSpecializer::run() {
  bool Changed = false;

  for (Function &F : M) {
    if (!isCandidateFunction(&F))
      continue;

    auto Cost = getSpecializationCost(&F);
    if (!Cost.isValid()) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Invalid specialization cost.\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "FnSpecialization: Specialization cost for "
                      << F.getName() << " is " << Cost << "\n");

    SmallVector<CallSpecBinding, 8> Specializations;
    if (!findSpecializations(&F, Cost, Specializations)) {
      LLVM_DEBUG(
          dbgs() << "FnSpecialization: No possible specializations found\n");
      continue;
    }

    Changed = true;

    SmallVector<Function *, 4> Clones;
    for (CallSpecBinding &Specialization : Specializations)
      Clones.push_back(createSpecialization(&F, Specialization));

    Solver.solveWhileResolvedUndefsIn(Clones);
    updateCallSites(&F, Specializations);
  }

  promoteConstantStackValues();

  LLVM_DEBUG(if (NbFunctionsSpecialized) dbgs()
             << "FnSpecialization: Specialized " << NbFunctionsSpecialized
             << " functions in module " << M.getName() << "\n");

  NumFuncSpecialized += NbFunctionsSpecialized;
  return Changed;
}

void FunctionSpecializer::removeDeadFunctions() {
  for (Function *F : FullySpecialized) {
    LLVM_DEBUG(dbgs() << "FnSpecialization: Removing dead function "
                      << F->getName() << "\n");
    if (FAM)
      FAM->clear(*F, F->getName());
    F->eraseFromParent();
  }
  FullySpecialized.clear();
}

// Compute the code metrics for function \p F.
CodeMetrics &FunctionSpecializer::analyzeFunction(Function *F) {
  auto I = FunctionMetrics.insert({F, CodeMetrics()});
  CodeMetrics &Metrics = I.first->second;
  if (I.second) {
    // The code metrics were not cached.
    SmallPtrSet<const Value *, 32> EphValues;
    CodeMetrics::collectEphemeralValues(F, &(GetAC)(*F), EphValues);
    for (BasicBlock &BB : *F)
      Metrics.analyzeBasicBlock(&BB, (GetTTI)(*F), EphValues);

    LLVM_DEBUG(dbgs() << "FnSpecialization: Code size of function "
                      << F->getName() << " is " << Metrics.NumInsts
                      << " instructions\n");
  }
  return Metrics;
}

/// Clone the function \p F and remove the ssa_copy intrinsics added by
/// the SCCPSolver in the cloned version.
static Function *cloneCandidateFunction(Function *F) {
  ValueToValueMapTy Mappings;
  Function *Clone = CloneFunction(F, Mappings);
  removeSSACopy(*Clone);
  return Clone;
}

/// This function decides whether it's worthwhile to specialize function
/// \p F based on the known constant values its arguments can take on. It
/// only discovers potential specialization opportunities without actually
/// applying them.
///
/// \returns true if any specializations have been found.
bool FunctionSpecializer::findSpecializations(
    Function *F, InstructionCost Cost,
    SmallVectorImpl<CallSpecBinding> &WorkList) {
  // Get a list of interesting arguments.
  SmallVector<Argument *, 4> Args;
  for (Argument &Arg : F->args())
    if (isArgumentInteresting(&Arg))
      Args.push_back(&Arg);

  if (!Args.size())
    return false;

  // Find all the call sites for the function.
  SpecializationMap Specializations;
  for (User *U : F->users()) {
    if (!isa<CallInst>(U) && !isa<InvokeInst>(U))
      continue;
    auto &CS = *cast<CallBase>(U);

    // Skip irrelevant users.
    if (CS.getCalledFunction() != F)
      continue;

    // If the call site has attribute minsize set, that callsite won't be
    // specialized.
    if (CS.hasFnAttr(Attribute::MinSize))
      continue;

    // If the parent of the call site will never be executed, we don't need
    // to worry about the passed value.
    if (!Solver.isBlockExecutable(CS.getParent()))
      continue;

    // Examine arguments and create specialization candidates from call sites
    // with constant arguments.
    bool Added = false;
    for (Argument *A : Args) {
      Constant *C = getCandidateConstant(CS.getArgOperand(A->getArgNo()));
      if (!C)
        continue;

      if (!Added) {
        Specializations[&CS] = {{}, 0 - Cost, nullptr};
        Added = true;
      }

      SpecializationInfo &S = Specializations.back().second;
      S.Gain += getSpecializationBonus(A, C, Solver.getLoopInfo(*F));
      S.Args.push_back({A, C});
    }
    Added = false;
  }

  // Remove unprofitable specializations.
  if (!ForceFunctionSpecialization)
    Specializations.remove_if(
        [](const auto &Entry) { return Entry.second.Gain <= 0; });

  // Clear the MapVector and return the underlying vector.
  WorkList = Specializations.takeVector();

  // Sort the candidates in descending order.
  llvm::stable_sort(WorkList, [](const auto &L, const auto &R) {
    return L.second.Gain > R.second.Gain;
  });

  // Truncate the worklist to 'MaxClonesThreshold' candidates if necessary.
  if (WorkList.size() > MaxClonesThreshold) {
    LLVM_DEBUG(dbgs() << "FnSpecialization: Number of candidates exceed "
                      << "the maximum number of clones threshold.\n"
                      << "FnSpecialization: Truncating worklist to "
                      << MaxClonesThreshold << " candidates.\n");
    WorkList.erase(WorkList.begin() + MaxClonesThreshold, WorkList.end());
  }

  LLVM_DEBUG(dbgs() << "FnSpecialization: Specializations for function "
                    << F->getName() << "\n";
             for (const auto &Entry
                  : WorkList) {
               dbgs() << "FnSpecialization:   Gain = " << Entry.second.Gain
                      << "\n";
               for (const ArgInfo &Arg : Entry.second.Args)
                 dbgs() << "FnSpecialization:   FormalArg = "
                        << Arg.Formal->getNameOrAsOperand()
                        << ", ActualArg = " << Arg.Actual->getNameOrAsOperand()
                        << "\n";
             });

  return !WorkList.empty();
}

bool FunctionSpecializer::isCandidateFunction(Function *F) {
  if (F->isDeclaration())
    return false;

  if (F->hasFnAttribute(Attribute::NoDuplicate))
    return false;

  if (!Solver.isArgumentTrackedFunction(F))
    return false;

  // Do not specialize the cloned function again.
  if (SpecializedFuncs.contains(F))
    return false;

  // If we're optimizing the function for size, we shouldn't specialize it.
  if (F->hasOptSize() ||
      shouldOptimizeForSize(F, nullptr, nullptr, PGSOQueryType::IRPass))
    return false;

  // Exit if the function is not executable. There's no point in specializing
  // a dead function.
  if (!Solver.isBlockExecutable(&F->getEntryBlock()))
    return false;

  // It wastes time to specialize a function which would get inlined finally.
  if (F->hasFnAttribute(Attribute::AlwaysInline))
    return false;

  LLVM_DEBUG(dbgs() << "FnSpecialization: Try function: " << F->getName()
                    << "\n");
  return true;
}

Function *
FunctionSpecializer::createSpecialization(Function *F,
                                          CallSpecBinding &Specialization) {
  Function *Clone = cloneCandidateFunction(F);
  Specialization.second.Clone = Clone;

  // Initialize the lattice state of the arguments of the function clone,
  // marking the argument on which we specialized the function constant
  // with the given value.
  Solver.markArgInFuncSpecialization(Clone, Specialization.second.Args);

  Solver.addArgumentTrackedFunction(Clone);
  Solver.markBlockExecutable(&Clone->front());

  // Mark all the specialized functions
  SpecializedFuncs.insert(Clone);
  NbFunctionsSpecialized++;

  return Clone;
}

/// Compute and return the cost of specializing function \p F.
InstructionCost FunctionSpecializer::getSpecializationCost(Function *F) {
  CodeMetrics &Metrics = analyzeFunction(F);
  // If the code metrics reveal that we shouldn't duplicate the function, we
  // shouldn't specialize it. Set the specialization cost to Invalid.
  // Or if the lines of codes implies that this function is easy to get
  // inlined so that we shouldn't specialize it.
  if (Metrics.notDuplicatable || !Metrics.NumInsts.isValid() ||
      (!ForceFunctionSpecialization &&
       !F->hasFnAttribute(Attribute::NoInline) &&
       Metrics.NumInsts < SmallFunctionThreshold))
    return InstructionCost::getInvalid();

  // Otherwise, set the specialization cost to be the cost of all the
  // instructions in the function and penalty for specializing more functions.
  unsigned Penalty = NbFunctionsSpecialized + 1;
  return Metrics.NumInsts * InlineConstants::getInstrCost() * Penalty;
}

static InstructionCost getUserBonus(User *U, llvm::TargetTransformInfo &TTI,
                                    const LoopInfo &LI) {
  auto *I = dyn_cast_or_null<Instruction>(U);
  // If not an instruction we do not know how to evaluate.
  // Keep minimum possible cost for now so that it doesnt affect
  // specialization.
  if (!I)
    return std::numeric_limits<unsigned>::min();

  InstructionCost Cost =
      TTI.getInstructionCost(U, TargetTransformInfo::TCK_SizeAndLatency);

  // Increase the cost if it is inside the loop.
  unsigned LoopDepth = LI.getLoopDepth(I->getParent());
  Cost *= std::pow((double)AvgLoopIterationCount, LoopDepth);

  // Traverse recursively if there are more uses.
  // TODO: Any other instructions to be added here?
  if (I->mayReadFromMemory() || I->isCast())
    for (auto *User : I->users())
      Cost += getUserBonus(User, TTI, LI);

  return Cost;
}

/// Compute a bonus for replacing argument \p A with constant \p C.
InstructionCost
FunctionSpecializer::getSpecializationBonus(Argument *A, Constant *C,
                                            const LoopInfo &LI) {
  Function *F = A->getParent();
  auto &TTI = (GetTTI)(*F);
  LLVM_DEBUG(dbgs() << "FnSpecialization: Analysing bonus for constant: "
                    << C->getNameOrAsOperand() << "\n");

  InstructionCost TotalCost = 0;
  for (auto *U : A->users()) {
    TotalCost += getUserBonus(U, TTI, LI);
    LLVM_DEBUG(dbgs() << "FnSpecialization:   User cost ";
               TotalCost.print(dbgs()); dbgs() << " for: " << *U << "\n");
  }

  // The below heuristic is only concerned with exposing inlining
  // opportunities via indirect call promotion. If the argument is not a
  // (potentially casted) function pointer, give up.
  Function *CalledFunction = dyn_cast<Function>(C->stripPointerCasts());
  if (!CalledFunction)
    return TotalCost;

  // Get TTI for the called function (used for the inline cost).
  auto &CalleeTTI = (GetTTI)(*CalledFunction);

  // Look at all the call sites whose called value is the argument.
  // Specializing the function on the argument would allow these indirect
  // calls to be promoted to direct calls. If the indirect call promotion
  // would likely enable the called function to be inlined, specializing is a
  // good idea.
  int Bonus = 0;
  for (User *U : A->users()) {
    if (!isa<CallInst>(U) && !isa<InvokeInst>(U))
      continue;
    auto *CS = cast<CallBase>(U);
    if (CS->getCalledOperand() != A)
      continue;

    // Get the cost of inlining the called function at this call site. Note
    // that this is only an estimate. The called function may eventually
    // change in a way that leads to it not being inlined here, even though
    // inlining looks profitable now. For example, one of its called
    // functions may be inlined into it, making the called function too large
    // to be inlined into this call site.
    //
    // We apply a boost for performing indirect call promotion by increasing
    // the default threshold by the threshold for indirect calls.
    auto Params = getInlineParams();
    Params.DefaultThreshold += InlineConstants::IndirectCallThreshold;
    InlineCost IC =
        getInlineCost(*CS, CalledFunction, Params, CalleeTTI, GetAC, GetTLI);

    // We clamp the bonus for this call to be between zero and the default
    // threshold.
    if (IC.isAlways())
      Bonus += Params.DefaultThreshold;
    else if (IC.isVariable() && IC.getCostDelta() > 0)
      Bonus += IC.getCostDelta();

    LLVM_DEBUG(dbgs() << "FnSpecialization:   Inlining bonus " << Bonus
                      << " for user " << *U << "\n");
  }

  return TotalCost + Bonus;
}

/// Determine if it is possible to specialise the function for constant values
/// of the formal parameter \p A.
bool FunctionSpecializer::isArgumentInteresting(Argument *A) {
  // No point in specialization if the argument is unused.
  if (A->user_empty())
    return false;

  // For now, don't attempt to specialize functions based on the values of
  // composite types.
  Type *ArgTy = A->getType();
  if (!ArgTy->isSingleValueType())
    return false;

  // Specialization of integer and floating point types needs to be explicitly
  // enabled.
  if (!EnableSpecializationForLiteralConstant &&
      (ArgTy->isIntegerTy() || ArgTy->isFloatingPointTy()))
    return false;

  // SCCP solver does not record an argument that will be constructed on
  // stack.
  if (A->hasByValAttr() && !A->getParent()->onlyReadsMemory())
    return false;

  // Check the lattice value and decide if we should attemt to specialize,
  // based on this argument. No point in specialization, if the lattice value
  // is already a constant.
  const ValueLatticeElement &LV = Solver.getLatticeValueFor(A);
  if (LV.isUnknownOrUndef() || LV.isConstant() ||
      (LV.isConstantRange() && LV.getConstantRange().isSingleElement())) {
    LLVM_DEBUG(dbgs() << "FnSpecialization: Nothing to do, argument "
                      << A->getNameOrAsOperand() << " is already constant\n");
    return false;
  }

  return true;
}

/// Check if the valuy \p V  (an actual argument) is a constant or can only
/// have a constant value. Return that constant.
Constant *FunctionSpecializer::getCandidateConstant(Value *V) {
  if (isa<PoisonValue>(V))
    return nullptr;

  // TrackValueOfGlobalVariable only tracks scalar global variables.
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    // Check if we want to specialize on the address of non-constant
    // global values.
    if (!GV->isConstant() && !SpecializeOnAddresses)
      return nullptr;

    if (!GV->getValueType()->isSingleValueType())
      return nullptr;
  }

  // Select for possible specialisation values that are constants or
  // are deduced to be constants or constant ranges with a single element.
  Constant *C = dyn_cast<Constant>(V);
  if (!C) {
    const ValueLatticeElement &LV = Solver.getLatticeValueFor(V);
    if (LV.isConstant())
      C = LV.getConstant();
    else if (LV.isConstantRange() && LV.getConstantRange().isSingleElement()) {
      assert(V->getType()->isIntegerTy() && "Non-integral constant range");
      C = Constant::getIntegerValue(V->getType(),
                                    *LV.getConstantRange().getSingleElement());
    } else
      return nullptr;
  }

  LLVM_DEBUG(dbgs() << "FnSpecialization: Found interesting argument "
                    << V->getNameOrAsOperand() << "\n");

  return C;
}

/// Redirects callsites of function \p F to its specialized copies.
void FunctionSpecializer::updateCallSites(
    Function *F, SmallVectorImpl<CallSpecBinding> &Specializations) {
  SmallVector<CallBase *, 8> ToUpdate;
  for (User *U : F->users()) {
    if (auto *CS = dyn_cast<CallBase>(U))
      if (CS->getCalledFunction() == F &&
          Solver.isBlockExecutable(CS->getParent()))
        ToUpdate.push_back(CS);
  }

  unsigned NCallsLeft = ToUpdate.size();
  for (CallBase *CS : ToUpdate) {
    // Decrement the counter if the callsite is either recursive or updated.
    bool ShouldDecrementCount = CS->getFunction() == F;
    for (CallSpecBinding &Specialization : Specializations) {
      Function *Clone = Specialization.second.Clone;
      SmallVectorImpl<ArgInfo> &Args = Specialization.second.Args;

      if (any_of(Args, [CS, this](const ArgInfo &Arg) {
            unsigned ArgNo = Arg.Formal->getArgNo();
            return getCandidateConstant(CS->getArgOperand(ArgNo)) != Arg.Actual;
          }))
        continue;

      LLVM_DEBUG(dbgs() << "FnSpecialization: Replacing call site " << *CS
                        << " with " << Clone->getName() << "\n");

      CS->setCalledFunction(Clone);
      ShouldDecrementCount = true;
      break;
    }
    if (ShouldDecrementCount)
      --NCallsLeft;
  }

  // If the function has been completely specialized, the original function
  // is no longer needed. Mark it unreachable.
  if (NCallsLeft == 0) {
    Solver.markFunctionUnreachable(F);
    FullySpecialized.insert(F);
  }
}
