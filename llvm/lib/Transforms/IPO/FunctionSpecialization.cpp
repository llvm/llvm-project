//===- FunctionSpecialization.cpp - Function Specialization ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This specialises functions with constant parameters (e.g. functions,
// globals). Constant parameters like function pointers and constant globals
// are propagated to the callee by specializing the function.
//
// Current limitations:
// - It does not yet handle integer ranges.
// - Only 1 argument per function is specialised,
// - The cost-model could be further looked into,
// - We are not yet caching analysis results.
//
// Ideas:
// - With a function specialization attribute for arguments, we could have
//   a direct way to steer function specialization, avoiding the cost-model,
//   and thus control compile-times / code-size.
//
// Todos:
// - Limit the times a recursive function get specialized when
// `func-specialization-max-iters`
//   increases linearly. See discussion in https://reviews.llvm.org/D106426 for
//   details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SizeOpts.h"
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "function-specialization"

STATISTIC(NumFuncSpecialized, "Number of functions specialized");

static cl::opt<bool> ForceFunctionSpecialization(
    "force-function-specialization", cl::init(false), cl::Hidden,
    cl::desc("Force function specialization for every call site with a "
             "constant argument"));

static cl::opt<unsigned> FuncSpecializationMaxIters(
    "func-specialization-max-iters", cl::Hidden,
    cl::desc("The maximum number of iterations function specialization is run"),
    cl::init(1));

static cl::opt<unsigned> MaxConstantsThreshold(
    "func-specialization-max-constants", cl::Hidden,
    cl::desc("The maximum number of clones allowed for a single function "
             "specialization"),
    cl::init(3));

static cl::opt<unsigned>
    AvgLoopIterationCount("func-specialization-avg-iters-cost", cl::Hidden,
                          cl::desc("Average loop iteration count cost"),
                          cl::init(10));

static cl::opt<bool> EnableSpecializationForLiteralConstant(
    "function-specialization-for-literal-constant", cl::init(false), cl::Hidden,
    cl::desc("Make function specialization available for literal constant."));

// Helper to check if \p LV is either a constant or a constant
// range with a single element. This should cover exactly the same cases as the
// old ValueLatticeElement::isConstant() and is intended to be used in the
// transition to ValueLatticeElement.
static bool isConstant(const ValueLatticeElement &LV) {
  return LV.isConstant() ||
         (LV.isConstantRange() && LV.getConstantRange().isSingleElement());
}

// Helper to check if \p LV is either overdefined or a constant int.
static bool isOverdefined(const ValueLatticeElement &LV) {
  return !LV.isUnknownOrUndef() && !isConstant(LV);
}

static Constant *getPromotableAlloca(AllocaInst *Alloca, CallInst *Call) {
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
  return dyn_cast_or_null<Constant>(StoreValue);
}

// A constant stack value is an AllocaInst that has a single constant
// value stored to it. Return this constant if such an alloca stack value
// is a function argument.
static Constant *getConstantStackValue(CallInst *Call, Value *Val,
                                       SCCPSolver &Solver) {
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
static void constantArgPropagation(SmallVectorImpl<Function *> &WorkList,
                                   Module &M, SCCPSolver &Solver) {
  // Iterate over the argument tracked functions see if there
  // are any new constant values for the call instruction via
  // stack variables.
  for (auto *F : WorkList) {
    // TODO: Generalize for any read only arguments.
    if (F->arg_size() != 1)
      continue;

    auto &Arg = *F->arg_begin();
    if (!Arg.onlyReadsMemory() || !Arg.getType()->isPointerTy())
      continue;

    for (auto *User : F->users()) {
      auto *Call = dyn_cast<CallInst>(User);
      if (!Call)
        break;
      auto *ArgOp = Call->getArgOperand(0);
      auto *ArgOpType = ArgOp->getType();
      auto *ConstVal = getConstantStackValue(Call, ArgOp, Solver);
      if (!ConstVal)
        break;

      Value *GV = new GlobalVariable(M, ConstVal->getType(), true,
                                     GlobalValue::InternalLinkage, ConstVal,
                                     "funcspec.arg");

      if (ArgOpType != ConstVal->getType())
        GV = ConstantExpr::getBitCast(cast<Constant>(GV), ArgOp->getType());

      Call->setArgOperand(0, GV);

      // Add the changed CallInst to Solver Worklist
      Solver.visitCall(*Call);
    }
  }
}

// ssa_copy intrinsics are introduced by the SCCP solver. These intrinsics
// interfere with the constantArgPropagation optimization.
static void removeSSACopy(Function &F) {
  for (BasicBlock &BB : F) {
    for (BasicBlock::iterator BI = BB.begin(), E = BB.end(); BI != E;) {
      Instruction *Inst = &*BI++;
      auto *II = dyn_cast<IntrinsicInst>(Inst);
      if (!II)
        continue;
      if (II->getIntrinsicID() != Intrinsic::ssa_copy)
        continue;
      Inst->replaceAllUsesWith(II->getOperand(0));
      Inst->eraseFromParent();
    }
  }
}

static void removeSSACopy(Module &M) {
  for (Function &F : M)
    removeSSACopy(F);
}

class FunctionSpecializer {

  /// The IPSCCP Solver.
  SCCPSolver &Solver;

  /// Analyses used to help determine if a function should be specialized.
  std::function<AssumptionCache &(Function &)> GetAC;
  std::function<TargetTransformInfo &(Function &)> GetTTI;
  std::function<TargetLibraryInfo &(Function &)> GetTLI;

  SmallPtrSet<Function *, 2> SpecializedFuncs;

public:
  FunctionSpecializer(SCCPSolver &Solver,
                      std::function<AssumptionCache &(Function &)> GetAC,
                      std::function<TargetTransformInfo &(Function &)> GetTTI,
                      std::function<TargetLibraryInfo &(Function &)> GetTLI)
      : Solver(Solver), GetAC(GetAC), GetTTI(GetTTI), GetTLI(GetTLI) {}

  /// Attempt to specialize functions in the module to enable constant
  /// propagation across function boundaries.
  ///
  /// \returns true if at least one function is specialized.
  bool
  specializeFunctions(SmallVectorImpl<Function *> &FuncDecls,
                      SmallVectorImpl<Function *> &CurrentSpecializations) {

    // Attempt to specialize the argument-tracked functions.
    bool Changed = false;
    for (auto *F : FuncDecls) {
      if (specializeFunction(F, CurrentSpecializations)) {
        Changed = true;
        LLVM_DEBUG(dbgs() << "FnSpecialization: Can specialize this func.\n");
      } else {
        LLVM_DEBUG(
            dbgs() << "FnSpecialization: Cannot specialize this func.\n");
      }
    }

    for (auto *SpecializedFunc : CurrentSpecializations) {
      SpecializedFuncs.insert(SpecializedFunc);

      // Initialize the state of the newly created functions, marking them
      // argument-tracked and executable.
      if (SpecializedFunc->hasExactDefinition() &&
          !SpecializedFunc->hasFnAttribute(Attribute::Naked))
        Solver.addTrackedFunction(SpecializedFunc);
      Solver.addArgumentTrackedFunction(SpecializedFunc);
      FuncDecls.push_back(SpecializedFunc);
      Solver.markBlockExecutable(&SpecializedFunc->front());

      // Replace the function arguments for the specialized functions.
      for (Argument &Arg : SpecializedFunc->args())
        if (!Arg.use_empty() && tryToReplaceWithConstant(&Arg))
          LLVM_DEBUG(dbgs() << "FnSpecialization: Replaced constant argument: "
                            << Arg.getName() << "\n");
    }

    NumFuncSpecialized += NbFunctionsSpecialized;
    return Changed;
  }

  bool tryToReplaceWithConstant(Value *V) {
    if (!V->getType()->isSingleValueType() || isa<CallBase>(V) ||
        V->user_empty())
      return false;

    const ValueLatticeElement &IV = Solver.getLatticeValueFor(V);
    if (isOverdefined(IV))
      return false;
    auto *Const =
        isConstant(IV) ? Solver.getConstant(IV) : UndefValue::get(V->getType());
    V->replaceAllUsesWith(Const);

    for (auto *U : Const->users())
      if (auto *I = dyn_cast<Instruction>(U))
        if (Solver.isBlockExecutable(I->getParent()))
          Solver.visit(I);

    // Remove the instruction from Block and Solver.
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (I->isSafeToRemove()) {
        I->eraseFromParent();
        Solver.removeLatticeValueFor(I);
      }
    }
    return true;
  }

private:
  // The number of functions specialised, used for collecting statistics and
  // also in the cost model.
  unsigned NbFunctionsSpecialized = 0;

  /// Clone the function \p F and remove the ssa_copy intrinsics added by
  /// the SCCPSolver in the cloned version.
  Function *cloneCandidateFunction(Function *F) {
    ValueToValueMapTy EmptyMap;
    Function *Clone = CloneFunction(F, EmptyMap);
    removeSSACopy(*Clone);
    return Clone;
  }

  /// This function decides whether to specialize function \p F based on the
  /// known constant values its arguments can take on. Specialization is
  /// performed on the first interesting argument. Specializations based on
  /// additional arguments will be evaluated on following iterations of the
  /// main IPSCCP solve loop. \returns true if the function is specialized and
  /// false otherwise.
  bool specializeFunction(Function *F,
                          SmallVectorImpl<Function *> &Specializations) {

    // Do not specialize the cloned function again.
    if (SpecializedFuncs.contains(F)) {
      return false;
    }

    // If we're optimizing the function for size, we shouldn't specialize it.
    if (F->hasOptSize() ||
        shouldOptimizeForSize(F, nullptr, nullptr, PGSOQueryType::IRPass))
      return false;

    // Exit if the function is not executable. There's no point in specializing
    // a dead function.
    if (!Solver.isBlockExecutable(&F->getEntryBlock()))
      return false;

    LLVM_DEBUG(dbgs() << "FnSpecialization: Try function: " << F->getName()
                      << "\n");
    // Determine if we should specialize the function based on the values the
    // argument can take on. If specialization is not profitable, we continue
    // on to the next argument.
    for (Argument &A : F->args()) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Analysing arg: " << A.getName()
                        << "\n");
      // True if this will be a partial specialization. We will need to keep
      // the original function around in addition to the added specializations.
      bool IsPartial = true;

      // Determine if this argument is interesting. If we know the argument can
      // take on any constant values, they are collected in Constants. If the
      // argument can only ever equal a constant value in Constants, the
      // function will be completely specialized, and the IsPartial flag will
      // be set to false by isArgumentInteresting (that function only adds
      // values to the Constants list that are deemed profitable).
      SmallVector<Constant *, 4> Constants;
      if (!isArgumentInteresting(&A, Constants, IsPartial)) {
        LLVM_DEBUG(dbgs() << "FnSpecialization: Argument is not interesting\n");
        continue;
      }

      assert(!Constants.empty() && "No constants on which to specialize");
      LLVM_DEBUG(dbgs() << "FnSpecialization: Argument is interesting!\n"
                        << "FnSpecialization: Specializing '" << F->getName()
                        << "' on argument: " << A << "\n"
                        << "FnSpecialization: Constants are:\n\n";
                 for (unsigned I = 0; I < Constants.size(); ++I) dbgs()
                 << *Constants[I] << "\n";
                 dbgs() << "FnSpecialization: End of constants\n\n");

      // Create a version of the function in which the argument is marked
      // constant with the given value.
      for (auto *C : Constants) {
        // Clone the function. We leave the ValueToValueMap empty to allow
        // IPSCCP to propagate the constant arguments.
        Function *Clone = cloneCandidateFunction(F);
        Argument *ClonedArg = Clone->arg_begin() + A.getArgNo();

        // Rewrite calls to the function so that they call the clone instead.
        rewriteCallSites(F, Clone, *ClonedArg, C);

        // Initialize the lattice state of the arguments of the function clone,
        // marking the argument on which we specialized the function constant
        // with the given value.
        Solver.markArgInFuncSpecialization(F, ClonedArg, C);

        // Mark all the specialized functions
        Specializations.push_back(Clone);
        NbFunctionsSpecialized++;
      }

      // If the function has been completely specialized, the original function
      // is no longer needed. Mark it unreachable.
      if (!IsPartial)
        Solver.markFunctionUnreachable(F);

      // FIXME: Only one argument per function.
      return true;
    }

    return false;
  }

  /// Compute the cost of specializing function \p F.
  InstructionCost getSpecializationCost(Function *F) {
    // Compute the code metrics for the function.
    SmallPtrSet<const Value *, 32> EphValues;
    CodeMetrics::collectEphemeralValues(F, &(GetAC)(*F), EphValues);
    CodeMetrics Metrics;
    for (BasicBlock &BB : *F)
      Metrics.analyzeBasicBlock(&BB, (GetTTI)(*F), EphValues);

    // If the code metrics reveal that we shouldn't duplicate the function, we
    // shouldn't specialize it. Set the specialization cost to Invalid.
    if (Metrics.notDuplicatable) {
      InstructionCost C{};
      C.setInvalid();
      return C;
    }

    // Otherwise, set the specialization cost to be the cost of all the
    // instructions in the function and penalty for specializing more functions.
    unsigned Penalty = NbFunctionsSpecialized + 1;
    return Metrics.NumInsts * InlineConstants::InstrCost * Penalty;
  }

  InstructionCost getUserBonus(User *U, llvm::TargetTransformInfo &TTI,
                               LoopInfo &LI) {
    auto *I = dyn_cast_or_null<Instruction>(U);
    // If not an instruction we do not know how to evaluate.
    // Keep minimum possible cost for now so that it doesnt affect
    // specialization.
    if (!I)
      return std::numeric_limits<unsigned>::min();

    auto Cost = TTI.getUserCost(U, TargetTransformInfo::TCK_SizeAndLatency);

    // Traverse recursively if there are more uses.
    // TODO: Any other instructions to be added here?
    if (I->mayReadFromMemory() || I->isCast())
      for (auto *User : I->users())
        Cost += getUserBonus(User, TTI, LI);

    // Increase the cost if it is inside the loop.
    auto LoopDepth = LI.getLoopDepth(I->getParent());
    Cost *= std::pow((double)AvgLoopIterationCount, LoopDepth);
    return Cost;
  }

  /// Compute a bonus for replacing argument \p A with constant \p C.
  InstructionCost getSpecializationBonus(Argument *A, Constant *C) {
    Function *F = A->getParent();
    DominatorTree DT(*F);
    LoopInfo LI(DT);
    auto &TTI = (GetTTI)(*F);
    LLVM_DEBUG(dbgs() << "FnSpecialization: Analysing bonus for: " << *A
                      << "\n");

    InstructionCost TotalCost = 0;
    for (auto *U : A->users()) {
      TotalCost += getUserBonus(U, TTI, LI);
      LLVM_DEBUG(dbgs() << "FnSpecialization: User cost ";
                 TotalCost.print(dbgs()); dbgs() << " for: " << *U << "\n");
    }

    // The below heuristic is only concerned with exposing inlining
    // opportunities via indirect call promotion. If the argument is not a
    // function pointer, give up.
    if (!isa<PointerType>(A->getType()) ||
        !isa<FunctionType>(A->getType()->getPointerElementType()))
      return TotalCost;

    // Since the argument is a function pointer, its incoming constant values
    // should be functions or constant expressions. The code below attempts to
    // look through cast expressions to find the function that will be called.
    Value *CalledValue = C;
    while (isa<ConstantExpr>(CalledValue) &&
           cast<ConstantExpr>(CalledValue)->isCast())
      CalledValue = cast<User>(CalledValue)->getOperand(0);
    Function *CalledFunction = dyn_cast<Function>(CalledValue);
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
    }

    return TotalCost + Bonus;
  }

  /// Determine if we should specialize a function based on the incoming values
  /// of the given argument.
  ///
  /// This function implements the goal-directed heuristic. It determines if
  /// specializing the function based on the incoming values of argument \p A
  /// would result in any significant optimization opportunities. If
  /// optimization opportunities exist, the constant values of \p A on which to
  /// specialize the function are collected in \p Constants. If the values in
  /// \p Constants represent the complete set of values that \p A can take on,
  /// the function will be completely specialized, and the \p IsPartial flag is
  /// set to false.
  ///
  /// \returns true if the function should be specialized on the given
  /// argument.
  bool isArgumentInteresting(Argument *A,
                             SmallVectorImpl<Constant *> &Constants,
                             bool &IsPartial) {
    Function *F = A->getParent();

    // For now, don't attempt to specialize functions based on the values of
    // composite types.
    if (!A->getType()->isSingleValueType() || A->user_empty())
      return false;

    // If the argument isn't overdefined, there's nothing to do. It should
    // already be constant.
    if (!Solver.getLatticeValueFor(A).isOverdefined()) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: nothing to do, arg is already "
                        << "constant?\n");
      return false;
    }

    // Collect the constant values that the argument can take on. If the
    // argument can't take on any constant values, we aren't going to
    // specialize the function. While it's possible to specialize the function
    // based on non-constant arguments, there's likely not much benefit to
    // constant propagation in doing so.
    //
    // TODO 1: currently it won't specialize if there are over the threshold of
    // calls using the same argument, e.g foo(a) x 4 and foo(b) x 1, but it
    // might be beneficial to take the occurrences into account in the cost
    // model, so we would need to find the unique constants.
    //
    // TODO 2: this currently does not support constants, i.e. integer ranges.
    //
    SmallVector<Constant *, 4> PossibleConstants;
    bool AllConstant = getPossibleConstants(A, PossibleConstants);
    if (PossibleConstants.empty()) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: no possible constants found\n");
      return false;
    }
    if (PossibleConstants.size() > MaxConstantsThreshold) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: number of constants found exceed "
                        << "the maximum number of constants threshold.\n");
      return false;
    }

    // Determine if it would be profitable to create a specialization of the
    // function where the argument takes on the given constant value. If so,
    // add the constant to Constants.
    auto FnSpecCost = getSpecializationCost(F);
    if (!FnSpecCost.isValid()) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Invalid specialisation cost.\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "FnSpecialization: func specialisation cost: ";
               FnSpecCost.print(dbgs()); dbgs() << "\n");

    for (auto *C : PossibleConstants) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Constant: " << *C << "\n");
      if (ForceFunctionSpecialization) {
        LLVM_DEBUG(dbgs() << "FnSpecialization: Forced!\n");
        Constants.push_back(C);
        continue;
      }
      if (getSpecializationBonus(A, C) > FnSpecCost) {
        LLVM_DEBUG(dbgs() << "FnSpecialization: profitable!\n");
        Constants.push_back(C);
      } else {
        LLVM_DEBUG(dbgs() << "FnSpecialization: not profitable\n");
      }
    }

    // None of the constant values the argument can take on were deemed good
    // candidates on which to specialize the function.
    if (Constants.empty())
      return false;

    // This will be a partial specialization if some of the constants were
    // rejected due to their profitability.
    IsPartial = !AllConstant || PossibleConstants.size() != Constants.size();

    return true;
  }

  /// Collect in \p Constants all the constant values that argument \p A can
  /// take on.
  ///
  /// \returns true if all of the values the argument can take on are constant
  /// (e.g., the argument's parent function cannot be called with an
  /// overdefined value).
  bool getPossibleConstants(Argument *A,
                            SmallVectorImpl<Constant *> &Constants) {
    Function *F = A->getParent();
    bool AllConstant = true;

    // Iterate over all the call sites of the argument's parent function.
    for (User *U : F->users()) {
      if (!isa<CallInst>(U) && !isa<InvokeInst>(U))
        continue;
      auto &CS = *cast<CallBase>(U);

      // If the parent of the call site will never be executed, we don't need
      // to worry about the passed value.
      if (!Solver.isBlockExecutable(CS.getParent()))
        continue;

      auto *V = CS.getArgOperand(A->getArgNo());
      // TrackValueOfGlobalVariable only tracks scalar global variables.
      if (auto *GV = dyn_cast<GlobalVariable>(V)) {
        if (!GV->getValueType()->isSingleValueType()) {
          return false;
        }
      }

      if (isa<Constant>(V) && (Solver.getLatticeValueFor(V).isConstant() ||
                               EnableSpecializationForLiteralConstant))
        Constants.push_back(cast<Constant>(V));
      else
        AllConstant = false;
    }

    // If the argument can only take on constant values, AllConstant will be
    // true.
    return AllConstant;
  }

  /// Rewrite calls to function \p F to call function \p Clone instead.
  ///
  /// This function modifies calls to function \p F whose argument at index \p
  /// ArgNo is equal to constant \p C. The calls are rewritten to call function
  /// \p Clone instead.
  void rewriteCallSites(Function *F, Function *Clone, Argument &Arg,
                        Constant *C) {
    unsigned ArgNo = Arg.getArgNo();
    SmallVector<CallBase *, 4> CallSitesToRewrite;
    for (auto *U : F->users()) {
      if (!isa<CallInst>(U) && !isa<InvokeInst>(U))
        continue;
      auto &CS = *cast<CallBase>(U);
      if (!CS.getCalledFunction() || CS.getCalledFunction() != F)
        continue;
      CallSitesToRewrite.push_back(&CS);
    }
    for (auto *CS : CallSitesToRewrite) {
      if ((CS->getFunction() == Clone && CS->getArgOperand(ArgNo) == &Arg) ||
          CS->getArgOperand(ArgNo) == C) {
        CS->setCalledFunction(Clone);
        Solver.markOverdefined(CS);
      }
    }
  }
};

bool llvm::runFunctionSpecialization(
    Module &M, const DataLayout &DL,
    std::function<TargetLibraryInfo &(Function &)> GetTLI,
    std::function<TargetTransformInfo &(Function &)> GetTTI,
    std::function<AssumptionCache &(Function &)> GetAC,
    function_ref<AnalysisResultsForFn(Function &)> GetAnalysis) {
  SCCPSolver Solver(DL, GetTLI, M.getContext());
  FunctionSpecializer FS(Solver, GetAC, GetTTI, GetTLI);
  bool Changed = false;

  // Loop over all functions, marking arguments to those with their addresses
  // taken or that are external as overdefined.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (F.hasFnAttribute(Attribute::NoDuplicate))
      continue;

    LLVM_DEBUG(dbgs() << "\nFnSpecialization: Analysing decl: " << F.getName()
                      << "\n");
    Solver.addAnalysis(F, GetAnalysis(F));

    // Determine if we can track the function's arguments. If so, add the
    // function to the solver's set of argument-tracked functions.
    if (canTrackArgumentsInterprocedurally(&F)) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Can track arguments\n");
      Solver.addArgumentTrackedFunction(&F);
      continue;
    } else {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Can't track arguments!\n"
                        << "FnSpecialization: Doesn't have local linkage, or "
                        << "has its address taken\n");
    }

    // Assume the function is called.
    Solver.markBlockExecutable(&F.front());

    // Assume nothing about the incoming arguments.
    for (Argument &AI : F.args())
      Solver.markOverdefined(&AI);
  }

  // Determine if we can track any of the module's global variables. If so, add
  // the global variables we can track to the solver's set of tracked global
  // variables.
  for (GlobalVariable &G : M.globals()) {
    G.removeDeadConstantUsers();
    if (canTrackGlobalVariableInterprocedurally(&G))
      Solver.trackValueOfGlobalVariable(&G);
  }

  // Solve for constants.
  auto RunSCCPSolver = [&](auto &WorkList) {
    bool ResolvedUndefs = true;

    while (ResolvedUndefs) {
      LLVM_DEBUG(dbgs() << "FnSpecialization: Running solver\n");
      Solver.solve();
      LLVM_DEBUG(dbgs() << "FnSpecialization: Resolving undefs\n");
      ResolvedUndefs = false;
      for (Function *F : WorkList)
        if (Solver.resolvedUndefsIn(*F))
          ResolvedUndefs = true;
    }

    for (auto *F : WorkList) {
      for (BasicBlock &BB : *F) {
        if (!Solver.isBlockExecutable(&BB))
          continue;
        for (auto &I : make_early_inc_range(BB))
          FS.tryToReplaceWithConstant(&I);
      }
    }
  };

  auto &TrackedFuncs = Solver.getArgumentTrackedFunctions();
  SmallVector<Function *, 16> FuncDecls(TrackedFuncs.begin(),
                                        TrackedFuncs.end());
#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "FnSpecialization: Worklist fn decls:\n");
  for (auto *F : FuncDecls)
    LLVM_DEBUG(dbgs() << "FnSpecialization: *) " << F->getName() << "\n");
#endif

  // Initially resolve the constants in all the argument tracked functions.
  RunSCCPSolver(FuncDecls);

  SmallVector<Function *, 2> CurrentSpecializations;
  unsigned I = 0;
  while (FuncSpecializationMaxIters != I++ &&
         FS.specializeFunctions(FuncDecls, CurrentSpecializations)) {

    // Run the solver for the specialized functions.
    RunSCCPSolver(CurrentSpecializations);

    // Replace some unresolved constant arguments
    constantArgPropagation(FuncDecls, M, Solver);

    CurrentSpecializations.clear();
    Changed = true;
  }

  // Clean up the IR by removing ssa_copy intrinsics.
  removeSSACopy(M);
  return Changed;
}
