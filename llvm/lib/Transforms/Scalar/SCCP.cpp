//===- SCCP.cpp - Sparse Conditional Constant Propagation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse conditional constant propagation and merging:
//
// Specifically, this:
//   * Assumes values are constant unless proven otherwise
//   * Assumes BasicBlocks are dead unless proven otherwise
//   * Proves values to be constant, and replaces them with constants
//   * Proves conditional branches to be unconditional
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SCCP.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueLattice.h"
#include "llvm/Analysis/ValueLatticeUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"
#include <cassert>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "sccp"

STATISTIC(NumInstRemoved, "Number of instructions removed");
STATISTIC(NumDeadBlocks , "Number of basic blocks unreachable");
STATISTIC(NumInstReplaced,
          "Number of instructions replaced with (simpler) instruction");

STATISTIC(IPNumInstRemoved, "Number of instructions removed by IPSCCP");
STATISTIC(IPNumArgsElimed ,"Number of arguments constant propagated by IPSCCP");
STATISTIC(IPNumGlobalConst, "Number of globals found to be constant by IPSCCP");
STATISTIC(
    IPNumInstReplaced,
    "Number of instructions replaced with (simpler) instruction by IPSCCP");

// Helper to check if \p LV is either a constant or a constant
// range with a single element. This should cover exactly the same cases as the
// old ValueLatticeElement::isConstant() and is intended to be used in the
// transition to ValueLatticeElement.
static bool isConstant(const ValueLatticeElement &LV) {
  return LV.isConstant() ||
         (LV.isConstantRange() && LV.getConstantRange().isSingleElement());
}

// Helper to check if \p LV is either overdefined or a constant range with more
// than a single element. This should cover exactly the same cases as the old
// ValueLatticeElement::isOverdefined() and is intended to be used in the
// transition to ValueLatticeElement.
static bool isOverdefined(const ValueLatticeElement &LV) {
  return !LV.isUnknownOrUndef() && !isConstant(LV);
}

static bool canRemoveInstruction(Instruction *I) {
  if (wouldInstructionBeTriviallyDead(I))
    return true;

  // Some instructions can be handled but are rejected above. Catch
  // those cases by falling through to here.
  // TODO: Mark globals as being constant earlier, so
  // TODO: wouldInstructionBeTriviallyDead() knows that atomic loads
  // TODO: are safe to remove.
  return isa<LoadInst>(I);
}

static bool tryToReplaceWithConstant(SCCPSolver &Solver, Value *V) {
  Constant *Const = nullptr;
  if (V->getType()->isStructTy()) {
    std::vector<ValueLatticeElement> IVs = Solver.getStructLatticeValueFor(V);
    if (llvm::any_of(IVs, isOverdefined))
      return false;
    std::vector<Constant *> ConstVals;
    auto *ST = cast<StructType>(V->getType());
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
      ValueLatticeElement V = IVs[i];
      ConstVals.push_back(isConstant(V)
                              ? Solver.getConstant(V)
                              : UndefValue::get(ST->getElementType(i)));
    }
    Const = ConstantStruct::get(ST, ConstVals);
  } else {
    const ValueLatticeElement &IV = Solver.getLatticeValueFor(V);
    if (isOverdefined(IV))
      return false;

    Const =
        isConstant(IV) ? Solver.getConstant(IV) : UndefValue::get(V->getType());
  }
  assert(Const && "Constant is nullptr here!");

  // Replacing `musttail` instructions with constant breaks `musttail` invariant
  // unless the call itself can be removed.
  // Calls with "clang.arc.attachedcall" implicitly use the return value and
  // those uses cannot be updated with a constant.
  CallBase *CB = dyn_cast<CallBase>(V);
  if (CB && ((CB->isMustTailCall() &&
              !canRemoveInstruction(CB)) ||
             CB->getOperandBundle(LLVMContext::OB_clang_arc_attachedcall))) {
    Function *F = CB->getCalledFunction();

    // Don't zap returns of the callee
    if (F)
      Solver.addToMustPreserveReturnsInFunctions(F);

    LLVM_DEBUG(dbgs() << "  Can\'t treat the result of call " << *CB
                      << " as a constant\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Constant: " << *Const << " = " << *V << '\n');

  // Replaces all of the uses of a variable with uses of the constant.
  V->replaceAllUsesWith(Const);
  return true;
}

static bool simplifyInstsInBlock(SCCPSolver &Solver, BasicBlock &BB,
                                 SmallPtrSetImpl<Value *> &InsertedValues,
                                 Statistic &InstRemovedStat,
                                 Statistic &InstReplacedStat) {
  bool MadeChanges = false;
  for (Instruction &Inst : make_early_inc_range(BB)) {
    if (Inst.getType()->isVoidTy())
      continue;
    if (tryToReplaceWithConstant(Solver, &Inst)) {
      if (canRemoveInstruction(&Inst))
        Inst.eraseFromParent();

      MadeChanges = true;
      ++InstRemovedStat;
    } else if (isa<SExtInst>(&Inst)) {
      Value *ExtOp = Inst.getOperand(0);
      if (isa<Constant>(ExtOp) || InsertedValues.count(ExtOp))
        continue;
      const ValueLatticeElement &IV = Solver.getLatticeValueFor(ExtOp);
      if (!IV.isConstantRange(/*UndefAllowed=*/false))
        continue;
      if (IV.getConstantRange().isAllNonNegative()) {
        auto *ZExt = new ZExtInst(ExtOp, Inst.getType(), "", &Inst);
        ZExt->takeName(&Inst);
        InsertedValues.insert(ZExt);
        Inst.replaceAllUsesWith(ZExt);
        Solver.removeLatticeValueFor(&Inst);
        Inst.eraseFromParent();
        InstReplacedStat++;
        MadeChanges = true;
      }
    }
  }
  return MadeChanges;
}

// runSCCP() - Run the Sparse Conditional Constant Propagation algorithm,
// and return true if the function was modified.
static bool runSCCP(Function &F, const DataLayout &DL,
                    const TargetLibraryInfo *TLI) {
  LLVM_DEBUG(dbgs() << "SCCP on function '" << F.getName() << "'\n");
  SCCPSolver Solver(
      DL, [TLI](Function &F) -> const TargetLibraryInfo & { return *TLI; },
      F.getContext());

  // Mark the first block of the function as being executable.
  Solver.markBlockExecutable(&F.front());

  // Mark all arguments to the function as being overdefined.
  for (Argument &AI : F.args())
    Solver.markOverdefined(&AI);

  // Solve for constants.
  bool ResolvedUndefs = true;
  while (ResolvedUndefs) {
    Solver.solve();
    LLVM_DEBUG(dbgs() << "RESOLVING UNDEFs\n");
    ResolvedUndefs = Solver.resolvedUndefsIn(F);
  }

  bool MadeChanges = false;

  // If we decided that there are basic blocks that are dead in this function,
  // delete their contents now.  Note that we cannot actually delete the blocks,
  // as we cannot modify the CFG of the function.

  SmallPtrSet<Value *, 32> InsertedValues;
  for (BasicBlock &BB : F) {
    if (!Solver.isBlockExecutable(&BB)) {
      LLVM_DEBUG(dbgs() << "  BasicBlock Dead:" << BB);

      ++NumDeadBlocks;
      NumInstRemoved += removeAllNonTerminatorAndEHPadInstructions(&BB).first;

      MadeChanges = true;
      continue;
    }

    MadeChanges |= simplifyInstsInBlock(Solver, BB, InsertedValues,
                                        NumInstRemoved, NumInstReplaced);
  }

  return MadeChanges;
}

PreservedAnalyses SCCPPass::run(Function &F, FunctionAnalysisManager &AM) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  if (!runSCCP(F, DL, &TLI))
    return PreservedAnalyses::all();

  auto PA = PreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

namespace {

//===--------------------------------------------------------------------===//
//
/// SCCP Class - This class uses the SCCPSolver to implement a per-function
/// Sparse Conditional Constant Propagator.
///
class SCCPLegacyPass : public FunctionPass {
public:
  // Pass identification, replacement for typeid
  static char ID;

  SCCPLegacyPass() : FunctionPass(ID) {
    initializeSCCPLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
    AU.setPreservesCFG();
  }

  // runOnFunction - Run the Sparse Conditional Constant Propagation
  // algorithm, and return true if the function was modified.
  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    const DataLayout &DL = F.getParent()->getDataLayout();
    const TargetLibraryInfo *TLI =
        &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    return runSCCP(F, DL, TLI);
  }
};

} // end anonymous namespace

char SCCPLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(SCCPLegacyPass, "sccp",
                      "Sparse Conditional Constant Propagation", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(SCCPLegacyPass, "sccp",
                    "Sparse Conditional Constant Propagation", false, false)

// createSCCPPass - This is the public interface to this file.
FunctionPass *llvm::createSCCPPass() { return new SCCPLegacyPass(); }

static void findReturnsToZap(Function &F,
                             SmallVector<ReturnInst *, 8> &ReturnsToZap,
                             SCCPSolver &Solver) {
  // We can only do this if we know that nothing else can call the function.
  if (!Solver.isArgumentTrackedFunction(&F))
    return;

  if (Solver.mustPreserveReturn(&F)) {
    LLVM_DEBUG(
        dbgs()
        << "Can't zap returns of the function : " << F.getName()
        << " due to present musttail or \"clang.arc.attachedcall\" call of "
           "it\n");
    return;
  }

  assert(
      all_of(F.users(),
             [&Solver](User *U) {
               if (isa<Instruction>(U) &&
                   !Solver.isBlockExecutable(cast<Instruction>(U)->getParent()))
                 return true;
               // Non-callsite uses are not impacted by zapping. Also, constant
               // uses (like blockaddresses) could stuck around, without being
               // used in the underlying IR, meaning we do not have lattice
               // values for them.
               if (!isa<CallBase>(U))
                 return true;
               if (U->getType()->isStructTy()) {
                 return all_of(Solver.getStructLatticeValueFor(U),
                               [](const ValueLatticeElement &LV) {
                                 return !isOverdefined(LV);
                               });
               }
               return !isOverdefined(Solver.getLatticeValueFor(U));
             }) &&
      "We can only zap functions where all live users have a concrete value");

  for (BasicBlock &BB : F) {
    if (CallInst *CI = BB.getTerminatingMustTailCall()) {
      LLVM_DEBUG(dbgs() << "Can't zap return of the block due to present "
                        << "musttail call : " << *CI << "\n");
      (void)CI;
      return;
    }

    if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
      if (!isa<UndefValue>(RI->getOperand(0)))
        ReturnsToZap.push_back(RI);
  }
}

static bool removeNonFeasibleEdges(const SCCPSolver &Solver, BasicBlock *BB,
                                   DomTreeUpdater &DTU,
                                   BasicBlock *&NewUnreachableBB) {
  SmallPtrSet<BasicBlock *, 8> FeasibleSuccessors;
  bool HasNonFeasibleEdges = false;
  for (BasicBlock *Succ : successors(BB)) {
    if (Solver.isEdgeFeasible(BB, Succ))
      FeasibleSuccessors.insert(Succ);
    else
      HasNonFeasibleEdges = true;
  }

  // All edges feasible, nothing to do.
  if (!HasNonFeasibleEdges)
    return false;

  // SCCP can only determine non-feasible edges for br, switch and indirectbr.
  Instruction *TI = BB->getTerminator();
  assert((isa<BranchInst>(TI) || isa<SwitchInst>(TI) ||
          isa<IndirectBrInst>(TI)) &&
         "Terminator must be a br, switch or indirectbr");

  if (FeasibleSuccessors.size() == 1) {
    // Replace with an unconditional branch to the only feasible successor.
    BasicBlock *OnlyFeasibleSuccessor = *FeasibleSuccessors.begin();
    SmallVector<DominatorTree::UpdateType, 8> Updates;
    bool HaveSeenOnlyFeasibleSuccessor = false;
    for (BasicBlock *Succ : successors(BB)) {
      if (Succ == OnlyFeasibleSuccessor && !HaveSeenOnlyFeasibleSuccessor) {
        // Don't remove the edge to the only feasible successor the first time
        // we see it. We still do need to remove any multi-edges to it though.
        HaveSeenOnlyFeasibleSuccessor = true;
        continue;
      }

      Succ->removePredecessor(BB);
      Updates.push_back({DominatorTree::Delete, BB, Succ});
    }

    BranchInst::Create(OnlyFeasibleSuccessor, BB);
    TI->eraseFromParent();
    DTU.applyUpdatesPermissive(Updates);
  } else if (FeasibleSuccessors.size() > 1) {
    SwitchInstProfUpdateWrapper SI(*cast<SwitchInst>(TI));
    SmallVector<DominatorTree::UpdateType, 8> Updates;

    // If the default destination is unfeasible it will never be taken. Replace
    // it with a new block with a single Unreachable instruction.
    BasicBlock *DefaultDest = SI->getDefaultDest();
    if (!FeasibleSuccessors.contains(DefaultDest)) {
      if (!NewUnreachableBB) {
        NewUnreachableBB =
            BasicBlock::Create(DefaultDest->getContext(), "default.unreachable",
                               DefaultDest->getParent(), DefaultDest);
        new UnreachableInst(DefaultDest->getContext(), NewUnreachableBB);
      }

      SI->setDefaultDest(NewUnreachableBB);
      Updates.push_back({DominatorTree::Delete, BB, DefaultDest});
      Updates.push_back({DominatorTree::Insert, BB, NewUnreachableBB});
    }

    for (auto CI = SI->case_begin(); CI != SI->case_end();) {
      if (FeasibleSuccessors.contains(CI->getCaseSuccessor())) {
        ++CI;
        continue;
      }

      BasicBlock *Succ = CI->getCaseSuccessor();
      Succ->removePredecessor(BB);
      Updates.push_back({DominatorTree::Delete, BB, Succ});
      SI.removeCase(CI);
      // Don't increment CI, as we removed a case.
    }

    DTU.applyUpdatesPermissive(Updates);
  } else {
    llvm_unreachable("Must have at least one feasible successor");
  }
  return true;
}

bool llvm::runIPSCCP(
    Module &M, const DataLayout &DL,
    std::function<const TargetLibraryInfo &(Function &)> GetTLI,
    function_ref<AnalysisResultsForFn(Function &)> getAnalysis) {
  SCCPSolver Solver(DL, GetTLI, M.getContext());

  // Loop over all functions, marking arguments to those with their addresses
  // taken or that are external as overdefined.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    Solver.addAnalysis(F, getAnalysis(F));

    // Determine if we can track the function's return values. If so, add the
    // function to the solver's set of return-tracked functions.
    if (canTrackReturnsInterprocedurally(&F))
      Solver.addTrackedFunction(&F);

    // Determine if we can track the function's arguments. If so, add the
    // function to the solver's set of argument-tracked functions.
    if (canTrackArgumentsInterprocedurally(&F)) {
      Solver.addArgumentTrackedFunction(&F);
      continue;
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
  bool ResolvedUndefs = true;
  Solver.solve();
  while (ResolvedUndefs) {
    LLVM_DEBUG(dbgs() << "RESOLVING UNDEFS\n");
    ResolvedUndefs = false;
    for (Function &F : M) {
      if (Solver.resolvedUndefsIn(F))
        ResolvedUndefs = true;
    }
    if (ResolvedUndefs)
      Solver.solve();
  }

  bool MadeChanges = false;

  // Iterate over all of the instructions in the module, replacing them with
  // constants if we have found them to be of constant values.

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    SmallVector<BasicBlock *, 512> BlocksToErase;

    if (Solver.isBlockExecutable(&F.front())) {
      bool ReplacedPointerArg = false;
      for (Argument &Arg : F.args()) {
        if (!Arg.use_empty() && tryToReplaceWithConstant(Solver, &Arg)) {
          ReplacedPointerArg |= Arg.getType()->isPointerTy();
          ++IPNumArgsElimed;
        }
      }

      // If we replaced an argument, the argmemonly and
      // inaccessiblemem_or_argmemonly attributes do not hold any longer. Remove
      // them from both the function and callsites.
      if (ReplacedPointerArg) {
        AttributeMask AttributesToRemove;
        AttributesToRemove.addAttribute(Attribute::ArgMemOnly);
        AttributesToRemove.addAttribute(Attribute::InaccessibleMemOrArgMemOnly);
        F.removeFnAttrs(AttributesToRemove);

        for (User *U : F.users()) {
          auto *CB = dyn_cast<CallBase>(U);
          if (!CB || CB->getCalledFunction() != &F)
            continue;

          CB->removeFnAttrs(AttributesToRemove);
        }
      }
      MadeChanges |= ReplacedPointerArg;
    }

    SmallPtrSet<Value *, 32> InsertedValues;
    for (BasicBlock &BB : F) {
      if (!Solver.isBlockExecutable(&BB)) {
        LLVM_DEBUG(dbgs() << "  BasicBlock Dead:" << BB);
        ++NumDeadBlocks;

        MadeChanges = true;

        if (&BB != &F.front())
          BlocksToErase.push_back(&BB);
        continue;
      }

      MadeChanges |= simplifyInstsInBlock(Solver, BB, InsertedValues,
                                          IPNumInstRemoved, IPNumInstReplaced);
    }

    DomTreeUpdater DTU = Solver.getDTU(F);
    // Change dead blocks to unreachable. We do it after replacing constants
    // in all executable blocks, because changeToUnreachable may remove PHI
    // nodes in executable blocks we found values for. The function's entry
    // block is not part of BlocksToErase, so we have to handle it separately.
    for (BasicBlock *BB : BlocksToErase) {
      NumInstRemoved += changeToUnreachable(BB->getFirstNonPHI(),
                                            /*PreserveLCSSA=*/false, &DTU);
    }
    if (!Solver.isBlockExecutable(&F.front()))
      NumInstRemoved += changeToUnreachable(F.front().getFirstNonPHI(),
                                            /*PreserveLCSSA=*/false, &DTU);

    BasicBlock *NewUnreachableBB = nullptr;
    for (BasicBlock &BB : F)
      MadeChanges |= removeNonFeasibleEdges(Solver, &BB, DTU, NewUnreachableBB);

    for (BasicBlock *DeadBB : BlocksToErase)
      if (!DeadBB->hasAddressTaken())
        DTU.deleteBB(DeadBB);

    for (BasicBlock &BB : F) {
      for (Instruction &Inst : llvm::make_early_inc_range(BB)) {
        if (Solver.getPredicateInfoFor(&Inst)) {
          if (auto *II = dyn_cast<IntrinsicInst>(&Inst)) {
            if (II->getIntrinsicID() == Intrinsic::ssa_copy) {
              Value *Op = II->getOperand(0);
              Inst.replaceAllUsesWith(Op);
              Inst.eraseFromParent();
            }
          }
        }
      }
    }
  }

  // If we inferred constant or undef return values for a function, we replaced
  // all call uses with the inferred value.  This means we don't need to bother
  // actually returning anything from the function.  Replace all return
  // instructions with return undef.
  //
  // Do this in two stages: first identify the functions we should process, then
  // actually zap their returns.  This is important because we can only do this
  // if the address of the function isn't taken.  In cases where a return is the
  // last use of a function, the order of processing functions would affect
  // whether other functions are optimizable.
  SmallVector<ReturnInst*, 8> ReturnsToZap;

  for (const auto &I : Solver.getTrackedRetVals()) {
    Function *F = I.first;
    const ValueLatticeElement &ReturnValue = I.second;

    // If there is a known constant range for the return value, add !range
    // metadata to the function's call sites.
    if (ReturnValue.isConstantRange() &&
        !ReturnValue.getConstantRange().isSingleElement()) {
      // Do not add range metadata if the return value may include undef.
      if (ReturnValue.isConstantRangeIncludingUndef())
        continue;

      auto &CR = ReturnValue.getConstantRange();
      for (User *User : F->users()) {
        auto *CB = dyn_cast<CallBase>(User);
        if (!CB || CB->getCalledFunction() != F)
          continue;

        // Limit to cases where the return value is guaranteed to be neither
        // poison nor undef. Poison will be outside any range and currently
        // values outside of the specified range cause immediate undefined
        // behavior.
        if (!isGuaranteedNotToBeUndefOrPoison(CB, nullptr, CB))
          continue;

        // Do not touch existing metadata for now.
        // TODO: We should be able to take the intersection of the existing
        // metadata and the inferred range.
        if (CB->getMetadata(LLVMContext::MD_range))
          continue;

        LLVMContext &Context = CB->getParent()->getContext();
        Metadata *RangeMD[] = {
            ConstantAsMetadata::get(ConstantInt::get(Context, CR.getLower())),
            ConstantAsMetadata::get(ConstantInt::get(Context, CR.getUpper()))};
        CB->setMetadata(LLVMContext::MD_range, MDNode::get(Context, RangeMD));
      }
      continue;
    }
    if (F->getReturnType()->isVoidTy())
      continue;
    if (isConstant(ReturnValue) || ReturnValue.isUnknownOrUndef())
      findReturnsToZap(*F, ReturnsToZap, Solver);
  }

  for (auto F : Solver.getMRVFunctionsTracked()) {
    assert(F->getReturnType()->isStructTy() &&
           "The return type should be a struct");
    StructType *STy = cast<StructType>(F->getReturnType());
    if (Solver.isStructLatticeConstant(F, STy))
      findReturnsToZap(*F, ReturnsToZap, Solver);
  }

  // Zap all returns which we've identified as zap to change.
  SmallSetVector<Function *, 8> FuncZappedReturn;
  for (unsigned i = 0, e = ReturnsToZap.size(); i != e; ++i) {
    Function *F = ReturnsToZap[i]->getParent()->getParent();
    ReturnsToZap[i]->setOperand(0, UndefValue::get(F->getReturnType()));
    // Record all functions that are zapped.
    FuncZappedReturn.insert(F);
  }

  // Remove the returned attribute for zapped functions and the
  // corresponding call sites.
  for (Function *F : FuncZappedReturn) {
    for (Argument &A : F->args())
      F->removeParamAttr(A.getArgNo(), Attribute::Returned);
    for (Use &U : F->uses()) {
      // Skip over blockaddr users.
      if (isa<BlockAddress>(U.getUser()))
        continue;
      CallBase *CB = cast<CallBase>(U.getUser());
      for (Use &Arg : CB->args())
        CB->removeParamAttr(CB->getArgOperandNo(&Arg), Attribute::Returned);
    }
  }

  // If we inferred constant or undef values for globals variables, we can
  // delete the global and any stores that remain to it.
  for (auto &I : make_early_inc_range(Solver.getTrackedGlobals())) {
    GlobalVariable *GV = I.first;
    if (isOverdefined(I.second))
      continue;
    LLVM_DEBUG(dbgs() << "Found that GV '" << GV->getName()
                      << "' is constant!\n");
    while (!GV->use_empty()) {
      StoreInst *SI = cast<StoreInst>(GV->user_back());
      SI->eraseFromParent();
      MadeChanges = true;
    }
    M.getGlobalList().erase(GV);
    ++IPNumGlobalConst;
  }

  return MadeChanges;
}
