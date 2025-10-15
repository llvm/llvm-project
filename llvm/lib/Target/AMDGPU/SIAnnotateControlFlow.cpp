//===- SIAnnotateControlFlow.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Annotates the control flow with hardware specific intrinsics.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "si-annotate-control-flow"

namespace {

// Complex types used in this pass
using StackEntry = std::pair<BasicBlock *, Value *>;
using StackVector = SmallVector<StackEntry, 16>;

class SIAnnotateControlFlow {
private:
  Function *F;
  UniformityInfo *UA;

  Type *Boolean;
  Type *Void;
  Type *IntMask;
  Type *ReturnStruct;

  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  PoisonValue *BoolPoison;
  Constant *IntMaskZero;

  Function *If = nullptr;
  Function *Else = nullptr;
  Function *IfBreak = nullptr;
  Function *Loop = nullptr;
  Function *EndCf = nullptr;

  DominatorTree *DT;
  StackVector Stack;

  LoopInfo *LI;

  void initialize(const GCNSubtarget &ST);

  bool isUniform(BranchInst *T);

  bool isTopOfStack(BasicBlock *BB);

  Value *popSaved();

  void push(BasicBlock *BB, Value *Saved);

  bool isElse(PHINode *Phi);

  bool hasKill(const BasicBlock *BB);

  bool eraseIfUnused(PHINode *Phi);

  bool openIf(BranchInst *Term);

  bool insertElse(BranchInst *Term);

  Value *
  handleLoopCondition(Value *Cond, PHINode *Broken, llvm::Loop *L,
                      BranchInst *Term);

  bool handleLoop(BranchInst *Term);

  bool closeControlFlow(BasicBlock *BB);

  Function *getDecl(Function *&Cache, Intrinsic::ID ID, ArrayRef<Type *> Tys) {
    if (!Cache)
      Cache = Intrinsic::getOrInsertDeclaration(F->getParent(), ID, Tys);
    return Cache;
  }

public:
  SIAnnotateControlFlow(Function &F, const GCNSubtarget &ST, DominatorTree &DT,
                        LoopInfo &LI, UniformityInfo &UA)
      : F(&F), UA(&UA), DT(&DT), LI(&LI) {
    initialize(ST);
  }

  bool run();
};

} // end anonymous namespace

/// Initialize all the types and constants used in the pass
void SIAnnotateControlFlow::initialize(const GCNSubtarget &ST) {
  LLVMContext &Context = F->getContext();

  Void = Type::getVoidTy(Context);
  Boolean = Type::getInt1Ty(Context);
  IntMask = ST.isWave32() ? Type::getInt32Ty(Context)
                           : Type::getInt64Ty(Context);
  ReturnStruct = StructType::get(Boolean, IntMask);

  BoolTrue = ConstantInt::getTrue(Context);
  BoolFalse = ConstantInt::getFalse(Context);
  BoolPoison = PoisonValue::get(Boolean);
  IntMaskZero = ConstantInt::get(IntMask, 0);
}

/// Is the branch condition uniform or did the StructurizeCFG pass
/// consider it as such?
bool SIAnnotateControlFlow::isUniform(BranchInst *T) {
  return UA->isUniform(T) || T->hasMetadata("structurizecfg.uniform");
}

/// Is BB the last block saved on the stack ?
bool SIAnnotateControlFlow::isTopOfStack(BasicBlock *BB) {
  return !Stack.empty() && Stack.back().first == BB;
}

/// Pop the last saved value from the control flow stack
Value *SIAnnotateControlFlow::popSaved() {
  return Stack.pop_back_val().second;
}

/// Push a BB and saved value to the control flow stack
void SIAnnotateControlFlow::push(BasicBlock *BB, Value *Saved) {
  Stack.push_back(std::pair(BB, Saved));
}

/// Can the condition represented by this PHI node treated like
/// an "Else" block?
bool SIAnnotateControlFlow::isElse(PHINode *Phi) {
  BasicBlock *IDom = DT->getNode(Phi->getParent())->getIDom()->getBlock();
  for (unsigned i = 0, e = Phi->getNumIncomingValues(); i != e; ++i) {
    if (Phi->getIncomingBlock(i) == IDom) {

      if (Phi->getIncomingValue(i) != BoolTrue)
        return false;

    } else {
      if (Phi->getIncomingValue(i) != BoolFalse)
        return false;

    }
  }
  return true;
}

bool SIAnnotateControlFlow::hasKill(const BasicBlock *BB) {
  for (const Instruction &I : *BB) {
    if (const CallInst *CI = dyn_cast<CallInst>(&I))
      if (CI->getIntrinsicID() == Intrinsic::amdgcn_kill)
        return true;
  }
  return false;
}

// Erase "Phi" if it is not used any more. Return true if any change was made.
bool SIAnnotateControlFlow::eraseIfUnused(PHINode *Phi) {
  bool Changed = RecursivelyDeleteDeadPHINode(Phi);
  if (Changed)
    LLVM_DEBUG(dbgs() << "Erased unused condition phi\n");
  return Changed;
}

/// Open a new "If" block
bool SIAnnotateControlFlow::openIf(BranchInst *Term) {
  if (isUniform(Term))
    return false;

  IRBuilder<> IRB(Term);
  Value *IfCall = IRB.CreateCall(getDecl(If, Intrinsic::amdgcn_if, IntMask),
                                 {Term->getCondition()});
  Value *Cond = IRB.CreateExtractValue(IfCall, {0});
  Value *Mask = IRB.CreateExtractValue(IfCall, {1});
  Term->setCondition(Cond);
  push(Term->getSuccessor(1), Mask);
  return true;
}

/// Close the last "If" block and open a new "Else" block
bool SIAnnotateControlFlow::insertElse(BranchInst *Term) {
  if (isUniform(Term)) {
    return false;
  }

  IRBuilder<> IRB(Term);
  Value *ElseCall = IRB.CreateCall(
      getDecl(Else, Intrinsic::amdgcn_else, {IntMask, IntMask}), {popSaved()});
  Value *Cond = IRB.CreateExtractValue(ElseCall, {0});
  Value *Mask = IRB.CreateExtractValue(ElseCall, {1});
  Term->setCondition(Cond);
  push(Term->getSuccessor(1), Mask);
  return true;
}

/// Recursively handle the condition leading to a loop
Value *SIAnnotateControlFlow::handleLoopCondition(
    Value *Cond, PHINode *Broken, llvm::Loop *L, BranchInst *Term) {

  auto CreateBreak = [this, Cond, Broken](Instruction *I) -> CallInst * {
    return IRBuilder<>(I).CreateCall(
        getDecl(IfBreak, Intrinsic::amdgcn_if_break, IntMask), {Cond, Broken});
  };

  if (Instruction *Inst = dyn_cast<Instruction>(Cond)) {
    BasicBlock *Parent = Inst->getParent();
    Instruction *Insert;
    if (LI->getLoopFor(Parent) == L) {
      // Insert IfBreak in the same BB as Cond, which can help
      // SILowerControlFlow to know that it does not have to insert an
      // AND with EXEC.
      Insert = Parent->getTerminator();
    } else if (L->contains(Inst)) {
      Insert = Term;
    } else {
      Insert = &*L->getHeader()->getFirstNonPHIOrDbgOrLifetime();
    }

    return CreateBreak(Insert);
  }

  // Insert IfBreak in the loop header TERM for constant COND other than true.
  if (isa<Constant>(Cond)) {
    Instruction *Insert = Cond == BoolTrue ?
      Term : L->getHeader()->getTerminator();

    return CreateBreak(Insert);
  }

  if (isa<Argument>(Cond)) {
    Instruction *Insert = &*L->getHeader()->getFirstNonPHIOrDbgOrLifetime();
    return CreateBreak(Insert);
  }

  llvm_unreachable("Unhandled loop condition!");
}

/// Handle a back edge (loop)
bool SIAnnotateControlFlow::handleLoop(BranchInst *Term) {
  if (isUniform(Term))
    return false;

  BasicBlock *BB = Term->getParent();
  llvm::Loop *L = LI->getLoopFor(BB);
  if (!L)
    return false;

  BasicBlock *Target = Term->getSuccessor(1);
  PHINode *Broken = PHINode::Create(IntMask, 0, "phi.broken");
  Broken->insertBefore(Target->begin());

  Value *Cond = Term->getCondition();
  Term->setCondition(BoolTrue);
  Value *Arg = handleLoopCondition(Cond, Broken, L, Term);

  for (BasicBlock *Pred : predecessors(Target)) {
    Value *PHIValue = IntMaskZero;
    if (Pred == BB) // Remember the value of the previous iteration.
      PHIValue = Arg;
    // If the backedge from Pred to Target could be executed before the exit
    // of the loop at BB, it should not reset or change "Broken", which keeps
    // track of the number of threads exited the loop at BB.
    else if (L->contains(Pred) && DT->dominates(Pred, BB))
      PHIValue = Broken;
    Broken->addIncoming(PHIValue, Pred);
  }

  CallInst *LoopCall = IRBuilder<>(Term).CreateCall(
      getDecl(Loop, Intrinsic::amdgcn_loop, IntMask), {Arg});
  Term->setCondition(LoopCall);

  push(Term->getSuccessor(0), Arg);

  return true;
}

/// Close the last opened control flow
bool SIAnnotateControlFlow::closeControlFlow(BasicBlock *BB) {
  llvm::Loop *L = LI->getLoopFor(BB);

  assert(Stack.back().first == BB);

  if (L && L->getHeader() == BB) {
    // We can't insert an EndCF call into a loop header, because it will
    // get executed on every iteration of the loop, when it should be
    // executed only once before the loop.
    SmallVector <BasicBlock *, 8> Latches;
    L->getLoopLatches(Latches);

    SmallVector<BasicBlock *, 2> Preds;
    for (BasicBlock *Pred : predecessors(BB)) {
      if (!is_contained(Latches, Pred))
        Preds.push_back(Pred);
    }

    BB = SplitBlockPredecessors(BB, Preds, "endcf.split", DT, LI, nullptr,
                                false);
  }

  Value *Exec = popSaved();
  BasicBlock::iterator FirstInsertionPt = BB->getFirstInsertionPt();
  if (!isa<UndefValue>(Exec) && !isa<UnreachableInst>(FirstInsertionPt)) {
    Instruction *ExecDef = cast<Instruction>(Exec);
    BasicBlock *DefBB = ExecDef->getParent();
    if (!DT->dominates(DefBB, BB)) {
      // Split edge to make Def dominate Use
      FirstInsertionPt = SplitEdge(DefBB, BB, DT, LI)->getFirstInsertionPt();
    }
    IRBuilder<> IRB(FirstInsertionPt->getParent(), FirstInsertionPt);
    // TODO: StructurizeCFG 'Flow' blocks have debug locations from the
    // condition, for now just avoid copying these DebugLocs so that stepping
    // out of the then/else block in a debugger doesn't step to the condition.
    IRB.SetCurrentDebugLocation(DebugLoc());
    IRB.CreateCall(getDecl(EndCf, Intrinsic::amdgcn_end_cf, IntMask), {Exec});
  }

  return true;
}

/// Annotate the control flow with intrinsics so the backend can
/// recognize if/then/else and loops.
bool SIAnnotateControlFlow::run() {
  bool Changed = false;

  for (df_iterator<BasicBlock *> I = df_begin(&F->getEntryBlock()),
                                 E = df_end(&F->getEntryBlock());
       I != E; ++I) {
    BasicBlock *BB = *I;
    BranchInst *Term = dyn_cast<BranchInst>(BB->getTerminator());

    if (!Term || Term->isUnconditional()) {
      if (isTopOfStack(BB))
        Changed |= closeControlFlow(BB);

      continue;
    }

    if (I.nodeVisited(Term->getSuccessor(1))) {
      if (isTopOfStack(BB))
        Changed |= closeControlFlow(BB);

      if (DT->dominates(Term->getSuccessor(1), BB))
        Changed |= handleLoop(Term);
      continue;
    }

    if (isTopOfStack(BB)) {
      PHINode *Phi = dyn_cast<PHINode>(Term->getCondition());
      if (Phi && Phi->getParent() == BB && isElse(Phi) && !hasKill(BB)) {
        Changed |= insertElse(Term);
        Changed |= eraseIfUnused(Phi);
        continue;
      }

      Changed |= closeControlFlow(BB);
    }

    Changed |= openIf(Term);
  }

  if (!Stack.empty()) {
    // CFG was probably not structured.
    report_fatal_error("failed to annotate CFG");
  }

  return Changed;
}

PreservedAnalyses SIAnnotateControlFlowPass::run(Function &F,
                                                 FunctionAnalysisManager &FAM) {
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);

  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

  SIAnnotateControlFlow Impl(F, ST, DT, LI, UI);

  bool Changed = Impl.run();
  if (!Changed)
    return PreservedAnalyses::all();

  // TODO: Is LoopInfo preserved?
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}

class SIAnnotateControlFlowLegacy : public FunctionPass {
public:
  static char ID;

  SIAnnotateControlFlowLegacy() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "SI annotate control flow"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    UniformityInfo &UI =
        getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
    TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
    const TargetMachine &TM = TPC.getTM<TargetMachine>();
    const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);

    SIAnnotateControlFlow Impl(F, ST, DT, LI, UI);
    return Impl.run();
  }
};

INITIALIZE_PASS_BEGIN(SIAnnotateControlFlowLegacy, DEBUG_TYPE,
                      "Annotate SI Control Flow", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(SIAnnotateControlFlowLegacy, DEBUG_TYPE,
                    "Annotate SI Control Flow", false, false)

char SIAnnotateControlFlowLegacy::ID = 0;

/// Create the annotation pass
FunctionPass *llvm::createSIAnnotateControlFlowLegacyPass() {
  return new SIAnnotateControlFlowLegacy();
}
