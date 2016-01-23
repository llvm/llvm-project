//===-- SIAnnotateControlFlow.cpp -  ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Annotates the control flow with hardware specific intrinsics.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "si-annotate-control-flow"

namespace {

// Complex types used in this pass
typedef std::pair<BasicBlock *, Value *> StackEntry;
typedef SmallVector<StackEntry, 16> StackVector;

// Intrinsic names the control flow is annotated with
static const char *const IfIntrinsic = "llvm.amdgcn.if";
static const char *const ElseIntrinsic = "llvm.amdgcn.else";
static const char *const BreakIntrinsic = "llvm.amdgcn.break";
static const char *const IfBreakIntrinsic = "llvm.amdgcn.if.break";
static const char *const ElseBreakIntrinsic = "llvm.amdgcn.else.break";
static const char *const LoopIntrinsic = "llvm.amdgcn.loop";
static const char *const EndCfIntrinsic = "llvm.amdgcn.end.cf";

class SIAnnotateControlFlow : public FunctionPass {

  Type *Boolean;
  Type *Void;
  Type *Int64;
  Type *ReturnStruct;

  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  UndefValue *BoolUndef;
  Constant *Int64Zero;

  Constant *If;
  Constant *Else;
  Constant *Break;
  Constant *IfBreak;
  Constant *ElseBreak;
  Constant *Loop;
  Constant *EndCf;

  DominatorTree *DT;
  StackVector Stack;

  LoopInfo *LI;

  bool isTopOfStack(BasicBlock *BB);

  Value *popSaved();

  void push(BasicBlock *BB, Value *Saved);

  bool isElse(PHINode *Phi);

  void eraseIfUnused(PHINode *Phi);

  void openIf(BranchInst *Term);

  void insertElse(BranchInst *Term);

  Value *handleLoopCondition(Value *Cond, PHINode *Broken, llvm::Loop *L);

  void handleLoop(BranchInst *Term);

  void closeControlFlow(BasicBlock *BB);

public:
  static char ID;

  SIAnnotateControlFlow():
    FunctionPass(ID) { }

  bool doInitialization(Module &M) override;

  bool runOnFunction(Function &F) override;

  const char *getPassName() const override {
    return "SI annotate control flow";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SIAnnotateControlFlow, DEBUG_TYPE,
                      "Annotate SI Control Flow", false, false)
INITIALIZE_PASS_END(SIAnnotateControlFlow, DEBUG_TYPE,
                    "Annotate SI Control Flow", false, false)

char SIAnnotateControlFlow::ID = 0;

/// \brief Initialize all the types and constants used in the pass
bool SIAnnotateControlFlow::doInitialization(Module &M) {
  LLVMContext &Context = M.getContext();

  Void = Type::getVoidTy(Context);
  Boolean = Type::getInt1Ty(Context);
  Int64 = Type::getInt64Ty(Context);
  ReturnStruct = StructType::get(Boolean, Int64, (Type *)nullptr);

  BoolTrue = ConstantInt::getTrue(Context);
  BoolFalse = ConstantInt::getFalse(Context);
  BoolUndef = UndefValue::get(Boolean);
  Int64Zero = ConstantInt::get(Int64, 0);

  If = M.getOrInsertFunction(
    IfIntrinsic, ReturnStruct, Boolean, (Type *)nullptr);

  Else = M.getOrInsertFunction(
    ElseIntrinsic, ReturnStruct, Int64, (Type *)nullptr);

  Break = M.getOrInsertFunction(
    BreakIntrinsic, Int64, Int64, (Type *)nullptr);

  IfBreak = M.getOrInsertFunction(
    IfBreakIntrinsic, Int64, Boolean, Int64, (Type *)nullptr);

  ElseBreak = M.getOrInsertFunction(
    ElseBreakIntrinsic, Int64, Int64, Int64, (Type *)nullptr);

  Loop = M.getOrInsertFunction(
    LoopIntrinsic, Boolean, Int64, (Type *)nullptr);

  EndCf = M.getOrInsertFunction(
    EndCfIntrinsic, Void, Int64, (Type *)nullptr);

  return false;
}

/// \brief Is BB the last block saved on the stack ?
bool SIAnnotateControlFlow::isTopOfStack(BasicBlock *BB) {
  return !Stack.empty() && Stack.back().first == BB;
}

/// \brief Pop the last saved value from the control flow stack
Value *SIAnnotateControlFlow::popSaved() {
  return Stack.pop_back_val().second;
}

/// \brief Push a BB and saved value to the control flow stack
void SIAnnotateControlFlow::push(BasicBlock *BB, Value *Saved) {
  Stack.push_back(std::make_pair(BB, Saved));
}

/// \brief Can the condition represented by this PHI node treated like
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

// \brief Erase "Phi" if it is not used any more
void SIAnnotateControlFlow::eraseIfUnused(PHINode *Phi) {
  if (!Phi->hasNUsesOrMore(1))
    Phi->eraseFromParent();
}

/// \brief Open a new "If" block
void SIAnnotateControlFlow::openIf(BranchInst *Term) {
  Value *Ret = CallInst::Create(If, Term->getCondition(), "", Term);
  Term->setCondition(ExtractValueInst::Create(Ret, 0, "", Term));
  push(Term->getSuccessor(1), ExtractValueInst::Create(Ret, 1, "", Term));
}

/// \brief Close the last "If" block and open a new "Else" block
void SIAnnotateControlFlow::insertElse(BranchInst *Term) {
  Value *Ret = CallInst::Create(Else, popSaved(), "", Term);
  Term->setCondition(ExtractValueInst::Create(Ret, 0, "", Term));
  push(Term->getSuccessor(1), ExtractValueInst::Create(Ret, 1, "", Term));
}

/// \brief Recursively handle the condition leading to a loop
Value *SIAnnotateControlFlow::handleLoopCondition(Value *Cond, PHINode *Broken,
                                                  llvm::Loop *L) {

  // Only search through PHI nodes which are inside the loop.  If we try this
  // with PHI nodes that are outside of the loop, we end up inserting new PHI
  // nodes outside of the loop which depend on values defined inside the loop.
  // This will break the module with
  // 'Instruction does not dominate all users!' errors.
  PHINode *Phi = nullptr;
  if ((Phi = dyn_cast<PHINode>(Cond)) && L->contains(Phi)) {

    BasicBlock *Parent = Phi->getParent();
    PHINode *NewPhi = PHINode::Create(Int64, 0, "", &Parent->front());
    Value *Ret = NewPhi;

    // Handle all non-constant incoming values first
    for (unsigned i = 0, e = Phi->getNumIncomingValues(); i != e; ++i) {
      Value *Incoming = Phi->getIncomingValue(i);
      BasicBlock *From = Phi->getIncomingBlock(i);
      if (isa<ConstantInt>(Incoming)) {
        NewPhi->addIncoming(Broken, From);
        continue;
      }

      Phi->setIncomingValue(i, BoolFalse);
      Value *PhiArg = handleLoopCondition(Incoming, Broken, L);
      NewPhi->addIncoming(PhiArg, From);
    }

    BasicBlock *IDom = DT->getNode(Parent)->getIDom()->getBlock();

    for (unsigned i = 0, e = Phi->getNumIncomingValues(); i != e; ++i) {

      Value *Incoming = Phi->getIncomingValue(i);
      if (Incoming != BoolTrue)
        continue;

      BasicBlock *From = Phi->getIncomingBlock(i);
      if (From == IDom) {
        CallInst *OldEnd = dyn_cast<CallInst>(Parent->getFirstInsertionPt());
        if (OldEnd && OldEnd->getCalledFunction() == EndCf) {
          Value *Args[] = { OldEnd->getArgOperand(0), NewPhi };
          Ret = CallInst::Create(ElseBreak, Args, "", OldEnd);
          continue;
        }
      }
      TerminatorInst *Insert = From->getTerminator();
      Value *PhiArg = CallInst::Create(Break, Broken, "", Insert);
      NewPhi->setIncomingValue(i, PhiArg);
    }
    eraseIfUnused(Phi);
    return Ret;

  } else if (Instruction *Inst = dyn_cast<Instruction>(Cond)) {
    BasicBlock *Parent = Inst->getParent();
    Instruction *Insert;
    if (L->contains(Inst)) {
      Insert = Parent->getTerminator();
    } else {
      Insert = L->getHeader()->getFirstNonPHIOrDbgOrLifetime();
    }
    Value *Args[] = { Cond, Broken };
    return CallInst::Create(IfBreak, Args, "", Insert);

  } else {
    llvm_unreachable("Unhandled loop condition!");
  }
  return 0;
}

/// \brief Handle a back edge (loop)
void SIAnnotateControlFlow::handleLoop(BranchInst *Term) {
  BasicBlock *BB = Term->getParent();
  llvm::Loop *L = LI->getLoopFor(BB);
  BasicBlock *Target = Term->getSuccessor(1);
  PHINode *Broken = PHINode::Create(Int64, 0, "", &Target->front());

  Value *Cond = Term->getCondition();
  Term->setCondition(BoolTrue);
  Value *Arg = handleLoopCondition(Cond, Broken, L);

  for (pred_iterator PI = pred_begin(Target), PE = pred_end(Target);
       PI != PE; ++PI) {

    Broken->addIncoming(*PI == BB ? Arg : Int64Zero, *PI);
  }

  Term->setCondition(CallInst::Create(Loop, Arg, "", Term));
  push(Term->getSuccessor(0), Arg);
}/// \brief Close the last opened control flow
void SIAnnotateControlFlow::closeControlFlow(BasicBlock *BB) {
  llvm::Loop *L = LI->getLoopFor(BB);

  if (L && L->getHeader() == BB) {
    // We can't insert an EndCF call into a loop header, because it will
    // get executed on every iteration of the loop, when it should be
    // executed only once before the loop.
    SmallVector <BasicBlock*, 8> Latches;
    L->getLoopLatches(Latches);

    std::vector<BasicBlock*> Preds;
    for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
      if (std::find(Latches.begin(), Latches.end(), *PI) == Latches.end())
        Preds.push_back(*PI);
    }
    BB = llvm::SplitBlockPredecessors(BB, Preds, "endcf.split", DT, LI, false);
  }

  CallInst::Create(EndCf, popSaved(), "", &*BB->getFirstInsertionPt());
}

/// \brief Annotate the control flow with intrinsics so the backend can
/// recognize if/then/else and loops.
bool SIAnnotateControlFlow::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  for (df_iterator<BasicBlock *> I = df_begin(&F.getEntryBlock()),
       E = df_end(&F.getEntryBlock()); I != E; ++I) {

    BranchInst *Term = dyn_cast<BranchInst>((*I)->getTerminator());

    if (!Term || Term->isUnconditional()) {
      if (isTopOfStack(*I))
        closeControlFlow(*I);
      continue;
    }

    if (I.nodeVisited(Term->getSuccessor(1))) {
      if (isTopOfStack(*I))
        closeControlFlow(*I);
      handleLoop(Term);
      continue;
    }

    if (isTopOfStack(*I)) {
      PHINode *Phi = dyn_cast<PHINode>(Term->getCondition());
      if (Phi && Phi->getParent() == *I && isElse(Phi)) {
        insertElse(Term);
        eraseIfUnused(Phi);
        continue;
      }
      closeControlFlow(*I);
    }
    openIf(Term);
  }

  assert(Stack.empty());
  return true;
}

/// \brief Create the annotation pass
FunctionPass *llvm::createSIAnnotateControlFlowPass() {
  return new SIAnnotateControlFlow();
}
