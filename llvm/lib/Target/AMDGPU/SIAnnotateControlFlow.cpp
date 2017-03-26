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
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "si-annotate-control-flow"

namespace {

// Complex types used in this pass
typedef std::pair<BasicBlock *, Value *> StackEntry;
typedef SmallVector<StackEntry, 16> StackVector;

class SIAnnotateControlFlow : public FunctionPass {
  DivergenceAnalysis *DA;

  Type *Boolean;
  Type *Void;
  Type *Int64;
  Type *ReturnStruct;

  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  UndefValue *BoolUndef;
  Constant *Int64Zero;

  Function *If;
  Function *Else;
  Function *Break;
  Function *IfBreak;
  Function *ElseBreak;
  Function *Loop;
  Function *EndCf;

  DominatorTree *DT;
  StackVector Stack;

  LoopInfo *LI;

  bool isUniform(BranchInst *T);

  bool isTopOfStack(BasicBlock *BB);

  Value *popSaved();

  void push(BasicBlock *BB, Value *Saved);

  bool isElse(PHINode *Phi);

  void eraseIfUnused(PHINode *Phi);

  void openIf(BranchInst *Term);

  void insertElse(BranchInst *Term);

  Value *handleLoopCondition(Value *Cond, PHINode *Broken,
                             llvm::Loop *L, BranchInst *Term,
                             SmallVectorImpl<WeakVH> &LoopPhiConditions);

  void handleLoop(BranchInst *Term);

  void closeControlFlow(BasicBlock *BB);

public:
  static char ID;

  SIAnnotateControlFlow():
    FunctionPass(ID) { }

  bool doInitialization(Module &M) override;

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "SI annotate control flow"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<DivergenceAnalysis>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }

};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SIAnnotateControlFlow, DEBUG_TYPE,
                      "Annotate SI Control Flow", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DivergenceAnalysis)
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

  If = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_if);
  Else = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_else);
  Break = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_break);
  IfBreak = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_if_break);
  ElseBreak = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_else_break);
  Loop = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_loop);
  EndCf = Intrinsic::getDeclaration(&M, Intrinsic::amdgcn_end_cf);
  return false;
}

/// \brief Is the branch condition uniform or did the StructurizeCFG pass
/// consider it as such?
bool SIAnnotateControlFlow::isUniform(BranchInst *T) {
  return DA->isUniform(T->getCondition()) ||
         T->getMetadata("structurizecfg.uniform") != nullptr;
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
  if (llvm::RecursivelyDeleteDeadPHINode(Phi)) {
    DEBUG(dbgs() << "Erased unused condition phi\n");
  }
}

/// \brief Open a new "If" block
void SIAnnotateControlFlow::openIf(BranchInst *Term) {
  if (isUniform(Term))
    return;

  Value *Ret = CallInst::Create(If, Term->getCondition(), "", Term);
  Term->setCondition(ExtractValueInst::Create(Ret, 0, "", Term));
  push(Term->getSuccessor(1), ExtractValueInst::Create(Ret, 1, "", Term));
}

/// \brief Close the last "If" block and open a new "Else" block
void SIAnnotateControlFlow::insertElse(BranchInst *Term) {
  if (isUniform(Term)) {
    return;
  }
  Value *Ret = CallInst::Create(Else, popSaved(), "", Term);
  Term->setCondition(ExtractValueInst::Create(Ret, 0, "", Term));
  push(Term->getSuccessor(1), ExtractValueInst::Create(Ret, 1, "", Term));
}

/// \brief Recursively handle the condition leading to a loop
Value *SIAnnotateControlFlow::handleLoopCondition(
  Value *Cond, PHINode *Broken,
  llvm::Loop *L, BranchInst *Term,
  SmallVectorImpl<WeakVH> &LoopPhiConditions) {

  // Only search through PHI nodes which are inside the loop.  If we try this
  // with PHI nodes that are outside of the loop, we end up inserting new PHI
  // nodes outside of the loop which depend on values defined inside the loop.
  // This will break the module with
  // 'Instruction does not dominate all users!' errors.
  PHINode *Phi = nullptr;
  if ((Phi = dyn_cast<PHINode>(Cond)) && L->contains(Phi)) {

    BasicBlock *Parent = Phi->getParent();
    PHINode *NewPhi = PHINode::Create(Int64, 0, "loop.phi", &Parent->front());
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
      Value *PhiArg = handleLoopCondition(Incoming, Broken, L,
                                          Term, LoopPhiConditions);
      NewPhi->addIncoming(PhiArg, From);
    }

    BasicBlock *IDom = DT->getNode(Parent)->getIDom()->getBlock();

    for (unsigned i = 0, e = Phi->getNumIncomingValues(); i != e; ++i) {
      Value *Incoming = Phi->getIncomingValue(i);
      if (Incoming != BoolTrue)
        continue;

      BasicBlock *From = Phi->getIncomingBlock(i);
      if (From == IDom) {
        // We're in the following situation:
        //   IDom/From
        //      |   \
        //      |   If-block
        //      |   /
        //     Parent
        // where we want to break out of the loop if the If-block is not taken.
        // Due to the depth-first traversal, there should be an end.cf
        // intrinsic in Parent, and we insert an else.break before it.
        //
        // Note that the end.cf need not be the first non-phi instruction
        // of parent, particularly when we're dealing with a multi-level
        // break, but it should occur within a group of intrinsic calls
        // at the beginning of the block.
        CallInst *OldEnd = dyn_cast<CallInst>(Parent->getFirstInsertionPt());
        while (OldEnd && OldEnd->getCalledFunction() != EndCf)
          OldEnd = dyn_cast<CallInst>(OldEnd->getNextNode());
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

    LoopPhiConditions.push_back(WeakVH(Phi));
    return Ret;
  }

  if (Instruction *Inst = dyn_cast<Instruction>(Cond)) {
    BasicBlock *Parent = Inst->getParent();
    Instruction *Insert;
    if (L->contains(Inst)) {
      Insert = Parent->getTerminator();
    } else {
      Insert = L->getHeader()->getFirstNonPHIOrDbgOrLifetime();
    }

    Value *Args[] = { Cond, Broken };
    return CallInst::Create(IfBreak, Args, "", Insert);
  }

  // Insert IfBreak in the loop header TERM for constant COND other than true.
  if (isa<Constant>(Cond)) {
    Instruction *Insert = Cond == BoolTrue ?
      Term : L->getHeader()->getTerminator();

    Value *Args[] = { Cond, Broken };
    return CallInst::Create(IfBreak, Args, "", Insert);
  }

  llvm_unreachable("Unhandled loop condition!");
}

/// \brief Handle a back edge (loop)
void SIAnnotateControlFlow::handleLoop(BranchInst *Term) {
  if (isUniform(Term))
    return;

  BasicBlock *BB = Term->getParent();
  llvm::Loop *L = LI->getLoopFor(BB);
  if (!L)
    return;

  BasicBlock *Target = Term->getSuccessor(1);
  PHINode *Broken = PHINode::Create(Int64, 0, "phi.broken", &Target->front());

  SmallVector<WeakVH, 8> LoopPhiConditions;
  Value *Cond = Term->getCondition();
  Term->setCondition(BoolTrue);
  Value *Arg = handleLoopCondition(Cond, Broken, L, Term, LoopPhiConditions);

  for (BasicBlock *Pred : predecessors(Target))
    Broken->addIncoming(Pred == BB ? Arg : Int64Zero, Pred);

  Term->setCondition(CallInst::Create(Loop, Arg, "", Term));

  for (WeakVH Val : reverse(LoopPhiConditions)) {
    if (PHINode *Cond = cast_or_null<PHINode>(Val))
      eraseIfUnused(Cond);
  }

  push(Term->getSuccessor(0), Arg);
}

/// \brief Close the last opened control flow
void SIAnnotateControlFlow::closeControlFlow(BasicBlock *BB) {
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

    BB = llvm::SplitBlockPredecessors(BB, Preds, "endcf.split", DT, LI, false);
  }

  Value *Exec = popSaved();
  Instruction *FirstInsertionPt = &*BB->getFirstInsertionPt();
  if (!isa<UndefValue>(Exec) && !isa<UnreachableInst>(FirstInsertionPt))
    CallInst::Create(EndCf, Exec, "", FirstInsertionPt);
}

/// \brief Annotate the control flow with intrinsics so the backend can
/// recognize if/then/else and loops.
bool SIAnnotateControlFlow::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DA = &getAnalysis<DivergenceAnalysis>();

  for (df_iterator<BasicBlock *> I = df_begin(&F.getEntryBlock()),
       E = df_end(&F.getEntryBlock()); I != E; ++I) {
    BasicBlock *BB = *I;
    BranchInst *Term = dyn_cast<BranchInst>(BB->getTerminator());

    if (!Term || Term->isUnconditional()) {
      if (isTopOfStack(BB))
        closeControlFlow(BB);

      continue;
    }

    if (I.nodeVisited(Term->getSuccessor(1))) {
      if (isTopOfStack(BB))
        closeControlFlow(BB);

      handleLoop(Term);
      continue;
    }

    if (isTopOfStack(BB)) {
      PHINode *Phi = dyn_cast<PHINode>(Term->getCondition());
      if (Phi && Phi->getParent() == BB && isElse(Phi)) {
        insertElse(Term);
        eraseIfUnused(Phi);
        continue;
      }

      closeControlFlow(BB);
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
