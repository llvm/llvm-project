//===- CodeGenPrepare.cpp - Prepare a function for code generation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach. It should eventually be removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "codegenprepare"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/ValueMap.h"
#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/BypassSlowDivision.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;
using namespace llvm::PatternMatch;

STATISTIC(NumBlocksElim, "Number of blocks eliminated");
STATISTIC(NumPHIsElim,   "Number of trivial PHIs eliminated");
STATISTIC(NumGEPsElim,   "Number of GEPs converted to casts");
STATISTIC(NumCmpUses, "Number of uses of Cmp expressions replaced with uses of "
                      "sunken Cmps");
STATISTIC(NumCastUses, "Number of uses of Cast expressions replaced with uses "
                       "of sunken Casts");
STATISTIC(NumMemoryInsts, "Number of memory instructions whose address "
                          "computations were sunk");
STATISTIC(NumExtsMoved,  "Number of [s|z]ext instructions combined with loads");
STATISTIC(NumExtUses,    "Number of uses of [s|z]ext instructions optimized");
STATISTIC(NumRetsDup,    "Number of return instructions duplicated");
STATISTIC(NumDbgValueMoved, "Number of debug value instructions moved");
STATISTIC(NumSelectsExpanded, "Number of selects turned into branches");

static cl::opt<bool> DisableBranchOpts(
  "disable-cgp-branch-opts", cl::Hidden, cl::init(false),
  cl::desc("Disable branch optimizations in CodeGenPrepare"));

static cl::opt<bool> DisableSelectToBranch(
  "disable-cgp-select2branch", cl::Hidden, cl::init(false),
  cl::desc("Disable select to branch conversion."));

namespace {
  class CodeGenPrepare : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// transformation profitability.
    const TargetLowering *TLI;
    const TargetLibraryInfo *TLInfo;
    DominatorTree *DT;
    ProfileInfo *PFI;

    /// CurInstIterator - As we scan instructions optimizing them, this is the
    /// next instruction to optimize.  Xforms that can invalidate this should
    /// update it.
    BasicBlock::iterator CurInstIterator;

    /// Keeps track of non-local addresses that have been sunk into a block.
    /// This allows us to avoid inserting duplicate code for blocks with
    /// multiple load/stores of the same address.
    ValueMap<Value*, Value*> SunkAddrs;

    /// ModifiedDT - If CFG is modified in anyway, dominator tree may need to
    /// be updated.
    bool ModifiedDT;

    /// OptSize - True if optimizing for size.
    bool OptSize;

  public:
    static char ID; // Pass identification, replacement for typeid
    explicit CodeGenPrepare(const TargetLowering *tli = 0)
      : FunctionPass(ID), TLI(tli) {
        initializeCodeGenPreparePass(*PassRegistry::getPassRegistry());
      }
    bool runOnFunction(Function &F);

    const char *getPassName() const { return "CodeGen Prepare"; }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<ProfileInfo>();
      AU.addRequired<TargetLibraryInfo>();
    }

  private:
    bool EliminateFallThrough(Function &F);
    bool EliminateMostlyEmptyBlocks(Function &F);
    bool CanMergeBlocks(const BasicBlock *BB, const BasicBlock *DestBB) const;
    void EliminateMostlyEmptyBlock(BasicBlock *BB);
    bool OptimizeBlock(BasicBlock &BB);
    bool OptimizeInst(Instruction *I);
    bool OptimizeMemoryInst(Instruction *I, Value *Addr, Type *AccessTy);
    bool OptimizeInlineAsmInst(CallInst *CS);
    bool OptimizeCallInst(CallInst *CI);
    bool MoveExtToFormExtLoad(Instruction *I);
    bool OptimizeExtUses(Instruction *I);
    bool OptimizeSelectInst(SelectInst *SI);
    bool DupRetToEnableTailCallOpts(BasicBlock *BB);
    bool PlaceDbgValues(Function &F);
  };
}

char CodeGenPrepare::ID = 0;
INITIALIZE_PASS_BEGIN(CodeGenPrepare, "codegenprepare",
                "Optimize for code generation", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(CodeGenPrepare, "codegenprepare",
                "Optimize for code generation", false, false)

FunctionPass *llvm::createCodeGenPreparePass(const TargetLowering *TLI) {
  return new CodeGenPrepare(TLI);
}

bool CodeGenPrepare::runOnFunction(Function &F) {
  bool EverMadeChange = false;

  ModifiedDT = false;
  TLInfo = &getAnalysis<TargetLibraryInfo>();
  DT = getAnalysisIfAvailable<DominatorTree>();
  PFI = getAnalysisIfAvailable<ProfileInfo>();
  OptSize = F.getAttributes().hasAttribute(AttributeSet::FunctionIndex,
                                           Attribute::OptimizeForSize);

  /// This optimization identifies DIV instructions that can be
  /// profitably bypassed and carried out with a shorter, faster divide.
  if (!OptSize && TLI && TLI->isSlowDivBypassed()) {
    const DenseMap<unsigned int, unsigned int> &BypassWidths =
       TLI->getBypassSlowDivWidths();
    for (Function::iterator I = F.begin(); I != F.end(); I++)
      EverMadeChange |= bypassSlowDivision(F, I, BypassWidths);
  }

  // Eliminate blocks that contain only PHI nodes and an
  // unconditional branch.
  EverMadeChange |= EliminateMostlyEmptyBlocks(F);

  // llvm.dbg.value is far away from the value then iSel may not be able
  // handle it properly. iSel will drop llvm.dbg.value if it can not
  // find a node corresponding to the value.
  EverMadeChange |= PlaceDbgValues(F);

  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (Function::iterator I = F.begin(); I != F.end(); ) {
      BasicBlock *BB = I++;
      MadeChange |= OptimizeBlock(*BB);
    }
    EverMadeChange |= MadeChange;
  }

  SunkAddrs.clear();

  if (!DisableBranchOpts) {
    MadeChange = false;
    SmallPtrSet<BasicBlock*, 8> WorkList;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
      SmallVector<BasicBlock*, 2> Successors(succ_begin(BB), succ_end(BB));
      MadeChange |= ConstantFoldTerminator(BB, true);
      if (!MadeChange) continue;

      for (SmallVectorImpl<BasicBlock*>::iterator
             II = Successors.begin(), IE = Successors.end(); II != IE; ++II)
        if (pred_begin(*II) == pred_end(*II))
          WorkList.insert(*II);
    }

    // Delete the dead blocks and any of their dead successors.
    MadeChange |= !WorkList.empty();
    while (!WorkList.empty()) {
      BasicBlock *BB = *WorkList.begin();
      WorkList.erase(BB);
      SmallVector<BasicBlock*, 2> Successors(succ_begin(BB), succ_end(BB));

      DeleteDeadBlock(BB);
      
      for (SmallVectorImpl<BasicBlock*>::iterator
             II = Successors.begin(), IE = Successors.end(); II != IE; ++II)
        if (pred_begin(*II) == pred_end(*II))
          WorkList.insert(*II);
    }

    // Merge pairs of basic blocks with unconditional branches, connected by
    // a single edge.
    if (EverMadeChange || MadeChange)
      MadeChange |= EliminateFallThrough(F);

    if (MadeChange)
      ModifiedDT = true;
    EverMadeChange |= MadeChange;
  }

  if (ModifiedDT && DT)
    DT->DT->recalculate(F);

  return EverMadeChange;
}

/// EliminateFallThrough - Merge basic blocks which are connected
/// by a single edge, where one of the basic blocks has a single successor
/// pointing to the other basic block, which has a single predecessor.
bool CodeGenPrepare::EliminateFallThrough(Function &F) {
  bool Changed = false;
  // Scan all of the blocks in the function, except for the entry block.
  for (Function::iterator I = ++F.begin(), E = F.end(); I != E; ) {
    BasicBlock *BB = I++;
    // If the destination block has a single pred, then this is a trivial
    // edge, just collapse it.
    BasicBlock *SinglePred = BB->getSinglePredecessor();

    // Don't merge if BB's address is taken.
    if (!SinglePred || SinglePred == BB || BB->hasAddressTaken()) continue;

    BranchInst *Term = dyn_cast<BranchInst>(SinglePred->getTerminator());
    if (Term && !Term->isConditional()) {
      Changed = true;
      DEBUG(dbgs() << "To merge:\n"<< *SinglePred << "\n\n\n");
      // Remember if SinglePred was the entry block of the function.
      // If so, we will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      MergeBasicBlockIntoOnlyPred(BB, this);

      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());

      // We have erased a block. Update the iterator.
      I = BB;
    }
  }
  return Changed;
}

/// EliminateMostlyEmptyBlocks - eliminate blocks that contain only PHI nodes,
/// debug info directives, and an unconditional branch.  Passes before isel
/// (e.g. LSR/loopsimplify) often split edges in ways that are non-optimal for
/// isel.  Start by eliminating these blocks so we can split them the way we
/// want them.
bool CodeGenPrepare::EliminateMostlyEmptyBlocks(Function &F) {
  bool MadeChange = false;
  // Note that this intentionally skips the entry block.
  for (Function::iterator I = ++F.begin(), E = F.end(); I != E; ) {
    BasicBlock *BB = I++;

    // If this block doesn't end with an uncond branch, ignore it.
    BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isUnconditional())
      continue;

    // If the instruction before the branch (skipping debug info) isn't a phi
    // node, then other stuff is happening here.
    BasicBlock::iterator BBI = BI;
    if (BBI != BB->begin()) {
      --BBI;
      while (isa<DbgInfoIntrinsic>(BBI)) {
        if (BBI == BB->begin())
          break;
        --BBI;
      }
      if (!isa<DbgInfoIntrinsic>(BBI) && !isa<PHINode>(BBI))
        continue;
    }

    // Do not break infinite loops.
    BasicBlock *DestBB = BI->getSuccessor(0);
    if (DestBB == BB)
      continue;

    if (!CanMergeBlocks(BB, DestBB))
      continue;

    EliminateMostlyEmptyBlock(BB);
    MadeChange = true;
  }
  return MadeChange;
}

/// CanMergeBlocks - Return true if we can merge BB into DestBB if there is a
/// single uncond branch between them, and BB contains no other non-phi
/// instructions.
bool CodeGenPrepare::CanMergeBlocks(const BasicBlock *BB,
                                    const BasicBlock *DestBB) const {
  // We only want to eliminate blocks whose phi nodes are used by phi nodes in
  // the successor.  If there are more complex condition (e.g. preheaders),
  // don't mess around with them.
  BasicBlock::const_iterator BBI = BB->begin();
  while (const PHINode *PN = dyn_cast<PHINode>(BBI++)) {
    for (Value::const_use_iterator UI = PN->use_begin(), E = PN->use_end();
         UI != E; ++UI) {
      const Instruction *User = cast<Instruction>(*UI);
      if (User->getParent() != DestBB || !isa<PHINode>(User))
        return false;
      // If User is inside DestBB block and it is a PHINode then check
      // incoming value. If incoming value is not from BB then this is
      // a complex condition (e.g. preheaders) we want to avoid here.
      if (User->getParent() == DestBB) {
        if (const PHINode *UPN = dyn_cast<PHINode>(User))
          for (unsigned I = 0, E = UPN->getNumIncomingValues(); I != E; ++I) {
            Instruction *Insn = dyn_cast<Instruction>(UPN->getIncomingValue(I));
            if (Insn && Insn->getParent() == BB &&
                Insn->getParent() != UPN->getIncomingBlock(I))
              return false;
          }
      }
    }
  }

  // If BB and DestBB contain any common predecessors, then the phi nodes in BB
  // and DestBB may have conflicting incoming values for the block.  If so, we
  // can't merge the block.
  const PHINode *DestBBPN = dyn_cast<PHINode>(DestBB->begin());
  if (!DestBBPN) return true;  // no conflict.

  // Collect the preds of BB.
  SmallPtrSet<const BasicBlock*, 16> BBPreds;
  if (const PHINode *BBPN = dyn_cast<PHINode>(BB->begin())) {
    // It is faster to get preds from a PHI than with pred_iterator.
    for (unsigned i = 0, e = BBPN->getNumIncomingValues(); i != e; ++i)
      BBPreds.insert(BBPN->getIncomingBlock(i));
  } else {
    BBPreds.insert(pred_begin(BB), pred_end(BB));
  }

  // Walk the preds of DestBB.
  for (unsigned i = 0, e = DestBBPN->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *Pred = DestBBPN->getIncomingBlock(i);
    if (BBPreds.count(Pred)) {   // Common predecessor?
      BBI = DestBB->begin();
      while (const PHINode *PN = dyn_cast<PHINode>(BBI++)) {
        const Value *V1 = PN->getIncomingValueForBlock(Pred);
        const Value *V2 = PN->getIncomingValueForBlock(BB);

        // If V2 is a phi node in BB, look up what the mapped value will be.
        if (const PHINode *V2PN = dyn_cast<PHINode>(V2))
          if (V2PN->getParent() == BB)
            V2 = V2PN->getIncomingValueForBlock(Pred);

        // If there is a conflict, bail out.
        if (V1 != V2) return false;
      }
    }
  }

  return true;
}


/// EliminateMostlyEmptyBlock - Eliminate a basic block that have only phi's and
/// an unconditional branch in it.
void CodeGenPrepare::EliminateMostlyEmptyBlock(BasicBlock *BB) {
  BranchInst *BI = cast<BranchInst>(BB->getTerminator());
  BasicBlock *DestBB = BI->getSuccessor(0);

  DEBUG(dbgs() << "MERGING MOSTLY EMPTY BLOCKS - BEFORE:\n" << *BB << *DestBB);

  // If the destination block has a single pred, then this is a trivial edge,
  // just collapse it.
  if (BasicBlock *SinglePred = DestBB->getSinglePredecessor()) {
    if (SinglePred != DestBB) {
      // Remember if SinglePred was the entry block of the function.  If so, we
      // will need to move BB back to the entry position.
      bool isEntry = SinglePred == &SinglePred->getParent()->getEntryBlock();
      MergeBasicBlockIntoOnlyPred(DestBB, this);

      if (isEntry && BB != &BB->getParent()->getEntryBlock())
        BB->moveBefore(&BB->getParent()->getEntryBlock());

      DEBUG(dbgs() << "AFTER:\n" << *DestBB << "\n\n\n");
      return;
    }
  }

  // Otherwise, we have multiple predecessors of BB.  Update the PHIs in DestBB
  // to handle the new incoming edges it is about to have.
  PHINode *PN;
  for (BasicBlock::iterator BBI = DestBB->begin();
       (PN = dyn_cast<PHINode>(BBI)); ++BBI) {
    // Remove the incoming value for BB, and remember it.
    Value *InVal = PN->removeIncomingValue(BB, false);

    // Two options: either the InVal is a phi node defined in BB or it is some
    // value that dominates BB.
    PHINode *InValPhi = dyn_cast<PHINode>(InVal);
    if (InValPhi && InValPhi->getParent() == BB) {
      // Add all of the input values of the input PHI as inputs of this phi.
      for (unsigned i = 0, e = InValPhi->getNumIncomingValues(); i != e; ++i)
        PN->addIncoming(InValPhi->getIncomingValue(i),
                        InValPhi->getIncomingBlock(i));
    } else {
      // Otherwise, add one instance of the dominating value for each edge that
      // we will be adding.
      if (PHINode *BBPN = dyn_cast<PHINode>(BB->begin())) {
        for (unsigned i = 0, e = BBPN->getNumIncomingValues(); i != e; ++i)
          PN->addIncoming(InVal, BBPN->getIncomingBlock(i));
      } else {
        for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
          PN->addIncoming(InVal, *PI);
      }
    }
  }

  // The PHIs are now updated, change everything that refers to BB to use
  // DestBB and remove BB.
  BB->replaceAllUsesWith(DestBB);
  if (DT && !ModifiedDT) {
    BasicBlock *BBIDom  = DT->getNode(BB)->getIDom()->getBlock();
    BasicBlock *DestBBIDom = DT->getNode(DestBB)->getIDom()->getBlock();
    BasicBlock *NewIDom = DT->findNearestCommonDominator(BBIDom, DestBBIDom);
    DT->changeImmediateDominator(DestBB, NewIDom);
    DT->eraseNode(BB);
  }
  if (PFI) {
    PFI->replaceAllUses(BB, DestBB);
    PFI->removeEdge(ProfileInfo::getEdge(BB, DestBB));
  }
  BB->eraseFromParent();
  ++NumBlocksElim;

  DEBUG(dbgs() << "AFTER:\n" << *DestBB << "\n\n\n");
}

/// OptimizeNoopCopyExpression - If the specified cast instruction is a noop
/// copy (e.g. it's casting from one pointer type to another, i32->i8 on PPC),
/// sink it into user blocks to reduce the number of virtual
/// registers that must be created and coalesced.
///
/// Return true if any changes are made.
///
static bool OptimizeNoopCopyExpression(CastInst *CI, const TargetLowering &TLI){
  // If this is a noop copy,
  EVT SrcVT = TLI.getValueType(CI->getOperand(0)->getType());
  EVT DstVT = TLI.getValueType(CI->getType());

  // This is an fp<->int conversion?
  if (SrcVT.isInteger() != DstVT.isInteger())
    return false;

  // If this is an extension, it will be a zero or sign extension, which
  // isn't a noop.
  if (SrcVT.bitsLT(DstVT)) return false;

  // If these values will be promoted, find out what they will be promoted
  // to.  This helps us consider truncates on PPC as noop copies when they
  // are.
  if (TLI.getTypeAction(CI->getContext(), SrcVT) ==
      TargetLowering::TypePromoteInteger)
    SrcVT = TLI.getTypeToTransformTo(CI->getContext(), SrcVT);
  if (TLI.getTypeAction(CI->getContext(), DstVT) ==
      TargetLowering::TypePromoteInteger)
    DstVT = TLI.getTypeToTransformTo(CI->getContext(), DstVT);

  // If, after promotion, these are the same types, this is a noop copy.
  if (SrcVT != DstVT)
    return false;

  BasicBlock *DefBB = CI->getParent();

  /// InsertedCasts - Only insert a cast in each block once.
  DenseMap<BasicBlock*, CastInst*> InsertedCasts;

  bool MadeChange = false;
  for (Value::use_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this cast is used in.  For PHI's this is the
    // appropriate predecessor block.
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User)) {
      UserBB = PN->getIncomingBlock(UI);
    }

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // If this user is in the same block as the cast, don't change the cast.
    if (UserBB == DefBB) continue;

    // If we have already inserted a cast into this block, use it.
    CastInst *&InsertedCast = InsertedCasts[UserBB];

    if (!InsertedCast) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedCast =
        CastInst::Create(CI->getOpcode(), CI->getOperand(0), CI->getType(), "",
                         InsertPt);
      MadeChange = true;
    }

    // Replace a use of the cast with a use of the new cast.
    TheUse = InsertedCast;
    ++NumCastUses;
  }

  // If we removed all uses, nuke the cast.
  if (CI->use_empty()) {
    CI->eraseFromParent();
    MadeChange = true;
  }

  return MadeChange;
}

/// OptimizeCmpExpression - sink the given CmpInst into user blocks to reduce
/// the number of virtual registers that must be created and coalesced.  This is
/// a clear win except on targets with multiple condition code registers
///  (PowerPC), where it might lose; some adjustment may be wanted there.
///
/// Return true if any changes are made.
static bool OptimizeCmpExpression(CmpInst *CI) {
  BasicBlock *DefBB = CI->getParent();

  /// InsertedCmp - Only insert a cmp in each block once.
  DenseMap<BasicBlock*, CmpInst*> InsertedCmps;

  bool MadeChange = false;
  for (Value::use_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; ) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Preincrement use iterator so we don't invalidate it.
    ++UI;

    // Don't bother for PHI nodes.
    if (isa<PHINode>(User))
      continue;

    // Figure out which BB this cmp is used in.
    BasicBlock *UserBB = User->getParent();

    // If this user is in the same block as the cmp, don't change the cmp.
    if (UserBB == DefBB) continue;

    // If we have already inserted a cmp into this block, use it.
    CmpInst *&InsertedCmp = InsertedCmps[UserBB];

    if (!InsertedCmp) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedCmp =
        CmpInst::Create(CI->getOpcode(),
                        CI->getPredicate(),  CI->getOperand(0),
                        CI->getOperand(1), "", InsertPt);
      MadeChange = true;
    }

    // Replace a use of the cmp with a use of the new cmp.
    TheUse = InsertedCmp;
    ++NumCmpUses;
  }

  // If we removed all uses, nuke the cmp.
  if (CI->use_empty())
    CI->eraseFromParent();

  return MadeChange;
}

namespace {
class CodeGenPrepareFortifiedLibCalls : public SimplifyFortifiedLibCalls {
protected:
  void replaceCall(Value *With) {
    CI->replaceAllUsesWith(With);
    CI->eraseFromParent();
  }
  bool isFoldable(unsigned SizeCIOp, unsigned, bool) const {
      if (ConstantInt *SizeCI =
                             dyn_cast<ConstantInt>(CI->getArgOperand(SizeCIOp)))
        return SizeCI->isAllOnesValue();
    return false;
  }
};
} // end anonymous namespace

bool CodeGenPrepare::OptimizeCallInst(CallInst *CI) {
  BasicBlock *BB = CI->getParent();

  // Lower inline assembly if we can.
  // If we found an inline asm expession, and if the target knows how to
  // lower it to normal LLVM code, do so now.
  if (TLI && isa<InlineAsm>(CI->getCalledValue())) {
    if (TLI->ExpandInlineAsm(CI)) {
      // Avoid invalidating the iterator.
      CurInstIterator = BB->begin();
      // Avoid processing instructions out of order, which could cause
      // reuse before a value is defined.
      SunkAddrs.clear();
      return true;
    }
    // Sink address computing for memory operands into the block.
    if (OptimizeInlineAsmInst(CI))
      return true;
  }

  // Lower all uses of llvm.objectsize.*
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI);
  if (II && II->getIntrinsicID() == Intrinsic::objectsize) {
    bool Min = (cast<ConstantInt>(II->getArgOperand(1))->getZExtValue() == 1);
    Type *ReturnTy = CI->getType();
    Constant *RetVal = ConstantInt::get(ReturnTy, Min ? 0 : -1ULL);

    // Substituting this can cause recursive simplifications, which can
    // invalidate our iterator.  Use a WeakVH to hold onto it in case this
    // happens.
    WeakVH IterHandle(CurInstIterator);

    replaceAndRecursivelySimplify(CI, RetVal, TLI ? TLI->getDataLayout() : 0,
                                  TLInfo, ModifiedDT ? 0 : DT);

    // If the iterator instruction was recursively deleted, start over at the
    // start of the block.
    if (IterHandle != CurInstIterator) {
      CurInstIterator = BB->begin();
      SunkAddrs.clear();
    }
    return true;
  }

  if (II && TLI) {
    SmallVector<Value*, 2> PtrOps;
    Type *AccessTy;
    if (TLI->GetAddrModeArguments(II, PtrOps, AccessTy))
      while (!PtrOps.empty())
        if (OptimizeMemoryInst(II, PtrOps.pop_back_val(), AccessTy))
          return true;
  }

  // From here on out we're working with named functions.
  if (CI->getCalledFunction() == 0) return false;

  // We'll need DataLayout from here on out.
  const DataLayout *TD = TLI ? TLI->getDataLayout() : 0;
  if (!TD) return false;

  // Lower all default uses of _chk calls.  This is very similar
  // to what InstCombineCalls does, but here we are only lowering calls
  // that have the default "don't know" as the objectsize.  Anything else
  // should be left alone.
  CodeGenPrepareFortifiedLibCalls Simplifier;
  return Simplifier.fold(CI, TD, TLInfo);
}

/// DupRetToEnableTailCallOpts - Look for opportunities to duplicate return
/// instructions to the predecessor to enable tail call optimizations. The
/// case it is currently looking for is:
/// @code
/// bb0:
///   %tmp0 = tail call i32 @f0()
///   br label %return
/// bb1:
///   %tmp1 = tail call i32 @f1()
///   br label %return
/// bb2:
///   %tmp2 = tail call i32 @f2()
///   br label %return
/// return:
///   %retval = phi i32 [ %tmp0, %bb0 ], [ %tmp1, %bb1 ], [ %tmp2, %bb2 ]
///   ret i32 %retval
/// @endcode
///
/// =>
///
/// @code
/// bb0:
///   %tmp0 = tail call i32 @f0()
///   ret i32 %tmp0
/// bb1:
///   %tmp1 = tail call i32 @f1()
///   ret i32 %tmp1
/// bb2:
///   %tmp2 = tail call i32 @f2()
///   ret i32 %tmp2
/// @endcode
bool CodeGenPrepare::DupRetToEnableTailCallOpts(BasicBlock *BB) {
  if (!TLI)
    return false;

  ReturnInst *RI = dyn_cast<ReturnInst>(BB->getTerminator());
  if (!RI)
    return false;

  PHINode *PN = 0;
  BitCastInst *BCI = 0;
  Value *V = RI->getReturnValue();
  if (V) {
    BCI = dyn_cast<BitCastInst>(V);
    if (BCI)
      V = BCI->getOperand(0);

    PN = dyn_cast<PHINode>(V);
    if (!PN)
      return false;
  }

  if (PN && PN->getParent() != BB)
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  // See llvm::isInTailCallPosition().
  const Function *F = BB->getParent();
  AttributeSet CallerAttrs = F->getAttributes();
  if (CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::ZExt) ||
      CallerAttrs.hasAttribute(AttributeSet::ReturnIndex, Attribute::SExt))
    return false;

  // Make sure there are no instructions between the PHI and return, or that the
  // return is the first instruction in the block.
  if (PN) {
    BasicBlock::iterator BI = BB->begin();
    do { ++BI; } while (isa<DbgInfoIntrinsic>(BI));
    if (&*BI == BCI)
      // Also skip over the bitcast.
      ++BI;
    if (&*BI != RI)
      return false;
  } else {
    BasicBlock::iterator BI = BB->begin();
    while (isa<DbgInfoIntrinsic>(BI)) ++BI;
    if (&*BI != RI)
      return false;
  }

  /// Only dup the ReturnInst if the CallInst is likely to be emitted as a tail
  /// call.
  SmallVector<CallInst*, 4> TailCalls;
  if (PN) {
    for (unsigned I = 0, E = PN->getNumIncomingValues(); I != E; ++I) {
      CallInst *CI = dyn_cast<CallInst>(PN->getIncomingValue(I));
      // Make sure the phi value is indeed produced by the tail call.
      if (CI && CI->hasOneUse() && CI->getParent() == PN->getIncomingBlock(I) &&
          TLI->mayBeEmittedAsTailCall(CI))
        TailCalls.push_back(CI);
    }
  } else {
    SmallPtrSet<BasicBlock*, 4> VisitedBBs;
    for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
      if (!VisitedBBs.insert(*PI))
        continue;

      BasicBlock::InstListType &InstList = (*PI)->getInstList();
      BasicBlock::InstListType::reverse_iterator RI = InstList.rbegin();
      BasicBlock::InstListType::reverse_iterator RE = InstList.rend();
      do { ++RI; } while (RI != RE && isa<DbgInfoIntrinsic>(&*RI));
      if (RI == RE)
        continue;

      CallInst *CI = dyn_cast<CallInst>(&*RI);
      if (CI && CI->use_empty() && TLI->mayBeEmittedAsTailCall(CI))
        TailCalls.push_back(CI);
    }
  }

  bool Changed = false;
  for (unsigned i = 0, e = TailCalls.size(); i != e; ++i) {
    CallInst *CI = TailCalls[i];
    CallSite CS(CI);

    // Conservatively require the attributes of the call to match those of the
    // return. Ignore noalias because it doesn't affect the call sequence.
    AttributeSet CalleeAttrs = CS.getAttributes();
    if (AttrBuilder(CalleeAttrs, AttributeSet::ReturnIndex).
          removeAttribute(Attribute::NoAlias) !=
        AttrBuilder(CalleeAttrs, AttributeSet::ReturnIndex).
          removeAttribute(Attribute::NoAlias))
      continue;

    // Make sure the call instruction is followed by an unconditional branch to
    // the return block.
    BasicBlock *CallBB = CI->getParent();
    BranchInst *BI = dyn_cast<BranchInst>(CallBB->getTerminator());
    if (!BI || !BI->isUnconditional() || BI->getSuccessor(0) != BB)
      continue;

    // Duplicate the return into CallBB.
    (void)FoldReturnIntoUncondBranch(RI, BB, CallBB);
    ModifiedDT = Changed = true;
    ++NumRetsDup;
  }

  // If we eliminated all predecessors of the block, delete the block now.
  if (Changed && !BB->hasAddressTaken() && pred_begin(BB) == pred_end(BB))
    BB->eraseFromParent();

  return Changed;
}

//===----------------------------------------------------------------------===//
// Memory Optimization
//===----------------------------------------------------------------------===//

namespace {

/// ExtAddrMode - This is an extended version of TargetLowering::AddrMode
/// which holds actual Value*'s for register values.
struct ExtAddrMode : public TargetLowering::AddrMode {
  Value *BaseReg;
  Value *ScaledReg;
  ExtAddrMode() : BaseReg(0), ScaledReg(0) {}
  void print(raw_ostream &OS) const;
  void dump() const;
  
  bool operator==(const ExtAddrMode& O) const {
    return (BaseReg == O.BaseReg) && (ScaledReg == O.ScaledReg) &&
           (BaseGV == O.BaseGV) && (BaseOffs == O.BaseOffs) &&
           (HasBaseReg == O.HasBaseReg) && (Scale == O.Scale);
  }
};

static inline raw_ostream &operator<<(raw_ostream &OS, const ExtAddrMode &AM) {
  AM.print(OS);
  return OS;
}

void ExtAddrMode::print(raw_ostream &OS) const {
  bool NeedPlus = false;
  OS << "[";
  if (BaseGV) {
    OS << (NeedPlus ? " + " : "")
       << "GV:";
    WriteAsOperand(OS, BaseGV, /*PrintType=*/false);
    NeedPlus = true;
  }

  if (BaseOffs)
    OS << (NeedPlus ? " + " : "") << BaseOffs, NeedPlus = true;

  if (BaseReg) {
    OS << (NeedPlus ? " + " : "")
       << "Base:";
    WriteAsOperand(OS, BaseReg, /*PrintType=*/false);
    NeedPlus = true;
  }
  if (Scale) {
    OS << (NeedPlus ? " + " : "")
       << Scale << "*";
    WriteAsOperand(OS, ScaledReg, /*PrintType=*/false);
    NeedPlus = true;
  }

  OS << ']';
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ExtAddrMode::dump() const {
  print(dbgs());
  dbgs() << '\n';
}
#endif


/// \brief A helper class for matching addressing modes.
///
/// This encapsulates the logic for matching the target-legal addressing modes.
class AddressingModeMatcher {
  SmallVectorImpl<Instruction*> &AddrModeInsts;
  const TargetLowering &TLI;

  /// AccessTy/MemoryInst - This is the type for the access (e.g. double) and
  /// the memory instruction that we're computing this address for.
  Type *AccessTy;
  Instruction *MemoryInst;
  
  /// AddrMode - This is the addressing mode that we're building up.  This is
  /// part of the return value of this addressing mode matching stuff.
  ExtAddrMode &AddrMode;
  
  /// IgnoreProfitability - This is set to true when we should not do
  /// profitability checks.  When true, IsProfitableToFoldIntoAddressingMode
  /// always returns true.
  bool IgnoreProfitability;
  
  AddressingModeMatcher(SmallVectorImpl<Instruction*> &AMI,
                        const TargetLowering &T, Type *AT,
                        Instruction *MI, ExtAddrMode &AM)
    : AddrModeInsts(AMI), TLI(T), AccessTy(AT), MemoryInst(MI), AddrMode(AM) {
    IgnoreProfitability = false;
  }
public:
  
  /// Match - Find the maximal addressing mode that a load/store of V can fold,
  /// give an access type of AccessTy.  This returns a list of involved
  /// instructions in AddrModeInsts.
  static ExtAddrMode Match(Value *V, Type *AccessTy,
                           Instruction *MemoryInst,
                           SmallVectorImpl<Instruction*> &AddrModeInsts,
                           const TargetLowering &TLI) {
    ExtAddrMode Result;

    bool Success = 
      AddressingModeMatcher(AddrModeInsts, TLI, AccessTy,
                            MemoryInst, Result).MatchAddr(V, 0);
    (void)Success; assert(Success && "Couldn't select *anything*?");
    return Result;
  }
private:
  bool MatchScaledValue(Value *ScaleReg, int64_t Scale, unsigned Depth);
  bool MatchAddr(Value *V, unsigned Depth);
  bool MatchOperationAddr(User *Operation, unsigned Opcode, unsigned Depth);
  bool IsProfitableToFoldIntoAddressingMode(Instruction *I,
                                            ExtAddrMode &AMBefore,
                                            ExtAddrMode &AMAfter);
  bool ValueAlreadyLiveAtInst(Value *Val, Value *KnownLive1, Value *KnownLive2);
};

/// MatchScaledValue - Try adding ScaleReg*Scale to the current addressing mode.
/// Return true and update AddrMode if this addr mode is legal for the target,
/// false if not.
bool AddressingModeMatcher::MatchScaledValue(Value *ScaleReg, int64_t Scale,
                                             unsigned Depth) {
  // If Scale is 1, then this is the same as adding ScaleReg to the addressing
  // mode.  Just process that directly.
  if (Scale == 1)
    return MatchAddr(ScaleReg, Depth);
  
  // If the scale is 0, it takes nothing to add this.
  if (Scale == 0)
    return true;
  
  // If we already have a scale of this value, we can add to it, otherwise, we
  // need an available scale field.
  if (AddrMode.Scale != 0 && AddrMode.ScaledReg != ScaleReg)
    return false;

  ExtAddrMode TestAddrMode = AddrMode;

  // Add scale to turn X*4+X*3 -> X*7.  This could also do things like
  // [A+B + A*7] -> [B+A*8].
  TestAddrMode.Scale += Scale;
  TestAddrMode.ScaledReg = ScaleReg;

  // If the new address isn't legal, bail out.
  if (!TLI.isLegalAddressingMode(TestAddrMode, AccessTy))
    return false;

  // It was legal, so commit it.
  AddrMode = TestAddrMode;
  
  // Okay, we decided that we can add ScaleReg+Scale to AddrMode.  Check now
  // to see if ScaleReg is actually X+C.  If so, we can turn this into adding
  // X*Scale + C*Scale to addr mode.
  ConstantInt *CI = 0; Value *AddLHS = 0;
  if (isa<Instruction>(ScaleReg) &&  // not a constant expr.
      match(ScaleReg, m_Add(m_Value(AddLHS), m_ConstantInt(CI)))) {
    TestAddrMode.ScaledReg = AddLHS;
    TestAddrMode.BaseOffs += CI->getSExtValue()*TestAddrMode.Scale;
      
    // If this addressing mode is legal, commit it and remember that we folded
    // this instruction.
    if (TLI.isLegalAddressingMode(TestAddrMode, AccessTy)) {
      AddrModeInsts.push_back(cast<Instruction>(ScaleReg));
      AddrMode = TestAddrMode;
      return true;
    }
  }

  // Otherwise, not (x+c)*scale, just return what we have.
  return true;
}

/// MightBeFoldableInst - This is a little filter, which returns true if an
/// addressing computation involving I might be folded into a load/store
/// accessing it.  This doesn't need to be perfect, but needs to accept at least
/// the set of instructions that MatchOperationAddr can.
static bool MightBeFoldableInst(Instruction *I) {
  switch (I->getOpcode()) {
  case Instruction::BitCast:
    // Don't touch identity bitcasts.
    if (I->getType() == I->getOperand(0)->getType())
      return false;
    return I->getType()->isPointerTy() || I->getType()->isIntegerTy();
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    return true;
  case Instruction::IntToPtr:
    // We know the input is intptr_t, so this is foldable.
    return true;
  case Instruction::Add:
    return true;
  case Instruction::Mul:
  case Instruction::Shl:
    // Can only handle X*C and X << C.
    return isa<ConstantInt>(I->getOperand(1));
  case Instruction::GetElementPtr:
    return true;
  default:
    return false;
  }
}

/// MatchOperationAddr - Given an instruction or constant expr, see if we can
/// fold the operation into the addressing mode.  If so, update the addressing
/// mode and return true, otherwise return false without modifying AddrMode.
bool AddressingModeMatcher::MatchOperationAddr(User *AddrInst, unsigned Opcode,
                                               unsigned Depth) {
  // Avoid exponential behavior on extremely deep expression trees.
  if (Depth >= 5) return false;
  
  switch (Opcode) {
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    return MatchAddr(AddrInst->getOperand(0), Depth);
  case Instruction::IntToPtr:
    // This inttoptr is a no-op if the integer type is pointer sized.
    if (TLI.getValueType(AddrInst->getOperand(0)->getType()) ==
        TLI.getPointerTy())
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  case Instruction::BitCast:
    // BitCast is always a noop, and we can handle it as long as it is
    // int->int or pointer->pointer (we don't want int<->fp or something).
    if ((AddrInst->getOperand(0)->getType()->isPointerTy() ||
         AddrInst->getOperand(0)->getType()->isIntegerTy()) &&
        // Don't touch identity bitcasts.  These were probably put here by LSR,
        // and we don't want to mess around with them.  Assume it knows what it
        // is doing.
        AddrInst->getOperand(0)->getType() != AddrInst->getType())
      return MatchAddr(AddrInst->getOperand(0), Depth);
    return false;
  case Instruction::Add: {
    // Check to see if we can merge in the RHS then the LHS.  If so, we win.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();
    if (MatchAddr(AddrInst->getOperand(1), Depth+1) &&
        MatchAddr(AddrInst->getOperand(0), Depth+1))
      return true;
    
    // Restore the old addr mode info.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    
    // Otherwise this was over-aggressive.  Try merging in the LHS then the RHS.
    if (MatchAddr(AddrInst->getOperand(0), Depth+1) &&
        MatchAddr(AddrInst->getOperand(1), Depth+1))
      return true;
    
    // Otherwise we definitely can't merge the ADD in.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    break;
  }
  //case Instruction::Or:
  // TODO: We can handle "Or Val, Imm" iff this OR is equivalent to an ADD.
  //break;
  case Instruction::Mul:
  case Instruction::Shl: {
    // Can only handle X*C and X << C.
    ConstantInt *RHS = dyn_cast<ConstantInt>(AddrInst->getOperand(1));
    if (!RHS) return false;
    int64_t Scale = RHS->getSExtValue();
    if (Opcode == Instruction::Shl)
      Scale = 1LL << Scale;
    
    return MatchScaledValue(AddrInst->getOperand(0), Scale, Depth);
  }
  case Instruction::GetElementPtr: {
    // Scan the GEP.  We check it if it contains constant offsets and at most
    // one variable offset.
    int VariableOperand = -1;
    unsigned VariableScale = 0;
    
    int64_t ConstantOffset = 0;
    const DataLayout *TD = TLI.getDataLayout();
    gep_type_iterator GTI = gep_type_begin(AddrInst);
    for (unsigned i = 1, e = AddrInst->getNumOperands(); i != e; ++i, ++GTI) {
      if (StructType *STy = dyn_cast<StructType>(*GTI)) {
        const StructLayout *SL = TD->getStructLayout(STy);
        unsigned Idx =
          cast<ConstantInt>(AddrInst->getOperand(i))->getZExtValue();
        ConstantOffset += SL->getElementOffset(Idx);
      } else {
        uint64_t TypeSize = TD->getTypeAllocSize(GTI.getIndexedType());
        if (ConstantInt *CI = dyn_cast<ConstantInt>(AddrInst->getOperand(i))) {
          ConstantOffset += CI->getSExtValue()*TypeSize;
        } else if (TypeSize) {  // Scales of zero don't do anything.
          // We only allow one variable index at the moment.
          if (VariableOperand != -1)
            return false;
          
          // Remember the variable index.
          VariableOperand = i;
          VariableScale = TypeSize;
        }
      }
    }
    
    // A common case is for the GEP to only do a constant offset.  In this case,
    // just add it to the disp field and check validity.
    if (VariableOperand == -1) {
      AddrMode.BaseOffs += ConstantOffset;
      if (ConstantOffset == 0 || TLI.isLegalAddressingMode(AddrMode, AccessTy)){
        // Check to see if we can fold the base pointer in too.
        if (MatchAddr(AddrInst->getOperand(0), Depth+1))
          return true;
      }
      AddrMode.BaseOffs -= ConstantOffset;
      return false;
    }

    // Save the valid addressing mode in case we can't match.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    // See if the scale and offset amount is valid for this target.
    AddrMode.BaseOffs += ConstantOffset;

    // Match the base operand of the GEP.
    if (!MatchAddr(AddrInst->getOperand(0), Depth+1)) {
      // If it couldn't be matched, just stuff the value in a register.
      if (AddrMode.HasBaseReg) {
        AddrMode = BackupAddrMode;
        AddrModeInsts.resize(OldSize);
        return false;
      }
      AddrMode.HasBaseReg = true;
      AddrMode.BaseReg = AddrInst->getOperand(0);
    }

    // Match the remaining variable portion of the GEP.
    if (!MatchScaledValue(AddrInst->getOperand(VariableOperand), VariableScale,
                          Depth)) {
      // If it couldn't be matched, try stuffing the base into a register
      // instead of matching it, and retrying the match of the scale.
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
      if (AddrMode.HasBaseReg)
        return false;
      AddrMode.HasBaseReg = true;
      AddrMode.BaseReg = AddrInst->getOperand(0);
      AddrMode.BaseOffs += ConstantOffset;
      if (!MatchScaledValue(AddrInst->getOperand(VariableOperand),
                            VariableScale, Depth)) {
        // If even that didn't work, bail.
        AddrMode = BackupAddrMode;
        AddrModeInsts.resize(OldSize);
        return false;
      }
    }

    return true;
  }
  }
  return false;
}

/// MatchAddr - If we can, try to add the value of 'Addr' into the current
/// addressing mode.  If Addr can't be added to AddrMode this returns false and
/// leaves AddrMode unmodified.  This assumes that Addr is either a pointer type
/// or intptr_t for the target.
///
bool AddressingModeMatcher::MatchAddr(Value *Addr, unsigned Depth) {
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Addr)) {
    // Fold in immediates if legal for the target.
    AddrMode.BaseOffs += CI->getSExtValue();
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.BaseOffs -= CI->getSExtValue();
  } else if (GlobalValue *GV = dyn_cast<GlobalValue>(Addr)) {
    // If this is a global variable, try to fold it into the addressing mode.
    if (AddrMode.BaseGV == 0) {
      AddrMode.BaseGV = GV;
      if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
        return true;
      AddrMode.BaseGV = 0;
    }
  } else if (Instruction *I = dyn_cast<Instruction>(Addr)) {
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();

    // Check to see if it is possible to fold this operation.
    if (MatchOperationAddr(I, I->getOpcode(), Depth)) {
      // Okay, it's possible to fold this.  Check to see if it is actually
      // *profitable* to do so.  We use a simple cost model to avoid increasing
      // register pressure too much.
      if (I->hasOneUse() ||
          IsProfitableToFoldIntoAddressingMode(I, BackupAddrMode, AddrMode)) {
        AddrModeInsts.push_back(I);
        return true;
      }
      
      // It isn't profitable to do this, roll back.
      //cerr << "NOT FOLDING: " << *I;
      AddrMode = BackupAddrMode;
      AddrModeInsts.resize(OldSize);
    }
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr)) {
    if (MatchOperationAddr(CE, CE->getOpcode(), Depth))
      return true;
  } else if (isa<ConstantPointerNull>(Addr)) {
    // Null pointer gets folded without affecting the addressing mode.
    return true;
  }

  // Worse case, the target should support [reg] addressing modes. :)
  if (!AddrMode.HasBaseReg) {
    AddrMode.HasBaseReg = true;
    AddrMode.BaseReg = Addr;
    // Still check for legality in case the target supports [imm] but not [i+r].
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.HasBaseReg = false;
    AddrMode.BaseReg = 0;
  }

  // If the base register is already taken, see if we can do [r+r].
  if (AddrMode.Scale == 0) {
    AddrMode.Scale = 1;
    AddrMode.ScaledReg = Addr;
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.Scale = 0;
    AddrMode.ScaledReg = 0;
  }
  // Couldn't match.
  return false;
}

/// IsOperandAMemoryOperand - Check to see if all uses of OpVal by the specified
/// inline asm call are due to memory operands.  If so, return true, otherwise
/// return false.
static bool IsOperandAMemoryOperand(CallInst *CI, InlineAsm *IA, Value *OpVal,
                                    const TargetLowering &TLI) {
  TargetLowering::AsmOperandInfoVector TargetConstraints = TLI.ParseConstraints(ImmutableCallSite(CI));
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];
    
    // Compute the constraint code and ConstraintType to use.
    TLI.ComputeConstraintToUse(OpInfo, SDValue());

    // If this asm operand is our Value*, and if it isn't an indirect memory
    // operand, we can't fold it!
    if (OpInfo.CallOperandVal == OpVal &&
        (OpInfo.ConstraintType != TargetLowering::C_Memory ||
         !OpInfo.isIndirect))
      return false;
  }

  return true;
}

/// FindAllMemoryUses - Recursively walk all the uses of I until we find a
/// memory use.  If we find an obviously non-foldable instruction, return true.
/// Add the ultimately found memory instructions to MemoryUses.
static bool FindAllMemoryUses(Instruction *I,
                SmallVectorImpl<std::pair<Instruction*,unsigned> > &MemoryUses,
                              SmallPtrSet<Instruction*, 16> &ConsideredInsts,
                              const TargetLowering &TLI) {
  // If we already considered this instruction, we're done.
  if (!ConsideredInsts.insert(I))
    return false;
  
  // If this is an obviously unfoldable instruction, bail out.
  if (!MightBeFoldableInst(I))
    return true;

  // Loop over all the uses, recursively processing them.
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    User *U = *UI;

    if (LoadInst *LI = dyn_cast<LoadInst>(U)) {
      MemoryUses.push_back(std::make_pair(LI, UI.getOperandNo()));
      continue;
    }
    
    if (StoreInst *SI = dyn_cast<StoreInst>(U)) {
      unsigned opNo = UI.getOperandNo();
      if (opNo == 0) return true; // Storing addr, not into addr.
      MemoryUses.push_back(std::make_pair(SI, opNo));
      continue;
    }
    
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledValue());
      if (!IA) return true;
      
      // If this is a memory operand, we're cool, otherwise bail out.
      if (!IsOperandAMemoryOperand(CI, IA, I, TLI))
        return true;
      continue;
    }
    
    if (FindAllMemoryUses(cast<Instruction>(U), MemoryUses, ConsideredInsts,
                          TLI))
      return true;
  }

  return false;
}

/// ValueAlreadyLiveAtInst - Retrn true if Val is already known to be live at
/// the use site that we're folding it into.  If so, there is no cost to
/// include it in the addressing mode.  KnownLive1 and KnownLive2 are two values
/// that we know are live at the instruction already.
bool AddressingModeMatcher::ValueAlreadyLiveAtInst(Value *Val,Value *KnownLive1,
                                                   Value *KnownLive2) {
  // If Val is either of the known-live values, we know it is live!
  if (Val == 0 || Val == KnownLive1 || Val == KnownLive2)
    return true;
  
  // All values other than instructions and arguments (e.g. constants) are live.
  if (!isa<Instruction>(Val) && !isa<Argument>(Val)) return true;
  
  // If Val is a constant sized alloca in the entry block, it is live, this is
  // true because it is just a reference to the stack/frame pointer, which is
  // live for the whole function.
  if (AllocaInst *AI = dyn_cast<AllocaInst>(Val))
    if (AI->isStaticAlloca())
      return true;
  
  // Check to see if this value is already used in the memory instruction's
  // block.  If so, it's already live into the block at the very least, so we
  // can reasonably fold it.
  return Val->isUsedInBasicBlock(MemoryInst->getParent());
}

/// IsProfitableToFoldIntoAddressingMode - It is possible for the addressing
/// mode of the machine to fold the specified instruction into a load or store
/// that ultimately uses it.  However, the specified instruction has multiple
/// uses.  Given this, it may actually increase register pressure to fold it
/// into the load.  For example, consider this code:
///
///     X = ...
///     Y = X+1
///     use(Y)   -> nonload/store
///     Z = Y+1
///     load Z
///
/// In this case, Y has multiple uses, and can be folded into the load of Z
/// (yielding load [X+2]).  However, doing this will cause both "X" and "X+1" to
/// be live at the use(Y) line.  If we don't fold Y into load Z, we use one
/// fewer register.  Since Y can't be folded into "use(Y)" we don't increase the
/// number of computations either.
///
/// Note that this (like most of CodeGenPrepare) is just a rough heuristic.  If
/// X was live across 'load Z' for other reasons, we actually *would* want to
/// fold the addressing mode in the Z case.  This would make Y die earlier.
bool AddressingModeMatcher::
IsProfitableToFoldIntoAddressingMode(Instruction *I, ExtAddrMode &AMBefore,
                                     ExtAddrMode &AMAfter) {
  if (IgnoreProfitability) return true;
  
  // AMBefore is the addressing mode before this instruction was folded into it,
  // and AMAfter is the addressing mode after the instruction was folded.  Get
  // the set of registers referenced by AMAfter and subtract out those
  // referenced by AMBefore: this is the set of values which folding in this
  // address extends the lifetime of.
  //
  // Note that there are only two potential values being referenced here,
  // BaseReg and ScaleReg (global addresses are always available, as are any
  // folded immediates).
  Value *BaseReg = AMAfter.BaseReg, *ScaledReg = AMAfter.ScaledReg;
  
  // If the BaseReg or ScaledReg was referenced by the previous addrmode, their
  // lifetime wasn't extended by adding this instruction.
  if (ValueAlreadyLiveAtInst(BaseReg, AMBefore.BaseReg, AMBefore.ScaledReg))
    BaseReg = 0;
  if (ValueAlreadyLiveAtInst(ScaledReg, AMBefore.BaseReg, AMBefore.ScaledReg))
    ScaledReg = 0;

  // If folding this instruction (and it's subexprs) didn't extend any live
  // ranges, we're ok with it.
  if (BaseReg == 0 && ScaledReg == 0)
    return true;

  // If all uses of this instruction are ultimately load/store/inlineasm's,
  // check to see if their addressing modes will include this instruction.  If
  // so, we can fold it into all uses, so it doesn't matter if it has multiple
  // uses.
  SmallVector<std::pair<Instruction*,unsigned>, 16> MemoryUses;
  SmallPtrSet<Instruction*, 16> ConsideredInsts;
  if (FindAllMemoryUses(I, MemoryUses, ConsideredInsts, TLI))
    return false;  // Has a non-memory, non-foldable use!
  
  // Now that we know that all uses of this instruction are part of a chain of
  // computation involving only operations that could theoretically be folded
  // into a memory use, loop over each of these uses and see if they could
  // *actually* fold the instruction.
  SmallVector<Instruction*, 32> MatchedAddrModeInsts;
  for (unsigned i = 0, e = MemoryUses.size(); i != e; ++i) {
    Instruction *User = MemoryUses[i].first;
    unsigned OpNo = MemoryUses[i].second;
    
    // Get the access type of this use.  If the use isn't a pointer, we don't
    // know what it accesses.
    Value *Address = User->getOperand(OpNo);
    if (!Address->getType()->isPointerTy())
      return false;
    Type *AddressAccessTy =
      cast<PointerType>(Address->getType())->getElementType();
    
    // Do a match against the root of this address, ignoring profitability. This
    // will tell us if the addressing mode for the memory operation will
    // *actually* cover the shared instruction.
    ExtAddrMode Result;
    AddressingModeMatcher Matcher(MatchedAddrModeInsts, TLI, AddressAccessTy,
                                  MemoryInst, Result);
    Matcher.IgnoreProfitability = true;
    bool Success = Matcher.MatchAddr(Address, 0);
    (void)Success; assert(Success && "Couldn't select *anything*?");

    // If the match didn't cover I, then it won't be shared by it.
    if (std::find(MatchedAddrModeInsts.begin(), MatchedAddrModeInsts.end(),
                  I) == MatchedAddrModeInsts.end())
      return false;
    
    MatchedAddrModeInsts.clear();
  }
  
  return true;
}

} // end anonymous namespace

/// IsNonLocalValue - Return true if the specified values are defined in a
/// different basic block than BB.
static bool IsNonLocalValue(Value *V, BasicBlock *BB) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() != BB;
  return false;
}

/// OptimizeMemoryInst - Load and Store Instructions often have
/// addressing modes that can do significant amounts of computation.  As such,
/// instruction selection will try to get the load or store to do as much
/// computation as possible for the program.  The problem is that isel can only
/// see within a single block.  As such, we sink as much legal addressing mode
/// stuff into the block as possible.
///
/// This method is used to optimize both load/store and inline asms with memory
/// operands.
bool CodeGenPrepare::OptimizeMemoryInst(Instruction *MemoryInst, Value *Addr,
                                        Type *AccessTy) {
  Value *Repl = Addr;

  // Try to collapse single-value PHI nodes.  This is necessary to undo
  // unprofitable PRE transformations.
  SmallVector<Value*, 8> worklist;
  SmallPtrSet<Value*, 16> Visited;
  worklist.push_back(Addr);

  // Use a worklist to iteratively look through PHI nodes, and ensure that
  // the addressing mode obtained from the non-PHI roots of the graph
  // are equivalent.
  Value *Consensus = 0;
  unsigned NumUsesConsensus = 0;
  bool IsNumUsesConsensusValid = false;
  SmallVector<Instruction*, 16> AddrModeInsts;
  ExtAddrMode AddrMode;
  while (!worklist.empty()) {
    Value *V = worklist.back();
    worklist.pop_back();

    // Break use-def graph loops.
    if (!Visited.insert(V)) {
      Consensus = 0;
      break;
    }

    // For a PHI node, push all of its incoming values.
    if (PHINode *P = dyn_cast<PHINode>(V)) {
      for (unsigned i = 0, e = P->getNumIncomingValues(); i != e; ++i)
        worklist.push_back(P->getIncomingValue(i));
      continue;
    }

    // For non-PHIs, determine the addressing mode being computed.
    SmallVector<Instruction*, 16> NewAddrModeInsts;
    ExtAddrMode NewAddrMode =
      AddressingModeMatcher::Match(V, AccessTy, MemoryInst,
                                   NewAddrModeInsts, *TLI);

    // This check is broken into two cases with very similar code to avoid using
    // getNumUses() as much as possible. Some values have a lot of uses, so
    // calling getNumUses() unconditionally caused a significant compile-time
    // regression.
    if (!Consensus) {
      Consensus = V;
      AddrMode = NewAddrMode;
      AddrModeInsts = NewAddrModeInsts;
      continue;
    } else if (NewAddrMode == AddrMode) {
      if (!IsNumUsesConsensusValid) {
        NumUsesConsensus = Consensus->getNumUses();
        IsNumUsesConsensusValid = true;
      }

      // Ensure that the obtained addressing mode is equivalent to that obtained
      // for all other roots of the PHI traversal.  Also, when choosing one
      // such root as representative, select the one with the most uses in order
      // to keep the cost modeling heuristics in AddressingModeMatcher
      // applicable.
      unsigned NumUses = V->getNumUses();
      if (NumUses > NumUsesConsensus) {
        Consensus = V;
        NumUsesConsensus = NumUses;
        AddrModeInsts = NewAddrModeInsts;
      }
      continue;
    }

    Consensus = 0;
    break;
  }

  // If the addressing mode couldn't be determined, or if multiple different
  // ones were determined, bail out now.
  if (!Consensus) return false;

  // Check to see if any of the instructions supersumed by this addr mode are
  // non-local to I's BB.
  bool AnyNonLocal = false;
  for (unsigned i = 0, e = AddrModeInsts.size(); i != e; ++i) {
    if (IsNonLocalValue(AddrModeInsts[i], MemoryInst->getParent())) {
      AnyNonLocal = true;
      break;
    }
  }

  // If all the instructions matched are already in this BB, don't do anything.
  if (!AnyNonLocal) {
    DEBUG(dbgs() << "CGP: Found      local addrmode: " << AddrMode << "\n");
    return false;
  }

  // Insert this computation right after this user.  Since our caller is
  // scanning from the top of the BB to the bottom, reuse of the expr are
  // guaranteed to happen later.
  IRBuilder<> Builder(MemoryInst);

  // Now that we determined the addressing expression we want to use and know
  // that we have to sink it into this block.  Check to see if we have already
  // done this for some other load/store instr in this block.  If so, reuse the
  // computation.
  Value *&SunkAddr = SunkAddrs[Addr];
  if (SunkAddr) {
    DEBUG(dbgs() << "CGP: Reusing nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst);
    if (SunkAddr->getType() != Addr->getType())
      SunkAddr = Builder.CreateBitCast(SunkAddr, Addr->getType());
  } else {
    DEBUG(dbgs() << "CGP: SINKING nonlocal addrmode: " << AddrMode << " for "
                 << *MemoryInst);
    Type *IntPtrTy =
          TLI->getDataLayout()->getIntPtrType(AccessTy->getContext());

    Value *Result = 0;

    // Start with the base register. Do this first so that subsequent address
    // matching finds it last, which will prevent it from trying to match it
    // as the scaled value in case it happens to be a mul. That would be
    // problematic if we've sunk a different mul for the scale, because then
    // we'd end up sinking both muls.
    if (AddrMode.BaseReg) {
      Value *V = AddrMode.BaseReg;
      if (V->getType()->isPointerTy())
        V = Builder.CreatePtrToInt(V, IntPtrTy, "sunkaddr");
      if (V->getType() != IntPtrTy)
        V = Builder.CreateIntCast(V, IntPtrTy, /*isSigned=*/true, "sunkaddr");
      Result = V;
    }

    // Add the scale value.
    if (AddrMode.Scale) {
      Value *V = AddrMode.ScaledReg;
      if (V->getType() == IntPtrTy) {
        // done.
      } else if (V->getType()->isPointerTy()) {
        V = Builder.CreatePtrToInt(V, IntPtrTy, "sunkaddr");
      } else if (cast<IntegerType>(IntPtrTy)->getBitWidth() <
                 cast<IntegerType>(V->getType())->getBitWidth()) {
        V = Builder.CreateTrunc(V, IntPtrTy, "sunkaddr");
      } else {
        V = Builder.CreateSExt(V, IntPtrTy, "sunkaddr");
      }
      if (AddrMode.Scale != 1)
        V = Builder.CreateMul(V, ConstantInt::get(IntPtrTy, AddrMode.Scale),
                              "sunkaddr");
      if (Result)
        Result = Builder.CreateAdd(Result, V, "sunkaddr");
      else
        Result = V;
    }

    // Add in the BaseGV if present.
    if (AddrMode.BaseGV) {
      Value *V = Builder.CreatePtrToInt(AddrMode.BaseGV, IntPtrTy, "sunkaddr");
      if (Result)
        Result = Builder.CreateAdd(Result, V, "sunkaddr");
      else
        Result = V;
    }

    // Add in the Base Offset if present.
    if (AddrMode.BaseOffs) {
      Value *V = ConstantInt::get(IntPtrTy, AddrMode.BaseOffs);
      if (Result)
        Result = Builder.CreateAdd(Result, V, "sunkaddr");
      else
        Result = V;
    }

    if (Result == 0)
      SunkAddr = Constant::getNullValue(Addr->getType());
    else
      SunkAddr = Builder.CreateIntToPtr(Result, Addr->getType(), "sunkaddr");
  }

  MemoryInst->replaceUsesOfWith(Repl, SunkAddr);

  // If we have no uses, recursively delete the value and all dead instructions
  // using it.
  if (Repl->use_empty()) {
    // This can cause recursive deletion, which can invalidate our iterator.
    // Use a WeakVH to hold onto it in case this happens.
    WeakVH IterHandle(CurInstIterator);
    BasicBlock *BB = CurInstIterator->getParent();

    RecursivelyDeleteTriviallyDeadInstructions(Repl, TLInfo);

    if (IterHandle != CurInstIterator) {
      // If the iterator instruction was recursively deleted, start over at the
      // start of the block.
      CurInstIterator = BB->begin();
      SunkAddrs.clear();
    }
  }
  ++NumMemoryInsts;
  return true;
}

/// OptimizeInlineAsmInst - If there are any memory operands, use
/// OptimizeMemoryInst to sink their address computing into the block when
/// possible / profitable.
bool CodeGenPrepare::OptimizeInlineAsmInst(CallInst *CS) {
  bool MadeChange = false;

  TargetLowering::AsmOperandInfoVector
    TargetConstraints = TLI->ParseConstraints(CS);
  unsigned ArgNo = 0;
  for (unsigned i = 0, e = TargetConstraints.size(); i != e; ++i) {
    TargetLowering::AsmOperandInfo &OpInfo = TargetConstraints[i];

    // Compute the constraint code and ConstraintType to use.
    TLI->ComputeConstraintToUse(OpInfo, SDValue());

    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        OpInfo.isIndirect) {
      Value *OpVal = CS->getArgOperand(ArgNo++);
      MadeChange |= OptimizeMemoryInst(CS, OpVal, OpVal->getType());
    } else if (OpInfo.Type == InlineAsm::isInput)
      ArgNo++;
  }

  return MadeChange;
}

/// MoveExtToFormExtLoad - Move a zext or sext fed by a load into the same
/// basic block as the load, unless conditions are unfavorable. This allows
/// SelectionDAG to fold the extend into the load.
///
bool CodeGenPrepare::MoveExtToFormExtLoad(Instruction *I) {
  // Look for a load being extended.
  LoadInst *LI = dyn_cast<LoadInst>(I->getOperand(0));
  if (!LI) return false;

  // If they're already in the same block, there's nothing to do.
  if (LI->getParent() == I->getParent())
    return false;

  // If the load has other users and the truncate is not free, this probably
  // isn't worthwhile.
  if (!LI->hasOneUse() &&
      TLI && (TLI->isTypeLegal(TLI->getValueType(LI->getType())) ||
              !TLI->isTypeLegal(TLI->getValueType(I->getType()))) &&
      !TLI->isTruncateFree(I->getType(), LI->getType()))
    return false;

  // Check whether the target supports casts folded into loads.
  unsigned LType;
  if (isa<ZExtInst>(I))
    LType = ISD::ZEXTLOAD;
  else {
    assert(isa<SExtInst>(I) && "Unexpected ext type!");
    LType = ISD::SEXTLOAD;
  }
  if (TLI && !TLI->isLoadExtLegal(LType, TLI->getValueType(LI->getType())))
    return false;

  // Move the extend into the same block as the load, so that SelectionDAG
  // can fold it.
  I->removeFromParent();
  I->insertAfter(LI);
  ++NumExtsMoved;
  return true;
}

bool CodeGenPrepare::OptimizeExtUses(Instruction *I) {
  BasicBlock *DefBB = I->getParent();

  // If the result of a {s|z}ext and its source are both live out, rewrite all
  // other uses of the source with result of extension.
  Value *Src = I->getOperand(0);
  if (Src->hasOneUse())
    return false;

  // Only do this xform if truncating is free.
  if (TLI && !TLI->isTruncateFree(I->getType(), Src->getType()))
    return false;

  // Only safe to perform the optimization if the source is also defined in
  // this block.
  if (!isa<Instruction>(Src) || DefBB != cast<Instruction>(Src)->getParent())
    return false;

  bool DefIsLiveOut = false;
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;
    DefIsLiveOut = true;
    break;
  }
  if (!DefIsLiveOut)
    return false;

  // Make sure none of the uses are PHI nodes.
  for (Value::use_iterator UI = Src->use_begin(), E = Src->use_end();
       UI != E; ++UI) {
    Instruction *User = cast<Instruction>(*UI);
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;
    // Be conservative. We don't want this xform to end up introducing
    // reloads just before load / store instructions.
    if (isa<PHINode>(User) || isa<LoadInst>(User) || isa<StoreInst>(User))
      return false;
  }

  // InsertedTruncs - Only insert one trunc in each block once.
  DenseMap<BasicBlock*, Instruction*> InsertedTruncs;

  bool MadeChange = false;
  for (Value::use_iterator UI = Src->use_begin(), E = Src->use_end();
       UI != E; ++UI) {
    Use &TheUse = UI.getUse();
    Instruction *User = cast<Instruction>(*UI);

    // Figure out which BB this ext is used in.
    BasicBlock *UserBB = User->getParent();
    if (UserBB == DefBB) continue;

    // Both src and def are live in this block. Rewrite the use.
    Instruction *&InsertedTrunc = InsertedTruncs[UserBB];

    if (!InsertedTrunc) {
      BasicBlock::iterator InsertPt = UserBB->getFirstInsertionPt();
      InsertedTrunc = new TruncInst(I, Src->getType(), "", InsertPt);
    }

    // Replace a use of the {s|z}ext source with a use of the result.
    TheUse = InsertedTrunc;
    ++NumExtUses;
    MadeChange = true;
  }

  return MadeChange;
}

/// isFormingBranchFromSelectProfitable - Returns true if a SelectInst should be
/// turned into an explicit branch.
static bool isFormingBranchFromSelectProfitable(SelectInst *SI) {
  // FIXME: This should use the same heuristics as IfConversion to determine
  // whether a select is better represented as a branch.  This requires that
  // branch probability metadata is preserved for the select, which is not the
  // case currently.

  CmpInst *Cmp = dyn_cast<CmpInst>(SI->getCondition());

  // If the branch is predicted right, an out of order CPU can avoid blocking on
  // the compare.  Emit cmovs on compares with a memory operand as branches to
  // avoid stalls on the load from memory.  If the compare has more than one use
  // there's probably another cmov or setcc around so it's not worth emitting a
  // branch.
  if (!Cmp)
    return false;

  Value *CmpOp0 = Cmp->getOperand(0);
  Value *CmpOp1 = Cmp->getOperand(1);

  // We check that the memory operand has one use to avoid uses of the loaded
  // value directly after the compare, making branches unprofitable.
  return Cmp->hasOneUse() &&
         ((isa<LoadInst>(CmpOp0) && CmpOp0->hasOneUse()) ||
          (isa<LoadInst>(CmpOp1) && CmpOp1->hasOneUse()));
}


/// If we have a SelectInst that will likely profit from branch prediction,
/// turn it into a branch.
bool CodeGenPrepare::OptimizeSelectInst(SelectInst *SI) {
  bool VectorCond = !SI->getCondition()->getType()->isIntegerTy(1);

  // Can we convert the 'select' to CF ?
  if (DisableSelectToBranch || OptSize || !TLI || VectorCond)
    return false;

  TargetLowering::SelectSupportKind SelectKind;
  if (VectorCond)
    SelectKind = TargetLowering::VectorMaskSelect;
  else if (SI->getType()->isVectorTy())
    SelectKind = TargetLowering::ScalarCondVectorVal;
  else
    SelectKind = TargetLowering::ScalarValSelect;

  // Do we have efficient codegen support for this kind of 'selects' ?
  if (TLI->isSelectSupported(SelectKind)) {
    // We have efficient codegen support for the select instruction.
    // Check if it is profitable to keep this 'select'.
    if (!TLI->isPredictableSelectExpensive() ||
        !isFormingBranchFromSelectProfitable(SI))
      return false;
  }

  ModifiedDT = true;

  // First, we split the block containing the select into 2 blocks.
  BasicBlock *StartBlock = SI->getParent();
  BasicBlock::iterator SplitPt = ++(BasicBlock::iterator(SI));
  BasicBlock *NextBlock = StartBlock->splitBasicBlock(SplitPt, "select.end");

  // Create a new block serving as the landing pad for the branch.
  BasicBlock *SmallBlock = BasicBlock::Create(SI->getContext(), "select.mid",
                                             NextBlock->getParent(), NextBlock);

  // Move the unconditional branch from the block with the select in it into our
  // landing pad block.
  StartBlock->getTerminator()->eraseFromParent();
  BranchInst::Create(NextBlock, SmallBlock);

  // Insert the real conditional branch based on the original condition.
  BranchInst::Create(NextBlock, SmallBlock, SI->getCondition(), SI);

  // The select itself is replaced with a PHI Node.
  PHINode *PN = PHINode::Create(SI->getType(), 2, "", NextBlock->begin());
  PN->takeName(SI);
  PN->addIncoming(SI->getTrueValue(), StartBlock);
  PN->addIncoming(SI->getFalseValue(), SmallBlock);
  SI->replaceAllUsesWith(PN);
  SI->eraseFromParent();

  // Instruct OptimizeBlock to skip to the next block.
  CurInstIterator = StartBlock->end();
  ++NumSelectsExpanded;
  return true;
}

bool CodeGenPrepare::OptimizeInst(Instruction *I) {
  if (PHINode *P = dyn_cast<PHINode>(I)) {
    // It is possible for very late stage optimizations (such as SimplifyCFG)
    // to introduce PHI nodes too late to be cleaned up.  If we detect such a
    // trivial PHI, go ahead and zap it here.
    if (Value *V = SimplifyInstruction(P)) {
      P->replaceAllUsesWith(V);
      P->eraseFromParent();
      ++NumPHIsElim;
      return true;
    }
    return false;
  }

  if (CastInst *CI = dyn_cast<CastInst>(I)) {
    // If the source of the cast is a constant, then this should have
    // already been constant folded.  The only reason NOT to constant fold
    // it is if something (e.g. LSR) was careful to place the constant
    // evaluation in a block other than then one that uses it (e.g. to hoist
    // the address of globals out of a loop).  If this is the case, we don't
    // want to forward-subst the cast.
    if (isa<Constant>(CI->getOperand(0)))
      return false;

    if (TLI && OptimizeNoopCopyExpression(CI, *TLI))
      return true;

    if (isa<ZExtInst>(I) || isa<SExtInst>(I)) {
      bool MadeChange = MoveExtToFormExtLoad(I);
      return MadeChange | OptimizeExtUses(I);
    }
    return false;
  }

  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    return OptimizeCmpExpression(CI);

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (TLI)
      return OptimizeMemoryInst(I, I->getOperand(0), LI->getType());
    return false;
  }

  if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (TLI)
      return OptimizeMemoryInst(I, SI->getOperand(1),
                                SI->getOperand(0)->getType());
    return false;
  }

  if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
    if (GEPI->hasAllZeroIndices()) {
      /// The GEP operand must be a pointer, so must its result -> BitCast
      Instruction *NC = new BitCastInst(GEPI->getOperand(0), GEPI->getType(),
                                        GEPI->getName(), GEPI);
      GEPI->replaceAllUsesWith(NC);
      GEPI->eraseFromParent();
      ++NumGEPsElim;
      OptimizeInst(NC);
      return true;
    }
    return false;
  }

  if (CallInst *CI = dyn_cast<CallInst>(I))
    return OptimizeCallInst(CI);

  if (SelectInst *SI = dyn_cast<SelectInst>(I))
    return OptimizeSelectInst(SI);

  return false;
}

// In this pass we look for GEP and cast instructions that are used
// across basic blocks and rewrite them to improve basic-block-at-a-time
// selection.
bool CodeGenPrepare::OptimizeBlock(BasicBlock &BB) {
  SunkAddrs.clear();
  bool MadeChange = false;

  CurInstIterator = BB.begin();
  while (CurInstIterator != BB.end())
    MadeChange |= OptimizeInst(CurInstIterator++);

  MadeChange |= DupRetToEnableTailCallOpts(&BB);

  return MadeChange;
}

// llvm.dbg.value is far away from the value then iSel may not be able
// handle it properly. iSel will drop llvm.dbg.value if it can not
// find a node corresponding to the value.
bool CodeGenPrepare::PlaceDbgValues(Function &F) {
  bool MadeChange = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    Instruction *PrevNonDbgInst = NULL;
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      Instruction *Insn = BI; ++BI;
      DbgValueInst *DVI = dyn_cast<DbgValueInst>(Insn);
      if (!DVI) {
        PrevNonDbgInst = Insn;
        continue;
      }

      Instruction *VI = dyn_cast_or_null<Instruction>(DVI->getValue());
      if (VI && VI != PrevNonDbgInst && !VI->isTerminator()) {
        DEBUG(dbgs() << "Moving Debug Value before :\n" << *DVI << ' ' << *VI);
        DVI->removeFromParent();
        if (isa<PHINode>(VI))
          DVI->insertBefore(VI->getParent()->getFirstInsertionPt());
        else
          DVI->insertAfter(VI);
        MadeChange = true;
        ++NumDbgValueMoved;
      }
    }
  }
  return MadeChange;
}
