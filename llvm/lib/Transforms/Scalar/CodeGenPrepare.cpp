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
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/DominatorInternals.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ProfileInfo.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/AddrModeMatcher.h"
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
    DenseMap<Value*, Value*> SunkAddrs;

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
    bool DupRetToEnableTailCallOpts(ReturnInst *RI);
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
  OptSize = F.getFnAttributes().hasAttribute(Attributes::OptimizeForSize);

  /// This optimization identifies DIV instructions that can be
  /// profitably bypassed and carried out with a shorter, faster divide.
  if (TLI && TLI->isSlowDivBypassed()) {
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

    for (SmallPtrSet<BasicBlock*, 8>::iterator
           I = WorkList.begin(), E = WorkList.end(); I != E; ++I)
      DeleteDeadBlock(*I);

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
bool CodeGenPrepare::DupRetToEnableTailCallOpts(ReturnInst *RI) {
  if (!TLI)
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

  BasicBlock *BB = RI->getParent();
  if (PN && PN->getParent() != BB)
    return false;

  // It's not safe to eliminate the sign / zero extension of the return value.
  // See llvm::isInTailCallPosition().
  const Function *F = BB->getParent();
  Attributes CallerRetAttr = F->getAttributes().getRetAttributes();
  if (CallerRetAttr.hasAttribute(Attributes::ZExt) ||
      CallerRetAttr.hasAttribute(Attributes::SExt))
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
    Attributes CalleeRetAttr = CS.getAttributes().getRetAttributes();
    if (AttrBuilder(CalleeRetAttr).
          removeAttribute(Attributes::NoAlias) !=
        AttrBuilder(CallerRetAttr).
          removeAttribute(Attributes::NoAlias))
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
    } else {
      // This address is now available for reassignment, so erase the table
      // entry; we don't want to match some completely different instruction.
      SunkAddrs[Addr] = 0;
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

  // Make sure non of the uses are PHI nodes.
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

  if (ReturnInst *RI = dyn_cast<ReturnInst>(I))
    return DupRetToEnableTailCallOpts(RI);

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
