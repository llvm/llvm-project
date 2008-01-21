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
// SelectionDAG-based code generation.  This works around limitations in it's
// basic-block-at-a-time approach.  It should eventually be removed.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "codegenprepare"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
using namespace llvm;

namespace {
  cl::opt<bool> OptExtUses("optimize-ext-uses",
                           cl::init(true), cl::Hidden);
}

namespace {  
  class VISIBILITY_HIDDEN CodeGenPrepare : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// transformation profitability.
    const TargetLowering *TLI;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit CodeGenPrepare(const TargetLowering *tli = 0)
      : FunctionPass((intptr_t)&ID), TLI(tli) {}
    bool runOnFunction(Function &F);
    
  private:
    bool EliminateMostlyEmptyBlocks(Function &F);
    bool CanMergeBlocks(const BasicBlock *BB, const BasicBlock *DestBB) const;
    void EliminateMostlyEmptyBlock(BasicBlock *BB);
    bool OptimizeBlock(BasicBlock &BB);
    bool OptimizeLoadStoreInst(Instruction *I, Value *Addr,
                               const Type *AccessTy,
                               DenseMap<Value*,Value*> &SunkAddrs);
    bool OptimizeExtUses(Instruction *I);
  };
}

char CodeGenPrepare::ID = 0;
static RegisterPass<CodeGenPrepare> X("codegenprepare",
                                      "Optimize for code generation");

FunctionPass *llvm::createCodeGenPreparePass(const TargetLowering *TLI) {
  return new CodeGenPrepare(TLI);
}


bool CodeGenPrepare::runOnFunction(Function &F) {
  bool EverMadeChange = false;
  
  // First pass, eliminate blocks that contain only PHI nodes and an
  // unconditional branch.
  EverMadeChange |= EliminateMostlyEmptyBlocks(F);
  
  bool MadeChange = true;
  while (MadeChange) {
    MadeChange = false;
    for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
      MadeChange |= OptimizeBlock(*BB);
    EverMadeChange |= MadeChange;
  }
  return EverMadeChange;
}

/// EliminateMostlyEmptyBlocks - eliminate blocks that contain only PHI nodes
/// and an unconditional branch.  Passes before isel (e.g. LSR/loopsimplify) 
/// often split edges in ways that are non-optimal for isel.  Start by
/// eliminating these blocks so we can split them the way we want them.
bool CodeGenPrepare::EliminateMostlyEmptyBlocks(Function &F) {
  bool MadeChange = false;
  // Note that this intentionally skips the entry block.
  for (Function::iterator I = ++F.begin(), E = F.end(); I != E; ) {
    BasicBlock *BB = I++;

    // If this block doesn't end with an uncond branch, ignore it.
    BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isUnconditional())
      continue;
    
    // If the instruction before the branch isn't a phi node, then other stuff
    // is happening here.
    BasicBlock::iterator BBI = BI;
    if (BBI != BB->begin()) {
      --BBI;
      if (!isa<PHINode>(BBI)) continue;
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
    for (Value::use_const_iterator UI = PN->use_begin(), E = PN->use_end();
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
  
  DOUT << "MERGING MOSTLY EMPTY BLOCKS - BEFORE:\n" << *BB << *DestBB;
  
  // If the destination block has a single pred, then this is a trivial edge,
  // just collapse it.
  if (DestBB->getSinglePredecessor()) {
    // If DestBB has single-entry PHI nodes, fold them.
    while (PHINode *PN = dyn_cast<PHINode>(DestBB->begin())) {
      PN->replaceAllUsesWith(PN->getIncomingValue(0));
      PN->eraseFromParent();
    }
    
    // Splice all the PHI nodes from BB over to DestBB.
    DestBB->getInstList().splice(DestBB->begin(), BB->getInstList(),
                                 BB->begin(), BI);
    
    // Anything that branched to BB now branches to DestBB.
    BB->replaceAllUsesWith(DestBB);
    
    // Nuke BB.
    BB->eraseFromParent();
    
    DOUT << "AFTER:\n" << *DestBB << "\n\n\n";
    return;
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
  BB->eraseFromParent();
  
  DOUT << "AFTER:\n" << *DestBB << "\n\n\n";
}


/// SplitEdgeNicely - Split the critical edge from TI to its specified
/// successor if it will improve codegen.  We only do this if the successor has
/// phi nodes (otherwise critical edges are ok).  If there is already another
/// predecessor of the succ that is empty (and thus has no phi nodes), use it
/// instead of introducing a new block.
static void SplitEdgeNicely(TerminatorInst *TI, unsigned SuccNum, Pass *P) {
  BasicBlock *TIBB = TI->getParent();
  BasicBlock *Dest = TI->getSuccessor(SuccNum);
  assert(isa<PHINode>(Dest->begin()) &&
         "This should only be called if Dest has a PHI!");
  
  // As a hack, never split backedges of loops.  Even though the copy for any
  // PHIs inserted on the backedge would be dead for exits from the loop, we
  // assume that the cost of *splitting* the backedge would be too high.
  if (Dest == TIBB)
    return;
  
  /// TIPHIValues - This array is lazily computed to determine the values of
  /// PHIs in Dest that TI would provide.
  SmallVector<Value*, 32> TIPHIValues;
  
  // Check to see if Dest has any blocks that can be used as a split edge for
  // this terminator.
  for (pred_iterator PI = pred_begin(Dest), E = pred_end(Dest); PI != E; ++PI) {
    BasicBlock *Pred = *PI;
    // To be usable, the pred has to end with an uncond branch to the dest.
    BranchInst *PredBr = dyn_cast<BranchInst>(Pred->getTerminator());
    if (!PredBr || !PredBr->isUnconditional() ||
        // Must be empty other than the branch.
        &Pred->front() != PredBr ||
        // Cannot be the entry block; its label does not get emitted.
        Pred == &(Dest->getParent()->getEntryBlock()))
      continue;
    
    // Finally, since we know that Dest has phi nodes in it, we have to make
    // sure that jumping to Pred will have the same affect as going to Dest in
    // terms of PHI values.
    PHINode *PN;
    unsigned PHINo = 0;
    bool FoundMatch = true;
    for (BasicBlock::iterator I = Dest->begin();
         (PN = dyn_cast<PHINode>(I)); ++I, ++PHINo) {
      if (PHINo == TIPHIValues.size())
        TIPHIValues.push_back(PN->getIncomingValueForBlock(TIBB));
      
      // If the PHI entry doesn't work, we can't use this pred.
      if (TIPHIValues[PHINo] != PN->getIncomingValueForBlock(Pred)) {
        FoundMatch = false;
        break;
      }
    }
    
    // If we found a workable predecessor, change TI to branch to Succ.
    if (FoundMatch) {
      Dest->removePredecessor(TIBB);
      TI->setSuccessor(SuccNum, Pred);
      return;
    }
  }
  
  SplitCriticalEdge(TI, SuccNum, P, true);  
}

/// OptimizeNoopCopyExpression - If the specified cast instruction is a noop
/// copy (e.g. it's casting from one pointer type to another, int->uint, or
/// int->sbyte on PPC), sink it into user blocks to reduce the number of virtual
/// registers that must be created and coalesced.
///
/// Return true if any changes are made.
static bool OptimizeNoopCopyExpression(CastInst *CI, const TargetLowering &TLI){
  // If this is a noop copy, 
  MVT::ValueType SrcVT = TLI.getValueType(CI->getOperand(0)->getType());
  MVT::ValueType DstVT = TLI.getValueType(CI->getType());
  
  // This is an fp<->int conversion?
  if (MVT::isInteger(SrcVT) != MVT::isInteger(DstVT))
    return false;
  
  // If this is an extension, it will be a zero or sign extension, which
  // isn't a noop.
  if (SrcVT < DstVT) return false;
  
  // If these values will be promoted, find out what they will be promoted
  // to.  This helps us consider truncates on PPC as noop copies when they
  // are.
  if (TLI.getTypeAction(SrcVT) == TargetLowering::Promote)
    SrcVT = TLI.getTypeToTransformTo(SrcVT);
  if (TLI.getTypeAction(DstVT) == TargetLowering::Promote)
    DstVT = TLI.getTypeToTransformTo(DstVT);
  
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
      unsigned OpVal = UI.getOperandNo()/2;
      UserBB = PN->getIncomingBlock(OpVal);
    }
    
    // Preincrement use iterator so we don't invalidate it.
    ++UI;
    
    // If this user is in the same block as the cast, don't change the cast.
    if (UserBB == DefBB) continue;
    
    // If we have already inserted a cast into this block, use it.
    CastInst *&InsertedCast = InsertedCasts[UserBB];

    if (!InsertedCast) {
      BasicBlock::iterator InsertPt = UserBB->begin();
      while (isa<PHINode>(InsertPt)) ++InsertPt;
      
      InsertedCast = 
        CastInst::create(CI->getOpcode(), CI->getOperand(0), CI->getType(), "", 
                         InsertPt);
      MadeChange = true;
    }
    
    // Replace a use of the cast with a use of the new cast.
    TheUse = InsertedCast;
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
static bool OptimizeCmpExpression(CmpInst *CI){

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
      BasicBlock::iterator InsertPt = UserBB->begin();
      while (isa<PHINode>(InsertPt)) ++InsertPt;
      
      InsertedCmp = 
        CmpInst::create(CI->getOpcode(), CI->getPredicate(), CI->getOperand(0), 
                        CI->getOperand(1), "", InsertPt);
      MadeChange = true;
    }
    
    // Replace a use of the cmp with a use of the new cmp.
    TheUse = InsertedCmp;
  }
  
  // If we removed all uses, nuke the cmp.
  if (CI->use_empty())
    CI->eraseFromParent();
  
  return MadeChange;
}

/// EraseDeadInstructions - Erase any dead instructions
static void EraseDeadInstructions(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I || !I->use_empty()) return;
  
  SmallPtrSet<Instruction*, 16> Insts;
  Insts.insert(I);
  
  while (!Insts.empty()) {
    I = *Insts.begin();
    Insts.erase(I);
    if (isInstructionTriviallyDead(I)) {
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
        if (Instruction *U = dyn_cast<Instruction>(I->getOperand(i)))
          Insts.insert(U);
      I->eraseFromParent();
    }
  }
}


/// ExtAddrMode - This is an extended version of TargetLowering::AddrMode which
/// holds actual Value*'s for register values.
struct ExtAddrMode : public TargetLowering::AddrMode {
  Value *BaseReg;
  Value *ScaledReg;
  ExtAddrMode() : BaseReg(0), ScaledReg(0) {}
  void dump() const;
};

static std::ostream &operator<<(std::ostream &OS, const ExtAddrMode &AM) {
  bool NeedPlus = false;
  OS << "[";
  if (AM.BaseGV)
    OS << (NeedPlus ? " + " : "")
       << "GV:%" << AM.BaseGV->getName(), NeedPlus = true;
  
  if (AM.BaseOffs)
    OS << (NeedPlus ? " + " : "") << AM.BaseOffs, NeedPlus = true;
  
  if (AM.BaseReg)
    OS << (NeedPlus ? " + " : "")
       << "Base:%" << AM.BaseReg->getName(), NeedPlus = true;
  if (AM.Scale)
    OS << (NeedPlus ? " + " : "")
       << AM.Scale << "*%" << AM.ScaledReg->getName(), NeedPlus = true;
  
  return OS << "]";
}

void ExtAddrMode::dump() const {
  cerr << *this << "\n";
}

static bool TryMatchingScaledValue(Value *ScaleReg, int64_t Scale,
                                   const Type *AccessTy, ExtAddrMode &AddrMode,
                                   SmallVector<Instruction*, 16> &AddrModeInsts,
                                   const TargetLowering &TLI, unsigned Depth);
  
/// FindMaximalLegalAddressingMode - If we can, try to merge the computation of
/// Addr into the specified addressing mode.  If Addr can't be added to AddrMode
/// this returns false.  This assumes that Addr is either a pointer type or
/// intptr_t for the target.
static bool FindMaximalLegalAddressingMode(Value *Addr, const Type *AccessTy,
                                           ExtAddrMode &AddrMode,
                                   SmallVector<Instruction*, 16> &AddrModeInsts,
                                           const TargetLowering &TLI,
                                           unsigned Depth) {
  
  // If this is a global variable, fold it into the addressing mode if possible.
  if (GlobalValue *GV = dyn_cast<GlobalValue>(Addr)) {
    if (AddrMode.BaseGV == 0) {
      AddrMode.BaseGV = GV;
      if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
        return true;
      AddrMode.BaseGV = 0;
    }
  } else if (ConstantInt *CI = dyn_cast<ConstantInt>(Addr)) {
    AddrMode.BaseOffs += CI->getSExtValue();
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy))
      return true;
    AddrMode.BaseOffs -= CI->getSExtValue();
  } else if (isa<ConstantPointerNull>(Addr)) {
    return true;
  }
  
  // Look through constant exprs and instructions.
  unsigned Opcode = ~0U;
  User *AddrInst = 0;
  if (Instruction *I = dyn_cast<Instruction>(Addr)) {
    Opcode = I->getOpcode();
    AddrInst = I;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr)) {
    Opcode = CE->getOpcode();
    AddrInst = CE;
  }

  // Limit recursion to avoid exponential behavior.
  if (Depth == 5) { AddrInst = 0; Opcode = ~0U; }

  // If this is really an instruction, add it to our list of related
  // instructions.
  if (Instruction *I = dyn_cast_or_null<Instruction>(AddrInst))
    AddrModeInsts.push_back(I);

  switch (Opcode) {
  case Instruction::PtrToInt:
    // PtrToInt is always a noop, as we know that the int type is pointer sized.
    if (FindMaximalLegalAddressingMode(AddrInst->getOperand(0), AccessTy,
                                       AddrMode, AddrModeInsts, TLI, Depth))
      return true;
    break;
  case Instruction::IntToPtr:
    // This inttoptr is a no-op if the integer type is pointer sized.
    if (TLI.getValueType(AddrInst->getOperand(0)->getType()) ==
        TLI.getPointerTy()) {
      if (FindMaximalLegalAddressingMode(AddrInst->getOperand(0), AccessTy,
                                         AddrMode, AddrModeInsts, TLI, Depth))
        return true;
    }
    break;
  case Instruction::Add: {
    // Check to see if we can merge in the RHS then the LHS.  If so, we win.
    ExtAddrMode BackupAddrMode = AddrMode;
    unsigned OldSize = AddrModeInsts.size();
    if (FindMaximalLegalAddressingMode(AddrInst->getOperand(1), AccessTy,
                                       AddrMode, AddrModeInsts, TLI, Depth+1) &&
        FindMaximalLegalAddressingMode(AddrInst->getOperand(0), AccessTy,
                                       AddrMode, AddrModeInsts, TLI, Depth+1))
      return true;

    // Restore the old addr mode info.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    
    // Otherwise this was over-aggressive.  Try merging in the LHS then the RHS.
    if (FindMaximalLegalAddressingMode(AddrInst->getOperand(0), AccessTy,
                                       AddrMode, AddrModeInsts, TLI, Depth+1) &&
        FindMaximalLegalAddressingMode(AddrInst->getOperand(1), AccessTy,
                                       AddrMode, AddrModeInsts, TLI, Depth+1))
      return true;
    
    // Otherwise we definitely can't merge the ADD in.
    AddrMode = BackupAddrMode;
    AddrModeInsts.resize(OldSize);
    break;    
  }
  case Instruction::Or: {
    ConstantInt *RHS = dyn_cast<ConstantInt>(AddrInst->getOperand(1));
    if (!RHS) break;
    // TODO: We can handle "Or Val, Imm" iff this OR is equivalent to an ADD.
    break;
  }
  case Instruction::Mul:
  case Instruction::Shl: {
    // Can only handle X*C and X << C, and can only handle this when the scale
    // field is available.
    ConstantInt *RHS = dyn_cast<ConstantInt>(AddrInst->getOperand(1));
    if (!RHS) break;
    int64_t Scale = RHS->getSExtValue();
    if (Opcode == Instruction::Shl)
      Scale = 1 << Scale;
    
    if (TryMatchingScaledValue(AddrInst->getOperand(0), Scale, AccessTy,
                               AddrMode, AddrModeInsts, TLI, Depth))
      return true;
    break;
  }
  case Instruction::GetElementPtr: {
    // Scan the GEP.  We check it if it contains constant offsets and at most
    // one variable offset.
    int VariableOperand = -1;
    unsigned VariableScale = 0;
    
    int64_t ConstantOffset = 0;
    const TargetData *TD = TLI.getTargetData();
    gep_type_iterator GTI = gep_type_begin(AddrInst);
    for (unsigned i = 1, e = AddrInst->getNumOperands(); i != e; ++i, ++GTI) {
      if (const StructType *STy = dyn_cast<StructType>(*GTI)) {
        const StructLayout *SL = TD->getStructLayout(STy);
        unsigned Idx =
          cast<ConstantInt>(AddrInst->getOperand(i))->getZExtValue();
        ConstantOffset += SL->getElementOffset(Idx);
      } else {
        uint64_t TypeSize = TD->getABITypeSize(GTI.getIndexedType());
        if (ConstantInt *CI = dyn_cast<ConstantInt>(AddrInst->getOperand(i))) {
          ConstantOffset += CI->getSExtValue()*TypeSize;
        } else if (TypeSize) {  // Scales of zero don't do anything.
          // We only allow one variable index at the moment.
          if (VariableOperand != -1) {
            VariableOperand = -2;
            break;
          }
          
          // Remember the variable index.
          VariableOperand = i;
          VariableScale = TypeSize;
        }
      }
    }

    // If the GEP had multiple variable indices, punt.
    if (VariableOperand == -2)
      break;

    // A common case is for the GEP to only do a constant offset.  In this case,
    // just add it to the disp field and check validity.
    if (VariableOperand == -1) {
      AddrMode.BaseOffs += ConstantOffset;
      if (ConstantOffset == 0 || TLI.isLegalAddressingMode(AddrMode, AccessTy)){
        // Check to see if we can fold the base pointer in too.
        if (FindMaximalLegalAddressingMode(AddrInst->getOperand(0), AccessTy,
                                           AddrMode, AddrModeInsts, TLI,
                                           Depth+1))
          return true;
      }
      AddrMode.BaseOffs -= ConstantOffset;
    } else {
      // Check that this has no base reg yet.  If so, we won't have a place to
      // put the base of the GEP (assuming it is not a null ptr).
      bool SetBaseReg = false;
      if (AddrMode.HasBaseReg) {
        if (!isa<ConstantPointerNull>(AddrInst->getOperand(0)))
          break;
      } else {
        AddrMode.HasBaseReg = true;
        AddrMode.BaseReg = AddrInst->getOperand(0);
        SetBaseReg = true;
      }
      
      // See if the scale amount is valid for this target.
      AddrMode.BaseOffs += ConstantOffset;
      if (TryMatchingScaledValue(AddrInst->getOperand(VariableOperand),
                                 VariableScale, AccessTy, AddrMode, 
                                 AddrModeInsts, TLI, Depth)) {
        if (!SetBaseReg) return true;

        // If this match succeeded, we know that we can form an address with the
        // GepBase as the basereg.  See if we can match *more*.
        AddrMode.HasBaseReg = false;
        AddrMode.BaseReg = 0;
        if (FindMaximalLegalAddressingMode(AddrInst->getOperand(0), AccessTy,
                                           AddrMode, AddrModeInsts, TLI,
                                           Depth+1))
          return true;
        // Strange, shouldn't happen.  Restore the base reg and succeed the easy
        // way.        
        AddrMode.HasBaseReg = true;
        AddrMode.BaseReg = AddrInst->getOperand(0);
        return true;
      }
      
      AddrMode.BaseOffs -= ConstantOffset;
      if (SetBaseReg) {
        AddrMode.HasBaseReg = false;
        AddrMode.BaseReg = 0;
      }
    }
    break;    
  }
  }
  
  if (Instruction *I = dyn_cast_or_null<Instruction>(AddrInst)) {
    assert(AddrModeInsts.back() == I && "Stack imbalance");
    AddrModeInsts.pop_back();
  }
  
  // Worse case, the target should support [reg] addressing modes. :)
  if (!AddrMode.HasBaseReg) {
    AddrMode.HasBaseReg = true;
    // Still check for legality in case the target supports [imm] but not [i+r].
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy)) {
      AddrMode.BaseReg = Addr;
      return true;
    }
    AddrMode.HasBaseReg = false;
  }
  
  // If the base register is already taken, see if we can do [r+r].
  if (AddrMode.Scale == 0) {
    AddrMode.Scale = 1;
    if (TLI.isLegalAddressingMode(AddrMode, AccessTy)) {
      AddrMode.ScaledReg = Addr;
      return true;
    }
    AddrMode.Scale = 0;
  }
  // Couldn't match.
  return false;
}

/// TryMatchingScaledValue - Try adding ScaleReg*Scale to the specified
/// addressing mode.  Return true if this addr mode is legal for the target,
/// false if not.
static bool TryMatchingScaledValue(Value *ScaleReg, int64_t Scale,
                                   const Type *AccessTy, ExtAddrMode &AddrMode,
                                   SmallVector<Instruction*, 16> &AddrModeInsts,
                                   const TargetLowering &TLI, unsigned Depth) {
  // If we already have a scale of this value, we can add to it, otherwise, we
  // need an available scale field.
  if (AddrMode.Scale != 0 && AddrMode.ScaledReg != ScaleReg)
    return false;
  
  ExtAddrMode InputAddrMode = AddrMode;
  
  // Add scale to turn X*4+X*3 -> X*7.  This could also do things like
  // [A+B + A*7] -> [B+A*8].
  AddrMode.Scale += Scale;
  AddrMode.ScaledReg = ScaleReg;
  
  if (TLI.isLegalAddressingMode(AddrMode, AccessTy)) {
    // Okay, we decided that we can add ScaleReg+Scale to AddrMode.  Check now
    // to see if ScaleReg is actually X+C.  If so, we can turn this into adding
    // X*Scale + C*Scale to addr mode.
    BinaryOperator *BinOp = dyn_cast<BinaryOperator>(ScaleReg);
    if (BinOp && BinOp->getOpcode() == Instruction::Add &&
        isa<ConstantInt>(BinOp->getOperand(1)) && InputAddrMode.ScaledReg ==0) {
      
      InputAddrMode.Scale = Scale;
      InputAddrMode.ScaledReg = BinOp->getOperand(0);
      InputAddrMode.BaseOffs += 
        cast<ConstantInt>(BinOp->getOperand(1))->getSExtValue()*Scale;
      if (TLI.isLegalAddressingMode(InputAddrMode, AccessTy)) {
        AddrModeInsts.push_back(BinOp);
        AddrMode = InputAddrMode;
        return true;
      }
    }

    // Otherwise, not (x+c)*scale, just return what we have.
    return true;
  }
  
  // Otherwise, back this attempt out.
  AddrMode.Scale -= Scale;
  if (AddrMode.Scale == 0) AddrMode.ScaledReg = 0;
  
  return false;
}


/// IsNonLocalValue - Return true if the specified values are defined in a
/// different basic block than BB.
static bool IsNonLocalValue(Value *V, BasicBlock *BB) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() != BB;
  return false;
}

/// OptimizeLoadStoreInst - Load and Store Instructions have often have
/// addressing modes that can do significant amounts of computation.  As such,
/// instruction selection will try to get the load or store to do as much
/// computation as possible for the program.  The problem is that isel can only
/// see within a single block.  As such, we sink as much legal addressing mode
/// stuff into the block as possible.
bool CodeGenPrepare::OptimizeLoadStoreInst(Instruction *LdStInst, Value *Addr,
                                           const Type *AccessTy,
                                           DenseMap<Value*,Value*> &SunkAddrs) {
  // Figure out what addressing mode will be built up for this operation.
  SmallVector<Instruction*, 16> AddrModeInsts;
  ExtAddrMode AddrMode;
  bool Success = FindMaximalLegalAddressingMode(Addr, AccessTy, AddrMode,
                                                AddrModeInsts, *TLI, 0);
  Success = Success; assert(Success && "Couldn't select *anything*?");
  
  // Check to see if any of the instructions supersumed by this addr mode are
  // non-local to I's BB.
  bool AnyNonLocal = false;
  for (unsigned i = 0, e = AddrModeInsts.size(); i != e; ++i) {
    if (IsNonLocalValue(AddrModeInsts[i], LdStInst->getParent())) {
      AnyNonLocal = true;
      break;
    }
  }
  
  // If all the instructions matched are already in this BB, don't do anything.
  if (!AnyNonLocal) {
    DEBUG(cerr << "CGP: Found      local addrmode: " << AddrMode << "\n");
    return false;
  }
  
  // Insert this computation right after this user.  Since our caller is
  // scanning from the top of the BB to the bottom, reuse of the expr are
  // guaranteed to happen later.
  BasicBlock::iterator InsertPt = LdStInst;
  
  // Now that we determined the addressing expression we want to use and know
  // that we have to sink it into this block.  Check to see if we have already
  // done this for some other load/store instr in this block.  If so, reuse the
  // computation.
  Value *&SunkAddr = SunkAddrs[Addr];
  if (SunkAddr) {
    DEBUG(cerr << "CGP: Reusing nonlocal addrmode: " << AddrMode << "\n");
    if (SunkAddr->getType() != Addr->getType())
      SunkAddr = new BitCastInst(SunkAddr, Addr->getType(), "tmp", InsertPt);
  } else {
    DEBUG(cerr << "CGP: SINKING nonlocal addrmode: " << AddrMode << "\n");
    const Type *IntPtrTy = TLI->getTargetData()->getIntPtrType();
    
    Value *Result = 0;
    // Start with the scale value.
    if (AddrMode.Scale) {
      Value *V = AddrMode.ScaledReg;
      if (V->getType() == IntPtrTy) {
        // done.
      } else if (isa<PointerType>(V->getType())) {
        V = new PtrToIntInst(V, IntPtrTy, "sunkaddr", InsertPt);
      } else if (cast<IntegerType>(IntPtrTy)->getBitWidth() <
                 cast<IntegerType>(V->getType())->getBitWidth()) {
        V = new TruncInst(V, IntPtrTy, "sunkaddr", InsertPt);
      } else {
        V = new SExtInst(V, IntPtrTy, "sunkaddr", InsertPt);
      }
      if (AddrMode.Scale != 1)
        V = BinaryOperator::createMul(V, ConstantInt::get(IntPtrTy,
                                                          AddrMode.Scale),
                                      "sunkaddr", InsertPt);
      Result = V;
    }

    // Add in the base register.
    if (AddrMode.BaseReg) {
      Value *V = AddrMode.BaseReg;
      if (V->getType() != IntPtrTy)
        V = new PtrToIntInst(V, IntPtrTy, "sunkaddr", InsertPt);
      if (Result)
        Result = BinaryOperator::createAdd(Result, V, "sunkaddr", InsertPt);
      else
        Result = V;
    }
    
    // Add in the BaseGV if present.
    if (AddrMode.BaseGV) {
      Value *V = new PtrToIntInst(AddrMode.BaseGV, IntPtrTy, "sunkaddr",
                                  InsertPt);
      if (Result)
        Result = BinaryOperator::createAdd(Result, V, "sunkaddr", InsertPt);
      else
        Result = V;
    }
    
    // Add in the Base Offset if present.
    if (AddrMode.BaseOffs) {
      Value *V = ConstantInt::get(IntPtrTy, AddrMode.BaseOffs);
      if (Result)
        Result = BinaryOperator::createAdd(Result, V, "sunkaddr", InsertPt);
      else
        Result = V;
    }

    if (Result == 0)
      SunkAddr = Constant::getNullValue(Addr->getType());
    else
      SunkAddr = new IntToPtrInst(Result, Addr->getType(), "sunkaddr",InsertPt);
  }
  
  LdStInst->replaceUsesOfWith(Addr, SunkAddr);
  
  if (Addr->use_empty())
    EraseDeadInstructions(Addr);
  return true;
}

bool CodeGenPrepare::OptimizeExtUses(Instruction *I) {
  BasicBlock *DefBB = I->getParent();

  // If both result of the {s|z}xt and its source are live out, rewrite all
  // other uses of the source with result of extension.
  Value *Src = I->getOperand(0);
  if (Src->hasOneUse())
    return false;

  // Only do this xform if truncating is free.
  if (!TLI->isTruncateFree(I->getType(), Src->getType()))
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
      BasicBlock::iterator InsertPt = UserBB->begin();
      while (isa<PHINode>(InsertPt)) ++InsertPt;
      
      InsertedTrunc = new TruncInst(I, Src->getType(), "", InsertPt);
    }

    // Replace a use of the {s|z}ext source with a use of the result.
    TheUse = InsertedTrunc;

    MadeChange = true;
  }

  return MadeChange;
}

// In this pass we look for GEP and cast instructions that are used
// across basic blocks and rewrite them to improve basic-block-at-a-time
// selection.
bool CodeGenPrepare::OptimizeBlock(BasicBlock &BB) {
  bool MadeChange = false;
  
  // Split all critical edges where the dest block has a PHI and where the phi
  // has shared immediate operands.
  TerminatorInst *BBTI = BB.getTerminator();
  if (BBTI->getNumSuccessors() > 1) {
    for (unsigned i = 0, e = BBTI->getNumSuccessors(); i != e; ++i)
      if (isa<PHINode>(BBTI->getSuccessor(i)->begin()) &&
          isCriticalEdge(BBTI, i, true))
        SplitEdgeNicely(BBTI, i, this);
  }
  
  
  // Keep track of non-local addresses that have been sunk into this block.
  // This allows us to avoid inserting duplicate code for blocks with multiple
  // load/stores of the same address.
  DenseMap<Value*, Value*> SunkAddrs;
  
  for (BasicBlock::iterator BBI = BB.begin(), E = BB.end(); BBI != E; ) {
    Instruction *I = BBI++;
    
    if (CastInst *CI = dyn_cast<CastInst>(I)) {
      // If the source of the cast is a constant, then this should have
      // already been constant folded.  The only reason NOT to constant fold
      // it is if something (e.g. LSR) was careful to place the constant
      // evaluation in a block other than then one that uses it (e.g. to hoist
      // the address of globals out of a loop).  If this is the case, we don't
      // want to forward-subst the cast.
      if (isa<Constant>(CI->getOperand(0)))
        continue;
      
      bool Change = false;
      if (TLI) {
        Change = OptimizeNoopCopyExpression(CI, *TLI);
        MadeChange |= Change;
      }

      if (OptExtUses && !Change && (isa<ZExtInst>(I) || isa<SExtInst>(I)))
        MadeChange |= OptimizeExtUses(I);
    } else if (CmpInst *CI = dyn_cast<CmpInst>(I)) {
      MadeChange |= OptimizeCmpExpression(CI);
    } else if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      if (TLI)
        MadeChange |= OptimizeLoadStoreInst(I, I->getOperand(0), LI->getType(),
                                            SunkAddrs);
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      if (TLI)
        MadeChange |= OptimizeLoadStoreInst(I, SI->getOperand(1),
                                            SI->getOperand(0)->getType(),
                                            SunkAddrs);
    } else if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
      if (GEPI->hasAllZeroIndices()) {
        /// The GEP operand must be a pointer, so must its result -> BitCast
        Instruction *NC = new BitCastInst(GEPI->getOperand(0), GEPI->getType(), 
                                          GEPI->getName(), GEPI);
        GEPI->replaceAllUsesWith(NC);
        GEPI->eraseFromParent();
        MadeChange = true;
        BBI = NC;
      }
    } else if (CallInst *CI = dyn_cast<CallInst>(I)) {
      // If we found an inline asm expession, and if the target knows how to
      // lower it to normal LLVM code, do so now.
      if (TLI && isa<InlineAsm>(CI->getCalledValue()))
        if (const TargetAsmInfo *TAI = 
            TLI->getTargetMachine().getTargetAsmInfo()) {
          if (TAI->ExpandInlineAsm(CI))
            BBI = BB.begin();
        }
    }
  }
    
  return MadeChange;
}

