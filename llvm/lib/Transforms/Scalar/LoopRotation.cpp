//===- LoopRotation.cpp - Loop Rotation Pass ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Rotation Pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-rotate"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

#define MAX_HEADER_SIZE 16

STATISTIC(NumRotated, "Number of loops rotated");
namespace {

  class VISIBILITY_HIDDEN RenameData {
  public:
    RenameData(Instruction *O, Value *P, Instruction *H) 
      : Original(O), PreHeader(P), Header(H) { }
  public:
    Instruction *Original; // Original instruction
    Value *PreHeader; // Original pre-header replacement
    Instruction *Header; // New header replacement
  };
  
  class VISIBILITY_HIDDEN LoopRotate : public LoopPass {

  public:
    static char ID; // Pass ID, replacement for typeid
    LoopRotate() : LoopPass(&ID) {}

    // Rotate Loop L as many times as possible. Return true if
    // loop is rotated at least once.
    bool runOnLoop(Loop *L, LPPassManager &LPM);

    // LCSSA form makes instruction renaming easier.
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(LoopSimplifyID);
      AU.addPreservedID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);
      AU.addPreservedID(LCSSAID);
      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<LoopInfo>();
      AU.addPreserved<DominatorTree>();
      AU.addPreserved<DominanceFrontier>();
    }

    // Helper functions

    /// Do actual work
    bool rotateLoop(Loop *L, LPPassManager &LPM);
    
    /// Initialize local data
    void initialize();

    /// Make sure all Exit block PHINodes have required incoming values.
    /// If incoming value is constant or defined outside the loop then
    /// PHINode may not have an entry for original pre-header. 
    void  updateExitBlock();

    /// Return true if this instruction is used outside original header.
    bool usedOutsideOriginalHeader(Instruction *In);

    /// Find Replacement information for instruction. Return NULL if it is
    /// not available.
    const RenameData *findReplacementData(Instruction *I);

    /// After loop rotation, loop pre-header has multiple sucessors.
    /// Insert one forwarding basic block to ensure that loop pre-header
    /// has only one successor.
    void preserveCanonicalLoopForm(LPPassManager &LPM);

  private:

    Loop *L;
    BasicBlock *OrigHeader;
    BasicBlock *OrigPreHeader;
    BasicBlock *OrigLatch;
    BasicBlock *NewHeader;
    BasicBlock *Exit;
    LPPassManager *LPM_Ptr;
    SmallVector<RenameData, MAX_HEADER_SIZE> LoopHeaderInfo;
  };
}
  
char LoopRotate::ID = 0;
static RegisterPass<LoopRotate> X("loop-rotate", "Rotate Loops");

Pass *llvm::createLoopRotatePass() { return new LoopRotate(); }

/// Rotate Loop L as many times as possible. Return true if
/// loop is rotated at least once.
bool LoopRotate::runOnLoop(Loop *Lp, LPPassManager &LPM) {

  bool RotatedOneLoop = false;
  initialize();
  LPM_Ptr = &LPM;

  // One loop can be rotated multiple times.
  while (rotateLoop(Lp,LPM)) {
    RotatedOneLoop = true;
    initialize();
  }

  return RotatedOneLoop;
}

/// Rotate loop LP. Return true if the loop is rotated.
bool LoopRotate::rotateLoop(Loop *Lp, LPPassManager &LPM) {
  L = Lp;

  OrigHeader =  L->getHeader();
  OrigPreHeader = L->getLoopPreheader();
  OrigLatch = L->getLoopLatch();

  // If loop has only one block then there is not much to rotate.
  if (L->getBlocks().size() == 1)
    return false;

  assert(OrigHeader && OrigLatch && OrigPreHeader &&
         "Loop is not in canonical form");

  // If loop header is not one of the loop exit block then
  // either this loop is already rotated or it is not 
  // suitable for loop rotation transformations.
  if (!L->isLoopExit(OrigHeader))
    return false;

  BranchInst *BI = dyn_cast<BranchInst>(OrigHeader->getTerminator());
  if (!BI)
    return false;
  assert(BI->isConditional() && "Branch Instruction is not conditional");

  // Updating PHInodes in loops with multiple exits adds complexity. 
  // Keep it simple, and restrict loop rotation to loops with one exit only.
  // In future, lift this restriction and support for multiple exits if
  // required.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() > 1)
    return false;

  // Check size of original header and reject
  // loop if it is very big.
  if (OrigHeader->size() > MAX_HEADER_SIZE)
    return false;

  // Now, this loop is suitable for rotation.

  // Find new Loop header. NewHeader is a Header's one and only successor
  // that is inside loop.  Header's other successor is outside the
  // loop.  Otherwise loop is not suitable for rotation.
  Exit = BI->getSuccessor(0);
  NewHeader = BI->getSuccessor(1);
  if (L->contains(Exit))
    std::swap(Exit, NewHeader);
  assert(NewHeader && "Unable to determine new loop header");
  assert(L->contains(NewHeader) && !L->contains(Exit) && 
         "Unable to determine loop header and exit blocks");
  
  // This code assumes that new header has exactly one predecessor.  Remove any
  // single entry PHI nodes in it.
  assert(NewHeader->getSinglePredecessor() &&
         "New header doesn't have one pred!");
  FoldSingleEntryPHINodes(NewHeader);

  // Copy PHI nodes and other instructions from original header
  // into original pre-header. Unlike original header, original pre-header is
  // not a member of loop. 
  //
  // New loop header is one and only successor of original header that 
  // is inside the loop. All other original header successors are outside 
  // the loop. Copy PHI Nodes from original header into new loop header. 
  // Add second incoming value, from original loop pre-header into these phi 
  // nodes. If a value defined in original header is used outside original 
  // header then new loop header will need new phi nodes with two incoming 
  // values, one definition from original header and second definition is 
  // from original loop pre-header.

  // Remove terminator from Original pre-header. Original pre-header will
  // receive a clone of original header terminator as a new terminator.
  OrigPreHeader->getInstList().pop_back();
  BasicBlock::iterator I = OrigHeader->begin(), E = OrigHeader->end();
  PHINode *PN = 0;
  for (; (PN = dyn_cast<PHINode>(I)); ++I) {
    // PHI nodes are not copied into original pre-header. Instead their values
    // are directly propagated.
    Value *NPV = PN->getIncomingValueForBlock(OrigPreHeader);

    // Create new PHI node with two incoming values for NewHeader.
    // One incoming value is from OrigLatch (through OrigHeader) and 
    // second incoming value is from original pre-header.
    PHINode *NH = PHINode::Create(PN->getType(), PN->getName(),
                                  NewHeader->begin());
    NH->addIncoming(PN->getIncomingValueForBlock(OrigLatch), OrigHeader);
    NH->addIncoming(NPV, OrigPreHeader);
    
    // "In" can be replaced by NH at various places.
    LoopHeaderInfo.push_back(RenameData(PN, NPV, NH));
  }

  // Now, handle non-phi instructions.
  for (; I != E; ++I) {
    Instruction *In = I;
    assert(!isa<PHINode>(In) && "PHINode is not expected here");
    
    // This is not a PHI instruction. Insert its clone into original pre-header.
    // If this instruction is using a value from same basic block then
    // update it to use value from cloned instruction.
    Instruction *C = In->clone();
    C->setName(In->getName());
    OrigPreHeader->getInstList().push_back(C);

    for (unsigned opi = 0, e = In->getNumOperands(); opi != e; ++opi) {
      Instruction *OpInsn = dyn_cast<Instruction>(In->getOperand(opi));
      if (!OpInsn) continue;  // Ignore non-instruction values.
      if (const RenameData *D = findReplacementData(OpInsn))
        C->setOperand(opi, D->PreHeader);
    }

    // If this instruction is used outside this basic block then
    // create new PHINode for this instruction.
    Instruction *NewHeaderReplacement = NULL;
    if (usedOutsideOriginalHeader(In)) {
      PHINode *PN = PHINode::Create(In->getType(), In->getName(),
                                    NewHeader->begin());
      PN->addIncoming(In, OrigHeader);
      PN->addIncoming(C, OrigPreHeader);
      NewHeaderReplacement = PN;
    }
    LoopHeaderInfo.push_back(RenameData(In, C, NewHeaderReplacement));
  }

  // Rename uses of original header instructions to reflect their new
  // definitions (either from original pre-header node or from newly created
  // new header PHINodes.
  //
  // Original header instructions are used in
  // 1) Original header:
  //
  //    If instruction is used in non-phi instructions then it is using
  //    defintion from original heder iteself. Do not replace this use
  //    with definition from new header or original pre-header.
  //
  //    If instruction is used in phi node then it is an incoming 
  //    value. Rename its use to reflect new definition from new-preheader
  //    or new header.
  //
  // 2) Inside loop but not in original header
  //
  //    Replace this use to reflect definition from new header.
  for (unsigned LHI = 0, LHI_E = LoopHeaderInfo.size(); LHI != LHI_E; ++LHI) {
    const RenameData &ILoopHeaderInfo = LoopHeaderInfo[LHI];

    if (!ILoopHeaderInfo.Header)
      continue;

    Instruction *OldPhi = ILoopHeaderInfo.Original;
    Instruction *NewPhi = ILoopHeaderInfo.Header;

    // Before replacing uses, collect them first, so that iterator is
    // not invalidated.
    SmallVector<Instruction *, 16> AllUses;
    for (Value::use_iterator UI = OldPhi->use_begin(), UE = OldPhi->use_end();
         UI != UE; ++UI)
      AllUses.push_back(cast<Instruction>(UI));

    for (SmallVector<Instruction *, 16>::iterator UI = AllUses.begin(), 
           UE = AllUses.end(); UI != UE; ++UI) {
      Instruction *U = *UI;
      BasicBlock *Parent = U->getParent();

      // Used inside original header
      if (Parent == OrigHeader) {
        // Do not rename uses inside original header non-phi instructions.
        PHINode *PU = dyn_cast<PHINode>(U);
        if (!PU)
          continue;

        // Do not rename uses inside original header phi nodes, if the
        // incoming value is for new header.
        if (PU->getBasicBlockIndex(NewHeader) != -1
            && PU->getIncomingValueForBlock(NewHeader) == U)
          continue;
        
       U->replaceUsesOfWith(OldPhi, NewPhi);
       continue;
      }

      // Used inside loop, but not in original header.
      if (L->contains(U->getParent())) {
        if (U != NewPhi)
          U->replaceUsesOfWith(OldPhi, NewPhi);
        continue;
      }
      
      // Used inside Exit Block. Since we are in LCSSA form, U must be PHINode.
      if (U->getParent() == Exit) {
        assert(isa<PHINode>(U) && "Use in Exit Block that is not PHINode");
        
        PHINode *UPhi = cast<PHINode>(U);
        // UPhi already has one incoming argument from original header. 
        // Add second incoming argument from new Pre header.
        UPhi->addIncoming(ILoopHeaderInfo.PreHeader, OrigPreHeader);
      } else {
        // Used outside Exit block. Create a new PHI node from exit block
        // to receive value from ne new header ane pre header.
        PHINode *PN = PHINode::Create(U->getType(), U->getName(),
                                      Exit->begin());
        PN->addIncoming(ILoopHeaderInfo.PreHeader, OrigPreHeader);
        PN->addIncoming(OldPhi, OrigHeader);
        U->replaceUsesOfWith(OldPhi, PN);
      }
    }
  }
  
  /// Make sure all Exit block PHINodes have required incoming values.
  updateExitBlock();

  // Update CFG

  // Removing incoming branch from loop preheader to original header.
  // Now original header is inside the loop.
  for (BasicBlock::iterator I = OrigHeader->begin(), E = OrigHeader->end();
       I != E; ++I)
    if (PHINode *PN = dyn_cast<PHINode>(I))
      PN->removeIncomingValue(OrigPreHeader);

  // Make NewHeader as the new header for the loop.
  L->moveToHeader(NewHeader);

  preserveCanonicalLoopForm(LPM);

  NumRotated++;
  return true;
}

/// Make sure all Exit block PHINodes have required incoming values.
/// If incoming value is constant or defined outside the loop then
/// PHINode may not have an entry for original pre-header. 
void LoopRotate::updateExitBlock() {

  for (BasicBlock::iterator I = Exit->begin(), E = Exit->end();
       I != E; ++I) {

    PHINode *PN = dyn_cast<PHINode>(I);
    if (!PN)
      break;

    // There is already one incoming value from original pre-header block.
    if (PN->getBasicBlockIndex(OrigPreHeader) != -1)
      continue;

    const RenameData *ILoopHeaderInfo;
    Value *V = PN->getIncomingValueForBlock(OrigHeader);
    if (isa<Instruction>(V) && 
        (ILoopHeaderInfo = findReplacementData(cast<Instruction>(V)))) {
      assert(ILoopHeaderInfo->PreHeader && "Missing New Preheader Instruction");
      PN->addIncoming(ILoopHeaderInfo->PreHeader, OrigPreHeader);
    } else {
      PN->addIncoming(V, OrigPreHeader);
    }
  }
}

/// Initialize local data
void LoopRotate::initialize() {
  L = NULL;
  OrigHeader = NULL;
  OrigPreHeader = NULL;
  NewHeader = NULL;
  Exit = NULL;

  LoopHeaderInfo.clear();
}

/// Return true if this instruction is used by any instructions in the loop that
/// aren't in original header.
bool LoopRotate::usedOutsideOriginalHeader(Instruction *In) {
  for (Value::use_iterator UI = In->use_begin(), UE = In->use_end();
       UI != UE; ++UI) {
    BasicBlock *UserBB = cast<Instruction>(UI)->getParent();
    if (UserBB != OrigHeader && L->contains(UserBB))
      return true;
  }

  return false;
}

/// Find Replacement information for instruction. Return NULL if it is
/// not available.
const RenameData *LoopRotate::findReplacementData(Instruction *In) {

  // Since LoopHeaderInfo is small, linear walk is OK.
  for (unsigned LHI = 0, LHI_E = LoopHeaderInfo.size(); LHI != LHI_E; ++LHI) {
    const RenameData &ILoopHeaderInfo = LoopHeaderInfo[LHI];
    if (ILoopHeaderInfo.Original == In)
      return &ILoopHeaderInfo;
  }
  return NULL;
}

/// After loop rotation, loop pre-header has multiple sucessors.
/// Insert one forwarding basic block to ensure that loop pre-header
/// has only one successor.
void LoopRotate::preserveCanonicalLoopForm(LPPassManager &LPM) {

  // Right now original pre-header has two successors, new header and
  // exit block. Insert new block between original pre-header and
  // new header such that loop's new pre-header has only one successor.
  BasicBlock *NewPreHeader = BasicBlock::Create("bb.nph",
                                                OrigHeader->getParent(), 
                                                NewHeader);
  LoopInfo &LI = LPM.getAnalysis<LoopInfo>();
  if (Loop *PL = LI.getLoopFor(OrigPreHeader))
    PL->addBasicBlockToLoop(NewPreHeader, LI.getBase());
  BranchInst::Create(NewHeader, NewPreHeader);
  
  BranchInst *OrigPH_BI = cast<BranchInst>(OrigPreHeader->getTerminator());
  if (OrigPH_BI->getSuccessor(0) == NewHeader)
    OrigPH_BI->setSuccessor(0, NewPreHeader);
  else {
    assert(OrigPH_BI->getSuccessor(1) == NewHeader &&
           "Unexpected original pre-header terminator");
    OrigPH_BI->setSuccessor(1, NewPreHeader);
  }
  
  for (BasicBlock::iterator I = NewHeader->begin(), E = NewHeader->end();
       I != E; ++I) {
    PHINode *PN = dyn_cast<PHINode>(I);
    if (!PN)
      break;

    int index = PN->getBasicBlockIndex(OrigPreHeader);
    assert(index != -1 && "Expected incoming value from Original PreHeader");
    PN->setIncomingBlock(index, NewPreHeader);
    assert(PN->getBasicBlockIndex(OrigPreHeader) == -1 && 
           "Expected only one incoming value from Original PreHeader");
  }

  if (DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>()) {
    DT->addNewBlock(NewPreHeader, OrigPreHeader);
    DT->changeImmediateDominator(L->getHeader(), NewPreHeader);
    DT->changeImmediateDominator(Exit, OrigPreHeader);
    for (Loop::block_iterator BI = L->block_begin(), BE = L->block_end();
         BI != BE; ++BI) {
      BasicBlock *B = *BI;
      if (L->getHeader() != B) {
        DomTreeNode *Node = DT->getNode(B);
        if (Node && Node->getBlock() == OrigHeader)
          DT->changeImmediateDominator(*BI, L->getHeader());
      }
    }
    DT->changeImmediateDominator(OrigHeader, OrigLatch);
  }

  if (DominanceFrontier *DF = getAnalysisIfAvailable<DominanceFrontier>()) {
    // New Preheader's dominance frontier is Exit block.
    DominanceFrontier::DomSetType NewPHSet;
    NewPHSet.insert(Exit);
    DF->addBasicBlock(NewPreHeader, NewPHSet);

    // New Header's dominance frontier now includes itself and Exit block
    DominanceFrontier::iterator HeadI = DF->find(L->getHeader());
    if (HeadI != DF->end()) {
      DominanceFrontier::DomSetType & HeaderSet = HeadI->second;
      HeaderSet.clear();
      HeaderSet.insert(L->getHeader());
      HeaderSet.insert(Exit);
    } else {
      DominanceFrontier::DomSetType HeaderSet;
      HeaderSet.insert(L->getHeader());
      HeaderSet.insert(Exit);
      DF->addBasicBlock(L->getHeader(), HeaderSet);
    }

    // Original header (new Loop Latch)'s dominance frontier is Exit.
    DominanceFrontier::iterator LatchI = DF->find(L->getLoopLatch());
    if (LatchI != DF->end()) {
      DominanceFrontier::DomSetType &LatchSet = LatchI->second;
      LatchSet = LatchI->second;
      LatchSet.clear();
      LatchSet.insert(Exit);
    } else {
      DominanceFrontier::DomSetType LatchSet;
      LatchSet.insert(Exit);
      DF->addBasicBlock(L->getHeader(), LatchSet);
    }

    // If a loop block dominates new loop latch then its frontier is
    // new header and Exit.
    BasicBlock *NewLatch = L->getLoopLatch();
    DominatorTree *DT = getAnalysisIfAvailable<DominatorTree>();
    for (Loop::block_iterator BI = L->block_begin(), BE = L->block_end();
         BI != BE; ++BI) {
      BasicBlock *B = *BI;
      if (DT->dominates(B, NewLatch)) {
        DominanceFrontier::iterator BDFI = DF->find(B);
        if (BDFI != DF->end()) {
          DominanceFrontier::DomSetType &BSet = BDFI->second;
          BSet = BDFI->second;
          BSet.clear();
          BSet.insert(L->getHeader());
          BSet.insert(Exit);
        } else {
          DominanceFrontier::DomSetType BSet;
          BSet.insert(L->getHeader());
          BSet.insert(Exit);
          DF->addBasicBlock(B, BSet);
        }
      }
    }
  }

  // Preserve canonical loop form, which means Exit block should
  // have only one predecessor.
  BasicBlock *NExit = SplitEdge(L->getLoopLatch(), Exit, this);

  // Preserve LCSSA.
  BasicBlock::iterator I = Exit->begin(), E = Exit->end();
  PHINode *PN = NULL;
  for (; (PN = dyn_cast<PHINode>(I)); ++I) {
    unsigned N = PN->getNumIncomingValues();
    for (unsigned index = 0; index < N; ++index)
      if (PN->getIncomingBlock(index) == NExit) {
        PHINode *NewPN = PHINode::Create(PN->getType(), PN->getName(),
                                         NExit->begin());
        NewPN->addIncoming(PN->getIncomingValue(index), L->getLoopLatch());
        PN->setIncomingValue(index, NewPN);
        PN->setIncomingBlock(index, NExit);
        break;
      }
  }

  assert(NewHeader && L->getHeader() == NewHeader &&
         "Invalid loop header after loop rotation");
  assert(NewPreHeader && L->getLoopPreheader() == NewPreHeader &&
         "Invalid loop preheader after loop rotation");
  assert(L->getLoopLatch() &&
         "Invalid loop latch after loop rotation");
}
