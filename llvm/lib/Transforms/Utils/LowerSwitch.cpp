//===- LowerSwitch.cpp - Eliminate Switch instructions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The LowerSwitch transformation rewrites switch instructions with a sequence
// of branches, which allows targets to get away with not implementing the
// switch instruction until it is convenient.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;

namespace {
  /// LowerSwitch Pass - Replace all SwitchInst instructions with chained branch
  /// instructions.
  class LowerSwitch : public FunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    LowerSwitch() : FunctionPass(ID) {
      initializeLowerSwitchPass(*PassRegistry::getPassRegistry());
    } 

    virtual bool runOnFunction(Function &F);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // This is a cluster of orthogonal Transforms
      AU.addPreserved<UnifyFunctionExitNodes>();
      AU.addPreserved("mem2reg");
      AU.addPreservedID(LowerInvokePassID);
    }

    struct CaseRange {
      Constant* Low;
      Constant* High;
      BasicBlock* BB;

      CaseRange(Constant *low = 0, Constant *high = 0, BasicBlock *bb = 0) :
        Low(low), High(high), BB(bb) { }
    };

    typedef std::vector<CaseRange>           CaseVector;
    typedef std::vector<CaseRange>::iterator CaseItr;
  private:
    void processSwitchInst(SwitchInst *SI);

    BasicBlock* switchConvert(CaseItr Begin, CaseItr End, Value* Val,
                              BasicBlock* OrigBlock, BasicBlock* Default);
    BasicBlock* newLeafBlock(CaseRange& Leaf, Value* Val,
                             BasicBlock* OrigBlock, BasicBlock* Default);
    unsigned Clusterify(CaseVector& Cases, SwitchInst *SI);
  };
}

char LowerSwitch::ID = 0;
INITIALIZE_PASS(LowerSwitch, "lowerswitch",
                "Lower SwitchInst's to branches", false, false)

// Publicly exposed interface to pass...
char &llvm::LowerSwitchID = LowerSwitch::ID;
// createLowerSwitchPass - Interface to this file...
FunctionPass *llvm::createLowerSwitchPass() {
  return new LowerSwitch();
}

bool LowerSwitch::runOnFunction(Function &F) {
  bool Changed = false;

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ) {
    BasicBlock *Cur = I++; // Advance over block so we don't traverse new blocks

    if (SwitchInst *SI = dyn_cast<SwitchInst>(Cur->getTerminator())) {
      Changed = true;
      processSwitchInst(SI);
    }
  }

  return Changed;
}

// operator<< - Used for debugging purposes.
//
static raw_ostream& operator<<(raw_ostream &O,
                               const LowerSwitch::CaseVector &C)
    LLVM_ATTRIBUTE_USED;
static raw_ostream& operator<<(raw_ostream &O,
                               const LowerSwitch::CaseVector &C) {
  O << "[";

  for (LowerSwitch::CaseVector::const_iterator B = C.begin(),
         E = C.end(); B != E; ) {
    O << *B->Low << " -" << *B->High;
    if (++B != E) O << ", ";
  }

  return O << "]";
}

// switchConvert - Convert the switch statement into a binary lookup of
// the case values. The function recursively builds this tree.
//
BasicBlock* LowerSwitch::switchConvert(CaseItr Begin, CaseItr End,
                                       Value* Val, BasicBlock* OrigBlock,
                                       BasicBlock* Default)
{
  unsigned Size = End - Begin;

  if (Size == 1)
    return newLeafBlock(*Begin, Val, OrigBlock, Default);

  unsigned Mid = Size / 2;
  std::vector<CaseRange> LHS(Begin, Begin + Mid);
  DEBUG(dbgs() << "LHS: " << LHS << "\n");
  std::vector<CaseRange> RHS(Begin + Mid, End);
  DEBUG(dbgs() << "RHS: " << RHS << "\n");

  CaseRange& Pivot = *(Begin + Mid);
  DEBUG(dbgs() << "Pivot ==> " 
               << cast<ConstantInt>(Pivot.Low)->getValue() << " -"
               << cast<ConstantInt>(Pivot.High)->getValue() << "\n");

  BasicBlock* LBranch = switchConvert(LHS.begin(), LHS.end(), Val,
                                      OrigBlock, Default);
  BasicBlock* RBranch = switchConvert(RHS.begin(), RHS.end(), Val,
                                      OrigBlock, Default);

  // Create a new node that checks if the value is < pivot. Go to the
  // left branch if it is and right branch if not.
  Function* F = OrigBlock->getParent();
  BasicBlock* NewNode = BasicBlock::Create(Val->getContext(), "NodeBlock");
  Function::iterator FI = OrigBlock;
  F->getBasicBlockList().insert(++FI, NewNode);

  ICmpInst* Comp = new ICmpInst(ICmpInst::ICMP_ULT,
                                Val, Pivot.Low, "Pivot");
  NewNode->getInstList().push_back(Comp);
  BranchInst::Create(LBranch, RBranch, Comp, NewNode);
  return NewNode;
}

// newLeafBlock - Create a new leaf block for the binary lookup tree. It
// checks if the switch's value == the case's value. If not, then it
// jumps to the default branch. At this point in the tree, the value
// can't be another valid case value, so the jump to the "default" branch
// is warranted.
//
BasicBlock* LowerSwitch::newLeafBlock(CaseRange& Leaf, Value* Val,
                                      BasicBlock* OrigBlock,
                                      BasicBlock* Default)
{
  Function* F = OrigBlock->getParent();
  BasicBlock* NewLeaf = BasicBlock::Create(Val->getContext(), "LeafBlock");
  Function::iterator FI = OrigBlock;
  F->getBasicBlockList().insert(++FI, NewLeaf);

  // Emit comparison
  ICmpInst* Comp = NULL;
  if (Leaf.Low == Leaf.High) {
    // Make the seteq instruction...
    Comp = new ICmpInst(*NewLeaf, ICmpInst::ICMP_EQ, Val,
                        Leaf.Low, "SwitchLeaf");
  } else {
    // Make range comparison
    if (cast<ConstantInt>(Leaf.Low)->isMinValue(true /*isSigned*/)) {
      // Val >= Min && Val <= Hi --> Val <= Hi
      Comp = new ICmpInst(*NewLeaf, ICmpInst::ICMP_SLE, Val, Leaf.High,
                          "SwitchLeaf");
    } else if (cast<ConstantInt>(Leaf.Low)->isZero()) {
      // Val >= 0 && Val <= Hi --> Val <=u Hi
      Comp = new ICmpInst(*NewLeaf, ICmpInst::ICMP_ULE, Val, Leaf.High,
                          "SwitchLeaf");      
    } else {
      // Emit V-Lo <=u Hi-Lo
      Constant* NegLo = ConstantExpr::getNeg(Leaf.Low);
      Instruction* Add = BinaryOperator::CreateAdd(Val, NegLo,
                                                   Val->getName()+".off",
                                                   NewLeaf);
      Constant *UpperBound = ConstantExpr::getAdd(NegLo, Leaf.High);
      Comp = new ICmpInst(*NewLeaf, ICmpInst::ICMP_ULE, Add, UpperBound,
                          "SwitchLeaf");
    }
  }

  // Make the conditional branch...
  BasicBlock* Succ = Leaf.BB;
  BranchInst::Create(Succ, Default, Comp, NewLeaf);

  // If there were any PHI nodes in this successor, rewrite one entry
  // from OrigBlock to come from NewLeaf.
  for (BasicBlock::iterator I = Succ->begin(); isa<PHINode>(I); ++I) {
    PHINode* PN = cast<PHINode>(I);
    // Remove all but one incoming entries from the cluster
    uint64_t Range = cast<ConstantInt>(Leaf.High)->getSExtValue() -
                     cast<ConstantInt>(Leaf.Low)->getSExtValue();    
    for (uint64_t j = 0; j < Range; ++j) {
      PN->removeIncomingValue(OrigBlock);
    }
    
    int BlockIdx = PN->getBasicBlockIndex(OrigBlock);
    assert(BlockIdx != -1 && "Switch didn't go to this successor??");
    PN->setIncomingBlock((unsigned)BlockIdx, NewLeaf);
  }

  return NewLeaf;
}

// Clusterify - Transform simple list of Cases into list of CaseRange's
unsigned LowerSwitch::Clusterify(CaseVector& Cases, SwitchInst *SI) {

  IntegersSubsetToBB TheClusterifier;

  // Start with "simple" cases
  for (SwitchInst::CaseIt i = SI->case_begin(), e = SI->case_end();
       i != e; ++i) {
    BasicBlock *SuccBB = i.getCaseSuccessor();
    IntegersSubset CaseRanges = i.getCaseValueEx();
    TheClusterifier.add(CaseRanges, SuccBB);
  }
  
  TheClusterifier.optimize();
  
  size_t numCmps = 0;
  for (IntegersSubsetToBB::RangeIterator i = TheClusterifier.begin(),
       e = TheClusterifier.end(); i != e; ++i, ++numCmps) {
    IntegersSubsetToBB::Cluster &C = *i;
    
    // FIXME: Currently work with ConstantInt based numbers.
    // Changing it to APInt based is a pretty heavy for this commit.
    Cases.push_back(CaseRange(C.first.getLow().toConstantInt(),
                              C.first.getHigh().toConstantInt(), C.second));
    if (C.first.isSingleNumber())
      // A range counts double, since it requires two compares.
      ++numCmps;
  }

  return numCmps;  
}

// processSwitchInst - Replace the specified switch instruction with a sequence
// of chained if-then insts in a balanced binary search.
//
void LowerSwitch::processSwitchInst(SwitchInst *SI) {
  BasicBlock *CurBlock = SI->getParent();
  BasicBlock *OrigBlock = CurBlock;
  Function *F = CurBlock->getParent();
  Value *Val = SI->getCondition();  // The value we are switching on...
  BasicBlock* Default = SI->getDefaultDest();

  // If there is only the default destination, don't bother with the code below.
  if (!SI->getNumCases()) {
    BranchInst::Create(SI->getDefaultDest(), CurBlock);
    CurBlock->getInstList().erase(SI);
    return;
  }

  // Create a new, empty default block so that the new hierarchy of
  // if-then statements go to this and the PHI nodes are happy.
  BasicBlock* NewDefault = BasicBlock::Create(SI->getContext(), "NewDefault");
  F->getBasicBlockList().insert(Default, NewDefault);

  BranchInst::Create(Default, NewDefault);

  // If there is an entry in any PHI nodes for the default edge, make sure
  // to update them as well.
  for (BasicBlock::iterator I = Default->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    int BlockIdx = PN->getBasicBlockIndex(OrigBlock);
    assert(BlockIdx != -1 && "Switch didn't go to this successor??");
    PN->setIncomingBlock((unsigned)BlockIdx, NewDefault);
  }

  // Prepare cases vector.
  CaseVector Cases;
  unsigned numCmps = Clusterify(Cases, SI);

  DEBUG(dbgs() << "Clusterify finished. Total clusters: " << Cases.size()
               << ". Total compares: " << numCmps << "\n");
  DEBUG(dbgs() << "Cases: " << Cases << "\n");
  (void)numCmps;
  
  BasicBlock* SwitchBlock = switchConvert(Cases.begin(), Cases.end(), Val,
                                          OrigBlock, NewDefault);

  // Branch to our shiny new if-then stuff...
  BranchInst::Create(SwitchBlock, OrigBlock);

  // We are now done with the switch instruction, delete it.
  CurBlock->getInstList().erase(SI);
}
