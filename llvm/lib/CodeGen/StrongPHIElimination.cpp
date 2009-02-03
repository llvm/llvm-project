//===- StrongPhiElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates machine instruction PHI nodes by inserting copy
// instructions, using an intelligent copy-folding technique based on
// dominator information.  This is technique is derived from:
// 
//    Budimlic, et al. Fast copy coalescing and live-range identification.
//    In Proceedings of the ACM SIGPLAN 2002 Conference on Programming Language
//    Design and Implementation (Berlin, Germany, June 17 - 19, 2002).
//    PLDI '02. ACM, New York, NY, 25-32.
//    DOI= http://doi.acm.org/10.1145/512529.512534
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strongphielim"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN StrongPHIElimination : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid
    StrongPHIElimination() : MachineFunctionPass(&ID) {}

    // Waiting stores, for each MBB, the set of copies that need to
    // be inserted into that MBB
    DenseMap<MachineBasicBlock*,
             std::multimap<unsigned, unsigned> > Waiting;
    
    // Stacks holds the renaming stack for each register
    std::map<unsigned, std::vector<unsigned> > Stacks;
    
    // Registers in UsedByAnother are PHI nodes that are themselves
    // used as operands to another another PHI node
    std::set<unsigned> UsedByAnother;
    
    // RenameSets are the is a map from a PHI-defined register
    // to the input registers to be coalesced along with the 
    // predecessor block for those input registers.
    std::map<unsigned, std::map<unsigned, MachineBasicBlock*> > RenameSets;
    
    // PhiValueNumber holds the ID numbers of the VNs for each phi that we're
    // eliminating, indexed by the register defined by that phi.
    std::map<unsigned, unsigned> PhiValueNumber;

    // Store the DFS-in number of each block
    DenseMap<MachineBasicBlock*, unsigned> preorder;
    
    // Store the DFS-out number of each block
    DenseMap<MachineBasicBlock*, unsigned> maxpreorder;

    bool runOnMachineFunction(MachineFunction &Fn);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<LiveIntervals>();
      
      // TODO: Actually make this true.
      AU.addPreserved<LiveIntervals>();
      AU.addPreserved<RegisterCoalescer>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    
    virtual void releaseMemory() {
      preorder.clear();
      maxpreorder.clear();
      
      Waiting.clear();
      Stacks.clear();
      UsedByAnother.clear();
      RenameSets.clear();
    }

  private:
    
    /// DomForestNode - Represents a node in the "dominator forest".  This is
    /// a forest in which the nodes represent registers and the edges
    /// represent a dominance relation in the block defining those registers.
    struct DomForestNode {
    private:
      // Store references to our children
      std::vector<DomForestNode*> children;
      // The register we represent
      unsigned reg;
      
      // Add another node as our child
      void addChild(DomForestNode* DFN) { children.push_back(DFN); }
      
    public:
      typedef std::vector<DomForestNode*>::iterator iterator;
      
      // Create a DomForestNode by providing the register it represents, and
      // the node to be its parent.  The virtual root node has register 0
      // and a null parent.
      DomForestNode(unsigned r, DomForestNode* parent) : reg(r) {
        if (parent)
          parent->addChild(this);
      }
      
      ~DomForestNode() {
        for (iterator I = begin(), E = end(); I != E; ++I)
          delete *I;
      }
      
      /// getReg - Return the regiser that this node represents
      inline unsigned getReg() { return reg; }
      
      // Provide iterator access to our children
      inline DomForestNode::iterator begin() { return children.begin(); }
      inline DomForestNode::iterator end() { return children.end(); }
    };
    
    void computeDFS(MachineFunction& MF);
    void processBlock(MachineBasicBlock* MBB);
    
    std::vector<DomForestNode*> computeDomForest(
                           std::map<unsigned, MachineBasicBlock*>& instrs,
                                                 MachineRegisterInfo& MRI);
    void processPHIUnion(MachineInstr* Inst,
                         std::map<unsigned, MachineBasicBlock*>& PHIUnion,
                         std::vector<StrongPHIElimination::DomForestNode*>& DF,
                         std::vector<std::pair<unsigned, unsigned> >& locals);
    void ScheduleCopies(MachineBasicBlock* MBB, std::set<unsigned>& pushed);
    void InsertCopies(MachineDomTreeNode* MBB,
                      SmallPtrSet<MachineBasicBlock*, 16>& v);
    bool mergeLiveIntervals(unsigned primary, unsigned secondary);
  };
}

char StrongPHIElimination::ID = 0;
static RegisterPass<StrongPHIElimination>
X("strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently");

const PassInfo *const llvm::StrongPHIEliminationID = &X;

/// computeDFS - Computes the DFS-in and DFS-out numbers of the dominator tree
/// of the given MachineFunction.  These numbers are then used in other parts
/// of the PHI elimination process.
void StrongPHIElimination::computeDFS(MachineFunction& MF) {
  SmallPtrSet<MachineDomTreeNode*, 8> frontier;
  SmallPtrSet<MachineDomTreeNode*, 8> visited;
  
  unsigned time = 0;
  
  MachineDominatorTree& DT = getAnalysis<MachineDominatorTree>();
  
  MachineDomTreeNode* node = DT.getRootNode();
  
  std::vector<MachineDomTreeNode*> worklist;
  worklist.push_back(node);
  
  while (!worklist.empty()) {
    MachineDomTreeNode* currNode = worklist.back();
    
    if (!frontier.count(currNode)) {
      frontier.insert(currNode);
      ++time;
      preorder.insert(std::make_pair(currNode->getBlock(), time));
    }
    
    bool inserted = false;
    for (MachineDomTreeNode::iterator I = currNode->begin(), E = currNode->end();
         I != E; ++I)
      if (!frontier.count(*I) && !visited.count(*I)) {
        worklist.push_back(*I);
        inserted = true;
        break;
      }
    
    if (!inserted) {
      frontier.erase(currNode);
      visited.insert(currNode);
      maxpreorder.insert(std::make_pair(currNode->getBlock(), time));
      
      worklist.pop_back();
    }
  }
}

namespace {

/// PreorderSorter - a helper class that is used to sort registers
/// according to the preorder number of their defining blocks
class PreorderSorter {
private:
  DenseMap<MachineBasicBlock*, unsigned>& preorder;
  MachineRegisterInfo& MRI;
  
public:
  PreorderSorter(DenseMap<MachineBasicBlock*, unsigned>& p,
                MachineRegisterInfo& M) : preorder(p), MRI(M) { }
  
  bool operator()(unsigned A, unsigned B) {
    if (A == B)
      return false;
    
    MachineBasicBlock* ABlock = MRI.getVRegDef(A)->getParent();
    MachineBasicBlock* BBlock = MRI.getVRegDef(B)->getParent();
    
    if (preorder[ABlock] < preorder[BBlock])
      return true;
    else if (preorder[ABlock] > preorder[BBlock])
      return false;
    
    return false;
  }
};

}

/// computeDomForest - compute the subforest of the DomTree corresponding
/// to the defining blocks of the registers in question
std::vector<StrongPHIElimination::DomForestNode*>
StrongPHIElimination::computeDomForest(
                  std::map<unsigned, MachineBasicBlock*>& regs, 
                                       MachineRegisterInfo& MRI) {
  // Begin by creating a virtual root node, since the actual results
  // may well be a forest.  Assume this node has maximum DFS-out number.
  DomForestNode* VirtualRoot = new DomForestNode(0, 0);
  maxpreorder.insert(std::make_pair((MachineBasicBlock*)0, ~0UL));
  
  // Populate a worklist with the registers
  std::vector<unsigned> worklist;
  worklist.reserve(regs.size());
  for (std::map<unsigned, MachineBasicBlock*>::iterator I = regs.begin(),
       E = regs.end(); I != E; ++I)
    worklist.push_back(I->first);
  
  // Sort the registers by the DFS-in number of their defining block
  PreorderSorter PS(preorder, MRI);
  std::sort(worklist.begin(), worklist.end(), PS);
  
  // Create a "current parent" stack, and put the virtual root on top of it
  DomForestNode* CurrentParent = VirtualRoot;
  std::vector<DomForestNode*> stack;
  stack.push_back(VirtualRoot);
  
  // Iterate over all the registers in the previously computed order
  for (std::vector<unsigned>::iterator I = worklist.begin(), E = worklist.end();
       I != E; ++I) {
    unsigned pre = preorder[MRI.getVRegDef(*I)->getParent()];
    MachineBasicBlock* parentBlock = CurrentParent->getReg() ?
                 MRI.getVRegDef(CurrentParent->getReg())->getParent() :
                 0;
    
    // If the DFS-in number of the register is greater than the DFS-out number
    // of the current parent, repeatedly pop the parent stack until it isn't.
    while (pre > maxpreorder[parentBlock]) {
      stack.pop_back();
      CurrentParent = stack.back();
      
      parentBlock = CurrentParent->getReg() ?
                   MRI.getVRegDef(CurrentParent->getReg())->getParent() :
                   0;
    }
    
    // Now that we've found the appropriate parent, create a DomForestNode for
    // this register and attach it to the forest
    DomForestNode* child = new DomForestNode(*I, CurrentParent);
    
    // Push this new node on the "current parent" stack
    stack.push_back(child);
    CurrentParent = child;
  }
  
  // Return a vector containing the children of the virtual root node
  std::vector<DomForestNode*> ret;
  ret.insert(ret.end(), VirtualRoot->begin(), VirtualRoot->end());
  return ret;
}

/// isLiveIn - helper method that determines, from a regno, if a register
/// is live into a block
static bool isLiveIn(unsigned r, MachineBasicBlock* MBB,
                     LiveIntervals& LI) {
  LiveInterval& I = LI.getOrCreateInterval(r);
  unsigned idx = LI.getMBBStartIdx(MBB);
  return I.liveAt(idx);
}

/// isLiveOut - help method that determines, from a regno, if a register is
/// live out of a block.
static bool isLiveOut(unsigned r, MachineBasicBlock* MBB,
                      LiveIntervals& LI) {
  for (MachineBasicBlock::succ_iterator PI = MBB->succ_begin(),
       E = MBB->succ_end(); PI != E; ++PI)
    if (isLiveIn(r, *PI, LI))
      return true;
  
  return false;
}

/// interferes - checks for local interferences by scanning a block.  The only
/// trick parameter is 'mode' which tells it the relationship of the two
/// registers. 0 - defined in the same block, 1 - first properly dominates
/// second, 2 - second properly dominates first 
static bool interferes(unsigned a, unsigned b, MachineBasicBlock* scan,
                       LiveIntervals& LV, unsigned mode) {
  MachineInstr* def = 0;
  MachineInstr* kill = 0;
  
  // The code is still in SSA form at this point, so there is only one
  // definition per VReg.  Thus we can safely use MRI->getVRegDef().
  const MachineRegisterInfo* MRI = &scan->getParent()->getRegInfo();
  
  bool interference = false;
  
  // Wallk the block, checking for interferences
  for (MachineBasicBlock::iterator MBI = scan->begin(), MBE = scan->end();
       MBI != MBE; ++MBI) {
    MachineInstr* curr = MBI;
    
    // Same defining block...
    if (mode == 0) {
      if (curr == MRI->getVRegDef(a)) {
        // If we find our first definition, save it
        if (!def) {
          def = curr;
        // If there's already an unkilled definition, then 
        // this is an interference
        } else if (!kill) {
          interference = true;
          break;
        // If there's a definition followed by a KillInst, then
        // they can't interfere
        } else {
          interference = false;
          break;
        }
      // Symmetric with the above
      } else if (curr == MRI->getVRegDef(b)) {
        if (!def) {
          def = curr;
        } else if (!kill) {
          interference = true;
          break;
        } else {
          interference = false;
          break;
        }
      // Store KillInsts if they match up with the definition
      } else if (curr->killsRegister(a)) {
        if (def == MRI->getVRegDef(a)) {
          kill = curr;
        } else if (curr->killsRegister(b)) {
          if (def == MRI->getVRegDef(b)) {
            kill = curr;
          }
        }
      }
    // First properly dominates second...
    } else if (mode == 1) {
      if (curr == MRI->getVRegDef(b)) {
        // Definition of second without kill of first is an interference
        if (!kill) {
          interference = true;
          break;
        // Definition after a kill is a non-interference
        } else {
          interference = false;
          break;
        }
      // Save KillInsts of First
      } else if (curr->killsRegister(a)) {
        kill = curr;
      }
    // Symmetric with the above
    } else if (mode == 2) {
      if (curr == MRI->getVRegDef(a)) {
        if (!kill) {
          interference = true;
          break;
        } else {
          interference = false;
          break;
        }
      } else if (curr->killsRegister(b)) {
        kill = curr;
      }
    }
  }
  
  return interference;
}

/// processBlock - Determine how to break up PHIs in the current block.  Each
/// PHI is broken up by some combination of renaming its operands and inserting
/// copies.  This method is responsible for determining which operands receive
/// which treatment.
void StrongPHIElimination::processBlock(MachineBasicBlock* MBB) {
  LiveIntervals& LI = getAnalysis<LiveIntervals>();
  MachineRegisterInfo& MRI = MBB->getParent()->getRegInfo();
  
  // Holds names that have been added to a set in any PHI within this block
  // before the current one.
  std::set<unsigned> ProcessedNames;
  
  // Iterate over all the PHI nodes in this block
  MachineBasicBlock::iterator P = MBB->begin();
  while (P != MBB->end() && P->getOpcode() == TargetInstrInfo::PHI) {
    unsigned DestReg = P->getOperand(0).getReg();
    
    // Don't both doing PHI elimination for dead PHI's.
    if (P->registerDefIsDead(DestReg)) {
      ++P;
      continue;
    }

    LiveInterval& PI = LI.getOrCreateInterval(DestReg);
    unsigned pIdx = LI.getDefIndex(LI.getInstructionIndex(P));
    VNInfo* PVN = PI.getLiveRangeContaining(pIdx)->valno;
    PhiValueNumber.insert(std::make_pair(DestReg, PVN->id));

    // PHIUnion is the set of incoming registers to the PHI node that
    // are going to be renames rather than having copies inserted.  This set
    // is refinded over the course of this function.  UnionedBlocks is the set
    // of corresponding MBBs.
    std::map<unsigned, MachineBasicBlock*> PHIUnion;
    SmallPtrSet<MachineBasicBlock*, 8> UnionedBlocks;
  
    // Iterate over the operands of the PHI node
    for (int i = P->getNumOperands() - 1; i >= 2; i-=2) {
      unsigned SrcReg = P->getOperand(i-1).getReg();
      
      // Don't need to try to coalesce a register with itself.
      if (SrcReg == DestReg) {
        ProcessedNames.insert(SrcReg);
        continue;
      }
      
      // We don't need to insert copies for implicit_defs.
      MachineInstr* DefMI = MRI.getVRegDef(SrcReg);
      if (DefMI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF)
        ProcessedNames.insert(SrcReg);
    
      // Check for trivial interferences via liveness information, allowing us
      // to avoid extra work later.  Any registers that interfere cannot both
      // be in the renaming set, so choose one and add copies for it instead.
      // The conditions are:
      //   1) if the operand is live into the PHI node's block OR
      //   2) if the PHI node is live out of the operand's defining block OR
      //   3) if the operand is itself a PHI node and the original PHI is
      //      live into the operand's defining block OR
      //   4) if the operand is already being renamed for another PHI node
      //      in this block OR
      //   5) if any two operands are defined in the same block, insert copies
      //      for one of them
      if (isLiveIn(SrcReg, P->getParent(), LI) ||
          isLiveOut(P->getOperand(0).getReg(),
                    MRI.getVRegDef(SrcReg)->getParent(), LI) ||
          ( MRI.getVRegDef(SrcReg)->getOpcode() == TargetInstrInfo::PHI &&
            isLiveIn(P->getOperand(0).getReg(),
                     MRI.getVRegDef(SrcReg)->getParent(), LI) ) ||
          ProcessedNames.count(SrcReg) ||
          UnionedBlocks.count(MRI.getVRegDef(SrcReg)->getParent())) {
        
        // Add a copy for the selected register
        MachineBasicBlock* From = P->getOperand(i).getMBB();
        Waiting[From].insert(std::make_pair(SrcReg, DestReg));
        UsedByAnother.insert(SrcReg);
      } else {
        // Otherwise, add it to the renaming set
        PHIUnion.insert(std::make_pair(SrcReg,P->getOperand(i).getMBB()));
        UnionedBlocks.insert(MRI.getVRegDef(SrcReg)->getParent());
      }
    }
    
    // Compute the dominator forest for the renaming set.  This is a forest
    // where the nodes are the registers and the edges represent dominance 
    // relations between the defining blocks of the registers
    std::vector<StrongPHIElimination::DomForestNode*> DF = 
                                                computeDomForest(PHIUnion, MRI);
    
    // Walk DomForest to resolve interferences at an inter-block level.  This
    // will remove registers from the renaming set (and insert copies for them)
    // if interferences are found.
    std::vector<std::pair<unsigned, unsigned> > localInterferences;
    processPHIUnion(P, PHIUnion, DF, localInterferences);
    
    // If one of the inputs is defined in the same block as the current PHI
    // then we need to check for a local interference between that input and
    // the PHI.
    for (std::map<unsigned, MachineBasicBlock*>::iterator I = PHIUnion.begin(),
         E = PHIUnion.end(); I != E; ++I)
      if (MRI.getVRegDef(I->first)->getParent() == P->getParent())
        localInterferences.push_back(std::make_pair(I->first,
                                                    P->getOperand(0).getReg()));
    
    // The dominator forest walk may have returned some register pairs whose
    // interference cannot be determined from dominator analysis.  We now 
    // examine these pairs for local interferences.
    for (std::vector<std::pair<unsigned, unsigned> >::iterator I =
        localInterferences.begin(), E = localInterferences.end(); I != E; ++I) {
      std::pair<unsigned, unsigned> p = *I;
      
      MachineDominatorTree& MDT = getAnalysis<MachineDominatorTree>();
      
      // Determine the block we need to scan and the relationship between
      // the two registers
      MachineBasicBlock* scan = 0;
      unsigned mode = 0;
      if (MRI.getVRegDef(p.first)->getParent() ==
          MRI.getVRegDef(p.second)->getParent()) {
        scan = MRI.getVRegDef(p.first)->getParent();
        mode = 0; // Same block
      } else if (MDT.dominates(MRI.getVRegDef(p.first)->getParent(),
                               MRI.getVRegDef(p.second)->getParent())) {
        scan = MRI.getVRegDef(p.second)->getParent();
        mode = 1; // First dominates second
      } else {
        scan = MRI.getVRegDef(p.first)->getParent();
        mode = 2; // Second dominates first
      }
      
      // If there's an interference, we need to insert  copies
      if (interferes(p.first, p.second, scan, LI, mode)) {
        // Insert copies for First
        for (int i = P->getNumOperands() - 1; i >= 2; i-=2) {
          if (P->getOperand(i-1).getReg() == p.first) {
            unsigned SrcReg = p.first;
            MachineBasicBlock* From = P->getOperand(i).getMBB();
            
            Waiting[From].insert(std::make_pair(SrcReg,
                                                P->getOperand(0).getReg()));
            UsedByAnother.insert(SrcReg);
            
            PHIUnion.erase(SrcReg);
          }
        }
      }
    }
    
    // Add the renaming set for this PHI node to our overall renaming information
    for (std::map<unsigned, MachineBasicBlock*>::iterator QI = PHIUnion.begin(),
         QE = PHIUnion.end(); QI != QE; ++QI) {
      DOUT << "Adding Renaming: " << QI->first << " -> "
           << P->getOperand(0).getReg() << "\n";
    }
    
    RenameSets.insert(std::make_pair(P->getOperand(0).getReg(), PHIUnion));
    
    // Remember which registers are already renamed, so that we don't try to 
    // rename them for another PHI node in this block
    for (std::map<unsigned, MachineBasicBlock*>::iterator I = PHIUnion.begin(),
         E = PHIUnion.end(); I != E; ++I)
      ProcessedNames.insert(I->first);
    
    ++P;
  }
}

/// processPHIUnion - Take a set of candidate registers to be coalesced when
/// decomposing the PHI instruction.  Use the DominanceForest to remove the ones
/// that are known to interfere, and flag others that need to be checked for
/// local interferences.
void StrongPHIElimination::processPHIUnion(MachineInstr* Inst,
                        std::map<unsigned, MachineBasicBlock*>& PHIUnion,
                        std::vector<StrongPHIElimination::DomForestNode*>& DF,
                        std::vector<std::pair<unsigned, unsigned> >& locals) {
  
  std::vector<DomForestNode*> worklist(DF.begin(), DF.end());
  SmallPtrSet<DomForestNode*, 4> visited;
  
  // Code is still in SSA form, so we can use MRI::getVRegDef()
  MachineRegisterInfo& MRI = Inst->getParent()->getParent()->getRegInfo();
  
  LiveIntervals& LI = getAnalysis<LiveIntervals>();
  unsigned DestReg = Inst->getOperand(0).getReg();
  
  // DF walk on the DomForest
  while (!worklist.empty()) {
    DomForestNode* DFNode = worklist.back();
    
    visited.insert(DFNode);
    
    bool inserted = false;
    for (DomForestNode::iterator CI = DFNode->begin(), CE = DFNode->end();
         CI != CE; ++CI) {
      DomForestNode* child = *CI;   
      
      // If the current node is live-out of the defining block of one of its
      // children, insert a copy for it.  NOTE: The paper actually calls for
      // a more elaborate heuristic for determining whether to insert copies
      // for the child or the parent.  In the interest of simplicity, we're
      // just always choosing the parent.
      if (isLiveOut(DFNode->getReg(),
          MRI.getVRegDef(child->getReg())->getParent(), LI)) {
        // Insert copies for parent
        for (int i = Inst->getNumOperands() - 1; i >= 2; i-=2) {
          if (Inst->getOperand(i-1).getReg() == DFNode->getReg()) {
            unsigned SrcReg = DFNode->getReg();
            MachineBasicBlock* From = Inst->getOperand(i).getMBB();
            
            Waiting[From].insert(std::make_pair(SrcReg, DestReg));
            UsedByAnother.insert(SrcReg);
            
            PHIUnion.erase(SrcReg);
          }
        }
      
      // If a node is live-in to the defining block of one of its children, but
      // not live-out, then we need to scan that block for local interferences.
      } else if (isLiveIn(DFNode->getReg(),
                          MRI.getVRegDef(child->getReg())->getParent(), LI) ||
                 MRI.getVRegDef(DFNode->getReg())->getParent() ==
                                 MRI.getVRegDef(child->getReg())->getParent()) {
        // Add (p, c) to possible local interferences
        locals.push_back(std::make_pair(DFNode->getReg(), child->getReg()));
      }
      
      if (!visited.count(child)) {
        worklist.push_back(child);
        inserted = true;
      }
    }
    
    if (!inserted) worklist.pop_back();
  }
}

/// ScheduleCopies - Insert copies into predecessor blocks, scheduling
/// them properly so as to avoid the 'lost copy' and the 'virtual swap'
/// problems.
///
/// Based on "Practical Improvements to the Construction and Destruction
/// of Static Single Assignment Form" by Briggs, et al.
void StrongPHIElimination::ScheduleCopies(MachineBasicBlock* MBB,
                                          std::set<unsigned>& pushed) {
  // FIXME: This function needs to update LiveIntervals
  std::multimap<unsigned, unsigned>& copy_set= Waiting[MBB];
  
  std::multimap<unsigned, unsigned> worklist;
  std::map<unsigned, unsigned> map;
  
  // Setup worklist of initial copies
  for (std::multimap<unsigned, unsigned>::iterator I = copy_set.begin(),
       E = copy_set.end(); I != E; ) {
    map.insert(std::make_pair(I->first, I->first));
    map.insert(std::make_pair(I->second, I->second));
         
    if (!UsedByAnother.count(I->second)) {
      worklist.insert(*I);
      
      // Avoid iterator invalidation
      std::multimap<unsigned, unsigned>::iterator OI = I;
      ++I;
      copy_set.erase(OI);
    } else {
      ++I;
    }
  }
  
  LiveIntervals& LI = getAnalysis<LiveIntervals>();
  MachineFunction* MF = MBB->getParent();
  MachineRegisterInfo& MRI = MF->getRegInfo();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();
  
  SmallVector<std::pair<unsigned, MachineInstr*>, 4> InsertedPHIDests;
  
  // Iterate over the worklist, inserting copies
  while (!worklist.empty() || !copy_set.empty()) {
    while (!worklist.empty()) {
      std::multimap<unsigned, unsigned>::iterator WI = worklist.begin();
      std::pair<unsigned, unsigned> curr = *WI;
      worklist.erase(WI);
      
      const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(curr.first);
      
      if (isLiveOut(curr.second, MBB, LI)) {
        // Create a temporary
        unsigned t = MF->getRegInfo().createVirtualRegister(RC);
        
        // Insert copy from curr.second to a temporary at
        // the Phi defining curr.second
        MachineBasicBlock::iterator PI = MRI.getVRegDef(curr.second);
        TII->copyRegToReg(*PI->getParent(), PI, t,
                          curr.second, RC, RC);
        
        DOUT << "Inserted copy from " << curr.second << " to " << t << "\n";
        
        // Push temporary on Stacks
        Stacks[curr.second].push_back(t);
        
        // Insert curr.second in pushed
        pushed.insert(curr.second);
        
        // Create a live interval for this temporary
        InsertedPHIDests.push_back(std::make_pair(t, --PI));
      }
      
      // Insert copy from map[curr.first] to curr.second
      TII->copyRegToReg(*MBB, MBB->getFirstTerminator(), curr.second,
                        map[curr.first], RC, RC);
      map[curr.first] = curr.second;
      DOUT << "Inserted copy from " << curr.first << " to "
           << curr.second << "\n";
      
      // Push this copy onto InsertedPHICopies so we can
      // update LiveIntervals with it.
      MachineBasicBlock::iterator MI = MBB->getFirstTerminator();
      InsertedPHIDests.push_back(std::make_pair(curr.second, --MI));
      
      // If curr.first is a destination in copy_set...
      for (std::multimap<unsigned, unsigned>::iterator I = copy_set.begin(),
           E = copy_set.end(); I != E; )
        if (curr.first == I->second) {
          std::pair<unsigned, unsigned> temp = *I;
          worklist.insert(temp);
          
          // Avoid iterator invalidation
          std::multimap<unsigned, unsigned>::iterator OI = I;
          ++I;
          copy_set.erase(OI);
          
          break;
        } else {
          ++I;
        }
    }
    
    if (!copy_set.empty()) {
      std::multimap<unsigned, unsigned>::iterator CI = copy_set.begin();
      std::pair<unsigned, unsigned> curr = *CI;
      worklist.insert(curr);
      copy_set.erase(CI);
      
      LiveInterval& I = LI.getInterval(curr.second);
      MachineBasicBlock::iterator term = MBB->getFirstTerminator();
      unsigned endIdx = 0;
      if (term != MBB->end())
        endIdx = LI.getInstructionIndex(term);
      else
        endIdx = LI.getMBBEndIdx(MBB);
      
      if (I.liveAt(endIdx)) {
        const TargetRegisterClass *RC =
                                       MF->getRegInfo().getRegClass(curr.first);
        
        // Insert a copy from dest to a new temporary t at the end of b
        unsigned t = MF->getRegInfo().createVirtualRegister(RC);
        TII->copyRegToReg(*MBB, MBB->getFirstTerminator(), t,
                          curr.second, RC, RC);
        map[curr.second] = t;
        
        MachineBasicBlock::iterator TI = MBB->getFirstTerminator();
        InsertedPHIDests.push_back(std::make_pair(t, --TI));
      }
    }
  }
  
  // Renumber the instructions so that we can perform the index computations
  // needed to create new live intervals.
  LI.computeNumbering();
  
  // For copies that we inserted at the ends of predecessors, we construct
  // live intervals.  This is pretty easy, since we know that the destination
  // register cannot have be in live at that point previously.  We just have
  // to make sure that, for registers that serve as inputs to more than one
  // PHI, we don't create multiple overlapping live intervals.
  std::set<unsigned> RegHandled;
  for (SmallVector<std::pair<unsigned, MachineInstr*>, 4>::iterator I =
       InsertedPHIDests.begin(), E = InsertedPHIDests.end(); I != E; ++I) {
    if (RegHandled.insert(I->first).second) {
      LiveInterval& Int = LI.getOrCreateInterval(I->first);
      unsigned instrIdx = LI.getInstructionIndex(I->second);
      if (Int.liveAt(LiveIntervals::getDefIndex(instrIdx)))
        Int.removeRange(LiveIntervals::getDefIndex(instrIdx),
                        LI.getMBBEndIdx(I->second->getParent())+1,
                        true);
      
      LiveRange R = LI.addLiveRangeToEndOfBlock(I->first, I->second);
      R.valno->copy = I->second;
      R.valno->def =
                  LiveIntervals::getDefIndex(LI.getInstructionIndex(I->second));
    }
  }
}

/// InsertCopies - insert copies into MBB and all of its successors
void StrongPHIElimination::InsertCopies(MachineDomTreeNode* MDTN,
                                 SmallPtrSet<MachineBasicBlock*, 16>& visited) {
  MachineBasicBlock* MBB = MDTN->getBlock();
  visited.insert(MBB);
  
  std::set<unsigned> pushed;
  
  LiveIntervals& LI = getAnalysis<LiveIntervals>();
  // Rewrite register uses from Stacks
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
      I != E; ++I) {
    if (I->getOpcode() == TargetInstrInfo::PHI)
      continue;
    
    for (unsigned i = 0; i < I->getNumOperands(); ++i)
      if (I->getOperand(i).isReg() &&
          Stacks[I->getOperand(i).getReg()].size()) {
        // Remove the live range for the old vreg.
        LiveInterval& OldInt = LI.getInterval(I->getOperand(i).getReg());
        LiveInterval::iterator OldLR = OldInt.FindLiveRangeContaining(
                  LiveIntervals::getUseIndex(LI.getInstructionIndex(I)));
        if (OldLR != OldInt.end())
          OldInt.removeRange(*OldLR, true);
        
        // Change the register
        I->getOperand(i).setReg(Stacks[I->getOperand(i).getReg()].back());
        
        // Add a live range for the new vreg
        LiveInterval& Int = LI.getInterval(I->getOperand(i).getReg());
        VNInfo* FirstVN = *Int.vni_begin();
        FirstVN->hasPHIKill = false;
        if (I->getOperand(i).isKill())
          FirstVN->kills.push_back(
                         LiveIntervals::getUseIndex(LI.getInstructionIndex(I)));
        
        LiveRange LR (LI.getMBBStartIdx(I->getParent()),
                      LiveIntervals::getUseIndex(LI.getInstructionIndex(I))+1,
                      FirstVN);
        
        Int.addRange(LR);
      }
  }    
  
  // Schedule the copies for this block
  ScheduleCopies(MBB, pushed);
  
  // Recur down the dominator tree.
  for (MachineDomTreeNode::iterator I = MDTN->begin(),
       E = MDTN->end(); I != E; ++I)
    if (!visited.count((*I)->getBlock()))
      InsertCopies(*I, visited);
  
  // As we exit this block, pop the names we pushed while processing it
  for (std::set<unsigned>::iterator I = pushed.begin(), 
       E = pushed.end(); I != E; ++I)
    Stacks[*I].pop_back();
}

bool StrongPHIElimination::mergeLiveIntervals(unsigned primary,
                                              unsigned secondary) {
  
  LiveIntervals& LI = getAnalysis<LiveIntervals>();
  LiveInterval& LHS = LI.getOrCreateInterval(primary);
  LiveInterval& RHS = LI.getOrCreateInterval(secondary);
  
  LI.computeNumbering();
  
  DenseMap<VNInfo*, VNInfo*> VNMap;
  for (LiveInterval::iterator I = RHS.begin(), E = RHS.end(); I != E; ++I) {
    LiveRange R = *I;
 
    unsigned Start = R.start;
    unsigned End = R.end;
    if (LHS.getLiveRangeContaining(Start))
      return false;
    
    if (LHS.getLiveRangeContaining(End))
      return false;
    
    LiveInterval::iterator RI = std::upper_bound(LHS.begin(), LHS.end(), R);
    if (RI != LHS.end() && RI->start < End)
      return false;
  }
  
  for (LiveInterval::iterator I = RHS.begin(), E = RHS.end(); I != E; ++I) {
    LiveRange R = *I;
    VNInfo* OldVN = R.valno;
    VNInfo*& NewVN = VNMap[OldVN];
    if (!NewVN) {
      NewVN = LHS.getNextValue(OldVN->def,
                               OldVN->copy,
                               LI.getVNInfoAllocator());
      NewVN->kills = OldVN->kills;
    }
    
    LiveRange LR (R.start, R.end, NewVN);
    LHS.addRange(LR);
  }
  
  LI.removeInterval(RHS.reg);
  
  return true;
}

bool StrongPHIElimination::runOnMachineFunction(MachineFunction &Fn) {
  LiveIntervals& LI = getAnalysis<LiveIntervals>();
  
  // Compute DFS numbers of each block
  computeDFS(Fn);
  
  // Determine which phi node operands need copies
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    if (!I->empty() &&
        I->begin()->getOpcode() == TargetInstrInfo::PHI)
      processBlock(I);
  
  // Break interferences where two different phis want to coalesce
  // in the same register.
  std::set<unsigned> seen;
  typedef std::map<unsigned, std::map<unsigned, MachineBasicBlock*> >
          RenameSetType;
  for (RenameSetType::iterator I = RenameSets.begin(), E = RenameSets.end();
       I != E; ++I) {
    for (std::map<unsigned, MachineBasicBlock*>::iterator
         OI = I->second.begin(), OE = I->second.end(); OI != OE; ) {
      if (!seen.count(OI->first)) {
        seen.insert(OI->first);
        ++OI;
      } else {
        Waiting[OI->second].insert(std::make_pair(OI->first, I->first));
        unsigned reg = OI->first;
        ++OI;
        I->second.erase(reg);
        DOUT << "Removing Renaming: " << reg << " -> " << I->first << "\n";
      }
    }
  }
  
  // Insert copies
  // FIXME: This process should probably preserve LiveIntervals
  SmallPtrSet<MachineBasicBlock*, 16> visited;
  MachineDominatorTree& MDT = getAnalysis<MachineDominatorTree>();
  InsertCopies(MDT.getRootNode(), visited);
  
  // Perform renaming
  for (RenameSetType::iterator I = RenameSets.begin(), E = RenameSets.end();
       I != E; ++I)
    while (I->second.size()) {
      std::map<unsigned, MachineBasicBlock*>::iterator SI = I->second.begin();
      
      DOUT << "Renaming: " << SI->first << " -> " << I->first << "\n";
      
      if (SI->first != I->first) {
        if (mergeLiveIntervals(I->first, SI->first)) {
          Fn.getRegInfo().replaceRegWith(SI->first, I->first);
      
          if (RenameSets.count(SI->first)) {
            I->second.insert(RenameSets[SI->first].begin(),
                             RenameSets[SI->first].end());
            RenameSets.erase(SI->first);
          }
        } else {
          // Insert a last-minute copy if a conflict was detected.
          const TargetInstrInfo *TII = Fn.getTarget().getInstrInfo();
          const TargetRegisterClass *RC = Fn.getRegInfo().getRegClass(I->first);
          TII->copyRegToReg(*SI->second, SI->second->getFirstTerminator(),
                            I->first, SI->first, RC, RC);
          
          LI.computeNumbering();
          
          LiveInterval& Int = LI.getOrCreateInterval(I->first);
          unsigned instrIdx =
                     LI.getInstructionIndex(--SI->second->getFirstTerminator());
          if (Int.liveAt(LiveIntervals::getDefIndex(instrIdx)))
            Int.removeRange(LiveIntervals::getDefIndex(instrIdx),
                            LI.getMBBEndIdx(SI->second)+1, true);

          LiveRange R = LI.addLiveRangeToEndOfBlock(I->first,
                                            --SI->second->getFirstTerminator());
          R.valno->copy = --SI->second->getFirstTerminator();
          R.valno->def = LiveIntervals::getDefIndex(instrIdx);
          
          DOUT << "Renaming failed: " << SI->first << " -> "
               << I->first << "\n";
        }
      }
      
      LiveInterval& Int = LI.getOrCreateInterval(I->first);
      const LiveRange* LR =
                       Int.getLiveRangeContaining(LI.getMBBEndIdx(SI->second));
      LR->valno->hasPHIKill = true;
      
      I->second.erase(SI->first);
    }
  
  // Remove PHIs
  std::vector<MachineInstr*> phis;
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    for (MachineBasicBlock::iterator BI = I->begin(), BE = I->end();
         BI != BE; ++BI)
      if (BI->getOpcode() == TargetInstrInfo::PHI)
        phis.push_back(BI);
  }
  
  for (std::vector<MachineInstr*>::iterator I = phis.begin(), E = phis.end();
       I != E; ) {
    MachineInstr* PInstr = *(I++);
    
    // If this is a dead PHI node, then remove it from LiveIntervals.
    unsigned DestReg = PInstr->getOperand(0).getReg();
    LiveInterval& PI = LI.getInterval(DestReg);
    if (PInstr->registerDefIsDead(DestReg)) {
      if (PI.containsOneValue()) {
        LI.removeInterval(DestReg);
      } else {
        unsigned idx = LI.getDefIndex(LI.getInstructionIndex(PInstr));
        PI.removeRange(*PI.getLiveRangeContaining(idx), true);
      }
    } else {
      // Trim live intervals of input registers.  They are no longer live into
      // this block if they died after the PHI.  If they lived after it, don't
      // trim them because they might have other legitimate uses.
      for (unsigned i = 1; i < PInstr->getNumOperands(); i += 2) {
        unsigned reg = PInstr->getOperand(i).getReg();
        
        MachineBasicBlock* MBB = PInstr->getOperand(i+1).getMBB();
        LiveInterval& InputI = LI.getInterval(reg);
        if (MBB != PInstr->getParent() &&
            InputI.liveAt(LI.getMBBStartIdx(PInstr->getParent())) &&
            InputI.expiredAt(LI.getInstructionIndex(PInstr) + 
                             LiveIntervals::InstrSlots::NUM))
          InputI.removeRange(LI.getMBBStartIdx(PInstr->getParent()),
                             LI.getInstructionIndex(PInstr),
                             true);
      }
      
      // If the PHI is not dead, then the valno defined by the PHI
      // now has an unknown def.
      unsigned idx = LI.getDefIndex(LI.getInstructionIndex(PInstr));
      const LiveRange* PLR = PI.getLiveRangeContaining(idx);
      PLR->valno->def = ~0U;
      LiveRange R (LI.getMBBStartIdx(PInstr->getParent()),
                   PLR->start, PLR->valno);
      PI.addRange(R);
    }
    
    LI.RemoveMachineInstrFromMaps(PInstr);
    PInstr->eraseFromParent();
  }
  
  LI.computeNumbering();
  
  return true;
}
