//===-- MachineCSE.cpp - Machine Common Subexpression Elimination Pass ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global common subexpression elimination on machine
// instructions using a scoped hash table based value numbering scheme. It
// must be run while the machine function is still in SSA form.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-cse"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

STATISTIC(NumCoalesces, "Number of copies coalesced");
STATISTIC(NumCSEs,      "Number of common subexpression eliminated");
STATISTIC(NumPhysCSEs,  "Number of phyreg defining common subexpr eliminated");

namespace {
  class MachineCSE : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    AliasAnalysis *AA;
    MachineDominatorTree *DT;
    MachineRegisterInfo *MRI;
  public:
    static char ID; // Pass identification
    MachineCSE() : MachineFunctionPass(ID), LookAheadLimit(5), CurrVN(0) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addPreservedID(MachineLoopInfoID);
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
    }

    virtual void releaseMemory() {
      ScopeMap.clear();
      Exps.clear();
    }

  private:
    const unsigned LookAheadLimit;
    typedef ScopedHashTableScope<MachineInstr*, unsigned,
                                 MachineInstrExpressionTrait> ScopeType;
    DenseMap<MachineBasicBlock*, ScopeType*> ScopeMap;
    ScopedHashTable<MachineInstr*, unsigned, MachineInstrExpressionTrait> VNT;
    SmallVector<MachineInstr*, 64> Exps;
    unsigned CurrVN;

    bool PerformTrivialCoalescing(MachineInstr *MI, MachineBasicBlock *MBB);
    bool isPhysDefTriviallyDead(unsigned Reg,
                                MachineBasicBlock::const_iterator I,
                                MachineBasicBlock::const_iterator E) const ;
    bool hasLivePhysRegDefUse(const MachineInstr *MI,
                              const MachineBasicBlock *MBB,
                              unsigned &PhysDef) const;
    bool PhysRegDefReaches(MachineInstr *CSMI, MachineInstr *MI,
                           unsigned PhysDef) const;
    bool isCSECandidate(MachineInstr *MI);
    bool isProfitableToCSE(unsigned CSReg, unsigned Reg,
                           MachineInstr *CSMI, MachineInstr *MI);
    void EnterScope(MachineBasicBlock *MBB);
    void ExitScope(MachineBasicBlock *MBB);
    bool ProcessBlock(MachineBasicBlock *MBB);
    void ExitScopeIfDone(MachineDomTreeNode *Node,
                 DenseMap<MachineDomTreeNode*, unsigned> &OpenChildren,
                 DenseMap<MachineDomTreeNode*, MachineDomTreeNode*> &ParentMap);
    bool PerformCSE(MachineDomTreeNode *Node);
  };
} // end anonymous namespace

char MachineCSE::ID = 0;
INITIALIZE_PASS(MachineCSE, "machine-cse",
                "Machine Common Subexpression Elimination", false, false);

FunctionPass *llvm::createMachineCSEPass() { return new MachineCSE(); }

bool MachineCSE::PerformTrivialCoalescing(MachineInstr *MI,
                                          MachineBasicBlock *MBB) {
  bool Changed = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (!MRI->hasOneNonDBGUse(Reg))
      // Only coalesce single use copies. This ensure the copy will be
      // deleted.
      continue;
    MachineInstr *DefMI = MRI->getVRegDef(Reg);
    if (DefMI->getParent() != MBB)
      continue;
    if (!DefMI->isCopy())
      continue;
    unsigned SrcReg = DefMI->getOperand(1).getReg();
    if (!TargetRegisterInfo::isVirtualRegister(SrcReg))
      continue;
    if (DefMI->getOperand(0).getSubReg() || DefMI->getOperand(1).getSubReg())
      continue;
    const TargetRegisterClass *SRC   = MRI->getRegClass(SrcReg);
    const TargetRegisterClass *RC    = MRI->getRegClass(Reg);
    const TargetRegisterClass *NewRC = getCommonSubClass(RC, SRC);
    if (!NewRC)
      continue;
    DEBUG(dbgs() << "Coalescing: " << *DefMI);
    DEBUG(dbgs() << "*** to: " << *MI);
    MO.setReg(SrcReg);
    MRI->clearKillFlags(SrcReg);
    if (NewRC != SRC)
      MRI->setRegClass(SrcReg, NewRC);
    DefMI->eraseFromParent();
    ++NumCoalesces;
    Changed = true;
  }

  return Changed;
}

bool
MachineCSE::isPhysDefTriviallyDead(unsigned Reg,
                                   MachineBasicBlock::const_iterator I,
                                   MachineBasicBlock::const_iterator E) const {
  unsigned LookAheadLeft = LookAheadLimit;
  while (LookAheadLeft) {
    // Skip over dbg_value's.
    while (I != E && I->isDebugValue())
      ++I;

    if (I == E)
      // Reached end of block, register is obviously dead.
      return true;

    bool SeenDef = false;
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = I->getOperand(i);
      if (!MO.isReg() || !MO.getReg())
        continue;
      if (!TRI->regsOverlap(MO.getReg(), Reg))
        continue;
      if (MO.isUse())
        // Found a use!
        return false;
      SeenDef = true;
    }
    if (SeenDef)
      // See a def of Reg (or an alias) before encountering any use, it's 
      // trivially dead.
      return true;

    --LookAheadLeft;
    ++I;
  }
  return false;
}

/// hasLivePhysRegDefUse - Return true if the specified instruction read / write
/// physical registers (except for dead defs of physical registers). It also
/// returns the physical register def by reference if it's the only one and the
/// instruction does not uses a physical register.
bool MachineCSE::hasLivePhysRegDefUse(const MachineInstr *MI,
                                      const MachineBasicBlock *MBB,
                                      unsigned &PhysDef) const {
  PhysDef = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (MO.isUse()) {
      // Can't touch anything to read a physical register.
      PhysDef = 0;
      return true;
    }
    if (MO.isDead())
      // If the def is dead, it's ok.
      continue;
    // Ok, this is a physical register def that's not marked "dead". That's
    // common since this pass is run before livevariables. We can scan
    // forward a few instructions and check if it is obviously dead.
    if (PhysDef) {
      // Multiple physical register defs. These are rare, forget about it.
      PhysDef = 0;
      return true;
    }
    PhysDef = Reg;
  }

  if (PhysDef) {
    MachineBasicBlock::const_iterator I = MI; I = llvm::next(I);
    if (!isPhysDefTriviallyDead(PhysDef, I, MBB->end()))
      return true;
  }
  return false;
}

bool MachineCSE::PhysRegDefReaches(MachineInstr *CSMI, MachineInstr *MI,
                                  unsigned PhysDef) const {
  // For now conservatively returns false if the common subexpression is
  // not in the same basic block as the given instruction.
  MachineBasicBlock *MBB = MI->getParent();
  if (CSMI->getParent() != MBB)
    return false;
  MachineBasicBlock::const_iterator I = CSMI; I = llvm::next(I);
  MachineBasicBlock::const_iterator E = MI;
  unsigned LookAheadLeft = LookAheadLimit;
  while (LookAheadLeft) {
    // Skip over dbg_value's.
    while (I != E && I->isDebugValue())
      ++I;

    if (I == E)
      return true;
    if (I->modifiesRegister(PhysDef, TRI))
      return false;

    --LookAheadLeft;
    ++I;
  }

  return false;
}

bool MachineCSE::isCSECandidate(MachineInstr *MI) {
  if (MI->isLabel() || MI->isPHI() || MI->isImplicitDef() ||
      MI->isKill() || MI->isInlineAsm() || MI->isDebugValue())
    return false;

  // Ignore copies.
  if (MI->isCopyLike())
    return false;

  // Ignore stuff that we obviously can't move.
  const TargetInstrDesc &TID = MI->getDesc();  
  if (TID.mayStore() || TID.isCall() || TID.isTerminator() ||
      TID.hasUnmodeledSideEffects())
    return false;

  if (TID.mayLoad()) {
    // Okay, this instruction does a load. As a refinement, we allow the target
    // to decide whether the loaded value is actually a constant. If so, we can
    // actually use it as a load.
    if (!MI->isInvariantLoad(AA))
      // FIXME: we should be able to hoist loads with no other side effects if
      // there are no other instructions which can change memory in this loop.
      // This is a trivial form of alias analysis.
      return false;
  }
  return true;
}

/// isProfitableToCSE - Return true if it's profitable to eliminate MI with a
/// common expression that defines Reg.
bool MachineCSE::isProfitableToCSE(unsigned CSReg, unsigned Reg,
                                   MachineInstr *CSMI, MachineInstr *MI) {
  // FIXME: Heuristics that works around the lack the live range splitting.

  // Heuristics #1: Don't cse "cheap" computating if the def is not local or in an
  // immediate predecessor. We don't want to increase register pressure and end up
  // causing other computation to be spilled.
  if (MI->getDesc().isAsCheapAsAMove()) {
    MachineBasicBlock *CSBB = CSMI->getParent();
    MachineBasicBlock *BB = MI->getParent();
    if (CSBB != BB && 
        find(CSBB->succ_begin(), CSBB->succ_end(), BB) == CSBB->succ_end())
      return false;
  }

  // Heuristics #2: If the expression doesn't not use a vr and the only use
  // of the redundant computation are copies, do not cse.
  bool HasVRegUse = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isUse() && MO.getReg() &&
        TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
      HasVRegUse = true;
      break;
    }
  }
  if (!HasVRegUse) {
    bool HasNonCopyUse = false;
    for (MachineRegisterInfo::use_nodbg_iterator I =  MRI->use_nodbg_begin(Reg),
           E = MRI->use_nodbg_end(); I != E; ++I) {
      MachineInstr *Use = &*I;
      // Ignore copies.
      if (!Use->isCopyLike()) {
        HasNonCopyUse = true;
        break;
      }
    }
    if (!HasNonCopyUse)
      return false;
  }

  // Heuristics #3: If the common subexpression is used by PHIs, do not reuse
  // it unless the defined value is already used in the BB of the new use.
  bool HasPHI = false;
  SmallPtrSet<MachineBasicBlock*, 4> CSBBs;
  for (MachineRegisterInfo::use_nodbg_iterator I =  MRI->use_nodbg_begin(CSReg),
       E = MRI->use_nodbg_end(); I != E; ++I) {
    MachineInstr *Use = &*I;
    HasPHI |= Use->isPHI();
    CSBBs.insert(Use->getParent());
  }

  if (!HasPHI)
    return true;
  return CSBBs.count(MI->getParent());
}

void MachineCSE::EnterScope(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Entering: " << MBB->getName() << '\n');
  ScopeType *Scope = new ScopeType(VNT);
  ScopeMap[MBB] = Scope;
}

void MachineCSE::ExitScope(MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Exiting: " << MBB->getName() << '\n');
  DenseMap<MachineBasicBlock*, ScopeType*>::iterator SI = ScopeMap.find(MBB);
  assert(SI != ScopeMap.end());
  ScopeMap.erase(SI);
  delete SI->second;
}

bool MachineCSE::ProcessBlock(MachineBasicBlock *MBB) {
  bool Changed = false;

  SmallVector<std::pair<unsigned, unsigned>, 8> CSEPairs;
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    if (!isCSECandidate(MI))
      continue;

    bool DefPhys = false;
    bool FoundCSE = VNT.count(MI);
    if (!FoundCSE) {
      // Look for trivial copy coalescing opportunities.
      if (PerformTrivialCoalescing(MI, MBB)) {
        // After coalescing MI itself may become a copy.
        if (MI->isCopyLike())
          continue;
        FoundCSE = VNT.count(MI);
      }
    }
    // FIXME: commute commutable instructions?

    // If the instruction defines a physical register and the value *may* be
    // used, then it's not safe to replace it with a common subexpression.
    unsigned PhysDef = 0;
    if (FoundCSE && hasLivePhysRegDefUse(MI, MBB, PhysDef)) {
      FoundCSE = false;

      // ... Unless the CS is local and it also defines the physical register
      // which is not clobbered in between.
      if (PhysDef) {
        unsigned CSVN = VNT.lookup(MI);
        MachineInstr *CSMI = Exps[CSVN];
        if (PhysRegDefReaches(CSMI, MI, PhysDef)) {
          FoundCSE = true;
          DefPhys = true;
        }
      }
    }

    if (!FoundCSE) {
      VNT.insert(MI, CurrVN++);
      Exps.push_back(MI);
      continue;
    }

    // Found a common subexpression, eliminate it.
    unsigned CSVN = VNT.lookup(MI);
    MachineInstr *CSMI = Exps[CSVN];
    DEBUG(dbgs() << "Examining: " << *MI);
    DEBUG(dbgs() << "*** Found a common subexpression: " << *CSMI);

    // Check if it's profitable to perform this CSE.
    bool DoCSE = true;
    unsigned NumDefs = MI->getDesc().getNumDefs();
    for (unsigned i = 0, e = MI->getNumOperands(); NumDefs && i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned OldReg = MO.getReg();
      unsigned NewReg = CSMI->getOperand(i).getReg();
      if (OldReg == NewReg)
        continue;
      assert(TargetRegisterInfo::isVirtualRegister(OldReg) &&
             TargetRegisterInfo::isVirtualRegister(NewReg) &&
             "Do not CSE physical register defs!");
      if (!isProfitableToCSE(NewReg, OldReg, CSMI, MI)) {
        DoCSE = false;
        break;
      }
      CSEPairs.push_back(std::make_pair(OldReg, NewReg));
      --NumDefs;
    }

    // Actually perform the elimination.
    if (DoCSE) {
      for (unsigned i = 0, e = CSEPairs.size(); i != e; ++i) {
        MRI->replaceRegWith(CSEPairs[i].first, CSEPairs[i].second);
        MRI->clearKillFlags(CSEPairs[i].second);
      }
      MI->eraseFromParent();
      ++NumCSEs;
      if (DefPhys)
        ++NumPhysCSEs;
    } else {
      DEBUG(dbgs() << "*** Not profitable, avoid CSE!\n");
      VNT.insert(MI, CurrVN++);
      Exps.push_back(MI);
    }
    CSEPairs.clear();
  }

  return Changed;
}

/// ExitScopeIfDone - Destroy scope for the MBB that corresponds to the given
/// dominator tree node if its a leaf or all of its children are done. Walk
/// up the dominator tree to destroy ancestors which are now done.
void
MachineCSE::ExitScopeIfDone(MachineDomTreeNode *Node,
                DenseMap<MachineDomTreeNode*, unsigned> &OpenChildren,
                DenseMap<MachineDomTreeNode*, MachineDomTreeNode*> &ParentMap) {
  if (OpenChildren[Node])
    return;

  // Pop scope.
  ExitScope(Node->getBlock());

  // Now traverse upwards to pop ancestors whose offsprings are all done.
  while (MachineDomTreeNode *Parent = ParentMap[Node]) {
    unsigned Left = --OpenChildren[Parent];
    if (Left != 0)
      break;
    ExitScope(Parent->getBlock());
    Node = Parent;
  }
}

bool MachineCSE::PerformCSE(MachineDomTreeNode *Node) {
  SmallVector<MachineDomTreeNode*, 32> Scopes;
  SmallVector<MachineDomTreeNode*, 8> WorkList;
  DenseMap<MachineDomTreeNode*, MachineDomTreeNode*> ParentMap;
  DenseMap<MachineDomTreeNode*, unsigned> OpenChildren;

  CurrVN = 0;

  // Perform a DFS walk to determine the order of visit.
  WorkList.push_back(Node);
  do {
    Node = WorkList.pop_back_val();
    Scopes.push_back(Node);
    const std::vector<MachineDomTreeNode*> &Children = Node->getChildren();
    unsigned NumChildren = Children.size();
    OpenChildren[Node] = NumChildren;
    for (unsigned i = 0; i != NumChildren; ++i) {
      MachineDomTreeNode *Child = Children[i];
      ParentMap[Child] = Node;
      WorkList.push_back(Child);
    }
  } while (!WorkList.empty());

  // Now perform CSE.
  bool Changed = false;
  for (unsigned i = 0, e = Scopes.size(); i != e; ++i) {
    MachineDomTreeNode *Node = Scopes[i];
    MachineBasicBlock *MBB = Node->getBlock();
    EnterScope(MBB);
    Changed |= ProcessBlock(MBB);
    // If it's a leaf node, it's done. Traverse upwards to pop ancestors.
    ExitScopeIfDone(Node, OpenChildren, ParentMap);
  }

  return Changed;
}

bool MachineCSE::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  TRI = MF.getTarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<MachineDominatorTree>();
  return PerformCSE(DT->getRootNode());
}
