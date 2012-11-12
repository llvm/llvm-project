//===- StrongPHIElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates PHI instructions by aggressively coalescing the copies
// that would be inserted by a naive algorithm and only inserting the copies
// that are necessary. The coalescing technique initially assumes that all
// registers appearing in a PHI instruction do not interfere. It then eliminates
// proven interferences, using dominators to only perform a linear number of
// interference tests instead of the quadratic number of interference tests
// that this would naively require. This is a technique derived from:
// 
//    Budimlic, et al. Fast copy coalescing and live-range identification.
//    In Proceedings of the ACM SIGPLAN 2002 Conference on Programming Language
//    Design and Implementation (Berlin, Germany, June 17 - 19, 2002).
//    PLDI '02. ACM, New York, NY, 25-32.
//
// The original implementation constructs a data structure they call a dominance
// forest for this purpose. The dominance forest was shown to be unnecessary,
// as it is possible to emulate the creation and traversal of a dominance forest
// by directly using the dominator tree, rather than actually constructing the
// dominance forest.  This technique is explained in:
//
//   Boissinot, et al. Revisiting Out-of-SSA Translation for Correctness, Code
//     Quality and Efficiency,
//   In Proceedings of the 7th annual IEEE/ACM International Symposium on Code
//   Generation and Optimization (Seattle, Washington, March 22 - 25, 2009).
//   CGO '09. IEEE, Washington, DC, 114-125.
//
// Careful implementation allows for all of the dominator forest interference
// checks to be performed at once in a single depth-first traversal of the
// dominator tree, which is what is implemented here.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strongphielim"
#include "PHIEliminationUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

namespace {
  class StrongPHIElimination : public MachineFunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    StrongPHIElimination() : MachineFunctionPass(ID) {
      initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
    }

    virtual void getAnalysisUsage(AnalysisUsage&) const;
    bool runOnMachineFunction(MachineFunction&);

  private:
    /// This struct represents a single node in the union-find data structure
    /// representing the variable congruence classes. There is one difference
    /// from a normal union-find data structure. We steal two bits from the parent
    /// pointer . One of these bits is used to represent whether the register
    /// itself has been isolated, and the other is used to represent whether the
    /// PHI with that register as its destination has been isolated.
    ///
    /// Note that this leads to the strange situation where the leader of a
    /// congruence class may no longer logically be a member, due to being
    /// isolated.
    struct Node {
      enum Flags {
        kRegisterIsolatedFlag = 1,
        kPHIIsolatedFlag = 2
      };
      Node(unsigned v) : value(v), rank(0) { parent.setPointer(this); }

      Node *getLeader();

      PointerIntPair<Node*, 2> parent;
      unsigned value;
      unsigned rank;
    };

    /// Add a register in a new congruence class containing only itself.
    void addReg(unsigned);

    /// Join the congruence classes of two registers. This function is biased
    /// towards the left argument, i.e. after
    ///
    /// addReg(r2);
    /// unionRegs(r1, r2);
    ///
    /// the leader of the unioned congruence class is the same as the leader of
    /// r1's congruence class prior to the union. This is actually relied upon
    /// in the copy insertion code.
    void unionRegs(unsigned, unsigned);

    /// Get the color of a register. The color is 0 if the register has been
    /// isolated.
    unsigned getRegColor(unsigned);

    // Isolate a register.
    void isolateReg(unsigned);

    /// Get the color of a PHI. The color of a PHI is 0 if the PHI has been
    /// isolated. Otherwise, it is the original color of its destination and
    /// all of its operands (before they were isolated, if they were).
    unsigned getPHIColor(MachineInstr*);

    /// Isolate a PHI.
    void isolatePHI(MachineInstr*);

    /// Traverses a basic block, splitting any interferences found between
    /// registers in the same congruence class. It takes two DenseMaps as
    /// arguments that it also updates: CurrentDominatingParent, which maps
    /// a color to the register in that congruence class whose definition was
    /// most recently seen, and ImmediateDominatingParent, which maps a register
    /// to the register in the same congruence class that most immediately
    /// dominates it.
    ///
    /// This function assumes that it is being called in a depth-first traversal
    /// of the dominator tree.
    void SplitInterferencesForBasicBlock(
      MachineBasicBlock&,
      DenseMap<unsigned, unsigned> &CurrentDominatingParent,
      DenseMap<unsigned, unsigned> &ImmediateDominatingParent);

    // Lowers a PHI instruction, inserting copies of the source and destination
    // registers as necessary.
    void InsertCopiesForPHI(MachineInstr*, MachineBasicBlock*);

    // Merges the live interval of Reg into NewReg and renames Reg to NewReg
    // everywhere that Reg appears. Requires Reg and NewReg to have non-
    // overlapping lifetimes.
    void MergeLIsAndRename(unsigned Reg, unsigned NewReg);

    MachineRegisterInfo *MRI;
    const TargetInstrInfo *TII;
    MachineDominatorTree *DT;
    LiveIntervals *LI;

    BumpPtrAllocator Allocator;

    DenseMap<unsigned, Node*> RegNodeMap;

    // Maps a basic block to a list of its defs of registers that appear as PHI
    // sources.
    DenseMap<MachineBasicBlock*, std::vector<MachineInstr*> > PHISrcDefs;

    // Maps a color to a pair of a MachineInstr* and a virtual register, which
    // is the operand of that PHI corresponding to the current basic block.
    DenseMap<unsigned, std::pair<MachineInstr*, unsigned> > CurrentPHIForColor;

    // FIXME: Can these two data structures be combined? Would a std::multimap
    // be any better?

    // Stores pairs of predecessor basic blocks and the source registers of
    // inserted copy instructions.
    typedef DenseSet<std::pair<MachineBasicBlock*, unsigned> > SrcCopySet;
    SrcCopySet InsertedSrcCopySet;

    // Maps pairs of predecessor basic blocks and colors to their defining copy
    // instructions.
    typedef DenseMap<std::pair<MachineBasicBlock*, unsigned>, MachineInstr*>
      SrcCopyMap;
    SrcCopyMap InsertedSrcCopyMap;

    // Maps inserted destination copy registers to their defining copy
    // instructions.
    typedef DenseMap<unsigned, MachineInstr*> DestCopyMap;
    DestCopyMap InsertedDestCopies;
  };

  struct MIIndexCompare {
    MIIndexCompare(LiveIntervals *LiveIntervals) : LI(LiveIntervals) { }

    bool operator()(const MachineInstr *LHS, const MachineInstr *RHS) const {
      return LI->getInstructionIndex(LHS) < LI->getInstructionIndex(RHS);
    }

    LiveIntervals *LI;
  };
} // namespace

STATISTIC(NumPHIsLowered, "Number of PHIs lowered");
STATISTIC(NumDestCopiesInserted, "Number of destination copies inserted");
STATISTIC(NumSrcCopiesInserted, "Number of source copies inserted");

char StrongPHIElimination::ID = 0;
INITIALIZE_PASS_BEGIN(StrongPHIElimination, "strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(StrongPHIElimination, "strong-phi-node-elimination",
  "Eliminate PHI nodes for register allocation, intelligently", false, false)

char &llvm::StrongPHIEliminationID = StrongPHIElimination::ID;

void StrongPHIElimination::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineDominatorTree>();
  AU.addRequired<SlotIndexes>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

static MachineOperand *findLastUse(MachineBasicBlock *MBB, unsigned Reg) {
  // FIXME: This only needs to check from the first terminator, as only the
  // first terminator can use a virtual register.
  for (MachineBasicBlock::reverse_iterator RI = MBB->rbegin(); ; ++RI) {
    assert (RI != MBB->rend());
    MachineInstr *MI = &*RI;

    for (MachineInstr::mop_iterator OI = MI->operands_begin(),
         OE = MI->operands_end(); OI != OE; ++OI) {
      MachineOperand &MO = *OI;
      if (MO.isReg() && MO.isUse() && MO.getReg() == Reg)
        return &MO;
    }
  }
}

bool StrongPHIElimination::runOnMachineFunction(MachineFunction &MF) {
  MRI = &MF.getRegInfo();
  TII = MF.getTarget().getInstrInfo();
  DT = &getAnalysis<MachineDominatorTree>();
  LI = &getAnalysis<LiveIntervals>();

  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    for (MachineBasicBlock::iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->isPHI(); ++BBI) {
      unsigned DestReg = BBI->getOperand(0).getReg();
      addReg(DestReg);
      PHISrcDefs[I].push_back(BBI);

      for (unsigned i = 1; i < BBI->getNumOperands(); i += 2) {
        MachineOperand &SrcMO = BBI->getOperand(i);
        unsigned SrcReg = SrcMO.getReg();
        addReg(SrcReg);
        unionRegs(DestReg, SrcReg);

        MachineInstr *DefMI = MRI->getVRegDef(SrcReg);
        if (DefMI)
          PHISrcDefs[DefMI->getParent()].push_back(DefMI);
      }
    }
  }

  // Perform a depth-first traversal of the dominator tree, splitting
  // interferences amongst PHI-congruence classes.
  DenseMap<unsigned, unsigned> CurrentDominatingParent;
  DenseMap<unsigned, unsigned> ImmediateDominatingParent;
  for (df_iterator<MachineDomTreeNode*> DI = df_begin(DT->getRootNode()),
       DE = df_end(DT->getRootNode()); DI != DE; ++DI) {
    SplitInterferencesForBasicBlock(*DI->getBlock(),
                                    CurrentDominatingParent,
                                    ImmediateDominatingParent);
  }

  // Insert copies for all PHI source and destination registers.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    for (MachineBasicBlock::iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->isPHI(); ++BBI) {
      InsertCopiesForPHI(BBI, I);
    }
  }

  // FIXME: Preserve the equivalence classes during copy insertion and use
  // the preversed equivalence classes instead of recomputing them.
  RegNodeMap.clear();
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    for (MachineBasicBlock::iterator BBI = I->begin(), BBE = I->end();
         BBI != BBE && BBI->isPHI(); ++BBI) {
      unsigned DestReg = BBI->getOperand(0).getReg();
      addReg(DestReg);

      for (unsigned i = 1; i < BBI->getNumOperands(); i += 2) {
        unsigned SrcReg = BBI->getOperand(i).getReg();
        addReg(SrcReg);
        unionRegs(DestReg, SrcReg);
      }
    }
  }

  DenseMap<unsigned, unsigned> RegRenamingMap;
  bool Changed = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    MachineBasicBlock::iterator BBI = I->begin(), BBE = I->end();
    while (BBI != BBE && BBI->isPHI()) {
      MachineInstr *PHI = BBI;

      assert(PHI->getNumOperands() > 0);

      unsigned SrcReg = PHI->getOperand(1).getReg();
      unsigned SrcColor = getRegColor(SrcReg);
      unsigned NewReg = RegRenamingMap[SrcColor];
      if (!NewReg) {
        NewReg = SrcReg;
        RegRenamingMap[SrcColor] = SrcReg;
      }
      MergeLIsAndRename(SrcReg, NewReg);

      unsigned DestReg = PHI->getOperand(0).getReg();
      if (!InsertedDestCopies.count(DestReg))
        MergeLIsAndRename(DestReg, NewReg);

      for (unsigned i = 3; i < PHI->getNumOperands(); i += 2) {
        unsigned SrcReg = PHI->getOperand(i).getReg();
        MergeLIsAndRename(SrcReg, NewReg);
      }

      ++BBI;
      LI->RemoveMachineInstrFromMaps(PHI);
      PHI->eraseFromParent();
      Changed = true;
    }
  }

  // Due to the insertion of copies to split live ranges, the live intervals are
  // guaranteed to not overlap, except in one case: an original PHI source and a
  // PHI destination copy. In this case, they have the same value and thus don't
  // truly intersect, so we merge them into the value live at that point.
  // FIXME: Is there some better way we can handle this?
  for (DestCopyMap::iterator I = InsertedDestCopies.begin(),
       E = InsertedDestCopies.end(); I != E; ++I) {
    unsigned DestReg = I->first;
    unsigned DestColor = getRegColor(DestReg);
    unsigned NewReg = RegRenamingMap[DestColor];

    LiveInterval &DestLI = LI->getInterval(DestReg);
    LiveInterval &NewLI = LI->getInterval(NewReg);

    assert(DestLI.ranges.size() == 1
           && "PHI destination copy's live interval should be a single live "
               "range from the beginning of the BB to the copy instruction.");
    LiveRange *DestLR = DestLI.begin();
    VNInfo *NewVNI = NewLI.getVNInfoAt(DestLR->start);
    if (!NewVNI) {
      NewVNI = NewLI.createValueCopy(DestLR->valno, LI->getVNInfoAllocator());
      MachineInstr *CopyInstr = I->second;
      CopyInstr->getOperand(1).setIsKill(true);
    }

    LiveRange NewLR(DestLR->start, DestLR->end, NewVNI);
    NewLI.addRange(NewLR);

    LI->removeInterval(DestReg);
    MRI->replaceRegWith(DestReg, NewReg);
  }

  // Adjust the live intervals of all PHI source registers to handle the case
  // where the PHIs in successor blocks were the only later uses of the source
  // register.
  for (SrcCopySet::iterator I = InsertedSrcCopySet.begin(),
       E = InsertedSrcCopySet.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I->first;
    unsigned SrcReg = I->second;
    if (unsigned RenamedRegister = RegRenamingMap[getRegColor(SrcReg)])
      SrcReg = RenamedRegister;

    LiveInterval &SrcLI = LI->getInterval(SrcReg);

    bool isLiveOut = false;
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
      if (SrcLI.liveAt(LI->getMBBStartIdx(*SI))) {
        isLiveOut = true;
        break;
      }
    }

    if (isLiveOut)
      continue;

    MachineOperand *LastUse = findLastUse(MBB, SrcReg);
    assert(LastUse);
    SlotIndex LastUseIndex = LI->getInstructionIndex(LastUse->getParent());
    SrcLI.removeRange(LastUseIndex.getRegSlot(), LI->getMBBEndIdx(MBB));
    LastUse->setIsKill(true);
  }

  Allocator.Reset();
  RegNodeMap.clear();
  PHISrcDefs.clear();
  InsertedSrcCopySet.clear();
  InsertedSrcCopyMap.clear();
  InsertedDestCopies.clear();

  return Changed;
}

void StrongPHIElimination::addReg(unsigned Reg) {
  Node *&N = RegNodeMap[Reg];
  if (!N)
    N = new (Allocator) Node(Reg);
}

StrongPHIElimination::Node*
StrongPHIElimination::Node::getLeader() {
  Node *N = this;
  Node *Parent = parent.getPointer();
  Node *Grandparent = Parent->parent.getPointer();

  while (Parent != Grandparent) {
    N->parent.setPointer(Grandparent);
    N = Grandparent;
    Parent = Parent->parent.getPointer();
    Grandparent = Parent->parent.getPointer();
  }

  return Parent;
}

unsigned StrongPHIElimination::getRegColor(unsigned Reg) {
  DenseMap<unsigned, Node*>::iterator RI = RegNodeMap.find(Reg);
  if (RI == RegNodeMap.end())
    return 0;
  Node *Node = RI->second;
  if (Node->parent.getInt() & Node::kRegisterIsolatedFlag)
    return 0;
  return Node->getLeader()->value;
}

void StrongPHIElimination::unionRegs(unsigned Reg1, unsigned Reg2) {
  Node *Node1 = RegNodeMap[Reg1]->getLeader();
  Node *Node2 = RegNodeMap[Reg2]->getLeader();

  if (Node1->rank > Node2->rank) {
    Node2->parent.setPointer(Node1->getLeader());
  } else if (Node1->rank < Node2->rank) {
    Node1->parent.setPointer(Node2->getLeader());
  } else if (Node1 != Node2) {
    Node2->parent.setPointer(Node1->getLeader());
    Node1->rank++;
  }
}

void StrongPHIElimination::isolateReg(unsigned Reg) {
  Node *Node = RegNodeMap[Reg];
  Node->parent.setInt(Node->parent.getInt() | Node::kRegisterIsolatedFlag);
}

unsigned StrongPHIElimination::getPHIColor(MachineInstr *PHI) {
  assert(PHI->isPHI());

  unsigned DestReg = PHI->getOperand(0).getReg();
  Node *DestNode = RegNodeMap[DestReg];
  if (DestNode->parent.getInt() & Node::kPHIIsolatedFlag)
    return 0;

  for (unsigned i = 1; i < PHI->getNumOperands(); i += 2) {
    unsigned SrcColor = getRegColor(PHI->getOperand(i).getReg());
    if (SrcColor)
      return SrcColor;
  }
  return 0;
}

void StrongPHIElimination::isolatePHI(MachineInstr *PHI) {
  assert(PHI->isPHI());
  Node *Node = RegNodeMap[PHI->getOperand(0).getReg()];
  Node->parent.setInt(Node->parent.getInt() | Node::kPHIIsolatedFlag);
}

/// SplitInterferencesForBasicBlock - traverses a basic block, splitting any
/// interferences found between registers in the same congruence class. It
/// takes two DenseMaps as arguments that it also updates:
///
/// 1) CurrentDominatingParent, which maps a color to the register in that
///    congruence class whose definition was most recently seen.
///
/// 2) ImmediateDominatingParent, which maps a register to the register in the
///    same congruence class that most immediately dominates it.
///
/// This function assumes that it is being called in a depth-first traversal
/// of the dominator tree.
///
/// The algorithm used here is a generalization of the dominance-based SSA test
/// for two variables. If there are variables a_1, ..., a_n such that
///
///   def(a_1) dom ... dom def(a_n),
///
/// then we can test for an interference between any two a_i by only using O(n)
/// interference tests between pairs of variables. If i < j and a_i and a_j
/// interfere, then a_i is alive at def(a_j), so it is also alive at def(a_i+1).
/// Thus, in order to test for an interference involving a_i, we need only check
/// for a potential interference with a_i+1.
///
/// This method can be generalized to arbitrary sets of variables by performing
/// a depth-first traversal of the dominator tree. As we traverse down a branch
/// of the dominator tree, we keep track of the current dominating variable and
/// only perform an interference test with that variable. However, when we go to
/// another branch of the dominator tree, the definition of the current dominating
/// variable may no longer dominate the current block. In order to correct this,
/// we need to use a stack of past choices of the current dominating variable
/// and pop from this stack until we find a variable whose definition actually
/// dominates the current block.
/// 
/// There will be one push on this stack for each variable that has become the
/// current dominating variable, so instead of using an explicit stack we can
/// simply associate the previous choice for a current dominating variable with
/// the new choice. This works better in our implementation, where we test for
/// interference in multiple distinct sets at once.
void
StrongPHIElimination::SplitInterferencesForBasicBlock(
    MachineBasicBlock &MBB,
    DenseMap<unsigned, unsigned> &CurrentDominatingParent,
    DenseMap<unsigned, unsigned> &ImmediateDominatingParent) {
  // Sort defs by their order in the original basic block, as the code below
  // assumes that it is processing definitions in dominance order.
  std::vector<MachineInstr*> &DefInstrs = PHISrcDefs[&MBB];
  std::sort(DefInstrs.begin(), DefInstrs.end(), MIIndexCompare(LI));

  for (std::vector<MachineInstr*>::const_iterator BBI = DefInstrs.begin(),
       BBE = DefInstrs.end(); BBI != BBE; ++BBI) {
    for (MachineInstr::const_mop_iterator I = (*BBI)->operands_begin(),
         E = (*BBI)->operands_end(); I != E; ++I) {
      const MachineOperand &MO = *I;

      // FIXME: This would be faster if it were possible to bail out of checking
      // an instruction's operands after the explicit defs, but this is incorrect
      // for variadic instructions, which may appear before register allocation
      // in the future.
      if (!MO.isReg() || !MO.isDef())
        continue;

      unsigned DestReg = MO.getReg();
      if (!DestReg || !TargetRegisterInfo::isVirtualRegister(DestReg))
        continue;

      // If the virtual register being defined is not used in any PHI or has
      // already been isolated, then there are no more interferences to check.
      unsigned DestColor = getRegColor(DestReg);
      if (!DestColor)
        continue;

      // The input to this pass sometimes is not in SSA form in every basic
      // block, as some virtual registers have redefinitions. We could eliminate
      // this by fixing the passes that generate the non-SSA code, or we could
      // handle it here by tracking defining machine instructions rather than
      // virtual registers. For now, we just handle the situation conservatively
      // in a way that will possibly lead to false interferences.
      unsigned &CurrentParent = CurrentDominatingParent[DestColor];
      unsigned NewParent = CurrentParent;
      if (NewParent == DestReg)
        continue;

      // Pop registers from the stack represented by ImmediateDominatingParent
      // until we find a parent that dominates the current instruction.
      while (NewParent && (!DT->dominates(MRI->getVRegDef(NewParent), *BBI)
                           || !getRegColor(NewParent)))
        NewParent = ImmediateDominatingParent[NewParent];

      // If NewParent is nonzero, then its definition dominates the current
      // instruction, so it is only necessary to check for the liveness of
      // NewParent in order to check for an interference.
      if (NewParent
          && LI->getInterval(NewParent).liveAt(LI->getInstructionIndex(*BBI))) {
        // If there is an interference, always isolate the new register. This
        // could be improved by using a heuristic that decides which of the two
        // registers to isolate.
        isolateReg(DestReg);
        CurrentParent = NewParent;
      } else {
        // If there is no interference, update ImmediateDominatingParent and set
        // the CurrentDominatingParent for this color to the current register.
        ImmediateDominatingParent[DestReg] = NewParent;
        CurrentParent = DestReg;
      }
    }
  }

  // We now walk the PHIs in successor blocks and check for interferences. This
  // is necessary because the use of a PHI's operands are logically contained in
  // the predecessor block. The def of a PHI's destination register is processed
  // along with the other defs in a basic block.

  CurrentPHIForColor.clear();

  for (MachineBasicBlock::succ_iterator SI = MBB.succ_begin(),
       SE = MBB.succ_end(); SI != SE; ++SI) {
    for (MachineBasicBlock::iterator BBI = (*SI)->begin(), BBE = (*SI)->end();
         BBI != BBE && BBI->isPHI(); ++BBI) {
      MachineInstr *PHI = BBI;

      // If a PHI is already isolated, either by being isolated directly or
      // having all of its operands isolated, ignore it.
      unsigned Color = getPHIColor(PHI);
      if (!Color)
        continue;

      // Find the index of the PHI operand that corresponds to this basic block.
      unsigned PredIndex;
      for (PredIndex = 1; PredIndex < PHI->getNumOperands(); PredIndex += 2) {
        if (PHI->getOperand(PredIndex + 1).getMBB() == &MBB)
          break;
      }
      assert(PredIndex < PHI->getNumOperands());
      unsigned PredOperandReg = PHI->getOperand(PredIndex).getReg();

      // Pop registers from the stack represented by ImmediateDominatingParent
      // until we find a parent that dominates the current instruction.
      unsigned &CurrentParent = CurrentDominatingParent[Color];
      unsigned NewParent = CurrentParent;
      while (NewParent
             && (!DT->dominates(MRI->getVRegDef(NewParent)->getParent(), &MBB)
                 || !getRegColor(NewParent)))
        NewParent = ImmediateDominatingParent[NewParent];
      CurrentParent = NewParent;

      // If there is an interference with a register, always isolate the
      // register rather than the PHI. It is also possible to isolate the
      // PHI, but that introduces copies for all of the registers involved
      // in that PHI.
      if (NewParent && LI->isLiveOutOfMBB(LI->getInterval(NewParent), &MBB)
                    && NewParent != PredOperandReg)
        isolateReg(NewParent);

      std::pair<MachineInstr*, unsigned>
        &CurrentPHI = CurrentPHIForColor[Color];

      // If two PHIs have the same operand from every shared predecessor, then
      // they don't actually interfere. Otherwise, isolate the current PHI. This
      // could possibly be improved, e.g. we could isolate the PHI with the
      // fewest operands.
      if (CurrentPHI.first && CurrentPHI.second != PredOperandReg)
        isolatePHI(PHI);
      else
        CurrentPHI = std::make_pair(PHI, PredOperandReg);
    }
  }
}

void StrongPHIElimination::InsertCopiesForPHI(MachineInstr *PHI,
                                              MachineBasicBlock *MBB) {
  assert(PHI->isPHI());
  ++NumPHIsLowered;
  unsigned PHIColor = getPHIColor(PHI);

  for (unsigned i = 1; i < PHI->getNumOperands(); i += 2) {
    MachineOperand &SrcMO = PHI->getOperand(i);

    // If a source is defined by an implicit def, there is no need to insert a
    // copy in the predecessor.
    if (SrcMO.isUndef())
      continue;

    unsigned SrcReg = SrcMO.getReg();
    assert(TargetRegisterInfo::isVirtualRegister(SrcReg) &&
           "Machine PHI Operands must all be virtual registers!");

    MachineBasicBlock *PredBB = PHI->getOperand(i + 1).getMBB();
    unsigned SrcColor = getRegColor(SrcReg);

    // If neither the PHI nor the operand were isolated, then we only need to
    // set the phi-kill flag on the VNInfo at this PHI.
    if (PHIColor && SrcColor == PHIColor) {
      LiveInterval &SrcInterval = LI->getInterval(SrcReg);
      SlotIndex PredIndex = LI->getMBBEndIdx(PredBB);
      VNInfo *SrcVNI = SrcInterval.getVNInfoBefore(PredIndex);
      (void)SrcVNI;
      assert(SrcVNI);
      continue;
    }

    unsigned CopyReg = 0;
    if (PHIColor) {
      SrcCopyMap::const_iterator I
        = InsertedSrcCopyMap.find(std::make_pair(PredBB, PHIColor));
      CopyReg
        = I != InsertedSrcCopyMap.end() ? I->second->getOperand(0).getReg() : 0;
    }

    if (!CopyReg) {
      const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
      CopyReg = MRI->createVirtualRegister(RC);

      MachineBasicBlock::iterator
        CopyInsertPoint = findPHICopyInsertPoint(PredBB, MBB, SrcReg);
      unsigned SrcSubReg = SrcMO.getSubReg();
      MachineInstr *CopyInstr = BuildMI(*PredBB,
                                        CopyInsertPoint,
                                        PHI->getDebugLoc(),
                                        TII->get(TargetOpcode::COPY),
                                        CopyReg).addReg(SrcReg, 0, SrcSubReg);
      LI->InsertMachineInstrInMaps(CopyInstr);
      ++NumSrcCopiesInserted;

      // addLiveRangeToEndOfBlock() also adds the phikill flag to the VNInfo for
      // the newly added range.
      LI->addLiveRangeToEndOfBlock(CopyReg, CopyInstr);
      InsertedSrcCopySet.insert(std::make_pair(PredBB, SrcReg));

      addReg(CopyReg);
      if (PHIColor) {
        unionRegs(PHIColor, CopyReg);
        assert(getRegColor(CopyReg) != CopyReg);
      } else {
        PHIColor = CopyReg;
        assert(getRegColor(CopyReg) == CopyReg);
      }

      // Insert into map if not already there.
      InsertedSrcCopyMap.insert(std::make_pair(std::make_pair(PredBB, PHIColor),
                                               CopyInstr));
    }

    SrcMO.setReg(CopyReg);

    // If SrcReg is not live beyond the PHI, trim its interval so that it is no
    // longer live-in to MBB. Note that SrcReg may appear in other PHIs that are
    // processed later, but this is still correct to do at this point because we
    // never rely on LiveIntervals being correct while inserting copies.
    // FIXME: Should this just count uses at PHIs like the normal PHIElimination
    // pass does?
    LiveInterval &SrcLI = LI->getInterval(SrcReg);
    SlotIndex MBBStartIndex = LI->getMBBStartIdx(MBB);
    SlotIndex PHIIndex = LI->getInstructionIndex(PHI);
    SlotIndex NextInstrIndex = PHIIndex.getNextIndex();
    if (SrcLI.liveAt(MBBStartIndex) && SrcLI.expiredAt(NextInstrIndex))
      SrcLI.removeRange(MBBStartIndex, PHIIndex, true);
  }

  unsigned DestReg = PHI->getOperand(0).getReg();
  unsigned DestColor = getRegColor(DestReg);

  if (PHIColor && DestColor == PHIColor) {
    LiveInterval &DestLI = LI->getInterval(DestReg);

    // Set the phi-def flag for the VN at this PHI.
    SlotIndex PHIIndex = LI->getInstructionIndex(PHI);
    VNInfo *DestVNI = DestLI.getVNInfoAt(PHIIndex.getRegSlot());
    assert(DestVNI);
  
    // Prior to PHI elimination, the live ranges of PHIs begin at their defining
    // instruction. After PHI elimination, PHI instructions are replaced by VNs
    // with the phi-def flag set, and the live ranges of these VNs start at the
    // beginning of the basic block.
    SlotIndex MBBStartIndex = LI->getMBBStartIdx(MBB);
    DestVNI->def = MBBStartIndex;
    DestLI.addRange(LiveRange(MBBStartIndex,
                              PHIIndex.getRegSlot(),
                              DestVNI));
    return;
  }

  const TargetRegisterClass *RC = MRI->getRegClass(DestReg);
  unsigned CopyReg = MRI->createVirtualRegister(RC);

  MachineInstr *CopyInstr = BuildMI(*MBB,
                                    MBB->SkipPHIsAndLabels(MBB->begin()),
                                    PHI->getDebugLoc(),
                                    TII->get(TargetOpcode::COPY),
                                    DestReg).addReg(CopyReg);
  LI->InsertMachineInstrInMaps(CopyInstr);
  PHI->getOperand(0).setReg(CopyReg);
  ++NumDestCopiesInserted;

  // Add the region from the beginning of MBB to the copy instruction to
  // CopyReg's live interval, and give the VNInfo the phidef flag.
  LiveInterval &CopyLI = LI->getOrCreateInterval(CopyReg);
  SlotIndex MBBStartIndex = LI->getMBBStartIdx(MBB);
  SlotIndex DestCopyIndex = LI->getInstructionIndex(CopyInstr);
  VNInfo *CopyVNI = CopyLI.getNextValue(MBBStartIndex,
                                        LI->getVNInfoAllocator());
  CopyLI.addRange(LiveRange(MBBStartIndex,
                            DestCopyIndex.getRegSlot(),
                            CopyVNI));

  // Adjust DestReg's live interval to adjust for its new definition at
  // CopyInstr.
  LiveInterval &DestLI = LI->getOrCreateInterval(DestReg);
  SlotIndex PHIIndex = LI->getInstructionIndex(PHI);
  DestLI.removeRange(PHIIndex.getRegSlot(), DestCopyIndex.getRegSlot());

  VNInfo *DestVNI = DestLI.getVNInfoAt(DestCopyIndex.getRegSlot());
  assert(DestVNI);
  DestVNI->def = DestCopyIndex.getRegSlot();

  InsertedDestCopies[CopyReg] = CopyInstr;
}

void StrongPHIElimination::MergeLIsAndRename(unsigned Reg, unsigned NewReg) {
  if (Reg == NewReg)
    return;

  LiveInterval &OldLI = LI->getInterval(Reg);
  LiveInterval &NewLI = LI->getInterval(NewReg);

  // Merge the live ranges of the two registers.
  DenseMap<VNInfo*, VNInfo*> VNMap;
  for (LiveInterval::iterator LRI = OldLI.begin(), LRE = OldLI.end();
       LRI != LRE; ++LRI) {
    LiveRange OldLR = *LRI;
    VNInfo *OldVN = OldLR.valno;

    VNInfo *&NewVN = VNMap[OldVN];
    if (!NewVN) {
      NewVN = NewLI.createValueCopy(OldVN, LI->getVNInfoAllocator());
      VNMap[OldVN] = NewVN;
    }

    LiveRange LR(OldLR.start, OldLR.end, NewVN);
    NewLI.addRange(LR);
  }

  // Remove the LiveInterval for the register being renamed and replace all
  // of its defs and uses with the new register.
  LI->removeInterval(Reg);
  MRI->replaceRegWith(Reg, NewReg);
}
