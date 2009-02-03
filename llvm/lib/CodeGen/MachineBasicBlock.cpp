//===-- llvm/CodeGen/MachineBasicBlock.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Collect the sequence of machine instructions for a basic block.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/BasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetInstrDesc.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/LeakDetector.h"
#include <algorithm>
using namespace llvm;

MachineBasicBlock::MachineBasicBlock(MachineFunction &mf, const BasicBlock *bb)
  : BB(bb), Number(-1), xParent(&mf), Alignment(0), IsLandingPad(false) {
  Insts.Parent = this;
}

MachineBasicBlock::~MachineBasicBlock() {
  LeakDetector::removeGarbageObject(this);
}

std::ostream& llvm::operator<<(std::ostream &OS, const MachineBasicBlock &MBB) {
  MBB.print(OS);
  return OS;
}

/// addNodeToList (MBB) - When an MBB is added to an MF, we need to update the 
/// parent pointer of the MBB, the MBB numbering, and any instructions in the
/// MBB to be on the right operand list for registers.
///
/// MBBs start out as #-1. When a MBB is added to a MachineFunction, it
/// gets the next available unique MBB number. If it is removed from a
/// MachineFunction, it goes back to being #-1.
void ilist_traits<MachineBasicBlock>::addNodeToList(MachineBasicBlock* N) {
  MachineFunction &MF = *N->getParent();
  N->Number = MF.addToMBBNumbering(N);

  // Make sure the instructions have their operands in the reginfo lists.
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  for (MachineBasicBlock::iterator I = N->begin(), E = N->end(); I != E; ++I)
    I->AddRegOperandsToUseLists(RegInfo);

  LeakDetector::removeGarbageObject(N);
}

void ilist_traits<MachineBasicBlock>::removeNodeFromList(MachineBasicBlock* N) {
  N->getParent()->removeFromMBBNumbering(N->Number);
  N->Number = -1;
  LeakDetector::addGarbageObject(N);
}


/// addNodeToList (MI) - When we add an instruction to a basic block
/// list, we update its parent pointer and add its operands from reg use/def
/// lists if appropriate.
void ilist_traits<MachineInstr>::addNodeToList(MachineInstr* N) {
  assert(N->getParent() == 0 && "machine instruction already in a basic block");
  N->setParent(Parent);
  
  // Add the instruction's register operands to their corresponding
  // use/def lists.
  MachineFunction *MF = Parent->getParent();
  N->AddRegOperandsToUseLists(MF->getRegInfo());

  LeakDetector::removeGarbageObject(N);
}

/// removeNodeFromList (MI) - When we remove an instruction from a basic block
/// list, we update its parent pointer and remove its operands from reg use/def
/// lists if appropriate.
void ilist_traits<MachineInstr>::removeNodeFromList(MachineInstr* N) {
  assert(N->getParent() != 0 && "machine instruction not in a basic block");

  // Remove from the use/def lists.
  N->RemoveRegOperandsFromUseLists();
  
  N->setParent(0);

  LeakDetector::addGarbageObject(N);
}

/// transferNodesFromList (MI) - When moving a range of instructions from one
/// MBB list to another, we need to update the parent pointers and the use/def
/// lists.
void ilist_traits<MachineInstr>::transferNodesFromList(
      ilist_traits<MachineInstr>& fromList,
      MachineBasicBlock::iterator first,
      MachineBasicBlock::iterator last) {
  assert(Parent->getParent() == fromList.Parent->getParent() &&
        "MachineInstr parent mismatch!");

  // Splice within the same MBB -> no change.
  if (Parent == fromList.Parent) return;

  // If splicing between two blocks within the same function, just update the
  // parent pointers.
  for (; first != last; ++first)
    first->setParent(Parent);
}

void ilist_traits<MachineInstr>::deleteNode(MachineInstr* MI) {
  assert(!MI->getParent() && "MI is still in a block!");
  Parent->getParent()->DeleteMachineInstr(MI);
}

MachineBasicBlock::iterator MachineBasicBlock::getFirstTerminator() {
  iterator I = end();
  while (I != begin() && (--I)->getDesc().isTerminator())
    ; /*noop */
  if (I != end() && !I->getDesc().isTerminator()) ++I;
  return I;
}

void MachineBasicBlock::dump() const {
  print(*cerr.stream());
}

static inline void OutputReg(std::ostream &os, unsigned RegNo,
                             const TargetRegisterInfo *TRI = 0) {
  if (!RegNo || TargetRegisterInfo::isPhysicalRegister(RegNo)) {
    if (TRI)
      os << " %" << TRI->get(RegNo).Name;
    else
      os << " %mreg(" << RegNo << ")";
  } else
    os << " %reg" << RegNo;
}

void MachineBasicBlock::print(std::ostream &OS) const {
  const MachineFunction *MF = getParent();
  if(!MF) {
    OS << "Can't print out MachineBasicBlock because parent MachineFunction"
       << " is null\n";
    return;
  }

  const BasicBlock *LBB = getBasicBlock();
  OS << "\n";
  if (LBB) OS << LBB->getName() << ": ";
  OS << (const void*)this
     << ", LLVM BB @" << (const void*) LBB << ", ID#" << getNumber();
  if (Alignment) OS << ", Alignment " << Alignment;
  if (isLandingPad()) OS << ", EH LANDING PAD";
  OS << ":\n";

  const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();  
  if (!livein_empty()) {
    OS << "Live Ins:";
    for (const_livein_iterator I = livein_begin(),E = livein_end(); I != E; ++I)
      OutputReg(OS, *I, TRI);
    OS << "\n";
  }
  // Print the preds of this block according to the CFG.
  if (!pred_empty()) {
    OS << "    Predecessors according to CFG:";
    for (const_pred_iterator PI = pred_begin(), E = pred_end(); PI != E; ++PI)
      OS << " " << *PI << " (#" << (*PI)->getNumber() << ")";
    OS << "\n";
  }
  
  for (const_iterator I = begin(); I != end(); ++I) {
    OS << "\t";
    I->print(OS, &getParent()->getTarget());
  }

  // Print the successors of this block according to the CFG.
  if (!succ_empty()) {
    OS << "    Successors according to CFG:";
    for (const_succ_iterator SI = succ_begin(), E = succ_end(); SI != E; ++SI)
      OS << " " << *SI << " (#" << (*SI)->getNumber() << ")";
    OS << "\n";
  }
}

void MachineBasicBlock::removeLiveIn(unsigned Reg) {
  livein_iterator I = std::find(livein_begin(), livein_end(), Reg);
  assert(I != livein_end() && "Not a live in!");
  LiveIns.erase(I);
}

bool MachineBasicBlock::isLiveIn(unsigned Reg) const {
  const_livein_iterator I = std::find(livein_begin(), livein_end(), Reg);
  return I != livein_end();
}

void MachineBasicBlock::moveBefore(MachineBasicBlock *NewAfter) {
  getParent()->splice(NewAfter, this);
}

void MachineBasicBlock::moveAfter(MachineBasicBlock *NewBefore) {
  MachineFunction::iterator BBI = NewBefore;
  getParent()->splice(++BBI, this);
}


void MachineBasicBlock::addSuccessor(MachineBasicBlock *succ) {
  Successors.push_back(succ);
  succ->addPredecessor(this);
}

void MachineBasicBlock::removeSuccessor(MachineBasicBlock *succ) {
  succ->removePredecessor(this);
  succ_iterator I = std::find(Successors.begin(), Successors.end(), succ);
  assert(I != Successors.end() && "Not a current successor!");
  Successors.erase(I);
}

MachineBasicBlock::succ_iterator 
MachineBasicBlock::removeSuccessor(succ_iterator I) {
  assert(I != Successors.end() && "Not a current successor!");
  (*I)->removePredecessor(this);
  return Successors.erase(I);
}

void MachineBasicBlock::addPredecessor(MachineBasicBlock *pred) {
  Predecessors.push_back(pred);
}

void MachineBasicBlock::removePredecessor(MachineBasicBlock *pred) {
  std::vector<MachineBasicBlock *>::iterator I =
    std::find(Predecessors.begin(), Predecessors.end(), pred);
  assert(I != Predecessors.end() && "Pred is not a predecessor of this block!");
  Predecessors.erase(I);
}

void MachineBasicBlock::transferSuccessors(MachineBasicBlock *fromMBB)
{
  if (this == fromMBB)
    return;
  
  for(MachineBasicBlock::succ_iterator iter = fromMBB->succ_begin(), 
      end = fromMBB->succ_end(); iter != end; ++iter) {
      addSuccessor(*iter);
  }
  while(!fromMBB->succ_empty())
    fromMBB->removeSuccessor(fromMBB->succ_begin());
}

bool MachineBasicBlock::isSuccessor(MachineBasicBlock *MBB) const {
  std::vector<MachineBasicBlock *>::const_iterator I =
    std::find(Successors.begin(), Successors.end(), MBB);
  return I != Successors.end();
}

bool MachineBasicBlock::isLayoutSuccessor(MachineBasicBlock *MBB) const {
  MachineFunction::const_iterator I(this);
  return next(I) == MachineFunction::const_iterator(MBB);
}

/// removeFromParent - This method unlinks 'this' from the containing function,
/// and returns it, but does not delete it.
MachineBasicBlock *MachineBasicBlock::removeFromParent() {
  assert(getParent() && "Not embedded in a function!");
  getParent()->remove(this);
  return this;
}


/// eraseFromParent - This method unlinks 'this' from the containing function,
/// and deletes it.
void MachineBasicBlock::eraseFromParent() {
  assert(getParent() && "Not embedded in a function!");
  getParent()->erase(this);
}


/// ReplaceUsesOfBlockWith - Given a machine basic block that branched to
/// 'Old', change the code and CFG so that it branches to 'New' instead.
void MachineBasicBlock::ReplaceUsesOfBlockWith(MachineBasicBlock *Old,
                                               MachineBasicBlock *New) {
  assert(Old != New && "Cannot replace self with self!");

  MachineBasicBlock::iterator I = end();
  while (I != begin()) {
    --I;
    if (!I->getDesc().isTerminator()) break;

    // Scan the operands of this machine instruction, replacing any uses of Old
    // with New.
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      if (I->getOperand(i).isMBB() &&
          I->getOperand(i).getMBB() == Old)
        I->getOperand(i).setMBB(New);
  }

  // Update the successor information.  If New was already a successor, just
  // remove the link to Old instead of creating another one.  PR 1444.
  removeSuccessor(Old);
  if (!isSuccessor(New))
    addSuccessor(New);
}

/// CorrectExtraCFGEdges - Various pieces of code can cause excess edges in the
/// CFG to be inserted.  If we have proven that MBB can only branch to DestA and
/// DestB, remove any other MBB successors from the CFG.  DestA and DestB can
/// be null.
/// Besides DestA and DestB, retain other edges leading to LandingPads
/// (currently there can be only one; we don't check or require that here).
/// Note it is possible that DestA and/or DestB are LandingPads.
bool MachineBasicBlock::CorrectExtraCFGEdges(MachineBasicBlock *DestA,
                                             MachineBasicBlock *DestB,
                                             bool isCond) {
  bool MadeChange = false;
  bool AddedFallThrough = false;

  MachineFunction::iterator FallThru = next(MachineFunction::iterator(this));
  
  // If this block ends with a conditional branch that falls through to its
  // successor, set DestB as the successor.
  if (isCond) {
    if (DestB == 0 && FallThru != getParent()->end()) {
      DestB = FallThru;
      AddedFallThrough = true;
    }
  } else {
    // If this is an unconditional branch with no explicit dest, it must just be
    // a fallthrough into DestB.
    if (DestA == 0 && FallThru != getParent()->end()) {
      DestA = FallThru;
      AddedFallThrough = true;
    }
  }
  
  MachineBasicBlock::succ_iterator SI = succ_begin();
  MachineBasicBlock *OrigDestA = DestA, *OrigDestB = DestB;
  while (SI != succ_end()) {
    if (*SI == DestA && DestA == DestB) {
      DestA = DestB = 0;
      ++SI;
    } else if (*SI == DestA) {
      DestA = 0;
      ++SI;
    } else if (*SI == DestB) {
      DestB = 0;
      ++SI;
    } else if ((*SI)->isLandingPad() && 
               *SI!=OrigDestA && *SI!=OrigDestB) {
      ++SI;
    } else {
      // Otherwise, this is a superfluous edge, remove it.
      SI = removeSuccessor(SI);
      MadeChange = true;
    }
  }
  if (!AddedFallThrough) {
    assert(DestA == 0 && DestB == 0 &&
           "MachineCFG is missing edges!");
  } else if (isCond) {
    assert(DestA == 0 && "MachineCFG is missing edges!");
  }
  return MadeChange;
}
