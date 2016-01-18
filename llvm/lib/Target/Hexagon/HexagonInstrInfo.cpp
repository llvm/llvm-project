//===-- HexagonInstrInfo.cpp - Hexagon Instruction Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "HexagonInstrInfo.h"
#include "Hexagon.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-instrinfo"

#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRMAP_INFO
#include "HexagonGenInstrInfo.inc"
#include "HexagonGenDFAPacketizer.inc"

///
/// Constants for Hexagon instructions.
///
const int Hexagon_MEMW_OFFSET_MAX = 4095;
const int Hexagon_MEMW_OFFSET_MIN = -4096;
const int Hexagon_MEMD_OFFSET_MAX = 8191;
const int Hexagon_MEMD_OFFSET_MIN = -8192;
const int Hexagon_MEMH_OFFSET_MAX = 2047;
const int Hexagon_MEMH_OFFSET_MIN = -2048;
const int Hexagon_MEMB_OFFSET_MAX = 1023;
const int Hexagon_MEMB_OFFSET_MIN = -1024;
const int Hexagon_ADDI_OFFSET_MAX = 32767;
const int Hexagon_ADDI_OFFSET_MIN = -32768;
const int Hexagon_MEMD_AUTOINC_MAX = 56;
const int Hexagon_MEMD_AUTOINC_MIN = -64;
const int Hexagon_MEMW_AUTOINC_MAX = 28;
const int Hexagon_MEMW_AUTOINC_MIN = -32;
const int Hexagon_MEMH_AUTOINC_MAX = 14;
const int Hexagon_MEMH_AUTOINC_MIN = -16;
const int Hexagon_MEMB_AUTOINC_MAX = 7;
const int Hexagon_MEMB_AUTOINC_MIN = -8;

// Pin the vtable to this file.
void HexagonInstrInfo::anchor() {}

HexagonInstrInfo::HexagonInstrInfo(HexagonSubtarget &ST)
    : HexagonGenInstrInfo(Hexagon::ADJCALLSTACKDOWN, Hexagon::ADJCALLSTACKUP),
      RI(), Subtarget(ST) {}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned HexagonInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {


  switch (MI->getOpcode()) {
  default: break;
  case Hexagon::L2_loadri_io:
  case Hexagon::L2_loadrd_io:
  case Hexagon::L2_loadrh_io:
  case Hexagon::L2_loadrb_io:
  case Hexagon::L2_loadrub_io:
    if (MI->getOperand(2).isFI() &&
        MI->getOperand(1).isImm() && (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}


/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned HexagonInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case Hexagon::S2_storeri_io:
  case Hexagon::S2_storerd_io:
  case Hexagon::S2_storerh_io:
  case Hexagon::S2_storerb_io:
    if (MI->getOperand(2).isFI() &&
        MI->getOperand(1).isImm() && (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
    break;
  }
  return 0;
}

// Find the hardware loop instruction used to set-up the specified loop.
// On Hexagon, we have two instructions used to set-up the hardware loop
// (LOOP0, LOOP1) with corresponding endloop (ENDLOOP0, ENDLOOP1) instructions
// to indicate the end of a loop.
static MachineInstr *
findLoopInstr(MachineBasicBlock *BB, int EndLoopOp,
              SmallPtrSet<MachineBasicBlock *, 8> &Visited) {
  int LOOPi;
  int LOOPr;
  if (EndLoopOp == Hexagon::ENDLOOP0) {
    LOOPi = Hexagon::J2_loop0i;
    LOOPr = Hexagon::J2_loop0r;
  } else { // EndLoopOp == Hexagon::EndLOOP1
    LOOPi = Hexagon::J2_loop1i;
    LOOPr = Hexagon::J2_loop1r;
  }

  // The loop set-up instruction will be in a predecessor block
  for (MachineBasicBlock::pred_iterator PB = BB->pred_begin(),
         PE = BB->pred_end(); PB != PE; ++PB) {
    // If this has been visited, already skip it.
    if (!Visited.insert(*PB).second)
      continue;
    if (*PB == BB)
      continue;
    for (MachineBasicBlock::reverse_instr_iterator I = (*PB)->instr_rbegin(),
           E = (*PB)->instr_rend(); I != E; ++I) {
      int Opc = I->getOpcode();
      if (Opc == LOOPi || Opc == LOOPr)
        return &*I;
      // We've reached a different loop, which means the loop0 has been removed.
      if (Opc == EndLoopOp)
        return 0;
    }
    // Check the predecessors for the LOOP instruction.
    MachineInstr *loop = findLoopInstr(*PB, EndLoopOp, Visited);
    if (loop)
      return loop;
  }
  return 0;
}

unsigned HexagonInstrInfo::InsertBranch(
    MachineBasicBlock &MBB,MachineBasicBlock *TBB, MachineBasicBlock *FBB,
    ArrayRef<MachineOperand> Cond, DebugLoc DL) const {

  Opcode_t BOpc   = Hexagon::J2_jump;
  Opcode_t BccOpc = Hexagon::J2_jumpt;

  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  // Check if ReverseBranchCondition has asked to reverse this branch
  // If we want to reverse the branch an odd number of times, we want
  // J2_jumpf.
  if (!Cond.empty() && Cond[0].isImm())
    BccOpc = Cond[0].getImm();

  if (!FBB) {
    if (Cond.empty()) {
      // Due to a bug in TailMerging/CFG Optimization, we need to add a
      // special case handling of a predicated jump followed by an
      // unconditional jump. If not, Tail Merging and CFG Optimization go
      // into an infinite loop.
      MachineBasicBlock *NewTBB, *NewFBB;
      SmallVector<MachineOperand, 4> Cond;
      MachineInstr *Term = MBB.getFirstTerminator();
      if (Term != MBB.end() && isPredicated(Term) &&
          !AnalyzeBranch(MBB, NewTBB, NewFBB, Cond, false)) {
        MachineBasicBlock *NextBB = &*++MBB.getIterator();
        if (NewTBB == NextBB) {
          ReverseBranchCondition(Cond);
          RemoveBranch(MBB);
          return InsertBranch(MBB, TBB, nullptr, Cond, DL);
        }
      }
      BuildMI(&MBB, DL, get(BOpc)).addMBB(TBB);
    } else if (isEndLoopN(Cond[0].getImm())) {
      int EndLoopOp = Cond[0].getImm();
      assert(Cond[1].isMBB());
      // Since we're adding an ENDLOOP, there better be a LOOP instruction.
      // Check for it, and change the BB target if needed.
      SmallPtrSet<MachineBasicBlock *, 8> VisitedBBs;
      MachineInstr *Loop = findLoopInstr(TBB, EndLoopOp, VisitedBBs);
      assert(Loop != 0 && "Inserting an ENDLOOP without a LOOP");
      Loop->getOperand(0).setMBB(TBB);
      // Add the ENDLOOP after the finding the LOOP0.
      BuildMI(&MBB, DL, get(EndLoopOp)).addMBB(TBB);
    } else if (isNewValueJump(Cond[0].getImm())) {
      assert((Cond.size() == 3) && "Only supporting rr/ri version of nvjump");
      // New value jump
      // (ins IntRegs:$src1, IntRegs:$src2, brtarget:$offset)
      // (ins IntRegs:$src1, u5Imm:$src2, brtarget:$offset)
      unsigned Flags1 = getUndefRegState(Cond[1].isUndef());
      DEBUG(dbgs() << "\nInserting NVJump for BB#" << MBB.getNumber(););
      if (Cond[2].isReg()) {
        unsigned Flags2 = getUndefRegState(Cond[2].isUndef());
        BuildMI(&MBB, DL, get(BccOpc)).addReg(Cond[1].getReg(), Flags1).
          addReg(Cond[2].getReg(), Flags2).addMBB(TBB);
      } else if(Cond[2].isImm()) {
        BuildMI(&MBB, DL, get(BccOpc)).addReg(Cond[1].getReg(), Flags1).
          addImm(Cond[2].getImm()).addMBB(TBB);
      } else
        llvm_unreachable("Invalid condition for branching");
    } else {
      assert((Cond.size() == 2) && "Malformed cond vector");
      const MachineOperand &RO = Cond[1];
      unsigned Flags = getUndefRegState(RO.isUndef());
      BuildMI(&MBB, DL, get(BccOpc)).addReg(RO.getReg(), Flags).addMBB(TBB);
    }
    return 1;
  }
  assert((!Cond.empty()) &&
         "Cond. cannot be empty when multiple branchings are required");
  assert((!isNewValueJump(Cond[0].getImm())) &&
         "NV-jump cannot be inserted with another branch");
  // Special case for hardware loops.  The condition is a basic block.
  if (isEndLoopN(Cond[0].getImm())) {
    int EndLoopOp = Cond[0].getImm();
    assert(Cond[1].isMBB());
    // Since we're adding an ENDLOOP, there better be a LOOP instruction.
    // Check for it, and change the BB target if needed.
    SmallPtrSet<MachineBasicBlock *, 8> VisitedBBs;
    MachineInstr *Loop = findLoopInstr(TBB, EndLoopOp, VisitedBBs);
    assert(Loop != 0 && "Inserting an ENDLOOP without a LOOP");
    Loop->getOperand(0).setMBB(TBB);
    // Add the ENDLOOP after the finding the LOOP0.
    BuildMI(&MBB, DL, get(EndLoopOp)).addMBB(TBB);
  } else {
    const MachineOperand &RO = Cond[1];
    unsigned Flags = getUndefRegState(RO.isUndef());
    BuildMI(&MBB, DL, get(BccOpc)).addReg(RO.getReg(), Flags).addMBB(TBB);
  }
  BuildMI(&MBB, DL, get(BOpc)).addMBB(FBB);

  return 2;
}


/// This function can analyze one/two way branching only and should (mostly) be
/// called by target independent side.
/// First entry is always the opcode of the branching instruction, except when
/// the Cond vector is supposed to be empty, e.g., when AnalyzeBranch fails, a
/// BB with only unconditional jump. Subsequent entries depend upon the opcode,
/// e.g. Jump_c p will have
/// Cond[0] = Jump_c
/// Cond[1] = p
/// HW-loop ENDLOOP:
/// Cond[0] = ENDLOOP
/// Cond[1] = MBB
/// New value jump:
/// Cond[0] = Hexagon::CMPEQri_f_Jumpnv_t_V4 -- specific opcode
/// Cond[1] = R
/// Cond[2] = Imm
/// @note Related function is \fn findInstrPredicate which fills in
/// Cond. vector when a predicated instruction is passed to it.
/// We follow same protocol in that case too.
///
bool HexagonInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *&TBB,
                                     MachineBasicBlock *&FBB,
                                     SmallVectorImpl<MachineOperand> &Cond,
                                     bool AllowModify) const {
  TBB = nullptr;
  FBB = nullptr;
  Cond.clear();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::instr_iterator I = MBB.instr_end();
  if (I == MBB.instr_begin())
    return false;

  // A basic block may looks like this:
  //
  //  [   insn
  //     EH_LABEL
  //      insn
  //      insn
  //      insn
  //     EH_LABEL
  //      insn     ]
  //
  // It has two succs but does not have a terminator
  // Don't know how to handle it.
  do {
    --I;
    if (I->isEHLabel())
      // Don't analyze EH branches.
      return true;
  } while (I != MBB.instr_begin());

  I = MBB.instr_end();
  --I;

  while (I->isDebugValue()) {
    if (I == MBB.instr_begin())
      return false;
    --I;
  }
  
  bool JumpToBlock = I->getOpcode() == Hexagon::J2_jump &&
                     I->getOperand(0).isMBB();
  // Delete the J2_jump if it's equivalent to a fall-through.
  if (AllowModify && JumpToBlock &&
      MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
    DEBUG(dbgs()<< "\nErasing the jump to successor block\n";);
    I->eraseFromParent();
    I = MBB.instr_end();
    if (I == MBB.instr_begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(&*I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = &*I;
  MachineInstr *SecondLastInst = nullptr;
  // Find one more terminator if present.
  for (;;) {
    if (&*I != LastInst && !I->isBundle() && isUnpredicatedTerminator(&*I)) {
      if (!SecondLastInst)
        SecondLastInst = &*I;
      else
        // This is a third branch.
        return true;
    }
    if (I == MBB.instr_begin())
      break;
    --I;
  }

  int LastOpcode = LastInst->getOpcode();
  int SecLastOpcode = SecondLastInst ? SecondLastInst->getOpcode() : 0;
  // If the branch target is not a basic block, it could be a tail call.
  // (It is, if the target is a function.)
  if (LastOpcode == Hexagon::J2_jump && !LastInst->getOperand(0).isMBB())
    return true;
  if (SecLastOpcode == Hexagon::J2_jump &&
      !SecondLastInst->getOperand(0).isMBB())
    return true;

  bool LastOpcodeHasJMP_c = PredOpcodeHasJMP_c(LastOpcode);
  bool LastOpcodeHasNVJump = isNewValueJump(LastInst);

  // If there is only one terminator instruction, process it.
  if (LastInst && !SecondLastInst) {
    if (LastOpcode == Hexagon::J2_jump) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (isEndLoopN(LastOpcode)) {
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    if (LastOpcodeHasJMP_c) {
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    // Only supporting rr/ri versions of new-value jumps.
    if (LastOpcodeHasNVJump && (LastInst->getNumExplicitOperands() == 3)) {
      TBB = LastInst->getOperand(2).getMBB();
      Cond.push_back(MachineOperand::CreateImm(LastInst->getOpcode()));
      Cond.push_back(LastInst->getOperand(0));
      Cond.push_back(LastInst->getOperand(1));
      return false;
    }
    DEBUG(dbgs() << "\nCant analyze BB#" << MBB.getNumber()
                 << " with one jump\n";);
    // Otherwise, don't know what this is.
    return true;
  }

  bool SecLastOpcodeHasJMP_c = PredOpcodeHasJMP_c(SecLastOpcode);
  bool SecLastOpcodeHasNVJump = isNewValueJump(SecondLastInst);
  if (SecLastOpcodeHasJMP_c && (LastOpcode == Hexagon::J2_jump)) {
    TBB =  SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(SecondLastInst->getOpcode()));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // Only supporting rr/ri versions of new-value jumps.
  if (SecLastOpcodeHasNVJump &&
      (SecondLastInst->getNumExplicitOperands() == 3) &&
      (LastOpcode == Hexagon::J2_jump)) {
    TBB = SecondLastInst->getOperand(2).getMBB();
    Cond.push_back(MachineOperand::CreateImm(SecondLastInst->getOpcode()));
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two Hexagon:JMPs, handle it.  The second one is not
  // executed, so remove it.
  if (SecLastOpcode == Hexagon::J2_jump && LastOpcode == Hexagon::J2_jump) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst->getIterator();
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // If the block ends with an ENDLOOP, and J2_jump, handle it.
  if (isEndLoopN(SecLastOpcode) && LastOpcode == Hexagon::J2_jump) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(MachineOperand::CreateImm(SecondLastInst->getOpcode()));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  DEBUG(dbgs() << "\nCant analyze BB#" << MBB.getNumber()
               << " with two jumps";);
  // Otherwise, can't handle this.
  return true;
}

unsigned HexagonInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  DEBUG(dbgs() << "\nRemoving branches out of BB#" << MBB.getNumber());
  MachineBasicBlock::iterator I = MBB.end();
  unsigned Count = 0;
  while (I != MBB.begin()) {
    --I;
    if (I->isDebugValue())
      continue;
    // Only removing branches from end of MBB.
    if (!I->isBranch())
      return Count;
    if (Count && (I->getOpcode() == Hexagon::J2_jump))
      llvm_unreachable("Malformed basic block: unconditional branch not last");
    MBB.erase(&MBB.back());
    I = MBB.end();
    ++Count;
  }
  return Count;
}

/// \brief For a comparison instruction, return the source registers in
/// \p SrcReg and \p SrcReg2 if having two register operands, and the value it
/// compares against in CmpValue. Return true if the comparison instruction
/// can be analyzed.
bool HexagonInstrInfo::analyzeCompare(const MachineInstr *MI,
                                      unsigned &SrcReg, unsigned &SrcReg2,
                                      int &Mask, int &Value) const {
  unsigned Opc = MI->getOpcode();

  // Set mask and the first source register.
  switch (Opc) {
    case Hexagon::C2_cmpeq:
    case Hexagon::C2_cmpeqp:
    case Hexagon::C2_cmpgt:
    case Hexagon::C2_cmpgtp:
    case Hexagon::C2_cmpgtu:
    case Hexagon::C2_cmpgtup:
    case Hexagon::C4_cmpneq:
    case Hexagon::C4_cmplte:
    case Hexagon::C4_cmplteu:
    case Hexagon::C2_cmpeqi:
    case Hexagon::C2_cmpgti:
    case Hexagon::C2_cmpgtui:
    case Hexagon::C4_cmpneqi:
    case Hexagon::C4_cmplteui:
    case Hexagon::C4_cmpltei:
      SrcReg = MI->getOperand(1).getReg();
      Mask = ~0;
      break;
    case Hexagon::A4_cmpbeq:
    case Hexagon::A4_cmpbgt:
    case Hexagon::A4_cmpbgtu:
    case Hexagon::A4_cmpbeqi:
    case Hexagon::A4_cmpbgti:
    case Hexagon::A4_cmpbgtui:
      SrcReg = MI->getOperand(1).getReg();
      Mask = 0xFF;
      break;
    case Hexagon::A4_cmpheq:
    case Hexagon::A4_cmphgt:
    case Hexagon::A4_cmphgtu:
    case Hexagon::A4_cmpheqi:
    case Hexagon::A4_cmphgti:
    case Hexagon::A4_cmphgtui:
      SrcReg = MI->getOperand(1).getReg();
      Mask = 0xFFFF;
      break;
  }

  // Set the value/second source register.
  switch (Opc) {
    case Hexagon::C2_cmpeq:
    case Hexagon::C2_cmpeqp:
    case Hexagon::C2_cmpgt:
    case Hexagon::C2_cmpgtp:
    case Hexagon::C2_cmpgtu:
    case Hexagon::C2_cmpgtup:
    case Hexagon::A4_cmpbeq:
    case Hexagon::A4_cmpbgt:
    case Hexagon::A4_cmpbgtu:
    case Hexagon::A4_cmpheq:
    case Hexagon::A4_cmphgt:
    case Hexagon::A4_cmphgtu:
    case Hexagon::C4_cmpneq:
    case Hexagon::C4_cmplte:
    case Hexagon::C4_cmplteu:
      SrcReg2 = MI->getOperand(2).getReg();
      return true;

    case Hexagon::C2_cmpeqi:
    case Hexagon::C2_cmpgtui:
    case Hexagon::C2_cmpgti:
    case Hexagon::C4_cmpneqi:
    case Hexagon::C4_cmplteui:
    case Hexagon::C4_cmpltei:
    case Hexagon::A4_cmpbeqi:
    case Hexagon::A4_cmpbgti:
    case Hexagon::A4_cmpbgtui:
    case Hexagon::A4_cmpheqi:
    case Hexagon::A4_cmphgti:
    case Hexagon::A4_cmphgtui:
      SrcReg2 = 0;
      Value = MI->getOperand(2).getImm();
      return true;
  }

  return false;
}


void HexagonInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I, DebugLoc DL,
                                 unsigned DestReg, unsigned SrcReg,
                                 bool KillSrc) const {
  if (Hexagon::IntRegsRegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::A2_tfr), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::DoubleRegsRegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::A2_tfrp), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::PredRegsRegClass.contains(SrcReg, DestReg)) {
    // Map Pd = Ps to Pd = or(Ps, Ps).
    BuildMI(MBB, I, DL, get(Hexagon::C2_or),
            DestReg).addReg(SrcReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::DoubleRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg)) {
    // We can have an overlap between single and double reg: r1:0 = r0.
    if(SrcReg == RI.getSubReg(DestReg, Hexagon::subreg_loreg)) {
        // r1:0 = r0
        BuildMI(MBB, I, DL, get(Hexagon::A2_tfrsi), (RI.getSubReg(DestReg,
                Hexagon::subreg_hireg))).addImm(0);
    } else {
        // r1:0 = r1 or no overlap.
        BuildMI(MBB, I, DL, get(Hexagon::A2_tfr), (RI.getSubReg(DestReg,
                Hexagon::subreg_loreg))).addReg(SrcReg);
        BuildMI(MBB, I, DL, get(Hexagon::A2_tfrsi), (RI.getSubReg(DestReg,
                Hexagon::subreg_hireg))).addImm(0);
    }
    return;
  }
  if (Hexagon::CtrRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::A2_tfrrcr), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::PredRegsRegClass.contains(SrcReg) &&
      Hexagon::IntRegsRegClass.contains(DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::C2_tfrpr), DestReg).
      addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (Hexagon::IntRegsRegClass.contains(SrcReg) &&
      Hexagon::PredRegsRegClass.contains(DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::C2_tfrrp), DestReg).
      addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  llvm_unreachable("Unimplemented");
}


void HexagonInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {

  DebugLoc DL = MBB.findDebugLoc(I);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOStore,
      MFI.getObjectSize(FI), Align);

  if (Hexagon::IntRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::S2_storeri_io))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else if (Hexagon::DoubleRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::S2_storerd_io))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else if (Hexagon::PredRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STriw_pred))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else {
    llvm_unreachable("Unimplemented");
  }
}


void HexagonInstrInfo::storeRegToAddr(
                                 MachineFunction &MF, unsigned SrcReg,
                                 bool isKill,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const
{
  llvm_unreachable("Unimplemented");
}


void HexagonInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBB.findDebugLoc(I);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FI), MachineMemOperand::MOLoad,
      MFI.getObjectSize(FI), Align);
  if (RC == &Hexagon::IntRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::L2_loadri_io), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else if (RC == &Hexagon::DoubleRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::L2_loadrd_io), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else if (RC == &Hexagon::PredRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDriw_pred), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else {
    llvm_unreachable("Can't store this register to stack slot");
  }
}


void HexagonInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                        SmallVectorImpl<MachineOperand> &Addr,
                                        const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  llvm_unreachable("Unimplemented");
}
bool
HexagonInstrInfo::expandPostRAPseudo(MachineBasicBlock::iterator MI) const {
  const HexagonRegisterInfo &HRI = getRegisterInfo();
  MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();
  MachineBasicBlock &MBB = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  unsigned Opc = MI->getOpcode();

  switch (Opc) {
    case Hexagon::ALIGNA:
      BuildMI(MBB, MI, DL, get(Hexagon::A2_andir), MI->getOperand(0).getReg())
          .addReg(HRI.getFrameRegister())
          .addImm(-MI->getOperand(1).getImm());
      MBB.erase(MI);
      return true;
    case Hexagon::TFR_PdTrue: {
      unsigned Reg = MI->getOperand(0).getReg();
      BuildMI(MBB, MI, DL, get(Hexagon::C2_orn), Reg)
        .addReg(Reg, RegState::Undef)
        .addReg(Reg, RegState::Undef);
      MBB.erase(MI);
      return true;
    }
    case Hexagon::TFR_PdFalse: {
      unsigned Reg = MI->getOperand(0).getReg();
      BuildMI(MBB, MI, DL, get(Hexagon::C2_andn), Reg)
        .addReg(Reg, RegState::Undef)
        .addReg(Reg, RegState::Undef);
      MBB.erase(MI);
      return true;
    }
    case Hexagon::VMULW: {
      // Expand a 64-bit vector multiply into 2 32-bit scalar multiplies.
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned Src1Reg = MI->getOperand(1).getReg();
      unsigned Src2Reg = MI->getOperand(2).getReg();
      unsigned Src1SubHi = HRI.getSubReg(Src1Reg, Hexagon::subreg_hireg);
      unsigned Src1SubLo = HRI.getSubReg(Src1Reg, Hexagon::subreg_loreg);
      unsigned Src2SubHi = HRI.getSubReg(Src2Reg, Hexagon::subreg_hireg);
      unsigned Src2SubLo = HRI.getSubReg(Src2Reg, Hexagon::subreg_loreg);
      BuildMI(MBB, MI, MI->getDebugLoc(), get(Hexagon::M2_mpyi),
              HRI.getSubReg(DstReg, Hexagon::subreg_hireg)).addReg(Src1SubHi)
          .addReg(Src2SubHi);
      BuildMI(MBB, MI, MI->getDebugLoc(), get(Hexagon::M2_mpyi),
              HRI.getSubReg(DstReg, Hexagon::subreg_loreg)).addReg(Src1SubLo)
          .addReg(Src2SubLo);
      MBB.erase(MI);
      MRI.clearKillFlags(Src1SubHi);
      MRI.clearKillFlags(Src1SubLo);
      MRI.clearKillFlags(Src2SubHi);
      MRI.clearKillFlags(Src2SubLo);
      return true;
    }
    case Hexagon::VMULW_ACC: {
      // Expand 64-bit vector multiply with addition into 2 scalar multiplies.
      unsigned DstReg = MI->getOperand(0).getReg();
      unsigned Src1Reg = MI->getOperand(1).getReg();
      unsigned Src2Reg = MI->getOperand(2).getReg();
      unsigned Src3Reg = MI->getOperand(3).getReg();
      unsigned Src1SubHi = HRI.getSubReg(Src1Reg, Hexagon::subreg_hireg);
      unsigned Src1SubLo = HRI.getSubReg(Src1Reg, Hexagon::subreg_loreg);
      unsigned Src2SubHi = HRI.getSubReg(Src2Reg, Hexagon::subreg_hireg);
      unsigned Src2SubLo = HRI.getSubReg(Src2Reg, Hexagon::subreg_loreg);
      unsigned Src3SubHi = HRI.getSubReg(Src3Reg, Hexagon::subreg_hireg);
      unsigned Src3SubLo = HRI.getSubReg(Src3Reg, Hexagon::subreg_loreg);
      BuildMI(MBB, MI, MI->getDebugLoc(), get(Hexagon::M2_maci),
              HRI.getSubReg(DstReg, Hexagon::subreg_hireg)).addReg(Src1SubHi)
          .addReg(Src2SubHi).addReg(Src3SubHi);
      BuildMI(MBB, MI, MI->getDebugLoc(), get(Hexagon::M2_maci),
              HRI.getSubReg(DstReg, Hexagon::subreg_loreg)).addReg(Src1SubLo)
          .addReg(Src2SubLo).addReg(Src3SubLo);
      MBB.erase(MI);
      MRI.clearKillFlags(Src1SubHi);
      MRI.clearKillFlags(Src1SubLo);
      MRI.clearKillFlags(Src2SubHi);
      MRI.clearKillFlags(Src2SubLo);
      MRI.clearKillFlags(Src3SubHi);
      MRI.clearKillFlags(Src3SubLo);
      return true;
    }
    case Hexagon::MUX64_rr: {
      const MachineOperand &Op0 = MI->getOperand(0);
      const MachineOperand &Op1 = MI->getOperand(1);
      const MachineOperand &Op2 = MI->getOperand(2);
      const MachineOperand &Op3 = MI->getOperand(3);
      unsigned Rd = Op0.getReg();
      unsigned Pu = Op1.getReg();
      unsigned Rs = Op2.getReg();
      unsigned Rt = Op3.getReg();
      DebugLoc DL = MI->getDebugLoc();
      unsigned K1 = getKillRegState(Op1.isKill());
      unsigned K2 = getKillRegState(Op2.isKill());
      unsigned K3 = getKillRegState(Op3.isKill());
      if (Rd != Rs)
        BuildMI(MBB, MI, DL, get(Hexagon::A2_tfrpt), Rd)
          .addReg(Pu, (Rd == Rt) ? K1 : 0)
          .addReg(Rs, K2);
      if (Rd != Rt)
        BuildMI(MBB, MI, DL, get(Hexagon::A2_tfrpf), Rd)
          .addReg(Pu, K1)
          .addReg(Rt, K3);
      MBB.erase(MI);
      return true;
    }
    case Hexagon::TCRETURNi:
      MI->setDesc(get(Hexagon::J2_jump));
      return true;
    case Hexagon::TCRETURNr:
      MI->setDesc(get(Hexagon::J2_jumpr));
      return true;
  }

  return false;
}

MachineInstr *HexagonInstrInfo::foldMemoryOperandImpl(
    MachineFunction &MF, MachineInstr *MI, ArrayRef<unsigned> Ops,
    MachineBasicBlock::iterator InsertPt, int FI) const {
  // Hexagon_TODO: Implement.
  return nullptr;
}

unsigned HexagonInstrInfo::createVR(MachineFunction* MF, MVT VT) const {

  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *TRC;
  if (VT == MVT::i1) {
    TRC = &Hexagon::PredRegsRegClass;
  } else if (VT == MVT::i32 || VT == MVT::f32) {
    TRC = &Hexagon::IntRegsRegClass;
  } else if (VT == MVT::i64 || VT == MVT::f64) {
    TRC = &Hexagon::DoubleRegsRegClass;
  } else {
    llvm_unreachable("Cannot handle this register class");
  }

  unsigned NewReg = RegInfo.createVirtualRegister(TRC);
  return NewReg;
}

bool HexagonInstrInfo::isExtendable(const MachineInstr *MI) const {
  const MCInstrDesc &MID = MI->getDesc();
  const uint64_t F = MID.TSFlags;
  if ((F >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask)
    return true;

  // TODO: This is largely obsolete now. Will need to be removed
  // in consecutive patches.
  switch(MI->getOpcode()) {
    // TFR_FI Remains a special case.
    case Hexagon::TFR_FI:
      return true;
    default:
      return false;
  }
  return  false;
}

// This returns true in two cases:
// - The OP code itself indicates that this is an extended instruction.
// - One of MOs has been marked with HMOTF_ConstExtended flag.
bool HexagonInstrInfo::isExtended(const MachineInstr *MI) const {
  // First check if this is permanently extended op code.
  const uint64_t F = MI->getDesc().TSFlags;
  if ((F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask)
    return true;
  // Use MO operand flags to determine if one of MI's operands
  // has HMOTF_ConstExtended flag set.
  for (MachineInstr::const_mop_iterator I = MI->operands_begin(),
       E = MI->operands_end(); I != E; ++I) {
    if (I->getTargetFlags() && HexagonII::HMOTF_ConstExtended)
      return true;
  }
  return  false;
}

bool HexagonInstrInfo::isBranch (const MachineInstr *MI) const {
  return MI->getDesc().isBranch();
}

bool HexagonInstrInfo::isNewValueInst(const MachineInstr *MI) const {
  if (isNewValueJump(MI))
    return true;

  if (isNewValueStore(MI))
    return true;

  return false;
}

bool HexagonInstrInfo::isNewValue(const MachineInstr* MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  return ((F >> HexagonII::NewValuePos) & HexagonII::NewValueMask);
}

bool HexagonInstrInfo::isNewValue(Opcode_t Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;
  return ((F >> HexagonII::NewValuePos) & HexagonII::NewValueMask);
}

bool HexagonInstrInfo::isSaveCalleeSavedRegsCall(const MachineInstr *MI) const {
  return MI->getOpcode() == Hexagon::SAVE_REGISTERS_CALL_V4;
}

bool HexagonInstrInfo::isPredicable(MachineInstr *MI) const {
  bool isPred = MI->getDesc().isPredicable();

  if (!isPred)
    return false;

  const int Opc = MI->getOpcode();

  switch(Opc) {
  case Hexagon::A2_tfrsi:
    return (isOperandExtended(MI, 1) && isConstExtended(MI)) || isInt<12>(MI->getOperand(1).getImm());

  case Hexagon::S2_storerd_io:
    return isShiftedUInt<6,3>(MI->getOperand(1).getImm());

  case Hexagon::S2_storeri_io:
  case Hexagon::S2_storerinew_io:
    return isShiftedUInt<6,2>(MI->getOperand(1).getImm());

  case Hexagon::S2_storerh_io:
  case Hexagon::S2_storerhnew_io:
    return isShiftedUInt<6,1>(MI->getOperand(1).getImm());

  case Hexagon::S2_storerb_io:
  case Hexagon::S2_storerbnew_io:
    return isUInt<6>(MI->getOperand(1).getImm());

  case Hexagon::L2_loadrd_io:
    return isShiftedUInt<6,3>(MI->getOperand(2).getImm());

  case Hexagon::L2_loadri_io:
    return isShiftedUInt<6,2>(MI->getOperand(2).getImm());

  case Hexagon::L2_loadrh_io:
  case Hexagon::L2_loadruh_io:
    return isShiftedUInt<6,1>(MI->getOperand(2).getImm());

  case Hexagon::L2_loadrb_io:
  case Hexagon::L2_loadrub_io:
    return isUInt<6>(MI->getOperand(2).getImm());

  case Hexagon::L2_loadrd_pi:
    return isShiftedInt<4,3>(MI->getOperand(3).getImm());

  case Hexagon::L2_loadri_pi:
    return isShiftedInt<4,2>(MI->getOperand(3).getImm());

  case Hexagon::L2_loadrh_pi:
  case Hexagon::L2_loadruh_pi:
    return isShiftedInt<4,1>(MI->getOperand(3).getImm());

  case Hexagon::L2_loadrb_pi:
  case Hexagon::L2_loadrub_pi:
    return isInt<4>(MI->getOperand(3).getImm());

  case Hexagon::S4_storeirb_io:
  case Hexagon::S4_storeirh_io:
  case Hexagon::S4_storeiri_io:
    return (isUInt<6>(MI->getOperand(1).getImm()) &&
            isInt<6>(MI->getOperand(2).getImm()));

  case Hexagon::A2_addi:
    return isInt<8>(MI->getOperand(2).getImm());

  case Hexagon::A2_aslh:
  case Hexagon::A2_asrh:
  case Hexagon::A2_sxtb:
  case Hexagon::A2_sxth:
  case Hexagon::A2_zxtb:
  case Hexagon::A2_zxth:
    return true;
  }

  return true;
}

// This function performs the following inversiones:
//
//  cPt    ---> cNotPt
//  cNotPt ---> cPt
//
unsigned HexagonInstrInfo::getInvertedPredicatedOpcode(const int Opc) const {
  int InvPredOpcode;
  InvPredOpcode = isPredicatedTrue(Opc) ? Hexagon::getFalsePredOpcode(Opc)
                                        : Hexagon::getTruePredOpcode(Opc);
  if (InvPredOpcode >= 0) // Valid instruction with the inverted predicate.
    return InvPredOpcode;

  switch(Opc) {
    default: llvm_unreachable("Unexpected predicated instruction");
    case Hexagon::C2_ccombinewt:
      return Hexagon::C2_ccombinewf;
    case Hexagon::C2_ccombinewf:
      return Hexagon::C2_ccombinewt;

      // Dealloc_return.
    case Hexagon::L4_return_t:
      return Hexagon::L4_return_f;
    case Hexagon::L4_return_f:
      return Hexagon::L4_return_t;
  }
}

// New Value Store instructions.
bool HexagonInstrInfo::isNewValueStore(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::NVStorePos) & HexagonII::NVStoreMask);
}

bool HexagonInstrInfo::isNewValueStore(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  return ((F >> HexagonII::NVStorePos) & HexagonII::NVStoreMask);
}

int HexagonInstrInfo::getCondOpcode(int Opc, bool invertPredicate) const {
  enum Hexagon::PredSense inPredSense;
  inPredSense = invertPredicate ? Hexagon::PredSense_false :
                                  Hexagon::PredSense_true;
  int CondOpcode = Hexagon::getPredOpcode(Opc, inPredSense);
  if (CondOpcode >= 0) // Valid Conditional opcode/instruction
    return CondOpcode;

  // This switch case will be removed once all the instructions have been
  // modified to use relation maps.
  switch(Opc) {
  case Hexagon::TFRI_f:
    return !invertPredicate ? Hexagon::TFRI_cPt_f :
                              Hexagon::TFRI_cNotPt_f;
  case Hexagon::A2_combinew:
    return !invertPredicate ? Hexagon::C2_ccombinewt :
                              Hexagon::C2_ccombinewf;

  // DEALLOC_RETURN.
  case Hexagon::L4_return:
    return !invertPredicate ? Hexagon::L4_return_t:
                              Hexagon::L4_return_f;
  }
  llvm_unreachable("Unexpected predicable instruction");
}


bool HexagonInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     ArrayRef<MachineOperand> Cond) const {
  if (Cond.empty() || isEndLoopN(Cond[0].getImm())) {
    DEBUG(dbgs() << "\nCannot predicate:"; MI->dump(););
    return false;
  }
  int Opc = MI->getOpcode();
  assert (isPredicable(MI) && "Expected predicable instruction");
  bool invertJump = predOpcodeHasNot(Cond);

  // We have to predicate MI "in place", i.e. after this function returns,
  // MI will need to be transformed into a predicated form. To avoid com-
  // plicated manipulations with the operands (handling tied operands,
  // etc.), build a new temporary instruction, then overwrite MI with it.

  MachineBasicBlock &B = *MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  unsigned PredOpc = getCondOpcode(Opc, invertJump);
  MachineInstrBuilder T = BuildMI(B, MI, DL, get(PredOpc));
  unsigned NOp = 0, NumOps = MI->getNumOperands();
  while (NOp < NumOps) {
    MachineOperand &Op = MI->getOperand(NOp);
    if (!Op.isReg() || !Op.isDef() || Op.isImplicit())
      break;
    T.addOperand(Op);
    NOp++;
  }

  unsigned PredReg, PredRegPos, PredRegFlags;
  bool GotPredReg = getPredReg(Cond, PredReg, PredRegPos, PredRegFlags);
  (void)GotPredReg;
  assert(GotPredReg);
  T.addReg(PredReg, PredRegFlags);
  while (NOp < NumOps)
    T.addOperand(MI->getOperand(NOp++));

  MI->setDesc(get(PredOpc));
  while (unsigned n = MI->getNumOperands())
    MI->RemoveOperand(n-1);
  for (unsigned i = 0, n = T->getNumOperands(); i < n; ++i)
    MI->addOperand(T->getOperand(i));

  MachineBasicBlock::instr_iterator TI = T->getIterator();
  B.erase(TI);

  MachineRegisterInfo &MRI = B.getParent()->getRegInfo();
  MRI.clearKillFlags(PredReg);

  return true;
}


bool
HexagonInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &MBB,
                    unsigned NumCycles,
                    unsigned ExtraPredCycles,
                    BranchProbability Probability) const {
  return true;
}


bool
HexagonInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &TMBB,
                    unsigned NumTCycles,
                    unsigned ExtraTCycles,
                    MachineBasicBlock &FMBB,
                    unsigned NumFCycles,
                    unsigned ExtraFCycles,
                    BranchProbability Probability) const {
  return true;
}

// Returns true if an instruction is predicated irrespective of the predicate
// sense. For example, all of the following will return true.
// if (p0) R1 = add(R2, R3)
// if (!p0) R1 = add(R2, R3)
// if (p0.new) R1 = add(R2, R3)
// if (!p0.new) R1 = add(R2, R3)
bool HexagonInstrInfo::isPredicated(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool HexagonInstrInfo::isPredicated(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool HexagonInstrInfo::isPredicatedTrue(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  assert(isPredicated(MI));
  return (!((F >> HexagonII::PredicatedFalsePos) &
            HexagonII::PredicatedFalseMask));
}

bool HexagonInstrInfo::isPredicatedTrue(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  // Make sure that the instruction is predicated.
  assert((F>> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
  return (!((F >> HexagonII::PredicatedFalsePos) &
            HexagonII::PredicatedFalseMask));
}

bool HexagonInstrInfo::isPredicatedNew(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  assert(isPredicated(MI));
  return ((F >> HexagonII::PredicatedNewPos) & HexagonII::PredicatedNewMask);
}

bool HexagonInstrInfo::isPredicatedNew(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  assert(isPredicated(Opcode));
  return ((F >> HexagonII::PredicatedNewPos) & HexagonII::PredicatedNewMask);
}

// Returns true, if a ST insn can be promoted to a new-value store.
bool HexagonInstrInfo::mayBeNewStore(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::mayNVStorePos) &
           HexagonII::mayNVStoreMask);
}

bool
HexagonInstrInfo::DefinesPredicate(MachineInstr *MI,
                                   std::vector<MachineOperand> &Pred) const {
  for (unsigned oper = 0; oper < MI->getNumOperands(); ++oper) {
    MachineOperand MO = MI->getOperand(oper);
    if (MO.isReg() && MO.isDef()) {
      const TargetRegisterClass* RC = RI.getMinimalPhysRegClass(MO.getReg());
      if (RC == &Hexagon::PredRegsRegClass) {
        Pred.push_back(MO);
        return true;
      }
    }
  }
  return false;
}


bool
HexagonInstrInfo::
SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                  ArrayRef<MachineOperand> Pred2) const {
  // TODO: Fix this
  return false;
}


//
// We indicate that we want to reverse the branch by
// inserting the reversed branching opcode.
//
bool HexagonInstrInfo::ReverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const {
  if (Cond.empty())
    return true;
  assert(Cond[0].isImm() && "First entry in the cond vector not imm-val");
  Opcode_t opcode = Cond[0].getImm();
  //unsigned temp;
  assert(get(opcode).isBranch() && "Should be a branching condition.");
  if (isEndLoopN(opcode))
    return true;
  Opcode_t NewOpcode = getInvertedPredicatedOpcode(opcode);
  Cond[0].setImm(NewOpcode);
  return false;
}


bool HexagonInstrInfo::
isProfitableToDupForIfCvt(MachineBasicBlock &MBB,unsigned NumInstrs,
                          BranchProbability Probability) const {
  return (NumInstrs <= 4);
}

bool HexagonInstrInfo::isDeallocRet(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default: return false;
  case Hexagon::L4_return:
  case Hexagon::L4_return_t:
  case Hexagon::L4_return_f:
  case Hexagon::L4_return_tnew_pnt:
  case Hexagon::L4_return_fnew_pnt:
  case Hexagon::L4_return_tnew_pt:
  case Hexagon::L4_return_fnew_pt:
   return true;
  }
}


bool HexagonInstrInfo::isValidOffset(unsigned Opcode, int Offset,
      bool Extend) const {
  // This function is to check whether the "Offset" is in the correct range of
  // the given "Opcode". If "Offset" is not in the correct range, "A2_addi" is
  // inserted to calculate the final address. Due to this reason, the function
  // assumes that the "Offset" has correct alignment.
  // We used to assert if the offset was not properly aligned, however,
  // there are cases where a misaligned pointer recast can cause this
  // problem, and we need to allow for it. The front end warns of such
  // misaligns with respect to load size.

  switch (Opcode) {
  case Hexagon::J2_loop0i:
  case Hexagon::J2_loop1i:
    return isUInt<10>(Offset);
  }

  if (Extend)
    return true;

  switch (Opcode) {
  case Hexagon::L2_loadri_io:
  case Hexagon::S2_storeri_io:
    return (Offset >= Hexagon_MEMW_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMW_OFFSET_MAX);

  case Hexagon::L2_loadrd_io:
  case Hexagon::S2_storerd_io:
    return (Offset >= Hexagon_MEMD_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMD_OFFSET_MAX);

  case Hexagon::L2_loadrh_io:
  case Hexagon::L2_loadruh_io:
  case Hexagon::S2_storerh_io:
    return (Offset >= Hexagon_MEMH_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMH_OFFSET_MAX);

  case Hexagon::L2_loadrb_io:
  case Hexagon::S2_storerb_io:
  case Hexagon::L2_loadrub_io:
    return (Offset >= Hexagon_MEMB_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMB_OFFSET_MAX);

  case Hexagon::A2_addi:
    return (Offset >= Hexagon_ADDI_OFFSET_MIN) &&
      (Offset <= Hexagon_ADDI_OFFSET_MAX);

  case Hexagon::L4_iadd_memopw_io:
  case Hexagon::L4_isub_memopw_io:
  case Hexagon::L4_add_memopw_io:
  case Hexagon::L4_sub_memopw_io:
  case Hexagon::L4_and_memopw_io:
  case Hexagon::L4_or_memopw_io:
    return (0 <= Offset && Offset <= 255);

  case Hexagon::L4_iadd_memoph_io:
  case Hexagon::L4_isub_memoph_io:
  case Hexagon::L4_add_memoph_io:
  case Hexagon::L4_sub_memoph_io:
  case Hexagon::L4_and_memoph_io:
  case Hexagon::L4_or_memoph_io:
    return (0 <= Offset && Offset <= 127);

  case Hexagon::L4_iadd_memopb_io:
  case Hexagon::L4_isub_memopb_io:
  case Hexagon::L4_add_memopb_io:
  case Hexagon::L4_sub_memopb_io:
  case Hexagon::L4_and_memopb_io:
  case Hexagon::L4_or_memopb_io:
    return (0 <= Offset && Offset <= 63);

  // LDri_pred and STriw_pred are pseudo operations, so it has to take offset of
  // any size. Later pass knows how to handle it.
  case Hexagon::STriw_pred:
  case Hexagon::LDriw_pred:
    return true;

  case Hexagon::TFR_FI:
  case Hexagon::TFR_FIA:
  case Hexagon::INLINEASM:
    return true;

  case Hexagon::L2_ploadrbt_io:
  case Hexagon::L2_ploadrbf_io:
  case Hexagon::L2_ploadrubt_io:
  case Hexagon::L2_ploadrubf_io:
  case Hexagon::S2_pstorerbt_io:
  case Hexagon::S2_pstorerbf_io:
  case Hexagon::S4_storeirb_io:
  case Hexagon::S4_storeirbt_io:
  case Hexagon::S4_storeirbf_io:
    return isUInt<6>(Offset);

  case Hexagon::L2_ploadrht_io:
  case Hexagon::L2_ploadrhf_io:
  case Hexagon::L2_ploadruht_io:
  case Hexagon::L2_ploadruhf_io:
  case Hexagon::S2_pstorerht_io:
  case Hexagon::S2_pstorerhf_io:
  case Hexagon::S4_storeirh_io:
  case Hexagon::S4_storeirht_io:
  case Hexagon::S4_storeirhf_io:
    return isShiftedUInt<6,1>(Offset);

  case Hexagon::L2_ploadrit_io:
  case Hexagon::L2_ploadrif_io:
  case Hexagon::S2_pstorerit_io:
  case Hexagon::S2_pstorerif_io:
  case Hexagon::S4_storeiri_io:
  case Hexagon::S4_storeirit_io:
  case Hexagon::S4_storeirif_io:
    return isShiftedUInt<6,2>(Offset);

  case Hexagon::L2_ploadrdt_io:
  case Hexagon::L2_ploadrdf_io:
  case Hexagon::S2_pstorerdt_io:
  case Hexagon::S2_pstorerdf_io:
    return isShiftedUInt<6,3>(Offset);
  } // switch

  llvm_unreachable("No offset range is defined for this opcode. "
                   "Please define it in the above switch statement!");
}


//
// Check if the Offset is a valid auto-inc imm by Load/Store Type.
//
bool HexagonInstrInfo::
isValidAutoIncImm(const EVT VT, const int Offset) const {

  if (VT == MVT::i64) {
      return (Offset >= Hexagon_MEMD_AUTOINC_MIN &&
              Offset <= Hexagon_MEMD_AUTOINC_MAX &&
              (Offset & 0x7) == 0);
  }
  if (VT == MVT::i32) {
      return (Offset >= Hexagon_MEMW_AUTOINC_MIN &&
              Offset <= Hexagon_MEMW_AUTOINC_MAX &&
              (Offset & 0x3) == 0);
  }
  if (VT == MVT::i16) {
      return (Offset >= Hexagon_MEMH_AUTOINC_MIN &&
              Offset <= Hexagon_MEMH_AUTOINC_MAX &&
              (Offset & 0x1) == 0);
  }
  if (VT == MVT::i8) {
      return (Offset >= Hexagon_MEMB_AUTOINC_MIN &&
              Offset <= Hexagon_MEMB_AUTOINC_MAX);
  }
  llvm_unreachable("Not an auto-inc opc!");
}


bool HexagonInstrInfo::
isMemOp(const MachineInstr *MI) const {
//  return MI->getDesc().mayLoad() && MI->getDesc().mayStore();

  switch (MI->getOpcode())
  {
  default: return false;
  case Hexagon::L4_iadd_memopw_io:
  case Hexagon::L4_isub_memopw_io:
  case Hexagon::L4_add_memopw_io:
  case Hexagon::L4_sub_memopw_io:
  case Hexagon::L4_and_memopw_io:
  case Hexagon::L4_or_memopw_io:
  case Hexagon::L4_iadd_memoph_io:
  case Hexagon::L4_isub_memoph_io:
  case Hexagon::L4_add_memoph_io:
  case Hexagon::L4_sub_memoph_io:
  case Hexagon::L4_and_memoph_io:
  case Hexagon::L4_or_memoph_io:
  case Hexagon::L4_iadd_memopb_io:
  case Hexagon::L4_isub_memopb_io:
  case Hexagon::L4_add_memopb_io:
  case Hexagon::L4_sub_memopb_io:
  case Hexagon::L4_and_memopb_io:
  case Hexagon::L4_or_memopb_io:
  case Hexagon::L4_ior_memopb_io:
  case Hexagon::L4_ior_memoph_io:
  case Hexagon::L4_ior_memopw_io:
  case Hexagon::L4_iand_memopb_io:
  case Hexagon::L4_iand_memoph_io:
  case Hexagon::L4_iand_memopw_io:
    return true;
  }
  return false;
}


bool HexagonInstrInfo::
isSpillPredRegOp(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::STriw_pred :
    case Hexagon::LDriw_pred :
      return true;
  }
}

bool HexagonInstrInfo::isNewValueJumpCandidate(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::C2_cmpeq:
    case Hexagon::C2_cmpeqi:
    case Hexagon::C2_cmpgt:
    case Hexagon::C2_cmpgti:
    case Hexagon::C2_cmpgtu:
    case Hexagon::C2_cmpgtui:
      return true;
  }
}

bool HexagonInstrInfo::
isConditionalTransfer (const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::A2_tfrt:
    case Hexagon::A2_tfrf:
    case Hexagon::C2_cmoveit:
    case Hexagon::C2_cmoveif:
    case Hexagon::A2_tfrtnew:
    case Hexagon::A2_tfrfnew:
    case Hexagon::C2_cmovenewit:
    case Hexagon::C2_cmovenewif:
      return true;
  }
}

bool HexagonInstrInfo::isConditionalALU32 (const MachineInstr* MI) const {
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::A2_paddf:
    case Hexagon::A2_paddfnew:
    case Hexagon::A2_paddt:
    case Hexagon::A2_paddtnew:
    case Hexagon::A2_pandf:
    case Hexagon::A2_pandfnew:
    case Hexagon::A2_pandt:
    case Hexagon::A2_pandtnew:
    case Hexagon::A4_paslhf:
    case Hexagon::A4_paslhfnew:
    case Hexagon::A4_paslht:
    case Hexagon::A4_paslhtnew:
    case Hexagon::A4_pasrhf:
    case Hexagon::A4_pasrhfnew:
    case Hexagon::A4_pasrht:
    case Hexagon::A4_pasrhtnew:
    case Hexagon::A2_porf:
    case Hexagon::A2_porfnew:
    case Hexagon::A2_port:
    case Hexagon::A2_portnew:
    case Hexagon::A2_psubf:
    case Hexagon::A2_psubfnew:
    case Hexagon::A2_psubt:
    case Hexagon::A2_psubtnew:
    case Hexagon::A2_pxorf:
    case Hexagon::A2_pxorfnew:
    case Hexagon::A2_pxort:
    case Hexagon::A2_pxortnew:
    case Hexagon::A4_psxthf:
    case Hexagon::A4_psxthfnew:
    case Hexagon::A4_psxtht:
    case Hexagon::A4_psxthtnew:
    case Hexagon::A4_psxtbf:
    case Hexagon::A4_psxtbfnew:
    case Hexagon::A4_psxtbt:
    case Hexagon::A4_psxtbtnew:
    case Hexagon::A4_pzxtbf:
    case Hexagon::A4_pzxtbfnew:
    case Hexagon::A4_pzxtbt:
    case Hexagon::A4_pzxtbtnew:
    case Hexagon::A4_pzxthf:
    case Hexagon::A4_pzxthfnew:
    case Hexagon::A4_pzxtht:
    case Hexagon::A4_pzxthtnew:
    case Hexagon::A2_paddit:
    case Hexagon::A2_paddif:
    case Hexagon::C2_ccombinewt:
    case Hexagon::C2_ccombinewf:
      return true;
  }
}

bool HexagonInstrInfo::
isConditionalLoad (const MachineInstr* MI) const {
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::L2_ploadrdt_io :
    case Hexagon::L2_ploadrdf_io:
    case Hexagon::L2_ploadrit_io:
    case Hexagon::L2_ploadrif_io:
    case Hexagon::L2_ploadrht_io:
    case Hexagon::L2_ploadrhf_io:
    case Hexagon::L2_ploadrbt_io:
    case Hexagon::L2_ploadrbf_io:
    case Hexagon::L2_ploadruht_io:
    case Hexagon::L2_ploadruhf_io:
    case Hexagon::L2_ploadrubt_io:
    case Hexagon::L2_ploadrubf_io:
    case Hexagon::L2_ploadrdt_pi:
    case Hexagon::L2_ploadrdf_pi:
    case Hexagon::L2_ploadrit_pi:
    case Hexagon::L2_ploadrif_pi:
    case Hexagon::L2_ploadrht_pi:
    case Hexagon::L2_ploadrhf_pi:
    case Hexagon::L2_ploadrbt_pi:
    case Hexagon::L2_ploadrbf_pi:
    case Hexagon::L2_ploadruht_pi:
    case Hexagon::L2_ploadruhf_pi:
    case Hexagon::L2_ploadrubt_pi:
    case Hexagon::L2_ploadrubf_pi:
    case Hexagon::L4_ploadrdt_rr:
    case Hexagon::L4_ploadrdf_rr:
    case Hexagon::L4_ploadrbt_rr:
    case Hexagon::L4_ploadrbf_rr:
    case Hexagon::L4_ploadrubt_rr:
    case Hexagon::L4_ploadrubf_rr:
    case Hexagon::L4_ploadrht_rr:
    case Hexagon::L4_ploadrhf_rr:
    case Hexagon::L4_ploadruht_rr:
    case Hexagon::L4_ploadruhf_rr:
    case Hexagon::L4_ploadrit_rr:
    case Hexagon::L4_ploadrif_rr:
      return true;
  }
}

// Returns true if an instruction is a conditional store.
//
// Note: It doesn't include conditional new-value stores as they can't be
// converted to .new predicate.
//
//               p.new NV store [ if(p0.new)memw(R0+#0)=R2.new ]
//                ^           ^
//               /             \ (not OK. it will cause new-value store to be
//              /               X conditional on p0.new while R2 producer is
//             /                 \ on p0)
//            /                   \.
//     p.new store                 p.old NV store
// [if(p0.new)memw(R0+#0)=R2]    [if(p0)memw(R0+#0)=R2.new]
//            ^                  ^
//             \                /
//              \              /
//               \            /
//                 p.old store
//             [if (p0)memw(R0+#0)=R2]
//
// The above diagram shows the steps involoved in the conversion of a predicated
// store instruction to its .new predicated new-value form.
//
// The following set of instructions further explains the scenario where
// conditional new-value store becomes invalid when promoted to .new predicate
// form.
//
// { 1) if (p0) r0 = add(r1, r2)
//   2) p0 = cmp.eq(r3, #0) }
//
//   3) if (p0) memb(r1+#0) = r0  --> this instruction can't be grouped with
// the first two instructions because in instr 1, r0 is conditional on old value
// of p0 but its use in instr 3 is conditional on p0 modified by instr 2 which
// is not valid for new-value stores.
bool HexagonInstrInfo::
isConditionalStore (const MachineInstr* MI) const {
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::S4_storeirbt_io:
    case Hexagon::S4_storeirbf_io:
    case Hexagon::S4_pstorerbt_rr:
    case Hexagon::S4_pstorerbf_rr:
    case Hexagon::S2_pstorerbt_io:
    case Hexagon::S2_pstorerbf_io:
    case Hexagon::S2_pstorerbt_pi:
    case Hexagon::S2_pstorerbf_pi:
    case Hexagon::S2_pstorerdt_io:
    case Hexagon::S2_pstorerdf_io:
    case Hexagon::S4_pstorerdt_rr:
    case Hexagon::S4_pstorerdf_rr:
    case Hexagon::S2_pstorerdt_pi:
    case Hexagon::S2_pstorerdf_pi:
    case Hexagon::S2_pstorerht_io:
    case Hexagon::S2_pstorerhf_io:
    case Hexagon::S4_storeirht_io:
    case Hexagon::S4_storeirhf_io:
    case Hexagon::S4_pstorerht_rr:
    case Hexagon::S4_pstorerhf_rr:
    case Hexagon::S2_pstorerht_pi:
    case Hexagon::S2_pstorerhf_pi:
    case Hexagon::S2_pstorerit_io:
    case Hexagon::S2_pstorerif_io:
    case Hexagon::S4_storeirit_io:
    case Hexagon::S4_storeirif_io:
    case Hexagon::S4_pstorerit_rr:
    case Hexagon::S4_pstorerif_rr:
    case Hexagon::S2_pstorerit_pi:
    case Hexagon::S2_pstorerif_pi:

    // V4 global address store before promoting to dot new.
    case Hexagon::S4_pstorerdt_abs:
    case Hexagon::S4_pstorerdf_abs:
    case Hexagon::S4_pstorerbt_abs:
    case Hexagon::S4_pstorerbf_abs:
    case Hexagon::S4_pstorerht_abs:
    case Hexagon::S4_pstorerhf_abs:
    case Hexagon::S4_pstorerit_abs:
    case Hexagon::S4_pstorerif_abs:
      return true;

    // Predicated new value stores (i.e. if (p0) memw(..)=r0.new) are excluded
    // from the "Conditional Store" list. Because a predicated new value store
    // would NOT be promoted to a double dot new store. See diagram below:
    // This function returns yes for those stores that are predicated but not
    // yet promoted to predicate dot new instructions.
    //
    //                          +---------------------+
    //                    /-----| if (p0) memw(..)=r0 |---------\~
    //                   ||     +---------------------+         ||
    //          promote  ||       /\       /\                   ||  promote
    //                   ||      /||\     /||\                  ||
    //                  \||/    demote     ||                  \||/
    //                   \/       ||       ||                   \/
    //       +-------------------------+   ||   +-------------------------+
    //       | if (p0.new) memw(..)=r0 |   ||   | if (p0) memw(..)=r0.new |
    //       +-------------------------+   ||   +-------------------------+
    //                        ||           ||         ||
    //                        ||         demote      \||/
    //                      promote        ||         \/ NOT possible
    //                        ||           ||         /\~
    //                       \||/          ||        /||\~
    //                        \/           ||         ||
    //                      +-----------------------------+
    //                      | if (p0.new) memw(..)=r0.new |
    //                      +-----------------------------+
    //                           Double Dot New Store
    //
  }
}


bool HexagonInstrInfo::isNewValueJump(const MachineInstr *MI) const {
  if (isNewValue(MI) && isBranch(MI))
    return true;
  return false;
}

bool HexagonInstrInfo::isNewValueJump(Opcode_t Opcode) const {
  return isNewValue(Opcode) && get(Opcode).isBranch() && isPredicated(Opcode);
}

bool HexagonInstrInfo::isPostIncrement (const MachineInstr* MI) const {
  return (getAddrMode(MI) == HexagonII::PostInc);
}

// Returns true, if any one of the operands is a dot new
// insn, whether it is predicated dot new or register dot new.
bool HexagonInstrInfo::isDotNewInst (const MachineInstr* MI) const {
  return (isNewValueInst(MI) ||
     (isPredicated(MI) && isPredicatedNew(MI)));
}

// Returns the most basic instruction for the .new predicated instructions and
// new-value stores.
// For example, all of the following instructions will be converted back to the
// same instruction:
// 1) if (p0.new) memw(R0+#0) = R1.new  --->
// 2) if (p0) memw(R0+#0)= R1.new      -------> if (p0) memw(R0+#0) = R1
// 3) if (p0.new) memw(R0+#0) = R1      --->
//

int HexagonInstrInfo::GetDotOldOp(const int opc) const {
  int NewOp = opc;
  if (isPredicated(NewOp) && isPredicatedNew(NewOp)) { // Get predicate old form
    NewOp = Hexagon::getPredOldOpcode(NewOp);
    assert(NewOp >= 0 &&
           "Couldn't change predicate new instruction to its old form.");
  }

  if (isNewValueStore(NewOp)) { // Convert into non-new-value format
    NewOp = Hexagon::getNonNVStore(NewOp);
    assert(NewOp >= 0 && "Couldn't change new-value store to its old form.");
  }
  return NewOp;
}

// Return the new value instruction for a given store.
int HexagonInstrInfo::GetDotNewOp(const MachineInstr* MI) const {
  int NVOpcode = Hexagon::getNewValueOpcode(MI->getOpcode());
  if (NVOpcode >= 0) // Valid new-value store instruction.
    return NVOpcode;

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown .new type");
  case Hexagon::S4_storerb_ur:
    return Hexagon::S4_storerbnew_ur;

  case Hexagon::S4_storerh_ur:
    return Hexagon::S4_storerhnew_ur;

  case Hexagon::S4_storeri_ur:
    return Hexagon::S4_storerinew_ur;

  case Hexagon::S2_storerb_pci:
    return Hexagon::S2_storerb_pci;

  case Hexagon::S2_storeri_pci:
    return Hexagon::S2_storeri_pci;

  case Hexagon::S2_storerh_pci:
    return Hexagon::S2_storerh_pci;

  case Hexagon::S2_storerd_pci:
    return Hexagon::S2_storerd_pci;

  case Hexagon::S2_storerf_pci:
    return Hexagon::S2_storerf_pci;
  }
  return 0;
}

// Return .new predicate version for an instruction.
int HexagonInstrInfo::GetDotNewPredOp(MachineInstr *MI,
                                      const MachineBranchProbabilityInfo
                                      *MBPI) const {

  int NewOpcode = Hexagon::getPredNewOpcode(MI->getOpcode());
  if (NewOpcode >= 0) // Valid predicate new instruction
    return NewOpcode;

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown .new type");
  // Condtional Jumps
  case Hexagon::J2_jumpt:
  case Hexagon::J2_jumpf:
    return getDotNewPredJumpOp(MI, MBPI);

  case Hexagon::J2_jumprt:
    return Hexagon::J2_jumptnewpt;

  case Hexagon::J2_jumprf:
    return Hexagon::J2_jumprfnewpt;

  case Hexagon::JMPrett:
    return Hexagon::J2_jumprtnewpt;

  case Hexagon::JMPretf:
    return Hexagon::J2_jumprfnewpt;


  // Conditional combine
  case Hexagon::C2_ccombinewt:
    return Hexagon::C2_ccombinewnewt;
  case Hexagon::C2_ccombinewf:
    return Hexagon::C2_ccombinewnewf;
  }
}


unsigned HexagonInstrInfo::getAddrMode(const MachineInstr* MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return((F >> HexagonII::AddrModePos) & HexagonII::AddrModeMask);
}

/// immediateExtend - Changes the instruction in place to one using an immediate
/// extender.
void HexagonInstrInfo::immediateExtend(MachineInstr *MI) const {
  assert((isExtendable(MI)||isConstExtended(MI)) &&
                               "Instruction must be extendable");
  // Find which operand is extendable.
  short ExtOpNum = getCExtOpNum(MI);
  MachineOperand &MO = MI->getOperand(ExtOpNum);
  // This needs to be something we understand.
  assert((MO.isMBB() || MO.isImm()) &&
         "Branch with unknown extendable field type");
  // Mark given operand as extended.
  MO.addTargetFlag(HexagonII::HMOTF_ConstExtended);
}

DFAPacketizer *HexagonInstrInfo::CreateTargetScheduleState(
    const TargetSubtargetInfo &STI) const {
  const InstrItineraryData *II = STI.getInstrItineraryData();
  return static_cast<const HexagonSubtarget &>(STI).createDFAPacketizer(II);
}

bool HexagonInstrInfo::isSchedulingBoundary(const MachineInstr *MI,
                                            const MachineBasicBlock *MBB,
                                            const MachineFunction &MF) const {
  // Debug info is never a scheduling boundary. It's necessary to be explicit
  // due to the special treatment of IT instructions below, otherwise a
  // dbg_value followed by an IT will result in the IT instruction being
  // considered a scheduling hazard, which is wrong. It should be the actual
  // instruction preceding the dbg_value instruction(s), just like it is
  // when debug info is not present.
  if (MI->isDebugValue())
    return false;

  // Terminators and labels can't be scheduled around.
  if (MI->getDesc().isTerminator() || MI->isPosition() || MI->isInlineAsm())
    return true;

  return false;
}

bool HexagonInstrInfo::isConstExtended(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  unsigned isExtended = (F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
  if (isExtended) // Instruction must be extended.
    return true;

  unsigned isExtendable =
    (F >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask;
  if (!isExtendable)
    return false;

  short ExtOpNum = getCExtOpNum(MI);
  const MachineOperand &MO = MI->getOperand(ExtOpNum);
  // Use MO operand flags to determine if MO
  // has the HMOTF_ConstExtended flag set.
  if (MO.getTargetFlags() && HexagonII::HMOTF_ConstExtended)
    return true;
  // If this is a Machine BB address we are talking about, and it is
  // not marked as extended, say so.
  if (MO.isMBB())
    return false;

  // We could be using an instruction with an extendable immediate and shoehorn
  // a global address into it. If it is a global address it will be constant
  // extended. We do this for COMBINE.
  // We currently only handle isGlobal() because it is the only kind of
  // object we are going to end up with here for now.
  // In the future we probably should add isSymbol(), etc.
  if (MO.isGlobal() || MO.isSymbol() || MO.isBlockAddress() ||
      MO.isJTI() || MO.isCPI())
    return true;

  // If the extendable operand is not 'Immediate' type, the instruction should
  // have 'isExtended' flag set.
  assert(MO.isImm() && "Extendable operand must be Immediate type");

  int MinValue = getMinValue(MI);
  int MaxValue = getMaxValue(MI);
  int ImmValue = MO.getImm();

  return (ImmValue < MinValue || ImmValue > MaxValue);
}

// Return the number of bytes required to encode the instruction.
// Hexagon instructions are fixed length, 4 bytes, unless they
// use a constant extender, which requires another 4 bytes.
// For debug instructions and prolog labels, return 0.
unsigned HexagonInstrInfo::getSize(const MachineInstr *MI) const {

  if (MI->isDebugValue() || MI->isPosition())
    return 0;

  unsigned Size = MI->getDesc().getSize();
  if (!Size)
    // Assume the default insn size in case it cannot be determined
    // for whatever reason.
    Size = HEXAGON_INSTR_SIZE;

  if (isConstExtended(MI) || isExtended(MI))
    Size += HEXAGON_INSTR_SIZE;

  return Size;
}

// Returns the opcode to use when converting MI, which is a conditional jump,
// into a conditional instruction which uses the .new value of the predicate.
// We also use branch probabilities to add a hint to the jump.
int
HexagonInstrInfo::getDotNewPredJumpOp(MachineInstr *MI,
                                  const
                                  MachineBranchProbabilityInfo *MBPI) const {

  // We assume that block can have at most two successors.
  bool taken = false;
  MachineBasicBlock *Src = MI->getParent();
  MachineOperand *BrTarget = &MI->getOperand(1);
  MachineBasicBlock *Dst = BrTarget->getMBB();

  const BranchProbability Prediction = MBPI->getEdgeProbability(Src, Dst);
  if (Prediction >= BranchProbability(1,2))
    taken = true;

  switch (MI->getOpcode()) {
  case Hexagon::J2_jumpt:
    return taken ? Hexagon::J2_jumptnewpt : Hexagon::J2_jumptnew;
  case Hexagon::J2_jumpf:
    return taken ? Hexagon::J2_jumpfnewpt : Hexagon::J2_jumpfnew;

  default:
    llvm_unreachable("Unexpected jump instruction.");
  }
}
// Returns true if a particular operand is extendable for an instruction.
bool HexagonInstrInfo::isOperandExtended(const MachineInstr *MI,
                                         unsigned short OperandNum) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask)
          == OperandNum;
}

// Returns Operand Index for the constant extended instruction.
unsigned short HexagonInstrInfo::getCExtOpNum(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

// Returns the min value that doesn't need to be extended.
int HexagonInstrInfo::getMinValue(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  unsigned isSigned = (F >> HexagonII::ExtentSignedPos)
                    & HexagonII::ExtentSignedMask;
  unsigned bits =  (F >> HexagonII::ExtentBitsPos)
                    & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return -1U << (bits - 1);
  else
    return 0;
}

// Returns the max value that doesn't need to be extended.
int HexagonInstrInfo::getMaxValue(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  unsigned isSigned = (F >> HexagonII::ExtentSignedPos)
                    & HexagonII::ExtentSignedMask;
  unsigned bits =  (F >> HexagonII::ExtentBitsPos)
                    & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return ~(-1U << (bits - 1));
  else
    return ~(-1U << bits);
}

// Returns true if an instruction can be converted into a non-extended
// equivalent instruction.
bool HexagonInstrInfo::NonExtEquivalentExists (const MachineInstr *MI) const {

  short NonExtOpcode;
  // Check if the instruction has a register form that uses register in place
  // of the extended operand, if so return that as the non-extended form.
  if (Hexagon::getRegForm(MI->getOpcode()) >= 0)
    return true;

  if (MI->getDesc().mayLoad() || MI->getDesc().mayStore()) {
    // Check addressing mode and retrieve non-ext equivalent instruction.

    switch (getAddrMode(MI)) {
    case HexagonII::Absolute :
      // Load/store with absolute addressing mode can be converted into
      // base+offset mode.
      NonExtOpcode = Hexagon::getBaseWithImmOffset(MI->getOpcode());
      break;
    case HexagonII::BaseImmOffset :
      // Load/store with base+offset addressing mode can be converted into
      // base+register offset addressing mode. However left shift operand should
      // be set to 0.
      NonExtOpcode = Hexagon::getBaseWithRegOffset(MI->getOpcode());
      break;
    default:
      return false;
    }
    if (NonExtOpcode < 0)
      return false;
    return true;
  }
  return false;
}

// Returns opcode of the non-extended equivalent instruction.
short HexagonInstrInfo::getNonExtOpcode (const MachineInstr *MI) const {

  // Check if the instruction has a register form that uses register in place
  // of the extended operand, if so return that as the non-extended form.
  short NonExtOpcode = Hexagon::getRegForm(MI->getOpcode());
    if (NonExtOpcode >= 0)
      return NonExtOpcode;

  if (MI->getDesc().mayLoad() || MI->getDesc().mayStore()) {
    // Check addressing mode and retrieve non-ext equivalent instruction.
    switch (getAddrMode(MI)) {
    case HexagonII::Absolute :
      return Hexagon::getBaseWithImmOffset(MI->getOpcode());
    case HexagonII::BaseImmOffset :
      return Hexagon::getBaseWithRegOffset(MI->getOpcode());
    default:
      return -1;
    }
  }
  return -1;
}

bool HexagonInstrInfo::PredOpcodeHasJMP_c(Opcode_t Opcode) const {
  return (Opcode == Hexagon::J2_jumpt) ||
         (Opcode == Hexagon::J2_jumpf) ||
         (Opcode == Hexagon::J2_jumptnewpt) ||
         (Opcode == Hexagon::J2_jumpfnewpt) ||
         (Opcode == Hexagon::J2_jumpt) ||
         (Opcode == Hexagon::J2_jumpf);
}

bool HexagonInstrInfo::predOpcodeHasNot(ArrayRef<MachineOperand> Cond) const {
  if (Cond.empty() || !isPredicated(Cond[0].getImm()))
    return false;
  return !isPredicatedTrue(Cond[0].getImm());
}

bool HexagonInstrInfo::isEndLoopN(Opcode_t Opcode) const {
  return (Opcode == Hexagon::ENDLOOP0 ||
          Opcode == Hexagon::ENDLOOP1);
}

bool HexagonInstrInfo::getPredReg(ArrayRef<MachineOperand> Cond,
                                  unsigned &PredReg, unsigned &PredRegPos,
                                  unsigned &PredRegFlags) const {
  if (Cond.empty())
    return false;
  assert(Cond.size() == 2);
  if (isNewValueJump(Cond[0].getImm()) || Cond[1].isMBB()) {
     DEBUG(dbgs() << "No predregs for new-value jumps/endloop");
     return false;
  }
  PredReg = Cond[1].getReg();
  PredRegPos = 1;
  // See IfConversion.cpp why we add RegState::Implicit | RegState::Undef
  PredRegFlags = 0;
  if (Cond[1].isImplicit())
    PredRegFlags = RegState::Implicit;
  if (Cond[1].isUndef())
    PredRegFlags |= RegState::Undef;
  return true;
}

