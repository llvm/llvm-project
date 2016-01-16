//===-- PPCInstrInfo.cpp - PowerPC Instruction Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "PPC.h"
#include "PPCHazardRecognizers.h"
#include "PPCInstrBuilder.h"
#include "PPCMachineFunctionInfo.h"
#include "PPCTargetMachine.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ppc-instr-info"

#define GET_INSTRMAP_INFO
#define GET_INSTRINFO_CTOR_DTOR
#include "PPCGenInstrInfo.inc"

static cl::
opt<bool> DisableCTRLoopAnal("disable-ppc-ctrloop-analysis", cl::Hidden,
            cl::desc("Disable analysis for CTR loops"));

static cl::opt<bool> DisableCmpOpt("disable-ppc-cmp-opt",
cl::desc("Disable compare instruction optimization"), cl::Hidden);

static cl::opt<bool> VSXSelfCopyCrash("crash-on-ppc-vsx-self-copy",
cl::desc("Causes the backend to crash instead of generating a nop VSX copy"),
cl::Hidden);

static cl::opt<bool>
UseOldLatencyCalc("ppc-old-latency-calc", cl::Hidden,
  cl::desc("Use the old (incorrect) instruction latency calculation"));

// Pin the vtable to this file.
void PPCInstrInfo::anchor() {}

PPCInstrInfo::PPCInstrInfo(PPCSubtarget &STI)
    : PPCGenInstrInfo(PPC::ADJCALLSTACKDOWN, PPC::ADJCALLSTACKUP),
      Subtarget(STI), RI(STI.getTargetMachine()) {}

/// CreateTargetHazardRecognizer - Return the hazard recognizer to use for
/// this target when scheduling the DAG.
ScheduleHazardRecognizer *
PPCInstrInfo::CreateTargetHazardRecognizer(const TargetSubtargetInfo *STI,
                                           const ScheduleDAG *DAG) const {
  unsigned Directive =
      static_cast<const PPCSubtarget *>(STI)->getDarwinDirective();
  if (Directive == PPC::DIR_440 || Directive == PPC::DIR_A2 ||
      Directive == PPC::DIR_E500mc || Directive == PPC::DIR_E5500) {
    const InstrItineraryData *II =
        static_cast<const PPCSubtarget *>(STI)->getInstrItineraryData();
    return new ScoreboardHazardRecognizer(II, DAG);
  }

  return TargetInstrInfo::CreateTargetHazardRecognizer(STI, DAG);
}

/// CreateTargetPostRAHazardRecognizer - Return the postRA hazard recognizer
/// to use for this target when scheduling the DAG.
ScheduleHazardRecognizer *
PPCInstrInfo::CreateTargetPostRAHazardRecognizer(const InstrItineraryData *II,
                                                 const ScheduleDAG *DAG) const {
  unsigned Directive =
      DAG->MF.getSubtarget<PPCSubtarget>().getDarwinDirective();

  if (Directive == PPC::DIR_PWR7 || Directive == PPC::DIR_PWR8)
    return new PPCDispatchGroupSBHazardRecognizer(II, DAG);

  // Most subtargets use a PPC970 recognizer.
  if (Directive != PPC::DIR_440 && Directive != PPC::DIR_A2 &&
      Directive != PPC::DIR_E500mc && Directive != PPC::DIR_E5500) {
    assert(DAG->TII && "No InstrInfo?");

    return new PPCHazardRecognizer970(*DAG);
  }

  return new ScoreboardHazardRecognizer(II, DAG);
}

unsigned PPCInstrInfo::getInstrLatency(const InstrItineraryData *ItinData,
                                       const MachineInstr *MI,
                                       unsigned *PredCost) const {
  if (!ItinData || UseOldLatencyCalc)
    return PPCGenInstrInfo::getInstrLatency(ItinData, MI, PredCost);

  // The default implementation of getInstrLatency calls getStageLatency, but
  // getStageLatency does not do the right thing for us. While we have
  // itinerary, most cores are fully pipelined, and so the itineraries only
  // express the first part of the pipeline, not every stage. Instead, we need
  // to use the listed output operand cycle number (using operand 0 here, which
  // is an output).

  unsigned Latency = 1;
  unsigned DefClass = MI->getDesc().getSchedClass();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isDef() || MO.isImplicit())
      continue;

    int Cycle = ItinData->getOperandCycle(DefClass, i);
    if (Cycle < 0)
      continue;

    Latency = std::max(Latency, (unsigned) Cycle);
  }

  return Latency;
}

int PPCInstrInfo::getOperandLatency(const InstrItineraryData *ItinData,
                                    const MachineInstr *DefMI, unsigned DefIdx,
                                    const MachineInstr *UseMI,
                                    unsigned UseIdx) const {
  int Latency = PPCGenInstrInfo::getOperandLatency(ItinData, DefMI, DefIdx,
                                                   UseMI, UseIdx);

  if (!DefMI->getParent())
    return Latency;

  const MachineOperand &DefMO = DefMI->getOperand(DefIdx);
  unsigned Reg = DefMO.getReg();

  bool IsRegCR;
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    const MachineRegisterInfo *MRI =
      &DefMI->getParent()->getParent()->getRegInfo();
    IsRegCR = MRI->getRegClass(Reg)->hasSuperClassEq(&PPC::CRRCRegClass) ||
              MRI->getRegClass(Reg)->hasSuperClassEq(&PPC::CRBITRCRegClass);
  } else {
    IsRegCR = PPC::CRRCRegClass.contains(Reg) ||
              PPC::CRBITRCRegClass.contains(Reg);
  }

  if (UseMI->isBranch() && IsRegCR) {
    if (Latency < 0)
      Latency = getInstrLatency(ItinData, DefMI);

    // On some cores, there is an additional delay between writing to a condition
    // register, and using it from a branch.
    unsigned Directive = Subtarget.getDarwinDirective();
    switch (Directive) {
    default: break;
    case PPC::DIR_7400:
    case PPC::DIR_750:
    case PPC::DIR_970:
    case PPC::DIR_E5500:
    case PPC::DIR_PWR4:
    case PPC::DIR_PWR5:
    case PPC::DIR_PWR5X:
    case PPC::DIR_PWR6:
    case PPC::DIR_PWR6X:
    case PPC::DIR_PWR7:
    case PPC::DIR_PWR8:
      Latency += 2;
      break;
    }
  }

  return Latency;
}

// This function does not list all associative and commutative operations, but
// only those worth feeding through the machine combiner in an attempt to
// reduce the critical path. Mostly, this means floating-point operations,
// because they have high latencies (compared to other operations, such and
// and/or, which are also associative and commutative, but have low latencies).
bool PPCInstrInfo::isAssociativeAndCommutative(const MachineInstr &Inst) const {
  switch (Inst.getOpcode()) {
  // FP Add:
  case PPC::FADD:
  case PPC::FADDS:
  // FP Multiply:
  case PPC::FMUL:
  case PPC::FMULS:
  // Altivec Add:
  case PPC::VADDFP:
  // VSX Add:
  case PPC::XSADDDP:
  case PPC::XVADDDP:
  case PPC::XVADDSP:
  case PPC::XSADDSP:
  // VSX Multiply:
  case PPC::XSMULDP:
  case PPC::XVMULDP:
  case PPC::XVMULSP:
  case PPC::XSMULSP:
  // QPX Add:
  case PPC::QVFADD:
  case PPC::QVFADDS:
  case PPC::QVFADDSs:
  // QPX Multiply:
  case PPC::QVFMUL:
  case PPC::QVFMULS:
  case PPC::QVFMULSs:
    return true;
  default:
    return false;
  }
}

bool PPCInstrInfo::getMachineCombinerPatterns(
    MachineInstr &Root,
    SmallVectorImpl<MachineCombinerPattern> &Patterns) const {
  // Using the machine combiner in this way is potentially expensive, so
  // restrict to when aggressive optimizations are desired.
  if (Subtarget.getTargetMachine().getOptLevel() != CodeGenOpt::Aggressive)
    return false;

  // FP reassociation is only legal when we don't need strict IEEE semantics.
  if (!Root.getParent()->getParent()->getTarget().Options.UnsafeFPMath)
    return false;

  return TargetInstrInfo::getMachineCombinerPatterns(Root, Patterns);
}

// Detect 32 -> 64-bit extensions where we may reuse the low sub-register.
bool PPCInstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                         unsigned &SrcReg, unsigned &DstReg,
                                         unsigned &SubIdx) const {
  switch (MI.getOpcode()) {
  default: return false;
  case PPC::EXTSW:
  case PPC::EXTSW_32_64:
    SrcReg = MI.getOperand(1).getReg();
    DstReg = MI.getOperand(0).getReg();
    SubIdx = PPC::sub_32;
    return true;
  }
}

unsigned PPCInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                           int &FrameIndex) const {
  // Note: This list must be kept consistent with LoadRegFromStackSlot.
  switch (MI->getOpcode()) {
  default: break;
  case PPC::LD:
  case PPC::LWZ:
  case PPC::LFS:
  case PPC::LFD:
  case PPC::RESTORE_CR:
  case PPC::RESTORE_CRBIT:
  case PPC::LVX:
  case PPC::LXVD2X:
  case PPC::QVLFDX:
  case PPC::QVLFSXs:
  case PPC::QVLFDXb:
  case PPC::RESTORE_VRSAVE:
    // Check for the operands added by addFrameReference (the immediate is the
    // offset which defaults to 0).
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned PPCInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                          int &FrameIndex) const {
  // Note: This list must be kept consistent with StoreRegToStackSlot.
  switch (MI->getOpcode()) {
  default: break;
  case PPC::STD:
  case PPC::STW:
  case PPC::STFS:
  case PPC::STFD:
  case PPC::SPILL_CR:
  case PPC::SPILL_CRBIT:
  case PPC::STVX:
  case PPC::STXVD2X:
  case PPC::QVSTFDX:
  case PPC::QVSTFSXs:
  case PPC::QVSTFDXb:
  case PPC::SPILL_VRSAVE:
    // Check for the operands added by addFrameReference (the immediate is the
    // offset which defaults to 0).
    if (MI->getOperand(1).isImm() && !MI->getOperand(1).getImm() &&
        MI->getOperand(2).isFI()) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

MachineInstr *PPCInstrInfo::commuteInstructionImpl(MachineInstr *MI,
                                                   bool NewMI,
                                                   unsigned OpIdx1,
                                                   unsigned OpIdx2) const {
  MachineFunction &MF = *MI->getParent()->getParent();

  // Normal instructions can be commuted the obvious way.
  if (MI->getOpcode() != PPC::RLWIMI &&
      MI->getOpcode() != PPC::RLWIMIo)
    return TargetInstrInfo::commuteInstructionImpl(MI, NewMI, OpIdx1, OpIdx2);
  // Note that RLWIMI can be commuted as a 32-bit instruction, but not as a
  // 64-bit instruction (so we don't handle PPC::RLWIMI8 here), because
  // changing the relative order of the mask operands might change what happens
  // to the high-bits of the mask (and, thus, the result).

  // Cannot commute if it has a non-zero rotate count.
  if (MI->getOperand(3).getImm() != 0)
    return nullptr;

  // If we have a zero rotate count, we have:
  //   M = mask(MB,ME)
  //   Op0 = (Op1 & ~M) | (Op2 & M)
  // Change this to:
  //   M = mask((ME+1)&31, (MB-1)&31)
  //   Op0 = (Op2 & ~M) | (Op1 & M)

  // Swap op1/op2
  assert(((OpIdx1 == 1 && OpIdx2 == 2) || (OpIdx1 == 2 && OpIdx2 == 1)) &&
         "Only the operands 1 and 2 can be swapped in RLSIMI/RLWIMIo.");
  unsigned Reg0 = MI->getOperand(0).getReg();
  unsigned Reg1 = MI->getOperand(1).getReg();
  unsigned Reg2 = MI->getOperand(2).getReg();
  unsigned SubReg1 = MI->getOperand(1).getSubReg();
  unsigned SubReg2 = MI->getOperand(2).getSubReg();
  bool Reg1IsKill = MI->getOperand(1).isKill();
  bool Reg2IsKill = MI->getOperand(2).isKill();
  bool ChangeReg0 = false;
  // If machine instrs are no longer in two-address forms, update
  // destination register as well.
  if (Reg0 == Reg1) {
    // Must be two address instruction!
    assert(MI->getDesc().getOperandConstraint(0, MCOI::TIED_TO) &&
           "Expecting a two-address instruction!");
    assert(MI->getOperand(0).getSubReg() == SubReg1 && "Tied subreg mismatch");
    Reg2IsKill = false;
    ChangeReg0 = true;
  }

  // Masks.
  unsigned MB = MI->getOperand(4).getImm();
  unsigned ME = MI->getOperand(5).getImm();

  // We can't commute a trivial mask (there is no way to represent an all-zero
  // mask).
  if (MB == 0 && ME == 31)
    return nullptr;

  if (NewMI) {
    // Create a new instruction.
    unsigned Reg0 = ChangeReg0 ? Reg2 : MI->getOperand(0).getReg();
    bool Reg0IsDead = MI->getOperand(0).isDead();
    return BuildMI(MF, MI->getDebugLoc(), MI->getDesc())
      .addReg(Reg0, RegState::Define | getDeadRegState(Reg0IsDead))
      .addReg(Reg2, getKillRegState(Reg2IsKill))
      .addReg(Reg1, getKillRegState(Reg1IsKill))
      .addImm((ME+1) & 31)
      .addImm((MB-1) & 31);
  }

  if (ChangeReg0) {
    MI->getOperand(0).setReg(Reg2);
    MI->getOperand(0).setSubReg(SubReg2);
  }
  MI->getOperand(2).setReg(Reg1);
  MI->getOperand(1).setReg(Reg2);
  MI->getOperand(2).setSubReg(SubReg1);
  MI->getOperand(1).setSubReg(SubReg2);
  MI->getOperand(2).setIsKill(Reg1IsKill);
  MI->getOperand(1).setIsKill(Reg2IsKill);

  // Swap the mask around.
  MI->getOperand(4).setImm((ME+1) & 31);
  MI->getOperand(5).setImm((MB-1) & 31);
  return MI;
}

bool PPCInstrInfo::findCommutedOpIndices(MachineInstr *MI, unsigned &SrcOpIdx1,
                                         unsigned &SrcOpIdx2) const {
  // For VSX A-Type FMA instructions, it is the first two operands that can be
  // commuted, however, because the non-encoded tied input operand is listed
  // first, the operands to swap are actually the second and third.

  int AltOpc = PPC::getAltVSXFMAOpcode(MI->getOpcode());
  if (AltOpc == -1)
    return TargetInstrInfo::findCommutedOpIndices(MI, SrcOpIdx1, SrcOpIdx2);

  // The commutable operand indices are 2 and 3. Return them in SrcOpIdx1
  // and SrcOpIdx2.
  return fixCommutedOpIndices(SrcOpIdx1, SrcOpIdx2, 2, 3);
}

void PPCInstrInfo::insertNoop(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI) const {
  // This function is used for scheduling, and the nop wanted here is the type
  // that terminates dispatch groups on the POWER cores.
  unsigned Directive = Subtarget.getDarwinDirective();
  unsigned Opcode;
  switch (Directive) {
  default:            Opcode = PPC::NOP; break;
  case PPC::DIR_PWR6: Opcode = PPC::NOP_GT_PWR6; break;
  case PPC::DIR_PWR7: Opcode = PPC::NOP_GT_PWR7; break;
  case PPC::DIR_PWR8: Opcode = PPC::NOP_GT_PWR7; break; /* FIXME: Update when P8 InstrScheduling model is ready */
  }

  DebugLoc DL;
  BuildMI(MBB, MI, DL, get(Opcode));
}

/// getNoopForMachoTarget - Return the noop instruction to use for a noop.
void PPCInstrInfo::getNoopForMachoTarget(MCInst &NopInst) const {
  NopInst.setOpcode(PPC::NOP);
}

// Branch analysis.
// Note: If the condition register is set to CTR or CTR8 then this is a
// BDNZ (imm == 1) or BDZ (imm == 0) branch.
bool PPCInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  bool isPPC64 = Subtarget.isPPC64();

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return false;

  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;

  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (LastInst->getOpcode() == PPC::B) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    } else if (LastInst->getOpcode() == PPC::BCC) {
      if (!LastInst->getOperand(2).isMBB())
        return true;
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(2).getMBB();
      Cond.push_back(LastInst->getOperand(0));
      Cond.push_back(LastInst->getOperand(1));
      return false;
    } else if (LastInst->getOpcode() == PPC::BC) {
      if (!LastInst->getOperand(1).isMBB())
        return true;
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(PPC::PRED_BIT_SET));
      Cond.push_back(LastInst->getOperand(0));
      return false;
    } else if (LastInst->getOpcode() == PPC::BCn) {
      if (!LastInst->getOperand(1).isMBB())
        return true;
      // Block ends with fall-through condbranch.
      TBB = LastInst->getOperand(1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(PPC::PRED_BIT_UNSET));
      Cond.push_back(LastInst->getOperand(0));
      return false;
    } else if (LastInst->getOpcode() == PPC::BDNZ8 ||
               LastInst->getOpcode() == PPC::BDNZ) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      if (DisableCTRLoopAnal)
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(1));
      Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                               true));
      return false;
    } else if (LastInst->getOpcode() == PPC::BDZ8 ||
               LastInst->getOpcode() == PPC::BDZ) {
      if (!LastInst->getOperand(0).isMBB())
        return true;
      if (DisableCTRLoopAnal)
        return true;
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(MachineOperand::CreateImm(0));
      Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                               true));
      return false;
    }

    // Otherwise, don't know what this is.
    return true;
  }

  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with PPC::B and PPC:BCC, handle it.
  if (SecondLastInst->getOpcode() == PPC::BCC &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(2).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    TBB =  SecondLastInst->getOperand(2).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    Cond.push_back(SecondLastInst->getOperand(1));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  } else if (SecondLastInst->getOpcode() == PPC::BC &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(1).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    TBB =  SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(PPC::PRED_BIT_SET));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  } else if (SecondLastInst->getOpcode() == PPC::BCn &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(1).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    TBB =  SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(PPC::PRED_BIT_UNSET));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  } else if ((SecondLastInst->getOpcode() == PPC::BDNZ8 ||
              SecondLastInst->getOpcode() == PPC::BDNZ) &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    if (DisableCTRLoopAnal)
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(MachineOperand::CreateImm(1));
    Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                             true));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  } else if ((SecondLastInst->getOpcode() == PPC::BDZ8 ||
              SecondLastInst->getOpcode() == PPC::BDZ) &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB() ||
        !LastInst->getOperand(0).isMBB())
      return true;
    if (DisableCTRLoopAnal)
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(MachineOperand::CreateImm(0));
    Cond.push_back(MachineOperand::CreateReg(isPPC64 ? PPC::CTR8 : PPC::CTR,
                                             true));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two PPC:Bs, handle it.  The second one is not
  // executed, so remove it.
  if (SecondLastInst->getOpcode() == PPC::B &&
      LastInst->getOpcode() == PPC::B) {
    if (!SecondLastInst->getOperand(0).isMBB())
      return true;
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned PPCInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.getLastNonDebugInstr();
  if (I == MBB.end())
    return 0;

  if (I->getOpcode() != PPC::B && I->getOpcode() != PPC::BCC &&
      I->getOpcode() != PPC::BC && I->getOpcode() != PPC::BCn &&
      I->getOpcode() != PPC::BDNZ8 && I->getOpcode() != PPC::BDNZ &&
      I->getOpcode() != PPC::BDZ8  && I->getOpcode() != PPC::BDZ)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != PPC::BCC &&
      I->getOpcode() != PPC::BC && I->getOpcode() != PPC::BCn &&
      I->getOpcode() != PPC::BDNZ8 && I->getOpcode() != PPC::BDNZ &&
      I->getOpcode() != PPC::BDZ8  && I->getOpcode() != PPC::BDZ)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

unsigned
PPCInstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                           MachineBasicBlock *FBB,
                           ArrayRef<MachineOperand> Cond,
                           DebugLoc DL) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "PPC branch conditions have two components!");

  bool isPPC64 = Subtarget.isPPC64();

  // One-way branch.
  if (!FBB) {
    if (Cond.empty())   // Unconditional branch
      BuildMI(&MBB, DL, get(PPC::B)).addMBB(TBB);
    else if (Cond[1].getReg() == PPC::CTR || Cond[1].getReg() == PPC::CTR8)
      BuildMI(&MBB, DL, get(Cond[0].getImm() ?
                              (isPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                              (isPPC64 ? PPC::BDZ8  : PPC::BDZ))).addMBB(TBB);
    else if (Cond[0].getImm() == PPC::PRED_BIT_SET)
      BuildMI(&MBB, DL, get(PPC::BC)).addOperand(Cond[1]).addMBB(TBB);
    else if (Cond[0].getImm() == PPC::PRED_BIT_UNSET)
      BuildMI(&MBB, DL, get(PPC::BCn)).addOperand(Cond[1]).addMBB(TBB);
    else                // Conditional branch
      BuildMI(&MBB, DL, get(PPC::BCC))
        .addImm(Cond[0].getImm()).addOperand(Cond[1]).addMBB(TBB);
    return 1;
  }

  // Two-way Conditional Branch.
  if (Cond[1].getReg() == PPC::CTR || Cond[1].getReg() == PPC::CTR8)
    BuildMI(&MBB, DL, get(Cond[0].getImm() ?
                            (isPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                            (isPPC64 ? PPC::BDZ8  : PPC::BDZ))).addMBB(TBB);
  else if (Cond[0].getImm() == PPC::PRED_BIT_SET)
    BuildMI(&MBB, DL, get(PPC::BC)).addOperand(Cond[1]).addMBB(TBB);
  else if (Cond[0].getImm() == PPC::PRED_BIT_UNSET)
    BuildMI(&MBB, DL, get(PPC::BCn)).addOperand(Cond[1]).addMBB(TBB);
  else
    BuildMI(&MBB, DL, get(PPC::BCC))
      .addImm(Cond[0].getImm()).addOperand(Cond[1]).addMBB(TBB);
  BuildMI(&MBB, DL, get(PPC::B)).addMBB(FBB);
  return 2;
}

// Select analysis.
bool PPCInstrInfo::canInsertSelect(const MachineBasicBlock &MBB,
                ArrayRef<MachineOperand> Cond,
                unsigned TrueReg, unsigned FalseReg,
                int &CondCycles, int &TrueCycles, int &FalseCycles) const {
  if (!Subtarget.hasISEL())
    return false;

  if (Cond.size() != 2)
    return false;

  // If this is really a bdnz-like condition, then it cannot be turned into a
  // select.
  if (Cond[1].getReg() == PPC::CTR || Cond[1].getReg() == PPC::CTR8)
    return false;

  // Check register classes.
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
    RI.getCommonSubClass(MRI.getRegClass(TrueReg), MRI.getRegClass(FalseReg));
  if (!RC)
    return false;

  // isel is for regular integer GPRs only.
  if (!PPC::GPRCRegClass.hasSubClassEq(RC) &&
      !PPC::GPRC_NOR0RegClass.hasSubClassEq(RC) &&
      !PPC::G8RCRegClass.hasSubClassEq(RC) &&
      !PPC::G8RC_NOX0RegClass.hasSubClassEq(RC))
    return false;

  // FIXME: These numbers are for the A2, how well they work for other cores is
  // an open question. On the A2, the isel instruction has a 2-cycle latency
  // but single-cycle throughput. These numbers are used in combination with
  // the MispredictPenalty setting from the active SchedMachineModel.
  CondCycles = 1;
  TrueCycles = 1;
  FalseCycles = 1;

  return true;
}

void PPCInstrInfo::insertSelect(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI, DebugLoc dl,
                                unsigned DestReg, ArrayRef<MachineOperand> Cond,
                                unsigned TrueReg, unsigned FalseReg) const {
  assert(Cond.size() == 2 &&
         "PPC branch conditions have two components!");

  assert(Subtarget.hasISEL() &&
         "Cannot insert select on target without ISEL support");

  // Get the register classes.
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC =
    RI.getCommonSubClass(MRI.getRegClass(TrueReg), MRI.getRegClass(FalseReg));
  assert(RC && "TrueReg and FalseReg must have overlapping register classes");

  bool Is64Bit = PPC::G8RCRegClass.hasSubClassEq(RC) ||
                 PPC::G8RC_NOX0RegClass.hasSubClassEq(RC);
  assert((Is64Bit ||
          PPC::GPRCRegClass.hasSubClassEq(RC) ||
          PPC::GPRC_NOR0RegClass.hasSubClassEq(RC)) &&
         "isel is for regular integer GPRs only");

  unsigned OpCode = Is64Bit ? PPC::ISEL8 : PPC::ISEL;
  auto SelectPred = static_cast<PPC::Predicate>(Cond[0].getImm());

  unsigned SubIdx = 0;
  bool SwapOps = false;
  switch (SelectPred) {
  case PPC::PRED_EQ:
  case PPC::PRED_EQ_MINUS:
  case PPC::PRED_EQ_PLUS:
      SubIdx = PPC::sub_eq; SwapOps = false; break;
  case PPC::PRED_NE:
  case PPC::PRED_NE_MINUS:
  case PPC::PRED_NE_PLUS:
      SubIdx = PPC::sub_eq; SwapOps = true; break;
  case PPC::PRED_LT:
  case PPC::PRED_LT_MINUS:
  case PPC::PRED_LT_PLUS:
      SubIdx = PPC::sub_lt; SwapOps = false; break;
  case PPC::PRED_GE:
  case PPC::PRED_GE_MINUS:
  case PPC::PRED_GE_PLUS:
      SubIdx = PPC::sub_lt; SwapOps = true; break;
  case PPC::PRED_GT:
  case PPC::PRED_GT_MINUS:
  case PPC::PRED_GT_PLUS:
      SubIdx = PPC::sub_gt; SwapOps = false; break;
  case PPC::PRED_LE:
  case PPC::PRED_LE_MINUS:
  case PPC::PRED_LE_PLUS:
      SubIdx = PPC::sub_gt; SwapOps = true; break;
  case PPC::PRED_UN:
  case PPC::PRED_UN_MINUS:
  case PPC::PRED_UN_PLUS:
      SubIdx = PPC::sub_un; SwapOps = false; break;
  case PPC::PRED_NU:
  case PPC::PRED_NU_MINUS:
  case PPC::PRED_NU_PLUS:
      SubIdx = PPC::sub_un; SwapOps = true; break;
  case PPC::PRED_BIT_SET:   SubIdx = 0; SwapOps = false; break;
  case PPC::PRED_BIT_UNSET: SubIdx = 0; SwapOps = true; break;
  }

  unsigned FirstReg =  SwapOps ? FalseReg : TrueReg,
           SecondReg = SwapOps ? TrueReg  : FalseReg;

  // The first input register of isel cannot be r0. If it is a member
  // of a register class that can be r0, then copy it first (the
  // register allocator should eliminate the copy).
  if (MRI.getRegClass(FirstReg)->contains(PPC::R0) ||
      MRI.getRegClass(FirstReg)->contains(PPC::X0)) {
    const TargetRegisterClass *FirstRC =
      MRI.getRegClass(FirstReg)->contains(PPC::X0) ?
        &PPC::G8RC_NOX0RegClass : &PPC::GPRC_NOR0RegClass;
    unsigned OldFirstReg = FirstReg;
    FirstReg = MRI.createVirtualRegister(FirstRC);
    BuildMI(MBB, MI, dl, get(TargetOpcode::COPY), FirstReg)
      .addReg(OldFirstReg);
  }

  BuildMI(MBB, MI, dl, get(OpCode), DestReg)
    .addReg(FirstReg).addReg(SecondReg)
    .addReg(Cond[1].getReg(), 0, SubIdx);
}

static unsigned getCRBitValue(unsigned CRBit) {
  unsigned Ret = 4;
  if (CRBit == PPC::CR0LT || CRBit == PPC::CR1LT ||
      CRBit == PPC::CR2LT || CRBit == PPC::CR3LT ||
      CRBit == PPC::CR4LT || CRBit == PPC::CR5LT ||
      CRBit == PPC::CR6LT || CRBit == PPC::CR7LT)
    Ret = 3;
  if (CRBit == PPC::CR0GT || CRBit == PPC::CR1GT ||
      CRBit == PPC::CR2GT || CRBit == PPC::CR3GT ||
      CRBit == PPC::CR4GT || CRBit == PPC::CR5GT ||
      CRBit == PPC::CR6GT || CRBit == PPC::CR7GT)
    Ret = 2;
  if (CRBit == PPC::CR0EQ || CRBit == PPC::CR1EQ ||
      CRBit == PPC::CR2EQ || CRBit == PPC::CR3EQ ||
      CRBit == PPC::CR4EQ || CRBit == PPC::CR5EQ ||
      CRBit == PPC::CR6EQ || CRBit == PPC::CR7EQ)
    Ret = 1;
  if (CRBit == PPC::CR0UN || CRBit == PPC::CR1UN ||
      CRBit == PPC::CR2UN || CRBit == PPC::CR3UN ||
      CRBit == PPC::CR4UN || CRBit == PPC::CR5UN ||
      CRBit == PPC::CR6UN || CRBit == PPC::CR7UN)
    Ret = 0;

  assert(Ret != 4 && "Invalid CR bit register");
  return Ret;
}

void PPCInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I, DebugLoc DL,
                               unsigned DestReg, unsigned SrcReg,
                               bool KillSrc) const {
  // We can end up with self copies and similar things as a result of VSX copy
  // legalization. Promote them here.
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  if (PPC::F8RCRegClass.contains(DestReg) &&
      PPC::VSRCRegClass.contains(SrcReg)) {
    unsigned SuperReg =
      TRI->getMatchingSuperReg(DestReg, PPC::sub_64, &PPC::VSRCRegClass);

    if (VSXSelfCopyCrash && SrcReg == SuperReg)
      llvm_unreachable("nop VSX copy");

    DestReg = SuperReg;
  } else if (PPC::VRRCRegClass.contains(DestReg) &&
             PPC::VSRCRegClass.contains(SrcReg)) {
    unsigned SuperReg =
      TRI->getMatchingSuperReg(DestReg, PPC::sub_128, &PPC::VSRCRegClass);

    if (VSXSelfCopyCrash && SrcReg == SuperReg)
      llvm_unreachable("nop VSX copy");

    DestReg = SuperReg;
  } else if (PPC::F8RCRegClass.contains(SrcReg) &&
             PPC::VSRCRegClass.contains(DestReg)) {
    unsigned SuperReg =
      TRI->getMatchingSuperReg(SrcReg, PPC::sub_64, &PPC::VSRCRegClass);

    if (VSXSelfCopyCrash && DestReg == SuperReg)
      llvm_unreachable("nop VSX copy");

    SrcReg = SuperReg;
  } else if (PPC::VRRCRegClass.contains(SrcReg) &&
             PPC::VSRCRegClass.contains(DestReg)) {
    unsigned SuperReg =
      TRI->getMatchingSuperReg(SrcReg, PPC::sub_128, &PPC::VSRCRegClass);

    if (VSXSelfCopyCrash && DestReg == SuperReg)
      llvm_unreachable("nop VSX copy");

    SrcReg = SuperReg;
  }

  // Different class register copy
  if (PPC::CRBITRCRegClass.contains(SrcReg) &&
      PPC::GPRCRegClass.contains(DestReg)) {
    unsigned CRReg = getCRFromCRBit(SrcReg);
    BuildMI(MBB, I, DL, get(PPC::MFOCRF), DestReg)
       .addReg(CRReg), getKillRegState(KillSrc);
    // Rotate the CR bit in the CR fields to be the least significant bit and
    // then mask with 0x1 (MB = ME = 31).
    BuildMI(MBB, I, DL, get(PPC::RLWINM), DestReg)
       .addReg(DestReg, RegState::Kill)
       .addImm(TRI->getEncodingValue(CRReg) * 4 + (4 - getCRBitValue(SrcReg)))
       .addImm(31)
       .addImm(31);
    return;
  } else if (PPC::CRRCRegClass.contains(SrcReg) &&
      PPC::G8RCRegClass.contains(DestReg)) {
    BuildMI(MBB, I, DL, get(PPC::MFOCRF8), DestReg)
       .addReg(SrcReg), getKillRegState(KillSrc);
    return;
  } else if (PPC::CRRCRegClass.contains(SrcReg) &&
      PPC::GPRCRegClass.contains(DestReg)) {
    BuildMI(MBB, I, DL, get(PPC::MFOCRF), DestReg)
       .addReg(SrcReg), getKillRegState(KillSrc);
    return;
   }

  unsigned Opc;
  if (PPC::GPRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::OR;
  else if (PPC::G8RCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::OR8;
  else if (PPC::F4RCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::FMR;
  else if (PPC::CRRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::MCRF;
  else if (PPC::VRRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::VOR;
  else if (PPC::VSRCRegClass.contains(DestReg, SrcReg))
    // There are two different ways this can be done:
    //   1. xxlor : This has lower latency (on the P7), 2 cycles, but can only
    //      issue in VSU pipeline 0.
    //   2. xmovdp/xmovsp: This has higher latency (on the P7), 6 cycles, but
    //      can go to either pipeline.
    // We'll always use xxlor here, because in practically all cases where
    // copies are generated, they are close enough to some use that the
    // lower-latency form is preferable.
    Opc = PPC::XXLOR;
  else if (PPC::VSFRCRegClass.contains(DestReg, SrcReg) ||
           PPC::VSSRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::XXLORf;
  else if (PPC::QFRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::QVFMR;
  else if (PPC::QSRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::QVFMRs;
  else if (PPC::QBRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::QVFMRb;
  else if (PPC::CRBITRCRegClass.contains(DestReg, SrcReg))
    Opc = PPC::CROR;
  else
    llvm_unreachable("Impossible reg-to-reg copy");

  const MCInstrDesc &MCID = get(Opc);
  if (MCID.getNumOperands() == 3)
    BuildMI(MBB, I, DL, MCID, DestReg)
      .addReg(SrcReg).addReg(SrcReg, getKillRegState(KillSrc));
  else
    BuildMI(MBB, I, DL, MCID, DestReg).addReg(SrcReg, getKillRegState(KillSrc));
}

// This function returns true if a CR spill is necessary and false otherwise.
bool
PPCInstrInfo::StoreRegToStackSlot(MachineFunction &MF,
                                  unsigned SrcReg, bool isKill,
                                  int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  SmallVectorImpl<MachineInstr*> &NewMIs,
                                  bool &NonRI, bool &SpillsVRS) const{
  // Note: If additional store instructions are added here,
  // update isStoreToStackSlot.

  DebugLoc DL;
  if (PPC::GPRCRegClass.hasSubClassEq(RC) ||
      PPC::GPRC_NOR0RegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STW))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (PPC::G8RCRegClass.hasSubClassEq(RC) ||
             PPC::G8RC_NOX0RegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STD))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (PPC::F8RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STFD))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (PPC::F4RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STFS))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
  } else if (PPC::CRRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::SPILL_CR))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    return true;
  } else if (PPC::CRBITRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::SPILL_CRBIT))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    return true;
  } else if (PPC::VRRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STVX))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VSRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STXVD2X))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VSFRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STXSDX))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VSSRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::STXSSPX))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VRSAVERCRegClass.hasSubClassEq(RC)) {
    assert(Subtarget.isDarwin() &&
           "VRSAVE only needs spill/restore on Darwin");
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::SPILL_VRSAVE))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    SpillsVRS = true;
  } else if (PPC::QFRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::QVSTFDX))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::QSRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::QVSTFSXs))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::QBRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::QVSTFDXb))
                                       .addReg(SrcReg,
                                               getKillRegState(isKill)),
                                       FrameIdx));
    NonRI = true;
  } else {
    llvm_unreachable("Unknown regclass!");
  }

  return false;
}

void
PPCInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MI,
                                  unsigned SrcReg, bool isKill, int FrameIdx,
                                  const TargetRegisterClass *RC,
                                  const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr*, 4> NewMIs;

  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();
  FuncInfo->setHasSpills();

  bool NonRI = false, SpillsVRS = false;
  if (StoreRegToStackSlot(MF, SrcReg, isKill, FrameIdx, RC, NewMIs,
                          NonRI, SpillsVRS))
    FuncInfo->setSpillsCR();

  if (SpillsVRS)
    FuncInfo->setSpillsVRSAVE();

  if (NonRI)
    FuncInfo->setHasNonRISpills();

  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);

  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIdx),
      MachineMemOperand::MOStore, MFI.getObjectSize(FrameIdx),
      MFI.getObjectAlignment(FrameIdx));
  NewMIs.back()->addMemOperand(MF, MMO);
}

bool
PPCInstrInfo::LoadRegFromStackSlot(MachineFunction &MF, DebugLoc DL,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   SmallVectorImpl<MachineInstr*> &NewMIs,
                                   bool &NonRI, bool &SpillsVRS) const{
  // Note: If additional load instructions are added here,
  // update isLoadFromStackSlot.

  if (PPC::GPRCRegClass.hasSubClassEq(RC) ||
      PPC::GPRC_NOR0RegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LWZ),
                                               DestReg), FrameIdx));
  } else if (PPC::G8RCRegClass.hasSubClassEq(RC) ||
             PPC::G8RC_NOX0RegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LD), DestReg),
                                       FrameIdx));
  } else if (PPC::F8RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LFD), DestReg),
                                       FrameIdx));
  } else if (PPC::F4RCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LFS), DestReg),
                                       FrameIdx));
  } else if (PPC::CRRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL,
                                               get(PPC::RESTORE_CR), DestReg),
                                       FrameIdx));
    return true;
  } else if (PPC::CRBITRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL,
                                               get(PPC::RESTORE_CRBIT), DestReg),
                                       FrameIdx));
    return true;
  } else if (PPC::VRRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LVX), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VSRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LXVD2X), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VSFRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LXSDX), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VSSRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::LXSSPX), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::VRSAVERCRegClass.hasSubClassEq(RC)) {
    assert(Subtarget.isDarwin() &&
           "VRSAVE only needs spill/restore on Darwin");
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL,
                                               get(PPC::RESTORE_VRSAVE),
                                               DestReg),
                                       FrameIdx));
    SpillsVRS = true;
  } else if (PPC::QFRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::QVLFDX), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::QSRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::QVLFSXs), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else if (PPC::QBRCRegClass.hasSubClassEq(RC)) {
    NewMIs.push_back(addFrameReference(BuildMI(MF, DL, get(PPC::QVLFDXb), DestReg),
                                       FrameIdx));
    NonRI = true;
  } else {
    llvm_unreachable("Unknown regclass!");
  }

  return false;
}

void
PPCInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   unsigned DestReg, int FrameIdx,
                                   const TargetRegisterClass *RC,
                                   const TargetRegisterInfo *TRI) const {
  MachineFunction &MF = *MBB.getParent();
  SmallVector<MachineInstr*, 4> NewMIs;
  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();

  PPCFunctionInfo *FuncInfo = MF.getInfo<PPCFunctionInfo>();
  FuncInfo->setHasSpills();

  bool NonRI = false, SpillsVRS = false;
  if (LoadRegFromStackSlot(MF, DL, DestReg, FrameIdx, RC, NewMIs,
                           NonRI, SpillsVRS))
    FuncInfo->setSpillsCR();

  if (SpillsVRS)
    FuncInfo->setSpillsVRSAVE();

  if (NonRI)
    FuncInfo->setHasNonRISpills();

  for (unsigned i = 0, e = NewMIs.size(); i != e; ++i)
    MBB.insert(MI, NewMIs[i]);

  const MachineFrameInfo &MFI = *MF.getFrameInfo();
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo::getFixedStack(MF, FrameIdx),
      MachineMemOperand::MOLoad, MFI.getObjectSize(FrameIdx),
      MFI.getObjectAlignment(FrameIdx));
  NewMIs.back()->addMemOperand(MF, MMO);
}

bool PPCInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  assert(Cond.size() == 2 && "Invalid PPC branch opcode!");
  if (Cond[1].getReg() == PPC::CTR8 || Cond[1].getReg() == PPC::CTR)
    Cond[0].setImm(Cond[0].getImm() == 0 ? 1 : 0);
  else
    // Leave the CR# the same, but invert the condition.
    Cond[0].setImm(PPC::InvertPredicate((PPC::Predicate)Cond[0].getImm()));
  return false;
}

bool PPCInstrInfo::FoldImmediate(MachineInstr *UseMI, MachineInstr *DefMI,
                             unsigned Reg, MachineRegisterInfo *MRI) const {
  // For some instructions, it is legal to fold ZERO into the RA register field.
  // A zero immediate should always be loaded with a single li.
  unsigned DefOpc = DefMI->getOpcode();
  if (DefOpc != PPC::LI && DefOpc != PPC::LI8)
    return false;
  if (!DefMI->getOperand(1).isImm())
    return false;
  if (DefMI->getOperand(1).getImm() != 0)
    return false;

  // Note that we cannot here invert the arguments of an isel in order to fold
  // a ZERO into what is presented as the second argument. All we have here
  // is the condition bit, and that might come from a CR-logical bit operation.

  const MCInstrDesc &UseMCID = UseMI->getDesc();

  // Only fold into real machine instructions.
  if (UseMCID.isPseudo())
    return false;

  unsigned UseIdx;
  for (UseIdx = 0; UseIdx < UseMI->getNumOperands(); ++UseIdx)
    if (UseMI->getOperand(UseIdx).isReg() &&
        UseMI->getOperand(UseIdx).getReg() == Reg)
      break;

  assert(UseIdx < UseMI->getNumOperands() && "Cannot find Reg in UseMI");
  assert(UseIdx < UseMCID.getNumOperands() && "No operand description for Reg");

  const MCOperandInfo *UseInfo = &UseMCID.OpInfo[UseIdx];

  // We can fold the zero if this register requires a GPRC_NOR0/G8RC_NOX0
  // register (which might also be specified as a pointer class kind).
  if (UseInfo->isLookupPtrRegClass()) {
    if (UseInfo->RegClass /* Kind */ != 1)
      return false;
  } else {
    if (UseInfo->RegClass != PPC::GPRC_NOR0RegClassID &&
        UseInfo->RegClass != PPC::G8RC_NOX0RegClassID)
      return false;
  }

  // Make sure this is not tied to an output register (or otherwise
  // constrained). This is true for ST?UX registers, for example, which
  // are tied to their output registers.
  if (UseInfo->Constraints != 0)
    return false;

  unsigned ZeroReg;
  if (UseInfo->isLookupPtrRegClass()) {
    bool isPPC64 = Subtarget.isPPC64();
    ZeroReg = isPPC64 ? PPC::ZERO8 : PPC::ZERO;
  } else {
    ZeroReg = UseInfo->RegClass == PPC::G8RC_NOX0RegClassID ?
              PPC::ZERO8 : PPC::ZERO;
  }

  bool DeleteDef = MRI->hasOneNonDBGUse(Reg);
  UseMI->getOperand(UseIdx).setReg(ZeroReg);

  if (DeleteDef)
    DefMI->eraseFromParent();

  return true;
}

static bool MBBDefinesCTR(MachineBasicBlock &MBB) {
  for (MachineBasicBlock::iterator I = MBB.begin(), IE = MBB.end();
       I != IE; ++I)
    if (I->definesRegister(PPC::CTR) || I->definesRegister(PPC::CTR8))
      return true;
  return false;
}

// We should make sure that, if we're going to predicate both sides of a
// condition (a diamond), that both sides don't define the counter register. We
// can predicate counter-decrement-based branches, but while that predicates
// the branching, it does not predicate the counter decrement. If we tried to
// merge the triangle into one predicated block, we'd decrement the counter
// twice.
bool PPCInstrInfo::isProfitableToIfCvt(MachineBasicBlock &TMBB,
                     unsigned NumT, unsigned ExtraT,
                     MachineBasicBlock &FMBB,
                     unsigned NumF, unsigned ExtraF,
                     BranchProbability Probability) const {
  return !(MBBDefinesCTR(TMBB) && MBBDefinesCTR(FMBB));
}


bool PPCInstrInfo::isPredicated(const MachineInstr *MI) const {
  // The predicated branches are identified by their type, not really by the
  // explicit presence of a predicate. Furthermore, some of them can be
  // predicated more than once. Because if conversion won't try to predicate
  // any instruction which already claims to be predicated (by returning true
  // here), always return false. In doing so, we let isPredicable() be the
  // final word on whether not the instruction can be (further) predicated.

  return false;
}

bool PPCInstrInfo::isUnpredicatedTerminator(const MachineInstr *MI) const {
  if (!MI->isTerminator())
    return false;

  // Conditional branch is a special case.
  if (MI->isBranch() && !MI->isBarrier())
    return true;

  return !isPredicated(MI);
}

bool PPCInstrInfo::PredicateInstruction(MachineInstr *MI,
                                        ArrayRef<MachineOperand> Pred) const {
  unsigned OpC = MI->getOpcode();
  if (OpC == PPC::BLR || OpC == PPC::BLR8) {
    if (Pred[1].getReg() == PPC::CTR8 || Pred[1].getReg() == PPC::CTR) {
      bool isPPC64 = Subtarget.isPPC64();
      MI->setDesc(get(Pred[0].getImm() ?
                      (isPPC64 ? PPC::BDNZLR8 : PPC::BDNZLR) :
                      (isPPC64 ? PPC::BDZLR8  : PPC::BDZLR)));
    } else if (Pred[0].getImm() == PPC::PRED_BIT_SET) {
      MI->setDesc(get(PPC::BCLR));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addReg(Pred[1].getReg());
    } else if (Pred[0].getImm() == PPC::PRED_BIT_UNSET) {
      MI->setDesc(get(PPC::BCLRn));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addReg(Pred[1].getReg());
    } else {
      MI->setDesc(get(PPC::BCCLR));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addImm(Pred[0].getImm())
        .addReg(Pred[1].getReg());
    }

    return true;
  } else if (OpC == PPC::B) {
    if (Pred[1].getReg() == PPC::CTR8 || Pred[1].getReg() == PPC::CTR) {
      bool isPPC64 = Subtarget.isPPC64();
      MI->setDesc(get(Pred[0].getImm() ?
                      (isPPC64 ? PPC::BDNZ8 : PPC::BDNZ) :
                      (isPPC64 ? PPC::BDZ8  : PPC::BDZ)));
    } else if (Pred[0].getImm() == PPC::PRED_BIT_SET) {
      MachineBasicBlock *MBB = MI->getOperand(0).getMBB();
      MI->RemoveOperand(0);

      MI->setDesc(get(PPC::BC));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addReg(Pred[1].getReg())
        .addMBB(MBB);
    } else if (Pred[0].getImm() == PPC::PRED_BIT_UNSET) {
      MachineBasicBlock *MBB = MI->getOperand(0).getMBB();
      MI->RemoveOperand(0);

      MI->setDesc(get(PPC::BCn));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addReg(Pred[1].getReg())
        .addMBB(MBB);
    } else {
      MachineBasicBlock *MBB = MI->getOperand(0).getMBB();
      MI->RemoveOperand(0);

      MI->setDesc(get(PPC::BCC));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addImm(Pred[0].getImm())
        .addReg(Pred[1].getReg())
        .addMBB(MBB);
    }

    return true;
  } else if (OpC == PPC::BCTR  || OpC == PPC::BCTR8 ||
             OpC == PPC::BCTRL || OpC == PPC::BCTRL8) {
    if (Pred[1].getReg() == PPC::CTR8 || Pred[1].getReg() == PPC::CTR)
      llvm_unreachable("Cannot predicate bctr[l] on the ctr register");

    bool setLR = OpC == PPC::BCTRL || OpC == PPC::BCTRL8;
    bool isPPC64 = Subtarget.isPPC64();

    if (Pred[0].getImm() == PPC::PRED_BIT_SET) {
      MI->setDesc(get(isPPC64 ? (setLR ? PPC::BCCTRL8 : PPC::BCCTR8) :
                                (setLR ? PPC::BCCTRL  : PPC::BCCTR)));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addReg(Pred[1].getReg());
      return true;
    } else if (Pred[0].getImm() == PPC::PRED_BIT_UNSET) {
      MI->setDesc(get(isPPC64 ? (setLR ? PPC::BCCTRL8n : PPC::BCCTR8n) :
                                (setLR ? PPC::BCCTRLn  : PPC::BCCTRn)));
      MachineInstrBuilder(*MI->getParent()->getParent(), MI)
        .addReg(Pred[1].getReg());
      return true;
    }

    MI->setDesc(get(isPPC64 ? (setLR ? PPC::BCCCTRL8 : PPC::BCCCTR8) :
                              (setLR ? PPC::BCCCTRL  : PPC::BCCCTR)));
    MachineInstrBuilder(*MI->getParent()->getParent(), MI)
      .addImm(Pred[0].getImm())
      .addReg(Pred[1].getReg());
    return true;
  }

  return false;
}

bool PPCInstrInfo::SubsumesPredicate(ArrayRef<MachineOperand> Pred1,
                                     ArrayRef<MachineOperand> Pred2) const {
  assert(Pred1.size() == 2 && "Invalid PPC first predicate");
  assert(Pred2.size() == 2 && "Invalid PPC second predicate");

  if (Pred1[1].getReg() == PPC::CTR8 || Pred1[1].getReg() == PPC::CTR)
    return false;
  if (Pred2[1].getReg() == PPC::CTR8 || Pred2[1].getReg() == PPC::CTR)
    return false;

  // P1 can only subsume P2 if they test the same condition register.
  if (Pred1[1].getReg() != Pred2[1].getReg())
    return false;

  PPC::Predicate P1 = (PPC::Predicate) Pred1[0].getImm();
  PPC::Predicate P2 = (PPC::Predicate) Pred2[0].getImm();

  if (P1 == P2)
    return true;

  // Does P1 subsume P2, e.g. GE subsumes GT.
  if (P1 == PPC::PRED_LE &&
      (P2 == PPC::PRED_LT || P2 == PPC::PRED_EQ))
    return true;
  if (P1 == PPC::PRED_GE &&
      (P2 == PPC::PRED_GT || P2 == PPC::PRED_EQ))
    return true;

  return false;
}

bool PPCInstrInfo::DefinesPredicate(MachineInstr *MI,
                                    std::vector<MachineOperand> &Pred) const {
  // Note: At the present time, the contents of Pred from this function is
  // unused by IfConversion. This implementation follows ARM by pushing the
  // CR-defining operand. Because the 'DZ' and 'DNZ' count as types of
  // predicate, instructions defining CTR or CTR8 are also included as
  // predicate-defining instructions.

  const TargetRegisterClass *RCs[] =
    { &PPC::CRRCRegClass, &PPC::CRBITRCRegClass,
      &PPC::CTRRCRegClass, &PPC::CTRRC8RegClass };

  bool Found = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    for (unsigned c = 0; c < array_lengthof(RCs) && !Found; ++c) {
      const TargetRegisterClass *RC = RCs[c];
      if (MO.isReg()) {
        if (MO.isDef() && RC->contains(MO.getReg())) {
          Pred.push_back(MO);
          Found = true;
        }
      } else if (MO.isRegMask()) {
        for (TargetRegisterClass::iterator I = RC->begin(),
             IE = RC->end(); I != IE; ++I)
          if (MO.clobbersPhysReg(*I)) {
            Pred.push_back(MO);
            Found = true;
          }
      }
    }
  }

  return Found;
}

bool PPCInstrInfo::isPredicable(MachineInstr *MI) const {
  unsigned OpC = MI->getOpcode();
  switch (OpC) {
  default:
    return false;
  case PPC::B:
  case PPC::BLR:
  case PPC::BLR8:
  case PPC::BCTR:
  case PPC::BCTR8:
  case PPC::BCTRL:
  case PPC::BCTRL8:
    return true;
  }
}

bool PPCInstrInfo::analyzeCompare(const MachineInstr *MI,
                                  unsigned &SrcReg, unsigned &SrcReg2,
                                  int &Mask, int &Value) const {
  unsigned Opc = MI->getOpcode();

  switch (Opc) {
  default: return false;
  case PPC::CMPWI:
  case PPC::CMPLWI:
  case PPC::CMPDI:
  case PPC::CMPLDI:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = 0;
    Value = MI->getOperand(2).getImm();
    Mask = 0xFFFF;
    return true;
  case PPC::CMPW:
  case PPC::CMPLW:
  case PPC::CMPD:
  case PPC::CMPLD:
  case PPC::FCMPUS:
  case PPC::FCMPUD:
    SrcReg = MI->getOperand(1).getReg();
    SrcReg2 = MI->getOperand(2).getReg();
    return true;
  }
}

bool PPCInstrInfo::optimizeCompareInstr(MachineInstr *CmpInstr,
                                        unsigned SrcReg, unsigned SrcReg2,
                                        int Mask, int Value,
                                        const MachineRegisterInfo *MRI) const {
  if (DisableCmpOpt)
    return false;

  int OpC = CmpInstr->getOpcode();
  unsigned CRReg = CmpInstr->getOperand(0).getReg();

  // FP record forms set CR1 based on the execption status bits, not a
  // comparison with zero.
  if (OpC == PPC::FCMPUS || OpC == PPC::FCMPUD)
    return false;

  // The record forms set the condition register based on a signed comparison
  // with zero (so says the ISA manual). This is not as straightforward as it
  // seems, however, because this is always a 64-bit comparison on PPC64, even
  // for instructions that are 32-bit in nature (like slw for example).
  // So, on PPC32, for unsigned comparisons, we can use the record forms only
  // for equality checks (as those don't depend on the sign). On PPC64,
  // we are restricted to equality for unsigned 64-bit comparisons and for
  // signed 32-bit comparisons the applicability is more restricted.
  bool isPPC64 = Subtarget.isPPC64();
  bool is32BitSignedCompare   = OpC ==  PPC::CMPWI || OpC == PPC::CMPW;
  bool is32BitUnsignedCompare = OpC == PPC::CMPLWI || OpC == PPC::CMPLW;
  bool is64BitUnsignedCompare = OpC == PPC::CMPLDI || OpC == PPC::CMPLD;

  // Get the unique definition of SrcReg.
  MachineInstr *MI = MRI->getUniqueVRegDef(SrcReg);
  if (!MI) return false;
  int MIOpC = MI->getOpcode();

  bool equalityOnly = false;
  bool noSub = false;
  if (isPPC64) {
    if (is32BitSignedCompare) {
      // We can perform this optimization only if MI is sign-extending.
      if (MIOpC == PPC::SRAW  || MIOpC == PPC::SRAWo ||
          MIOpC == PPC::SRAWI || MIOpC == PPC::SRAWIo ||
          MIOpC == PPC::EXTSB || MIOpC == PPC::EXTSBo ||
          MIOpC == PPC::EXTSH || MIOpC == PPC::EXTSHo ||
          MIOpC == PPC::EXTSW || MIOpC == PPC::EXTSWo) {
        noSub = true;
      } else
        return false;
    } else if (is32BitUnsignedCompare) {
      // We can perform this optimization, equality only, if MI is
      // zero-extending.
      if (MIOpC == PPC::CNTLZW || MIOpC == PPC::CNTLZWo ||
          MIOpC == PPC::SLW    || MIOpC == PPC::SLWo ||
          MIOpC == PPC::SRW    || MIOpC == PPC::SRWo) {
        noSub = true;
        equalityOnly = true;
      } else
        return false;
    } else
      equalityOnly = is64BitUnsignedCompare;
  } else
    equalityOnly = is32BitUnsignedCompare;

  if (equalityOnly) {
    // We need to check the uses of the condition register in order to reject
    // non-equality comparisons.
    for (MachineRegisterInfo::use_instr_iterator I =MRI->use_instr_begin(CRReg),
         IE = MRI->use_instr_end(); I != IE; ++I) {
      MachineInstr *UseMI = &*I;
      if (UseMI->getOpcode() == PPC::BCC) {
        unsigned Pred = UseMI->getOperand(0).getImm();
        if (Pred != PPC::PRED_EQ && Pred != PPC::PRED_NE)
          return false;
      } else if (UseMI->getOpcode() == PPC::ISEL ||
                 UseMI->getOpcode() == PPC::ISEL8) {
        unsigned SubIdx = UseMI->getOperand(3).getSubReg();
        if (SubIdx != PPC::sub_eq)
          return false;
      } else
        return false;
    }
  }

  MachineBasicBlock::iterator I = CmpInstr;

  // Scan forward to find the first use of the compare.
  for (MachineBasicBlock::iterator EL = CmpInstr->getParent()->end();
       I != EL; ++I) {
    bool FoundUse = false;
    for (MachineRegisterInfo::use_instr_iterator J =MRI->use_instr_begin(CRReg),
         JE = MRI->use_instr_end(); J != JE; ++J)
      if (&*J == &*I) {
        FoundUse = true;
        break;
      }

    if (FoundUse)
      break;
  }

  // There are two possible candidates which can be changed to set CR[01].
  // One is MI, the other is a SUB instruction.
  // For CMPrr(r1,r2), we are looking for SUB(r1,r2) or SUB(r2,r1).
  MachineInstr *Sub = nullptr;
  if (SrcReg2 != 0)
    // MI is not a candidate for CMPrr.
    MI = nullptr;
  // FIXME: Conservatively refuse to convert an instruction which isn't in the
  // same BB as the comparison. This is to allow the check below to avoid calls
  // (and other explicit clobbers); instead we should really check for these
  // more explicitly (in at least a few predecessors).
  else if (MI->getParent() != CmpInstr->getParent() || Value != 0) {
    // PPC does not have a record-form SUBri.
    return false;
  }

  // Search for Sub.
  const TargetRegisterInfo *TRI = &getRegisterInfo();
  --I;

  // Get ready to iterate backward from CmpInstr.
  MachineBasicBlock::iterator E = MI,
                              B = CmpInstr->getParent()->begin();

  for (; I != E && !noSub; --I) {
    const MachineInstr &Instr = *I;
    unsigned IOpC = Instr.getOpcode();

    if (&*I != CmpInstr && (
        Instr.modifiesRegister(PPC::CR0, TRI) ||
        Instr.readsRegister(PPC::CR0, TRI)))
      // This instruction modifies or uses the record condition register after
      // the one we want to change. While we could do this transformation, it
      // would likely not be profitable. This transformation removes one
      // instruction, and so even forcing RA to generate one move probably
      // makes it unprofitable.
      return false;

    // Check whether CmpInstr can be made redundant by the current instruction.
    if ((OpC == PPC::CMPW || OpC == PPC::CMPLW ||
         OpC == PPC::CMPD || OpC == PPC::CMPLD) &&
        (IOpC == PPC::SUBF || IOpC == PPC::SUBF8) &&
        ((Instr.getOperand(1).getReg() == SrcReg &&
          Instr.getOperand(2).getReg() == SrcReg2) ||
        (Instr.getOperand(1).getReg() == SrcReg2 &&
         Instr.getOperand(2).getReg() == SrcReg))) {
      Sub = &*I;
      break;
    }

    if (I == B)
      // The 'and' is below the comparison instruction.
      return false;
  }

  // Return false if no candidates exist.
  if (!MI && !Sub)
    return false;

  // The single candidate is called MI.
  if (!MI) MI = Sub;

  int NewOpC = -1;
  MIOpC = MI->getOpcode();
  if (MIOpC == PPC::ANDIo || MIOpC == PPC::ANDIo8)
    NewOpC = MIOpC;
  else {
    NewOpC = PPC::getRecordFormOpcode(MIOpC);
    if (NewOpC == -1 && PPC::getNonRecordFormOpcode(MIOpC) != -1)
      NewOpC = MIOpC;
  }

  // FIXME: On the non-embedded POWER architectures, only some of the record
  // forms are fast, and we should use only the fast ones.

  // The defining instruction has a record form (or is already a record
  // form). It is possible, however, that we'll need to reverse the condition
  // code of the users.
  if (NewOpC == -1)
    return false;

  SmallVector<std::pair<MachineOperand*, PPC::Predicate>, 4> PredsToUpdate;
  SmallVector<std::pair<MachineOperand*, unsigned>, 4> SubRegsToUpdate;

  // If we have SUB(r1, r2) and CMP(r2, r1), the condition code based on CMP
  // needs to be updated to be based on SUB.  Push the condition code
  // operands to OperandsToUpdate.  If it is safe to remove CmpInstr, the
  // condition code of these operands will be modified.
  bool ShouldSwap = false;
  if (Sub) {
    ShouldSwap = SrcReg2 != 0 && Sub->getOperand(1).getReg() == SrcReg2 &&
      Sub->getOperand(2).getReg() == SrcReg;

    // The operands to subf are the opposite of sub, so only in the fixed-point
    // case, invert the order.
    ShouldSwap = !ShouldSwap;
  }

  if (ShouldSwap)
    for (MachineRegisterInfo::use_instr_iterator
         I = MRI->use_instr_begin(CRReg), IE = MRI->use_instr_end();
         I != IE; ++I) {
      MachineInstr *UseMI = &*I;
      if (UseMI->getOpcode() == PPC::BCC) {
        PPC::Predicate Pred = (PPC::Predicate) UseMI->getOperand(0).getImm();
        assert((!equalityOnly ||
                Pred == PPC::PRED_EQ || Pred == PPC::PRED_NE) &&
               "Invalid predicate for equality-only optimization");
        PredsToUpdate.push_back(std::make_pair(&(UseMI->getOperand(0)),
                                PPC::getSwappedPredicate(Pred)));
      } else if (UseMI->getOpcode() == PPC::ISEL ||
                 UseMI->getOpcode() == PPC::ISEL8) {
        unsigned NewSubReg = UseMI->getOperand(3).getSubReg();
        assert((!equalityOnly || NewSubReg == PPC::sub_eq) &&
               "Invalid CR bit for equality-only optimization");

        if (NewSubReg == PPC::sub_lt)
          NewSubReg = PPC::sub_gt;
        else if (NewSubReg == PPC::sub_gt)
          NewSubReg = PPC::sub_lt;

        SubRegsToUpdate.push_back(std::make_pair(&(UseMI->getOperand(3)),
                                                 NewSubReg));
      } else // We need to abort on a user we don't understand.
        return false;
    }

  // Create a new virtual register to hold the value of the CR set by the
  // record-form instruction. If the instruction was not previously in
  // record form, then set the kill flag on the CR.
  CmpInstr->eraseFromParent();

  MachineBasicBlock::iterator MII = MI;
  BuildMI(*MI->getParent(), std::next(MII), MI->getDebugLoc(),
          get(TargetOpcode::COPY), CRReg)
    .addReg(PPC::CR0, MIOpC != NewOpC ? RegState::Kill : 0);

  if (MIOpC != NewOpC) {
    // We need to be careful here: we're replacing one instruction with
    // another, and we need to make sure that we get all of the right
    // implicit uses and defs. On the other hand, the caller may be holding
    // an iterator to this instruction, and so we can't delete it (this is
    // specifically the case if this is the instruction directly after the
    // compare).

    const MCInstrDesc &NewDesc = get(NewOpC);
    MI->setDesc(NewDesc);

    if (NewDesc.ImplicitDefs)
      for (const MCPhysReg *ImpDefs = NewDesc.getImplicitDefs();
           *ImpDefs; ++ImpDefs)
        if (!MI->definesRegister(*ImpDefs))
          MI->addOperand(*MI->getParent()->getParent(),
                         MachineOperand::CreateReg(*ImpDefs, true, true));
    if (NewDesc.ImplicitUses)
      for (const MCPhysReg *ImpUses = NewDesc.getImplicitUses();
           *ImpUses; ++ImpUses)
        if (!MI->readsRegister(*ImpUses))
          MI->addOperand(*MI->getParent()->getParent(),
                         MachineOperand::CreateReg(*ImpUses, false, true));
  }

  // Modify the condition code of operands in OperandsToUpdate.
  // Since we have SUB(r1, r2) and CMP(r2, r1), the condition code needs to
  // be changed from r2 > r1 to r1 < r2, from r2 < r1 to r1 > r2, etc.
  for (unsigned i = 0, e = PredsToUpdate.size(); i < e; i++)
    PredsToUpdate[i].first->setImm(PredsToUpdate[i].second);

  for (unsigned i = 0, e = SubRegsToUpdate.size(); i < e; i++)
    SubRegsToUpdate[i].first->setSubReg(SubRegsToUpdate[i].second);

  return true;
}

/// GetInstSize - Return the number of bytes of code the specified
/// instruction may be.  This returns the maximum number of bytes.
///
unsigned PPCInstrInfo::GetInstSizeInBytes(const MachineInstr *MI) const {
  unsigned Opcode = MI->getOpcode();

  if (Opcode == PPC::INLINEASM) {
    const MachineFunction *MF = MI->getParent()->getParent();
    const char *AsmStr = MI->getOperand(0).getSymbolName();
    return getInlineAsmLength(AsmStr, *MF->getTarget().getMCAsmInfo());
  } else if (Opcode == TargetOpcode::STACKMAP) {
    return MI->getOperand(1).getImm();
  } else if (Opcode == TargetOpcode::PATCHPOINT) {
    PatchPointOpers Opers(MI);
    return Opers.getMetaOper(PatchPointOpers::NBytesPos).getImm();
  } else {
    const MCInstrDesc &Desc = get(Opcode);
    return Desc.getSize();
  }
}

std::pair<unsigned, unsigned>
PPCInstrInfo::decomposeMachineOperandsTargetFlags(unsigned TF) const {
  const unsigned Mask = PPCII::MO_ACCESS_MASK;
  return std::make_pair(TF & Mask, TF & ~Mask);
}

ArrayRef<std::pair<unsigned, const char *>>
PPCInstrInfo::getSerializableDirectMachineOperandTargetFlags() const {
  using namespace PPCII;
  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_LO, "ppc-lo"},
      {MO_HA, "ppc-ha"},
      {MO_TPREL_LO, "ppc-tprel-lo"},
      {MO_TPREL_HA, "ppc-tprel-ha"},
      {MO_DTPREL_LO, "ppc-dtprel-lo"},
      {MO_TLSLD_LO, "ppc-tlsld-lo"},
      {MO_TOC_LO, "ppc-toc-lo"},
      {MO_TLS, "ppc-tls"}};
  return makeArrayRef(TargetFlags);
}

ArrayRef<std::pair<unsigned, const char *>>
PPCInstrInfo::getSerializableBitmaskMachineOperandTargetFlags() const {
  using namespace PPCII;
  static const std::pair<unsigned, const char *> TargetFlags[] = {
      {MO_PLT_OR_STUB, "ppc-plt-or-stub"},
      {MO_PIC_FLAG, "ppc-pic"},
      {MO_NLP_FLAG, "ppc-nlp"},
      {MO_NLP_HIDDEN_FLAG, "ppc-nlp-hidden"}};
  return makeArrayRef(TargetFlags);
}

