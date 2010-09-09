//===-- Thumb2SizeReduction.cpp - Thumb2 code size reduction pass -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "t2-reduce-size"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMBaseInstrInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumNarrows,  "Number of 32-bit instrs reduced to 16-bit ones");
STATISTIC(Num2Addrs,   "Number of 32-bit instrs reduced to 2addr 16-bit ones");
STATISTIC(NumLdSts,    "Number of 32-bit load / store reduced to 16-bit ones");

static cl::opt<int> ReduceLimit("t2-reduce-limit",
                                cl::init(-1), cl::Hidden);
static cl::opt<int> ReduceLimit2Addr("t2-reduce-limit2",
                                     cl::init(-1), cl::Hidden);
static cl::opt<int> ReduceLimitLdSt("t2-reduce-limit3",
                                     cl::init(-1), cl::Hidden);

namespace {
  /// ReduceTable - A static table with information on mapping from wide
  /// opcodes to narrow
  struct ReduceEntry {
    unsigned WideOpc;      // Wide opcode
    unsigned NarrowOpc1;   // Narrow opcode to transform to
    unsigned NarrowOpc2;   // Narrow opcode when it's two-address
    uint8_t  Imm1Limit;    // Limit of immediate field (bits)
    uint8_t  Imm2Limit;    // Limit of immediate field when it's two-address
    unsigned LowRegs1 : 1; // Only possible if low-registers are used
    unsigned LowRegs2 : 1; // Only possible if low-registers are used (2addr)
    unsigned PredCC1  : 2; // 0 - If predicated, cc is on and vice versa.
                           // 1 - No cc field.
                           // 2 - Always set CPSR.
    unsigned PredCC2  : 2;
    unsigned Special  : 1; // Needs to be dealt with specially
  };

  static const ReduceEntry ReduceTable[] = {
    // Wide,        Narrow1,      Narrow2,     imm1,imm2,  lo1, lo2, P/C, S
    { ARM::t2ADCrr, 0,            ARM::tADC,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2ADDri, ARM::tADDi3,  ARM::tADDi8,   3,   8,    1,   1,  0,0, 0 },
    { ARM::t2ADDrr, ARM::tADDrr,  ARM::tADDhirr, 0,   0,    1,   0,  0,1, 0 },
    // Note: immediate scale is 4.
    { ARM::t2ADDrSPi,ARM::tADDrSPi,0,            8,   0,    1,   0,  1,0, 0 },
    { ARM::t2ADDSri,ARM::tADDi3,  ARM::tADDi8,   3,   8,    1,   1,  2,2, 1 },
    { ARM::t2ADDSrr,ARM::tADDrr,  0,             0,   0,    1,   0,  2,0, 1 },
    { ARM::t2ANDrr, 0,            ARM::tAND,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2ASRri, ARM::tASRri,  0,             5,   0,    1,   0,  0,0, 0 },
    { ARM::t2ASRrr, 0,            ARM::tASRrr,   0,   0,    0,   1,  0,0, 0 },
    { ARM::t2BICrr, 0,            ARM::tBIC,     0,   0,    0,   1,  0,0, 0 },
    //FIXME: Disable CMN, as CCodes are backwards from compare expectations
    //{ ARM::t2CMNrr, ARM::tCMN,    0,             0,   0,    1,   0,  2,0, 0 },
    { ARM::t2CMPri, ARM::tCMPi8,  0,             8,   0,    1,   0,  2,0, 0 },
    { ARM::t2CMPrr, ARM::tCMPhir, 0,             0,   0,    0,   0,  2,0, 0 },
    { ARM::t2CMPzri,ARM::tCMPzi8, 0,             8,   0,    1,   0,  2,0, 0 },
    { ARM::t2CMPzrr,ARM::tCMPzhir,0,             0,   0,    0,   0,  2,0, 0 },
    { ARM::t2EORrr, 0,            ARM::tEOR,     0,   0,    0,   1,  0,0, 0 },
    // FIXME: adr.n immediate offset must be multiple of 4.
    //{ ARM::t2LEApcrelJT,ARM::tLEApcrelJT, 0,     0,   0,    1,   0,  1,0, 0 },
    { ARM::t2LSLri, ARM::tLSLri,  0,             5,   0,    1,   0,  0,0, 0 },
    { ARM::t2LSLrr, 0,            ARM::tLSLrr,   0,   0,    0,   1,  0,0, 0 },
    { ARM::t2LSRri, ARM::tLSRri,  0,             5,   0,    1,   0,  0,0, 0 },
    { ARM::t2LSRrr, 0,            ARM::tLSRrr,   0,   0,    0,   1,  0,0, 0 },
    { ARM::t2MOVi,  ARM::tMOVi8,  0,             8,   0,    1,   0,  0,0, 0 },
    { ARM::t2MOVi16,ARM::tMOVi8,  0,             8,   0,    1,   0,  0,0, 1 },
    // FIXME: Do we need the 16-bit 'S' variant?
    { ARM::t2MOVr,ARM::tMOVgpr2gpr,0,            0,   0,    0,   0,  1,0, 0 },
    { ARM::t2MOVCCr,0,            ARM::tMOVCCr,  0,   0,    0,   0,  0,1, 0 },
    { ARM::t2MOVCCi,0,            ARM::tMOVCCi,  0,   8,    0,   1,  0,1, 0 },
    { ARM::t2MUL,   0,            ARM::tMUL,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2MVNr,  ARM::tMVN,    0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2ORRrr, 0,            ARM::tORR,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2REV,   ARM::tREV,    0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2REV16, ARM::tREV16,  0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2REVSH, ARM::tREVSH,  0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2RORrr, 0,            ARM::tROR,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2RSBri, ARM::tRSB,    0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2RSBSri,ARM::tRSB,    0,             0,   0,    1,   0,  2,0, 1 },
    { ARM::t2SBCrr, 0,            ARM::tSBC,     0,   0,    0,   1,  0,0, 0 },
    { ARM::t2SUBri, ARM::tSUBi3,  ARM::tSUBi8,   3,   8,    1,   1,  0,0, 0 },
    { ARM::t2SUBrr, ARM::tSUBrr,  0,             0,   0,    1,   0,  0,0, 0 },
    { ARM::t2SUBSri,ARM::tSUBi3,  ARM::tSUBi8,   3,   8,    1,   1,  2,2, 0 },
    { ARM::t2SUBSrr,ARM::tSUBrr,  0,             0,   0,    1,   0,  2,0, 0 },
    { ARM::t2SXTBr, ARM::tSXTB,   0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2SXTHr, ARM::tSXTH,   0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2TSTrr, ARM::tTST,    0,             0,   0,    1,   0,  2,0, 0 },
    { ARM::t2UXTBr, ARM::tUXTB,   0,             0,   0,    1,   0,  1,0, 0 },
    { ARM::t2UXTHr, ARM::tUXTH,   0,             0,   0,    1,   0,  1,0, 0 },

    // FIXME: Clean this up after splitting each Thumb load / store opcode
    // into multiple ones.
    { ARM::t2LDRi12,ARM::tLDR,    ARM::tLDRspi,  5,   8,    1,   0,  0,0, 1 },
    { ARM::t2LDRs,  ARM::tLDR,    0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2LDRBi12,ARM::tLDRB,  0,             5,   0,    1,   0,  0,0, 1 },
    { ARM::t2LDRBs, ARM::tLDRB,   0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2LDRHi12,ARM::tLDRH,  0,             5,   0,    1,   0,  0,0, 1 },
    { ARM::t2LDRHs, ARM::tLDRH,   0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2LDRSBs,ARM::tLDRSB,  0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2LDRSHs,ARM::tLDRSH,  0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2STRi12,ARM::tSTR,    ARM::tSTRspi,  5,   8,    1,   0,  0,0, 1 },
    { ARM::t2STRs,  ARM::tSTR,    0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2STRBi12,ARM::tSTRB,  0,             5,   0,    1,   0,  0,0, 1 },
    { ARM::t2STRBs, ARM::tSTRB,   0,             0,   0,    1,   0,  0,0, 1 },
    { ARM::t2STRHi12,ARM::tSTRH,  0,             5,   0,    1,   0,  0,0, 1 },
    { ARM::t2STRHs, ARM::tSTRH,   0,             0,   0,    1,   0,  0,0, 1 },

    { ARM::t2LDM,   ARM::tLDM,    0,             0,   0,    1,   1,  1,1, 1 },
    { ARM::t2LDM_RET,0,           ARM::tPOP_RET, 0,   0,    1,   1,  1,1, 1 },
    { ARM::t2LDM_UPD,ARM::tLDM_UPD,ARM::tPOP,    0,   0,    1,   1,  1,1, 1 },
    // ARM::t2STM (with no basereg writeback) has no Thumb1 equivalent
    { ARM::t2STM_UPD,ARM::tSTM_UPD,ARM::tPUSH,   0,   0,    1,   1,  1,1, 1 },
  };

  class Thumb2SizeReduce : public MachineFunctionPass {
  public:
    static char ID;
    Thumb2SizeReduce();

    const Thumb2InstrInfo *TII;

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "Thumb2 instruction size reduction pass";
    }

  private:
    /// ReduceOpcodeMap - Maps wide opcode to index of entry in ReduceTable.
    DenseMap<unsigned, unsigned> ReduceOpcodeMap;

    bool VerifyPredAndCC(MachineInstr *MI, const ReduceEntry &Entry,
                         bool is2Addr, ARMCC::CondCodes Pred,
                         bool LiveCPSR, bool &HasCC, bool &CCDead);

    bool ReduceLoadStore(MachineBasicBlock &MBB, MachineInstr *MI,
                         const ReduceEntry &Entry);

    bool ReduceSpecial(MachineBasicBlock &MBB, MachineInstr *MI,
                       const ReduceEntry &Entry, bool LiveCPSR);

    /// ReduceTo2Addr - Reduce a 32-bit instruction to a 16-bit two-address
    /// instruction.
    bool ReduceTo2Addr(MachineBasicBlock &MBB, MachineInstr *MI,
                       const ReduceEntry &Entry,
                       bool LiveCPSR);

    /// ReduceToNarrow - Reduce a 32-bit instruction to a 16-bit
    /// non-two-address instruction.
    bool ReduceToNarrow(MachineBasicBlock &MBB, MachineInstr *MI,
                        const ReduceEntry &Entry,
                        bool LiveCPSR);

    /// ReduceMBB - Reduce width of instructions in the specified basic block.
    bool ReduceMBB(MachineBasicBlock &MBB);
  };
  char Thumb2SizeReduce::ID = 0;
}

Thumb2SizeReduce::Thumb2SizeReduce() : MachineFunctionPass(ID) {
  for (unsigned i = 0, e = array_lengthof(ReduceTable); i != e; ++i) {
    unsigned FromOpc = ReduceTable[i].WideOpc;
    if (!ReduceOpcodeMap.insert(std::make_pair(FromOpc, i)).second)
      assert(false && "Duplicated entries?");
  }
}

static bool HasImplicitCPSRDef(const TargetInstrDesc &TID) {
  for (const unsigned *Regs = TID.ImplicitDefs; *Regs; ++Regs)
    if (*Regs == ARM::CPSR)
      return true;
  return false;
}

bool
Thumb2SizeReduce::VerifyPredAndCC(MachineInstr *MI, const ReduceEntry &Entry,
                                  bool is2Addr, ARMCC::CondCodes Pred,
                                  bool LiveCPSR, bool &HasCC, bool &CCDead) {
  if ((is2Addr  && Entry.PredCC2 == 0) ||
      (!is2Addr && Entry.PredCC1 == 0)) {
    if (Pred == ARMCC::AL) {
      // Not predicated, must set CPSR.
      if (!HasCC) {
        // Original instruction was not setting CPSR, but CPSR is not
        // currently live anyway. It's ok to set it. The CPSR def is
        // dead though.
        if (!LiveCPSR) {
          HasCC = true;
          CCDead = true;
          return true;
        }
        return false;
      }
    } else {
      // Predicated, must not set CPSR.
      if (HasCC)
        return false;
    }
  } else if ((is2Addr  && Entry.PredCC2 == 2) ||
             (!is2Addr && Entry.PredCC1 == 2)) {
    /// Old opcode has an optional def of CPSR.
    if (HasCC)
      return true;
    // If both old opcode does not implicit CPSR def, then it's not ok since
    // these new opcodes CPSR def is not meant to be thrown away. e.g. CMP.
    if (!HasImplicitCPSRDef(MI->getDesc()))
      return false;
    HasCC = true;
  } else {
    // 16-bit instruction does not set CPSR.
    if (HasCC)
      return false;
  }

  return true;
}

static bool VerifyLowRegs(MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();
  bool isPCOk = (Opc == ARM::t2LDM_RET || Opc == ARM::t2LDM ||
                 Opc == ARM::t2LDM_UPD);
  bool isLROk = (Opc == ARM::t2STM_UPD);
  bool isSPOk = isPCOk || isLROk || (Opc == ARM::t2ADDrSPi);
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isImplicit())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || Reg == ARM::CPSR)
      continue;
    if (isPCOk && Reg == ARM::PC)
      continue;
    if (isLROk && Reg == ARM::LR)
      continue;
    if (Reg == ARM::SP) {
      if (isSPOk)
        continue;
      if (i == 1 && (Opc == ARM::t2LDRi12 || Opc == ARM::t2STRi12))
        // Special case for these ldr / str with sp as base register.
        continue;
    }
    if (!isARMLowRegister(Reg))
      return false;
  }
  return true;
}

bool
Thumb2SizeReduce::ReduceLoadStore(MachineBasicBlock &MBB, MachineInstr *MI,
                                  const ReduceEntry &Entry) {
  if (ReduceLimitLdSt != -1 && ((int)NumLdSts >= ReduceLimitLdSt))
    return false;

  unsigned Scale = 1;
  bool HasImmOffset = false;
  bool HasShift = false;
  bool HasOffReg = true;
  bool isLdStMul = false;
  unsigned Opc = Entry.NarrowOpc1;
  unsigned OpNum = 3; // First 'rest' of operands.
  uint8_t  ImmLimit = Entry.Imm1Limit;
  switch (Entry.WideOpc) {
  default:
    llvm_unreachable("Unexpected Thumb2 load / store opcode!");
  case ARM::t2LDRi12:
  case ARM::t2STRi12: {
    unsigned BaseReg = MI->getOperand(1).getReg();
    if (BaseReg == ARM::SP) {
      Opc = Entry.NarrowOpc2;
      ImmLimit = Entry.Imm2Limit;
      HasOffReg = false;
    }
    Scale = 4;
    HasImmOffset = true;
    break;
  }
  case ARM::t2LDRBi12:
  case ARM::t2STRBi12:
    HasImmOffset = true;
    break;
  case ARM::t2LDRHi12:
  case ARM::t2STRHi12:
    Scale = 2;
    HasImmOffset = true;
    break;
  case ARM::t2LDRs:
  case ARM::t2LDRBs:
  case ARM::t2LDRHs:
  case ARM::t2LDRSBs:
  case ARM::t2LDRSHs:
  case ARM::t2STRs:
  case ARM::t2STRBs:
  case ARM::t2STRHs:
    HasShift = true;
    OpNum = 4;
    break;
  case ARM::t2LDM: {
    unsigned BaseReg = MI->getOperand(0).getReg();
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MI->getOperand(1).getImm());
    if (!isARMLowRegister(BaseReg) || Mode != ARM_AM::ia)
      return false;
    // For the non-writeback version (this one), the base register must be
    // one of the registers being loaded.
    bool isOK = false;
    for (unsigned i = 4; i < MI->getNumOperands(); ++i) {
      if (MI->getOperand(i).getReg() == BaseReg) {
        isOK = true;
        break;
      }
    }
    if (!isOK)
      return false;

    OpNum = 0;
    isLdStMul = true;
    break;
  }
  case ARM::t2LDM_RET: {
    unsigned BaseReg = MI->getOperand(1).getReg();
    if (BaseReg != ARM::SP)
      return false;
    Opc = Entry.NarrowOpc2; // tPOP_RET
    OpNum = 3;
    isLdStMul = true;
    break;
  }
  case ARM::t2LDM_UPD:
  case ARM::t2STM_UPD: {
    OpNum = 0;
    unsigned BaseReg = MI->getOperand(1).getReg();
    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MI->getOperand(2).getImm());
    if (BaseReg == ARM::SP &&
        ((Entry.WideOpc == ARM::t2LDM_UPD && Mode == ARM_AM::ia) ||
         (Entry.WideOpc == ARM::t2STM_UPD && Mode == ARM_AM::db))) {
      Opc = Entry.NarrowOpc2; // tPOP or tPUSH
      OpNum = 3;
    } else if (!isARMLowRegister(BaseReg) || Mode != ARM_AM::ia) {
      return false;
    }
    isLdStMul = true;
    break;
  }
  }

  unsigned OffsetReg = 0;
  bool OffsetKill = false;
  if (HasShift) {
    OffsetReg  = MI->getOperand(2).getReg();
    OffsetKill = MI->getOperand(2).isKill();
    if (MI->getOperand(3).getImm())
      // Thumb1 addressing mode doesn't support shift.
      return false;
  }

  unsigned OffsetImm = 0;
  if (HasImmOffset) {
    OffsetImm = MI->getOperand(2).getImm();
    unsigned MaxOffset = ((1 << ImmLimit) - 1) * Scale;
    if ((OffsetImm & (Scale-1)) || OffsetImm > MaxOffset)
      // Make sure the immediate field fits.
      return false;
  }

  // Add the 16-bit load / store instruction.
  // FIXME: Thumb1 addressing mode encode both immediate and register offset.
  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, *MI, dl, TII->get(Opc));
  if (!isLdStMul) {
    MIB.addOperand(MI->getOperand(0)).addOperand(MI->getOperand(1));
    if (Opc != ARM::tLDRSB && Opc != ARM::tLDRSH) {
      // tLDRSB and tLDRSH do not have an immediate offset field. On the other
      // hand, it must have an offset register.
      // FIXME: Remove this special case.
      MIB.addImm(OffsetImm/Scale);
    }
    assert((!HasShift || OffsetReg) && "Invalid so_reg load / store address!");

    if (HasOffReg)
      MIB.addReg(OffsetReg, getKillRegState(OffsetKill));
  }

  // Transfer the rest of operands.
  for (unsigned e = MI->getNumOperands(); OpNum != e; ++OpNum)
    MIB.addOperand(MI->getOperand(OpNum));

  // Transfer memoperands.
  (*MIB).setMemRefs(MI->memoperands_begin(), MI->memoperands_end());

  DEBUG(errs() << "Converted 32-bit: " << *MI << "       to 16-bit: " << *MIB);

  MBB.erase(MI);
  ++NumLdSts;
  return true;
}

bool
Thumb2SizeReduce::ReduceSpecial(MachineBasicBlock &MBB, MachineInstr *MI,
                                const ReduceEntry &Entry,
                                bool LiveCPSR) {
  if (Entry.LowRegs1 && !VerifyLowRegs(MI))
    return false;

  const TargetInstrDesc &TID = MI->getDesc();
  if (TID.mayLoad() || TID.mayStore())
    return ReduceLoadStore(MBB, MI, Entry);

  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  default: break;
  case ARM::t2ADDSri: 
  case ARM::t2ADDSrr: {
    unsigned PredReg = 0;
    if (getInstrPredicate(MI, PredReg) == ARMCC::AL) {
      switch (Opc) {
      default: break;
      case ARM::t2ADDSri: {
        if (ReduceTo2Addr(MBB, MI, Entry, LiveCPSR))
          return true;
        // fallthrough
      }
      case ARM::t2ADDSrr:
        return ReduceToNarrow(MBB, MI, Entry, LiveCPSR);
      }
    }
    break;
  }
  case ARM::t2RSBri:
  case ARM::t2RSBSri:
    if (MI->getOperand(2).getImm() == 0)
      return ReduceToNarrow(MBB, MI, Entry, LiveCPSR);
    break;
  case ARM::t2MOVi16:
    // Can convert only 'pure' immediate operands, not immediates obtained as
    // globals' addresses.
    if (MI->getOperand(1).isImm())
      return ReduceToNarrow(MBB, MI, Entry, LiveCPSR);
    break;
  }
  return false;
}

bool
Thumb2SizeReduce::ReduceTo2Addr(MachineBasicBlock &MBB, MachineInstr *MI,
                                const ReduceEntry &Entry,
                                bool LiveCPSR) {

  if (ReduceLimit2Addr != -1 && ((int)Num2Addrs >= ReduceLimit2Addr))
    return false;

  unsigned Reg0 = MI->getOperand(0).getReg();
  unsigned Reg1 = MI->getOperand(1).getReg();
  if (Reg0 != Reg1) {
    // Try to commute the operands to make it a 2-address instruction.
    unsigned CommOpIdx1, CommOpIdx2;
    if (!TII->findCommutedOpIndices(MI, CommOpIdx1, CommOpIdx2) ||
        CommOpIdx1 != 1 || MI->getOperand(CommOpIdx2).getReg() != Reg0)
      return false;
    MachineInstr *CommutedMI = TII->commuteInstruction(MI);
    if (!CommutedMI)
      return false;
  }
  if (Entry.LowRegs2 && !isARMLowRegister(Reg0))
    return false;
  if (Entry.Imm2Limit) {
    unsigned Imm = MI->getOperand(2).getImm();
    unsigned Limit = (1 << Entry.Imm2Limit) - 1;
    if (Imm > Limit)
      return false;
  } else {
    unsigned Reg2 = MI->getOperand(2).getReg();
    if (Entry.LowRegs2 && !isARMLowRegister(Reg2))
      return false;
  }

  // Check if it's possible / necessary to transfer the predicate.
  const TargetInstrDesc &NewTID = TII->get(Entry.NarrowOpc2);
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  bool SkipPred = false;
  if (Pred != ARMCC::AL) {
    if (!NewTID.isPredicable())
      // Can't transfer predicate, fail.
      return false;
  } else {
    SkipPred = !NewTID.isPredicable();
  }

  bool HasCC = false;
  bool CCDead = false;
  const TargetInstrDesc &TID = MI->getDesc();
  if (TID.hasOptionalDef()) {
    unsigned NumOps = TID.getNumOperands();
    HasCC = (MI->getOperand(NumOps-1).getReg() == ARM::CPSR);
    if (HasCC && MI->getOperand(NumOps-1).isDead())
      CCDead = true;
  }
  if (!VerifyPredAndCC(MI, Entry, true, Pred, LiveCPSR, HasCC, CCDead))
    return false;

  // Add the 16-bit instruction.
  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, *MI, dl, NewTID);
  MIB.addOperand(MI->getOperand(0));
  if (NewTID.hasOptionalDef()) {
    if (HasCC)
      AddDefaultT1CC(MIB, CCDead);
    else
      AddNoT1CC(MIB);
  }

  // Transfer the rest of operands.
  unsigned NumOps = TID.getNumOperands();
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; ++i) {
    if (i < NumOps && TID.OpInfo[i].isOptionalDef())
      continue;
    if (SkipPred && TID.OpInfo[i].isPredicate())
      continue;
    MIB.addOperand(MI->getOperand(i));
  }

  DEBUG(errs() << "Converted 32-bit: " << *MI << "       to 16-bit: " << *MIB);

  MBB.erase(MI);
  ++Num2Addrs;
  return true;
}

bool
Thumb2SizeReduce::ReduceToNarrow(MachineBasicBlock &MBB, MachineInstr *MI,
                                 const ReduceEntry &Entry,
                                 bool LiveCPSR) {
  if (ReduceLimit != -1 && ((int)NumNarrows >= ReduceLimit))
    return false;

  unsigned Limit = ~0U;
  unsigned Scale = (Entry.WideOpc == ARM::t2ADDrSPi) ? 4 : 1;
  if (Entry.Imm1Limit)
    Limit = ((1 << Entry.Imm1Limit) - 1) * Scale;

  const TargetInstrDesc &TID = MI->getDesc();
  for (unsigned i = 0, e = TID.getNumOperands(); i != e; ++i) {
    if (TID.OpInfo[i].isPredicate())
      continue;
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg()) {
      unsigned Reg = MO.getReg();
      if (!Reg || Reg == ARM::CPSR)
        continue;
      if (Entry.WideOpc == ARM::t2ADDrSPi && Reg == ARM::SP)
        continue;
      if (Entry.LowRegs1 && !isARMLowRegister(Reg))
        return false;
    } else if (MO.isImm() &&
               !TID.OpInfo[i].isPredicate()) {
      if (((unsigned)MO.getImm()) > Limit || (MO.getImm() & (Scale-1)) != 0)
        return false;
    }
  }

  // Check if it's possible / necessary to transfer the predicate.
  const TargetInstrDesc &NewTID = TII->get(Entry.NarrowOpc1);
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  bool SkipPred = false;
  if (Pred != ARMCC::AL) {
    if (!NewTID.isPredicable())
      // Can't transfer predicate, fail.
      return false;
  } else {
    SkipPred = !NewTID.isPredicable();
  }

  bool HasCC = false;
  bool CCDead = false;
  if (TID.hasOptionalDef()) {
    unsigned NumOps = TID.getNumOperands();
    HasCC = (MI->getOperand(NumOps-1).getReg() == ARM::CPSR);
    if (HasCC && MI->getOperand(NumOps-1).isDead())
      CCDead = true;
  }
  if (!VerifyPredAndCC(MI, Entry, false, Pred, LiveCPSR, HasCC, CCDead))
    return false;

  // Add the 16-bit instruction.
  DebugLoc dl = MI->getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, *MI, dl, NewTID);
  MIB.addOperand(MI->getOperand(0));
  if (NewTID.hasOptionalDef()) {
    if (HasCC)
      AddDefaultT1CC(MIB, CCDead);
    else
      AddNoT1CC(MIB);
  }

  // Transfer the rest of operands.
  unsigned NumOps = TID.getNumOperands();
  for (unsigned i = 1, e = MI->getNumOperands(); i != e; ++i) {
    if (i < NumOps && TID.OpInfo[i].isOptionalDef())
      continue;
    if ((TID.getOpcode() == ARM::t2RSBSri ||
         TID.getOpcode() == ARM::t2RSBri) && i == 2)
      // Skip the zero immediate operand, it's now implicit.
      continue;
    bool isPred = (i < NumOps && TID.OpInfo[i].isPredicate());
    if (SkipPred && isPred)
        continue;
    const MachineOperand &MO = MI->getOperand(i);
    if (Scale > 1 && !isPred && MO.isImm())
      MIB.addImm(MO.getImm() / Scale);
    else {
      if (MO.isReg() && MO.isImplicit() && MO.getReg() == ARM::CPSR)
        // Skip implicit def of CPSR. Either it's modeled as an optional
        // def now or it's already an implicit def on the new instruction.
        continue;
      MIB.addOperand(MO);
    }
  }
  if (!TID.isPredicable() && NewTID.isPredicable())
    AddDefaultPred(MIB);

  DEBUG(errs() << "Converted 32-bit: " << *MI << "       to 16-bit: " << *MIB);

  MBB.erase(MI);
  ++NumNarrows;
  return true;
}

static bool UpdateCPSRDef(MachineInstr &MI, bool LiveCPSR) {
  bool HasDef = false;
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || MO.isUndef() || MO.isUse())
      continue;
    if (MO.getReg() != ARM::CPSR)
      continue;
    if (!MO.isDead())
      HasDef = true;
  }

  return HasDef || LiveCPSR;
}

static bool UpdateCPSRUse(MachineInstr &MI, bool LiveCPSR) {
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI.getOperand(i);
    if (!MO.isReg() || MO.isUndef() || MO.isDef())
      continue;
    if (MO.getReg() != ARM::CPSR)
      continue;
    assert(LiveCPSR && "CPSR liveness tracking is wrong!");
    if (MO.isKill()) {
      LiveCPSR = false;
      break;
    }
  }

  return LiveCPSR;
}

bool Thumb2SizeReduce::ReduceMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  // Yes, CPSR could be livein.
  bool LiveCPSR = MBB.isLiveIn(ARM::CPSR);

  MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end();
  MachineBasicBlock::iterator NextMII;
  for (; MII != E; MII = NextMII) {
    NextMII = llvm::next(MII);

    MachineInstr *MI = &*MII;
    LiveCPSR = UpdateCPSRUse(*MI, LiveCPSR);

    unsigned Opcode = MI->getOpcode();
    DenseMap<unsigned, unsigned>::iterator OPI = ReduceOpcodeMap.find(Opcode);
    if (OPI != ReduceOpcodeMap.end()) {
      const ReduceEntry &Entry = ReduceTable[OPI->second];
      // Ignore "special" cases for now.
      if (Entry.Special) {
        if (ReduceSpecial(MBB, MI, Entry, LiveCPSR)) {
          Modified = true;
          MachineBasicBlock::iterator I = prior(NextMII);
          MI = &*I;
        }
        goto ProcessNext;
      }

      // Try to transform to a 16-bit two-address instruction.
      if (Entry.NarrowOpc2 && ReduceTo2Addr(MBB, MI, Entry, LiveCPSR)) {
        Modified = true;
        MachineBasicBlock::iterator I = prior(NextMII);
        MI = &*I;
        goto ProcessNext;
      }

      // Try to transform to a 16-bit non-two-address instruction.
      if (Entry.NarrowOpc1 && ReduceToNarrow(MBB, MI, Entry, LiveCPSR)) {
        Modified = true;
        MachineBasicBlock::iterator I = prior(NextMII);
        MI = &*I;
      }
    }

  ProcessNext:
    LiveCPSR = UpdateCPSRDef(*MI, LiveCPSR);
  }

  return Modified;
}

bool Thumb2SizeReduce::runOnMachineFunction(MachineFunction &MF) {
  const TargetMachine &TM = MF.getTarget();
  TII = static_cast<const Thumb2InstrInfo*>(TM.getInstrInfo());

  bool Modified = false;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    Modified |= ReduceMBB(*I);
  return Modified;
}

/// createThumb2SizeReductionPass - Returns an instance of the Thumb2 size
/// reduction pass.
FunctionPass *llvm::createThumb2SizeReductionPass() {
  return new Thumb2SizeReduce();
}
