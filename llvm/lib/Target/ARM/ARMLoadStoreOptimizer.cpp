//===-- ARMLoadStoreOptimizer.cpp - ARM load / store opt. pass ----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs load / store related peephole
// optimizations. This pass should be run after register allocation.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-ldst-opt"
#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMRegisterInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

STATISTIC(NumLDMGened , "Number of ldm instructions generated");
STATISTIC(NumSTMGened , "Number of stm instructions generated");
STATISTIC(NumFLDMGened, "Number of fldm instructions generated");
STATISTIC(NumFSTMGened, "Number of fstm instructions generated");

namespace {
  struct VISIBILITY_HIDDEN ARMLoadStoreOpt : public MachineFunctionPass {
    static char ID;
    ARMLoadStoreOpt() : MachineFunctionPass(&ID) {}

    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    ARMFunctionInfo *AFI;
    RegScavenger *RS;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM load / store optimization pass";
    }

  private:
    struct MemOpQueueEntry {
      int Offset;
      unsigned Position;
      MachineBasicBlock::iterator MBBI;
      bool Merged;
      MemOpQueueEntry(int o, int p, MachineBasicBlock::iterator i)
        : Offset(o), Position(p), MBBI(i), Merged(false) {};
    };
    typedef SmallVector<MemOpQueueEntry,8> MemOpQueue;
    typedef MemOpQueue::iterator MemOpQueueIter;

    SmallVector<MachineBasicBlock::iterator, 4>
    MergeLDR_STR(MachineBasicBlock &MBB, unsigned SIndex, unsigned Base,
                 int Opcode, unsigned Size,
                 ARMCC::CondCodes Pred, unsigned PredReg,
                 unsigned Scratch, MemOpQueue &MemOps);

    void AdvanceRS(MachineBasicBlock &MBB, MemOpQueue &MemOps);
    bool LoadStoreMultipleOpti(MachineBasicBlock &MBB);
    bool MergeReturnIntoLDM(MachineBasicBlock &MBB);
  };
  char ARMLoadStoreOpt::ID = 0;
}

/// createARMLoadStoreOptimizationPass - returns an instance of the load / store
/// optimization pass.
FunctionPass *llvm::createARMLoadStoreOptimizationPass() {
  return new ARMLoadStoreOpt();
}

static int getLoadStoreMultipleOpcode(int Opcode) {
  switch (Opcode) {
  case ARM::LDR:
    NumLDMGened++;
    return ARM::LDM;
  case ARM::STR:
    NumSTMGened++;
    return ARM::STM;
  case ARM::FLDS:
    NumFLDMGened++;
    return ARM::FLDMS;
  case ARM::FSTS:
    NumFSTMGened++;
    return ARM::FSTMS;
  case ARM::FLDD:
    NumFLDMGened++;
    return ARM::FLDMD;
  case ARM::FSTD:
    NumFSTMGened++;
    return ARM::FSTMD;
  default: abort();
  }
  return 0;
}

/// mergeOps - Create and insert a LDM or STM with Base as base register and
/// registers in Regs as the register operands that would be loaded / stored.
/// It returns true if the transformation is done. 
static bool mergeOps(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     int Offset, unsigned Base, bool BaseKill, int Opcode,
                     ARMCC::CondCodes Pred, unsigned PredReg, unsigned Scratch,
                     SmallVector<std::pair<unsigned, bool>, 8> &Regs,
                     const TargetInstrInfo *TII) {
  // Only a single register to load / store. Don't bother.
  unsigned NumRegs = Regs.size();
  if (NumRegs <= 1)
    return false;

  ARM_AM::AMSubMode Mode = ARM_AM::ia;
  bool isAM4 = Opcode == ARM::LDR || Opcode == ARM::STR;
  if (isAM4 && Offset == 4)
    Mode = ARM_AM::ib;
  else if (isAM4 && Offset == -4 * (int)NumRegs + 4)
    Mode = ARM_AM::da;
  else if (isAM4 && Offset == -4 * (int)NumRegs)
    Mode = ARM_AM::db;
  else if (Offset != 0) {
    // If starting offset isn't zero, insert a MI to materialize a new base.
    // But only do so if it is cost effective, i.e. merging more than two
    // loads / stores.
    if (NumRegs <= 2)
      return false;

    unsigned NewBase;
    if (Opcode == ARM::LDR)
      // If it is a load, then just use one of the destination register to
      // use as the new base.
      NewBase = Regs[NumRegs-1].first;
    else {
      // Use the scratch register to use as a new base.
      NewBase = Scratch;
      if (NewBase == 0)
        return false;
    }
    int BaseOpc = ARM::ADDri;
    if (Offset < 0) {
      BaseOpc = ARM::SUBri;
      Offset = - Offset;
    }
    int ImmedOffset = ARM_AM::getSOImmVal(Offset);
    if (ImmedOffset == -1)
      return false;  // Probably not worth it then.

    BuildMI(MBB, MBBI, TII->get(BaseOpc), NewBase)
      .addReg(Base, false, false, BaseKill).addImm(ImmedOffset)
      .addImm(Pred).addReg(PredReg).addReg(0);
    Base = NewBase;
    BaseKill = true;  // New base is always killed right its use.
  }

  bool isDPR = Opcode == ARM::FLDD || Opcode == ARM::FSTD;
  bool isDef = Opcode == ARM::LDR || Opcode == ARM::FLDS || Opcode == ARM::FLDD;
  Opcode = getLoadStoreMultipleOpcode(Opcode);
  MachineInstrBuilder MIB = (isAM4)
    ? BuildMI(MBB, MBBI, TII->get(Opcode)).addReg(Base, false, false, BaseKill)
        .addImm(ARM_AM::getAM4ModeImm(Mode)).addImm(Pred).addReg(PredReg)
    : BuildMI(MBB, MBBI, TII->get(Opcode)).addReg(Base, false, false, BaseKill)
        .addImm(ARM_AM::getAM5Opc(Mode, false, isDPR ? NumRegs<<1 : NumRegs))
        .addImm(Pred).addReg(PredReg);
  for (unsigned i = 0; i != NumRegs; ++i)
    MIB = MIB.addReg(Regs[i].first, isDef, false, Regs[i].second);

  return true;
}

/// MergeLDR_STR - Merge a number of load / store instructions into one or more
/// load / store multiple instructions.
SmallVector<MachineBasicBlock::iterator, 4>
ARMLoadStoreOpt::MergeLDR_STR(MachineBasicBlock &MBB, unsigned SIndex,
                              unsigned Base, int Opcode, unsigned Size,
                              ARMCC::CondCodes Pred, unsigned PredReg,
                              unsigned Scratch, MemOpQueue &MemOps) {
  SmallVector<MachineBasicBlock::iterator, 4> Merges;
  bool isAM4 = Opcode == ARM::LDR || Opcode == ARM::STR;
  int Offset = MemOps[SIndex].Offset;
  int SOffset = Offset;
  unsigned Pos = MemOps[SIndex].Position;
  MachineBasicBlock::iterator Loc = MemOps[SIndex].MBBI;
  unsigned PReg = MemOps[SIndex].MBBI->getOperand(0).getReg();
  unsigned PRegNum = ARMRegisterInfo::getRegisterNumbering(PReg);
  bool isKill = MemOps[SIndex].MBBI->getOperand(0).isKill();

  SmallVector<std::pair<unsigned,bool>, 8> Regs;
  Regs.push_back(std::make_pair(PReg, isKill));
  for (unsigned i = SIndex+1, e = MemOps.size(); i != e; ++i) {
    int NewOffset = MemOps[i].Offset;
    unsigned Reg = MemOps[i].MBBI->getOperand(0).getReg();
    unsigned RegNum = ARMRegisterInfo::getRegisterNumbering(Reg);
    isKill = MemOps[i].MBBI->getOperand(0).isKill();
    // AM4 - register numbers in ascending order.
    // AM5 - consecutive register numbers in ascending order.
    if (NewOffset == Offset + (int)Size &&
        ((isAM4 && RegNum > PRegNum) || RegNum == PRegNum+1)) {
      Offset += Size;
      Regs.push_back(std::make_pair(Reg, isKill));
      PRegNum = RegNum;
    } else {
      // Can't merge this in. Try merge the earlier ones first.
      if (mergeOps(MBB, ++Loc, SOffset, Base, false, Opcode, Pred, PredReg,
                   Scratch, Regs, TII)) {
        Merges.push_back(prior(Loc));
        for (unsigned j = SIndex; j < i; ++j) {
          MBB.erase(MemOps[j].MBBI);
          MemOps[j].Merged = true;
        }
      }
      SmallVector<MachineBasicBlock::iterator, 4> Merges2 =
        MergeLDR_STR(MBB, i, Base, Opcode, Size, Pred, PredReg, Scratch,MemOps);
      Merges.append(Merges2.begin(), Merges2.end());
      return Merges;
    }

    if (MemOps[i].Position > Pos) {
      Pos = MemOps[i].Position;
      Loc = MemOps[i].MBBI;
    }
  }

  bool BaseKill = Loc->findRegisterUseOperandIdx(Base, true) != -1;
  if (mergeOps(MBB, ++Loc, SOffset, Base, BaseKill, Opcode, Pred, PredReg,
               Scratch, Regs, TII)) {
    Merges.push_back(prior(Loc));
    for (unsigned i = SIndex, e = MemOps.size(); i != e; ++i) {
      MBB.erase(MemOps[i].MBBI);
      MemOps[i].Merged = true;
    }
  }

  return Merges;
}

/// getInstrPredicate - If instruction is predicated, returns its predicate
/// condition, otherwise returns AL. It also returns the condition code
/// register by reference.
static ARMCC::CondCodes getInstrPredicate(MachineInstr *MI, unsigned &PredReg) {
  int PIdx = MI->findFirstPredOperandIdx();
  if (PIdx == -1) {
    PredReg = 0;
    return ARMCC::AL;
  }

  PredReg = MI->getOperand(PIdx+1).getReg();
  return (ARMCC::CondCodes)MI->getOperand(PIdx).getImm();
}

static inline bool isMatchingDecrement(MachineInstr *MI, unsigned Base,
                                       unsigned Bytes, ARMCC::CondCodes Pred,
                                       unsigned PredReg) {
  unsigned MyPredReg = 0;
  return (MI && MI->getOpcode() == ARM::SUBri &&
          MI->getOperand(0).getReg() == Base &&
          MI->getOperand(1).getReg() == Base &&
          ARM_AM::getAM2Offset(MI->getOperand(2).getImm()) == Bytes &&
          getInstrPredicate(MI, MyPredReg) == Pred &&
          MyPredReg == PredReg);
}

static inline bool isMatchingIncrement(MachineInstr *MI, unsigned Base,
                                       unsigned Bytes, ARMCC::CondCodes Pred,
                                       unsigned PredReg) {
  unsigned MyPredReg = 0;
  return (MI && MI->getOpcode() == ARM::ADDri &&
          MI->getOperand(0).getReg() == Base &&
          MI->getOperand(1).getReg() == Base &&
          ARM_AM::getAM2Offset(MI->getOperand(2).getImm()) == Bytes &&
          getInstrPredicate(MI, MyPredReg) == Pred &&
          MyPredReg == PredReg);
}

static inline unsigned getLSMultipleTransferSize(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: return 0;
  case ARM::LDR:
  case ARM::STR:
  case ARM::FLDS:
  case ARM::FSTS:
    return 4;
  case ARM::FLDD:
  case ARM::FSTD:
    return 8;
  case ARM::LDM:
  case ARM::STM:
    return (MI->getNumOperands() - 4) * 4;
  case ARM::FLDMS:
  case ARM::FSTMS:
  case ARM::FLDMD:
  case ARM::FSTMD:
    return ARM_AM::getAM5Offset(MI->getOperand(1).getImm()) * 4;
  }
}

/// mergeBaseUpdateLSMultiple - Fold proceeding/trailing inc/dec of base
/// register into the LDM/STM/FLDM{D|S}/FSTM{D|S} op when possible:
///
/// stmia rn, <ra, rb, rc>
/// rn := rn + 4 * 3;
/// =>
/// stmia rn!, <ra, rb, rc>
///
/// rn := rn - 4 * 3;
/// ldmia rn, <ra, rb, rc>
/// =>
/// ldmdb rn!, <ra, rb, rc>
static bool mergeBaseUpdateLSMultiple(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      bool &Advance,
                                      MachineBasicBlock::iterator &I) {
  MachineInstr *MI = MBBI;
  unsigned Base = MI->getOperand(0).getReg();
  unsigned Bytes = getLSMultipleTransferSize(MI);
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  int Opcode = MI->getOpcode();
  bool isAM4 = Opcode == ARM::LDM || Opcode == ARM::STM;

  if (isAM4) {
    if (ARM_AM::getAM4WBFlag(MI->getOperand(1).getImm()))
      return false;

    // Can't use the updating AM4 sub-mode if the base register is also a dest
    // register. e.g. ldmdb r0!, {r0, r1, r2}. The behavior is undefined.
    for (unsigned i = 3, e = MI->getNumOperands(); i != e; ++i) {
      if (MI->getOperand(i).getReg() == Base)
        return false;
    }

    ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MI->getOperand(1).getImm());
    if (MBBI != MBB.begin()) {
      MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
      if (Mode == ARM_AM::ia &&
          isMatchingDecrement(PrevMBBI, Base, Bytes, Pred, PredReg)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(ARM_AM::db, true));
        MBB.erase(PrevMBBI);
        return true;
      } else if (Mode == ARM_AM::ib &&
                 isMatchingDecrement(PrevMBBI, Base, Bytes, Pred, PredReg)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(ARM_AM::da, true));
        MBB.erase(PrevMBBI);
        return true;
      }
    }

    if (MBBI != MBB.end()) {
      MachineBasicBlock::iterator NextMBBI = next(MBBI);
      if ((Mode == ARM_AM::ia || Mode == ARM_AM::ib) &&
          isMatchingIncrement(NextMBBI, Base, Bytes, Pred, PredReg)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(Mode, true));
        if (NextMBBI == I) {
          Advance = true;
          ++I;
        }
        MBB.erase(NextMBBI);
        return true;
      } else if ((Mode == ARM_AM::da || Mode == ARM_AM::db) &&
                 isMatchingDecrement(NextMBBI, Base, Bytes, Pred, PredReg)) {
        MI->getOperand(1).setImm(ARM_AM::getAM4ModeImm(Mode, true));
        if (NextMBBI == I) {
          Advance = true;
          ++I;
        }
        MBB.erase(NextMBBI);
        return true;
      }
    }
  } else {
    // FLDM{D|S}, FSTM{D|S} addressing mode 5 ops.
    if (ARM_AM::getAM5WBFlag(MI->getOperand(1).getImm()))
      return false;

    ARM_AM::AMSubMode Mode = ARM_AM::getAM5SubMode(MI->getOperand(1).getImm());
    unsigned Offset = ARM_AM::getAM5Offset(MI->getOperand(1).getImm());
    if (MBBI != MBB.begin()) {
      MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
      if (Mode == ARM_AM::ia &&
          isMatchingDecrement(PrevMBBI, Base, Bytes, Pred, PredReg)) {
        MI->getOperand(1).setImm(ARM_AM::getAM5Opc(ARM_AM::db, true, Offset));
        MBB.erase(PrevMBBI);
        return true;
      }
    }

    if (MBBI != MBB.end()) {
      MachineBasicBlock::iterator NextMBBI = next(MBBI);
      if (Mode == ARM_AM::ia &&
          isMatchingIncrement(NextMBBI, Base, Bytes, Pred, PredReg)) {
        MI->getOperand(1).setImm(ARM_AM::getAM5Opc(ARM_AM::ia, true, Offset));
        if (NextMBBI == I) {
          Advance = true;
          ++I;
        }
        MBB.erase(NextMBBI);
      }
      return true;
    }
  }

  return false;
}

static unsigned getPreIndexedLoadStoreOpcode(unsigned Opc) {
  switch (Opc) {
  case ARM::LDR: return ARM::LDR_PRE;
  case ARM::STR: return ARM::STR_PRE;
  case ARM::FLDS: return ARM::FLDMS;
  case ARM::FLDD: return ARM::FLDMD;
  case ARM::FSTS: return ARM::FSTMS;
  case ARM::FSTD: return ARM::FSTMD;
  default: abort();
  }
  return 0;
}

static unsigned getPostIndexedLoadStoreOpcode(unsigned Opc) {
  switch (Opc) {
  case ARM::LDR: return ARM::LDR_POST;
  case ARM::STR: return ARM::STR_POST;
  case ARM::FLDS: return ARM::FLDMS;
  case ARM::FLDD: return ARM::FLDMD;
  case ARM::FSTS: return ARM::FSTMS;
  case ARM::FSTD: return ARM::FSTMD;
  default: abort();
  }
  return 0;
}

/// mergeBaseUpdateLoadStore - Fold proceeding/trailing inc/dec of base
/// register into the LDR/STR/FLD{D|S}/FST{D|S} op when possible:
static bool mergeBaseUpdateLoadStore(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     const TargetInstrInfo *TII,
                                     bool &Advance,
                                     MachineBasicBlock::iterator &I) {
  MachineInstr *MI = MBBI;
  unsigned Base = MI->getOperand(1).getReg();
  bool BaseKill = MI->getOperand(1).isKill();
  unsigned Bytes = getLSMultipleTransferSize(MI);
  int Opcode = MI->getOpcode();
  bool isAM2 = Opcode == ARM::LDR || Opcode == ARM::STR;
  if ((isAM2 && ARM_AM::getAM2Offset(MI->getOperand(3).getImm()) != 0) ||
      (!isAM2 && ARM_AM::getAM5Offset(MI->getOperand(2).getImm()) != 0))
    return false;

  bool isLd = Opcode == ARM::LDR || Opcode == ARM::FLDS || Opcode == ARM::FLDD;
  // Can't do the merge if the destination register is the same as the would-be
  // writeback register.
  if (isLd && MI->getOperand(0).getReg() == Base)
    return false;

  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  bool DoMerge = false;
  ARM_AM::AddrOpc AddSub = ARM_AM::add;
  unsigned NewOpc = 0;
  if (MBBI != MBB.begin()) {
    MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
    if (isMatchingDecrement(PrevMBBI, Base, Bytes, Pred, PredReg)) {
      DoMerge = true;
      AddSub = ARM_AM::sub;
      NewOpc = getPreIndexedLoadStoreOpcode(Opcode);
    } else if (isAM2 && isMatchingIncrement(PrevMBBI, Base, Bytes,
                                            Pred, PredReg)) {
      DoMerge = true;
      NewOpc = getPreIndexedLoadStoreOpcode(Opcode);
    }
    if (DoMerge)
      MBB.erase(PrevMBBI);
  }

  if (!DoMerge && MBBI != MBB.end()) {
    MachineBasicBlock::iterator NextMBBI = next(MBBI);
    if (isAM2 && isMatchingDecrement(NextMBBI, Base, Bytes, Pred, PredReg)) {
      DoMerge = true;
      AddSub = ARM_AM::sub;
      NewOpc = getPostIndexedLoadStoreOpcode(Opcode);
    } else if (isMatchingIncrement(NextMBBI, Base, Bytes, Pred, PredReg)) {
      DoMerge = true;
      NewOpc = getPostIndexedLoadStoreOpcode(Opcode);
    }
    if (DoMerge) {
      if (NextMBBI == I) {
        Advance = true;
        ++I;
      }
      MBB.erase(NextMBBI);
    }
  }

  if (!DoMerge)
    return false;

  bool isDPR = NewOpc == ARM::FLDMD || NewOpc == ARM::FSTMD;
  unsigned Offset = isAM2 ? ARM_AM::getAM2Opc(AddSub, Bytes, ARM_AM::no_shift)
    : ARM_AM::getAM5Opc((AddSub == ARM_AM::sub) ? ARM_AM::db : ARM_AM::ia,
                        true, isDPR ? 2 : 1);
  if (isLd) {
    if (isAM2)
      // LDR_PRE, LDR_POST;
      BuildMI(MBB, MBBI, TII->get(NewOpc), MI->getOperand(0).getReg())
        .addReg(Base, true)
        .addReg(Base).addReg(0).addImm(Offset).addImm(Pred).addReg(PredReg);
    else
      // FLDMS, FLDMD
      BuildMI(MBB, MBBI, TII->get(NewOpc)).addReg(Base, false, false, BaseKill)
        .addImm(Offset).addImm(Pred).addReg(PredReg)
        .addReg(MI->getOperand(0).getReg(), true);
  } else {
    MachineOperand &MO = MI->getOperand(0);
    if (isAM2)
      // STR_PRE, STR_POST;
      BuildMI(MBB, MBBI, TII->get(NewOpc), Base)
        .addReg(MO.getReg(), false, false, MO.isKill())
        .addReg(Base).addReg(0).addImm(Offset).addImm(Pred).addReg(PredReg);
    else
      // FSTMS, FSTMD
      BuildMI(MBB, MBBI, TII->get(NewOpc)).addReg(Base).addImm(Offset)
        .addImm(Pred).addReg(PredReg)
        .addReg(MO.getReg(), false, false, MO.isKill());
  }
  MBB.erase(MBBI);

  return true;
}

/// isMemoryOp - Returns true if instruction is a memory operations (that this
/// pass is capable of operating on).
static bool isMemoryOp(MachineInstr *MI) {
  int Opcode = MI->getOpcode();
  switch (Opcode) {
  default: break;
  case ARM::LDR:
  case ARM::STR:
    return MI->getOperand(1).isReg() && MI->getOperand(2).getReg() == 0;
  case ARM::FLDS:
  case ARM::FSTS:
    return MI->getOperand(1).isReg();
  case ARM::FLDD:
  case ARM::FSTD:
    return MI->getOperand(1).isReg();
  }
  return false;
}

/// AdvanceRS - Advance register scavenger to just before the earliest memory
/// op that is being merged.
void ARMLoadStoreOpt::AdvanceRS(MachineBasicBlock &MBB, MemOpQueue &MemOps) {
  MachineBasicBlock::iterator Loc = MemOps[0].MBBI;
  unsigned Position = MemOps[0].Position;
  for (unsigned i = 1, e = MemOps.size(); i != e; ++i) {
    if (MemOps[i].Position < Position) {
      Position = MemOps[i].Position;
      Loc = MemOps[i].MBBI;
    }
  }

  if (Loc != MBB.begin())
    RS->forward(prior(Loc));
}

/// LoadStoreMultipleOpti - An optimization pass to turn multiple LDR / STR
/// ops of the same base and incrementing offset into LDM / STM ops.
bool ARMLoadStoreOpt::LoadStoreMultipleOpti(MachineBasicBlock &MBB) {
  unsigned NumMerges = 0;
  unsigned NumMemOps = 0;
  MemOpQueue MemOps;
  unsigned CurrBase = 0;
  int CurrOpc = -1;
  unsigned CurrSize = 0;
  ARMCC::CondCodes CurrPred = ARMCC::AL;
  unsigned CurrPredReg = 0;
  unsigned Position = 0;

  RS->enterBasicBlock(&MBB);
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    bool Advance  = false;
    bool TryMerge = false;
    bool Clobber  = false;

    bool isMemOp = isMemoryOp(MBBI);
    if (isMemOp) {
      int Opcode = MBBI->getOpcode();
      bool isAM2 = Opcode == ARM::LDR || Opcode == ARM::STR;
      unsigned Size = getLSMultipleTransferSize(MBBI);
      unsigned Base = MBBI->getOperand(1).getReg();
      unsigned PredReg = 0;
      ARMCC::CondCodes Pred = getInstrPredicate(MBBI, PredReg);
      unsigned NumOperands = MBBI->getDesc().getNumOperands();
      unsigned OffField = MBBI->getOperand(NumOperands-3).getImm();
      int Offset = isAM2
        ? ARM_AM::getAM2Offset(OffField) : ARM_AM::getAM5Offset(OffField) * 4;
      if (isAM2) {
        if (ARM_AM::getAM2Op(OffField) == ARM_AM::sub)
          Offset = -Offset;
      } else {
        if (ARM_AM::getAM5Op(OffField) == ARM_AM::sub)
          Offset = -Offset;
      }
      // Watch out for:
      // r4 := ldr [r5]
      // r5 := ldr [r5, #4]
      // r6 := ldr [r5, #8]
      //
      // The second ldr has effectively broken the chain even though it
      // looks like the later ldr(s) use the same base register. Try to
      // merge the ldr's so far, including this one. But don't try to
      // combine the following ldr(s).
      Clobber = (Opcode == ARM::LDR && Base == MBBI->getOperand(0).getReg());
      if (CurrBase == 0 && !Clobber) {
        // Start of a new chain.
        CurrBase = Base;
        CurrOpc  = Opcode;
        CurrSize = Size;
        CurrPred = Pred;
        CurrPredReg = PredReg;
        MemOps.push_back(MemOpQueueEntry(Offset, Position, MBBI));
        NumMemOps++;
        Advance = true;
      } else {
        if (Clobber) {
          TryMerge = true;
          Advance = true;
        }

        if (CurrOpc == Opcode && CurrBase == Base && CurrPred == Pred) {
          // No need to match PredReg.
          // Continue adding to the queue.
          if (Offset > MemOps.back().Offset) {
            MemOps.push_back(MemOpQueueEntry(Offset, Position, MBBI));
            NumMemOps++;
            Advance = true;
          } else {
            for (MemOpQueueIter I = MemOps.begin(), E = MemOps.end();
                 I != E; ++I) {
              if (Offset < I->Offset) {
                MemOps.insert(I, MemOpQueueEntry(Offset, Position, MBBI));
                NumMemOps++;
                Advance = true;
                break;
              } else if (Offset == I->Offset) {
                // Collision! This can't be merged!
                break;
              }
            }
          }
        }
      }
    }

    if (Advance) {
      ++Position;
      ++MBBI;
    } else
      TryMerge = true;

    if (TryMerge) {
      if (NumMemOps > 1) {
        // Try to find a free register to use as a new base in case it's needed.
        // First advance to the instruction just before the start of the chain.
        AdvanceRS(MBB, MemOps);
        // Find a scratch register. Make sure it's a call clobbered register or
        // a spilled callee-saved register.
        unsigned Scratch = RS->FindUnusedReg(&ARM::GPRRegClass, true);
        if (!Scratch)
          Scratch = RS->FindUnusedReg(&ARM::GPRRegClass,
                                      AFI->getSpilledCSRegisters());
        // Process the load / store instructions.
        RS->forward(prior(MBBI));

        // Merge ops.
        SmallVector<MachineBasicBlock::iterator,4> MBBII =
          MergeLDR_STR(MBB, 0, CurrBase, CurrOpc, CurrSize,
                       CurrPred, CurrPredReg, Scratch, MemOps);

        // Try folding preceeding/trailing base inc/dec into the generated
        // LDM/STM ops.
        for (unsigned i = 0, e = MBBII.size(); i < e; ++i)
          if (mergeBaseUpdateLSMultiple(MBB, MBBII[i], Advance, MBBI))
            NumMerges++;
        NumMerges += MBBII.size();

        // Try folding preceeding/trailing base inc/dec into those load/store
        // that were not merged to form LDM/STM ops.
        for (unsigned i = 0; i != NumMemOps; ++i)
          if (!MemOps[i].Merged)
            if (mergeBaseUpdateLoadStore(MBB, MemOps[i].MBBI, TII,Advance,MBBI))
              NumMerges++;

        // RS may be pointing to an instruction that's deleted. 
        RS->skipTo(prior(MBBI));
      }

      CurrBase = 0;
      CurrOpc = -1;
      CurrSize = 0;
      CurrPred = ARMCC::AL;
      CurrPredReg = 0;
      if (NumMemOps) {
        MemOps.clear();
        NumMemOps = 0;
      }

      // If iterator hasn't been advanced and this is not a memory op, skip it.
      // It can't start a new chain anyway.
      if (!Advance && !isMemOp && MBBI != E) {
        ++Position;
        ++MBBI;
      }
    }
  }
  return NumMerges > 0;
}

/// MergeReturnIntoLDM - If this is a exit BB, try merging the return op
/// (bx lr) into the preceeding stack restore so it directly restore the value
/// of LR into pc.
///   ldmfd sp!, {r7, lr}
///   bx lr
/// =>
///   ldmfd sp!, {r7, pc}
bool ARMLoadStoreOpt::MergeReturnIntoLDM(MachineBasicBlock &MBB) {
  if (MBB.empty()) return false;

  MachineBasicBlock::iterator MBBI = prior(MBB.end());
  if (MBBI->getOpcode() == ARM::BX_RET && MBBI != MBB.begin()) {
    MachineInstr *PrevMI = prior(MBBI);
    if (PrevMI->getOpcode() == ARM::LDM) {
      MachineOperand &MO = PrevMI->getOperand(PrevMI->getNumOperands()-1);
      if (MO.getReg() == ARM::LR) {
        PrevMI->setDesc(TII->get(ARM::LDM_RET));
        MO.setReg(ARM::PC);
        MBB.erase(MBBI);
        return true;
      }
    }
  }
  return false;
}

bool ARMLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  const TargetMachine &TM = Fn.getTarget();
  AFI = Fn.getInfo<ARMFunctionInfo>();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  RS = new RegScavenger();

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= LoadStoreMultipleOpti(MBB);
    Modified |= MergeReturnIntoLDM(MBB);
  }

  delete RS;
  return Modified;
}
