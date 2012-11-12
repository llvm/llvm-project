//===-- ARMLoadStoreOptimizer.cpp - ARM load / store opt. pass ------------===//
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
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumLDMGened , "Number of ldm instructions generated");
STATISTIC(NumSTMGened , "Number of stm instructions generated");
STATISTIC(NumVLDMGened, "Number of vldm instructions generated");
STATISTIC(NumVSTMGened, "Number of vstm instructions generated");
STATISTIC(NumLdStMoved, "Number of load / store instructions moved");
STATISTIC(NumLDRDFormed,"Number of ldrd created before allocation");
STATISTIC(NumSTRDFormed,"Number of strd created before allocation");
STATISTIC(NumLDRD2LDM,  "Number of ldrd instructions turned back into ldm");
STATISTIC(NumSTRD2STM,  "Number of strd instructions turned back into stm");
STATISTIC(NumLDRD2LDR,  "Number of ldrd instructions turned back into ldr's");
STATISTIC(NumSTRD2STR,  "Number of strd instructions turned back into str's");

/// ARMAllocLoadStoreOpt - Post- register allocation pass the combine
/// load / store instructions to form ldm / stm instructions.

namespace {
  struct ARMLoadStoreOpt : public MachineFunctionPass {
    static char ID;
    ARMLoadStoreOpt() : MachineFunctionPass(ID) {}

    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const ARMSubtarget *STI;
    ARMFunctionInfo *AFI;
    RegScavenger *RS;
    bool isThumb2;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM load / store optimization pass";
    }

  private:
    struct MemOpQueueEntry {
      int Offset;
      unsigned Reg;
      bool isKill;
      unsigned Position;
      MachineBasicBlock::iterator MBBI;
      bool Merged;
      MemOpQueueEntry(int o, unsigned r, bool k, unsigned p,
                      MachineBasicBlock::iterator i)
        : Offset(o), Reg(r), isKill(k), Position(p), MBBI(i), Merged(false) {}
    };
    typedef SmallVector<MemOpQueueEntry,8> MemOpQueue;
    typedef MemOpQueue::iterator MemOpQueueIter;

    bool MergeOps(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  int Offset, unsigned Base, bool BaseKill, int Opcode,
                  ARMCC::CondCodes Pred, unsigned PredReg, unsigned Scratch,
                  DebugLoc dl,
                  ArrayRef<std::pair<unsigned, bool> > Regs,
                  ArrayRef<unsigned> ImpDefs);
    void MergeOpsUpdate(MachineBasicBlock &MBB,
                        MemOpQueue &MemOps,
                        unsigned memOpsBegin,
                        unsigned memOpsEnd,
                        unsigned insertAfter,
                        int Offset,
                        unsigned Base,
                        bool BaseKill,
                        int Opcode,
                        ARMCC::CondCodes Pred,
                        unsigned PredReg,
                        unsigned Scratch,
                        DebugLoc dl,
                        SmallVector<MachineBasicBlock::iterator, 4> &Merges);
    void MergeLDR_STR(MachineBasicBlock &MBB, unsigned SIndex, unsigned Base,
                      int Opcode, unsigned Size,
                      ARMCC::CondCodes Pred, unsigned PredReg,
                      unsigned Scratch, MemOpQueue &MemOps,
                      SmallVector<MachineBasicBlock::iterator, 4> &Merges);

    void AdvanceRS(MachineBasicBlock &MBB, MemOpQueue &MemOps);
    bool FixInvalidRegPairOp(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator &MBBI);
    bool MergeBaseUpdateLoadStore(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI,
                                  const TargetInstrInfo *TII,
                                  bool &Advance,
                                  MachineBasicBlock::iterator &I);
    bool MergeBaseUpdateLSMultiple(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   bool &Advance,
                                   MachineBasicBlock::iterator &I);
    bool LoadStoreMultipleOpti(MachineBasicBlock &MBB);
    bool MergeReturnIntoLDM(MachineBasicBlock &MBB);
  };
  char ARMLoadStoreOpt::ID = 0;
}

static int getLoadStoreMultipleOpcode(int Opcode, ARM_AM::AMSubMode Mode) {
  switch (Opcode) {
  default: llvm_unreachable("Unhandled opcode!");
  case ARM::LDRi12:
    ++NumLDMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::LDMIA;
    case ARM_AM::da: return ARM::LDMDA;
    case ARM_AM::db: return ARM::LDMDB;
    case ARM_AM::ib: return ARM::LDMIB;
    }
  case ARM::STRi12:
    ++NumSTMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::STMIA;
    case ARM_AM::da: return ARM::STMDA;
    case ARM_AM::db: return ARM::STMDB;
    case ARM_AM::ib: return ARM::STMIB;
    }
  case ARM::t2LDRi8:
  case ARM::t2LDRi12:
    ++NumLDMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::t2LDMIA;
    case ARM_AM::db: return ARM::t2LDMDB;
    }
  case ARM::t2STRi8:
  case ARM::t2STRi12:
    ++NumSTMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::t2STMIA;
    case ARM_AM::db: return ARM::t2STMDB;
    }
  case ARM::VLDRS:
    ++NumVLDMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VLDMSIA;
    case ARM_AM::db: return 0; // Only VLDMSDB_UPD exists.
    }
  case ARM::VSTRS:
    ++NumVSTMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VSTMSIA;
    case ARM_AM::db: return 0; // Only VSTMSDB_UPD exists.
    }
  case ARM::VLDRD:
    ++NumVLDMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VLDMDIA;
    case ARM_AM::db: return 0; // Only VLDMDDB_UPD exists.
    }
  case ARM::VSTRD:
    ++NumVSTMGened;
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VSTMDIA;
    case ARM_AM::db: return 0; // Only VSTMDDB_UPD exists.
    }
  }
}

namespace llvm {
  namespace ARM_AM {

AMSubMode getLoadStoreMultipleSubMode(int Opcode) {
  switch (Opcode) {
  default: llvm_unreachable("Unhandled opcode!");
  case ARM::LDMIA_RET:
  case ARM::LDMIA:
  case ARM::LDMIA_UPD:
  case ARM::STMIA:
  case ARM::STMIA_UPD:
  case ARM::t2LDMIA_RET:
  case ARM::t2LDMIA:
  case ARM::t2LDMIA_UPD:
  case ARM::t2STMIA:
  case ARM::t2STMIA_UPD:
  case ARM::VLDMSIA:
  case ARM::VLDMSIA_UPD:
  case ARM::VSTMSIA:
  case ARM::VSTMSIA_UPD:
  case ARM::VLDMDIA:
  case ARM::VLDMDIA_UPD:
  case ARM::VSTMDIA:
  case ARM::VSTMDIA_UPD:
    return ARM_AM::ia;

  case ARM::LDMDA:
  case ARM::LDMDA_UPD:
  case ARM::STMDA:
  case ARM::STMDA_UPD:
    return ARM_AM::da;

  case ARM::LDMDB:
  case ARM::LDMDB_UPD:
  case ARM::STMDB:
  case ARM::STMDB_UPD:
  case ARM::t2LDMDB:
  case ARM::t2LDMDB_UPD:
  case ARM::t2STMDB:
  case ARM::t2STMDB_UPD:
  case ARM::VLDMSDB_UPD:
  case ARM::VSTMSDB_UPD:
  case ARM::VLDMDDB_UPD:
  case ARM::VSTMDDB_UPD:
    return ARM_AM::db;

  case ARM::LDMIB:
  case ARM::LDMIB_UPD:
  case ARM::STMIB:
  case ARM::STMIB_UPD:
    return ARM_AM::ib;
  }
}

  } // end namespace ARM_AM
} // end namespace llvm

static bool isT2i32Load(unsigned Opc) {
  return Opc == ARM::t2LDRi12 || Opc == ARM::t2LDRi8;
}

static bool isi32Load(unsigned Opc) {
  return Opc == ARM::LDRi12 || isT2i32Load(Opc);
}

static bool isT2i32Store(unsigned Opc) {
  return Opc == ARM::t2STRi12 || Opc == ARM::t2STRi8;
}

static bool isi32Store(unsigned Opc) {
  return Opc == ARM::STRi12 || isT2i32Store(Opc);
}

/// MergeOps - Create and insert a LDM or STM with Base as base register and
/// registers in Regs as the register operands that would be loaded / stored.
/// It returns true if the transformation is done.
bool
ARMLoadStoreOpt::MergeOps(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MBBI,
                          int Offset, unsigned Base, bool BaseKill,
                          int Opcode, ARMCC::CondCodes Pred,
                          unsigned PredReg, unsigned Scratch, DebugLoc dl,
                          ArrayRef<std::pair<unsigned, bool> > Regs,
                          ArrayRef<unsigned> ImpDefs) {
  // Only a single register to load / store. Don't bother.
  unsigned NumRegs = Regs.size();
  if (NumRegs <= 1)
    return false;

  ARM_AM::AMSubMode Mode = ARM_AM::ia;
  // VFP and Thumb2 do not support IB or DA modes.
  bool isNotVFP = isi32Load(Opcode) || isi32Store(Opcode);
  bool haveIBAndDA = isNotVFP && !isThumb2;
  if (Offset == 4 && haveIBAndDA)
    Mode = ARM_AM::ib;
  else if (Offset == -4 * (int)NumRegs + 4 && haveIBAndDA)
    Mode = ARM_AM::da;
  else if (Offset == -4 * (int)NumRegs && isNotVFP)
    // VLDM/VSTM do not support DB mode without also updating the base reg.
    Mode = ARM_AM::db;
  else if (Offset != 0) {
    // Check if this is a supported opcode before we insert instructions to
    // calculate a new base register.
    if (!getLoadStoreMultipleOpcode(Opcode, Mode)) return false;

    // If starting offset isn't zero, insert a MI to materialize a new base.
    // But only do so if it is cost effective, i.e. merging more than two
    // loads / stores.
    if (NumRegs <= 2)
      return false;

    unsigned NewBase;
    if (isi32Load(Opcode))
      // If it is a load, then just use one of the destination register to
      // use as the new base.
      NewBase = Regs[NumRegs-1].first;
    else {
      // Use the scratch register to use as a new base.
      NewBase = Scratch;
      if (NewBase == 0)
        return false;
    }
    int BaseOpc = !isThumb2 ? ARM::ADDri : ARM::t2ADDri;
    if (Offset < 0) {
      BaseOpc = !isThumb2 ? ARM::SUBri : ARM::t2SUBri;
      Offset = - Offset;
    }
    int ImmedOffset = isThumb2
      ? ARM_AM::getT2SOImmVal(Offset) : ARM_AM::getSOImmVal(Offset);
    if (ImmedOffset == -1)
      // FIXME: Try t2ADDri12 or t2SUBri12?
      return false;  // Probably not worth it then.

    BuildMI(MBB, MBBI, dl, TII->get(BaseOpc), NewBase)
      .addReg(Base, getKillRegState(BaseKill)).addImm(Offset)
      .addImm(Pred).addReg(PredReg).addReg(0);
    Base = NewBase;
    BaseKill = true;  // New base is always killed right its use.
  }

  bool isDef = (isi32Load(Opcode) || Opcode == ARM::VLDRS ||
                Opcode == ARM::VLDRD);
  Opcode = getLoadStoreMultipleOpcode(Opcode, Mode);
  if (!Opcode) return false;
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII->get(Opcode))
    .addReg(Base, getKillRegState(BaseKill))
    .addImm(Pred).addReg(PredReg);
  for (unsigned i = 0; i != NumRegs; ++i)
    MIB = MIB.addReg(Regs[i].first, getDefRegState(isDef)
                     | getKillRegState(Regs[i].second));

  // Add implicit defs for super-registers.
  for (unsigned i = 0, e = ImpDefs.size(); i != e; ++i)
    MIB.addReg(ImpDefs[i], RegState::ImplicitDefine);

  return true;
}

// MergeOpsUpdate - call MergeOps and update MemOps and merges accordingly on
// success.
void ARMLoadStoreOpt::MergeOpsUpdate(MachineBasicBlock &MBB,
                                     MemOpQueue &memOps,
                                     unsigned memOpsBegin, unsigned memOpsEnd,
                                     unsigned insertAfter, int Offset,
                                     unsigned Base, bool BaseKill,
                                     int Opcode,
                                     ARMCC::CondCodes Pred, unsigned PredReg,
                                     unsigned Scratch,
                                     DebugLoc dl,
                          SmallVector<MachineBasicBlock::iterator, 4> &Merges) {
  // First calculate which of the registers should be killed by the merged
  // instruction.
  const unsigned insertPos = memOps[insertAfter].Position;
  SmallSet<unsigned, 4> KilledRegs;
  DenseMap<unsigned, unsigned> Killer;
  for (unsigned i = 0, e = memOps.size(); i != e; ++i) {
    if (i == memOpsBegin) {
      i = memOpsEnd;
      if (i == e)
        break;
    }
    if (memOps[i].Position < insertPos && memOps[i].isKill) {
      unsigned Reg = memOps[i].Reg;
      KilledRegs.insert(Reg);
      Killer[Reg] = i;
    }
  }

  SmallVector<std::pair<unsigned, bool>, 8> Regs;
  SmallVector<unsigned, 8> ImpDefs;
  for (unsigned i = memOpsBegin; i < memOpsEnd; ++i) {
    unsigned Reg = memOps[i].Reg;
    // If we are inserting the merged operation after an operation that
    // uses the same register, make sure to transfer any kill flag.
    bool isKill = memOps[i].isKill || KilledRegs.count(Reg);
    Regs.push_back(std::make_pair(Reg, isKill));

    // Collect any implicit defs of super-registers. They must be preserved.
    for (MIOperands MO(memOps[i].MBBI); MO.isValid(); ++MO) {
      if (!MO->isReg() || !MO->isDef() || !MO->isImplicit() || MO->isDead())
        continue;
      unsigned DefReg = MO->getReg();
      if (std::find(ImpDefs.begin(), ImpDefs.end(), DefReg) == ImpDefs.end())
        ImpDefs.push_back(DefReg);
    }
  }

  // Try to do the merge.
  MachineBasicBlock::iterator Loc = memOps[insertAfter].MBBI;
  ++Loc;
  if (!MergeOps(MBB, Loc, Offset, Base, BaseKill, Opcode,
                Pred, PredReg, Scratch, dl, Regs, ImpDefs))
    return;

  // Merge succeeded, update records.
  Merges.push_back(prior(Loc));
  for (unsigned i = memOpsBegin; i < memOpsEnd; ++i) {
    // Remove kill flags from any memops that come before insertPos.
    if (Regs[i-memOpsBegin].second) {
      unsigned Reg = Regs[i-memOpsBegin].first;
      if (KilledRegs.count(Reg)) {
        unsigned j = Killer[Reg];
        int Idx = memOps[j].MBBI->findRegisterUseOperandIdx(Reg, true);
        assert(Idx >= 0 && "Cannot find killing operand");
        memOps[j].MBBI->getOperand(Idx).setIsKill(false);
        memOps[j].isKill = false;
      }
      memOps[i].isKill = true;
    }
    MBB.erase(memOps[i].MBBI);
    // Update this memop to refer to the merged instruction.
    // We may need to move kill flags again.
    memOps[i].Merged = true;
    memOps[i].MBBI = Merges.back();
    memOps[i].Position = insertPos;
  }
}

/// MergeLDR_STR - Merge a number of load / store instructions into one or more
/// load / store multiple instructions.
void
ARMLoadStoreOpt::MergeLDR_STR(MachineBasicBlock &MBB, unsigned SIndex,
                          unsigned Base, int Opcode, unsigned Size,
                          ARMCC::CondCodes Pred, unsigned PredReg,
                          unsigned Scratch, MemOpQueue &MemOps,
                          SmallVector<MachineBasicBlock::iterator, 4> &Merges) {
  bool isNotVFP = isi32Load(Opcode) || isi32Store(Opcode);
  int Offset = MemOps[SIndex].Offset;
  int SOffset = Offset;
  unsigned insertAfter = SIndex;
  MachineBasicBlock::iterator Loc = MemOps[SIndex].MBBI;
  DebugLoc dl = Loc->getDebugLoc();
  const MachineOperand &PMO = Loc->getOperand(0);
  unsigned PReg = PMO.getReg();
  unsigned PRegNum = PMO.isUndef() ? UINT_MAX : TRI->getEncodingValue(PReg);
  unsigned Count = 1;
  unsigned Limit = ~0U;

  // vldm / vstm limit are 32 for S variants, 16 for D variants.

  switch (Opcode) {
  default: break;
  case ARM::VSTRS:
    Limit = 32;
    break;
  case ARM::VSTRD:
    Limit = 16;
    break;
  case ARM::VLDRD:
    Limit = 16;
    break;
  case ARM::VLDRS:
    Limit = 32;
    break;
  }

  for (unsigned i = SIndex+1, e = MemOps.size(); i != e; ++i) {
    int NewOffset = MemOps[i].Offset;
    const MachineOperand &MO = MemOps[i].MBBI->getOperand(0);
    unsigned Reg = MO.getReg();
    unsigned RegNum = MO.isUndef() ? UINT_MAX : TRI->getEncodingValue(Reg);
    // Register numbers must be in ascending order. For VFP / NEON load and
    // store multiples, the registers must also be consecutive and within the
    // limit on the number of registers per instruction.
    if (Reg != ARM::SP &&
        NewOffset == Offset + (int)Size &&
        ((isNotVFP && RegNum > PRegNum) ||
         ((Count < Limit) && RegNum == PRegNum+1))) {
      Offset += Size;
      PRegNum = RegNum;
      ++Count;
    } else {
      // Can't merge this in. Try merge the earlier ones first.
      MergeOpsUpdate(MBB, MemOps, SIndex, i, insertAfter, SOffset,
                     Base, false, Opcode, Pred, PredReg, Scratch, dl, Merges);
      MergeLDR_STR(MBB, i, Base, Opcode, Size, Pred, PredReg, Scratch,
                   MemOps, Merges);
      return;
    }

    if (MemOps[i].Position > MemOps[insertAfter].Position)
      insertAfter = i;
  }

  bool BaseKill = Loc->findRegisterUseOperandIdx(Base, true) != -1;
  MergeOpsUpdate(MBB, MemOps, SIndex, MemOps.size(), insertAfter, SOffset,
                 Base, BaseKill, Opcode, Pred, PredReg, Scratch, dl, Merges);
  return;
}

static bool definesCPSR(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    if (MO.isDef() && MO.getReg() == ARM::CPSR && !MO.isDead())
      // If the instruction has live CPSR def, then it's not safe to fold it
      // into load / store.
      return true;
  }

  return false;
}

static bool isMatchingDecrement(MachineInstr *MI, unsigned Base,
                                unsigned Bytes, unsigned Limit,
                                ARMCC::CondCodes Pred, unsigned PredReg) {
  unsigned MyPredReg = 0;
  if (!MI)
    return false;

  bool CheckCPSRDef = false;
  switch (MI->getOpcode()) {
  default: return false;
  case ARM::t2SUBri:
  case ARM::SUBri:
    CheckCPSRDef = true;
  // fallthrough
  case ARM::tSUBspi:
    break;
  }

  // Make sure the offset fits in 8 bits.
  if (Bytes == 0 || (Limit && Bytes >= Limit))
    return false;

  unsigned Scale = (MI->getOpcode() == ARM::tSUBspi) ? 4 : 1; // FIXME
  if (!(MI->getOperand(0).getReg() == Base &&
        MI->getOperand(1).getReg() == Base &&
        (MI->getOperand(2).getImm()*Scale) == Bytes &&
        getInstrPredicate(MI, MyPredReg) == Pred &&
        MyPredReg == PredReg))
    return false;

  return CheckCPSRDef ? !definesCPSR(MI) : true;
}

static bool isMatchingIncrement(MachineInstr *MI, unsigned Base,
                                unsigned Bytes, unsigned Limit,
                                ARMCC::CondCodes Pred, unsigned PredReg) {
  unsigned MyPredReg = 0;
  if (!MI)
    return false;

  bool CheckCPSRDef = false;
  switch (MI->getOpcode()) {
  default: return false;
  case ARM::t2ADDri:
  case ARM::ADDri:
    CheckCPSRDef = true;
  // fallthrough
  case ARM::tADDspi:
    break;
  }

  if (Bytes == 0 || (Limit && Bytes >= Limit))
    // Make sure the offset fits in 8 bits.
    return false;

  unsigned Scale = (MI->getOpcode() == ARM::tADDspi) ? 4 : 1; // FIXME
  if (!(MI->getOperand(0).getReg() == Base &&
        MI->getOperand(1).getReg() == Base &&
        (MI->getOperand(2).getImm()*Scale) == Bytes &&
        getInstrPredicate(MI, MyPredReg) == Pred &&
        MyPredReg == PredReg))
    return false;

  return CheckCPSRDef ? !definesCPSR(MI) : true;
}

static inline unsigned getLSMultipleTransferSize(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default: return 0;
  case ARM::LDRi12:
  case ARM::STRi12:
  case ARM::t2LDRi8:
  case ARM::t2LDRi12:
  case ARM::t2STRi8:
  case ARM::t2STRi12:
  case ARM::VLDRS:
  case ARM::VSTRS:
    return 4;
  case ARM::VLDRD:
  case ARM::VSTRD:
    return 8;
  case ARM::LDMIA:
  case ARM::LDMDA:
  case ARM::LDMDB:
  case ARM::LDMIB:
  case ARM::STMIA:
  case ARM::STMDA:
  case ARM::STMDB:
  case ARM::STMIB:
  case ARM::t2LDMIA:
  case ARM::t2LDMDB:
  case ARM::t2STMIA:
  case ARM::t2STMDB:
  case ARM::VLDMSIA:
  case ARM::VSTMSIA:
    return (MI->getNumOperands() - MI->getDesc().getNumOperands() + 1) * 4;
  case ARM::VLDMDIA:
  case ARM::VSTMDIA:
    return (MI->getNumOperands() - MI->getDesc().getNumOperands() + 1) * 8;
  }
}

static unsigned getUpdatingLSMultipleOpcode(unsigned Opc,
                                            ARM_AM::AMSubMode Mode) {
  switch (Opc) {
  default: llvm_unreachable("Unhandled opcode!");
  case ARM::LDMIA:
  case ARM::LDMDA:
  case ARM::LDMDB:
  case ARM::LDMIB:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::LDMIA_UPD;
    case ARM_AM::ib: return ARM::LDMIB_UPD;
    case ARM_AM::da: return ARM::LDMDA_UPD;
    case ARM_AM::db: return ARM::LDMDB_UPD;
    }
  case ARM::STMIA:
  case ARM::STMDA:
  case ARM::STMDB:
  case ARM::STMIB:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::STMIA_UPD;
    case ARM_AM::ib: return ARM::STMIB_UPD;
    case ARM_AM::da: return ARM::STMDA_UPD;
    case ARM_AM::db: return ARM::STMDB_UPD;
    }
  case ARM::t2LDMIA:
  case ARM::t2LDMDB:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::t2LDMIA_UPD;
    case ARM_AM::db: return ARM::t2LDMDB_UPD;
    }
  case ARM::t2STMIA:
  case ARM::t2STMDB:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::t2STMIA_UPD;
    case ARM_AM::db: return ARM::t2STMDB_UPD;
    }
  case ARM::VLDMSIA:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VLDMSIA_UPD;
    case ARM_AM::db: return ARM::VLDMSDB_UPD;
    }
  case ARM::VLDMDIA:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VLDMDIA_UPD;
    case ARM_AM::db: return ARM::VLDMDDB_UPD;
    }
  case ARM::VSTMSIA:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VSTMSIA_UPD;
    case ARM_AM::db: return ARM::VSTMSDB_UPD;
    }
  case ARM::VSTMDIA:
    switch (Mode) {
    default: llvm_unreachable("Unhandled submode!");
    case ARM_AM::ia: return ARM::VSTMDIA_UPD;
    case ARM_AM::db: return ARM::VSTMDDB_UPD;
    }
  }
}

/// MergeBaseUpdateLSMultiple - Fold proceeding/trailing inc/dec of base
/// register into the LDM/STM/VLDM{D|S}/VSTM{D|S} op when possible:
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
bool ARMLoadStoreOpt::MergeBaseUpdateLSMultiple(MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator MBBI,
                                               bool &Advance,
                                               MachineBasicBlock::iterator &I) {
  MachineInstr *MI = MBBI;
  unsigned Base = MI->getOperand(0).getReg();
  bool BaseKill = MI->getOperand(0).isKill();
  unsigned Bytes = getLSMultipleTransferSize(MI);
  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  int Opcode = MI->getOpcode();
  DebugLoc dl = MI->getDebugLoc();

  // Can't use an updating ld/st if the base register is also a dest
  // register. e.g. ldmdb r0!, {r0, r1, r2}. The behavior is undefined.
  for (unsigned i = 2, e = MI->getNumOperands(); i != e; ++i)
    if (MI->getOperand(i).getReg() == Base)
      return false;

  bool DoMerge = false;
  ARM_AM::AMSubMode Mode = ARM_AM::getLoadStoreMultipleSubMode(Opcode);

  // Try merging with the previous instruction.
  MachineBasicBlock::iterator BeginMBBI = MBB.begin();
  if (MBBI != BeginMBBI) {
    MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
    while (PrevMBBI != BeginMBBI && PrevMBBI->isDebugValue())
      --PrevMBBI;
    if (Mode == ARM_AM::ia &&
        isMatchingDecrement(PrevMBBI, Base, Bytes, 0, Pred, PredReg)) {
      Mode = ARM_AM::db;
      DoMerge = true;
    } else if (Mode == ARM_AM::ib &&
               isMatchingDecrement(PrevMBBI, Base, Bytes, 0, Pred, PredReg)) {
      Mode = ARM_AM::da;
      DoMerge = true;
    }
    if (DoMerge)
      MBB.erase(PrevMBBI);
  }

  // Try merging with the next instruction.
  MachineBasicBlock::iterator EndMBBI = MBB.end();
  if (!DoMerge && MBBI != EndMBBI) {
    MachineBasicBlock::iterator NextMBBI = llvm::next(MBBI);
    while (NextMBBI != EndMBBI && NextMBBI->isDebugValue())
      ++NextMBBI;
    if ((Mode == ARM_AM::ia || Mode == ARM_AM::ib) &&
        isMatchingIncrement(NextMBBI, Base, Bytes, 0, Pred, PredReg)) {
      DoMerge = true;
    } else if ((Mode == ARM_AM::da || Mode == ARM_AM::db) &&
               isMatchingDecrement(NextMBBI, Base, Bytes, 0, Pred, PredReg)) {
      DoMerge = true;
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

  unsigned NewOpc = getUpdatingLSMultipleOpcode(Opcode, Mode);
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, dl, TII->get(NewOpc))
    .addReg(Base, getDefRegState(true)) // WB base register
    .addReg(Base, getKillRegState(BaseKill))
    .addImm(Pred).addReg(PredReg);

  // Transfer the rest of operands.
  for (unsigned OpNum = 3, e = MI->getNumOperands(); OpNum != e; ++OpNum)
    MIB.addOperand(MI->getOperand(OpNum));

  // Transfer memoperands.
  MIB->setMemRefs(MI->memoperands_begin(), MI->memoperands_end());

  MBB.erase(MBBI);
  return true;
}

static unsigned getPreIndexedLoadStoreOpcode(unsigned Opc,
                                             ARM_AM::AddrOpc Mode) {
  switch (Opc) {
  case ARM::LDRi12:
    return ARM::LDR_PRE_IMM;
  case ARM::STRi12:
    return ARM::STR_PRE_IMM;
  case ARM::VLDRS:
    return Mode == ARM_AM::add ? ARM::VLDMSIA_UPD : ARM::VLDMSDB_UPD;
  case ARM::VLDRD:
    return Mode == ARM_AM::add ? ARM::VLDMDIA_UPD : ARM::VLDMDDB_UPD;
  case ARM::VSTRS:
    return Mode == ARM_AM::add ? ARM::VSTMSIA_UPD : ARM::VSTMSDB_UPD;
  case ARM::VSTRD:
    return Mode == ARM_AM::add ? ARM::VSTMDIA_UPD : ARM::VSTMDDB_UPD;
  case ARM::t2LDRi8:
  case ARM::t2LDRi12:
    return ARM::t2LDR_PRE;
  case ARM::t2STRi8:
  case ARM::t2STRi12:
    return ARM::t2STR_PRE;
  default: llvm_unreachable("Unhandled opcode!");
  }
}

static unsigned getPostIndexedLoadStoreOpcode(unsigned Opc,
                                              ARM_AM::AddrOpc Mode) {
  switch (Opc) {
  case ARM::LDRi12:
    return ARM::LDR_POST_IMM;
  case ARM::STRi12:
    return ARM::STR_POST_IMM;
  case ARM::VLDRS:
    return Mode == ARM_AM::add ? ARM::VLDMSIA_UPD : ARM::VLDMSDB_UPD;
  case ARM::VLDRD:
    return Mode == ARM_AM::add ? ARM::VLDMDIA_UPD : ARM::VLDMDDB_UPD;
  case ARM::VSTRS:
    return Mode == ARM_AM::add ? ARM::VSTMSIA_UPD : ARM::VSTMSDB_UPD;
  case ARM::VSTRD:
    return Mode == ARM_AM::add ? ARM::VSTMDIA_UPD : ARM::VSTMDDB_UPD;
  case ARM::t2LDRi8:
  case ARM::t2LDRi12:
    return ARM::t2LDR_POST;
  case ARM::t2STRi8:
  case ARM::t2STRi12:
    return ARM::t2STR_POST;
  default: llvm_unreachable("Unhandled opcode!");
  }
}

/// MergeBaseUpdateLoadStore - Fold proceeding/trailing inc/dec of base
/// register into the LDR/STR/FLD{D|S}/FST{D|S} op when possible:
bool ARMLoadStoreOpt::MergeBaseUpdateLoadStore(MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator MBBI,
                                               const TargetInstrInfo *TII,
                                               bool &Advance,
                                               MachineBasicBlock::iterator &I) {
  MachineInstr *MI = MBBI;
  unsigned Base = MI->getOperand(1).getReg();
  bool BaseKill = MI->getOperand(1).isKill();
  unsigned Bytes = getLSMultipleTransferSize(MI);
  int Opcode = MI->getOpcode();
  DebugLoc dl = MI->getDebugLoc();
  bool isAM5 = (Opcode == ARM::VLDRD || Opcode == ARM::VLDRS ||
                Opcode == ARM::VSTRD || Opcode == ARM::VSTRS);
  bool isAM2 = (Opcode == ARM::LDRi12 || Opcode == ARM::STRi12);
  if (isi32Load(Opcode) || isi32Store(Opcode))
    if (MI->getOperand(2).getImm() != 0)
      return false;
  if (isAM5 && ARM_AM::getAM5Offset(MI->getOperand(2).getImm()) != 0)
    return false;

  bool isLd = isi32Load(Opcode) || Opcode == ARM::VLDRS || Opcode == ARM::VLDRD;
  // Can't do the merge if the destination register is the same as the would-be
  // writeback register.
  if (isLd && MI->getOperand(0).getReg() == Base)
    return false;

  unsigned PredReg = 0;
  ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);
  bool DoMerge = false;
  ARM_AM::AddrOpc AddSub = ARM_AM::add;
  unsigned NewOpc = 0;
  // AM2 - 12 bits, thumb2 - 8 bits.
  unsigned Limit = isAM5 ? 0 : (isAM2 ? 0x1000 : 0x100);

  // Try merging with the previous instruction.
  MachineBasicBlock::iterator BeginMBBI = MBB.begin();
  if (MBBI != BeginMBBI) {
    MachineBasicBlock::iterator PrevMBBI = prior(MBBI);
    while (PrevMBBI != BeginMBBI && PrevMBBI->isDebugValue())
      --PrevMBBI;
    if (isMatchingDecrement(PrevMBBI, Base, Bytes, Limit, Pred, PredReg)) {
      DoMerge = true;
      AddSub = ARM_AM::sub;
    } else if (!isAM5 &&
               isMatchingIncrement(PrevMBBI, Base, Bytes, Limit,Pred,PredReg)) {
      DoMerge = true;
    }
    if (DoMerge) {
      NewOpc = getPreIndexedLoadStoreOpcode(Opcode, AddSub);
      MBB.erase(PrevMBBI);
    }
  }

  // Try merging with the next instruction.
  MachineBasicBlock::iterator EndMBBI = MBB.end();
  if (!DoMerge && MBBI != EndMBBI) {
    MachineBasicBlock::iterator NextMBBI = llvm::next(MBBI);
    while (NextMBBI != EndMBBI && NextMBBI->isDebugValue())
      ++NextMBBI;
    if (!isAM5 &&
        isMatchingDecrement(NextMBBI, Base, Bytes, Limit, Pred, PredReg)) {
      DoMerge = true;
      AddSub = ARM_AM::sub;
    } else if (isMatchingIncrement(NextMBBI, Base, Bytes, Limit,Pred,PredReg)) {
      DoMerge = true;
    }
    if (DoMerge) {
      NewOpc = getPostIndexedLoadStoreOpcode(Opcode, AddSub);
      if (NextMBBI == I) {
        Advance = true;
        ++I;
      }
      MBB.erase(NextMBBI);
    }
  }

  if (!DoMerge)
    return false;

  if (isAM5) {
    // VLDM[SD}_UPD, VSTM[SD]_UPD
    // (There are no base-updating versions of VLDR/VSTR instructions, but the
    // updating load/store-multiple instructions can be used with only one
    // register.)
    MachineOperand &MO = MI->getOperand(0);
    BuildMI(MBB, MBBI, dl, TII->get(NewOpc))
      .addReg(Base, getDefRegState(true)) // WB base register
      .addReg(Base, getKillRegState(isLd ? BaseKill : false))
      .addImm(Pred).addReg(PredReg)
      .addReg(MO.getReg(), (isLd ? getDefRegState(true) :
                            getKillRegState(MO.isKill())));
  } else if (isLd) {
    if (isAM2) {
      // LDR_PRE, LDR_POST
      if (NewOpc == ARM::LDR_PRE_IMM || NewOpc == ARM::LDRB_PRE_IMM) {
        int Offset = AddSub == ARM_AM::sub ? -Bytes : Bytes;
        BuildMI(MBB, MBBI, dl, TII->get(NewOpc), MI->getOperand(0).getReg())
          .addReg(Base, RegState::Define)
          .addReg(Base).addImm(Offset).addImm(Pred).addReg(PredReg);
      } else {
        int Offset = ARM_AM::getAM2Opc(AddSub, Bytes, ARM_AM::no_shift);
        BuildMI(MBB, MBBI, dl, TII->get(NewOpc), MI->getOperand(0).getReg())
          .addReg(Base, RegState::Define)
          .addReg(Base).addReg(0).addImm(Offset).addImm(Pred).addReg(PredReg);
      }
    } else {
      int Offset = AddSub == ARM_AM::sub ? -Bytes : Bytes;
      // t2LDR_PRE, t2LDR_POST
      BuildMI(MBB, MBBI, dl, TII->get(NewOpc), MI->getOperand(0).getReg())
        .addReg(Base, RegState::Define)
        .addReg(Base).addImm(Offset).addImm(Pred).addReg(PredReg);
    }
  } else {
    MachineOperand &MO = MI->getOperand(0);
    // FIXME: post-indexed stores use am2offset_imm, which still encodes
    // the vestigal zero-reg offset register. When that's fixed, this clause
    // can be removed entirely.
    if (isAM2 && NewOpc == ARM::STR_POST_IMM) {
      int Offset = ARM_AM::getAM2Opc(AddSub, Bytes, ARM_AM::no_shift);
      // STR_PRE, STR_POST
      BuildMI(MBB, MBBI, dl, TII->get(NewOpc), Base)
        .addReg(MO.getReg(), getKillRegState(MO.isKill()))
        .addReg(Base).addReg(0).addImm(Offset).addImm(Pred).addReg(PredReg);
    } else {
      int Offset = AddSub == ARM_AM::sub ? -Bytes : Bytes;
      // t2STR_PRE, t2STR_POST
      BuildMI(MBB, MBBI, dl, TII->get(NewOpc), Base)
        .addReg(MO.getReg(), getKillRegState(MO.isKill()))
        .addReg(Base).addImm(Offset).addImm(Pred).addReg(PredReg);
    }
  }
  MBB.erase(MBBI);

  return true;
}

/// isMemoryOp - Returns true if instruction is a memory operation that this
/// pass is capable of operating on.
static bool isMemoryOp(const MachineInstr *MI) {
  // When no memory operands are present, conservatively assume unaligned,
  // volatile, unfoldable.
  if (!MI->hasOneMemOperand())
    return false;

  const MachineMemOperand *MMO = *MI->memoperands_begin();

  // Don't touch volatile memory accesses - we may be changing their order.
  if (MMO->isVolatile())
    return false;

  // Unaligned ldr/str is emulated by some kernels, but unaligned ldm/stm is
  // not.
  if (MMO->getAlignment() < 4)
    return false;

  // str <undef> could probably be eliminated entirely, but for now we just want
  // to avoid making a mess of it.
  // FIXME: Use str <undef> as a wildcard to enable better stm folding.
  if (MI->getNumOperands() > 0 && MI->getOperand(0).isReg() &&
      MI->getOperand(0).isUndef())
    return false;

  // Likewise don't mess with references to undefined addresses.
  if (MI->getNumOperands() > 1 && MI->getOperand(1).isReg() &&
      MI->getOperand(1).isUndef())
    return false;

  int Opcode = MI->getOpcode();
  switch (Opcode) {
  default: break;
  case ARM::VLDRS:
  case ARM::VSTRS:
    return MI->getOperand(1).isReg();
  case ARM::VLDRD:
  case ARM::VSTRD:
    return MI->getOperand(1).isReg();
  case ARM::LDRi12:
  case ARM::STRi12:
  case ARM::t2LDRi8:
  case ARM::t2LDRi12:
  case ARM::t2STRi8:
  case ARM::t2STRi12:
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

static int getMemoryOpOffset(const MachineInstr *MI) {
  int Opcode = MI->getOpcode();
  bool isAM3 = Opcode == ARM::LDRD || Opcode == ARM::STRD;
  unsigned NumOperands = MI->getDesc().getNumOperands();
  unsigned OffField = MI->getOperand(NumOperands-3).getImm();

  if (Opcode == ARM::t2LDRi12 || Opcode == ARM::t2LDRi8 ||
      Opcode == ARM::t2STRi12 || Opcode == ARM::t2STRi8 ||
      Opcode == ARM::t2LDRDi8 || Opcode == ARM::t2STRDi8 ||
      Opcode == ARM::LDRi12   || Opcode == ARM::STRi12)
    return OffField;

  int Offset = isAM3 ? ARM_AM::getAM3Offset(OffField)
    : ARM_AM::getAM5Offset(OffField) * 4;
  if (isAM3) {
    if (ARM_AM::getAM3Op(OffField) == ARM_AM::sub)
      Offset = -Offset;
  } else {
    if (ARM_AM::getAM5Op(OffField) == ARM_AM::sub)
      Offset = -Offset;
  }
  return Offset;
}

static void InsertLDR_STR(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator &MBBI,
                          int Offset, bool isDef,
                          DebugLoc dl, unsigned NewOpc,
                          unsigned Reg, bool RegDeadKill, bool RegUndef,
                          unsigned BaseReg, bool BaseKill, bool BaseUndef,
                          bool OffKill, bool OffUndef,
                          ARMCC::CondCodes Pred, unsigned PredReg,
                          const TargetInstrInfo *TII, bool isT2) {
  if (isDef) {
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MBBI->getDebugLoc(),
                                      TII->get(NewOpc))
      .addReg(Reg, getDefRegState(true) | getDeadRegState(RegDeadKill))
      .addReg(BaseReg, getKillRegState(BaseKill)|getUndefRegState(BaseUndef));
    MIB.addImm(Offset).addImm(Pred).addReg(PredReg);
  } else {
    MachineInstrBuilder MIB = BuildMI(MBB, MBBI, MBBI->getDebugLoc(),
                                      TII->get(NewOpc))
      .addReg(Reg, getKillRegState(RegDeadKill) | getUndefRegState(RegUndef))
      .addReg(BaseReg, getKillRegState(BaseKill)|getUndefRegState(BaseUndef));
    MIB.addImm(Offset).addImm(Pred).addReg(PredReg);
  }
}

bool ARMLoadStoreOpt::FixInvalidRegPairOp(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator &MBBI) {
  MachineInstr *MI = &*MBBI;
  unsigned Opcode = MI->getOpcode();
  if (Opcode == ARM::LDRD || Opcode == ARM::STRD ||
      Opcode == ARM::t2LDRDi8 || Opcode == ARM::t2STRDi8) {
    const MachineOperand &BaseOp = MI->getOperand(2);
    unsigned BaseReg = BaseOp.getReg();
    unsigned EvenReg = MI->getOperand(0).getReg();
    unsigned OddReg  = MI->getOperand(1).getReg();
    unsigned EvenRegNum = TRI->getDwarfRegNum(EvenReg, false);
    unsigned OddRegNum  = TRI->getDwarfRegNum(OddReg, false);
    // ARM errata 602117: LDRD with base in list may result in incorrect base
    // register when interrupted or faulted.
    bool Errata602117 = EvenReg == BaseReg && STI->isCortexM3();
    if (!Errata602117 &&
        ((EvenRegNum & 1) == 0 && (EvenRegNum + 1) == OddRegNum))
      return false;

    MachineBasicBlock::iterator NewBBI = MBBI;
    bool isT2 = Opcode == ARM::t2LDRDi8 || Opcode == ARM::t2STRDi8;
    bool isLd = Opcode == ARM::LDRD || Opcode == ARM::t2LDRDi8;
    bool EvenDeadKill = isLd ?
      MI->getOperand(0).isDead() : MI->getOperand(0).isKill();
    bool EvenUndef = MI->getOperand(0).isUndef();
    bool OddDeadKill  = isLd ?
      MI->getOperand(1).isDead() : MI->getOperand(1).isKill();
    bool OddUndef = MI->getOperand(1).isUndef();
    bool BaseKill = BaseOp.isKill();
    bool BaseUndef = BaseOp.isUndef();
    bool OffKill = isT2 ? false : MI->getOperand(3).isKill();
    bool OffUndef = isT2 ? false : MI->getOperand(3).isUndef();
    int OffImm = getMemoryOpOffset(MI);
    unsigned PredReg = 0;
    ARMCC::CondCodes Pred = getInstrPredicate(MI, PredReg);

    if (OddRegNum > EvenRegNum && OffImm == 0) {
      // Ascending register numbers and no offset. It's safe to change it to a
      // ldm or stm.
      unsigned NewOpc = (isLd)
        ? (isT2 ? ARM::t2LDMIA : ARM::LDMIA)
        : (isT2 ? ARM::t2STMIA : ARM::STMIA);
      if (isLd) {
        BuildMI(MBB, MBBI, MBBI->getDebugLoc(), TII->get(NewOpc))
          .addReg(BaseReg, getKillRegState(BaseKill))
          .addImm(Pred).addReg(PredReg)
          .addReg(EvenReg, getDefRegState(isLd) | getDeadRegState(EvenDeadKill))
          .addReg(OddReg,  getDefRegState(isLd) | getDeadRegState(OddDeadKill));
        ++NumLDRD2LDM;
      } else {
        BuildMI(MBB, MBBI, MBBI->getDebugLoc(), TII->get(NewOpc))
          .addReg(BaseReg, getKillRegState(BaseKill))
          .addImm(Pred).addReg(PredReg)
          .addReg(EvenReg,
                  getKillRegState(EvenDeadKill) | getUndefRegState(EvenUndef))
          .addReg(OddReg,
                  getKillRegState(OddDeadKill)  | getUndefRegState(OddUndef));
        ++NumSTRD2STM;
      }
      NewBBI = llvm::prior(MBBI);
    } else {
      // Split into two instructions.
      unsigned NewOpc = (isLd)
        ? (isT2 ? (OffImm < 0 ? ARM::t2LDRi8 : ARM::t2LDRi12) : ARM::LDRi12)
        : (isT2 ? (OffImm < 0 ? ARM::t2STRi8 : ARM::t2STRi12) : ARM::STRi12);
      // Be extra careful for thumb2. t2LDRi8 can't reference a zero offset,
      // so adjust and use t2LDRi12 here for that.
      unsigned NewOpc2 = (isLd)
        ? (isT2 ? (OffImm+4 < 0 ? ARM::t2LDRi8 : ARM::t2LDRi12) : ARM::LDRi12)
        : (isT2 ? (OffImm+4 < 0 ? ARM::t2STRi8 : ARM::t2STRi12) : ARM::STRi12);
      DebugLoc dl = MBBI->getDebugLoc();
      // If this is a load and base register is killed, it may have been
      // re-defed by the load, make sure the first load does not clobber it.
      if (isLd &&
          (BaseKill || OffKill) &&
          (TRI->regsOverlap(EvenReg, BaseReg))) {
        assert(!TRI->regsOverlap(OddReg, BaseReg));
        InsertLDR_STR(MBB, MBBI, OffImm+4, isLd, dl, NewOpc2,
                      OddReg, OddDeadKill, false,
                      BaseReg, false, BaseUndef, false, OffUndef,
                      Pred, PredReg, TII, isT2);
        NewBBI = llvm::prior(MBBI);
        InsertLDR_STR(MBB, MBBI, OffImm, isLd, dl, NewOpc,
                      EvenReg, EvenDeadKill, false,
                      BaseReg, BaseKill, BaseUndef, OffKill, OffUndef,
                      Pred, PredReg, TII, isT2);
      } else {
        if (OddReg == EvenReg && EvenDeadKill) {
          // If the two source operands are the same, the kill marker is
          // probably on the first one. e.g.
          // t2STRDi8 %R5<kill>, %R5, %R9<kill>, 0, 14, %reg0
          EvenDeadKill = false;
          OddDeadKill = true;
        }
        // Never kill the base register in the first instruction.
        // <rdar://problem/11101911>
        if (EvenReg == BaseReg)
          EvenDeadKill = false;
        InsertLDR_STR(MBB, MBBI, OffImm, isLd, dl, NewOpc,
                      EvenReg, EvenDeadKill, EvenUndef,
                      BaseReg, false, BaseUndef, false, OffUndef,
                      Pred, PredReg, TII, isT2);
        NewBBI = llvm::prior(MBBI);
        InsertLDR_STR(MBB, MBBI, OffImm+4, isLd, dl, NewOpc2,
                      OddReg, OddDeadKill, OddUndef,
                      BaseReg, BaseKill, BaseUndef, OffKill, OffUndef,
                      Pred, PredReg, TII, isT2);
      }
      if (isLd)
        ++NumLDRD2LDR;
      else
        ++NumSTRD2STR;
    }

    MBB.erase(MI);
    MBBI = NewBBI;
    return true;
  }
  return false;
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
  SmallVector<MachineBasicBlock::iterator,4> Merges;

  RS->enterBasicBlock(&MBB);
  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    if (FixInvalidRegPairOp(MBB, MBBI))
      continue;

    bool Advance  = false;
    bool TryMerge = false;
    bool Clobber  = false;

    bool isMemOp = isMemoryOp(MBBI);
    if (isMemOp) {
      int Opcode = MBBI->getOpcode();
      unsigned Size = getLSMultipleTransferSize(MBBI);
      const MachineOperand &MO = MBBI->getOperand(0);
      unsigned Reg = MO.getReg();
      bool isKill = MO.isDef() ? false : MO.isKill();
      unsigned Base = MBBI->getOperand(1).getReg();
      unsigned PredReg = 0;
      ARMCC::CondCodes Pred = getInstrPredicate(MBBI, PredReg);
      int Offset = getMemoryOpOffset(MBBI);
      // Watch out for:
      // r4 := ldr [r5]
      // r5 := ldr [r5, #4]
      // r6 := ldr [r5, #8]
      //
      // The second ldr has effectively broken the chain even though it
      // looks like the later ldr(s) use the same base register. Try to
      // merge the ldr's so far, including this one. But don't try to
      // combine the following ldr(s).
      Clobber = (isi32Load(Opcode) && Base == MBBI->getOperand(0).getReg());
      if (CurrBase == 0 && !Clobber) {
        // Start of a new chain.
        CurrBase = Base;
        CurrOpc  = Opcode;
        CurrSize = Size;
        CurrPred = Pred;
        CurrPredReg = PredReg;
        MemOps.push_back(MemOpQueueEntry(Offset, Reg, isKill, Position, MBBI));
        ++NumMemOps;
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
            MemOps.push_back(MemOpQueueEntry(Offset, Reg, isKill,
                                             Position, MBBI));
            ++NumMemOps;
            Advance = true;
          } else {
            for (MemOpQueueIter I = MemOps.begin(), E = MemOps.end();
                 I != E; ++I) {
              if (Offset < I->Offset) {
                MemOps.insert(I, MemOpQueueEntry(Offset, Reg, isKill,
                                                 Position, MBBI));
                ++NumMemOps;
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

    if (MBBI->isDebugValue()) {
      ++MBBI;
      if (MBBI == E)
        // Reach the end of the block, try merging the memory instructions.
        TryMerge = true;
    } else if (Advance) {
      ++Position;
      ++MBBI;
      if (MBBI == E)
        // Reach the end of the block, try merging the memory instructions.
        TryMerge = true;
    } else
      TryMerge = true;

    if (TryMerge) {
      if (NumMemOps > 1) {
        // Try to find a free register to use as a new base in case it's needed.
        // First advance to the instruction just before the start of the chain.
        AdvanceRS(MBB, MemOps);
        // Find a scratch register.
        unsigned Scratch = RS->FindUnusedReg(&ARM::GPRRegClass);
        // Process the load / store instructions.
        RS->forward(prior(MBBI));

        // Merge ops.
        Merges.clear();
        MergeLDR_STR(MBB, 0, CurrBase, CurrOpc, CurrSize,
                     CurrPred, CurrPredReg, Scratch, MemOps, Merges);

        // Try folding preceding/trailing base inc/dec into the generated
        // LDM/STM ops.
        for (unsigned i = 0, e = Merges.size(); i < e; ++i)
          if (MergeBaseUpdateLSMultiple(MBB, Merges[i], Advance, MBBI))
            ++NumMerges;
        NumMerges += Merges.size();

        // Try folding preceding/trailing base inc/dec into those load/store
        // that were not merged to form LDM/STM ops.
        for (unsigned i = 0; i != NumMemOps; ++i)
          if (!MemOps[i].Merged)
            if (MergeBaseUpdateLoadStore(MBB, MemOps[i].MBBI, TII,Advance,MBBI))
              ++NumMerges;

        // RS may be pointing to an instruction that's deleted.
        RS->skipTo(prior(MBBI));
      } else if (NumMemOps == 1) {
        // Try folding preceding/trailing base inc/dec into the single
        // load/store.
        if (MergeBaseUpdateLoadStore(MBB, MemOps[0].MBBI, TII, Advance, MBBI)) {
          ++NumMerges;
          RS->forward(prior(MBBI));
        }
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

/// MergeReturnIntoLDM - If this is a exit BB, try merging the return ops
/// ("bx lr" and "mov pc, lr") into the preceding stack restore so it
/// directly restore the value of LR into pc.
///   ldmfd sp!, {..., lr}
///   bx lr
/// or
///   ldmfd sp!, {..., lr}
///   mov pc, lr
/// =>
///   ldmfd sp!, {..., pc}
bool ARMLoadStoreOpt::MergeReturnIntoLDM(MachineBasicBlock &MBB) {
  if (MBB.empty()) return false;

  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  if (MBBI != MBB.begin() &&
      (MBBI->getOpcode() == ARM::BX_RET ||
       MBBI->getOpcode() == ARM::tBX_RET ||
       MBBI->getOpcode() == ARM::MOVPCLR)) {
    MachineInstr *PrevMI = prior(MBBI);
    unsigned Opcode = PrevMI->getOpcode();
    if (Opcode == ARM::LDMIA_UPD || Opcode == ARM::LDMDA_UPD ||
        Opcode == ARM::LDMDB_UPD || Opcode == ARM::LDMIB_UPD ||
        Opcode == ARM::t2LDMIA_UPD || Opcode == ARM::t2LDMDB_UPD) {
      MachineOperand &MO = PrevMI->getOperand(PrevMI->getNumOperands()-1);
      if (MO.getReg() != ARM::LR)
        return false;
      unsigned NewOpc = (isThumb2 ? ARM::t2LDMIA_RET : ARM::LDMIA_RET);
      assert(((isThumb2 && Opcode == ARM::t2LDMIA_UPD) ||
              Opcode == ARM::LDMIA_UPD) && "Unsupported multiple load-return!");
      PrevMI->setDesc(TII->get(NewOpc));
      MO.setReg(ARM::PC);
      PrevMI->copyImplicitOps(&*MBBI);
      MBB.erase(MBBI);
      return true;
    }
  }
  return false;
}

bool ARMLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  const TargetMachine &TM = Fn.getTarget();
  AFI = Fn.getInfo<ARMFunctionInfo>();
  TII = TM.getInstrInfo();
  TRI = TM.getRegisterInfo();
  STI = &TM.getSubtarget<ARMSubtarget>();
  RS = new RegScavenger();
  isThumb2 = AFI->isThumb2Function();

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= LoadStoreMultipleOpti(MBB);
    if (TM.getSubtarget<ARMSubtarget>().hasV5TOps())
      Modified |= MergeReturnIntoLDM(MBB);
  }

  delete RS;
  return Modified;
}


/// ARMPreAllocLoadStoreOpt - Pre- register allocation pass that move
/// load / stores from consecutive locations close to make it more
/// likely they will be combined later.

namespace {
  struct ARMPreAllocLoadStoreOpt : public MachineFunctionPass{
    static char ID;
    ARMPreAllocLoadStoreOpt() : MachineFunctionPass(ID) {}

    const DataLayout *TD;
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    const ARMSubtarget *STI;
    MachineRegisterInfo *MRI;
    MachineFunction *MF;

    virtual bool runOnMachineFunction(MachineFunction &Fn);

    virtual const char *getPassName() const {
      return "ARM pre- register allocation load / store optimization pass";
    }

  private:
    bool CanFormLdStDWord(MachineInstr *Op0, MachineInstr *Op1, DebugLoc &dl,
                          unsigned &NewOpc, unsigned &EvenReg,
                          unsigned &OddReg, unsigned &BaseReg,
                          int &Offset,
                          unsigned &PredReg, ARMCC::CondCodes &Pred,
                          bool &isT2);
    bool RescheduleOps(MachineBasicBlock *MBB,
                       SmallVector<MachineInstr*, 4> &Ops,
                       unsigned Base, bool isLd,
                       DenseMap<MachineInstr*, unsigned> &MI2LocMap);
    bool RescheduleLoadStoreInstrs(MachineBasicBlock *MBB);
  };
  char ARMPreAllocLoadStoreOpt::ID = 0;
}

bool ARMPreAllocLoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  TD  = Fn.getTarget().getDataLayout();
  TII = Fn.getTarget().getInstrInfo();
  TRI = Fn.getTarget().getRegisterInfo();
  STI = &Fn.getTarget().getSubtarget<ARMSubtarget>();
  MRI = &Fn.getRegInfo();
  MF  = &Fn;

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI)
    Modified |= RescheduleLoadStoreInstrs(MFI);

  return Modified;
}

static bool IsSafeAndProfitableToMove(bool isLd, unsigned Base,
                                      MachineBasicBlock::iterator I,
                                      MachineBasicBlock::iterator E,
                                      SmallPtrSet<MachineInstr*, 4> &MemOps,
                                      SmallSet<unsigned, 4> &MemRegs,
                                      const TargetRegisterInfo *TRI) {
  // Are there stores / loads / calls between them?
  // FIXME: This is overly conservative. We should make use of alias information
  // some day.
  SmallSet<unsigned, 4> AddedRegPressure;
  while (++I != E) {
    if (I->isDebugValue() || MemOps.count(&*I))
      continue;
    if (I->isCall() || I->isTerminator() || I->hasUnmodeledSideEffects())
      return false;
    if (isLd && I->mayStore())
      return false;
    if (!isLd) {
      if (I->mayLoad())
        return false;
      // It's not safe to move the first 'str' down.
      // str r1, [r0]
      // strh r5, [r0]
      // str r4, [r0, #+4]
      if (I->mayStore())
        return false;
    }
    for (unsigned j = 0, NumOps = I->getNumOperands(); j != NumOps; ++j) {
      MachineOperand &MO = I->getOperand(j);
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (MO.isDef() && TRI->regsOverlap(Reg, Base))
        return false;
      if (Reg != Base && !MemRegs.count(Reg))
        AddedRegPressure.insert(Reg);
    }
  }

  // Estimate register pressure increase due to the transformation.
  if (MemRegs.size() <= 4)
    // Ok if we are moving small number of instructions.
    return true;
  return AddedRegPressure.size() <= MemRegs.size() * 2;
}


/// Copy Op0 and Op1 operands into a new array assigned to MI.
static void concatenateMemOperands(MachineInstr *MI, MachineInstr *Op0,
                                   MachineInstr *Op1) {
  assert(MI->memoperands_empty() && "expected a new machineinstr");
  size_t numMemRefs = (Op0->memoperands_end() - Op0->memoperands_begin())
    + (Op1->memoperands_end() - Op1->memoperands_begin());

  MachineFunction *MF = MI->getParent()->getParent();
  MachineSDNode::mmo_iterator MemBegin = MF->allocateMemRefsArray(numMemRefs);
  MachineSDNode::mmo_iterator MemEnd =
    std::copy(Op0->memoperands_begin(), Op0->memoperands_end(), MemBegin);
  MemEnd =
    std::copy(Op1->memoperands_begin(), Op1->memoperands_end(), MemEnd);
  MI->setMemRefs(MemBegin, MemEnd);
}

bool
ARMPreAllocLoadStoreOpt::CanFormLdStDWord(MachineInstr *Op0, MachineInstr *Op1,
                                          DebugLoc &dl,
                                          unsigned &NewOpc, unsigned &EvenReg,
                                          unsigned &OddReg, unsigned &BaseReg,
                                          int &Offset, unsigned &PredReg,
                                          ARMCC::CondCodes &Pred,
                                          bool &isT2) {
  // Make sure we're allowed to generate LDRD/STRD.
  if (!STI->hasV5TEOps())
    return false;

  // FIXME: VLDRS / VSTRS -> VLDRD / VSTRD
  unsigned Scale = 1;
  unsigned Opcode = Op0->getOpcode();
  if (Opcode == ARM::LDRi12)
    NewOpc = ARM::LDRD;
  else if (Opcode == ARM::STRi12)
    NewOpc = ARM::STRD;
  else if (Opcode == ARM::t2LDRi8 || Opcode == ARM::t2LDRi12) {
    NewOpc = ARM::t2LDRDi8;
    Scale = 4;
    isT2 = true;
  } else if (Opcode == ARM::t2STRi8 || Opcode == ARM::t2STRi12) {
    NewOpc = ARM::t2STRDi8;
    Scale = 4;
    isT2 = true;
  } else
    return false;

  // Make sure the base address satisfies i64 ld / st alignment requirement.
  if (!Op0->hasOneMemOperand() ||
      !(*Op0->memoperands_begin())->getValue() ||
      (*Op0->memoperands_begin())->isVolatile())
    return false;

  unsigned Align = (*Op0->memoperands_begin())->getAlignment();
  const Function *Func = MF->getFunction();
  unsigned ReqAlign = STI->hasV6Ops()
    ? TD->getABITypeAlignment(Type::getInt64Ty(Func->getContext()))
    : 8;  // Pre-v6 need 8-byte align
  if (Align < ReqAlign)
    return false;

  // Then make sure the immediate offset fits.
  int OffImm = getMemoryOpOffset(Op0);
  if (isT2) {
    int Limit = (1 << 8) * Scale;
    if (OffImm >= Limit || (OffImm <= -Limit) || (OffImm & (Scale-1)))
      return false;
    Offset = OffImm;
  } else {
    ARM_AM::AddrOpc AddSub = ARM_AM::add;
    if (OffImm < 0) {
      AddSub = ARM_AM::sub;
      OffImm = - OffImm;
    }
    int Limit = (1 << 8) * Scale;
    if (OffImm >= Limit || (OffImm & (Scale-1)))
      return false;
    Offset = ARM_AM::getAM3Opc(AddSub, OffImm);
  }
  EvenReg = Op0->getOperand(0).getReg();
  OddReg  = Op1->getOperand(0).getReg();
  if (EvenReg == OddReg)
    return false;
  BaseReg = Op0->getOperand(1).getReg();
  Pred = getInstrPredicate(Op0, PredReg);
  dl = Op0->getDebugLoc();
  return true;
}

namespace {
  struct OffsetCompare {
    bool operator()(const MachineInstr *LHS, const MachineInstr *RHS) const {
      int LOffset = getMemoryOpOffset(LHS);
      int ROffset = getMemoryOpOffset(RHS);
      assert(LHS == RHS || LOffset != ROffset);
      return LOffset > ROffset;
    }
  };
}

bool ARMPreAllocLoadStoreOpt::RescheduleOps(MachineBasicBlock *MBB,
                                 SmallVector<MachineInstr*, 4> &Ops,
                                 unsigned Base, bool isLd,
                                 DenseMap<MachineInstr*, unsigned> &MI2LocMap) {
  bool RetVal = false;

  // Sort by offset (in reverse order).
  std::sort(Ops.begin(), Ops.end(), OffsetCompare());

  // The loads / stores of the same base are in order. Scan them from first to
  // last and check for the following:
  // 1. Any def of base.
  // 2. Any gaps.
  while (Ops.size() > 1) {
    unsigned FirstLoc = ~0U;
    unsigned LastLoc = 0;
    MachineInstr *FirstOp = 0;
    MachineInstr *LastOp = 0;
    int LastOffset = 0;
    unsigned LastOpcode = 0;
    unsigned LastBytes = 0;
    unsigned NumMove = 0;
    for (int i = Ops.size() - 1; i >= 0; --i) {
      MachineInstr *Op = Ops[i];
      unsigned Loc = MI2LocMap[Op];
      if (Loc <= FirstLoc) {
        FirstLoc = Loc;
        FirstOp = Op;
      }
      if (Loc >= LastLoc) {
        LastLoc = Loc;
        LastOp = Op;
      }

      unsigned LSMOpcode
        = getLoadStoreMultipleOpcode(Op->getOpcode(), ARM_AM::ia);
      if (LastOpcode && LSMOpcode != LastOpcode)
        break;

      int Offset = getMemoryOpOffset(Op);
      unsigned Bytes = getLSMultipleTransferSize(Op);
      if (LastBytes) {
        if (Bytes != LastBytes || Offset != (LastOffset + (int)Bytes))
          break;
      }
      LastOffset = Offset;
      LastBytes = Bytes;
      LastOpcode = LSMOpcode;
      if (++NumMove == 8) // FIXME: Tune this limit.
        break;
    }

    if (NumMove <= 1)
      Ops.pop_back();
    else {
      SmallPtrSet<MachineInstr*, 4> MemOps;
      SmallSet<unsigned, 4> MemRegs;
      for (int i = NumMove-1; i >= 0; --i) {
        MemOps.insert(Ops[i]);
        MemRegs.insert(Ops[i]->getOperand(0).getReg());
      }

      // Be conservative, if the instructions are too far apart, don't
      // move them. We want to limit the increase of register pressure.
      bool DoMove = (LastLoc - FirstLoc) <= NumMove*4; // FIXME: Tune this.
      if (DoMove)
        DoMove = IsSafeAndProfitableToMove(isLd, Base, FirstOp, LastOp,
                                           MemOps, MemRegs, TRI);
      if (!DoMove) {
        for (unsigned i = 0; i != NumMove; ++i)
          Ops.pop_back();
      } else {
        // This is the new location for the loads / stores.
        MachineBasicBlock::iterator InsertPos = isLd ? FirstOp : LastOp;
        while (InsertPos != MBB->end()
               && (MemOps.count(InsertPos) || InsertPos->isDebugValue()))
          ++InsertPos;

        // If we are moving a pair of loads / stores, see if it makes sense
        // to try to allocate a pair of registers that can form register pairs.
        MachineInstr *Op0 = Ops.back();
        MachineInstr *Op1 = Ops[Ops.size()-2];
        unsigned EvenReg = 0, OddReg = 0;
        unsigned BaseReg = 0, PredReg = 0;
        ARMCC::CondCodes Pred = ARMCC::AL;
        bool isT2 = false;
        unsigned NewOpc = 0;
        int Offset = 0;
        DebugLoc dl;
        if (NumMove == 2 && CanFormLdStDWord(Op0, Op1, dl, NewOpc,
                                             EvenReg, OddReg, BaseReg,
                                             Offset, PredReg, Pred, isT2)) {
          Ops.pop_back();
          Ops.pop_back();

          const MCInstrDesc &MCID = TII->get(NewOpc);
          const TargetRegisterClass *TRC = TII->getRegClass(MCID, 0, TRI, *MF);
          MRI->constrainRegClass(EvenReg, TRC);
          MRI->constrainRegClass(OddReg, TRC);

          // Form the pair instruction.
          if (isLd) {
            MachineInstrBuilder MIB = BuildMI(*MBB, InsertPos, dl, MCID)
              .addReg(EvenReg, RegState::Define)
              .addReg(OddReg, RegState::Define)
              .addReg(BaseReg);
            // FIXME: We're converting from LDRi12 to an insn that still
            // uses addrmode2, so we need an explicit offset reg. It should
            // always by reg0 since we're transforming LDRi12s.
            if (!isT2)
              MIB.addReg(0);
            MIB.addImm(Offset).addImm(Pred).addReg(PredReg);
            concatenateMemOperands(MIB, Op0, Op1);
            DEBUG(dbgs() << "Formed " << *MIB << "\n");
            ++NumLDRDFormed;
          } else {
            MachineInstrBuilder MIB = BuildMI(*MBB, InsertPos, dl, MCID)
              .addReg(EvenReg)
              .addReg(OddReg)
              .addReg(BaseReg);
            // FIXME: We're converting from LDRi12 to an insn that still
            // uses addrmode2, so we need an explicit offset reg. It should
            // always by reg0 since we're transforming STRi12s.
            if (!isT2)
              MIB.addReg(0);
            MIB.addImm(Offset).addImm(Pred).addReg(PredReg);
            concatenateMemOperands(MIB, Op0, Op1);
            DEBUG(dbgs() << "Formed " << *MIB << "\n");
            ++NumSTRDFormed;
          }
          MBB->erase(Op0);
          MBB->erase(Op1);

          // Add register allocation hints to form register pairs.
          MRI->setRegAllocationHint(EvenReg, ARMRI::RegPairEven, OddReg);
          MRI->setRegAllocationHint(OddReg,  ARMRI::RegPairOdd, EvenReg);
        } else {
          for (unsigned i = 0; i != NumMove; ++i) {
            MachineInstr *Op = Ops.back();
            Ops.pop_back();
            MBB->splice(InsertPos, MBB, Op);
          }
        }

        NumLdStMoved += NumMove;
        RetVal = true;
      }
    }
  }

  return RetVal;
}

bool
ARMPreAllocLoadStoreOpt::RescheduleLoadStoreInstrs(MachineBasicBlock *MBB) {
  bool RetVal = false;

  DenseMap<MachineInstr*, unsigned> MI2LocMap;
  DenseMap<unsigned, SmallVector<MachineInstr*, 4> > Base2LdsMap;
  DenseMap<unsigned, SmallVector<MachineInstr*, 4> > Base2StsMap;
  SmallVector<unsigned, 4> LdBases;
  SmallVector<unsigned, 4> StBases;

  unsigned Loc = 0;
  MachineBasicBlock::iterator MBBI = MBB->begin();
  MachineBasicBlock::iterator E = MBB->end();
  while (MBBI != E) {
    for (; MBBI != E; ++MBBI) {
      MachineInstr *MI = MBBI;
      if (MI->isCall() || MI->isTerminator()) {
        // Stop at barriers.
        ++MBBI;
        break;
      }

      if (!MI->isDebugValue())
        MI2LocMap[MI] = ++Loc;

      if (!isMemoryOp(MI))
        continue;
      unsigned PredReg = 0;
      if (getInstrPredicate(MI, PredReg) != ARMCC::AL)
        continue;

      int Opc = MI->getOpcode();
      bool isLd = isi32Load(Opc) || Opc == ARM::VLDRS || Opc == ARM::VLDRD;
      unsigned Base = MI->getOperand(1).getReg();
      int Offset = getMemoryOpOffset(MI);

      bool StopHere = false;
      if (isLd) {
        DenseMap<unsigned, SmallVector<MachineInstr*, 4> >::iterator BI =
          Base2LdsMap.find(Base);
        if (BI != Base2LdsMap.end()) {
          for (unsigned i = 0, e = BI->second.size(); i != e; ++i) {
            if (Offset == getMemoryOpOffset(BI->second[i])) {
              StopHere = true;
              break;
            }
          }
          if (!StopHere)
            BI->second.push_back(MI);
        } else {
          SmallVector<MachineInstr*, 4> MIs;
          MIs.push_back(MI);
          Base2LdsMap[Base] = MIs;
          LdBases.push_back(Base);
        }
      } else {
        DenseMap<unsigned, SmallVector<MachineInstr*, 4> >::iterator BI =
          Base2StsMap.find(Base);
        if (BI != Base2StsMap.end()) {
          for (unsigned i = 0, e = BI->second.size(); i != e; ++i) {
            if (Offset == getMemoryOpOffset(BI->second[i])) {
              StopHere = true;
              break;
            }
          }
          if (!StopHere)
            BI->second.push_back(MI);
        } else {
          SmallVector<MachineInstr*, 4> MIs;
          MIs.push_back(MI);
          Base2StsMap[Base] = MIs;
          StBases.push_back(Base);
        }
      }

      if (StopHere) {
        // Found a duplicate (a base+offset combination that's seen earlier).
        // Backtrack.
        --Loc;
        break;
      }
    }

    // Re-schedule loads.
    for (unsigned i = 0, e = LdBases.size(); i != e; ++i) {
      unsigned Base = LdBases[i];
      SmallVector<MachineInstr*, 4> &Lds = Base2LdsMap[Base];
      if (Lds.size() > 1)
        RetVal |= RescheduleOps(MBB, Lds, Base, true, MI2LocMap);
    }

    // Re-schedule stores.
    for (unsigned i = 0, e = StBases.size(); i != e; ++i) {
      unsigned Base = StBases[i];
      SmallVector<MachineInstr*, 4> &Sts = Base2StsMap[Base];
      if (Sts.size() > 1)
        RetVal |= RescheduleOps(MBB, Sts, Base, false, MI2LocMap);
    }

    if (MBBI != E) {
      Base2LdsMap.clear();
      Base2StsMap.clear();
      LdBases.clear();
      StBases.clear();
    }
  }

  return RetVal;
}


/// createARMLoadStoreOptimizationPass - returns an instance of the load / store
/// optimization pass.
FunctionPass *llvm::createARMLoadStoreOptimizationPass(bool PreAlloc) {
  if (PreAlloc)
    return new ARMPreAllocLoadStoreOpt();
  return new ARMLoadStoreOpt();
}
