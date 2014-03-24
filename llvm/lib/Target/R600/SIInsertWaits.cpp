//===-- SILowerControlFlow.cpp - Use predicates for control flow ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Insert wait instructions for memory reads and writes.
///
/// Memory reads and writes are issued asynchronously, so we need to insert
/// S_WAITCNT instructions when we want to access any of their results or
/// overwrite any register that's used asynchronously.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace {

/// \brief One variable for each of the hardware counters
typedef union {
  struct {
    unsigned VM;
    unsigned EXP;
    unsigned LGKM;
  } Named;
  unsigned Array[3];

} Counters;

typedef Counters RegCounters[512];
typedef std::pair<unsigned, unsigned> RegInterval;

class SIInsertWaits : public MachineFunctionPass {

private:
  static char ID;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;

  /// \brief Constant hardware limits
  static const Counters WaitCounts;

  /// \brief Constant zero value
  static const Counters ZeroCounts;

  /// \brief Counter values we have already waited on.
  Counters WaitedOn;

  /// \brief Counter values for last instruction issued.
  Counters LastIssued;

  /// \brief Registers used by async instructions.
  RegCounters UsedRegs;

  /// \brief Registers defined by async instructions.
  RegCounters DefinedRegs;

  /// \brief Different export instruction types seen since last wait.
  unsigned ExpInstrTypesSeen;

  /// \brief Get increment/decrement amount for this instruction.
  Counters getHwCounts(MachineInstr &MI);

  /// \brief Is operand relevant for async execution?
  bool isOpRelevant(MachineOperand &Op);

  /// \brief Get register interval an operand affects.
  RegInterval getRegInterval(MachineOperand &Op);

  /// \brief Handle instructions async components
  void pushInstruction(MachineInstr &MI);

  /// \brief Insert the actual wait instruction
  bool insertWait(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  const Counters &Counts);

  /// \brief Do we need def2def checks?
  bool unorderedDefines(MachineInstr &MI);

  /// \brief Resolve all operand dependencies to counter requirements
  Counters handleOperands(MachineInstr &MI);

public:
  SIInsertWaits(TargetMachine &tm) :
    MachineFunctionPass(ID),
    TII(0),
    TRI(0),
    ExpInstrTypesSeen(0) { }

  virtual bool runOnMachineFunction(MachineFunction &MF);

  const char *getPassName() const {
    return "SI insert wait  instructions";
  }

};

} // End anonymous namespace

char SIInsertWaits::ID = 0;

const Counters SIInsertWaits::WaitCounts = { { 15, 7, 7 } };
const Counters SIInsertWaits::ZeroCounts = { { 0, 0, 0 } };

FunctionPass *llvm::createSIInsertWaits(TargetMachine &tm) {
  return new SIInsertWaits(tm);
}

Counters SIInsertWaits::getHwCounts(MachineInstr &MI) {

  uint64_t TSFlags = TII->get(MI.getOpcode()).TSFlags;
  Counters Result;

  Result.Named.VM = !!(TSFlags & SIInstrFlags::VM_CNT);

  // Only consider stores or EXP for EXP_CNT
  Result.Named.EXP = !!(TSFlags & SIInstrFlags::EXP_CNT &&
      (MI.getOpcode() == AMDGPU::EXP || MI.getDesc().mayStore()));

  // LGKM may uses larger values
  if (TSFlags & SIInstrFlags::LGKM_CNT) {

    if (TII->isSMRD(MI.getOpcode())) {

      MachineOperand &Op = MI.getOperand(0);
      assert(Op.isReg() && "First LGKM operand must be a register!");

      unsigned Reg = Op.getReg();
      unsigned Size = TRI->getMinimalPhysRegClass(Reg)->getSize();
      Result.Named.LGKM = Size > 4 ? 2 : 1;

    } else {
      // DS
      Result.Named.LGKM = 1;
    }

  } else {
    Result.Named.LGKM = 0;
  }

  return Result;
}

bool SIInsertWaits::isOpRelevant(MachineOperand &Op) {

  // Constants are always irrelevant
  if (!Op.isReg())
    return false;

  // Defines are always relevant
  if (Op.isDef())
    return true;

  // For exports all registers are relevant
  MachineInstr &MI = *Op.getParent();
  if (MI.getOpcode() == AMDGPU::EXP)
    return true;

  // For stores the stored value is also relevant
  if (!MI.getDesc().mayStore())
    return false;

  for (MachineInstr::mop_iterator I = MI.operands_begin(),
       E = MI.operands_end(); I != E; ++I) {

    if (I->isReg() && I->isUse())
      return Op.isIdenticalTo(*I);
  }

  return false;
}

RegInterval SIInsertWaits::getRegInterval(MachineOperand &Op) {

  if (!Op.isReg() || !TRI->isInAllocatableClass(Op.getReg()))
    return std::make_pair(0, 0);

  unsigned Reg = Op.getReg();
  unsigned Size = TRI->getMinimalPhysRegClass(Reg)->getSize();

  assert(Size >= 4);

  RegInterval Result;
  Result.first = TRI->getEncodingValue(Reg);
  Result.second = Result.first + Size / 4;

  return Result;
}

void SIInsertWaits::pushInstruction(MachineInstr &MI) {

  // Get the hardware counter increments and sum them up
  Counters Increment = getHwCounts(MI);
  unsigned Sum = 0;

  for (unsigned i = 0; i < 3; ++i) {
    LastIssued.Array[i] += Increment.Array[i];
    Sum += Increment.Array[i];
  }

  // If we don't increase anything then that's it
  if (Sum == 0)
    return;

  // Remember which export instructions we have seen
  if (Increment.Named.EXP) {
    ExpInstrTypesSeen |= MI.getOpcode() == AMDGPU::EXP ? 1 : 2;
  }

  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {

    MachineOperand &Op = MI.getOperand(i);
    if (!isOpRelevant(Op))
      continue;

    RegInterval Interval = getRegInterval(Op);
    for (unsigned j = Interval.first; j < Interval.second; ++j) {

      // Remember which registers we define
      if (Op.isDef())
        DefinedRegs[j] = LastIssued;

      // and which one we are using
      if (Op.isUse())
        UsedRegs[j] = LastIssued;
    }
  }
}

bool SIInsertWaits::insertWait(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               const Counters &Required) {

  // End of program? No need to wait on anything
  if (I != MBB.end() && I->getOpcode() == AMDGPU::S_ENDPGM)
    return false;

  // Figure out if the async instructions execute in order
  bool Ordered[3];

  // VM_CNT is always ordered
  Ordered[0] = true;

  // EXP_CNT is unordered if we have both EXP & VM-writes
  Ordered[1] = ExpInstrTypesSeen == 3;

  // LGKM_CNT is handled as always unordered. TODO: Handle LDS and GDS
  Ordered[2] = false;

  // The values we are going to put into the S_WAITCNT instruction
  Counters Counts = WaitCounts;

  // Do we really need to wait?
  bool NeedWait = false;

  for (unsigned i = 0; i < 3; ++i) {

    if (Required.Array[i] <= WaitedOn.Array[i])
      continue;

    NeedWait = true;
    
    if (Ordered[i]) {
      unsigned Value = LastIssued.Array[i] - Required.Array[i];

      // adjust the value to the real hardware posibilities
      Counts.Array[i] = std::min(Value, WaitCounts.Array[i]);

    } else
      Counts.Array[i] = 0;

    // Remember on what we have waited on
    WaitedOn.Array[i] = LastIssued.Array[i] - Counts.Array[i];
  }

  if (!NeedWait)
    return false;

  // Reset EXP_CNT instruction types
  if (Counts.Named.EXP == 0)
    ExpInstrTypesSeen = 0;

  // Build the wait instruction
  BuildMI(MBB, I, DebugLoc(), TII->get(AMDGPU::S_WAITCNT))
          .addImm((Counts.Named.VM & 0xF) |
                  ((Counts.Named.EXP & 0x7) << 4) |
                  ((Counts.Named.LGKM & 0x7) << 8));

  return true;
}

/// \brief helper function for handleOperands
static void increaseCounters(Counters &Dst, const Counters &Src) {

  for (unsigned i = 0; i < 3; ++i)
    Dst.Array[i] = std::max(Dst.Array[i], Src.Array[i]);
}

Counters SIInsertWaits::handleOperands(MachineInstr &MI) {

  Counters Result = ZeroCounts;

  // S_SENDMSG implicitly waits for all outstanding LGKM transfers to finish,
  // but we also want to wait for any other outstanding transfers before
  // signalling other hardware blocks
  if (MI.getOpcode() == AMDGPU::S_SENDMSG)
    return LastIssued;

  // For each register affected by this
  // instruction increase the result sequence
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {

    MachineOperand &Op = MI.getOperand(i);
    RegInterval Interval = getRegInterval(Op);
    for (unsigned j = Interval.first; j < Interval.second; ++j) {

      if (Op.isDef()) {
        increaseCounters(Result, UsedRegs[j]);
        increaseCounters(Result, DefinedRegs[j]);
      }

      if (Op.isUse())
        increaseCounters(Result, DefinedRegs[j]);
    }
  }

  return Result;
}

bool SIInsertWaits::runOnMachineFunction(MachineFunction &MF) {
  bool Changes = false;

  TII = static_cast<const SIInstrInfo*>(MF.getTarget().getInstrInfo());
  TRI = static_cast<const SIRegisterInfo*>(MF.getTarget().getRegisterInfo());

  MRI = &MF.getRegInfo();

  WaitedOn = ZeroCounts;
  LastIssued = ZeroCounts;

  memset(&UsedRegs, 0, sizeof(UsedRegs));
  memset(&DefinedRegs, 0, sizeof(DefinedRegs));

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {

      Changes |= insertWait(MBB, I, handleOperands(*I));
      pushInstruction(*I);
    }

    // Wait for everything at the end of the MBB
    Changes |= insertWait(MBB, MBB.getFirstTerminator(), LastIssued);
  }

  return Changes;
}
