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
#include "AMDGPUSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "si-insert-waits"

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

typedef enum {
  OTHER,
  SMEM,
  VMEM
} InstType;

typedef Counters RegCounters[512];
typedef std::pair<unsigned, unsigned> RegInterval;

class SIInsertWaits : public MachineFunctionPass {

private:
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

  /// \brief Type of the last opcode.
  InstType LastOpcodeType;

  bool LastInstWritesM0;

  /// \brief Whether the machine function returns void
  bool ReturnsVoid;

  /// \brief Get increment/decrement amount for this instruction.
  Counters getHwCounts(MachineInstr &MI);

  /// \brief Is operand relevant for async execution?
  bool isOpRelevant(MachineOperand &Op);

  /// \brief Get register interval an operand affects.
  RegInterval getRegInterval(const TargetRegisterClass *RC,
                             const MachineOperand &Reg) const;

  /// \brief Handle instructions async components
  void pushInstruction(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator I);

  /// \brief Insert the actual wait instruction
  bool insertWait(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  const Counters &Counts);

  /// \brief Do we need def2def checks?
  bool unorderedDefines(MachineInstr &MI);

  /// \brief Resolve all operand dependencies to counter requirements
  Counters handleOperands(MachineInstr &MI);

  /// \brief Insert S_NOP between an instruction writing M0 and S_SENDMSG.
  void handleSendMsg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I);

public:
  static char ID;

  SIInsertWaits() :
    MachineFunctionPass(ID),
    TII(nullptr),
    TRI(nullptr),
    ExpInstrTypesSeen(0) { }

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI insert wait instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace

INITIALIZE_PASS_BEGIN(SIInsertWaits, DEBUG_TYPE,
                      "SI Insert Waits", false, false)
INITIALIZE_PASS_END(SIInsertWaits, DEBUG_TYPE,
                    "SI Insert Waits", false, false)

char SIInsertWaits::ID = 0;

char &llvm::SIInsertWaitsID = SIInsertWaits::ID;

FunctionPass *llvm::createSIInsertWaitsPass() {
  return new SIInsertWaits();
}

const Counters SIInsertWaits::WaitCounts = { { 15, 7, 15 } };
const Counters SIInsertWaits::ZeroCounts = { { 0, 0, 0 } };


Counters SIInsertWaits::getHwCounts(MachineInstr &MI) {
  uint64_t TSFlags = MI.getDesc().TSFlags;
  Counters Result = { { 0, 0, 0 } };

  Result.Named.VM = !!(TSFlags & SIInstrFlags::VM_CNT);

  // Only consider stores or EXP for EXP_CNT
  Result.Named.EXP = !!(TSFlags & SIInstrFlags::EXP_CNT &&
      (MI.getOpcode() == AMDGPU::EXP || MI.getDesc().mayStore()));

  // LGKM may uses larger values
  if (TSFlags & SIInstrFlags::LGKM_CNT) {

    if (TII->isSMRD(MI)) {

      if (MI.getNumOperands() != 0) {
        assert(MI.getOperand(0).isReg() &&
               "First LGKM operand must be a register!");

        // XXX - What if this is a write into a super register?
        const TargetRegisterClass *RC = TII->getOpRegClass(MI, 0);
        unsigned Size = RC->getSize();
        Result.Named.LGKM = Size > 4 ? 2 : 1;
      } else {
        // s_dcache_inv etc. do not have a a destination register. Assume we
        // want a wait on these.
        // XXX - What is the right value?
        Result.Named.LGKM = 1;
      }
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
  if (!Op.isReg() || !TRI->isInAllocatableClass(Op.getReg()))
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

  // Check if this operand is the value being stored.
  // Special case for DS instructions, since the address
  // operand comes before the value operand and it may have
  // multiple data operands.

  if (TII->isDS(MI)) {
    MachineOperand *Data = TII->getNamedOperand(MI, AMDGPU::OpName::data);
    if (Data && Op.isIdenticalTo(*Data))
      return true;

    MachineOperand *Data0 = TII->getNamedOperand(MI, AMDGPU::OpName::data0);
    if (Data0 && Op.isIdenticalTo(*Data0))
      return true;

    MachineOperand *Data1 = TII->getNamedOperand(MI, AMDGPU::OpName::data1);
    if (Data1 && Op.isIdenticalTo(*Data1))
      return true;

    return false;
  }

  // NOTE: This assumes that the value operand is before the
  // address operand, and that there is only one value operand.
  for (MachineInstr::mop_iterator I = MI.operands_begin(),
       E = MI.operands_end(); I != E; ++I) {

    if (I->isReg() && I->isUse())
      return Op.isIdenticalTo(*I);
  }

  return false;
}

RegInterval SIInsertWaits::getRegInterval(const TargetRegisterClass *RC,
                                          const MachineOperand &Reg) const {
  unsigned Size = RC->getSize();
  assert(Size >= 4);

  RegInterval Result;
  Result.first = TRI->getEncodingValue(Reg.getReg());
  Result.second = Result.first + Size / 4;

  return Result;
}

void SIInsertWaits::pushInstruction(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I) {

  // Get the hardware counter increments and sum them up
  Counters Increment = getHwCounts(*I);
  Counters Limit = ZeroCounts;
  unsigned Sum = 0;

  for (unsigned i = 0; i < 3; ++i) {
    LastIssued.Array[i] += Increment.Array[i];
    if (Increment.Array[i])
      Limit.Array[i] = LastIssued.Array[i];
    Sum += Increment.Array[i];
  }

  // If we don't increase anything then that's it
  if (Sum == 0) {
    LastOpcodeType = OTHER;
    return;
  }

  if (MBB.getParent()->getSubtarget<AMDGPUSubtarget>().getGeneration() >=
      AMDGPUSubtarget::VOLCANIC_ISLANDS) {
    // Any occurrence of consecutive VMEM or SMEM instructions forms a VMEM
    // or SMEM clause, respectively.
    //
    // The temporary workaround is to break the clauses with S_NOP.
    //
    // The proper solution would be to allocate registers such that all source
    // and destination registers don't overlap, e.g. this is illegal:
    //   r0 = load r2
    //   r2 = load r0
    if ((LastOpcodeType == SMEM && TII->isSMRD(*I)) ||
        (LastOpcodeType == VMEM && Increment.Named.VM)) {
      // Insert a NOP to break the clause.
      BuildMI(MBB, I, DebugLoc(), TII->get(AMDGPU::S_NOP))
          .addImm(0);
      LastInstWritesM0 = false;
    }

    if (TII->isSMRD(*I))
      LastOpcodeType = SMEM;
    else if (Increment.Named.VM)
      LastOpcodeType = VMEM;
  }

  // Remember which export instructions we have seen
  if (Increment.Named.EXP) {
    ExpInstrTypesSeen |= I->getOpcode() == AMDGPU::EXP ? 1 : 2;
  }

  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    MachineOperand &Op = I->getOperand(i);
    if (!isOpRelevant(Op))
      continue;

    const TargetRegisterClass *RC = TII->getOpRegClass(*I, i);
    RegInterval Interval = getRegInterval(RC, Op);
    for (unsigned j = Interval.first; j < Interval.second; ++j) {

      // Remember which registers we define
      if (Op.isDef())
        DefinedRegs[j] = Limit;

      // and which one we are using
      if (Op.isUse())
        UsedRegs[j] = Limit;
    }
  }
}

bool SIInsertWaits::insertWait(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator I,
                               const Counters &Required) {

  // End of program? No need to wait on anything
  // A function not returning void needs to wait, because other bytecode will
  // be appended after it and we don't know what it will be.
  if (I != MBB.end() && I->getOpcode() == AMDGPU::S_ENDPGM && ReturnsVoid)
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

      // Adjust the value to the real hardware possibilities.
      Counts.Array[i] = std::min(Value, WaitCounts.Array[i]);

    } else
      Counts.Array[i] = 0;

    // Remember on what we have waited on.
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
                  ((Counts.Named.LGKM & 0xF) << 8));

  LastOpcodeType = OTHER;
  LastInstWritesM0 = false;
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

  // For each register affected by this instruction increase the result
  // sequence.
  //
  // TODO: We could probably just look at explicit operands if we removed VCC /
  // EXEC from SMRD dest reg classes.
  for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
    MachineOperand &Op = MI.getOperand(i);
    if (!Op.isReg() || !TRI->isInAllocatableClass(Op.getReg()))
      continue;

    const TargetRegisterClass *RC = TII->getOpRegClass(MI, i);
    RegInterval Interval = getRegInterval(RC, Op);
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

void SIInsertWaits::handleSendMsg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I) {
  if (MBB.getParent()->getSubtarget<AMDGPUSubtarget>().getGeneration() <
      AMDGPUSubtarget::VOLCANIC_ISLANDS)
    return;

  // There must be "S_NOP 0" between an instruction writing M0 and S_SENDMSG.
  if (LastInstWritesM0 && I->getOpcode() == AMDGPU::S_SENDMSG) {
    BuildMI(MBB, I, DebugLoc(), TII->get(AMDGPU::S_NOP)).addImm(0);
    LastInstWritesM0 = false;
    return;
  }

  // Set whether this instruction sets M0
  LastInstWritesM0 = false;

  unsigned NumOperands = I->getNumOperands();
  for (unsigned i = 0; i < NumOperands; i++) {
    const MachineOperand &Op = I->getOperand(i);

    if (Op.isReg() && Op.isDef() && Op.getReg() == AMDGPU::M0)
      LastInstWritesM0 = true;
  }
}

// FIXME: Insert waits listed in Table 4.2 "Required User-Inserted Wait States"
// around other non-memory instructions.
bool SIInsertWaits::runOnMachineFunction(MachineFunction &MF) {
  bool Changes = false;

  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  const AMDGPUSubtarget &ST = MF.getSubtarget<AMDGPUSubtarget>();
  MRI = &MF.getRegInfo();

  WaitedOn = ZeroCounts;
  LastIssued = ZeroCounts;
  LastOpcodeType = OTHER;
  LastInstWritesM0 = false;
  ReturnsVoid = MF.getInfo<SIMachineFunctionInfo>()->returnsVoid();

  memset(&UsedRegs, 0, sizeof(UsedRegs));
  memset(&DefinedRegs, 0, sizeof(DefinedRegs));

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {

      // Insert required wait states for SMRD reading an SGPR written by a VALU
      // instruction.
      if (ST.getGeneration() <= AMDGPUSubtarget::SOUTHERN_ISLANDS &&
          I->getOpcode() == AMDGPU::V_READFIRSTLANE_B32)
        TII->insertWaitStates(std::next(I), 4);

      // Wait for everything before a barrier.
      if (I->getOpcode() == AMDGPU::S_BARRIER)
        Changes |= insertWait(MBB, I, LastIssued);
      else
        Changes |= insertWait(MBB, I, handleOperands(*I));

      pushInstruction(MBB, I);
      handleSendMsg(MBB, I);
    }

    // Wait for everything at the end of the MBB
    Changes |= insertWait(MBB, MBB.getFirstTerminator(), LastIssued);
  }

  return Changes;
}
