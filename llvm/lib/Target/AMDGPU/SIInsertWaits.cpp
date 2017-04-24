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
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

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
  const SISubtarget *ST = nullptr;
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI;
  AMDGPU::IsaInfo::IsaVersion ISA;

  /// \brief Constant zero value
  static const Counters ZeroCounts;

  /// \brief Hardware limits
  Counters HardwareLimits;

  /// \brief Counter values we have already waited on.
  Counters WaitedOn;

  /// \brief Counter values that we must wait on before the next counter
  /// increase.
  Counters DelayedWaitOn;

  /// \brief Counter values for last instruction issued.
  Counters LastIssued;

  /// \brief Registers used by async instructions.
  RegCounters UsedRegs;

  /// \brief Registers defined by async instructions.
  RegCounters DefinedRegs;

  /// \brief Different export instruction types seen since last wait.
  unsigned ExpInstrTypesSeen = 0;

  /// \brief Type of the last opcode.
  InstType LastOpcodeType;

  bool LastInstWritesM0;

  /// Whether or not we have flat operations outstanding.
  bool IsFlatOutstanding;

  /// \brief Whether the machine function returns void
  bool ReturnsVoid;

  /// Whether the VCCZ bit is possibly corrupt
  bool VCCZCorrupt = false;

  /// \brief Get increment/decrement amount for this instruction.
  Counters getHwCounts(MachineInstr &MI);

  /// \brief Is operand relevant for async execution?
  bool isOpRelevant(MachineOperand &Op);

  /// \brief Get register interval an operand affects.
  RegInterval getRegInterval(const TargetRegisterClass *RC,
                             const MachineOperand &Reg) const;

  /// \brief Handle instructions async components
  void pushInstruction(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator I,
                       const Counters& Increment);

  /// \brief Insert the actual wait instruction
  bool insertWait(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator I,
                  const Counters &Counts);

  /// \brief Handle existing wait instructions (from intrinsics)
  void handleExistingWait(MachineBasicBlock::iterator I);

  /// \brief Do we need def2def checks?
  bool unorderedDefines(MachineInstr &MI);

  /// \brief Resolve all operand dependencies to counter requirements
  Counters handleOperands(MachineInstr &MI);

  /// \brief Insert S_NOP between an instruction writing M0 and S_SENDMSG.
  void handleSendMsg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I);

  /// Return true if there are LGKM instrucitons that haven't been waited on
  /// yet.
  bool hasOutstandingLGKM() const;

public:
  static char ID;

  SIInsertWaits() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI insert wait instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SIInsertWaits, DEBUG_TYPE,
                      "SI Insert Waits", false, false)
INITIALIZE_PASS_END(SIInsertWaits, DEBUG_TYPE,
                    "SI Insert Waits", false, false)

char SIInsertWaits::ID = 0;

char &llvm::SIInsertWaitsID = SIInsertWaits::ID;

FunctionPass *llvm::createSIInsertWaitsPass() {
  return new SIInsertWaits();
}

const Counters SIInsertWaits::ZeroCounts = { { 0, 0, 0 } };

static bool readsVCCZ(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return (Opc == AMDGPU::S_CBRANCH_VCCNZ || Opc == AMDGPU::S_CBRANCH_VCCZ) &&
         !MI.getOperand(1).isUndef();
}

bool SIInsertWaits::hasOutstandingLGKM() const {
  return WaitedOn.Named.LGKM != LastIssued.Named.LGKM;
}

Counters SIInsertWaits::getHwCounts(MachineInstr &MI) {
  uint64_t TSFlags = MI.getDesc().TSFlags;
  Counters Result = { { 0, 0, 0 } };

  Result.Named.VM = !!(TSFlags & SIInstrFlags::VM_CNT);

  // Only consider stores or EXP for EXP_CNT
  Result.Named.EXP = !!(TSFlags & SIInstrFlags::EXP_CNT) && MI.mayStore();

  // LGKM may uses larger values
  if (TSFlags & SIInstrFlags::LGKM_CNT) {

    if (TII->isSMRD(MI)) {

      if (MI.getNumOperands() != 0) {
        assert(MI.getOperand(0).isReg() &&
               "First LGKM operand must be a register!");

        // XXX - What if this is a write into a super register?
        const TargetRegisterClass *RC = TII->getOpRegClass(MI, 0);
        unsigned Size = TRI->getRegSizeInBits(*RC);
        Result.Named.LGKM = Size > 32 ? 2 : 1;
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

  // For exports all registers are relevant.
  // TODO: Skip undef/disabled registers.
  MachineInstr &MI = *Op.getParent();
  if (TII->isEXP(MI))
    return true;

  // For stores the stored value is also relevant
  if (!MI.getDesc().mayStore())
    return false;

  // Check if this operand is the value being stored.
  // Special case for DS/FLAT instructions, since the address
  // operand comes before the value operand and it may have
  // multiple data operands.

  if (TII->isDS(MI)) {
    MachineOperand *Data0 = TII->getNamedOperand(MI, AMDGPU::OpName::data0);
    if (Data0 && Op.isIdenticalTo(*Data0))
      return true;

    MachineOperand *Data1 = TII->getNamedOperand(MI, AMDGPU::OpName::data1);
    return Data1 && Op.isIdenticalTo(*Data1);
  }

  if (TII->isFLAT(MI)) {
    MachineOperand *Data = TII->getNamedOperand(MI, AMDGPU::OpName::vdata);
    if (Data && Op.isIdenticalTo(*Data))
      return true;
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
  unsigned Size = TRI->getRegSizeInBits(*RC);
  assert(Size >= 32);

  RegInterval Result;
  Result.first = TRI->getEncodingValue(Reg.getReg());
  Result.second = Result.first + Size / 32;

  return Result;
}

void SIInsertWaits::pushInstruction(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator I,
                                    const Counters &Increment) {
  // Get the hardware counter increments and sum them up
  Counters Limit = ZeroCounts;
  unsigned Sum = 0;

  if (TII->mayAccessFlatAddressSpace(*I))
    IsFlatOutstanding = true;

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

  if (ST->getGeneration() >= SISubtarget::VOLCANIC_ISLANDS) {
    // Any occurrence of consecutive VMEM or SMEM instructions forms a VMEM
    // or SMEM clause, respectively.
    //
    // The temporary workaround is to break the clauses with S_NOP.
    //
    // The proper solution would be to allocate registers such that all source
    // and destination registers don't overlap, e.g. this is illegal:
    //   r0 = load r2
    //   r2 = load r0
    if (LastOpcodeType == VMEM && Increment.Named.VM) {
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
    ExpInstrTypesSeen |= TII->isEXP(*I) ? 1 : 2;
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

  // VM_CNT is always ordered except when there are flat instructions, which
  // can return out of order.
  Ordered[0] = !IsFlatOutstanding;

  // EXP_CNT is unordered if we have both EXP & VM-writes
  Ordered[1] = ExpInstrTypesSeen == 3;

  // LGKM_CNT is handled as always unordered. TODO: Handle LDS and GDS
  Ordered[2] = false;

  // The values we are going to put into the S_WAITCNT instruction
  Counters Counts = HardwareLimits;

  // Do we really need to wait?
  bool NeedWait = false;

  for (unsigned i = 0; i < 3; ++i) {
    if (Required.Array[i] <= WaitedOn.Array[i])
      continue;

    NeedWait = true;

    if (Ordered[i]) {
      unsigned Value = LastIssued.Array[i] - Required.Array[i];

      // Adjust the value to the real hardware possibilities.
      Counts.Array[i] = std::min(Value, HardwareLimits.Array[i]);

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
    .addImm(AMDGPU::encodeWaitcnt(ISA,
                                  Counts.Named.VM,
                                  Counts.Named.EXP,
                                  Counts.Named.LGKM));

  LastOpcodeType = OTHER;
  LastInstWritesM0 = false;
  IsFlatOutstanding = false;
  return true;
}

/// \brief helper function for handleOperands
static void increaseCounters(Counters &Dst, const Counters &Src) {
  for (unsigned i = 0; i < 3; ++i)
    Dst.Array[i] = std::max(Dst.Array[i], Src.Array[i]);
}

/// \brief check whether any of the counters is non-zero
static bool countersNonZero(const Counters &Counter) {
  for (unsigned i = 0; i < 3; ++i)
    if (Counter.Array[i])
      return true;
  return false;
}

void SIInsertWaits::handleExistingWait(MachineBasicBlock::iterator I) {
  assert(I->getOpcode() == AMDGPU::S_WAITCNT);

  unsigned Imm = I->getOperand(0).getImm();
  Counters Counts, WaitOn;

  Counts.Named.VM = AMDGPU::decodeVmcnt(ISA, Imm);
  Counts.Named.EXP = AMDGPU::decodeExpcnt(ISA, Imm);
  Counts.Named.LGKM = AMDGPU::decodeLgkmcnt(ISA, Imm);

  for (unsigned i = 0; i < 3; ++i) {
    if (Counts.Array[i] <= LastIssued.Array[i])
      WaitOn.Array[i] = LastIssued.Array[i] - Counts.Array[i];
    else
      WaitOn.Array[i] = 0;
  }

  increaseCounters(DelayedWaitOn, WaitOn);
}

Counters SIInsertWaits::handleOperands(MachineInstr &MI) {
  Counters Result = ZeroCounts;

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
  if (ST->getGeneration() < SISubtarget::VOLCANIC_ISLANDS)
    return;

  // There must be "S_NOP 0" between an instruction writing M0 and S_SENDMSG.
  if (LastInstWritesM0 && (I->getOpcode() == AMDGPU::S_SENDMSG || I->getOpcode() == AMDGPU::S_SENDMSGHALT)) {
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

/// Return true if \p MBB has one successor immediately following, and is its
/// only predecessor
static bool hasTrivialSuccessor(const MachineBasicBlock &MBB) {
  if (MBB.succ_size() != 1)
    return false;

  const MachineBasicBlock *Succ = *MBB.succ_begin();
  return (Succ->pred_size() == 1) && MBB.isLayoutSuccessor(Succ);
}

// FIXME: Insert waits listed in Table 4.2 "Required User-Inserted Wait States"
// around other non-memory instructions.
bool SIInsertWaits::runOnMachineFunction(MachineFunction &MF) {
  bool Changes = false;

  ST = &MF.getSubtarget<SISubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  ISA = AMDGPU::IsaInfo::getIsaVersion(ST->getFeatureBits());
  const SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  HardwareLimits.Named.VM = AMDGPU::getVmcntBitMask(ISA);
  HardwareLimits.Named.EXP = AMDGPU::getExpcntBitMask(ISA);
  HardwareLimits.Named.LGKM = AMDGPU::getLgkmcntBitMask(ISA);

  WaitedOn = ZeroCounts;
  DelayedWaitOn = ZeroCounts;
  LastIssued = ZeroCounts;
  LastOpcodeType = OTHER;
  LastInstWritesM0 = false;
  IsFlatOutstanding = false;
  ReturnsVoid = MFI->returnsVoid();

  memset(&UsedRegs, 0, sizeof(UsedRegs));
  memset(&DefinedRegs, 0, sizeof(DefinedRegs));

  SmallVector<MachineInstr *, 4> RemoveMI;
  SmallVector<MachineBasicBlock *, 4> EndPgmBlocks;

  bool HaveScalarStores = false;

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
       BI != BE; ++BI) {

    MachineBasicBlock &MBB = *BI;

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {

      if (!HaveScalarStores && TII->isScalarStore(*I))
        HaveScalarStores = true;

      if (ST->getGeneration() <= SISubtarget::SEA_ISLANDS) {
        // There is a hardware bug on CI/SI where SMRD instruction may corrupt
        // vccz bit, so when we detect that an instruction may read from a
        // corrupt vccz bit, we need to:
        // 1. Insert s_waitcnt lgkm(0) to wait for all outstanding SMRD operations to
        //    complete.
        // 2. Restore the correct value of vccz by writing the current value
        //    of vcc back to vcc.

        if (TII->isSMRD(I->getOpcode())) {
          VCCZCorrupt = true;
        } else if (!hasOutstandingLGKM() && I->modifiesRegister(AMDGPU::VCC, TRI)) {
          // FIXME: We only care about SMRD instructions here, not LDS or GDS.
          // Whenever we store a value in vcc, the correct value of vccz is
          // restored.
          VCCZCorrupt = false;
        }

        // Check if we need to apply the bug work-around
        if (VCCZCorrupt && readsVCCZ(*I)) {
          DEBUG(dbgs() << "Inserting vccz bug work-around before: " << *I << '\n');

          // Wait on everything, not just LGKM.  vccz reads usually come from
          // terminators, and we always wait on everything at the end of the
          // block, so if we only wait on LGKM here, we might end up with
          // another s_waitcnt inserted right after this if there are non-LGKM
          // instructions still outstanding.
          insertWait(MBB, I, LastIssued);

          // Restore the vccz bit.  Any time a value is written to vcc, the vcc
          // bit is updated, so we can restore the bit by reading the value of
          // vcc and then writing it back to the register.
          BuildMI(MBB, I, I->getDebugLoc(), TII->get(AMDGPU::S_MOV_B64),
                  AMDGPU::VCC)
            .addReg(AMDGPU::VCC);
        }
      }

      // Record pre-existing, explicitly requested waits
      if (I->getOpcode() == AMDGPU::S_WAITCNT) {
        handleExistingWait(*I);
        RemoveMI.push_back(&*I);
        continue;
      }

      Counters Required;

      // Wait for everything before a barrier.
      //
      // S_SENDMSG implicitly waits for all outstanding LGKM transfers to finish,
      // but we also want to wait for any other outstanding transfers before
      // signalling other hardware blocks
      if ((I->getOpcode() == AMDGPU::S_BARRIER &&
               ST->needWaitcntBeforeBarrier()) ||
           I->getOpcode() == AMDGPU::S_SENDMSG ||
           I->getOpcode() == AMDGPU::S_SENDMSGHALT)
        Required = LastIssued;
      else
        Required = handleOperands(*I);

      Counters Increment = getHwCounts(*I);

      if (countersNonZero(Required) || countersNonZero(Increment))
        increaseCounters(Required, DelayedWaitOn);

      Changes |= insertWait(MBB, I, Required);

      pushInstruction(MBB, I, Increment);
      handleSendMsg(MBB, I);

      if (I->getOpcode() == AMDGPU::S_ENDPGM ||
          I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG)
        EndPgmBlocks.push_back(&MBB);
    }

    // Wait for everything at the end of the MBB. If there is only one
    // successor, we can defer this until the uses there.
    if (!hasTrivialSuccessor(MBB))
      Changes |= insertWait(MBB, MBB.getFirstTerminator(), LastIssued);
  }

  if (HaveScalarStores) {
    // If scalar writes are used, the cache must be flushed or else the next
    // wave to reuse the same scratch memory can be clobbered.
    //
    // Insert s_dcache_wb at wave termination points if there were any scalar
    // stores, and only if the cache hasn't already been flushed. This could be
    // improved by looking across blocks for flushes in postdominating blocks
    // from the stores but an explicitly requested flush is probably very rare.
    for (MachineBasicBlock *MBB : EndPgmBlocks) {
      bool SeenDCacheWB = false;

      for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
           I != E; ++I) {

        if (I->getOpcode() == AMDGPU::S_DCACHE_WB)
          SeenDCacheWB = true;
        else if (TII->isScalarStore(*I))
          SeenDCacheWB = false;

        // FIXME: It would be better to insert this before a waitcnt if any.
        if ((I->getOpcode() == AMDGPU::S_ENDPGM ||
             I->getOpcode() == AMDGPU::SI_RETURN_TO_EPILOG) && !SeenDCacheWB) {
          Changes = true;
          BuildMI(*MBB, I, I->getDebugLoc(), TII->get(AMDGPU::S_DCACHE_WB));
        }
      }
    }
  }

  for (MachineInstr *I : RemoveMI)
    I->eraseFromParent();

  if (!MFI->isEntryFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to to the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    MachineBasicBlock &EntryBB = MF.front();
    BuildMI(EntryBB, EntryBB.getFirstNonPHI(), DebugLoc(), TII->get(AMDGPU::S_WAITCNT))
      .addImm(0);

    Changes = true;
  }

  return Changes;
}
