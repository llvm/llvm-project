//===- AArch64SpeculationHardening.cpp - Harden Against Missspeculation  --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass to insert code to mitigate against side channel
// vulnerabilities that may happen under control flow miss-speculation.
//
// The pass implements tracking of control flow miss-speculation into a "taint"
// register. That taint register can then be used to mask off registers with
// sensitive data when executing under miss-speculation, a.k.a. "transient
// execution".
// This pass is aimed at mitigating against SpectreV1-style vulnarabilities.
//
// At the moment, it implements the tracking of miss-speculation of control
// flow into a taint register, but doesn't implement a mechanism yet to then
// use that taint register to mask of vulnerable data in registers (something
// for a follow-on improvement). Possible strategies to mask out vulnerable
// data that can be implemented on top of this are:
// - speculative load hardening to automatically mask of data loaded
//   in registers.
// - using intrinsics to mask of data in registers as indicated by the
//   programmer (see https://lwn.net/Articles/759423/).
//
// For AArch64, the following implementation choices are made below.
// Some of these are different than the implementation choices made in
// the similar pass implemented in X86SpeculativeLoadHardening.cpp, as
// the instruction set characteristics result in different trade-offs.
// - The speculation hardening is done after register allocation. With a
//   relative abundance of registers, one register is reserved (X16) to be
//   the taint register. X16 is expected to not clash with other register
//   reservation mechanisms with very high probability because:
//   . The AArch64 ABI doesn't guarantee X16 to be retained across any call.
//   . The only way to request X16 to be used as a programmer is through
//     inline assembly. In the rare case a function explicitly demands to
//     use X16/W16, this pass falls back to hardening against speculation
//     by inserting a DSB SYS/ISB barrier pair which will prevent control
//     flow speculation.
// - It is easy to insert mask operations at this late stage as we have
//   mask operations available that don't set flags.
// - The taint variable contains all-ones when no miss-speculation is detected,
//   and contains all-zeros when miss-speculation is detected. Therefore, when
//   masking, an AND instruction (which only changes the register to be masked,
//   no other side effects) can easily be inserted anywhere that's needed.
// - The tracking of miss-speculation is done by using a data-flow conditional
//   select instruction (CSEL) to evaluate the flags that were also used to
//   make conditional branch direction decisions. Speculation of the CSEL
//   instruction can be limited with a CSDB instruction - so the combination of
//   CSEL + a later CSDB gives the guarantee that the flags as used in the CSEL
//   aren't speculated. When conditional branch direction gets miss-speculated,
//   the semantics of the inserted CSEL instruction is such that the taint
//   register will contain all zero bits.
//   One key requirement for this to work is that the conditional branch is
//   followed by an execution of the CSEL instruction, where the CSEL
//   instruction needs to use the same flags status as the conditional branch.
//   This means that the conditional branches must not be implemented as one
//   of the AArch64 conditional branches that do not use the flags as input
//   (CB(N)Z and TB(N)Z). This is implemented by ensuring in the instruction
//   selectors to not produce these instructions when speculation hardening
//   is enabled. This pass will assert if it does encounter such an instruction.
// - On function call boundaries, the miss-speculation state is transferred from
//   the taint register X16 to be encoded in the SP register as value 0.
//
// Future extensions/improvements could be:
// - Implement this functionality using full speculation barriers, akin to the
//   x86-slh-lfence option. This may be more useful for the intrinsics-based
//   approach than for the SLH approach to masking.
//   Note that this pass already inserts the full speculation barriers if the
//   function for some niche reason makes use of X16/W16.
// - no indirect branch misprediction gets protected/instrumented; but this
//   could be done for some indirect branches, such as switch jump tables.
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "aarch64-speculation-hardening"

#define AARCH64_SPECULATION_HARDENING_NAME "AArch64 speculation hardening pass"

namespace {

class AArch64SpeculationHardening : public MachineFunctionPass {
public:
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  static char ID;

  AArch64SpeculationHardening() : MachineFunctionPass(ID) {
    initializeAArch64SpeculationHardeningPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override {
    return AARCH64_SPECULATION_HARDENING_NAME;
  }

private:
  unsigned MisspeculatingTaintReg;
  bool UseControlFlowSpeculationBarrier;

  bool functionUsesHardeningRegister(MachineFunction &MF) const;
  bool instrumentControlFlow(MachineBasicBlock &MBB);
  bool endsWithCondControlFlow(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                               MachineBasicBlock *&FBB,
                               AArch64CC::CondCode &CondCode) const;
  void insertTrackingCode(MachineBasicBlock &SplitEdgeBB,
                          AArch64CC::CondCode &CondCode, DebugLoc DL) const;
  void insertSPToRegTaintPropagation(MachineBasicBlock *MBB,
                                     MachineBasicBlock::iterator MBBI) const;
  void insertRegToSPTaintPropagation(MachineBasicBlock *MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned TmpReg) const;
};

} // end anonymous namespace

char AArch64SpeculationHardening::ID = 0;

INITIALIZE_PASS(AArch64SpeculationHardening, "aarch64-speculation-hardening",
                AARCH64_SPECULATION_HARDENING_NAME, false, false)

bool AArch64SpeculationHardening::endsWithCondControlFlow(
    MachineBasicBlock &MBB, MachineBasicBlock *&TBB, MachineBasicBlock *&FBB,
    AArch64CC::CondCode &CondCode) const {
  SmallVector<MachineOperand, 1> analyzeBranchCondCode;
  if (TII->analyzeBranch(MBB, TBB, FBB, analyzeBranchCondCode, false))
    return false;

  // Ignore if the BB ends in an unconditional branch/fall-through.
  if (analyzeBranchCondCode.empty())
    return false;

  // If the BB ends with a single conditional branch, FBB will be set to
  // nullptr (see API docs for TII->analyzeBranch). For the rest of the
  // analysis we want the FBB block to be set always.
  assert(TBB != nullptr);
  if (FBB == nullptr)
    FBB = MBB.getFallThrough();

  // If both the true and the false condition jump to the same basic block,
  // there isn't need for any protection - whether the branch is speculated
  // correctly or not, we end up executing the architecturally correct code.
  if (TBB == FBB)
    return false;

  assert(MBB.succ_size() == 2);
  // translate analyzeBranchCondCode to CondCode.
  assert(analyzeBranchCondCode.size() == 1 && "unknown Cond array format");
  CondCode = AArch64CC::CondCode(analyzeBranchCondCode[0].getImm());
  return true;
}

void AArch64SpeculationHardening::insertTrackingCode(
    MachineBasicBlock &SplitEdgeBB, AArch64CC::CondCode &CondCode,
    DebugLoc DL) const {
  if (UseControlFlowSpeculationBarrier) {
    // insert full control flow speculation barrier (DSB SYS + ISB)
    BuildMI(SplitEdgeBB, SplitEdgeBB.begin(), DL, TII->get(AArch64::ISB))
        .addImm(0xf);
    BuildMI(SplitEdgeBB, SplitEdgeBB.begin(), DL, TII->get(AArch64::DSB))
        .addImm(0xf);
  } else {
    BuildMI(SplitEdgeBB, SplitEdgeBB.begin(), DL, TII->get(AArch64::CSELXr))
        .addDef(MisspeculatingTaintReg)
        .addUse(MisspeculatingTaintReg)
        .addUse(AArch64::XZR)
        .addImm(CondCode);
    SplitEdgeBB.addLiveIn(AArch64::NZCV);
  }
}

bool AArch64SpeculationHardening::instrumentControlFlow(
    MachineBasicBlock &MBB) {
  LLVM_DEBUG(dbgs() << "Instrument control flow tracking on MBB: " << MBB);

  bool Modified = false;
  MachineBasicBlock *TBB = nullptr;
  MachineBasicBlock *FBB = nullptr;
  AArch64CC::CondCode CondCode;

  if (!endsWithCondControlFlow(MBB, TBB, FBB, CondCode)) {
    LLVM_DEBUG(dbgs() << "... doesn't end with CondControlFlow\n");
  } else {
    // Now insert:
    // "CSEL MisSpeculatingR, MisSpeculatingR, XZR, cond" on the True edge and
    // "CSEL MisSpeculatingR, MisSpeculatingR, XZR, Invertcond" on the False
    // edge.
    AArch64CC::CondCode InvCondCode = AArch64CC::getInvertedCondCode(CondCode);

    MachineBasicBlock *SplitEdgeTBB = MBB.SplitCriticalEdge(TBB, *this);
    MachineBasicBlock *SplitEdgeFBB = MBB.SplitCriticalEdge(FBB, *this);

    assert(SplitEdgeTBB != nullptr);
    assert(SplitEdgeFBB != nullptr);

    DebugLoc DL;
    if (MBB.instr_end() != MBB.instr_begin())
      DL = (--MBB.instr_end())->getDebugLoc();

    insertTrackingCode(*SplitEdgeTBB, CondCode, DL);
    insertTrackingCode(*SplitEdgeFBB, InvCondCode, DL);

    LLVM_DEBUG(dbgs() << "SplitEdgeTBB: " << *SplitEdgeTBB << "\n");
    LLVM_DEBUG(dbgs() << "SplitEdgeFBB: " << *SplitEdgeFBB << "\n");
    Modified = true;
  }

  // Perform correct code generation around function calls and before returns.
  {
    SmallVector<MachineInstr *, 4> ReturnInstructions;
    SmallVector<MachineInstr *, 4> CallInstructions;

    for (MachineInstr &MI : MBB) {
      if (MI.isReturn())
        ReturnInstructions.push_back(&MI);
      else if (MI.isCall())
        CallInstructions.push_back(&MI);
    }

    Modified |=
        (ReturnInstructions.size() > 0) || (CallInstructions.size() > 0);

    for (MachineInstr *Return : ReturnInstructions)
      insertRegToSPTaintPropagation(Return->getParent(), Return, AArch64::X17);
    for (MachineInstr *Call : CallInstructions) {
      // Just after the call:
      MachineBasicBlock::iterator i = Call;
      i++;
      insertSPToRegTaintPropagation(Call->getParent(), i);
      // Just before the call:
      insertRegToSPTaintPropagation(Call->getParent(), Call, AArch64::X17);
    }
  }

  return Modified;
}

void AArch64SpeculationHardening::insertSPToRegTaintPropagation(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI) const {
  // If full control flow speculation barriers are used, emit a control flow
  // barrier to block potential miss-speculation in flight coming in to this
  // function.
  if (UseControlFlowSpeculationBarrier) {
    // insert full control flow speculation barrier (DSB SYS + ISB)
    BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::DSB)).addImm(0xf);
    BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::ISB)).addImm(0xf);
    return;
  }

  // CMP   SP, #0   === SUBS   xzr, SP, #0
  BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::SUBSXri))
      .addDef(AArch64::XZR)
      .addUse(AArch64::SP)
      .addImm(0)
      .addImm(0); // no shift
  // CSETM x16, NE  === CSINV  x16, xzr, xzr, EQ
  BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::CSINVXr))
      .addDef(MisspeculatingTaintReg)
      .addUse(AArch64::XZR)
      .addUse(AArch64::XZR)
      .addImm(AArch64CC::EQ);
}

void AArch64SpeculationHardening::insertRegToSPTaintPropagation(
    MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
    unsigned TmpReg) const {
  // If full control flow speculation barriers are used, there will not be
  // miss-speculation when returning from this function, and therefore, also
  // no need to encode potential miss-speculation into the stack pointer.
  if (UseControlFlowSpeculationBarrier)
    return;

  // mov   Xtmp, SP  === ADD  Xtmp, SP, #0
  BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::ADDXri))
      .addDef(TmpReg)
      .addUse(AArch64::SP)
      .addImm(0)
      .addImm(0); // no shift
  // and   Xtmp, Xtmp, TaintReg === AND Xtmp, Xtmp, TaintReg, #0
  BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::ANDXrs))
      .addDef(TmpReg, RegState::Renamable)
      .addUse(TmpReg, RegState::Kill | RegState::Renamable)
      .addUse(MisspeculatingTaintReg, RegState::Kill)
      .addImm(0);
  // mov   SP, Xtmp === ADD SP, Xtmp, #0
  BuildMI(*MBB, MBBI, DebugLoc(), TII->get(AArch64::ADDXri))
      .addDef(AArch64::SP)
      .addUse(TmpReg, RegState::Kill)
      .addImm(0)
      .addImm(0); // no shift
}

bool AArch64SpeculationHardening::functionUsesHardeningRegister(
    MachineFunction &MF) const {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // treat function calls specially, as the hardening register does not
      // need to remain live across function calls.
      if (MI.isCall())
        continue;
      if (MI.readsRegister(MisspeculatingTaintReg, TRI) ||
          MI.modifiesRegister(MisspeculatingTaintReg, TRI))
        return true;
    }
  }
  return false;
}

bool AArch64SpeculationHardening::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().hasFnAttribute(Attribute::SpeculativeLoadHardening))
    return false;

  MisspeculatingTaintReg = AArch64::X16;
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  bool Modified = false;

  UseControlFlowSpeculationBarrier = functionUsesHardeningRegister(MF);

  // Instrument control flow speculation tracking, if requested.
  LLVM_DEBUG(
      dbgs()
      << "***** AArch64SpeculationHardening - track control flow *****\n");

  // 1. Add instrumentation code to function entry and exits.
  SmallVector<MachineBasicBlock *, 2> EntryBlocks;
  EntryBlocks.push_back(&MF.front());
  for (const LandingPadInfo &LPI : MF.getLandingPads())
    EntryBlocks.push_back(LPI.LandingPadBlock);
  for (auto Entry : EntryBlocks)
    insertSPToRegTaintPropagation(
        Entry, Entry->SkipPHIsLabelsAndDebug(Entry->begin()));

  // 2. Add instrumentation code to every basic block.
  for (auto &MBB : MF)
    Modified |= instrumentControlFlow(MBB);

  return Modified;
}

/// \brief Returns an instance of the pseudo instruction expansion pass.
FunctionPass *llvm::createAArch64SpeculationHardeningPass() {
  return new AArch64SpeculationHardening();
}
