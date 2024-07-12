//===- AArch64SLSHardening.cpp - Harden Straight Line Missspeculation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass to insert code to mitigate against side channel
// vulnerabilities that may happen under straight line miss-speculation.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/CodeGen/IndirectThunks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "aarch64-sls-hardening"

#define AARCH64_SLS_HARDENING_NAME "AArch64 sls hardening pass"

static const char SLSBLRNamePrefix[] = "__llvm_slsblr_thunk_";

namespace {

// Set of inserted thunks: bitmask with bits corresponding to
// indexes in SLSBLRThunks array.
typedef uint32_t ThunksSet;

struct SLSHardeningInserter : ThunkInserter<SLSHardeningInserter, ThunksSet> {
public:
  const char *getThunkPrefix() { return SLSBLRNamePrefix; }
  bool mayUseThunk(const MachineFunction &MF) {
    ComdatThunks &= !MF.getSubtarget<AArch64Subtarget>().hardenSlsNoComdat();
    // We are inserting barriers aside from thunk calls, so
    // check hardenSlsRetBr() as well.
    return MF.getSubtarget<AArch64Subtarget>().hardenSlsBlr() ||
           MF.getSubtarget<AArch64Subtarget>().hardenSlsRetBr();
  }
  ThunksSet insertThunks(MachineModuleInfo &MMI, MachineFunction &MF,
                         ThunksSet ExistingThunks);
  void populateThunk(MachineFunction &MF);

private:
  bool ComdatThunks = true;

  bool hardenReturnsAndBRs(MachineModuleInfo &MMI, MachineBasicBlock &MBB);
  bool hardenBLRs(MachineModuleInfo &MMI, MachineBasicBlock &MBB,
                  ThunksSet &Thunks);

  void convertBLRToBL(MachineModuleInfo &MMI, MachineBasicBlock &MBB,
                      MachineBasicBlock::instr_iterator MBBI,
                      ThunksSet &Thunks);
};

} // end anonymous namespace

static void insertSpeculationBarrier(const AArch64Subtarget *ST,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     DebugLoc DL,
                                     bool AlwaysUseISBDSB = false) {
  assert(MBBI != MBB.begin() &&
         "Must not insert SpeculationBarrierEndBB as only instruction in MBB.");
  assert(std::prev(MBBI)->isBarrier() &&
         "SpeculationBarrierEndBB must only follow unconditional control flow "
         "instructions.");
  assert(std::prev(MBBI)->isTerminator() &&
         "SpeculationBarrierEndBB must only follow terminators.");
  const TargetInstrInfo *TII = ST->getInstrInfo();
  unsigned BarrierOpc = ST->hasSB() && !AlwaysUseISBDSB
                            ? AArch64::SpeculationBarrierSBEndBB
                            : AArch64::SpeculationBarrierISBDSBEndBB;
  if (MBBI == MBB.end() ||
      (MBBI->getOpcode() != AArch64::SpeculationBarrierSBEndBB &&
       MBBI->getOpcode() != AArch64::SpeculationBarrierISBDSBEndBB))
    BuildMI(MBB, MBBI, DL, TII->get(BarrierOpc));
}

ThunksSet SLSHardeningInserter::insertThunks(MachineModuleInfo &MMI,
                                             MachineFunction &MF,
                                             ThunksSet ExistingThunks) {
  const AArch64Subtarget *ST = &MF.getSubtarget<AArch64Subtarget>();

  for (auto &MBB : MF) {
    if (ST->hardenSlsRetBr())
      hardenReturnsAndBRs(MMI, MBB);
    if (ST->hardenSlsBlr())
      hardenBLRs(MMI, MBB, ExistingThunks);
  }
  return ExistingThunks;
}

static bool isBLR(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AArch64::BLR:
  case AArch64::BLRNoIP:
    return true;
  case AArch64::BLRAA:
  case AArch64::BLRAB:
  case AArch64::BLRAAZ:
  case AArch64::BLRABZ:
    llvm_unreachable("Currently, LLVM's code generator does not support "
                     "producing BLRA* instructions. Therefore, there's no "
                     "support in this pass for those instructions.");
  }
  return false;
}

bool SLSHardeningInserter::hardenReturnsAndBRs(MachineModuleInfo &MMI,
                                               MachineBasicBlock &MBB) {
  const AArch64Subtarget *ST =
      &MBB.getParent()->getSubtarget<AArch64Subtarget>();
  bool Modified = false;
  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator(), E = MBB.end();
  MachineBasicBlock::iterator NextMBBI;
  for (; MBBI != E; MBBI = NextMBBI) {
    MachineInstr &MI = *MBBI;
    NextMBBI = std::next(MBBI);
    if (MI.isReturn() || isIndirectBranchOpcode(MI.getOpcode())) {
      assert(MI.isTerminator());
      insertSpeculationBarrier(ST, MBB, std::next(MBBI), MI.getDebugLoc());
      Modified = true;
    }
  }
  return Modified;
}

static const unsigned NumPermittedRegs = 29;
static const struct ThunkNameAndReg {
  const char* Name;
  Register Reg;
} SLSBLRThunks[NumPermittedRegs] = {
    {"__llvm_slsblr_thunk_x0", AArch64::X0},
    {"__llvm_slsblr_thunk_x1", AArch64::X1},
    {"__llvm_slsblr_thunk_x2", AArch64::X2},
    {"__llvm_slsblr_thunk_x3", AArch64::X3},
    {"__llvm_slsblr_thunk_x4", AArch64::X4},
    {"__llvm_slsblr_thunk_x5", AArch64::X5},
    {"__llvm_slsblr_thunk_x6", AArch64::X6},
    {"__llvm_slsblr_thunk_x7", AArch64::X7},
    {"__llvm_slsblr_thunk_x8", AArch64::X8},
    {"__llvm_slsblr_thunk_x9", AArch64::X9},
    {"__llvm_slsblr_thunk_x10", AArch64::X10},
    {"__llvm_slsblr_thunk_x11", AArch64::X11},
    {"__llvm_slsblr_thunk_x12", AArch64::X12},
    {"__llvm_slsblr_thunk_x13", AArch64::X13},
    {"__llvm_slsblr_thunk_x14", AArch64::X14},
    {"__llvm_slsblr_thunk_x15", AArch64::X15},
    // X16 and X17 are deliberately missing, as the mitigation requires those
    // register to not be used in BLR. See comment in ConvertBLRToBL for more
    // details.
    {"__llvm_slsblr_thunk_x18", AArch64::X18},
    {"__llvm_slsblr_thunk_x19", AArch64::X19},
    {"__llvm_slsblr_thunk_x20", AArch64::X20},
    {"__llvm_slsblr_thunk_x21", AArch64::X21},
    {"__llvm_slsblr_thunk_x22", AArch64::X22},
    {"__llvm_slsblr_thunk_x23", AArch64::X23},
    {"__llvm_slsblr_thunk_x24", AArch64::X24},
    {"__llvm_slsblr_thunk_x25", AArch64::X25},
    {"__llvm_slsblr_thunk_x26", AArch64::X26},
    {"__llvm_slsblr_thunk_x27", AArch64::X27},
    {"__llvm_slsblr_thunk_x28", AArch64::X28},
    {"__llvm_slsblr_thunk_x29", AArch64::FP},
    // X30 is deliberately missing, for similar reasons as X16 and X17 are
    // missing.
    {"__llvm_slsblr_thunk_x31", AArch64::XZR},
};

unsigned getThunkIndex(Register Reg) {
  for (unsigned I = 0; I < NumPermittedRegs; ++I)
    if (SLSBLRThunks[I].Reg == Reg)
      return I;
  llvm_unreachable("Unexpected register");
}

void SLSHardeningInserter::populateThunk(MachineFunction &MF) {
  assert(MF.getFunction().hasComdat() == ComdatThunks &&
         "ComdatThunks value changed since MF creation");
  // FIXME: How to better communicate Register number, rather than through
  // name and lookup table?
  assert(MF.getName().starts_with(getThunkPrefix()));
  auto ThunkIt = llvm::find_if(
      SLSBLRThunks, [&MF](auto T) { return T.Name == MF.getName(); });
  assert(ThunkIt != std::end(SLSBLRThunks));
  Register ThunkReg = ThunkIt->Reg;

  const TargetInstrInfo *TII =
      MF.getSubtarget<AArch64Subtarget>().getInstrInfo();

  // Depending on whether this pass is in the same FunctionPassManager as the
  // IR->MIR conversion, the thunk may be completely empty, or contain a single
  // basic block with a single return instruction. Normalise it to contain a
  // single empty basic block.
  if (MF.size() == 1) {
    assert(MF.front().size() == 1);
    assert(MF.front().front().getOpcode() == AArch64::RET);
    MF.front().erase(MF.front().begin());
  } else {
    assert(MF.size() == 0);
    MF.push_back(MF.CreateMachineBasicBlock());
  }

  MachineBasicBlock *Entry = &MF.front();
  Entry->clear();

  //  These thunks need to consist of the following instructions:
  //  __llvm_slsblr_thunk_xN:
  //      BR xN
  //      barrierInsts
  Entry->addLiveIn(ThunkReg);
  // MOV X16, ThunkReg == ORR X16, XZR, ThunkReg, LSL #0
  BuildMI(Entry, DebugLoc(), TII->get(AArch64::ORRXrs), AArch64::X16)
      .addReg(AArch64::XZR)
      .addReg(ThunkReg)
      .addImm(0);
  BuildMI(Entry, DebugLoc(), TII->get(AArch64::BR)).addReg(AArch64::X16);
  // Make sure the thunks do not make use of the SB extension in case there is
  // a function somewhere that will call to it that for some reason disabled
  // the SB extension locally on that function, even though it's enabled for
  // the module otherwise. Therefore set AlwaysUseISBSDB to true.
  insertSpeculationBarrier(&MF.getSubtarget<AArch64Subtarget>(), *Entry,
                           Entry->end(), DebugLoc(), true /*AlwaysUseISBDSB*/);
}

void SLSHardeningInserter::convertBLRToBL(
    MachineModuleInfo &MMI, MachineBasicBlock &MBB,
    MachineBasicBlock::instr_iterator MBBI, ThunksSet &Thunks) {
  // Transform a BLR to a BL as follows:
  // Before:
  //   |-----------------------------|
  //   |      ...                    |
  //   |  instI                      |
  //   |  BLR xN                     |
  //   |  instJ                      |
  //   |      ...                    |
  //   |-----------------------------|
  //
  // After:
  //   |-----------------------------|
  //   |      ...                    |
  //   |  instI                      |
  //   |  BL __llvm_slsblr_thunk_xN  |
  //   |  instJ                      |
  //   |      ...                    |
  //   |-----------------------------|
  //
  //   __llvm_slsblr_thunk_xN:
  //   |-----------------------------|
  //   |  BR xN                      |
  //   |  barrierInsts               |
  //   |-----------------------------|
  //
  // This function merely needs to transform BLR xN into BL
  // __llvm_slsblr_thunk_xN.
  //
  // Since linkers are allowed to clobber X16 and X17 on function calls, the
  // above mitigation only works if the original BLR instruction was not
  // BLR X16 nor BLR X17. Code generation before must make sure that no BLR
  // X16|X17 was produced if the mitigation is enabled.

  MachineInstr &BLR = *MBBI;
  assert(isBLR(BLR));
  unsigned BLOpcode;
  Register Reg;
  bool RegIsKilled;
  switch (BLR.getOpcode()) {
  case AArch64::BLR:
  case AArch64::BLRNoIP:
    BLOpcode = AArch64::BL;
    Reg = BLR.getOperand(0).getReg();
    assert(Reg != AArch64::X16 && Reg != AArch64::X17 && Reg != AArch64::LR);
    RegIsKilled = BLR.getOperand(0).isKill();
    break;
  case AArch64::BLRAA:
  case AArch64::BLRAB:
  case AArch64::BLRAAZ:
  case AArch64::BLRABZ:
    llvm_unreachable("BLRA instructions cannot yet be produced by LLVM, "
                     "therefore there is no need to support them for now.");
  default:
    llvm_unreachable("unhandled BLR");
  }
  DebugLoc DL = BLR.getDebugLoc();

  MachineFunction &MF = *MBBI->getMF();
  MCContext &Context = MBB.getParent()->getContext();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  unsigned ThunkIndex = getThunkIndex(Reg);
  StringRef ThunkName = SLSBLRThunks[ThunkIndex].Name;
  MCSymbol *Sym = Context.getOrCreateSymbol(ThunkName);
  if (!(Thunks & (1u << ThunkIndex))) {
    Thunks |= 1u << ThunkIndex;
    createThunkFunction(MMI, ThunkName, ComdatThunks);
  }

  MachineInstr *BL = BuildMI(MBB, MBBI, DL, TII->get(BLOpcode)).addSym(Sym);

  // Now copy the implicit operands from BLR to BL and copy other necessary
  // info.
  // However, both BLR and BL instructions implictly use SP and implicitly
  // define LR. Blindly copying implicit operands would result in SP and LR
  // operands to be present multiple times. While this may not be too much of
  // an issue, let's avoid that for cleanliness, by removing those implicit
  // operands from the BL created above before we copy over all implicit
  // operands from the BLR.
  int ImpLROpIdx = -1;
  int ImpSPOpIdx = -1;
  for (unsigned OpIdx = BL->getNumExplicitOperands();
       OpIdx < BL->getNumOperands(); OpIdx++) {
    MachineOperand Op = BL->getOperand(OpIdx);
    if (!Op.isReg())
      continue;
    if (Op.getReg() == AArch64::LR && Op.isDef())
      ImpLROpIdx = OpIdx;
    if (Op.getReg() == AArch64::SP && !Op.isDef())
      ImpSPOpIdx = OpIdx;
  }
  assert(ImpLROpIdx != -1);
  assert(ImpSPOpIdx != -1);
  int FirstOpIdxToRemove = std::max(ImpLROpIdx, ImpSPOpIdx);
  int SecondOpIdxToRemove = std::min(ImpLROpIdx, ImpSPOpIdx);
  BL->removeOperand(FirstOpIdxToRemove);
  BL->removeOperand(SecondOpIdxToRemove);
  // Now copy over the implicit operands from the original BLR
  BL->copyImplicitOps(MF, BLR);
  MF.moveCallSiteInfo(&BLR, BL);
  // Also add the register called in the BLR as being used in the called thunk.
  BL->addOperand(MachineOperand::CreateReg(Reg, false /*isDef*/, true /*isImp*/,
                                           RegIsKilled /*isKill*/));
  // Remove BLR instruction
  MBB.erase(MBBI);
}

bool SLSHardeningInserter::hardenBLRs(MachineModuleInfo &MMI,
                                      MachineBasicBlock &MBB,
                                      ThunksSet &Thunks) {
  bool Modified = false;
  MachineBasicBlock::instr_iterator MBBI = MBB.instr_begin(),
                                    E = MBB.instr_end();
  MachineBasicBlock::instr_iterator NextMBBI;
  for (; MBBI != E; MBBI = NextMBBI) {
    MachineInstr &MI = *MBBI;
    NextMBBI = std::next(MBBI);
    if (isBLR(MI)) {
      convertBLRToBL(MMI, MBB, MBBI, Thunks);
      Modified = true;
    }
  }
  return Modified;
}

namespace {
class AArch64SLSHardening : public ThunkInserterPass<SLSHardeningInserter> {
public:
  static char ID;

  AArch64SLSHardening() : ThunkInserterPass(ID) {}

  StringRef getPassName() const override { return AARCH64_SLS_HARDENING_NAME; }
};

} // end anonymous namespace

char AArch64SLSHardening::ID = 0;

INITIALIZE_PASS(AArch64SLSHardening, "aarch64-sls-hardening",
                AARCH64_SLS_HARDENING_NAME, false, false)

FunctionPass *llvm::createAArch64SLSHardeningPass() {
  return new AArch64SLSHardening();
}
