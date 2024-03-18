//===----------- RISCVSpillRewrite.cpp - RISC-V Spill Rewrite -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that rewrite spills and reloads to
// reduce the instruction latency by changing full register
// store/load(VS1R/VL1R) to fractional store/load(VSE/VLE) needed and expands.
//
// The algorithm finds and rewrites spills(VS1R) to VSE if the spilled vreg only
// needs fraction of a vreg(determined by the last write instruction's LMUL),
// note that if the spilled register comes from different BB, it will find the
// union LMUL of each defined BB. After then, it rewrites reloads(VL1R) to VLE
// follows the corresponding spills in the spill slots. The algorithm runs until
// there's no any rewrite.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/TargetParser/RISCVTargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-spill-rewrite"
#define RISCV_SPILL_REWRITE_NAME "RISC-V Spill Rewrite pass"

namespace {
static inline bool isSpillInst(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::VS1R_V;
}

static inline bool isReloadInst(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::VL1RE8_V ||
         MI.getOpcode() == RISCV::VL2RE8_V ||
         MI.getOpcode() == RISCV::VL4RE8_V ||
         MI.getOpcode() == RISCV::VL8RE8_V ||
         MI.getOpcode() == RISCV::VL1RE16_V ||
         MI.getOpcode() == RISCV::VL2RE16_V ||
         MI.getOpcode() == RISCV::VL4RE16_V ||
         MI.getOpcode() == RISCV::VL8RE16_V ||
         MI.getOpcode() == RISCV::VL1RE32_V ||
         MI.getOpcode() == RISCV::VL2RE32_V ||
         MI.getOpcode() == RISCV::VL4RE32_V ||
         MI.getOpcode() == RISCV::VL8RE32_V ||
         MI.getOpcode() == RISCV::VL1RE64_V ||
         MI.getOpcode() == RISCV::VL2RE64_V ||
         MI.getOpcode() == RISCV::VL4RE64_V ||
         MI.getOpcode() == RISCV::VL8RE64_V;
}

static inline bool hasSpillSlotObject(const MachineFrameInfo *MFI,
                                      const MachineInstr &MI,
                                      bool IsReload = false) {
  unsigned MemOpIdx = IsReload ? 2 : 1;
  if (MI.getNumOperands() <= MemOpIdx || !MI.getOperand(MemOpIdx).isFI())
    return false;

  int FI = MI.getOperand(MemOpIdx).getIndex();
  return MFI->isSpillSlotObjectIndex(FI);
}

static inline RISCVII::VLMUL maxLMUL(RISCVII::VLMUL LMUL1,
                                     RISCVII::VLMUL LMUL2) {
  int LMUL1Val = std::numeric_limits<int>::min();
  int LMUL2Val = std::numeric_limits<int>::min();

  if (LMUL1 != RISCVII::LMUL_RESERVED) {
    auto DecodedLMUL1 = RISCVVType::decodeVLMUL(LMUL1);
    LMUL1Val = DecodedLMUL1.second ? -DecodedLMUL1.first : DecodedLMUL1.first;
  }
  if (LMUL2 != RISCVII::LMUL_RESERVED) {
    auto DecodedLMUL2 = RISCVVType::decodeVLMUL(LMUL2);
    LMUL2Val = DecodedLMUL2.second ? -DecodedLMUL2.first : DecodedLMUL2.first;
  }

  return LMUL1Val > LMUL2Val ? LMUL1 : LMUL2;
}

static inline RISCVII::VLMUL getWidenedFracLMUL(RISCVII::VLMUL LMUL) {
  if (LMUL == RISCVII::LMUL_F8)
    return RISCVII::LMUL_F4;
  if (LMUL == RISCVII::LMUL_F4)
    return RISCVII::LMUL_F2;
  if (LMUL == RISCVII::LMUL_F2)
    return RISCVII::LMUL_1;

  llvm_unreachable("The LMUL is supposed to be fractional.");
}

class RISCVSpillRewrite : public MachineFunctionPass {
  const RISCVSubtarget *ST = nullptr;
  const TargetInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineFrameInfo *MFI = nullptr;
  LiveIntervals *LIS = nullptr;

public:
  static char ID;
  RISCVSpillRewrite() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return RISCV_SPILL_REWRITE_NAME; }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  bool tryToRewrite(MachineFunction &MF);

  // This function find the Reg's LMUL in its defining inst, if there're
  // multiple instructions that define the Reg in different BB, recursively find
  // them and return the maximum LMUL that are found. If it can't be found due
  // to any reason such as the register is dead, it returns
  // RISCVII::LMUL_RESERVE which means the Reg can't be rewritten.
  // BegI represents the starting instruction in the beginning, this is used to
  // determine whether it encounters a loop, if so then the defining instruction
  // doesn't exist in this MBB.
  RISCVII::VLMUL
  findDefiningInstUnionLMUL(MachineBasicBlock &MBB, Register Reg,
                            DenseMap<MachineInstr *, bool> &Visited,
                            MachineBasicBlock::reverse_iterator BegI = nullptr);
  bool tryToRewriteSpill(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                         std::map<int, RISCVII::VLMUL> &SpillLMUL);
  bool tryToRewriteReload(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                          int FI,
                          const std::map<int, RISCVII::VLMUL> &SpillLMUL);
};

} // end anonymous namespace

char RISCVSpillRewrite::ID = 0;

INITIALIZE_PASS(RISCVSpillRewrite, DEBUG_TYPE, RISCV_SPILL_REWRITE_NAME, false,
                false)

void RISCVSpillRewrite::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();

  AU.addPreserved<LiveIntervalsWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addRequired<SlotIndexesWrapperPass>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addPreserved<LiveStacks>();

  MachineFunctionPass::getAnalysisUsage(AU);
}

RISCVII::VLMUL RISCVSpillRewrite::findDefiningInstUnionLMUL(
    MachineBasicBlock &MBB, Register Reg,
    DenseMap<MachineInstr *, bool> &Visited,
    MachineBasicBlock::reverse_iterator BegI) {
  for (auto I = (BegI == nullptr ? MBB.rbegin() : BegI); I != MBB.rend(); ++I) {
    if (I->isDebugInstr())
      continue;

    // Return the minimum LMUL if this MBB is a loop body and we meet the
    // instruction that is already visited, it means the LMUL in this MBB is
    // dont-care.
    if (Visited.contains(&*I))
      return RISCVII::LMUL_F8;

    Visited[&*I];
    if (I->definesRegister(Reg, nullptr)) {
      if (I->registerDefIsDead(Reg, nullptr))
        return RISCVII::LMUL_RESERVED;

      if (isReloadInst(*I))
        return RISCVII::LMUL_1;

      if (auto DstSrcPair = TII->isCopyInstr(*I))
        return findDefiningInstUnionLMUL(MBB, DstSrcPair->Source->getReg(),
                                         Visited, *++I);

      const uint64_t TSFlags = I->getDesc().TSFlags;
      assert(RISCVII::hasSEWOp(TSFlags));

      // If the instruction is tail undisturbed, we need to preserve the full
      // vector register since the tail data might be used somewhere.
      if (RISCVII::hasVecPolicyOp(TSFlags)) {
        const MachineOperand &PolicyOp =
            I->getOperand(I->getNumExplicitOperands() - 1);
        if ((PolicyOp.getImm() & RISCVII::TAIL_AGNOSTIC) == 0)
          return RISCVII::VLMUL::LMUL_1;
      }

      RISCVII::VLMUL LMUL = RISCVII::getLMul(TSFlags);
      if (RISCVII::isRVVWideningReduction(TSFlags)) {
        // Widening reduction produces only single element result, so we just
        // need to calculate LMUL for single element.
        int Log2SEW =
            I->getOperand(RISCVII::getSEWOpNum(I->getDesc())).getImm();
        int Log2LMUL = Log2SEW - Log2_64(ST->getELen());
        LMUL =
            static_cast<RISCVII::VLMUL>(Log2LMUL < 0 ? Log2LMUL + 8 : Log2LMUL);
      }
      if (RISCVII::isWiden(TSFlags))
        LMUL = getWidenedFracLMUL(LMUL);

      return LMUL;
    }
  }

  assert(MBB.isLiveIn(Reg));

  // If Reg's defining inst is not found in this BB, find it in it's
  // predecessors.
  RISCVII::VLMUL LMUL = RISCVII::LMUL_RESERVED;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    RISCVII::VLMUL PredLMUL = findDefiningInstUnionLMUL(*P, Reg, Visited);
    if (PredLMUL == RISCVII::LMUL_RESERVED)
      continue;

    if (LMUL == RISCVII::LMUL_RESERVED) {
      LMUL = PredLMUL;
      continue;
    }

    LMUL = maxLMUL(LMUL, PredLMUL);
  }

  return LMUL;
}

bool RISCVSpillRewrite::tryToRewriteSpill(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
    std::map<int, RISCVII::VLMUL> &SpillLMUL) {
  Register SrcReg = I->getOperand(0).getReg();
  unsigned Opcode = 0;
  DenseMap<MachineInstr *, bool> Visited;
  // Find the nearest inst defines this spilled reg.
  RISCVII::VLMUL LMUL = findDefiningInstUnionLMUL(MBB, SrcReg, Visited, *I);
  // If the register's defined inst just defines partial of register, we only
  // need to store partial register.
  switch (LMUL) {
  case RISCVII::LMUL_F2:
    Opcode = RISCV::PseudoVSE8_V_MF2;
    break;
  case RISCVII::LMUL_F4:
    Opcode = RISCV::PseudoVSE8_V_MF4;
    break;
  case RISCVII::LMUL_F8:
    Opcode = RISCV::PseudoVSE8_V_MF8;
    break;
  default:
    break;
  }

  // No need to rewrite.
  if (!Opcode)
    return false;

  int FI = I->getOperand(1).getIndex();
  auto updateLMUL = [&](RISCVII::VLMUL LMUL) {
    assert(!SpillLMUL.count(FI) &&
           "Each frame index should only be used once.");
    SpillLMUL[FI] = LMUL;
  };

  if (Opcode == RISCV::PseudoVSE8_V_MF2)
    updateLMUL(RISCVII::LMUL_F2);
  else if (Opcode == RISCV::PseudoVSE8_V_MF4)
    updateLMUL(RISCVII::LMUL_F4);
  else if (Opcode == RISCV::PseudoVSE8_V_MF8)
    updateLMUL(RISCVII::LMUL_F8);

  MachineInstr *Vse = BuildMI(MBB, I, DebugLoc(), TII->get(Opcode))
                          .add(I->getOperand(0))
                          .addFrameIndex(FI)
                          .addImm(-1 /*VL Max*/)
                          .addImm(3 /*SEW = 8*/);
  LIS->InsertMachineInstrInMaps(*Vse);

  return true;
}

bool RISCVSpillRewrite::tryToRewriteReload(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I, int FI,
    const std::map<int, RISCVII::VLMUL> &SpillLMUL) {
  // Partial reload case
  // If this frame doesn't have corresponding reload op, just skip it.
  if (!SpillLMUL.count(FI))
    return false;

  unsigned Opcode = 0;
  switch (SpillLMUL.at(FI)) {
  case RISCVII::LMUL_F2:
    Opcode = RISCV::PseudoVLE8_V_MF2;
    break;
  case RISCVII::LMUL_F4:
    Opcode = RISCV::PseudoVLE8_V_MF4;
    break;
  case RISCVII::LMUL_F8:
    Opcode = RISCV::PseudoVLE8_V_MF8;
    break;
  default:
    break;
  }

  if (!Opcode)
    return false;

  bool IsRenamable = I->getOperand(0).isRenamable();
  MachineInstr *Vle =
      BuildMI(MBB, I, I->getDebugLoc(), TII->get(Opcode))
          .addReg(I->getOperand(0).getReg(),
                  RegState::Define | getRenamableRegState(IsRenamable))
          .addReg(I->getOperand(0).getReg(),
                  RegState::Undef | getRenamableRegState(IsRenamable))
          .addFrameIndex(FI)
          .addImm(-1 /*VL Max*/)
          .addImm(3 /*SEW = 8*/)
          .addImm(3 /*TAMA*/);
  LIS->InsertMachineInstrInMaps(*Vle);

  return true;
}

bool RISCVSpillRewrite::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  MFI = &MF.getFrameInfo();
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();

  // 1. If the MI is vector spill and need partial spill, record frame number
  // and the corresponding LMUL in SpillLMUL.
  // 2. If the MI is vector reload, change it's load instruction if found in
  // SpillLMUL.
  // Note that this pass is run before stack slot coloring pass, so it doesn't
  // need to consider stack slot reuse.
  bool Changed = false;
  bool TempChanged = false;
  std::map<int, RISCVII::VLMUL> SpillLMUL;
  do {
    TempChanged = false;
    for (MachineBasicBlock &MBB : MF) {
      // We are not able to insert vsetvli for inline assembly, so the
      // following case will fail if the spill rewrite is presented:
      // ```
      // %0 = vadd.vv v8, v9 (e8, mf2)
      // vs1r %0, %stack.0
      // ...
      // inline_asm("vsetvli 888, e8, m1")
      // %1 = vl1r %stack.0
      // inline_asm("vadd.vv %a, %b, %c", %a=v8, %b=%1, %c=%1)
      // ```
      // The vs1r %0, %stack.0 will be rewrite to vse8.v with lmul=mf2, then
      // %1 will also reload lmul=mf2 elements which break the original
      // context which need to be lmul=1.
      if (any_of(MBB, [](const MachineInstr &MI) { return MI.isInlineAsm(); }))
        continue;

      for (auto &MI : llvm::make_early_inc_range(MBB)) {
        int FI;
        if (hasSpillSlotObject(MFI, MI))
          FI = MI.getOperand(1).getIndex();
        else if (hasSpillSlotObject(MFI, MI, true))
          FI = MI.getOperand(2).getIndex();
        else
          continue;

        if (isSpillInst(MI) && tryToRewriteSpill(MBB, MI, SpillLMUL)) {
          LIS->RemoveMachineInstrFromMaps(MI);
          MI.eraseFromParent();
          Changed = true;
          TempChanged = true;
        } else if (isReloadInst(MI) &&
                   tryToRewriteReload(MBB, MI, FI, SpillLMUL)) {
          LIS->RemoveMachineInstrFromMaps(MI);
          MI.eraseFromParent();
          Changed = true;
          TempChanged = true;
        }

        if (MI == MBB.end())
          break;
      }
    }
  } while (TempChanged);

  return Changed;
}

/// Returns an instance of the RVV Spill Rewrite pass.
FunctionPass *llvm::createRISCVSpillRewritePass() {
  return new RISCVSpillRewrite();
}
