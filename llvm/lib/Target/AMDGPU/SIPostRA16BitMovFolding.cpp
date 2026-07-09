//===-- SIPostRA16BitMovFolding.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass performs the post RA 16bit Mov folding
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/BranchProbability.h"
using namespace llvm;

#define DEBUG_TYPE "si-post-ra-16bit-mov-folding"

namespace {

class SIPostRA16BitMovFolding {
private:
  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;

  void getMovB16Info(const MachineInstr &MI, const SIRegisterInfo *TRI,
                     MCRegister &SrcReg16, bool &SrcIsVGPR,
                     MCRegister &SrcReg32, bool &SrcIsHi, bool &SrcIsImm,
                     int64_t &ImmVal) const;

  bool mergeSingleMovB16Pair(MachineInstr &Lo, MachineInstr &Hi,
                             bool IsHiFirst) const;
  bool mergeMovB16Pairs(MachineFunction &MF) const;
public:
  bool run(MachineFunction &MF);
};

class SIPostRA16BitMovFoldingLegacy: public MachineFunctionPass {
public:
  static char ID;

  SIPostRA16BitMovFoldingLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "SI post-RA 16bit Mov Folding";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
	AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    return SIPostRA16BitMovFolding().run(MF);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(SIPostRA16BitMovFoldingLegacy, DEBUG_TYPE,
                "SI Post RA 16bit Mov Folding", false, false)

char SIPostRA16BitMovFoldingLegacy::ID = 0;

char &llvm::SIPostRA16BitMovFoldingLegacyID = SIPostRA16BitMovFoldingLegacy::ID;

// Helper: extract the src operand and whether it is from the hi16 half.
// Post-RA, both V_MOV_B16_t16_e32 and V_MOV_B16_t16_e64 use VGPR_16 dst
// physical registers whose encoding already encodes hi/lo (IS_HI16 bit).
void SIPostRA16BitMovFolding::getMovB16Info(const MachineInstr &MI,
                                      const SIRegisterInfo *TRI,
                                      MCRegister &SrcReg16, bool &SrcIsVGPR,
                                      MCRegister &SrcReg32, bool &SrcIsHi,
                                      bool &SrcIsImm, int64_t &ImmVal) const {
  SrcIsImm = false;
  SrcIsHi = false;
  SrcIsVGPR = false;
  SrcReg16 = MCRegister();
  SrcReg32 = MCRegister();

  const MachineOperand *SrcOp = TII->getNamedOperand(MI, AMDGPU::OpName::src0);

  if (SrcOp->isImm()) {
    SrcIsImm = true;
    ImmVal = SrcOp->getImm();
    return;
  }

  SrcReg16 = SrcOp->getReg().asMCReg();
  SrcIsVGPR = AMDGPU::VGPR_16RegClass.contains(SrcReg16);
  if (SrcIsVGPR) {
    SrcIsHi = AMDGPU::isHi16Reg(SrcReg16, *TRI);
    SrcReg32 = TRI->get32BitRegister(SrcReg16);
  } else {
    SrcIsHi = false;
    SrcReg32 = SrcReg16;
  }
}

// clang-format off
// Try to merge a pair of v_mov_b16 instructions targeting the lo16 and hi16
// halves of the same VGPR into a single 32-bit instruction.
//
// Caller guarantee the pair to be two v_mov_b16 and targets the same dst32
//
// Patterns:
//   v_mov_b16 v0.h, 0        v_mov_b16 v0.l, v2.l/s2  => v_and_b32  v0,0xffff,v2/s2
//   v_mov_b16 v0.h, 0        v_mov_b16 v0.l, v2.h     => v_lshrrev_b32 v0,16,v2
//   v_mov_b16 v0.l, 0        v_mov_b16 v0.h, v2.l/s2  => v_lshlrev_b32 v0,16,v2/s2
//   v_mov_b16 v0.l, 0        v_mov_b16 v0.h, v2.h     => v_and_b32  v0,0xffff0000,v2
//   v_mov_b16 v0.l, v.x/s    v_mov_b16 v0.h, v.y/s    => v_pack_b32_f16 v0, v/s, v/s
// clang-format on
bool SIPostRA16BitMovFolding::mergeSingleMovB16Pair(MachineInstr &Lo,
                                              MachineInstr &Hi,
                                              bool IsHiFirst) const {
  // Lo and Hi share the same Dst32
  MCRegister LoDst = Lo.getOperand(0).getReg().asMCReg();
  MCRegister HiDst = Hi.getOperand(0).getReg().asMCReg();
  MCRegister Dst32 = TRI->get32BitRegister(LoDst);

  // Extract source info for Lo and Hi.
  MCRegister LoSrc16, LoSrc32, HiSrc16, HiSrc32;
  bool LoSrcIsHi, HiSrcIsHi, LoSrcIsImm, HiSrcIsImm, LoSrcIsVGPR, HiSrcIsVGPR;
  int64_t LoImm = 0, HiImm = 0;

  getMovB16Info(Lo, TRI, LoSrc16, LoSrcIsVGPR, LoSrc32, LoSrcIsHi, LoSrcIsImm,
                LoImm);
  getMovB16Info(Hi, TRI, HiSrc16, HiSrcIsVGPR, HiSrc32, HiSrcIsHi, HiSrcIsImm,
                HiImm);

  MachineInstr &FirstMI = IsHiFirst ? Hi : Lo;
  MachineInstr &SecondMI = IsHiFirst ? Lo : Hi;

  // Data Conflict counter
  MachineBasicBlock::iterator UpperBound = SecondMI.getIterator();
  MachineBasicBlock::iterator LowerBound = FirstMI.getIterator();
  unsigned LoopCnt = 0, UpperBoundCnt = UINT_MAX, LowerBoundCnt = 0;

  MachineBasicBlock &MBB = *Lo.getParent();

  // Check that between Lo and Hi, there are no instructions that:
  // - modify Dst32
  // - modify LoSrc16 or HiSrc16 depending on order (data dependency)
  // We scan from the instruction after the first mov up to (but not including)
  // the second mov.
  MCRegister FirstSrc16 = IsHiFirst ? HiSrc16 : LoSrc16;
  MCRegister FirstDst16 = IsHiFirst ? HiDst : LoDst;
  MCRegister SecondSrc16 = IsHiFirst ? LoSrc16 : HiSrc16;
  MCRegister SecondDst16 = IsHiFirst ? LoDst : HiDst;
  for (MachineInstr &Scan :
       drop_begin(make_range(FirstMI.getIterator(), SecondMI.getIterator()))) {
    if (Scan.modifiesRegister(Dst32, TRI))
      return false;
    LoopCnt++;
    if (LoopCnt < UpperBoundCnt &&
        ((FirstSrc16 && Scan.modifiesRegister(FirstSrc16, TRI)) ||
         Scan.readsRegister(FirstDst16, TRI))) {
      UpperBound = Scan.getIterator();
      UpperBoundCnt = LoopCnt;
    }
    if (LoopCnt > LowerBoundCnt &&
        ((SecondSrc16 && Scan.modifiesRegister(SecondSrc16, TRI)) ||
         Scan.readsRegister(SecondDst16, TRI))) {
      LowerBound = Scan.getIterator();
      LowerBoundCnt = LoopCnt;
    }
  }

  // No spot maintains data dependency
  if (LowerBoundCnt >= UpperBoundCnt)
    return false;

  // Insert MI before selected. Any spots between (LowerBound, UpperBound] would
  // work
  MachineInstr &Selected = *UpperBound;
  const DebugLoc &DL = Selected.getDebugLoc();

  // Now match patterns and emit the replacement instruction.
  // Insert on Selected MI location, then remove both mov.

  // Pattern: v_mov_b16 v0.l, v2.x/s2 + v_mov_b16 v0.h, v3.y/s3
  //   => v_pack_b32_f16 v0,v2.x/s2,v3.y/s3
  if (!HiSrcIsImm && !LoSrcIsImm) {
    BuildMI(MBB, Selected, DL, TII->get(AMDGPU::V_PACK_B32_F16_t16_e64), Dst32)
        .addImm(0) // SrcMod
        .addReg(LoSrc16)
        .addImm(0) // SrcMod
        .addReg(HiSrc16)
        .addImm(0)  // Clamp
        .addImm(0); // Opsel
    Lo.eraseFromParent();
    Hi.eraseFromParent();
    return true;
  }

  bool Usevop2 =
      AMDGPU::VGPR_32_Lo128RegClass.contains(Dst32) &&
      (LoSrcIsImm ||
       (LoSrcIsVGPR && AMDGPU::VGPR_32_Lo128RegClass.contains(LoSrc32))) &&
      (HiSrcIsImm ||
       (HiSrcIsVGPR && AMDGPU::VGPR_32_Lo128RegClass.contains(HiSrc32)));

  // Pattern: v_mov_b16 v0.h, 0  +  v_mov_b16 v0.l, v2.l/s2
  //   => v_and_b32 v0, 0x0000ffff, v2/s2
  if (HiSrcIsImm && HiImm == 0 && !LoSrcIsImm && !LoSrcIsHi) {
    BuildMI(MBB, Selected, DL,
            TII->get(Usevop2 ? AMDGPU::V_AND_B32_e32 : AMDGPU::V_AND_B32_e64),
            Dst32)
        .addImm(0x0000ffff)
        .addReg(LoSrc32);
    Lo.eraseFromParent();
    Hi.eraseFromParent();
    return true;
  }

  // Pattern: v_mov_b16 v0.h, 0  +  v_mov_b16 v0.l, v2.h
  //   => v_lshrrev_b32 v0, 16, v2
  if (HiSrcIsImm && HiImm == 0 && !LoSrcIsImm && LoSrcIsHi) {
    BuildMI(MBB, Selected, DL,
            TII->get(Usevop2 ? AMDGPU::V_LSHRREV_B32_e32
                             : AMDGPU::V_LSHRREV_B32_e64),
            Dst32)
        .addImm(16)
        .addReg(LoSrc32);
    Lo.eraseFromParent();
    Hi.eraseFromParent();
    return true;
  }

  // Pattern: v_mov_b16 v0.l, 0  +  v_mov_b16 v0.h, v2.l/s2
  //   => v_lshlrev_b32 v0, 16, v2/s2
  if (LoSrcIsImm && LoImm == 0 && !HiSrcIsImm && !HiSrcIsHi) {
    BuildMI(MBB, Selected, DL,
            TII->get(Usevop2 ? AMDGPU::V_LSHLREV_B32_e32
                             : AMDGPU::V_LSHLREV_B32_e64),
            Dst32)
        .addImm(16)
        .addReg(HiSrc32);
    Lo.eraseFromParent();
    Hi.eraseFromParent();
    return true;
  }

  // Pattern: v_mov_b16 v0.l, 0  +  v_mov_b16 v0.h, v2.h
  //   => v_and_b32 v0, 0xffff0000, v2
  if (LoSrcIsImm && LoImm == 0 && !HiSrcIsImm && HiSrcIsHi) {
    BuildMI(MBB, Selected, DL,
            TII->get(Usevop2 ? AMDGPU::V_AND_B32_e32 : AMDGPU::V_AND_B32_e64),
            Dst32)
        .addImm(0xffff0000)
        .addReg(HiSrc32);
    Lo.eraseFromParent();
    Hi.eraseFromParent();
    return true;
  }

  return false;
}

// Merge pairs of v_mov_b16 targeting the lo16 and hi16 halves of the same
// VGPR into a single 32-bit instruction (true16 mode only).
bool SIPostRA16BitMovFolding::mergeMovB16Pairs(MachineFunction &MF) const {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    // Map from 32-bit VGPR to the pending v_mov_b16 and its age.
    // Tracks how many non-mov-b16 instructions have passed since the
    // 16-bit write using a fixed size circular buffer
    struct Pending {
      MCRegister Dst32;
      MachineInstr *MI;
      unsigned IsHi;
    };
    // Search window size
    const unsigned ScanLimit = 16;
    std::array<Pending, ScanLimit> CirBuf = {{{MCRegister(), nullptr, false}}};
    SmallDenseMap<MCRegister, unsigned> PendingWrites;
    unsigned Head = 0;

    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      if (MI.isDebugInstr())
        continue;

      unsigned Opc = MI.getOpcode();
      bool IsMovB16 = (Opc == AMDGPU::V_MOV_B16_t16_e32 ||
                       Opc == AMDGPU::V_MOV_B16_t16_e64);

      if (++Head == ScanLimit)
        Head = 0;

      // Expire the last one
      PendingWrites.erase(CirBuf[Head].Dst32);

      if (!IsMovB16) {
        CirBuf[Head] = {MCRegister(), nullptr, false};
        continue;
      }

	  LLVM_DEBUG(dbgs() << "Checking MI:" << MI << "\n");
      MCRegister DstReg = MI.getOperand(0).getReg().asMCReg();
      bool DstIsHi = AMDGPU::isHi16Reg(DstReg, *TRI);
      MCRegister Dst32 = TRI->get32BitRegister(DstReg);

      // Insert new one
      CirBuf[Head] = {Dst32, &MI, DstIsHi};

      auto [It, Inserted] = PendingWrites.insert({Dst32, Head});
      if (!Inserted) {
        if (CirBuf[It->second].IsHi == DstIsHi) {
          It->second = Head;
          continue;
        }

        // Look for a matching pending write.
        MachineInstr &LoMI = !DstIsHi ? MI : *CirBuf[It->second].MI;
        MachineInstr &HiMI = DstIsHi ? MI : *CirBuf[It->second].MI;
        bool IsHiFirst = CirBuf[It->second].IsHi;
        if (mergeSingleMovB16Pair(LoMI, HiMI, IsHiFirst)) {
          Changed = true;
          PendingWrites.erase(It);
        } else {
          It->second = Head;
        }
      }
    }
  }

  return Changed;
}

PreservedAnalyses
llvm::SIPostRA16BitMovFoldingPass::run(MachineFunction &MF,
                                       MachineFunctionAnalysisManager &MFAM) {
  SIPostRA16BitMovFolding().run(MF);
  return PreservedAnalyses::all();
}

bool SIPostRA16BitMovFolding::run(MachineFunction &MF) {
  const GCNSubtarget& ST = MF.getSubtarget<GCNSubtarget>();
  TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  TII = ST.getInstrInfo();
  bool Changed = false;

  // Try merge B16 Pair in true16 mode
  if (ST.useRealTrue16Insts())
    Changed |= mergeMovB16Pairs(MF);

  return Changed;
}
