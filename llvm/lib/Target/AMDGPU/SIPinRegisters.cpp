//===-- SIPinRegisters.cpp - Register pinning hints -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lowers the PIN_{VGPR,AGPR}_B* pseudos (from llvm.amdgcn.pin.{vgpr,agpr})
/// into a copy plus an allocation hint naming the requested register tuple.
/// The hint is a preference rather than an assignment: the allocator takes the
/// tuple when it is free and falls back to its normal order otherwise, so a pin
/// can change where a value lands but never whether the code is correct.
///
/// Runs pre-RA in SSA form (before PHIElimination / TwoAddressInstruction), so
/// each pinned value still has a single reaching def and one hint covers all of
/// it. The hint reaches RegAllocGreedy through MachineRegisterInfo, and follows
/// the value through coalescing via SIRegisterInfo::updateRegAllocHint().
///
/// A hint places a value as reliably as a fixed assignment for as long as the
/// value stays one live range, loop-carried PHIs included. What it cannot
/// express is a pin whose chain the control flow splits in two: an accumulator
/// running through an unrolled main loop and then a remainder loop that may
/// iterate zero times leaves the main loop's range live across the remainder
/// loop, so the two ranges overlap and coalescing cannot merge them. Both then
/// demand the one tuple, only one can have it, and the loser is split into
/// fresh vregs that do not inherit the hint.
///
/// Placing fewer values is not by itself worse. On a 4-wave gfx1250 GEMM the
/// hint placed 432 of 640 accumulators where pre-coloring placed all 640, yet
/// emitted fewer instructions, fewer S_SET_VGPR_MSB and fewer waits at equal
/// VGPR count, with no spill either way and the same throughput. What the
/// placement buys is not the registers themselves but the distance between a
/// write and the read of it: pinning an accumulation chain to its own tuple
/// keeps unrelated values out of it, and SIInsertWaitcnts, which scores
/// physical registers after allocation, can then prove the longer distance and
/// relax s_wait_alu depctr_va_vdst from a full pipeline drain to a bound that
/// leaves work in flight.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-pin-regs"

STATISTIC(NumPins, "Number of values hinted at the requested register");
STATISTIC(NumNoOpPins, "Number of pins dropped as a no-op");

// Which allocation hint a pin uses: the AMDGPU Pin hint (which can make the
// pinned tuple the only candidate, see SIRegisterInfo) or a plain copy
// preference.
static cl::opt<bool> PinHintKind(
    "amdgpu-pin-hint-kind", cl::init(true), cl::Hidden,
    cl::desc("Use the AMDGPU Pin allocation hint for a register pin (else a "
             "plain simple hint)"));

// Whether a pin also grows the VGPR budget to cover its tuple. Without this a
// high pin can never be honored, since the allocation order stops at the
// occupancy-derived limit.
static cl::opt<bool> PinReservesVGPRs(
    "amdgpu-pin-soft-reserves-vgprs", cl::init(false), cl::Hidden,
    cl::desc("Let a register pin raise the VGPR budget to cover the pinned "
             "tuple"));

// Whether a pin lowers to a copy rather than rewriting the pin's uses.
static cl::opt<bool> PinCopy(
    "amdgpu-pin-soft-copy", cl::init(true), cl::Hidden,
    cl::desc("Lower a pin to a copy instead of rewriting its uses"));

// If set, convert an AGPR-pinned input's MFMA to the mixed vgprcd form
// (v[C], a[A], a[B]) so the accumulator stays in VGPR; else keep the native
// all-AGPR form (a[D], a[A], a[B], a[C]).
static cl::opt<bool> PinAgprVgprC(
    "amdgpu-pin-agpr-vgpr-c", cl::init(true), cl::Hidden,
    cl::desc("Convert an AGPR-input MFMA to vgprcd to keep its accumulator in "
             "VGPR (else keep the native all-AGPR form)"));

namespace {

class SIPinRegisters : public MachineFunctionPass {
public:
  static char ID;

  SIPinRegisters() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI pin registers"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIPinRegisters::ID = 0;

char &llvm::SIPinRegistersID = SIPinRegisters::ID;

INITIALIZE_PASS(SIPinRegisters, DEBUG_TYPE, "SI pin registers", false, false)

FunctionPass *llvm::createSIPinRegistersPass() { return new SIPinRegisters(); }

static bool isPinPseudo(const SIInstrInfo *TII, const MachineInstr &MI) {
  StringRef N = TII->getName(MI.getOpcode());
  return N.starts_with("PIN_VGPR_B") || N.starts_with("PIN_AGPR_B");
}

// Physical register tuple a pin targets, or 0 if it is not a legal member of
// the destination register class (e.g. a misaligned start on a target that
// requires aligned tuples).
static MCRegister getPinPhysReg(const SIRegisterInfo *TRI,
                                const TargetRegisterClass *RC, unsigned RegNo) {
  unsigned First =
      (TRI->isAGPRClass(RC) ? AMDGPU::AGPR0 : AMDGPU::VGPR0) + RegNo;
  MCRegister PR = TRI->getRegSizeInBits(*RC) == 32
                      ? MCRegister(First)
                      : TRI->getMatchingSuperReg(First, AMDGPU::sub0, RC);
  if (PR && RC->contains(PR))
    return PR;
  return MCRegister();
}

bool SIPinRegisters::runOnMachineFunction(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  SmallVector<MachineInstr *, 8> Pins;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (isPinPseudo(TII, MI))
        Pins.push_back(&MI);

  if (Pins.empty())
    return false;

  // A pin that cannot reach its register leaves correct but possibly slower
  // code, so it is a remark rather than a diagnostic. It is worth reporting at
  // all because nothing in the source says which of the rules below a value
  // fell foul of: -Rpass-missed=si-pin-regs names the pin and the reason.
  MachineOptimizationRemarkEmitter ORE(MF, /*MBFI=*/nullptr);
  auto remark = [&](const MachineInstr *Pin, bool WantAGPR, unsigned RegNo,
                    StringRef Why) {
    std::string Tgt = (WantAGPR ? "a" : "v") + std::to_string(RegNo);
    ORE.emit([&] {
      return MachineOptimizationRemarkMissed(DEBUG_TYPE, "PinDropped",
                                             Pin->getDebugLoc(),
                                             Pin->getParent())
             << "pin to " << Tgt << " was dropped: " << Why;
    });
  };

  // Highest VGPR a pin needs, +1 (drives the occupancy cap).
  unsigned ReqVGPRs = 0;

  for (MachineInstr *Pin : Pins) {
    assert(Pin->getNumExplicitOperands() == 3 &&
           "pin pseudo must be (dst, src, regno)");
    Register Dst = Pin->getOperand(0).getReg();
    Register Src = Pin->getOperand(1).getReg();
    unsigned RegNo = Pin->getOperand(2).getImm();
    const TargetRegisterClass *RC = MRI.getRegClass(Dst);
    MCRegister PR = getPinPhysReg(TRI, RC, RegNo);

    unsigned NumRegs = TRI->getRegSizeInBits(*RC) / 32;
    bool WantAGPR = TRI->isAGPRClass(RC);

    // Targets without an AGPR file (e.g. RDNA) cannot honor an AGPR pin.
    // Degrade to a no-op -- forward the source to the uses and drop the pin --
    // so the value stays in its natural VGPR location instead of failing
    // register allocation with "no registers from class available".
    if (WantAGPR && !ST.hasMAIInsts()) {
      for (MachineOperand &MO :
           llvm::make_early_inc_range(MRI.use_operands(Dst)))
        MO.setReg(Src);
      if (Src.isVirtual())
        MRI.constrainRegClass(Src, TRI->getEquivalentVGPRClass(RC));
      remark(Pin, WantAGPR, RegNo,
             "the target has no AGPRs; the value stays in a VGPR");
      Pin->eraseFromParent();
      ++NumNoOpPins;
      continue;
    }

    // Narrow the pinned value's register file to VGPR or AGPR (a class
    // narrowing, not a physreg assignment, so it also works for loop-carried
    // PHIs and no-ops when the file is incompatible).
    {
      // Constrain the copy/REG_SEQUENCE/PHI/tie-connected component of `Seeds`.
      // MFMA src2<->vdst edges are followed only when `FollowAcc`; otherwise an
      // MFMA using a member as src0/src1 is recorded in `Inputs` as a leaf, so
      // an input pin does not drag the loop-carried accumulator into the AGPR
      // file. `Recompute` re-derives classes from defs first (needed after an
      // opcode conversion, since constrainRegClass cannot cross the AGPR/VGPR
      // files).
      auto constrainComponent = [&](ArrayRef<Register> Seeds, bool AGPRFile,
                                    bool FollowAcc, bool Recompute,
                                    SmallPtrSetImpl<MachineInstr *> &Inputs) {
        DenseSet<Register> Seen;
        SmallVector<Register, 16> WL;
        auto Add = [&](Register R) {
          if (R.isVirtual() && Seen.insert(R).second)
            WL.push_back(R);
        };
        for (Register R : Seeds)
          Add(R);
        for (unsigned I = 0; I < WL.size(); ++I) {
          for (MachineOperand &MO : MRI.reg_operands(WL[I])) {
            MachineInstr *MI = MO.getParent();
            // Copy/REG_SEQUENCE/PHI just move the value between vregs; pull in
            // every register operand. PHI keeps a loop-carried accumulator in
            // one file (else it needs an agpr<->vgpr copy each iteration).
            if (MI->isCopy() || MI->isRegSequence() || MI->isPHI()) {
              for (MachineOperand &O : MI->operands())
                if (O.isReg())
                  Add(O.getReg());
            }
            if (MO.isTied())
              Add(MI->getOperand(MI->findTiedOperandIdx(MO.getOperandNo()))
                      .getReg());
            if (TII->isMAI(*MI)) {
              int S0 = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                  AMDGPU::OpName::src0);
              int S1 = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                  AMDGPU::OpName::src1);
              int S2 = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                  AMDGPU::OpName::src2);
              unsigned OpNo = MO.getOperandNo();
              bool IsInput = (S0 >= 0 && OpNo == (unsigned)S0) ||
                             (S1 >= 0 && OpNo == (unsigned)S1);
              if (IsInput && !FollowAcc) {
                Inputs.insert(MI);
              } else if (FollowAcc && S2 >= 0) {
                if (MI->getOperand(0).isReg())
                  Add(MI->getOperand(0).getReg());
                if (MI->getOperand(S2).isReg())
                  Add(MI->getOperand(S2).getReg());
              }
            }
          }
        }
        for (Register R : WL) {
          // A constant accumulator init (e.g. clear()==0) placed in an AGPR by
          // V_ACCVGPR_WRITE can't be constrained to VGPR; rewrite it to V_MOV
          // so the constant is born in VGPR instead of copied from AGPR each
          // launch.
          if (!AGPRFile)
            for (MachineInstr &Def :
                 make_early_inc_range(MRI.def_instructions(R))) {
              if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64 &&
                  Def.getNumOperands() >= 2 && Def.getOperand(1).isImm())
                Def.setDesc(TII->get(AMDGPU::V_MOV_B32_e32));
            }
          if (Recompute)
            MRI.recomputeRegClass(R);
          unsigned Sz = TRI->getRegSizeInBits(*MRI.getRegClass(R));
          const TargetRegisterClass *Want =
              AGPRFile ? TRI->getAGPRClassForBitWidth(Sz)
                       : TRI->getVGPRClassForBitWidth(Sz);
          if (Want)
            MRI.constrainRegClass(R, Want);
        }
      };

      SmallPtrSet<MachineInstr *, 8> InputMFMAs;
      Register Seeds[] = {Src, Dst};
      // Constrain the pinned value's own component to its file. For an AGPR
      // input pin, stop at the MFMAs that consume it (recorded in InputMFMAs).
      constrainComponent(Seeds, /*AGPRFile=*/WantAGPR, /*FollowAcc=*/!WantAGPR,
                         /*Recompute=*/false, InputMFMAs);

      // ISel picks the all-AGPR MFMA form when the function needs AGPRs. To
      // keep the accumulator in VGPR, convert each consuming MFMA to vgprcd and
      // constrain its accumulator (vdst/srcC chain) to VGPR, re-deriving
      // classes from the converted defs. The chain stays coalesced in VGPR (no
      // chunked pins, no agpr<->vgpr shuffle).
      if (WantAGPR && PinAgprVgprC && !InputMFMAs.empty()) {
        SmallVector<Register, 8> AccSeeds;
        for (MachineInstr *MI : InputMFMAs) {
          int VOp = AMDGPU::getMFMASrcCVDstVGPROp(MI->getOpcode());
          if (VOp == -1)
            continue; // already vgprcd form
          MI->setDesc(TII->get(VOp));
          if (MI->getOperand(0).isReg())
            AccSeeds.push_back(MI->getOperand(0).getReg());
          int S2 =
              AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::src2);
          if (S2 >= 0 && MI->getOperand(S2).isReg())
            AccSeeds.push_back(MI->getOperand(S2).getReg());
        }
        if (!AccSeeds.empty()) {
          SmallPtrSet<MachineInstr *, 8> Ignore;
          constrainComponent(AccSeeds, /*AGPRFile=*/false, /*FollowAcc=*/true,
                             /*Recompute=*/true, Ignore);
        }
      }
    }

    // A sub-register source means the value is a slice of a shared register
    // (e.g. one ds_read2 loads two pinned fragments into one wide reg). Hinting
    // it would ask the allocator to move overlapping physreg sub-slices. The
    // shared reg is already in the right file (above), so the pin is redundant:
    // forward the source (sub)register to the uses and drop it.
    if (Pin->getOperand(1).getSubReg()) {
      unsigned SubIdx = Pin->getOperand(1).getSubReg();
      for (MachineOperand &MO :
           llvm::make_early_inc_range(MRI.use_operands(Dst))) {
        MO.setSubReg(TRI->composeSubRegIndices(SubIdx, MO.getSubReg()));
        MO.setReg(Src);
      }
      remark(Pin, WantAGPR, RegNo,
             "the value is a slice of a wider register, so placing it would "
             "move the lanes it shares; pin at the width the value is used");
      Pin->eraseFromParent();
      ++NumNoOpPins;
      continue;
    }

    if (!PR)
      remark(Pin, WantAGPR, RegNo,
             "no register tuple of this width and alignment starts there");

    // The copy a pin lowers to is redundant by construction -- source and
    // destination hold the same value -- and coalescing normally removes it.
    // But each copy is one more merge that has to succeed, and a merge that
    // fails leaves the chain as two overlapping live ranges competing for the
    // one tuple, which no hint can satisfy. Rewriting the uses instead keeps
    // the chain in one piece from the start.
    bool Rewrote = false;
    if (!PinCopy && Src.isVirtual() &&
        MRI.getRegClassOrNull(Src) == MRI.getRegClassOrNull(Dst)) {
      MRI.replaceRegWith(Dst, Src);
      Dst = Src;
      Rewrote = true;
    }
    if (!Rewrote)
      BuildMI(*Pin->getParent(), Pin, Pin->getDebugLoc(),
              TII->get(TargetOpcode::COPY), Dst)
          .addReg(Src);

    if (PR) {
      auto SetHint = [&](Register R) {
        if (PinHintKind)
          MRI.setRegAllocationHint(R, AMDGPURI::Pin, PR);
        else
          MRI.setSimpleHint(R, PR);
      };
      // A REG_SEQUENCE that assembles the pinned value out of several loads
      // needs no hint of its own: coalescing folds it into the hinted value, so
      // long as the tuple is used as a whole. A tuple read back in pieces
      // instead has a disconnected live range, gets split into fresh vregs
      // (which do not inherit the hint) and is not placed -- see the
      // slot-granularity note in the pin docs.
      SetHint(Dst);
      if (Src.isVirtual())
        SetHint(Src);
      // A hint can only be honored if the tuple is inside the VGPR budget: the
      // allocation order stops at the occupancy-derived limit, so a pin above
      // it is not merely unlikely, it is unreachable. Reserving costs occupancy
      // even when the allocator then ignores the hint, hence the flag. AGPRs
      // are a separate file that does not affect the VGPR budget.
      if (PinReservesVGPRs && !WantAGPR)
        ReqVGPRs = std::max(ReqVGPRs, RegNo + NumRegs);
      ++NumPins;
    }
    Pin->eraseFromParent();
  }

  // Cap occupancy so a wide VGPR-resident pinned value fits the per-wave
  // budget without the user setting __launch_bounds__.
  if (unsigned Req = ReqVGPRs) {
    auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
    // Occupancy achievable while reserving `Req` registers per wave; cap the
    // waves-per-EU (and hence the RA's VGPR budget) so the pinned range fits.
    unsigned Occ =
        ST.getOccupancyWithNumVGPRs(Req, MFI->getDynamicVGPRBlockSize());
    auto WPE = MFI->getWavesPerEU();
    unsigned NewMax = WPE.second ? std::min(WPE.second, Occ) : Occ;
    // Only cap the *max* occupancy; keep the min low (1 unless the function
    // already required more), since forcing min==max over-constrains the
    // allocator.
    unsigned NewMin = std::min(WPE.first ? WPE.first : 1u, NewMax);
    MFI->setWavesPerEU(NewMin, NewMax);
    MFI->limitOccupancy(NewMax);
  }

  return true;
}
