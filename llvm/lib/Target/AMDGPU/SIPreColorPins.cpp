//===-- SIPreColorPins.cpp - Hard register pinning ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lowers the PIN_{VGPR,AGPR}_B* pseudos (from llvm.amdgcn.pin.{vgpr,agpr})
/// into a hard physical-register assignment ("pre-coloring"): the pinned
/// value's def and uses are rewritten to reference the requested VGPR/AGPR
/// tuple directly, so the allocator treats it as fixed interference and cannot
/// override it (unlike a soft hint). The whole tie-connected component is
/// rewritten together, so a pin on an MFMA accumulator input also pins its tied
/// output.
///
/// When hard pinning is unsafe (a PHI/REG_SEQUENCE/IMPLICIT_DEF def, a physreg
/// illegal for some operand's class, or a tuple conflicting with an existing
/// hard pin) the pass falls back to a COPY plus a soft allocation hint, so it
/// never regresses correctness. Runs pre-RA in SSA form (before PHIElimination
/// / TwoAddressInstruction), so each value has a single reaching def.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-pre-color-pins"

static cl::opt<bool> EnableHardPin(
    "amdgpu-hard-pin-regs", cl::init(true), cl::Hidden,
    cl::desc("Use hard register pre-coloring for llvm.amdgcn.pin.* (else soft "
             "allocation hints only)"));

// If set, convert an AGPR-pinned input's MFMA to the mixed vgprcd form
// (v[C], a[A], a[B]) so the accumulator stays in VGPR; else keep the native
// all-AGPR form (a[D], a[A], a[B], a[C]).
static cl::opt<bool> PinAgprVgprC(
    "amdgpu-pin-agpr-vgpr-c", cl::init(true), cl::Hidden,
    cl::desc("Convert an AGPR-input MFMA to vgprcd to keep its accumulator in "
             "VGPR (else keep the native all-AGPR form)"));

namespace {

class SIPreColorPins : public MachineFunctionPass {
public:
  static char ID;

  SIPreColorPins() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI pre-color pinned registers";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char SIPreColorPins::ID = 0;

char &llvm::SIPreColorPinsID = SIPreColorPins::ID;

INITIALIZE_PASS(SIPreColorPins, DEBUG_TYPE, "SI pre-color pinned registers",
                false, false)

FunctionPass *llvm::createSIPreColorPinsPass() { return new SIPreColorPins(); }

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

bool SIPreColorPins::runOnMachineFunction(MachineFunction &MF) {
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

  // Regunits already claimed by a hard pin. A later pin overlapping any claimed
  // unit falls back to soft, so two distinct live values never share a physreg.
  // Reuse by a single value (e.g. an accumulation chain) is instead absorbed
  // into the first pin's tie-connected component, making later pins on it
  // no-ops.
  DenseSet<MCRegUnit> Claimed;
  bool NeedRecomputeLiveIns = false;
  unsigned ReqVGPRs =
      0; // highest VGPR a pin needs, +1 (drives the occupancy cap)
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
    // Degrade to a soft no-op -- forward the source to the uses and drop the
    // pin -- so the value stays in its natural VGPR location instead of failing
    // register allocation with "no registers from class available".
    if (WantAGPR && !ST.hasMAIInsts()) {
      for (MachineOperand &MO :
           llvm::make_early_inc_range(MRI.use_operands(Dst)))
        MO.setReg(Src);
      if (Src.isVirtual())
        MRI.constrainRegClass(Src, TRI->getEquivalentVGPRClass(RC));
      Pin->eraseFromParent();
      continue;
    }
    // Only VGPR pins drive the occupancy cap (see below); AGPRs are a separate
    // file that does not affect the VGPR budget.
    if (!WantAGPR)
      ReqVGPRs = std::max(ReqVGPRs, RegNo + NumRegs);

    // Narrow the pinned value's register file to VGPR or AGPR (a class
    // narrowing, not a physreg pin, so it also works for loop-carried PHIs and
    // no-ops when the file is incompatible).
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
    // (e.g. one ds_read2 loads two pinned fragments into one wide reg). Pinning
    // it -- hard or soft -- would move overlapping physreg sub-slices and
    // miscompile. The shared reg is already in the right file (above), so the
    // pin is redundant: forward the source (sub)register to the uses and drop
    // it.
    if (Pin->getOperand(1).getSubReg()) {
      unsigned SubIdx = Pin->getOperand(1).getSubReg();
      for (MachineOperand &MO :
           llvm::make_early_inc_range(MRI.use_operands(Dst))) {
        MO.setSubReg(TRI->composeSubRegIndices(SubIdx, MO.getSubReg()));
        MO.setReg(Src);
      }
      Pin->eraseFromParent();
      continue;
    }

    bool Hard = EnableHardPin && PR && Src.isVirtual() && Dst.isVirtual();

    // Deterministic AGPR placement for a load tuple: when the pinned value is a
    // REG_SEQUENCE of (folded) AGPR loads, rewrite each element's def to a
    // fixed physical AGPR sub-register. Otherwise the MFMA A/B operands are AV
    // and the allocator moves them back to VGPR under low pressure
    // (non-deterministic).
    if (Hard && WantAGPR) {
      MachineInstr *RS = MRI.getVRegDef(Src);
      MachineBasicBlock *PinMBB = Pin->getParent();
      bool Ok = RS && RS->isRegSequence() && RS->getParent() == PinMBB;
      for (MCRegUnit U : TRI->regunits(PR))
        if (Ok && Claimed.contains(U))
          Ok = false;

      // Collect element (reg, subreg-index) pairs.
      SmallVector<std::pair<Register, unsigned>, 16> Elems;
      if (Ok)
        for (unsigned I = 1; I + 1 < RS->getNumOperands(); I += 2) {
          const MachineOperand &Reg = RS->getOperand(I);
          const MachineOperand &Sub = RS->getOperand(I + 1);
          if (!Reg.isReg() || !Reg.getReg().isVirtual() || Reg.getSubReg() ||
              !Sub.isImm() || !TRI->getSubReg(PR, Sub.getImm())) {
            Ok = false;
            break;
          }
          Elems.push_back({Reg.getReg(), (unsigned)Sub.getImm()});
        }

      // Every use of the pinned result and of each element must legally accept
      // the physical (sub)register and live in this block.
      auto LegalHere = [&](MachineOperand &MO, MCRegister T) {
        if (!T || MO.getParent()->getParent() != PinMBB)
          return false;
        const TargetRegisterClass *OpRC =
            MO.getParent()->getRegClassConstraint(MO.getOperandNo(), TII, TRI);
        return !OpRC || OpRC->contains(T);
      };
      if (Ok)
        for (MachineOperand &MO : MRI.reg_operands(Dst)) {
          if (MO.getParent() == Pin)
            continue;
          MCRegister T =
              MO.getSubReg() ? TRI->getSubReg(PR, MO.getSubReg()) : PR;
          if (!LegalHere(MO, T)) {
            Ok = false;
            break;
          }
        }
      if (Ok)
        for (auto [Elem, SubIdx] : Elems) {
          MCRegister PhysSub = TRI->getSubReg(PR, SubIdx);
          for (MachineOperand &MO : MRI.reg_operands(Elem))
            if (!LegalHere(MO, PhysSub)) {
              Ok = false;
              break;
            }
          if (!Ok)
            break;
        }

      if (Ok) {
        // Point each element's def/uses at its physical AGPR sub-register.
        for (auto [Elem, SubIdx] : Elems) {
          MCRegister PhysSub = TRI->getSubReg(PR, SubIdx);
          SmallVector<MachineOperand *, 4> Ops;
          for (MachineOperand &MO : MRI.reg_operands(Elem))
            Ops.push_back(&MO);
          for (MachineOperand *MO : Ops) {
            MO->setReg(PhysSub);
            MO->setSubReg(0);
            MO->setIsRenamable(false);
          }
        }
        // Point the pinned-result uses at the physical tuple.
        SmallVector<MachineOperand *, 16> Ops;
        for (MachineOperand &MO : MRI.reg_operands(Dst))
          if (MO.getParent() != Pin)
            Ops.push_back(&MO);
        for (MachineOperand *MO : Ops) {
          MCRegister T =
              MO->getSubReg() ? TRI->getSubReg(PR, MO->getSubReg()) : PR;
          MO->setReg(T);
          MO->setSubReg(0);
          MO->setIsRenamable(false);
        }
        for (MCRegUnit U : TRI->regunits(PR))
          Claimed.insert(U);
        RS->eraseFromParent();
        Pin->eraseFromParent();
        NeedRecomputeLiveIns = true;
        continue;
      }
    }

    // Grow the set of virtual registers that must share PR by following tie
    // edges (both ends of a tied operand pair must be the same register).
    SmallVector<Register, 8> Comp;
    if (Hard) {
      DenseSet<Register> Seen;
      auto Add = [&](Register R) {
        if (R.isVirtual() && Seen.insert(R).second)
          Comp.push_back(R);
      };
      Add(Src);
      Add(Dst);
      for (unsigned I = 0; I < Comp.size(); ++I) {
        Register R = Comp[I];
        for (MachineOperand &MO : MRI.reg_operands(R)) {
          MachineInstr *MI = MO.getParent();
          // Follow tie edges (both ends of a tie must share the register).
          if (MO.isTied())
            Add(MI->getOperand(MI->findTiedOperandIdx(MO.getOperandNo()))
                    .getReg());
          // Follow the MFMA accumulator edge (src2 <-> vdst). The VGPR (vgprcd)
          // MFMA form is 3-address, so an accumulation chain is connected by
          // src2->vdst def-use rather than ties; pin the whole chain as a unit.
          if (TII->isMAI(*MI)) {
            int Src2 = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                  AMDGPU::OpName::src2);
            if (Src2 >= 0) {
              const MachineOperand &V2 = MI->getOperand(Src2);
              const MachineOperand &VD = MI->getOperand(0);
              if ((unsigned)Src2 == MO.getOperandNo() && MI->getNumDefs() > 0 &&
                  VD.isReg())
                Add(VD.getReg()); // src2 -> vdst
              else if (MO.isDef() && V2.isReg())
                Add(V2.getReg()); // vdst -> src2
            }
          }
        }
      }
    }

    // Collect and validate every operand referencing a component register.
    SmallVector<MachineOperand *, 16> ToRewrite;
    if (Hard) {
      for (Register R : Comp) {
        for (MachineInstr &DefMI : MRI.def_instructions(R)) {
          if (DefMI.isPHI() || DefMI.isRegSequence() || DefMI.isImplicitDef()) {
            Hard = false;
            break;
          }
        }
        if (!Hard)
          break;
        for (MachineOperand &MO : MRI.reg_operands(R)) {
          if (MO.getParent() == Pin)
            continue; // the pin itself is erased
          MCRegister Tgt =
              MO.getSubReg() ? TRI->getSubReg(PR, MO.getSubReg()) : PR;
          if (!Tgt) {
            Hard = false;
            break;
          }
          const TargetRegisterClass *OpRC =
              MO.getParent()->getRegClassConstraint(MO.getOperandNo(), TII,
                                                    TRI);
          if (OpRC && !OpRC->contains(Tgt)) {
            Hard = false;
            break;
          }
          ToRewrite.push_back(&MO);
        }
        if (!Hard)
          break;
      }
    }

    // Track whether any operand crosses basic blocks; if so we must recompute
    // physreg live-ins after rewriting (done once at the end).
    if (Hard) {
      MachineBasicBlock *MBB = nullptr;
      for (MachineOperand *MO : ToRewrite) {
        MachineBasicBlock *B = MO->getParent()->getParent();
        if (!MBB)
          MBB = B;
        else if (B != MBB) {
          NeedRecomputeLiveIns = true;
          break;
        }
      }
    }

    // Conflict with an existing hard pin on overlapping regunits?
    if (Hard) {
      for (MCRegUnit U : TRI->regunits(PR)) {
        if (Claimed.contains(U)) {
          Hard = false;
          break;
        }
      }
    }

    // Partition operands. Non-tied *subregister uses* (e.g. the per-lane reads
    // a wide accumulator feeds into stores) are not rewritten to physical
    // subregisters -- that yields fragile physical-subreg live ranges. Instead
    // they read a virtual copy-out of the whole tuple.
    SmallVector<MachineOperand *, 16> DirectOps, SubUses;
    if (Hard) {
      for (MachineOperand *MO : ToRewrite) {
        if (MO->isUse() && MO->getSubReg() && !MO->isTied())
          SubUses.push_back(MO);
        else
          DirectOps.push_back(MO);
      }
    }

    // If there are subregister uses, insert one "%out = COPY PR" that dominates
    // them. Only handle the single-block case; otherwise fall back to soft.
    MachineBasicBlock *CopyMBB = nullptr;
    MachineBasicBlock::iterator CopyPt;
    if (Hard && !SubUses.empty()) {
      CopyMBB = SubUses.front()->getParent()->getParent();
      for (MachineOperand *MO : SubUses)
        if (MO->getParent()->getParent() != CopyMBB) {
          Hard = false;
          break;
        }
      if (Hard) {
        // Earliest sub-use in program order becomes the insertion point.
        DenseSet<MachineInstr *> SubMIs;
        for (MachineOperand *MO : SubUses)
          SubMIs.insert(MO->getParent());
        CopyPt = CopyMBB->end();
        for (MachineInstr &MI : *CopyMBB)
          if (SubMIs.contains(&MI)) {
            CopyPt = MI.getIterator();
            break;
          }
      }
    }

    if (Hard) {
      for (MachineOperand *MO : DirectOps) {
        MCRegister Tgt =
            MO->getSubReg() ? TRI->getSubReg(PR, MO->getSubReg()) : PR;
        MO->setReg(Tgt);
        MO->setSubReg(0);
        MO->setIsRenamable(false);
      }
      if (!SubUses.empty()) {
        Register Out = MRI.createVirtualRegister(RC);
        BuildMI(*CopyMBB, CopyPt, CopyPt->getDebugLoc(),
                TII->get(TargetOpcode::COPY), Out)
            .addReg(PR);
        for (MachineOperand *MO : SubUses)
          MO->setReg(Out); // keep the subregister index
      }
      for (MCRegUnit U : TRI->regunits(PR))
        Claimed.insert(U);
      Pin->eraseFromParent();
      continue;
    }

    // Soft fallback: COPY + register-allocation hint (a no-op hint if the
    // physical tuple was illegal).
    BuildMI(*Pin->getParent(), Pin, Pin->getDebugLoc(),
            TII->get(TargetOpcode::COPY), Dst)
        .addReg(Src);
    if (PR) {
      MRI.setSimpleHint(Dst, PR);
      if (Src.isVirtual())
        MRI.setSimpleHint(Src, PR);
    }
    Pin->eraseFromParent();
  }

  // Cross-BB hard pins introduce physical registers that are live across basic
  // block boundaries; recompute physreg live-in lists so the verifier and the
  // allocator see correct liveness.
  if (NeedRecomputeLiveIns) {
    SmallVector<MachineBasicBlock *, 16> MBBs;
    for (MachineBasicBlock &MBB : MF)
      MBBs.push_back(&MBB);
    fullyRecomputeLiveIns(MBBs);
  }

  // Cap occupancy so a wide VGPR-resident pinned value fits the per-wave budget
  // without the user setting __launch_bounds__. Only VGPR footprints drive
  // this: AGPRs are a separate file, so feeding an AGPR count into the VGPR
  // occupancy formula would wrongly raise occupancy and spill the VGPR
  // accumulator.
  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (unsigned Req = ReqVGPRs) {
    // Occupancy achievable while reserving `Req` registers per wave; cap the
    // waves-per-EU (and hence the RA's VGPR budget) so the pinned range fits.
    unsigned Occ = ST.getOccupancyWithNumVGPRs(Req);
    auto WPE = MFI->getWavesPerEU();
    unsigned NewMax = WPE.second ? std::min(WPE.second, Occ) : Occ;
    // Only cap the *max* occupancy; keep the min low (1 unless the function
    // already required more). Forcing min==max over-constrains the allocator
    // and breaks physreg liveness for hard-pinned loop-body tuples at low
    // occupancy.
    unsigned NewMin = std::min(WPE.first ? WPE.first : 1u, NewMax);
    MFI->setWavesPerEU(NewMin, NewMax);
    MFI->limitOccupancy(NewMax);
  }

  return true;
}
