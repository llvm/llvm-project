//===-- SIPreColorPins.cpp - Hard register pinning ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Lowers the PIN_{VGPR,AGPR}_B* pseudos produced from
/// llvm.amdgcn.pin.{vgpr,agpr} into a hard register assignment ("pre-coloring").
///
/// The value being pinned is rewritten so that its def and all its uses
/// reference the requested physical VGPR/AGPR tuple directly. Because the value
/// is then a physical register in the MIR, the register allocator treats it as
/// fixed interference and can never place it elsewhere or let another value
/// clobber it -- unlike the soft allocation hint, this cannot be overridden by
/// competing coalescer copy-hints (e.g. an MFMA accumulator chain).
///
/// Tied operands (e.g. the in-place MFMA accumulator, whose vdst is tied to
/// src2) require care: both ends of a tie must share the same register. The
/// pass therefore rewrites the whole *tie-connected component* of virtual
/// registers, so a pin placed on the accumulator input also pins the tied
/// output. Subregister references are rewritten to the corresponding physical
/// subregister.
///
/// When hard pinning is not safe (a def in the component is a PHI, REG_SEQUENCE
/// or IMPLICIT_DEF, the physical (sub)register is not a legal member of some
/// rewritten operand's register class, or the tuple conflicts with an already
/// hard-pinned value) the pass falls back to the soft behaviour: a COPY plus a
/// register-allocation hint. This guarantees the pass never regresses
/// correctness.
///
/// Runs pre-RA while the function is still in SSA form (before PHIElimination /
/// TwoAddressInstruction), so each value has a single reaching def.
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

// Physical register tuple a pin targets, or 0 if it is not a legal member of the
// destination register class (e.g. a misaligned start on a target that requires
// aligned tuples).
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

  // Regunits already claimed by a hard pin. A later pin whose tuple overlaps any
  // claimed unit falls back to soft, so two distinct simultaneously-live values
  // can never be forced into the same physical register. (Legitimate reuse of a
  // register by a single value -- e.g. an accumulation chain -- is absorbed by
  // the tie-connected component of the first pin, after which the value is
  // already physical and later pins on it are no-ops.)
  DenseSet<MCRegUnit> Claimed;
  bool NeedRecomputeLiveIns = false;
  unsigned ReqVGPRs = 0, ReqAGPRs = 0; // highest register a pin needs, +1

  for (MachineInstr *Pin : Pins) {
    Register Dst = Pin->getOperand(0).getReg();
    Register Src = Pin->getOperand(1).getReg();
    unsigned RegNo = Pin->getOperand(2).getImm();
    const TargetRegisterClass *RC = MRI.getRegClass(Dst);
    MCRegister PR = getPinPhysReg(TRI, RC, RegNo);

    // Record how many registers this pin needs so the pin itself can drive the
    // occupancy target (the register budget must cover the pinned range).
    unsigned NumRegs = TRI->getRegSizeInBits(*RC) / 32;
    bool WantAGPR = TRI->isAGPRClass(RC);
    if (WantAGPR)
      ReqAGPRs = std::max(ReqAGPRs, RegNo + NumRegs);
    else
      ReqVGPRs = std::max(ReqVGPRs, RegNo + NumRegs);

    // Constrain the register *file* of the pinned value and every vreg reachable
    // through copies / REG_SEQUENCE / the MFMA accumulator edge to VGPR (for
    // pin_vgpr) or AGPR (for pin_agpr). This keeps a VGPR-pinned accumulator in
    // VGPRs even when its MFMA inputs are pinned to AGPRs (the MFMA then uses the
    // mixed v[D], a[A], a[B] form). Unlike a physreg pin this is just a class
    // narrowing, so it works for loop-carried PHI values too. constrainRegClass
    // is a no-op when the target file is incompatible (e.g. a VGPR load feeding
    // an AGPR-pinned input keeps its VGPR def and gets a copy).
    {
      DenseSet<Register> Seen;
      SmallVector<Register, 8> WL;
      SmallPtrSet<MachineInstr *, 8> AccMFMAs; // MFMAs whose vdst is pinned
      auto AddC = [&](Register R) {
        if (R.isVirtual() && Seen.insert(R).second)
          WL.push_back(R);
      };
      AddC(Src);
      AddC(Dst);
      for (unsigned I = 0; I < WL.size(); ++I) {
        for (MachineOperand &MO : MRI.reg_operands(WL[I])) {
          MachineInstr *MI = MO.getParent();
          if (MI->isCopy() || MI->isRegSequence()) {
            for (MachineOperand &O : MI->operands())
              if (O.isReg())
                AddC(O.getReg());
          }
          if (MO.isTied())
            AddC(MI->getOperand(MI->findTiedOperandIdx(MO.getOperandNo()))
                     .getReg());
          if (TII->isMAI(*MI)) {
            int S2 = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                                AMDGPU::OpName::src2);
            if (S2 >= 0) {
              if (MI->getOperand(0).isReg())
                AddC(MI->getOperand(0).getReg());
              if (MI->getOperand(S2).isReg())
                AddC(MI->getOperand(S2).getReg());
            }
            // vdst of this MFMA is in the pinned component.
            if (MO.isDef())
              AccMFMAs.insert(MI);
          }
        }
      }
      // Pinning an accumulator to VGPR while its MFMA inputs are in AGPR needs
      // the vgprcd MFMA form (VGPR dst/srcC, AGPR-or-VGPR srcA/B). ISel picks
      // the all-AGPR form because the function needs AGPRs; convert the reached
      // accumulator MFMAs to the vgprcd form, then re-derive the component's
      // register classes from the rewritten (VGPR-producing) defs. src0/src1
      // (the AGPR-pinned inputs) stay put -- vgprcd's AVSrc accepts them.
      bool Converted = false;
      if (!WantAGPR) {
        for (MachineInstr *MI : AccMFMAs) {
          int VOp = AMDGPU::getMFMASrcCVDstVGPROp(MI->getOpcode());
          if (VOp != -1) {
            MI->setDesc(TII->get(VOp));
            Converted = true;
          }
        }
      }
      for (Register R : WL) {
        // constrainRegClass cannot cross register files (AGPR<->VGPR are
        // disjoint); after an opcode conversion the class is re-derived instead.
        if (Converted)
          MRI.recomputeRegClass(R);
        unsigned Sz = TRI->getRegSizeInBits(*MRI.getRegClass(R));
        const TargetRegisterClass *Want =
            WantAGPR ? TRI->getAGPRClassForBitWidth(Sz)
                     : TRI->getVGPRClassForBitWidth(Sz);
        if (Want)
          MRI.constrainRegClass(R, Want);
      }
    }

    bool Hard = EnableHardPin && PR && Src.isVirtual() && Dst.isVirtual();

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
            int Src2 =
                AMDGPU::getNamedOperandIdx(MI->getOpcode(), AMDGPU::OpName::src2);
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
          const TargetRegisterClass *OpRC = MO.getParent()->getRegClassConstraint(
              MO.getOperandNo(), TII, TRI);
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

    // Partition operands. Non-tied *subregister uses* (e.g. the per-lane reads a
    // wide accumulator feeds into stores) are not rewritten to physical
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

  // Let the pins drive occupancy: the register budget must be large enough to
  // hold every pinned register, so cap the occupancy accordingly. This lets a
  // wide pinned accumulator (e.g. 192 VGPRs) force occupancy down without the
  // user having to set __launch_bounds__ / amdgpu-waves-per-eu by hand.
  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  unsigned Req = std::max(ReqVGPRs, ReqAGPRs);
  if (Req) {
    // Occupancy achievable while reserving `Req` registers per wave; cap the
    // waves-per-EU (and hence the RA's VGPR budget) so the pinned range fits.
    unsigned Occ = ST.getOccupancyWithNumVGPRs(Req);
    auto WPE = MFI->getWavesPerEU();
    unsigned NewMax = WPE.second ? std::min(WPE.second, Occ) : Occ;
    unsigned NewMin = std::min(WPE.first ? WPE.first : NewMax, NewMax);
    MFI->setWavesPerEU(NewMin, NewMax);
    MFI->limitOccupancy(NewMax);
  }

  return true;
}
