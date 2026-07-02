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

// When an MFMA input is pinned to AGPR, either force the accumulator into VGPR
// via the mixed vgprcd form (option 1: v[C], a[A], a[B]) or leave the
// hardware-native all-AGPR form untouched (option 2: a[D], a[A], a[B], a[C]).
static cl::opt<bool> PinAgprVgprC(
    "amdgpu-pin-agpr-vgpr-c", cl::init(true), cl::Hidden,
    cl::desc("For an AGPR-pinned MFMA input, convert the consuming MFMA to the "
             "vgprcd form so its accumulator stays in VGPR (else keep the "
             "native all-AGPR form)"));

// Extra VGPRs (beyond the pinned accumulator's own footprint) the occupancy cap
// must reserve for addressing / load temporaries so the accumulator stays
// resident. Chosen so both a 64-VGPR (128x128) and a 96-VGPR (192x128) tile stay
// spill-free without __launch_bounds__.
// Experimental: when >0, an AGPR-input pin caps occupancy so the vgprcd-pinned
// accumulator (plus this many VGPRs of headroom) stays resident, avoiding
// __launch_bounds__. Default 0 (off): auto-driving occupancy from this pass
// currently perturbs the hard-pinned physreg live ranges and can produce invalid
// MIR at low occupancy -- use __launch_bounds__ to control occupancy instead.
static cl::opt<unsigned> PinAccVGPRMargin(
    "amdgpu-pin-acc-vgpr-margin", cl::init(0), cl::Hidden,
    cl::desc("If nonzero, VGPRs reserved on top of a vgprcd-pinned accumulator "
             "so an AGPR-input pin can drive occupancy (experimental)"));

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
  // Accumulator tiles moved to VGPR by the vgprcd conversion (A,B->AGPR pins).
  // Their total VGPR footprint drives occupancy: moving A/B out of the VGPR file
  // lets the compiler raise occupancy, shrinking the per-wave VGPR budget until
  // the (now VGPR) accumulator no longer fits and spills/rotates through AGPRs.
  // Capping occupancy so the accumulator stays resident avoids that.
  DenseSet<Register> AccTiles;

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
      // Gather the copy/REG_SEQUENCE/tie-connected component of `Seeds` and
      // constrain every member to the requested register file. MFMA
      // src2<->vdst accumulator edges are followed only when `FollowAcc` is set.
      // Otherwise an MFMA that *uses* a component register as src0/src1 is
      // recorded in `Inputs` and treated as a leaf, so pinning an input to AGPR
      // does not drag the (large, loop-carried) accumulator into the AGPR file.
      // `Recompute` re-derives each class from its defs first -- needed after an
      // opcode conversion, since constrainRegClass cannot cross the disjoint
      // AGPR/VGPR files.
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
            // Copy / REG_SEQUENCE / PHI all just move the value between vregs;
            // pull every register operand into the component. PHI matters for
            // the loop-carried accumulator: without it the carried value stays
            // in its original file (AGPR) while the vgprcd MFMA computes in VGPR,
            // forcing an agpr<->vgpr copy every iteration.
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
          // A constant accumulator init (e.g. clear()==0) materialized in an
          // AGPR via V_ACCVGPR_WRITE cannot be constrained to VGPR (its dst is
          // AGPR-only), so it would stay in AGPR and be copied into the VGPR
          // accumulator every kernel launch (write-0-to-agpr then read-to-vgpr).
          // When routing the accumulator to VGPR, rewrite such an init to a
          // plain VGPR V_MOV so the constant is born in VGPR (no agpr<->vgpr copy).
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

      // An AGPR-pinned MFMA input needs the mixed vgprcd form (VGPR dst/srcC,
      // AGPR-or-VGPR srcA/B) so the accumulator can stay in VGPR. ISel picks the
      // all-AGPR form because the function needs AGPRs; convert each consuming
      // MFMA to vgprcd, then constrain its accumulator (vdst/srcC chain) to VGPR
      // -- re-deriving classes from the converted, VGPR-producing defs. This
      // keeps the whole accumulation chain in VGPR without pinning it, so it
      // stays coalesced (no chunked pins, no agpr<->vgpr shuffle).
      if (WantAGPR && PinAgprVgprC && !InputMFMAs.empty()) {
        SmallVector<Register, 8> AccSeeds;
        for (MachineInstr *MI : InputMFMAs) {
          int VOp = AMDGPU::getMFMASrcCVDstVGPROp(MI->getOpcode());
          if (VOp == -1)
            continue; // already vgprcd form
          MI->setDesc(TII->get(VOp));
          if (MI->getOperand(0).isReg()) {
            AccSeeds.push_back(MI->getOperand(0).getReg());
            // Each converted MFMA's vdst is one accumulator tile now living in
            // VGPR; track distinct tiles for the occupancy cap below.
            AccTiles.insert(MI->getOperand(0).getReg());
          }
          int S2 = AMDGPU::getNamedOperandIdx(MI->getOpcode(),
                                              AMDGPU::OpName::src2);
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

    // When the pinned source is a *subregister* of a larger value, that register
    // is shared -- e.g. a combined ds_read2 loads two pinned fragments into one
    // wide register, each pin taking a sub-slice. Neither a hard pin (rewriting
    // the whole wide reg to one narrow physreg) nor a soft COPY+hint is safe: the
    // soft copies read/write overlapping physreg sub-slices and the allocator
    // clobbers one before the other is read (miscompile). But the shared load was
    // already class-constrained to the requested file above (and tryFoldLoad put
    // it in AGPR), so the pin is redundant -- make it a no-op: replace uses of the
    // pin result with the source (sub)register directly and erase the pin.
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

    // Deterministic AGPR placement for a load tuple. When an AGPR pin's value is
    // a REG_SEQUENCE of (folded) AGPR loads, rewrite each element's def to a
    // fixed physical AGPR sub-register so A/B are *born* in fixed AGPRs. Without
    // this the MFMA A/B operands are AV (agpr-or-vgpr) and the coalescer /
    // allocator moves them back to VGPR whenever pressure is low, making the pin
    // non-deterministic. The accumulator was already routed to VGPR (vgprcd) by
    // the file-constraint step above, so this yields v[D], a[A], a[B] with the
    // accumulator free to occupy the whole VGPR file.
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
          MCRegister T = MO.getSubReg() ? TRI->getSubReg(PR, MO.getSubReg()) : PR;
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
          MCRegister T = MO->getSubReg() ? TRI->getSubReg(PR, MO->getSubReg()) : PR;
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

  // Let a *VGPR* pin drive occupancy: a wide pinned VGPR value (e.g. a 192-VGPR
  // accumulator) must fit the per-wave VGPR budget, so cap occupancy to make
  // room without the user setting __launch_bounds__ / amdgpu-waves-per-eu.
  // AGPR pins must NOT drive this: AGPRs are a separate file, and feeding an
  // AGPR count into the VGPR occupancy formula wrongly raises occupancy and
  // shrinks the VGPR budget, spilling the (VGPR) accumulator into AGPRs.
  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  // Total VGPR footprint of the accumulator tiles routed to VGPR, plus a margin
  // for addressing/temps. When A/B are pinned to AGPR the accumulator must stay
  // VGPR-resident; this caps occupancy so its budget is large enough.
  unsigned AccVGPRs = 0;
  if (PinAccVGPRMargin) {
    for (Register R : AccTiles)
      if (R.isVirtual())
        AccVGPRs += TRI->getRegSizeInBits(*MRI.getRegClass(R)) / 32;
    if (AccVGPRs)
      AccVGPRs += PinAccVGPRMargin;
  }
  unsigned Req = std::max(ReqVGPRs, AccVGPRs);
  if (Req) {
    // Occupancy achievable while reserving `Req` registers per wave; cap the
    // waves-per-EU (and hence the RA's VGPR budget) so the pinned range fits.
    unsigned Occ = ST.getOccupancyWithNumVGPRs(Req);
    auto WPE = MFI->getWavesPerEU();
    unsigned NewMax = WPE.second ? std::min(WPE.second, Occ) : Occ;
    // Only cap the *max* occupancy; keep the min low (1 unless the function
    // already required more). Forcing min==max over-constrains the allocator and
    // breaks physreg liveness for hard-pinned loop-body tuples at low occupancy.
    unsigned NewMin = std::min(WPE.first ? WPE.first : 1u, NewMax);
    MFI->setWavesPerEU(NewMin, NewMax);
    MFI->limitOccupancy(NewMax);
  }

  return true;
}
