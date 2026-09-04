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
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-pre-color-pins"

STATISTIC(NumHardPins, "Number of values pre-colored to the requested register");
STATISTIC(NumSoftPins, "Number of pins degraded to a soft allocation hint");
STATISTIC(NumNoOpPins, "Number of pins dropped as a no-op");

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

// Rewriting a pinned value's operands can leave a REG_SEQUENCE that does
// nothing but reassemble a run of the pinned tuple back into a virtual
// register -- the two halves a 32-byte store reads, say. The allocator
// materializes that as one copy per lane, undoing the placement. If the run
// names a physical tuple outright, forward it to the uses and drop the
// REG_SEQUENCE. Returns true if it was folded away.
static bool foldPhysRegSequence(const SIInstrInfo *TII,
                                const SIRegisterInfo *TRI,
                                MachineRegisterInfo &MRI, MachineInstr &RS) {
  Register Def = RS.getOperand(0).getReg();
  if (!Def.isVirtual() || RS.getNumOperands() < 3)
    return false;

  MCRegister Tuple;
  for (unsigned I = 1; I + 1 < RS.getNumOperands(); I += 2) {
    const MachineOperand &Src = RS.getOperand(I);
    const MachineOperand &Sub = RS.getOperand(I + 1);
    if (!Src.isReg() || !Src.getReg().isPhysical() || Src.getSubReg() ||
        !Sub.isImm())
      return false;
    MCRegister Phys = Src.getReg().asMCReg();
    // The first lane fixes the candidate tuple; the rest must agree with it,
    // which rejects a permuted, gapped or misaligned run.
    if (!Tuple)
      Tuple =
          TRI->getMatchingSuperReg(Phys, Sub.getImm(), MRI.getRegClass(Def));
    else if (TRI->getSubReg(Tuple, Sub.getImm()) != Phys)
      return false;
    if (!Tuple)
      return false;
  }

  SmallVector<MachineOperand *, 8> Uses;
  for (MachineOperand &MO : MRI.reg_operands(Def)) {
    if (MO.getParent() == &RS)
      continue;
    MCRegister T = MO.getSubReg() ? TRI->getSubReg(Tuple, MO.getSubReg())
                                  : MCRegister(Tuple);
    if (MO.isDef() || !T)
      return false;
    const TargetRegisterClass *OpRC =
        MO.getParent()->getRegClassConstraint(MO.getOperandNo(), TII, TRI);
    if (OpRC && !OpRC->contains(T))
      return false;
    Uses.push_back(&MO);
  }

  for (MachineOperand *MO : Uses) {
    MO->setReg(MO->getSubReg() ? TRI->getSubReg(Tuple, MO->getSubReg())
                               : MCRegister(Tuple));
    MO->setSubReg(0);
    MO->setIsRenamable(false);
  }
  RS.eraseFromParent();
  return true;
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
  // unit may only take the tuple over once the earlier occupant is dead there
  // (see canTakeOverClaim); otherwise it falls back to soft, so two live values
  // never share a physreg. Reuse by a single value (e.g. an accumulation chain)
  // is instead absorbed into the first pin's tie-connected component, making
  // later pins on it no-ops.
  DenseSet<MCRegUnit> Claimed;
  bool NeedRecomputeLiveIns = false;
  bool AnyHardPin = false;
  unsigned ReqVGPRs =
      0; // highest VGPR a pin needs, +1 (drives the occupancy cap)

  // Whether a new value, defined by `Defs` as (instruction, physical target)
  // pairs covering `PR`, can take `PR` over from whatever holds it now.
  //
  // Clang wraps every store to a pinned variable in its own pin, so
  // `x = load; x = f(x);` arrives as two pins on one tuple. Refusing the second
  // would leave the variable's later value wherever the allocator likes, which
  // defeats the point of pinning it. The takeover is safe exactly when no lane
  // of PR is touched after the instruction that rewrites it: judge that per
  // lane, since each half of a tuple can be updated at a different point, and
  // within one block, so instruction order is a plain index comparison. A lane
  // may be rewritten by the very instruction that last reads it -- an in-place
  // update reads its sources before writing its result.
  auto canTakeOverClaim =
      [&](MCRegister PR, MachineBasicBlock *MBB,
          ArrayRef<std::pair<MachineInstr *, MCRegister>> Defs) {
        DenseMap<MachineInstr *, unsigned> Order;
        for (MachineInstr &MI : *MBB)
          Order.insert({&MI, Order.size()});

        DenseMap<MCRegUnit, unsigned> DefAt;
        for (auto [DefMI, Phys] : Defs) {
          if (!DefMI || DefMI->getParent() != MBB)
            return false;
          for (MCRegUnit U : TRI->regunits(Phys)) {
            auto [It, New] = DefAt.try_emplace(U, Order[DefMI]);
            if (!New)
              It->second = std::min(It->second, Order[DefMI]);
          }
        }
        // A lane the new value never writes would keep the old one alive under
        // it, with no def to order the accesses against.
        for (MCRegUnit U : TRI->regunits(PR))
          if (!DefAt.contains(U))
            return false;

        for (MachineBasicBlock &B : MF)
          for (MachineInstr &MI : B)
            for (const MachineOperand &MO : MI.operands()) {
              if (MO.isRegMask() && MO.clobbersPhysReg(PR))
                return false;
              if (!MO.isReg() || !MO.getReg().isPhysical())
                continue;
              for (MCRegUnit U : TRI->regunits(MO.getReg().asMCReg())) {
                auto It = DefAt.find(U);
                if (It == DefAt.end())
                  continue;
                if (&B != MBB)
                  return false;
                unsigned At = Order[&MI];
                if (At > It->second || (At == It->second && MO.isDef()))
                  return false;
              }
            }
        return true;
      };

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
      ++NumNoOpPins;
      continue;
    }
    // Only a VGPR pin that is actually honored drives the occupancy cap (see
    // below), so this is recorded at the pre-coloring sites rather than here:
    // a soft hint is free to go unused, and paying occupancy for a hint the
    // allocator then ignores costs waves for nothing. AGPRs are a separate
    // file that does not affect the VGPR budget.
    auto RecordVGPRFootprint = [&] {
      if (!WantAGPR)
        ReqVGPRs = std::max(ReqVGPRs, RegNo + NumRegs);
    };

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
    // Why a pin could not be pre-colored, for -debug-only=si-pre-color-pins.
    const char *SoftWhy = Hard ? "?" : "disabled or non-virtual";

    // Deterministic placement for a load tuple: when the pinned value is a
    // REG_SEQUENCE of loads, rewrite each element's def to a fixed physical
    // sub-register. For an AGPR pin this stops the allocator moving MFMA A/B
    // operands back to VGPR under low pressure (they are AV-classed, so the
    // placement is otherwise non-deterministic). For a VGPR pin it is the only
    // way to place a value the hardware builds from more than one load, such as
    // a WMMA B operand that two ds_reads assemble out of LDS -- the general path
    // below rejects any REG_SEQUENCE def.
    if (Hard) {
      MachineInstr *RS = MRI.getVRegDef(Src);
      MachineBasicBlock *PinMBB = Pin->getParent();
      bool Ok = RS && RS->isRegSequence() && RS->getParent() == PinMBB;
      // A scaled MFMA (mfma_scale_*, f8f6f4) consuming a wide AGPR tuple hits a
      // machine-scheduler liveness error under the direct physical rewrite; leave
      // those to the soft path (which still places the inputs in AGPRs). Walk the
      // pinned value's uses (through copy/reg_sequence/subreg ops) for one.
      if (Ok) {
        SmallVector<Register, 8> WL{Dst};
        DenseSet<Register> WSeen{Dst};
        for (unsigned I = 0; I < WL.size() && Ok; ++I)
          for (MachineInstr &U : MRI.use_nodbg_instructions(WL[I])) {
            if (TII->getName(U.getOpcode()).contains("F8F6F4")) {
              Ok = false;
              break;
            }
            // A VGPR pin reaches shapes the AGPR path never sees, because the
            // AGPR path only ever pins MFMA A/B inputs. A tied (two-address)
            // use is a WMMA/MFMA accumulator, and TwoAddressInstruction
            // requires both ends of a tie to be virtual; a PHI operand must
            // stay virtual as well, since LiveVariables walks PHI sources with
            // getVarInfo(). Leave both to the general path.
            if (!WantAGPR) {
              if (U.isPHI()) {
                Ok = false;
                break;
              }
              for (const MachineOperand &O : U.operands())
                if (O.isReg() && O.isUse() && O.isTied() && O.getReg() == WL[I])
                  Ok = false;
              if (!Ok)
                break;
            }
            if (U.isCopy() || U.isRegSequence() || U.isPHI() ||
                U.getOpcode() == TargetOpcode::INSERT_SUBREG ||
                U.getOpcode() == TargetOpcode::EXTRACT_SUBREG)
              for (const MachineOperand &D : U.defs())
                if (D.getReg().isVirtual() && WSeen.insert(D.getReg()).second)
                  WL.push_back(D.getReg());
          }
      }
      // Map each element's defining register onto a physical (sub)register of
      // PR. An element either covers its REG_SEQUENCE slot outright, or is a
      // subregister slice of a wider def: a 32-byte load, for instance, is
      // selected as two dwordx4 loads whose lanes reach the REG_SEQUENCE as
      // %wide.subN. Retargeting such a lane on its own would leave the rest of
      // the wider def behind, so the whole def is placed instead --
      // getMatchingSuperReg derives the tuple it must occupy and rejects a
      // permuted or misaligned layout, and the tuple has to stay inside PR.
      SmallVector<std::pair<Register, MCRegister>, 16> Elems;
      if (Ok)
        for (unsigned I = 1; I + 1 < RS->getNumOperands(); I += 2) {
          const MachineOperand &Reg = RS->getOperand(I);
          const MachineOperand &Sub = RS->getOperand(I + 1);
          if (!Reg.isReg() || !Reg.getReg().isVirtual() || !Sub.isImm()) {
            Ok = false;
            break;
          }
          MCRegister Tgt = TRI->getSubReg(PR, Sub.getImm());
          if (Tgt && Reg.getSubReg())
            Tgt = TRI->getMatchingSuperReg(Tgt, Reg.getSubReg(),
                                           MRI.getRegClass(Reg.getReg()));
          if (!Tgt || !TRI->isSubRegisterEq(PR, Tgt)) {
            Ok = false;
            break;
          }
          auto *Prev =
              find_if(Elems, [&](const std::pair<Register, MCRegister> &E) {
                return E.first == Reg.getReg();
              });
          if (Prev == Elems.end())
            Elems.emplace_back(Reg.getReg(), Tgt);
          else if (Prev->second != Tgt) {
            Ok = false;
            break;
          }
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
        for (auto [Elem, Phys] : Elems) {
          for (MachineOperand &MO : MRI.reg_operands(Elem)) {
            MCRegister T =
                MO.getSubReg() ? TRI->getSubReg(Phys, MO.getSubReg()) : Phys;
            if (!LegalHere(MO, T)) {
              Ok = false;
              break;
            }
          }
          if (!Ok)
            break;
        }

      // Each element's def is where its share of the tuple is written, which
      // is what decides whether an earlier occupant can be displaced.
      if (Ok && any_of(TRI->regunits(PR),
                       [&](MCRegUnit U) { return Claimed.contains(U); })) {
        SmallVector<std::pair<MachineInstr *, MCRegister>, 16> Defs;
        for (auto [Elem, Phys] : Elems)
          Defs.emplace_back(MRI.getVRegDef(Elem), Phys);
        Ok = canTakeOverClaim(PR, PinMBB, Defs);
      }

      if (Ok) {
        // Point each element's def/uses at its physical (sub)register.
        for (auto [Elem, Phys] : Elems) {
          SmallVector<MachineOperand *, 4> Ops;
          for (MachineOperand &MO : MRI.reg_operands(Elem))
            Ops.push_back(&MO);
          for (MachineOperand *MO : Ops) {
            MCRegister T =
                MO->getSubReg() ? TRI->getSubReg(Phys, MO->getSubReg()) : Phys;
            MO->setReg(T);
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
        RecordVGPRFootprint();
        AnyHardPin = true;
        ++NumHardPins;
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

    // Conflict with an existing hard pin on overlapping regunits? Checked
    // before the operand walk below, so an occupied tuple is reported as such
    // rather than as whatever else the value's shape happens to trip over.
    if (Hard && any_of(TRI->regunits(PR),
                       [&](MCRegUnit U) { return Claimed.contains(U); })) {
      SmallVector<std::pair<MachineInstr *, MCRegister>, 8> Defs;
      for (Register R : Comp)
        for (MachineInstr &D : MRI.def_instructions(R)) {
          if (&D == Pin)
            continue; // erased below; Dst is written by the component's def
          for (const MachineOperand &MO : D.defs())
            if (MO.isReg() && MO.getReg() == R)
              Defs.emplace_back(
                  &D, MO.getSubReg() ? TRI->getSubReg(PR, MO.getSubReg()) : PR);
        }
      if (!canTakeOverClaim(PR, Pin->getParent(), Defs)) {
        Hard = false;
        SoftWhy = "overlaps a live earlier hard pin";
      }
    }

    // Collect and validate every operand referencing a component register.
    SmallVector<MachineOperand *, 16> ToRewrite;
    if (Hard) {
      for (Register R : Comp) {
        for (MachineInstr &DefMI : MRI.def_instructions(R)) {
          if (DefMI.isPHI() || DefMI.isRegSequence() || DefMI.isImplicitDef()) {
            Hard = false;
            SoftWhy = DefMI.isPHI()          ? "PHI def"
                      : DefMI.isRegSequence() ? "REG_SEQUENCE def"
                                              : "IMPLICIT_DEF";
            break;
          }
        }
        if (!Hard)
          break;
        for (MachineOperand &MO : MRI.reg_operands(R)) {
          if (MO.getParent() == Pin)
            continue; // the pin itself is erased
          // A PHI operand must stay virtual: LiveVariables walks PHI sources
          // through getVarInfo(), which only accepts virtual registers, so a
          // physreg there crashes before PHIElimination can lower it. This is
          // reached when a pinned value defined in one block flows into a
          // loop-carried PHI (several pinned accumulators initialised outside
          // the loop). Leave the whole pin to the soft path.
          if (MO.getParent()->isPHI()) {
            Hard = false;
            SoftWhy = "PHI use";
            break;
          }
          MCRegister Tgt =
              MO.getSubReg() ? TRI->getSubReg(PR, MO.getSubReg()) : PR;
          if (!Tgt) {
            Hard = false;
            SoftWhy = "no such subregister";
            break;
          }
          const TargetRegisterClass *OpRC =
              MO.getParent()->getRegClassConstraint(MO.getOperandNo(), TII,
                                                    TRI);
          if (OpRC && !OpRC->contains(Tgt)) {
            Hard = false;
            SoftWhy = "operand class rejects the physreg";
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
          SoftWhy = "subregister uses span blocks";
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
      RecordVGPRFootprint();
      AnyHardPin = true;
      ++NumHardPins;
      continue;
    }

    ++NumSoftPins;
    LLVM_DEBUG(dbgs() << "pin to " << (WantAGPR ? 'a' : 'v') << RegNo
                      << " not pre-colored: " << SoftWhy << '\n');
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

  // Clean up the REG_SEQUENCEs the rewrite left behind. Folding one can expose
  // another (a wide tuple reassembled in stages), so iterate to a fixpoint.
  for (bool Folded = AnyHardPin; Folded;) {
    Folded = false;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB))
        if (MI.isRegSequence() && foldPhysRegSequence(TII, TRI, MRI, MI)) {
          Folded = true;
          NeedRecomputeLiveIns = true;
        }
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
  // without the user setting __launch_bounds__. Only the VGPR footprint of a
  // pre-colored pin drives this: AGPRs are a separate file, so feeding an AGPR
  // count into the VGPR occupancy formula would wrongly raise occupancy and
  // spill the VGPR accumulator.
  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  if (unsigned Req = ReqVGPRs) {
    // Occupancy achievable while reserving `Req` registers per wave; cap the
    // waves-per-EU (and hence the RA's VGPR budget) so the pinned range fits.
    unsigned Occ =
        ST.getOccupancyWithNumVGPRs(Req, MFI->getDynamicVGPRBlockSize());
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
