//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

namespace {

static bool isImmConstant(const MachineOperand &Op, int64_t Val) {
  return Op.isImm() && Op.getImm() == Val;
}

class GCNPreRAOptimizationsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  MachineLoopInfo *MLI;

  bool processReg(Register Reg);

  bool isSingleUseVReg(Register Reg) const {
    return Reg.isVirtual() && MRI->hasOneUse(Reg);
  }

  bool isConstMove(MachineInstr &MI, int64_t C) const {
    return TII->isFoldableCopy(MI) && isImmConstant(MI.getOperand(1), C);
  }

  bool revertConditionalFMAPattern(MachineInstr &FMAInstr);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS, MachineLoopInfo *MLI)
      : LIS(LS), MLI(MLI) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {
    initializeGCNPreRAOptimizationsLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

FunctionPass *llvm::createGCNPreRAOptimizationsLegacyPass() {
  return new GCNPreRAOptimizationsLegacy();
}

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return GCNPreRAOptimizationsImpl(LIS, MLI).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  MachineLoopInfo *MLI = &MFAM.getResult<MachineLoopAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS, MLI).run(MF);
  return PreservedAnalyses::all();
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->MC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  if (ST.shouldUseConditionalSubToFMAF64()) {
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : make_early_inc_range(MBB)) {
        if (MI.getOpcode() == AMDGPU::V_FMAC_F64_e32)
          Changed |= revertConditionalFMAPattern(MI);
      }
    }
  }

  if (!ST.useRealTrue16Insts())
    return Changed;

  // Add RA hints to improve True16 COPY elimination.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AMDGPU::COPY)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      Register Src = MI.getOperand(1).getReg();
      if (Dst.isVirtual() &&
          MRI->getRegClass(Dst) == &AMDGPU::VGPR_16RegClass &&
          Src.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
      if (Src.isVirtual() &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
          Dst.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Dst) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
      if (!Dst.isVirtual() || !Src.isVirtual())
        continue;
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
        MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
      }
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_16RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
    }
  }

  return Changed;
}

/// Revert conditional subtraction to conditional FMA optimization happened
/// earlier in the selector. The reason is that the optimization uses more
/// instructions and registers to hold constants than original pattern and after
/// rematerializer it becomes clear if those constants are shared with other
/// code.
///
/// Detects a pattern where an FMA is used to conditionally subtract a value:
///   FMA(dst, cond ? -1.0 : 0.0, value, accum) -> accum - (cond ? value : 0)
///
/// Pattern detected:
///   v_mov_b32_e32 vNegOneHi, 0xbff00000   ; -1.0 high bits (single use)
///   v_mov_b32_e32 vMul.lo, 0                             ; (single use)
///   v_cndmask_b32_e64 vMul.hi, 0, vNegOneHi, vCondReg    ; (single use)
///   v_fmac_f64_e32 vDst[0:1], vMul[0:1], vValue[0:1], vAccum[0:1]
///
/// Transformed to (3 instructions instead of 4, lower register pressure):
///   v_cndmask_b32_e64 vCondValue.lo, 0, vValue.lo, vCondReg
///   v_cndmask_b32_e64 vCondValue.hi, 0, vValue.hi, vCondReg
///   v_add_f64_e64 vDst[0:1], vAccum[0:1], -vCondValue[0:1]
///
/// For loops: if both constants are initialized before the loop where the
/// v_fmac resides, we keep the original pattern. Ignoring case when v_fmac and
/// v_cndmask aren't in the same loop context as the selector doesn't generate
/// the pattern if v_cndmask is loop invariant.
bool GCNPreRAOptimizationsImpl::revertConditionalFMAPattern(
    MachineInstr &FMAInstr) {
  assert(FMAInstr.getOpcode() == AMDGPU::V_FMAC_F64_e32);

  MachineOperand *MulOp = TII->getNamedOperand(FMAInstr, AMDGPU::OpName::src0);
  assert(MulOp);
  if (!MulOp->isReg() || !isSingleUseVReg(MulOp->getReg()))
    return false;

  // Find subregister definitions for the 64-bit multiplicand register
  MachineInstr *MulLoDefMI = nullptr;
  MachineInstr *MulHiDefMI = nullptr;

  for (auto &DefMI : MRI->def_instructions(MulOp->getReg())) {
    if (DefMI.getOperand(0).getSubReg() == AMDGPU::sub0) {
      MulLoDefMI = &DefMI;
    } else if (DefMI.getOperand(0).getSubReg() == AMDGPU::sub1) {
      MulHiDefMI = &DefMI;
    }
  }

  if (!MulLoDefMI || !isConstMove(*MulLoDefMI, 0))
    return false;

  if (!MulHiDefMI || MulHiDefMI->getOpcode() != AMDGPU::V_CNDMASK_B32_e64)
    return false;

  MachineInstr *CndMaskMI = MulHiDefMI;
  MachineOperand *CndMaskFalseOp =
      TII->getNamedOperand(*CndMaskMI, AMDGPU::OpName::src0);
  assert(CndMaskFalseOp);
  if (!isImmConstant(*CndMaskFalseOp, 0))
    return false;

  MachineOperand *CndMaskTrueOp =
      TII->getNamedOperand(*CndMaskMI, AMDGPU::OpName::src1);
  assert(CndMaskTrueOp);
  if (!isSingleUseVReg(CndMaskTrueOp->getReg()))
    return false;

  // Check that the true operand is -1.0's high 32 bits (0xbff00000)
  MachineOperand *NegOneHiDef = MRI->getOneDef(CndMaskTrueOp->getReg());
  if (!NegOneHiDef ||
      !isConstMove(*NegOneHiDef->getParent(), -1074790400 /* 0xbff00000 */))
    return false;

  MachineInstr *NegOneHiMovMI = NegOneHiDef->getParent();

  if (MachineLoop *L = MLI->getLoopFor(FMAInstr.getParent())) {
    // The selector skips optimization if 'select' is loop invariant, so this is
    // more like an assert.
    if (MLI->getLoopFor(CndMaskMI->getParent()) != L)
      return false;

    // If both constants are initialized before the loop it's still beneficial
    // to keep the pattern.
    if (MLI->getLoopFor(NegOneHiMovMI->getParent()) != L &&
        MLI->getLoopFor(MulLoDefMI->getParent()) != L)
      return false;
  }

  // Perform the revert
  auto *DstOpnd = TII->getNamedOperand(FMAInstr, AMDGPU::OpName::vdst);
  auto *ValueOpnd = TII->getNamedOperand(FMAInstr, AMDGPU::OpName::src1);
  auto *AccumOpnd = TII->getNamedOperand(FMAInstr, AMDGPU::OpName::src2);
  auto *CondOpnd = TII->getNamedOperand(*CndMaskMI, AMDGPU::OpName::src2);
  assert(DstOpnd && ValueOpnd && AccumOpnd && CondOpnd);

  Register DstReg = DstOpnd->getReg();
  Register ValueReg = ValueOpnd->getReg();
  Register AccumReg = AccumOpnd->getReg();
  Register CondReg = CondOpnd->getReg();

  // Create a new 64-bit register for the conditional value
  Register CondValueReg =
      MRI->createVirtualRegister(MRI->getRegClass(ValueReg));

  MachineBasicBlock::iterator InsertPt = FMAInstr.getIterator();
  DebugLoc DL = FMAInstr.getDebugLoc();

  // Build: vCondValue.lo = condition ? vValue.lo : 0
  MachineBasicBlock *MBB = FMAInstr.getParent();
  MachineInstr *SelLo =
      BuildMI(*MBB, InsertPt, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64))
          .addReg(CondValueReg, RegState::DefineNoRead, AMDGPU::sub0)
          .addImm(0)                         // src0_modifiers
          .addImm(0)                         // src0 (false value = 0)
          .addImm(0)                         // src1_modifiers
          .addReg(ValueReg, 0, AMDGPU::sub0) // src1 (true value = vValue.lo)
          .addReg(CondReg)                   // condition
          .getInstr();

  // Build: vCondValue.hi = condition ? vValue.hi : 0
  MachineInstr *SelHi =
      BuildMI(*MBB, InsertPt, DL, TII->get(AMDGPU::V_CNDMASK_B32_e64))
          .addReg(CondValueReg, RegState::Define, AMDGPU::sub1)
          .addImm(0)                         // src0_modifiers
          .addImm(0)                         // src0 (false value = 0)
          .addImm(0)                         // src1_modifiers
          .addReg(ValueReg, 0, AMDGPU::sub1) // src1 (true value = vValue.hi)
          .addReg(CondReg)                   // condition
          .getInstr();

  // Build: vDst = vAccum - vCondValue (negation via src1_modifiers bit)
  MachineInstr *Sub =
      BuildMI(*MBB, InsertPt, DL, TII->get(AMDGPU::V_ADD_F64_e64))
          .addReg(DstReg, RegState::Define)
          .addImm(0)            // src0_modifiers
          .addReg(AccumReg)     // src0 (accumulator)
          .addImm(1)            // src1_modifiers (negation bit)
          .addReg(CondValueReg) // src1 (negated conditional value)
          .addImm(0)            // clamp
          .addImm(0)            // omod
          .getInstr();

  // Delete the old instructions
  for (MachineInstr *MI : {&FMAInstr, MulLoDefMI, CndMaskMI, NegOneHiMovMI}) {
    LIS->RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
  }

  LIS->InsertMachineInstrInMaps(*SelLo);
  LIS->InsertMachineInstrInMaps(*SelHi);
  LIS->InsertMachineInstrInMaps(*Sub);

  // Removed registers.
  LIS->removeInterval(MulOp->getReg());
  LIS->removeInterval(CndMaskTrueOp->getReg());

  // Reused registers.
  LIS->removeInterval(CondReg);
  LIS->createAndComputeVirtRegInterval(CondReg);

  LIS->removeInterval(DstReg);
  LIS->createAndComputeVirtRegInterval(DstReg);

  // Update AccumReg if it's different from DstReg.
  if (AccumReg != DstReg) {
    LIS->removeInterval(AccumReg);
    LIS->createAndComputeVirtRegInterval(AccumReg);
  }

  LIS->removeInterval(ValueReg);
  LIS->createAndComputeVirtRegInterval(ValueReg);

  // New register.
  LIS->createAndComputeVirtRegInterval(CondValueReg);

  return true;
}
