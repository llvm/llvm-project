//===-------------- RISCVVLOptimizer.cpp - VL Optimizer -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass reduces the VL where possible at the MI level, before VSETVLI
// instructions are inserted.
//
// The purpose of this optimization is to make the VL argument, for instructions
// that have a VL argument, as small as possible. This is implemented by
// visiting each instruction in reverse order and checking that if it has a VL
// argument, whether the VL can be reduced.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-vl-optimizer"
#define PASS_NAME "RISC-V VL Optimizer"

namespace {

class RISCVVLOptimizer : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
  const MachineDominatorTree *MDT;

public:
  static char ID;

  RISCVVLOptimizer() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return PASS_NAME; }

private:
  bool checkUsers(const MachineOperand *&CommonVL, MachineInstr &MI);
  bool tryReduceVL(MachineInstr &MI);
  bool isCandidate(const MachineInstr &MI) const;
};

} // end anonymous namespace

char RISCVVLOptimizer::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVVLOptimizer, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVVLOptimizer, DEBUG_TYPE, PASS_NAME, false, false)

FunctionPass *llvm::createRISCVVLOptimizerPass() {
  return new RISCVVLOptimizer();
}

/// Return true if R is a physical or virtual vector register, false otherwise.
static bool isVectorRegClass(Register R, const MachineRegisterInfo *MRI) {
  if (R.isPhysical())
    return RISCV::VRRegClass.contains(R);
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  return RISCVRI::isVRegClass(RC->TSFlags);
}

/// Represents the EMUL and EEW of a MachineOperand.
struct OperandInfo {
  enum class State {
    Unknown,
    Known,
  } S;

  // Represent as 1,2,4,8, ... and fractional indicator. This is because
  // EMUL can take on values that don't map to RISCVII::VLMUL values exactly.
  // For example, a mask operand can have an EMUL less than MF8.
  std::optional<std::pair<unsigned, bool>> EMUL;

  unsigned Log2EEW;

  OperandInfo(RISCVII::VLMUL EMUL, unsigned Log2EEW)
      : S(State::Known), EMUL(RISCVVType::decodeVLMUL(EMUL)), Log2EEW(Log2EEW) {
  }

  OperandInfo(std::pair<unsigned, bool> EMUL, unsigned Log2EEW)
      : S(State::Known), EMUL(EMUL), Log2EEW(Log2EEW) {}

  OperandInfo() : S(State::Unknown) {}

  bool isUnknown() const { return S == State::Unknown; }
  bool isKnown() const { return S == State::Known; }

  static bool EMULAndEEWAreEqual(const OperandInfo &A, const OperandInfo &B) {
    assert(A.isKnown() && B.isKnown() && "Both operands must be known");

    return A.Log2EEW == B.Log2EEW && A.EMUL->first == B.EMUL->first &&
           A.EMUL->second == B.EMUL->second;
  }

  void print(raw_ostream &OS) const {
    if (isUnknown()) {
      OS << "Unknown";
      return;
    }
    assert(EMUL && "Expected EMUL to have value");
    OS << "EMUL: m";
    if (EMUL->second)
      OS << "f";
    OS << EMUL->first;
    OS << ", EEW: " << (1 << Log2EEW);
  }
};

LLVM_ATTRIBUTE_UNUSED
static raw_ostream &operator<<(raw_ostream &OS, const OperandInfo &OI) {
  OI.print(OS);
  return OS;
}

namespace llvm {
namespace RISCVVType {
/// Return the RISCVII::VLMUL that is two times VLMul.
/// Precondition: VLMul is not LMUL_RESERVED or LMUL_8.
static RISCVII::VLMUL twoTimesVLMUL(RISCVII::VLMUL VLMul) {
  switch (VLMul) {
  case RISCVII::VLMUL::LMUL_F8:
    return RISCVII::VLMUL::LMUL_F4;
  case RISCVII::VLMUL::LMUL_F4:
    return RISCVII::VLMUL::LMUL_F2;
  case RISCVII::VLMUL::LMUL_F2:
    return RISCVII::VLMUL::LMUL_1;
  case RISCVII::VLMUL::LMUL_1:
    return RISCVII::VLMUL::LMUL_2;
  case RISCVII::VLMUL::LMUL_2:
    return RISCVII::VLMUL::LMUL_4;
  case RISCVII::VLMUL::LMUL_4:
    return RISCVII::VLMUL::LMUL_8;
  case RISCVII::VLMUL::LMUL_8:
  default:
    llvm_unreachable("Could not multiply VLMul by 2");
  }
}

/// Return EMUL = (EEW / SEW) * LMUL where EEW comes from Log2EEW and LMUL and
/// SEW are from the TSFlags of MI.
static std::pair<unsigned, bool>
getEMULEqualsEEWDivSEWTimesLMUL(unsigned Log2EEW, const MachineInstr &MI) {
  RISCVII::VLMUL MIVLMUL = RISCVII::getLMul(MI.getDesc().TSFlags);
  auto [MILMUL, MILMULIsFractional] = RISCVVType::decodeVLMUL(MIVLMUL);
  unsigned MILog2SEW =
      MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();

  // Mask instructions will have 0 as the SEW operand. But the LMUL of these
  // instructions is calculated is as if the SEW operand was 3 (e8).
  if (MILog2SEW == 0)
    MILog2SEW = 3;

  unsigned MISEW = 1 << MILog2SEW;

  unsigned EEW = 1 << Log2EEW;
  // Calculate (EEW/SEW)*LMUL preserving fractions less than 1. Use GCD
  // to put fraction in simplest form.
  unsigned Num = EEW, Denom = MISEW;
  int GCD = MILMULIsFractional ? std::gcd(Num, Denom * MILMUL)
                               : std::gcd(Num * MILMUL, Denom);
  Num = MILMULIsFractional ? Num / GCD : Num * MILMUL / GCD;
  Denom = MILMULIsFractional ? Denom * MILMUL / GCD : Denom / GCD;
  return std::make_pair(Num > Denom ? Num : Denom, Denom > Num);
}
} // end namespace RISCVVType
} // end namespace llvm

/// Dest has EEW=SEW and EMUL=LMUL. Source EEW=SEW/Factor (i.e. F2 => EEW/2).
/// Source has EMUL=(EEW/SEW)*LMUL. LMUL and SEW comes from TSFlags of MI.
static OperandInfo getIntegerExtensionOperandInfo(unsigned Factor,
                                                  const MachineInstr &MI,
                                                  const MachineOperand &MO) {
  RISCVII::VLMUL MIVLMul = RISCVII::getLMul(MI.getDesc().TSFlags);
  unsigned MILog2SEW =
      MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();

  if (MO.getOperandNo() == 0)
    return OperandInfo(MIVLMul, MILog2SEW);

  unsigned MISEW = 1 << MILog2SEW;
  unsigned EEW = MISEW / Factor;
  unsigned Log2EEW = Log2_32(EEW);

  return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(Log2EEW, MI),
                     Log2EEW);
}

/// Check whether MO is a mask operand of MI.
static bool isMaskOperand(const MachineInstr &MI, const MachineOperand &MO,
                          const MachineRegisterInfo *MRI) {

  if (!MO.isReg() || !isVectorRegClass(MO.getReg(), MRI))
    return false;

  const MCInstrDesc &Desc = MI.getDesc();
  return Desc.operands()[MO.getOperandNo()].RegClass == RISCV::VMV0RegClassID;
}

/// Return the OperandInfo for MO.
static OperandInfo getOperandInfo(const MachineOperand &MO,
                                  const MachineRegisterInfo *MRI) {
  const MachineInstr &MI = *MO.getParent();
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());
  assert(RVV && "Could not find MI in PseudoTable");

  // MI has a VLMUL and SEW associated with it. The RVV specification defines
  // the LMUL and SEW of each operand and definition in relation to MI.VLMUL and
  // MI.SEW.
  RISCVII::VLMUL MIVLMul = RISCVII::getLMul(MI.getDesc().TSFlags);
  unsigned MILog2SEW =
      MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();

  const bool HasPassthru = RISCVII::isFirstDefTiedToFirstUse(MI.getDesc());

  // We bail out early for instructions that have passthru with non NoRegister,
  // which means they are using TU policy. We are not interested in these
  // since they must preserve the entire register content.
  if (HasPassthru && MO.getOperandNo() == MI.getNumExplicitDefs() &&
      (MO.getReg() != RISCV::NoRegister))
    return {};

  bool IsMODef = MO.getOperandNo() == 0;

  // All mask operands have EEW=1, EMUL=(EEW/SEW)*LMUL
  if (isMaskOperand(MI, MO, MRI))
    return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);

  // switch against BaseInstr to reduce number of cases that need to be
  // considered.
  switch (RVV->BaseInstr) {

  // 6. Configuration-Setting Instructions
  // Configuration setting instructions do not read or write vector registers
  case RISCV::VSETIVLI:
  case RISCV::VSETVL:
  case RISCV::VSETVLI:
    llvm_unreachable("Configuration setting instructions do not read or write "
                     "vector registers");

  // Vector Loads and Stores
  // Vector Unit-Stride Instructions
  // Vector Strided Instructions
  /// Dest EEW encoded in the instruction and EMUL=(EEW/SEW)*LMUL
  case RISCV::VSE8_V:
  case RISCV::VSSE8_V:
    return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(3, MI), 3);
  case RISCV::VSE16_V:
  case RISCV::VSSE16_V:
    return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(4, MI), 4);
  case RISCV::VSE32_V:
  case RISCV::VSSE32_V:
    return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(5, MI), 5);
  case RISCV::VSE64_V:
  case RISCV::VSSE64_V:
    return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(6, MI), 6);

  // Vector Integer Arithmetic Instructions
  // Vector Single-Width Integer Add and Subtract
  case RISCV::VADD_VI:
  case RISCV::VADD_VV:
  case RISCV::VADD_VX:
  case RISCV::VSUB_VV:
  case RISCV::VSUB_VX:
  case RISCV::VRSUB_VI:
  case RISCV::VRSUB_VX:
  // Vector Bitwise Logical Instructions
  // Vector Single-Width Shift Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VAND_VI:
  case RISCV::VAND_VV:
  case RISCV::VAND_VX:
  case RISCV::VOR_VI:
  case RISCV::VOR_VV:
  case RISCV::VOR_VX:
  case RISCV::VXOR_VI:
  case RISCV::VXOR_VV:
  case RISCV::VXOR_VX:
  case RISCV::VSLL_VI:
  case RISCV::VSLL_VV:
  case RISCV::VSLL_VX:
  case RISCV::VSRL_VI:
  case RISCV::VSRL_VV:
  case RISCV::VSRL_VX:
  case RISCV::VSRA_VI:
  case RISCV::VSRA_VV:
  case RISCV::VSRA_VX:
  // Vector Integer Min/Max Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VMINU_VV:
  case RISCV::VMINU_VX:
  case RISCV::VMIN_VV:
  case RISCV::VMIN_VX:
  case RISCV::VMAXU_VV:
  case RISCV::VMAXU_VX:
  case RISCV::VMAX_VV:
  case RISCV::VMAX_VX:
  // Vector Single-Width Integer Multiply Instructions
  // Source and Dest EEW=SEW and EMUL=LMUL.
  case RISCV::VMUL_VV:
  case RISCV::VMUL_VX:
  case RISCV::VMULH_VV:
  case RISCV::VMULH_VX:
  case RISCV::VMULHU_VV:
  case RISCV::VMULHU_VX:
  case RISCV::VMULHSU_VV:
  case RISCV::VMULHSU_VX:
  // Vector Integer Divide Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VDIVU_VV:
  case RISCV::VDIVU_VX:
  case RISCV::VDIV_VV:
  case RISCV::VDIV_VX:
  case RISCV::VREMU_VV:
  case RISCV::VREMU_VX:
  case RISCV::VREM_VV:
  case RISCV::VREM_VX:
  // Vector Single-Width Integer Multiply-Add Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VMACC_VV:
  case RISCV::VMACC_VX:
  case RISCV::VNMSAC_VV:
  case RISCV::VNMSAC_VX:
  case RISCV::VMADD_VV:
  case RISCV::VMADD_VX:
  case RISCV::VNMSUB_VV:
  case RISCV::VNMSUB_VX:
  // Vector Integer Merge Instructions
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  // EEW=SEW and EMUL=LMUL, except the mask operand has EEW=1 and EMUL=
  // (EEW/SEW)*LMUL. Mask operand is handled before this switch.
  case RISCV::VMERGE_VIM:
  case RISCV::VMERGE_VVM:
  case RISCV::VMERGE_VXM:
  case RISCV::VADC_VIM:
  case RISCV::VADC_VVM:
  case RISCV::VADC_VXM:
  case RISCV::VSBC_VVM:
  case RISCV::VSBC_VXM:
  // Vector Integer Move Instructions
  // Vector Fixed-Point Arithmetic Instructions
  // Vector Single-Width Saturating Add and Subtract
  // Vector Single-Width Averaging Add and Subtract
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VMV_V_I:
  case RISCV::VMV_V_V:
  case RISCV::VMV_V_X:
  case RISCV::VSADDU_VI:
  case RISCV::VSADDU_VV:
  case RISCV::VSADDU_VX:
  case RISCV::VSADD_VI:
  case RISCV::VSADD_VV:
  case RISCV::VSADD_VX:
  case RISCV::VSSUBU_VV:
  case RISCV::VSSUBU_VX:
  case RISCV::VSSUB_VV:
  case RISCV::VSSUB_VX:
  case RISCV::VAADDU_VV:
  case RISCV::VAADDU_VX:
  case RISCV::VAADD_VV:
  case RISCV::VAADD_VX:
  case RISCV::VASUBU_VV:
  case RISCV::VASUBU_VX:
  case RISCV::VASUB_VV:
  case RISCV::VASUB_VX:
  // Vector Single-Width Scaling Shift Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VSSRL_VI:
  case RISCV::VSSRL_VV:
  case RISCV::VSSRL_VX:
  case RISCV::VSSRA_VI:
  case RISCV::VSSRA_VV:
  case RISCV::VSSRA_VX:
  // Vector Permutation Instructions
  // Integer Scalar Move Instructions
  // Floating-Point Scalar Move Instructions
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VMV_X_S:
  case RISCV::VMV_S_X:
  case RISCV::VFMV_F_S:
  case RISCV::VFMV_S_F:
  // Vector Slide Instructions
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VSLIDEUP_VI:
  case RISCV::VSLIDEUP_VX:
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDE1UP_VX:
  case RISCV::VFSLIDE1UP_VF:
  case RISCV::VSLIDE1DOWN_VX:
  case RISCV::VFSLIDE1DOWN_VF:
  // Vector Register Gather Instructions
  // EMUL=LMUL. EEW=SEW. For mask operand, EMUL=1 and EEW=1.
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
  // Vector Compress Instruction
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VCOMPRESS_VM:
    return OperandInfo(MIVLMul, MILog2SEW);

  // Vector Widening Integer Add/Subtract
  // Def uses EEW=2*SEW and EMUL=2*LMUL. Operands use EEW=SEW and EMUL=LMUL.
  case RISCV::VWADDU_VV:
  case RISCV::VWADDU_VX:
  case RISCV::VWSUBU_VV:
  case RISCV::VWSUBU_VX:
  case RISCV::VWADD_VV:
  case RISCV::VWADD_VX:
  case RISCV::VWSUB_VV:
  case RISCV::VWSUB_VX:
  case RISCV::VWSLL_VI:
  // Vector Widening Integer Multiply Instructions
  // Source and Destination EMUL=LMUL. Destination EEW=2*SEW. Source EEW=SEW.
  case RISCV::VWMUL_VV:
  case RISCV::VWMUL_VX:
  case RISCV::VWMULSU_VV:
  case RISCV::VWMULSU_VX:
  case RISCV::VWMULU_VV:
  case RISCV::VWMULU_VX:
  // Vector Widening Integer Multiply-Add Instructions
  // Destination EEW=2*SEW and EMUL=2*LMUL. Source EEW=SEW and EMUL=LMUL.
  // A SEW-bit*SEW-bit multiply of the sources forms a 2*SEW-bit value, which
  // is then added to the 2*SEW-bit Dest. These instructions never have a
  // passthru operand.
  case RISCV::VWMACCU_VV:
  case RISCV::VWMACCU_VX:
  case RISCV::VWMACC_VV:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCSU_VV:
  case RISCV::VWMACCSU_VX:
  case RISCV::VWMACCUS_VX: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL =
        IsMODef ? RISCVVType::twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }

  // Def and Op1 uses EEW=2*SEW and EMUL=2*LMUL. Op2 uses EEW=SEW and EMUL=LMUL
  case RISCV::VWADDU_WV:
  case RISCV::VWADDU_WX:
  case RISCV::VWSUBU_WV:
  case RISCV::VWSUBU_WX:
  case RISCV::VWADD_WV:
  case RISCV::VWADD_WX:
  case RISCV::VWSUB_WV:
  case RISCV::VWSUB_WX: {
    bool IsOp1 = HasPassthru ? MO.getOperandNo() == 2 : MO.getOperandNo() == 1;
    bool TwoTimes = IsMODef || IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL =
        TwoTimes ? RISCVVType::twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }

  // Vector Integer Extension
  case RISCV::VZEXT_VF2:
  case RISCV::VSEXT_VF2:
    return getIntegerExtensionOperandInfo(2, MI, MO);
  case RISCV::VZEXT_VF4:
  case RISCV::VSEXT_VF4:
    return getIntegerExtensionOperandInfo(4, MI, MO);
  case RISCV::VZEXT_VF8:
  case RISCV::VSEXT_VF8:
    return getIntegerExtensionOperandInfo(8, MI, MO);

  // Vector Narrowing Integer Right Shift Instructions
  // Destination EEW=SEW and EMUL=LMUL, Op 1 has EEW=2*SEW EMUL=2*LMUL. Op2 has
  // EEW=SEW EMUL=LMUL.
  case RISCV::VNSRL_WX:
  case RISCV::VNSRL_WI:
  case RISCV::VNSRL_WV:
  case RISCV::VNSRA_WI:
  case RISCV::VNSRA_WV:
  case RISCV::VNSRA_WX:
  // Vector Narrowing Fixed-Point Clip Instructions
  // Destination and Op1 EEW=SEW and EMUL=LMUL. Op2 EEW=2*SEW and EMUL=2*LMUL
  case RISCV::VNCLIPU_WI:
  case RISCV::VNCLIPU_WV:
  case RISCV::VNCLIPU_WX:
  case RISCV::VNCLIP_WI:
  case RISCV::VNCLIP_WV:
  case RISCV::VNCLIP_WX: {
    bool IsOp1 = HasPassthru ? MO.getOperandNo() == 2 : MO.getOperandNo() == 1;
    bool TwoTimes = IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL =
        TwoTimes ? RISCVVType::twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }

  // Vector Mask Instructions
  // Vector Mask-Register Logical Instructions
  // vmsbf.m set-before-first mask bit
  // vmsif.m set-including-first mask bit
  // vmsof.m set-only-first mask bit
  // EEW=1 and EMUL=(EEW/SEW)*LMUL
  // We handle the cases when operand is a v0 mask operand above the switch,
  // but these instructions may use non-v0 mask operands and need to be handled
  // specifically.
  case RISCV::VMAND_MM:
  case RISCV::VMNAND_MM:
  case RISCV::VMANDN_MM:
  case RISCV::VMXOR_MM:
  case RISCV::VMOR_MM:
  case RISCV::VMNOR_MM:
  case RISCV::VMORN_MM:
  case RISCV::VMXNOR_MM:
  case RISCV::VMSBF_M:
  case RISCV::VMSIF_M:
  case RISCV::VMSOF_M: {
    return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
  }

  // Vector Integer Compare Instructions
  // Dest EEW=1 and EMUL=(EEW/SEW)*LMUL. Source EEW=SEW and EMUL=LMUL.
  case RISCV::VMSEQ_VI:
  case RISCV::VMSEQ_VV:
  case RISCV::VMSEQ_VX:
  case RISCV::VMSNE_VI:
  case RISCV::VMSNE_VV:
  case RISCV::VMSNE_VX:
  case RISCV::VMSLTU_VV:
  case RISCV::VMSLTU_VX:
  case RISCV::VMSLT_VV:
  case RISCV::VMSLT_VX:
  case RISCV::VMSLEU_VV:
  case RISCV::VMSLEU_VI:
  case RISCV::VMSLEU_VX:
  case RISCV::VMSLE_VV:
  case RISCV::VMSLE_VI:
  case RISCV::VMSLE_VX:
  case RISCV::VMSGTU_VI:
  case RISCV::VMSGTU_VX:
  case RISCV::VMSGT_VI:
  case RISCV::VMSGT_VX:
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  // Dest EEW=1 and EMUL=(EEW/SEW)*LMUL. Source EEW=SEW and EMUL=LMUL. Mask
  // source operand handled above this switch.
  case RISCV::VMADC_VIM:
  case RISCV::VMADC_VVM:
  case RISCV::VMADC_VXM:
  case RISCV::VMSBC_VVM:
  case RISCV::VMSBC_VXM:
  // Dest EEW=1 and EMUL=(EEW/SEW)*LMUL. Source EEW=SEW and EMUL=LMUL.
  case RISCV::VMADC_VV:
  case RISCV::VMADC_VI:
  case RISCV::VMADC_VX:
  case RISCV::VMSBC_VV:
  case RISCV::VMSBC_VX: {
    if (IsMODef)
      return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
    return OperandInfo(MIVLMul, MILog2SEW);
  }

  default:
    return {};
  }
}

/// Return true if this optimization should consider MI for VL reduction. This
/// white-list approach simplifies this optimization for instructions that may
/// have more complex semantics with relation to how it uses VL.
static bool isSupportedInstr(const MachineInstr &MI) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());

  if (!RVV)
    return false;

  switch (RVV->BaseInstr) {
  // Vector Single-Width Integer Add and Subtract
  case RISCV::VADD_VI:
  case RISCV::VADD_VV:
  case RISCV::VADD_VX:
  case RISCV::VSUB_VV:
  case RISCV::VSUB_VX:
  case RISCV::VRSUB_VI:
  case RISCV::VRSUB_VX:
  // Vector Bitwise Logical Instructions
  // Vector Single-Width Shift Instructions
  case RISCV::VAND_VI:
  case RISCV::VAND_VV:
  case RISCV::VAND_VX:
  case RISCV::VOR_VI:
  case RISCV::VOR_VV:
  case RISCV::VOR_VX:
  case RISCV::VXOR_VI:
  case RISCV::VXOR_VV:
  case RISCV::VXOR_VX:
  case RISCV::VSLL_VI:
  case RISCV::VSLL_VV:
  case RISCV::VSLL_VX:
  case RISCV::VSRL_VI:
  case RISCV::VSRL_VV:
  case RISCV::VSRL_VX:
  case RISCV::VSRA_VI:
  case RISCV::VSRA_VV:
  case RISCV::VSRA_VX:
  // Vector Widening Integer Add/Subtract
  case RISCV::VWADDU_VV:
  case RISCV::VWADDU_VX:
  case RISCV::VWSUBU_VV:
  case RISCV::VWSUBU_VX:
  case RISCV::VWADD_VV:
  case RISCV::VWADD_VX:
  case RISCV::VWSUB_VV:
  case RISCV::VWSUB_VX:
  case RISCV::VWADDU_WV:
  case RISCV::VWADDU_WX:
  case RISCV::VWSUBU_WV:
  case RISCV::VWSUBU_WX:
  case RISCV::VWADD_WV:
  case RISCV::VWADD_WX:
  case RISCV::VWSUB_WV:
  case RISCV::VWSUB_WX:
  // Vector Integer Extension
  case RISCV::VZEXT_VF2:
  case RISCV::VSEXT_VF2:
  case RISCV::VZEXT_VF4:
  case RISCV::VSEXT_VF4:
  case RISCV::VZEXT_VF8:
  case RISCV::VSEXT_VF8:
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  // FIXME: Add support
  case RISCV::VMADC_VV:
  case RISCV::VMADC_VI:
  case RISCV::VMADC_VX:
  case RISCV::VMSBC_VV:
  case RISCV::VMSBC_VX:
  // Vector Narrowing Integer Right Shift Instructions
  case RISCV::VNSRL_WX:
  case RISCV::VNSRL_WI:
  case RISCV::VNSRL_WV:
  case RISCV::VNSRA_WI:
  case RISCV::VNSRA_WV:
  case RISCV::VNSRA_WX:
  // Vector Integer Compare Instructions
  case RISCV::VMSEQ_VI:
  case RISCV::VMSEQ_VV:
  case RISCV::VMSEQ_VX:
  case RISCV::VMSNE_VI:
  case RISCV::VMSNE_VV:
  case RISCV::VMSNE_VX:
  case RISCV::VMSLTU_VV:
  case RISCV::VMSLTU_VX:
  case RISCV::VMSLT_VV:
  case RISCV::VMSLT_VX:
  case RISCV::VMSLEU_VV:
  case RISCV::VMSLEU_VI:
  case RISCV::VMSLEU_VX:
  case RISCV::VMSLE_VV:
  case RISCV::VMSLE_VI:
  case RISCV::VMSLE_VX:
  case RISCV::VMSGTU_VI:
  case RISCV::VMSGTU_VX:
  case RISCV::VMSGT_VI:
  case RISCV::VMSGT_VX:
  // Vector Integer Min/Max Instructions
  case RISCV::VMINU_VV:
  case RISCV::VMINU_VX:
  case RISCV::VMIN_VV:
  case RISCV::VMIN_VX:
  case RISCV::VMAXU_VV:
  case RISCV::VMAXU_VX:
  case RISCV::VMAX_VV:
  case RISCV::VMAX_VX:
  // Vector Single-Width Integer Multiply Instructions
  case RISCV::VMUL_VV:
  case RISCV::VMUL_VX:
  case RISCV::VMULH_VV:
  case RISCV::VMULH_VX:
  case RISCV::VMULHU_VV:
  case RISCV::VMULHU_VX:
  case RISCV::VMULHSU_VV:
  case RISCV::VMULHSU_VX:
  // Vector Integer Divide Instructions
  case RISCV::VDIVU_VV:
  case RISCV::VDIVU_VX:
  case RISCV::VDIV_VV:
  case RISCV::VDIV_VX:
  case RISCV::VREMU_VV:
  case RISCV::VREMU_VX:
  case RISCV::VREM_VV:
  case RISCV::VREM_VX:
  // Vector Widening Integer Multiply Instructions
  case RISCV::VWMUL_VV:
  case RISCV::VWMUL_VX:
  case RISCV::VWMULSU_VV:
  case RISCV::VWMULSU_VX:
  case RISCV::VWMULU_VV:
  case RISCV::VWMULU_VX:
  // Vector Single-Width Integer Multiply-Add Instructions
  case RISCV::VMACC_VV:
  case RISCV::VMACC_VX:
  case RISCV::VNMSAC_VV:
  case RISCV::VNMSAC_VX:
  case RISCV::VMADD_VV:
  case RISCV::VMADD_VX:
  case RISCV::VNMSUB_VV:
  case RISCV::VNMSUB_VX:
  // Vector Widening Integer Multiply-Add Instructions
  case RISCV::VWMACCU_VV:
  case RISCV::VWMACCU_VX:
  case RISCV::VWMACC_VV:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCSU_VV:
  case RISCV::VWMACCSU_VX:
  case RISCV::VWMACCUS_VX:
  // Vector Integer Merge Instructions
  // FIXME: Add support
  // Vector Integer Move Instructions
  // FIXME: Add support
  case RISCV::VMV_V_I:
  case RISCV::VMV_V_X:
  case RISCV::VMV_V_V:

  // Vector Crypto
  case RISCV::VWSLL_VI:

  // Vector Mask Instructions
  // Vector Mask-Register Logical Instructions
  // vmsbf.m set-before-first mask bit
  // vmsif.m set-including-first mask bit
  // vmsof.m set-only-first mask bit
  case RISCV::VMAND_MM:
  case RISCV::VMNAND_MM:
  case RISCV::VMANDN_MM:
  case RISCV::VMXOR_MM:
  case RISCV::VMOR_MM:
  case RISCV::VMNOR_MM:
  case RISCV::VMORN_MM:
  case RISCV::VMXNOR_MM:
  case RISCV::VMSBF_M:
  case RISCV::VMSIF_M:
  case RISCV::VMSOF_M:
    return true;
  }

  return false;
}

/// Return true if MO is a vector operand but is used as a scalar operand.
static bool isVectorOpUsedAsScalarOp(MachineOperand &MO) {
  MachineInstr *MI = MO.getParent();
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI->getOpcode());

  if (!RVV)
    return false;

  switch (RVV->BaseInstr) {
  // Reductions only use vs1[0] of vs1
  case RISCV::VREDAND_VS:
  case RISCV::VREDMAX_VS:
  case RISCV::VREDMAXU_VS:
  case RISCV::VREDMIN_VS:
  case RISCV::VREDMINU_VS:
  case RISCV::VREDOR_VS:
  case RISCV::VREDSUM_VS:
  case RISCV::VREDXOR_VS:
  case RISCV::VWREDSUM_VS:
  case RISCV::VWREDSUMU_VS:
  case RISCV::VFREDMAX_VS:
  case RISCV::VFREDMIN_VS:
  case RISCV::VFREDOSUM_VS:
  case RISCV::VFREDUSUM_VS:
  case RISCV::VFWREDOSUM_VS:
  case RISCV::VFWREDUSUM_VS:
    return MO.getOperandNo() == 3;
  default:
    return false;
  }
}

/// Return true if MI may read elements past VL.
static bool mayReadPastVL(const MachineInstr &MI) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());
  if (!RVV)
    return true;

  switch (RVV->BaseInstr) {
  // vslidedown instructions may read elements past VL. They are handled
  // according to current tail policy.
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDE1DOWN_VX:
  case RISCV::VFSLIDE1DOWN_VF:

  // vrgather instructions may read the source vector at any index < VLMAX,
  // regardless of VL.
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
  case RISCV::VRGATHEREI16_VV:
    return true;

  default:
    return false;
  }
}

bool RISCVVLOptimizer::isCandidate(const MachineInstr &MI) const {
  const MCInstrDesc &Desc = MI.getDesc();
  if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags))
    return false;
  if (MI.getNumDefs() != 1)
    return false;

  // If we're not using VLMAX, then we need to be careful whether we are using
  // TA/TU when there is a non-undef Passthru. But when we are using VLMAX, it
  // does not matter whether we are using TA/TU with a non-undef Passthru, since
  // there are no tail elements to be preserved.
  unsigned VLOpNum = RISCVII::getVLOpNum(Desc);
  const MachineOperand &VLOp = MI.getOperand(VLOpNum);
  if (VLOp.isReg() || VLOp.getImm() != RISCV::VLMaxSentinel) {
    // If MI has a non-undef passthru, we will not try to optimize it since
    // that requires us to preserve tail elements according to TA/TU.
    // Otherwise, The MI has an undef Passthru, so it doesn't matter whether we
    // are using TA/TU.
    bool HasPassthru = RISCVII::isFirstDefTiedToFirstUse(Desc);
    unsigned PassthruOpIdx = MI.getNumExplicitDefs();
    if (HasPassthru &&
        MI.getOperand(PassthruOpIdx).getReg() != RISCV::NoRegister) {
      LLVM_DEBUG(
          dbgs() << "  Not a candidate because it uses non-undef passthru"
                    " with non-VLMAX VL\n");
      return false;
    }
  }

  // If the VL is 1, then there is no need to reduce it. This is an
  // optimization, not needed to preserve correctness.
  if (VLOp.isImm() && VLOp.getImm() == 1) {
    LLVM_DEBUG(dbgs() << "  Not a candidate because VL is already 1\n");
    return false;
  }

  // Some instructions that produce vectors have semantics that make it more
  // difficult to determine whether the VL can be reduced. For example, some
  // instructions, such as reductions, may write lanes past VL to a scalar
  // register. Other instructions, such as some loads or stores, may write
  // lower lanes using data from higher lanes. There may be other complex
  // semantics not mentioned here that make it hard to determine whether
  // the VL can be optimized. As a result, a white-list of supported
  // instructions is used. Over time, more instructions can be supported
  // upon careful examination of their semantics under the logic in this
  // optimization.
  // TODO: Use a better approach than a white-list, such as adding
  // properties to instructions using something like TSFlags.
  if (!isSupportedInstr(MI)) {
    LLVM_DEBUG(dbgs() << "Not a candidate due to unsupported instruction\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Found a candidate for VL reduction: " << MI << "\n");
  return true;
}

bool RISCVVLOptimizer::checkUsers(const MachineOperand *&CommonVL,
                                  MachineInstr &MI) {
  // FIXME: Avoid visiting each user for each time we visit something on the
  // worklist, combined with an extra visit from the outer loop. Restructure
  // along lines of an instcombine style worklist which integrates the outer
  // pass.
  bool CanReduceVL = true;
  for (auto &UserOp : MRI->use_operands(MI.getOperand(0).getReg())) {
    const MachineInstr &UserMI = *UserOp.getParent();
    LLVM_DEBUG(dbgs() << "  Checking user: " << UserMI << "\n");

    // Instructions like reductions may use a vector register as a scalar
    // register. In this case, we should treat it like a scalar register which
    // does not impact the decision on whether to optimize VL.
    // TODO: Treat it like a scalar register instead of bailing out.
    if (isVectorOpUsedAsScalarOp(UserOp)) {
      CanReduceVL = false;
      break;
    }

    if (mayReadPastVL(UserMI)) {
      LLVM_DEBUG(dbgs() << "    Abort because used by unsafe instruction\n");
      CanReduceVL = false;
      break;
    }

    // Tied operands might pass through.
    if (UserOp.isTied()) {
      LLVM_DEBUG(dbgs() << "    Abort because user used as tied operand\n");
      CanReduceVL = false;
      break;
    }

    const MCInstrDesc &Desc = UserMI.getDesc();
    if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags)) {
      LLVM_DEBUG(dbgs() << "    Abort due to lack of VL or SEW, assume that"
                           " use VLMAX\n");
      CanReduceVL = false;
      break;
    }

    unsigned VLOpNum = RISCVII::getVLOpNum(Desc);
    const MachineOperand &VLOp = UserMI.getOperand(VLOpNum);

    // Looking for an immediate or a register VL that isn't X0.
    assert((!VLOp.isReg() || VLOp.getReg() != RISCV::X0) &&
           "Did not expect X0 VL");

    if (!CommonVL) {
      CommonVL = &VLOp;
      LLVM_DEBUG(dbgs() << "    User VL is: " << VLOp << "\n");
    } else if (!CommonVL->isIdenticalTo(VLOp)) {
      // FIXME: This check requires all users to have the same VL. We can relax
      // this and get the largest VL amongst all users.
      LLVM_DEBUG(dbgs() << "    Abort because users have different VL\n");
      CanReduceVL = false;
      break;
    }

    // The SEW and LMUL of destination and source registers need to match.
    OperandInfo ConsumerInfo = getOperandInfo(UserOp, MRI);
    OperandInfo ProducerInfo = getOperandInfo(MI.getOperand(0), MRI);
    if (ConsumerInfo.isUnknown() || ProducerInfo.isUnknown() ||
        !OperandInfo::EMULAndEEWAreEqual(ConsumerInfo, ProducerInfo)) {
      LLVM_DEBUG(dbgs() << "    Abort due to incompatible or unknown "
                           "information for EMUL or EEW.\n");
      LLVM_DEBUG(dbgs() << "      ConsumerInfo is: " << ConsumerInfo << "\n");
      LLVM_DEBUG(dbgs() << "      ProducerInfo is: " << ProducerInfo << "\n");
      CanReduceVL = false;
      break;
    }
  }
  return CanReduceVL;
}

bool RISCVVLOptimizer::tryReduceVL(MachineInstr &OrigMI) {
  SetVector<MachineInstr *> Worklist;
  Worklist.insert(&OrigMI);

  bool MadeChange = false;
  while (!Worklist.empty()) {
    MachineInstr &MI = *Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "Trying to reduce VL for " << MI << "\n");

    const MachineOperand *CommonVL = nullptr;
    bool CanReduceVL = true;
    if (isVectorRegClass(MI.getOperand(0).getReg(), MRI))
      CanReduceVL = checkUsers(CommonVL, MI);

    if (!CanReduceVL || !CommonVL)
      continue;

    assert((CommonVL->isImm() || CommonVL->getReg().isVirtual()) &&
           "Expected VL to be an Imm or virtual Reg");

    unsigned VLOpNum = RISCVII::getVLOpNum(MI.getDesc());
    MachineOperand &VLOp = MI.getOperand(VLOpNum);

    if (!RISCV::isVLKnownLE(*CommonVL, VLOp)) {
      LLVM_DEBUG(dbgs() << "    Abort due to CommonVL not <= VLOp.\n");
      continue;
    }

    if (CommonVL->isImm()) {
      LLVM_DEBUG(dbgs() << "  Reduce VL from " << VLOp << " to "
                        << CommonVL->getImm() << " for " << MI << "\n");
      VLOp.ChangeToImmediate(CommonVL->getImm());
    } else {
      const MachineInstr *VLMI = MRI->getVRegDef(CommonVL->getReg());
      if (!MDT->dominates(VLMI, &MI))
        continue;
      LLVM_DEBUG(
          dbgs() << "  Reduce VL from " << VLOp << " to "
                 << printReg(CommonVL->getReg(), MRI->getTargetRegisterInfo())
                 << " for " << MI << "\n");

      // All our checks passed. We can reduce VL.
      VLOp.ChangeToRegister(CommonVL->getReg(), false);
    }

    MadeChange = true;

    // Now add all inputs to this instruction to the worklist.
    for (auto &Op : MI.operands()) {
      if (!Op.isReg() || !Op.isUse() || !Op.getReg().isVirtual())
        continue;

      if (!isVectorRegClass(Op.getReg(), MRI))
        continue;

      MachineInstr *DefMI = MRI->getVRegDef(Op.getReg());

      if (!isCandidate(*DefMI))
        continue;

      Worklist.insert(DefMI);
    }
  }

  return MadeChange;
}

bool RISCVVLOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    // Visit instructions in reverse order.
    for (auto &MI : make_range(MBB.rbegin(), MBB.rend())) {
      if (!isCandidate(MI))
        continue;

      MadeChange |= tryReduceVL(MI);
    }
  }

  return MadeChange;
}
