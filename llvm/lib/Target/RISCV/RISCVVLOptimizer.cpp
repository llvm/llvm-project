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
#include "llvm/ADT/PostOrderIterator.h"
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
  const TargetInstrInfo *TII;

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
  std::optional<MachineOperand> getMinimumVLForUser(MachineOperand &UserOp);
  /// Returns the largest common VL MachineOperand that may be used to optimize
  /// MI. Returns std::nullopt if it failed to find a suitable VL.
  std::optional<MachineOperand> checkUsers(MachineInstr &MI);
  bool tryReduceVL(MachineInstr &MI);
  bool isCandidate(const MachineInstr &MI) const;

  /// For a given instruction, records what elements of it are demanded by
  /// downstream users.
  DenseMap<const MachineInstr *, MachineOperand> DemandedVLs;
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
  // Represent as 1,2,4,8, ... and fractional indicator. This is because
  // EMUL can take on values that don't map to RISCVII::VLMUL values exactly.
  // For example, a mask operand can have an EMUL less than MF8.
  std::optional<std::pair<unsigned, bool>> EMUL;

  unsigned Log2EEW;

  OperandInfo(RISCVII::VLMUL EMUL, unsigned Log2EEW)
      : EMUL(RISCVVType::decodeVLMUL(EMUL)), Log2EEW(Log2EEW) {}

  OperandInfo(std::pair<unsigned, bool> EMUL, unsigned Log2EEW)
      : EMUL(EMUL), Log2EEW(Log2EEW) {}

  OperandInfo(unsigned Log2EEW) : Log2EEW(Log2EEW) {}

  OperandInfo() = delete;

  static bool EMULAndEEWAreEqual(const OperandInfo &A, const OperandInfo &B) {
    return A.Log2EEW == B.Log2EEW && A.EMUL->first == B.EMUL->first &&
           A.EMUL->second == B.EMUL->second;
  }

  static bool EEWAreEqual(const OperandInfo &A, const OperandInfo &B) {
    return A.Log2EEW == B.Log2EEW;
  }

  void print(raw_ostream &OS) const {
    if (EMUL) {
      OS << "EMUL: m";
      if (EMUL->second)
        OS << "f";
      OS << EMUL->first;
    } else
      OS << "EMUL: unknown\n";
    OS << ", EEW: " << (1 << Log2EEW);
  }
};

LLVM_ATTRIBUTE_UNUSED
static raw_ostream &operator<<(raw_ostream &OS, const OperandInfo &OI) {
  OI.print(OS);
  return OS;
}

LLVM_ATTRIBUTE_UNUSED
static raw_ostream &operator<<(raw_ostream &OS,
                               const std::optional<OperandInfo> &OI) {
  if (OI)
    OI->print(OS);
  else
    OS << "nullopt";
  return OS;
}

namespace llvm {
namespace RISCVVType {
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

/// Dest has EEW=SEW. Source EEW=SEW/Factor (i.e. F2 => EEW/2).
/// SEW comes from TSFlags of MI.
static unsigned getIntegerExtensionOperandEEW(unsigned Factor,
                                              const MachineInstr &MI,
                                              const MachineOperand &MO) {
  unsigned MILog2SEW =
      MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();

  if (MO.getOperandNo() == 0)
    return MILog2SEW;

  unsigned MISEW = 1 << MILog2SEW;
  unsigned EEW = MISEW / Factor;
  unsigned Log2EEW = Log2_32(EEW);

  return Log2EEW;
}

/// Check whether MO is a mask operand of MI.
static bool isMaskOperand(const MachineInstr &MI, const MachineOperand &MO,
                          const MachineRegisterInfo *MRI) {

  if (!MO.isReg() || !isVectorRegClass(MO.getReg(), MRI))
    return false;

  const MCInstrDesc &Desc = MI.getDesc();
  return Desc.operands()[MO.getOperandNo()].RegClass == RISCV::VMV0RegClassID;
}

static std::optional<unsigned>
getOperandLog2EEW(const MachineOperand &MO, const MachineRegisterInfo *MRI) {
  const MachineInstr &MI = *MO.getParent();
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());
  assert(RVV && "Could not find MI in PseudoTable");

  // MI has a SEW associated with it. The RVV specification defines
  // the EEW of each operand and definition in relation to MI.SEW.
  unsigned MILog2SEW =
      MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();

  const bool HasPassthru = RISCVII::isFirstDefTiedToFirstUse(MI.getDesc());
  const bool IsTied = RISCVII::isTiedPseudo(MI.getDesc().TSFlags);

  bool IsMODef = MO.getOperandNo() == 0;

  // All mask operands have EEW=1
  if (isMaskOperand(MI, MO, MRI))
    return 0;

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
  /// Dest EEW encoded in the instruction
  case RISCV::VLM_V:
  case RISCV::VSM_V:
    return 0;
  case RISCV::VLE8_V:
  case RISCV::VSE8_V:
  case RISCV::VLSE8_V:
  case RISCV::VSSE8_V:
    return 3;
  case RISCV::VLE16_V:
  case RISCV::VSE16_V:
  case RISCV::VLSE16_V:
  case RISCV::VSSE16_V:
    return 4;
  case RISCV::VLE32_V:
  case RISCV::VSE32_V:
  case RISCV::VLSE32_V:
  case RISCV::VSSE32_V:
    return 5;
  case RISCV::VLE64_V:
  case RISCV::VSE64_V:
  case RISCV::VLSE64_V:
  case RISCV::VSSE64_V:
    return 6;

  // Vector Indexed Instructions
  // vs(o|u)xei<eew>.v
  // Dest/Data (operand 0) EEW=SEW.  Source EEW=<eew>.
  case RISCV::VLUXEI8_V:
  case RISCV::VLOXEI8_V:
  case RISCV::VSUXEI8_V:
  case RISCV::VSOXEI8_V: {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 3;
  }
  case RISCV::VLUXEI16_V:
  case RISCV::VLOXEI16_V:
  case RISCV::VSUXEI16_V:
  case RISCV::VSOXEI16_V: {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 4;
  }
  case RISCV::VLUXEI32_V:
  case RISCV::VLOXEI32_V:
  case RISCV::VSUXEI32_V:
  case RISCV::VSOXEI32_V: {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 5;
  }
  case RISCV::VLUXEI64_V:
  case RISCV::VLOXEI64_V:
  case RISCV::VSUXEI64_V:
  case RISCV::VSOXEI64_V: {
    if (MO.getOperandNo() == 0)
      return MILog2SEW;
    return 6;
  }

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
  // EEW=SEW.
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
  // EEW=SEW.
  case RISCV::VMINU_VV:
  case RISCV::VMINU_VX:
  case RISCV::VMIN_VV:
  case RISCV::VMIN_VX:
  case RISCV::VMAXU_VV:
  case RISCV::VMAXU_VX:
  case RISCV::VMAX_VV:
  case RISCV::VMAX_VX:
  // Vector Single-Width Integer Multiply Instructions
  // Source and Dest EEW=SEW.
  case RISCV::VMUL_VV:
  case RISCV::VMUL_VX:
  case RISCV::VMULH_VV:
  case RISCV::VMULH_VX:
  case RISCV::VMULHU_VV:
  case RISCV::VMULHU_VX:
  case RISCV::VMULHSU_VV:
  case RISCV::VMULHSU_VX:
  // Vector Integer Divide Instructions
  // EEW=SEW.
  case RISCV::VDIVU_VV:
  case RISCV::VDIVU_VX:
  case RISCV::VDIV_VV:
  case RISCV::VDIV_VX:
  case RISCV::VREMU_VV:
  case RISCV::VREMU_VX:
  case RISCV::VREM_VV:
  case RISCV::VREM_VX:
  // Vector Single-Width Integer Multiply-Add Instructions
  // EEW=SEW.
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
  // EEW=SEW, except the mask operand has EEW=1. Mask operand is handled
  // before this switch.
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
  // EEW=SEW.
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
  // Vector Single-Width Fractional Multiply with Rounding and Saturation
  // EEW=SEW. The instruction produces 2*SEW product internally but
  // saturates to fit into SEW bits.
  case RISCV::VSMUL_VV:
  case RISCV::VSMUL_VX:
  // Vector Single-Width Scaling Shift Instructions
  // EEW=SEW.
  case RISCV::VSSRL_VI:
  case RISCV::VSSRL_VV:
  case RISCV::VSSRL_VX:
  case RISCV::VSSRA_VI:
  case RISCV::VSSRA_VV:
  case RISCV::VSSRA_VX:
  // Vector Permutation Instructions
  // Integer Scalar Move Instructions
  // Floating-Point Scalar Move Instructions
  // EEW=SEW.
  case RISCV::VMV_X_S:
  case RISCV::VMV_S_X:
  case RISCV::VFMV_F_S:
  case RISCV::VFMV_S_F:
  // Vector Slide Instructions
  // EEW=SEW.
  case RISCV::VSLIDEUP_VI:
  case RISCV::VSLIDEUP_VX:
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDE1UP_VX:
  case RISCV::VFSLIDE1UP_VF:
  case RISCV::VSLIDE1DOWN_VX:
  case RISCV::VFSLIDE1DOWN_VF:
  // Vector Register Gather Instructions
  // EEW=SEW. For mask operand, EEW=1.
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
  // Vector Compress Instruction
  // EEW=SEW.
  case RISCV::VCOMPRESS_VM:
  // Vector Element Index Instruction
  case RISCV::VID_V:
  // Vector Single-Width Floating-Point Add/Subtract Instructions
  case RISCV::VFADD_VF:
  case RISCV::VFADD_VV:
  case RISCV::VFSUB_VF:
  case RISCV::VFSUB_VV:
  case RISCV::VFRSUB_VF:
  // Vector Single-Width Floating-Point Multiply/Divide Instructions
  case RISCV::VFMUL_VF:
  case RISCV::VFMUL_VV:
  case RISCV::VFDIV_VF:
  case RISCV::VFDIV_VV:
  case RISCV::VFRDIV_VF:
  // Vector Floating-Point Square-Root Instruction
  case RISCV::VFSQRT_V:
  // Vector Floating-Point Reciprocal Square-Root Estimate Instruction
  case RISCV::VFRSQRT7_V:
  // Vector Floating-Point Reciprocal Estimate Instruction
  case RISCV::VFREC7_V:
  // Vector Floating-Point MIN/MAX Instructions
  case RISCV::VFMIN_VF:
  case RISCV::VFMIN_VV:
  case RISCV::VFMAX_VF:
  case RISCV::VFMAX_VV:
  // Vector Floating-Point Sign-Injection Instructions
  case RISCV::VFSGNJ_VF:
  case RISCV::VFSGNJ_VV:
  case RISCV::VFSGNJN_VV:
  case RISCV::VFSGNJN_VF:
  case RISCV::VFSGNJX_VF:
  case RISCV::VFSGNJX_VV:
  // Vector Floating-Point Classify Instruction
  case RISCV::VFCLASS_V:
  // Vector Floating-Point Move Instruction
  case RISCV::VFMV_V_F:
  // Single-Width Floating-Point/Integer Type-Convert Instructions
  case RISCV::VFCVT_XU_F_V:
  case RISCV::VFCVT_X_F_V:
  case RISCV::VFCVT_RTZ_XU_F_V:
  case RISCV::VFCVT_RTZ_X_F_V:
  case RISCV::VFCVT_F_XU_V:
  case RISCV::VFCVT_F_X_V:
  // Vector Floating-Point Merge Instruction
  case RISCV::VFMERGE_VFM:
  // Vector count population in mask vcpop.m
  // vfirst find-first-set mask bit
  case RISCV::VCPOP_M:
  case RISCV::VFIRST_M:
    return MILog2SEW;

  // Vector Widening Integer Add/Subtract
  // Def uses EEW=2*SEW . Operands use EEW=SEW.
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
  // Destination EEW=2*SEW. Source EEW=SEW.
  case RISCV::VWMUL_VV:
  case RISCV::VWMUL_VX:
  case RISCV::VWMULSU_VV:
  case RISCV::VWMULSU_VX:
  case RISCV::VWMULU_VV:
  case RISCV::VWMULU_VX:
  // Vector Widening Integer Multiply-Add Instructions
  // Destination EEW=2*SEW. Source EEW=SEW.
  // A SEW-bit*SEW-bit multiply of the sources forms a 2*SEW-bit value, which
  // is then added to the 2*SEW-bit Dest. These instructions never have a
  // passthru operand.
  case RISCV::VWMACCU_VV:
  case RISCV::VWMACCU_VX:
  case RISCV::VWMACC_VV:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCSU_VV:
  case RISCV::VWMACCSU_VX:
  case RISCV::VWMACCUS_VX:
  // Vector Widening Floating-Point Fused Multiply-Add Instructions
  case RISCV::VFWMACC_VF:
  case RISCV::VFWMACC_VV:
  case RISCV::VFWNMACC_VF:
  case RISCV::VFWNMACC_VV:
  case RISCV::VFWMSAC_VF:
  case RISCV::VFWMSAC_VV:
  case RISCV::VFWNMSAC_VF:
  case RISCV::VFWNMSAC_VV:
  // Vector Widening Floating-Point Add/Subtract Instructions
  // Dest EEW=2*SEW. Source EEW=SEW.
  case RISCV::VFWADD_VV:
  case RISCV::VFWADD_VF:
  case RISCV::VFWSUB_VV:
  case RISCV::VFWSUB_VF:
  // Vector Widening Floating-Point Multiply
  case RISCV::VFWMUL_VF:
  case RISCV::VFWMUL_VV:
  // Widening Floating-Point/Integer Type-Convert Instructions
  case RISCV::VFWCVT_XU_F_V:
  case RISCV::VFWCVT_X_F_V:
  case RISCV::VFWCVT_RTZ_XU_F_V:
  case RISCV::VFWCVT_RTZ_X_F_V:
  case RISCV::VFWCVT_F_XU_V:
  case RISCV::VFWCVT_F_X_V:
  case RISCV::VFWCVT_F_F_V:
  case RISCV::VFWCVTBF16_F_F_V:
    return IsMODef ? MILog2SEW + 1 : MILog2SEW;

  // Def and Op1 uses EEW=2*SEW. Op2 uses EEW=SEW.
  case RISCV::VWADDU_WV:
  case RISCV::VWADDU_WX:
  case RISCV::VWSUBU_WV:
  case RISCV::VWSUBU_WX:
  case RISCV::VWADD_WV:
  case RISCV::VWADD_WX:
  case RISCV::VWSUB_WV:
  case RISCV::VWSUB_WX:
  // Vector Widening Floating-Point Add/Subtract Instructions
  case RISCV::VFWADD_WF:
  case RISCV::VFWADD_WV:
  case RISCV::VFWSUB_WF:
  case RISCV::VFWSUB_WV: {
    bool IsOp1 = (HasPassthru && !IsTied) ? MO.getOperandNo() == 2
                                          : MO.getOperandNo() == 1;
    bool TwoTimes = IsMODef || IsOp1;
    return TwoTimes ? MILog2SEW + 1 : MILog2SEW;
  }

  // Vector Integer Extension
  case RISCV::VZEXT_VF2:
  case RISCV::VSEXT_VF2:
    return getIntegerExtensionOperandEEW(2, MI, MO);
  case RISCV::VZEXT_VF4:
  case RISCV::VSEXT_VF4:
    return getIntegerExtensionOperandEEW(4, MI, MO);
  case RISCV::VZEXT_VF8:
  case RISCV::VSEXT_VF8:
    return getIntegerExtensionOperandEEW(8, MI, MO);

  // Vector Narrowing Integer Right Shift Instructions
  // Destination EEW=SEW, Op 1 has EEW=2*SEW. Op2 has EEW=SEW
  case RISCV::VNSRL_WX:
  case RISCV::VNSRL_WI:
  case RISCV::VNSRL_WV:
  case RISCV::VNSRA_WI:
  case RISCV::VNSRA_WV:
  case RISCV::VNSRA_WX:
  // Vector Narrowing Fixed-Point Clip Instructions
  // Destination and Op1 EEW=SEW. Op2 EEW=2*SEW.
  case RISCV::VNCLIPU_WI:
  case RISCV::VNCLIPU_WV:
  case RISCV::VNCLIPU_WX:
  case RISCV::VNCLIP_WI:
  case RISCV::VNCLIP_WV:
  case RISCV::VNCLIP_WX:
  // Narrowing Floating-Point/Integer Type-Convert Instructions
  case RISCV::VFNCVT_XU_F_W:
  case RISCV::VFNCVT_X_F_W:
  case RISCV::VFNCVT_RTZ_XU_F_W:
  case RISCV::VFNCVT_RTZ_X_F_W:
  case RISCV::VFNCVT_F_XU_W:
  case RISCV::VFNCVT_F_X_W:
  case RISCV::VFNCVT_F_F_W:
  case RISCV::VFNCVT_ROD_F_F_W:
  case RISCV::VFNCVTBF16_F_F_W: {
    assert(!IsTied);
    bool IsOp1 = HasPassthru ? MO.getOperandNo() == 2 : MO.getOperandNo() == 1;
    bool TwoTimes = IsOp1;
    return TwoTimes ? MILog2SEW + 1 : MILog2SEW;
  }

  // Vector Mask Instructions
  // Vector Mask-Register Logical Instructions
  // vmsbf.m set-before-first mask bit
  // vmsif.m set-including-first mask bit
  // vmsof.m set-only-first mask bit
  // EEW=1
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
    return MILog2SEW;
  }

  // Vector Iota Instruction
  // EEW=SEW, except the mask operand has EEW=1. Mask operand is not handled
  // before this switch.
  case RISCV::VIOTA_M: {
    if (IsMODef || MO.getOperandNo() == 1)
      return MILog2SEW;
    return 0;
  }

  // Vector Integer Compare Instructions
  // Dest EEW=1. Source EEW=SEW.
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
  // Dest EEW=1. Source EEW=SEW. Mask source operand handled above this switch.
  case RISCV::VMADC_VIM:
  case RISCV::VMADC_VVM:
  case RISCV::VMADC_VXM:
  case RISCV::VMSBC_VVM:
  case RISCV::VMSBC_VXM:
  // Dest EEW=1. Source EEW=SEW.
  case RISCV::VMADC_VV:
  case RISCV::VMADC_VI:
  case RISCV::VMADC_VX:
  case RISCV::VMSBC_VV:
  case RISCV::VMSBC_VX:
  // 13.13. Vector Floating-Point Compare Instructions
  // Dest EEW=1. Source EEW=SEW
  case RISCV::VMFEQ_VF:
  case RISCV::VMFEQ_VV:
  case RISCV::VMFNE_VF:
  case RISCV::VMFNE_VV:
  case RISCV::VMFLT_VF:
  case RISCV::VMFLT_VV:
  case RISCV::VMFLE_VF:
  case RISCV::VMFLE_VV:
  case RISCV::VMFGT_VF:
  case RISCV::VMFGE_VF: {
    if (IsMODef)
      return 0;
    return MILog2SEW;
  }

  // Vector Reduction Operations
  // Vector Single-Width Integer Reduction Instructions
  case RISCV::VREDAND_VS:
  case RISCV::VREDMAX_VS:
  case RISCV::VREDMAXU_VS:
  case RISCV::VREDMIN_VS:
  case RISCV::VREDMINU_VS:
  case RISCV::VREDOR_VS:
  case RISCV::VREDSUM_VS:
  case RISCV::VREDXOR_VS:
  // Vector Single-Width Floating-Point Reduction Instructions
  case RISCV::VFREDMAX_VS:
  case RISCV::VFREDMIN_VS:
  case RISCV::VFREDOSUM_VS:
  case RISCV::VFREDUSUM_VS: {
    return MILog2SEW;
  }

  // Vector Widening Integer Reduction Instructions
  // The Dest and VS1 read only element 0 for the vector register. Return
  // 2*EEW for these. VS2 has EEW=SEW and EMUL=LMUL.
  case RISCV::VWREDSUM_VS:
  case RISCV::VWREDSUMU_VS:
  // Vector Widening Floating-Point Reduction Instructions
  case RISCV::VFWREDOSUM_VS:
  case RISCV::VFWREDUSUM_VS: {
    bool TwoTimes = IsMODef || MO.getOperandNo() == 3;
    return TwoTimes ? MILog2SEW + 1 : MILog2SEW;
  }

  default:
    return std::nullopt;
  }
}

static std::optional<OperandInfo>
getOperandInfo(const MachineOperand &MO, const MachineRegisterInfo *MRI) {
  const MachineInstr &MI = *MO.getParent();
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());
  assert(RVV && "Could not find MI in PseudoTable");

  std::optional<unsigned> Log2EEW = getOperandLog2EEW(MO, MRI);
  if (!Log2EEW)
    return std::nullopt;

  switch (RVV->BaseInstr) {
  // Vector Reduction Operations
  // Vector Single-Width Integer Reduction Instructions
  // Vector Widening Integer Reduction Instructions
  // Vector Widening Floating-Point Reduction Instructions
  // The Dest and VS1 only read element 0 of the vector register. Return just
  // the EEW for these.
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
  case RISCV::VFWREDOSUM_VS:
  case RISCV::VFWREDUSUM_VS:
    if (MO.getOperandNo() != 2)
      return OperandInfo(*Log2EEW);
    break;
  };

  // All others have EMUL=EEW/SEW*LMUL
  return OperandInfo(RISCVVType::getEMULEqualsEEWDivSEWTimesLMUL(*Log2EEW, MI),
                     *Log2EEW);
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
  // Vector Unit-Stride Instructions
  // Vector Strided Instructions
  case RISCV::VLM_V:
  case RISCV::VLE8_V:
  case RISCV::VLSE8_V:
  case RISCV::VLE16_V:
  case RISCV::VLSE16_V:
  case RISCV::VLE32_V:
  case RISCV::VLSE32_V:
  case RISCV::VLE64_V:
  case RISCV::VLSE64_V:
  // Vector Indexed Instructions
  case RISCV::VLUXEI8_V:
  case RISCV::VLOXEI8_V:
  case RISCV::VLUXEI16_V:
  case RISCV::VLOXEI16_V:
  case RISCV::VLUXEI32_V:
  case RISCV::VLOXEI32_V:
  case RISCV::VLUXEI64_V:
  case RISCV::VLOXEI64_V: {
    for (const MachineMemOperand *MMO : MI.memoperands())
      if (MMO->isVolatile())
        return false;
    return true;
  }

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
  // Vector Integer Merge Instructions
  case RISCV::VMERGE_VIM:
  case RISCV::VMERGE_VVM:
  case RISCV::VMERGE_VXM:
  // Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  case RISCV::VADC_VIM:
  case RISCV::VADC_VVM:
  case RISCV::VADC_VXM:
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
  // Vector Single-Width Averaging Add and Subtract
  case RISCV::VAADDU_VV:
  case RISCV::VAADDU_VX:
  case RISCV::VAADD_VV:
  case RISCV::VAADD_VX:
  case RISCV::VASUBU_VV:
  case RISCV::VASUBU_VX:
  case RISCV::VASUB_VV:
  case RISCV::VASUB_VX:

  // Vector Crypto
  case RISCV::VWSLL_VI:

  // Vector Mask Instructions
  // Vector Mask-Register Logical Instructions
  // vmsbf.m set-before-first mask bit
  // vmsif.m set-including-first mask bit
  // vmsof.m set-only-first mask bit
  // Vector Iota Instruction
  // Vector Element Index Instruction
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
  case RISCV::VIOTA_M:
  case RISCV::VID_V:
  // Vector Single-Width Floating-Point Add/Subtract Instructions
  case RISCV::VFADD_VF:
  case RISCV::VFADD_VV:
  case RISCV::VFSUB_VF:
  case RISCV::VFSUB_VV:
  case RISCV::VFRSUB_VF:
  // Vector Widening Floating-Point Add/Subtract Instructions
  case RISCV::VFWADD_VV:
  case RISCV::VFWADD_VF:
  case RISCV::VFWSUB_VV:
  case RISCV::VFWSUB_VF:
  case RISCV::VFWADD_WF:
  case RISCV::VFWADD_WV:
  case RISCV::VFWSUB_WF:
  case RISCV::VFWSUB_WV:
  // Vector Single-Width Floating-Point Multiply/Divide Instructions
  case RISCV::VFMUL_VF:
  case RISCV::VFMUL_VV:
  case RISCV::VFDIV_VF:
  case RISCV::VFDIV_VV:
  case RISCV::VFRDIV_VF:
  // Vector Widening Floating-Point Multiply
  case RISCV::VFWMUL_VF:
  case RISCV::VFWMUL_VV:
  // Vector Floating-Point MIN/MAX Instructions
  case RISCV::VFMIN_VF:
  case RISCV::VFMIN_VV:
  case RISCV::VFMAX_VF:
  case RISCV::VFMAX_VV:
  // Vector Floating-Point Sign-Injection Instructions
  case RISCV::VFSGNJ_VF:
  case RISCV::VFSGNJ_VV:
  case RISCV::VFSGNJN_VV:
  case RISCV::VFSGNJN_VF:
  case RISCV::VFSGNJX_VF:
  case RISCV::VFSGNJX_VV:
  // Vector Floating-Point Compare Instructions
  case RISCV::VMFEQ_VF:
  case RISCV::VMFEQ_VV:
  case RISCV::VMFNE_VF:
  case RISCV::VMFNE_VV:
  case RISCV::VMFLT_VF:
  case RISCV::VMFLT_VV:
  case RISCV::VMFLE_VF:
  case RISCV::VMFLE_VV:
  case RISCV::VMFGT_VF:
  case RISCV::VMFGE_VF:
  // Single-Width Floating-Point/Integer Type-Convert Instructions
  case RISCV::VFCVT_XU_F_V:
  case RISCV::VFCVT_X_F_V:
  case RISCV::VFCVT_RTZ_XU_F_V:
  case RISCV::VFCVT_RTZ_X_F_V:
  case RISCV::VFCVT_F_XU_V:
  case RISCV::VFCVT_F_X_V:
  // Widening Floating-Point/Integer Type-Convert Instructions
  case RISCV::VFWCVT_XU_F_V:
  case RISCV::VFWCVT_X_F_V:
  case RISCV::VFWCVT_RTZ_XU_F_V:
  case RISCV::VFWCVT_RTZ_X_F_V:
  case RISCV::VFWCVT_F_XU_V:
  case RISCV::VFWCVT_F_X_V:
  case RISCV::VFWCVT_F_F_V:
  case RISCV::VFWCVTBF16_F_F_V:
  // Narrowing Floating-Point/Integer Type-Convert Instructions
  case RISCV::VFNCVT_XU_F_W:
  case RISCV::VFNCVT_X_F_W:
  case RISCV::VFNCVT_RTZ_XU_F_W:
  case RISCV::VFNCVT_RTZ_X_F_W:
  case RISCV::VFNCVT_F_XU_W:
  case RISCV::VFNCVT_F_X_W:
  case RISCV::VFNCVT_F_F_W:
  case RISCV::VFNCVT_ROD_F_F_W:
  case RISCV::VFNCVTBF16_F_F_W:
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
  case RISCV::VMV_X_S:
  case RISCV::VFMV_F_S:
    return MO.getOperandNo() == 1;
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

  if (MI.mayRaiseFPException()) {
    LLVM_DEBUG(dbgs() << "Not a candidate because may raise FP exception\n");
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

  assert(MI.getOperand(0).isReg() &&
         isVectorRegClass(MI.getOperand(0).getReg(), MRI) &&
         "All supported instructions produce a vector register result");

  LLVM_DEBUG(dbgs() << "Found a candidate for VL reduction: " << MI << "\n");
  return true;
}

std::optional<MachineOperand>
RISCVVLOptimizer::getMinimumVLForUser(MachineOperand &UserOp) {
  const MachineInstr &UserMI = *UserOp.getParent();
  const MCInstrDesc &Desc = UserMI.getDesc();

  if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags)) {
    LLVM_DEBUG(dbgs() << "    Abort due to lack of VL, assume that"
                         " use VLMAX\n");
    return std::nullopt;
  }

  // Instructions like reductions may use a vector register as a scalar
  // register. In this case, we should treat it as only reading the first lane.
  if (isVectorOpUsedAsScalarOp(UserOp)) {
    [[maybe_unused]] Register R = UserOp.getReg();
    [[maybe_unused]] const TargetRegisterClass *RC = MRI->getRegClass(R);
    assert(RISCV::VRRegClass.hasSubClassEq(RC) &&
           "Expect LMUL 1 register class for vector as scalar operands!");
    LLVM_DEBUG(dbgs() << "    Used this operand as a scalar operand\n");

    return MachineOperand::CreateImm(1);
  }

  unsigned VLOpNum = RISCVII::getVLOpNum(Desc);
  const MachineOperand &VLOp = UserMI.getOperand(VLOpNum);
  // Looking for an immediate or a register VL that isn't X0.
  assert((!VLOp.isReg() || VLOp.getReg() != RISCV::X0) &&
         "Did not expect X0 VL");

  // If we know the demanded VL of UserMI, then we can reduce the VL it
  // requires.
  if (DemandedVLs.contains(&UserMI)) {
    // We can only shrink the VL used if the elementwise result doesn't depend
    // on VL (i.e. not vredsum/viota etc.)
    if (!RISCVII::elementsDependOnVL(
            TII->get(RISCV::getRVVMCOpcode(UserMI.getOpcode())).TSFlags)) {
      const MachineOperand &DemandedVL = DemandedVLs.at(&UserMI);
      if (RISCV::isVLKnownLE(DemandedVL, VLOp))
        return DemandedVL;
    }
  }

  return VLOp;
}

std::optional<MachineOperand> RISCVVLOptimizer::checkUsers(MachineInstr &MI) {
  std::optional<MachineOperand> CommonVL;
  for (auto &UserOp : MRI->use_operands(MI.getOperand(0).getReg())) {
    const MachineInstr &UserMI = *UserOp.getParent();
    LLVM_DEBUG(dbgs() << "  Checking user: " << UserMI << "\n");
    if (mayReadPastVL(UserMI)) {
      LLVM_DEBUG(dbgs() << "    Abort because used by unsafe instruction\n");
      return std::nullopt;
    }

    // If used as a passthru, elements past VL will be read.
    if (UserOp.isTied()) {
      LLVM_DEBUG(dbgs() << "    Abort because user used as tied operand\n");
      return std::nullopt;
    }

    auto VLOp = getMinimumVLForUser(UserOp);
    if (!VLOp)
      return std::nullopt;

    // Use the largest VL among all the users. If we cannot determine this
    // statically, then we cannot optimize the VL.
    if (!CommonVL || RISCV::isVLKnownLE(*CommonVL, *VLOp)) {
      CommonVL = *VLOp;
      LLVM_DEBUG(dbgs() << "    User VL is: " << VLOp << "\n");
    } else if (!RISCV::isVLKnownLE(*VLOp, *CommonVL)) {
      LLVM_DEBUG(dbgs() << "    Abort because cannot determine a common VL\n");
      return std::nullopt;
    }

    if (!RISCVII::hasSEWOp(UserMI.getDesc().TSFlags)) {
      LLVM_DEBUG(dbgs() << "    Abort due to lack of SEW operand\n");
      return std::nullopt;
    }

    std::optional<OperandInfo> ConsumerInfo = getOperandInfo(UserOp, MRI);
    std::optional<OperandInfo> ProducerInfo =
        getOperandInfo(MI.getOperand(0), MRI);
    if (!ConsumerInfo || !ProducerInfo) {
      LLVM_DEBUG(dbgs() << "    Abort due to unknown operand information.\n");
      LLVM_DEBUG(dbgs() << "      ConsumerInfo is: " << ConsumerInfo << "\n");
      LLVM_DEBUG(dbgs() << "      ProducerInfo is: " << ProducerInfo << "\n");
      return std::nullopt;
    }

    // If the operand is used as a scalar operand, then the EEW must be
    // compatible. Otherwise, the EMUL *and* EEW must be compatible.
    bool IsVectorOpUsedAsScalarOp = isVectorOpUsedAsScalarOp(UserOp);
    if ((IsVectorOpUsedAsScalarOp &&
         !OperandInfo::EEWAreEqual(*ConsumerInfo, *ProducerInfo)) ||
        (!IsVectorOpUsedAsScalarOp &&
         !OperandInfo::EMULAndEEWAreEqual(*ConsumerInfo, *ProducerInfo))) {
      LLVM_DEBUG(
          dbgs()
          << "    Abort due to incompatible information for EMUL or EEW.\n");
      LLVM_DEBUG(dbgs() << "      ConsumerInfo is: " << ConsumerInfo << "\n");
      LLVM_DEBUG(dbgs() << "      ProducerInfo is: " << ProducerInfo << "\n");
      return std::nullopt;
    }
  }

  return CommonVL;
}

bool RISCVVLOptimizer::tryReduceVL(MachineInstr &MI) {
  LLVM_DEBUG(dbgs() << "Trying to reduce VL for " << MI << "\n");

  unsigned VLOpNum = RISCVII::getVLOpNum(MI.getDesc());
  MachineOperand &VLOp = MI.getOperand(VLOpNum);

  // If the VL is 1, then there is no need to reduce it. This is an
  // optimization, not needed to preserve correctness.
  if (VLOp.isImm() && VLOp.getImm() == 1) {
    LLVM_DEBUG(dbgs() << "  Abort due to VL == 1, no point in reducing.\n");
    return false;
  }

  auto CommonVL = DemandedVLs[&MI];
  if (!CommonVL)
    return false;

  assert((CommonVL->isImm() || CommonVL->getReg().isVirtual()) &&
         "Expected VL to be an Imm or virtual Reg");

  if (!RISCV::isVLKnownLE(*CommonVL, VLOp)) {
    LLVM_DEBUG(dbgs() << "    Abort due to CommonVL not <= VLOp.\n");
    return false;
  }

  if (CommonVL->isIdenticalTo(VLOp)) {
    LLVM_DEBUG(
        dbgs() << "    Abort due to CommonVL == VLOp, no point in reducing.\n");
    return false;
  }

  if (CommonVL->isImm()) {
    LLVM_DEBUG(dbgs() << "  Reduce VL from " << VLOp << " to "
                      << CommonVL->getImm() << " for " << MI << "\n");
    VLOp.ChangeToImmediate(CommonVL->getImm());
    return true;
  }
  const MachineInstr *VLMI = MRI->getVRegDef(CommonVL->getReg());
  if (!MDT->dominates(VLMI, &MI))
    return false;
  LLVM_DEBUG(
      dbgs() << "  Reduce VL from " << VLOp << " to "
             << printReg(CommonVL->getReg(), MRI->getTargetRegisterInfo())
             << " for " << MI << "\n");

  // All our checks passed. We can reduce VL.
  VLOp.ChangeToRegister(CommonVL->getReg(), false);
  return true;
}

bool RISCVVLOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();


  // For each instruction that defines a vector, compute what VL its
  // downstream users demand.
  for (MachineBasicBlock *MBB : post_order(&MF)) {
    // Avoid unreachable blocks as they have degenerate dominance
    assert(MDT->isReachableFromEntry(MBB));
    for (MachineInstr &MI : reverse(*MBB)) {
      if (!isCandidate(MI))
        continue;
      if (auto DemandedVL = checkUsers(MI))
	DemandedVLs.insert({&MI, *DemandedVL});
    }
  }

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    // Then go through and see if we can reduce the VL of any instructions to
    // only what's demanded.
    for (auto &MI : MBB) {
      if (!isCandidate(MI))
        continue;
      if (!tryReduceVL(MI))
        continue;
      MadeChange = true;
    }
  }

  return MadeChange;
}
