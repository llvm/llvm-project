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
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "riscv-vl-optimizer"

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

  StringRef getPassName() const override { return "RISC-V VL Optimizer"; }

private:
  bool tryReduceVL(MachineInstr &MI);
  bool isCandidate(const MachineInstr &MI) const;
};

} // end anonymous namespace

char RISCVVLOptimizer::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVVLOptimizer, DEBUG_TYPE, "RISC-V VL Optimizer",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVVLOptimizer, DEBUG_TYPE, "RISC-V VL Optimizer", false,
                    false)

FunctionPass *llvm::createRISCVVLOptimizerPass() {
  return new RISCVVLOptimizer();
}

/// Return true if R is a physical or virtual vector register, false otherwise.
static bool isVectorRegClass(Register R, const MachineRegisterInfo *MRI) {
  if (R.isPhysical())
    return RISCV::VRRegClass.contains(R);
  const TargetRegisterClass *RC = MRI->getRegClass(R);
  return RISCV::VRRegClass.hasSubClassEq(RC) ||
         RISCV::VRM2RegClass.hasSubClassEq(RC) ||
         RISCV::VRM4RegClass.hasSubClassEq(RC) ||
         RISCV::VRM8RegClass.hasSubClassEq(RC);
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
  std::pair<unsigned, bool> EMUL;

  unsigned Log2EEW;

  OperandInfo(RISCVII::VLMUL EMUL, unsigned Log2EEW)
      : S(State::Known), EMUL(RISCVVType::decodeVLMUL(EMUL)), Log2EEW(Log2EEW) {
  }

  OperandInfo(std::pair<unsigned, bool> EMUL, unsigned Log2EEW)
      : S(State::Known), EMUL(EMUL), Log2EEW(Log2EEW) {}

  OperandInfo(State S) : S(S) {
    assert(S != State::Known &&
           "This constructor may only be used to construct "
           "an Unknown OperandInfo");
  }

  bool isUnknown() const { return S == State::Unknown; }
  bool isKnown() const { return S == State::Known; }

  static bool EMULAndEEWAreEqual(const OperandInfo &A, const OperandInfo &B) {
    assert(A.isKnown() && B.isKnown() && "Both operands must be known");
    return A.Log2EEW == B.Log2EEW && A.EMUL.first == B.EMUL.first &&
           A.EMUL.second == B.EMUL.second;
  }

  void print(raw_ostream &OS) const {
    if (isUnknown()) {
      OS << "Unknown";
      return;
    }
    OS << "EMUL: ";
    if (EMUL.second)
      OS << "m";
    OS << "f" << EMUL.first;
    OS << ", EEW: " << (1 << Log2EEW);
  }
};

static raw_ostream &operator<<(raw_ostream &OS, const OperandInfo &OI) {
  OI.print(OS);
  return OS;
}

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

/// Return the RISCVII::VLMUL that is VLMul / 2.
/// Precondition: VLMul is not LMUL_RESERVED or LMUL_MF8.
static RISCVII::VLMUL halfVLMUL(RISCVII::VLMUL VLMul) {
  switch (VLMul) {
  case RISCVII::VLMUL::LMUL_F4:
    return RISCVII::VLMUL::LMUL_F8;
  case RISCVII::VLMUL::LMUL_F2:
    return RISCVII::VLMUL::LMUL_F4;
  case RISCVII::VLMUL::LMUL_1:
    return RISCVII::VLMUL::LMUL_F2;
  case RISCVII::VLMUL::LMUL_2:
    return RISCVII::VLMUL::LMUL_1;
  case RISCVII::VLMUL::LMUL_4:
    return RISCVII::VLMUL::LMUL_2;
  case RISCVII::VLMUL::LMUL_8:
    return RISCVII::VLMUL::LMUL_4;
  case RISCVII::VLMUL::LMUL_F8:
  default:
    llvm_unreachable("Could not divide VLMul by 2");
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

static bool isOpN(const MachineOperand &MO, unsigned OpN) {
  const MachineInstr &MI = *MO.getParent();
  bool HasPassthru = RISCVII::isFirstDefTiedToFirstUse(MI.getDesc());

  if (HasPassthru)
    return MO.getOperandNo() == OpN + 1;

  return MO.getOperandNo() == OpN;
}

/// An index segment load or store operand has the form v.*seg<nf>ei<eeew>.v.
/// Data has EEW=SEW, EMUL=LMUL. Index has EEW=<eew>, EMUL=(EEW/SEW)*LMUL. LMUL
/// and SEW comes from TSFlags of MI.
static OperandInfo getIndexSegmentLoadStoreOperandInfo(unsigned Log2EEW,
                                                       const MachineInstr &MI,
                                                       const MachineOperand &MO,
                                                       bool IsLoad) {
  // Operand 0 is data register
  // Data vector register group has EEW=SEW, EMUL=LMUL.
  if (MO.getOperandNo() == 0) {
    RISCVII::VLMUL MIVLMul = RISCVII::getLMul(MI.getDesc().TSFlags);
    unsigned MILog2SEW =
        MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();
    return OperandInfo(MIVLMul, MILog2SEW);
  }

  // Operand 2 is index vector register
  // v.*seg<nf>ei<eeew>.v
  // Index vector register group has EEW=<eew>, EMUL=(EEW/SEW)*LMUL.
  if (isOpN(MO, 2))
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(Log2EEW, MI), Log2EEW);

  llvm_unreachable("Could not get OperandInfo for non-vector register of an "
                   "indexed segment load or store instruction");
}

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

  return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(Log2EEW, MI), Log2EEW);
}

/// Check whether MO is a mask operand of MI.
static bool isMaskOperand(const MachineInstr &MI, const MachineOperand &MO,
                          const MachineRegisterInfo *MRI) {

  if (!MO.isReg() || !isVectorRegClass(MO.getReg(), MRI))
    return false;

  const MCInstrDesc &Desc = MI.getDesc();
  return Desc.operands()[MO.getOperandNo()].RegClass == RISCV::VMV0RegClassID;
}

/// Return the OperandInfo for MO, which is an operand of MI.
static OperandInfo getOperandInfo(const MachineInstr &MI,
                                  const MachineOperand &MO,
                                  const MachineRegisterInfo *MRI) {
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
    return OperandInfo(OperandInfo::State::Unknown);

  bool IsMODef = MO.getOperandNo() == 0;
  bool IsOp1 = isOpN(MO, 1);
  bool IsOp2 = isOpN(MO, 2);
  bool IsOp3 = isOpN(MO, 3);

  // All mask operands have EEW=1, EMUL=(EEW/SEW)*LMUL
  if (isMaskOperand(MI, MO, MRI))
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);

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

  // 7. Vector Loads and Stores
  // 7.4. Vector Unit-Stride Instructions
  // 7.5. Vector Strided Instructions
  // 7.7. Unit-stride Fault-Only-First Loads
  /// Dest EEW encoded in the instruction and EMUL=(EEW/SEW)*LMUL
  case RISCV::VLE8_V:
  case RISCV::VSE8_V:
  case RISCV::VLM_V:
  case RISCV::VSM_V:
  case RISCV::VLSE8_V:
  case RISCV::VSSE8_V:
  case RISCV::VLE8FF_V:
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(3, MI), 3);
  case RISCV::VLE16_V:
  case RISCV::VSE16_V:
  case RISCV::VLSE16_V:
  case RISCV::VSSE16_V:
  case RISCV::VLE16FF_V:
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(4, MI), 4);
  case RISCV::VLE32_V:
  case RISCV::VSE32_V:
  case RISCV::VLSE32_V:
  case RISCV::VSSE32_V:
  case RISCV::VLE32FF_V:
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(5, MI), 5);
  case RISCV::VLE64_V:
  case RISCV::VSE64_V:
  case RISCV::VLSE64_V:
  case RISCV::VSSE64_V:
  case RISCV::VLE64FF_V:
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(6, MI), 6);

  // 7.6. Vector Indexed Instructions
  // Data EEW=SEW, EMUL=LMUL. Index EEW=<eew> and EMUL=(EEW/SEW)*LMUL
  case RISCV::VLUXEI8_V:
  case RISCV::VLOXEI8_V:
  case RISCV::VSUXEI8_V:
  case RISCV::VSOXEI8_V:
    if (MO.getOperandNo() == 0)
      return OperandInfo(MIVLMul, MILog2SEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(3, MI), 3);
  case RISCV::VLUXEI16_V:
  case RISCV::VLOXEI16_V:
  case RISCV::VSUXEI16_V:
  case RISCV::VSOXEI16_V:
    if (MO.getOperandNo() == 0)
      return OperandInfo(MIVLMul, MILog2SEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(4, MI), 4);
  case RISCV::VLUXEI32_V:
  case RISCV::VLOXEI32_V:
  case RISCV::VSUXEI32_V:
  case RISCV::VSOXEI32_V:
    if (MO.getOperandNo() == 0)
      return OperandInfo(MIVLMul, MILog2SEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(5, MI), 5);
  case RISCV::VLUXEI64_V:
  case RISCV::VLOXEI64_V:
  case RISCV::VSUXEI64_V:
  case RISCV::VSOXEI64_V:
    if (MO.getOperandNo() == 0)
      return OperandInfo(MIVLMul, MILog2SEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(6, MI), 6);

  // 7.8. Vector Load/Store Segment Instructions
  // 7.8.1. Vector Unit-Stride Segment Loads and Stores
  // v.*seg<nf>e<eew>.*
  // EEW=eew, EMUL=LMUL
  case RISCV::VLSEG2E8_V:
  case RISCV::VLSEG2E8FF_V:
  case RISCV::VLSEG3E8_V:
  case RISCV::VLSEG3E8FF_V:
  case RISCV::VLSEG4E8_V:
  case RISCV::VLSEG4E8FF_V:
  case RISCV::VLSEG5E8_V:
  case RISCV::VLSEG5E8FF_V:
  case RISCV::VLSEG6E8_V:
  case RISCV::VLSEG6E8FF_V:
  case RISCV::VLSEG7E8_V:
  case RISCV::VLSEG7E8FF_V:
  case RISCV::VLSEG8E8_V:
  case RISCV::VLSEG8E8FF_V:
  case RISCV::VSSEG2E8_V:
  case RISCV::VSSEG3E8_V:
  case RISCV::VSSEG4E8_V:
  case RISCV::VSSEG5E8_V:
  case RISCV::VSSEG6E8_V:
  case RISCV::VSSEG7E8_V:
  case RISCV::VSSEG8E8_V:
    return OperandInfo(MIVLMul, 3);
  case RISCV::VLSEG2E16_V:
  case RISCV::VLSEG2E16FF_V:
  case RISCV::VLSEG3E16_V:
  case RISCV::VLSEG3E16FF_V:
  case RISCV::VLSEG4E16_V:
  case RISCV::VLSEG4E16FF_V:
  case RISCV::VLSEG5E16_V:
  case RISCV::VLSEG5E16FF_V:
  case RISCV::VLSEG6E16_V:
  case RISCV::VLSEG6E16FF_V:
  case RISCV::VLSEG7E16_V:
  case RISCV::VLSEG7E16FF_V:
  case RISCV::VLSEG8E16_V:
  case RISCV::VLSEG8E16FF_V:
  case RISCV::VSSEG2E16_V:
  case RISCV::VSSEG3E16_V:
  case RISCV::VSSEG4E16_V:
  case RISCV::VSSEG5E16_V:
  case RISCV::VSSEG6E16_V:
  case RISCV::VSSEG7E16_V:
  case RISCV::VSSEG8E16_V:
    return OperandInfo(MIVLMul, 4);
  case RISCV::VLSEG2E32_V:
  case RISCV::VLSEG2E32FF_V:
  case RISCV::VLSEG3E32_V:
  case RISCV::VLSEG3E32FF_V:
  case RISCV::VLSEG4E32_V:
  case RISCV::VLSEG4E32FF_V:
  case RISCV::VLSEG5E32_V:
  case RISCV::VLSEG5E32FF_V:
  case RISCV::VLSEG6E32_V:
  case RISCV::VLSEG6E32FF_V:
  case RISCV::VLSEG7E32_V:
  case RISCV::VLSEG7E32FF_V:
  case RISCV::VLSEG8E32_V:
  case RISCV::VLSEG8E32FF_V:
  case RISCV::VSSEG2E32_V:
  case RISCV::VSSEG3E32_V:
  case RISCV::VSSEG4E32_V:
  case RISCV::VSSEG5E32_V:
  case RISCV::VSSEG6E32_V:
  case RISCV::VSSEG7E32_V:
  case RISCV::VSSEG8E32_V:
    return OperandInfo(MIVLMul, 5);
  case RISCV::VLSEG2E64_V:
  case RISCV::VLSEG2E64FF_V:
  case RISCV::VLSEG3E64_V:
  case RISCV::VLSEG3E64FF_V:
  case RISCV::VLSEG4E64_V:
  case RISCV::VLSEG4E64FF_V:
  case RISCV::VLSEG5E64_V:
  case RISCV::VLSEG5E64FF_V:
  case RISCV::VLSEG6E64_V:
  case RISCV::VLSEG6E64FF_V:
  case RISCV::VLSEG7E64_V:
  case RISCV::VLSEG7E64FF_V:
  case RISCV::VLSEG8E64_V:
  case RISCV::VLSEG8E64FF_V:
  case RISCV::VSSEG2E64_V:
  case RISCV::VSSEG3E64_V:
  case RISCV::VSSEG4E64_V:
  case RISCV::VSSEG5E64_V:
  case RISCV::VSSEG6E64_V:
  case RISCV::VSSEG7E64_V:
  case RISCV::VSSEG8E64_V:
    return OperandInfo(MIVLMul, 6);

  // 7.8.2. Vector Strided Segment Loads and Stores
  case RISCV::VLSSEG2E8_V:
  case RISCV::VLSSEG3E8_V:
  case RISCV::VLSSEG4E8_V:
  case RISCV::VLSSEG5E8_V:
  case RISCV::VLSSEG6E8_V:
  case RISCV::VLSSEG7E8_V:
  case RISCV::VLSSEG8E8_V:
  case RISCV::VSSSEG2E8_V:
  case RISCV::VSSSEG3E8_V:
  case RISCV::VSSSEG4E8_V:
  case RISCV::VSSSEG5E8_V:
  case RISCV::VSSSEG6E8_V:
  case RISCV::VSSSEG7E8_V:
  case RISCV::VSSSEG8E8_V:
    return OperandInfo(MIVLMul, 3);
  case RISCV::VLSSEG2E16_V:
  case RISCV::VLSSEG3E16_V:
  case RISCV::VLSSEG4E16_V:
  case RISCV::VLSSEG5E16_V:
  case RISCV::VLSSEG6E16_V:
  case RISCV::VLSSEG7E16_V:
  case RISCV::VLSSEG8E16_V:
  case RISCV::VSSSEG2E16_V:
  case RISCV::VSSSEG3E16_V:
  case RISCV::VSSSEG4E16_V:
  case RISCV::VSSSEG5E16_V:
  case RISCV::VSSSEG6E16_V:
  case RISCV::VSSSEG7E16_V:
  case RISCV::VSSSEG8E16_V:
    return OperandInfo(MIVLMul, 4);
  case RISCV::VLSSEG2E32_V:
  case RISCV::VLSSEG3E32_V:
  case RISCV::VLSSEG4E32_V:
  case RISCV::VLSSEG5E32_V:
  case RISCV::VLSSEG6E32_V:
  case RISCV::VLSSEG7E32_V:
  case RISCV::VLSSEG8E32_V:
  case RISCV::VSSSEG2E32_V:
  case RISCV::VSSSEG3E32_V:
  case RISCV::VSSSEG4E32_V:
  case RISCV::VSSSEG5E32_V:
  case RISCV::VSSSEG6E32_V:
  case RISCV::VSSSEG7E32_V:
  case RISCV::VSSSEG8E32_V:
    return OperandInfo(MIVLMul, 5);
  case RISCV::VLSSEG2E64_V:
  case RISCV::VLSSEG3E64_V:
  case RISCV::VLSSEG4E64_V:
  case RISCV::VLSSEG5E64_V:
  case RISCV::VLSSEG6E64_V:
  case RISCV::VLSSEG7E64_V:
  case RISCV::VLSSEG8E64_V:
  case RISCV::VSSSEG2E64_V:
  case RISCV::VSSSEG3E64_V:
  case RISCV::VSSSEG4E64_V:
  case RISCV::VSSSEG5E64_V:
  case RISCV::VSSSEG6E64_V:
  case RISCV::VSSSEG7E64_V:
  case RISCV::VSSSEG8E64_V:
    return OperandInfo(MIVLMul, 6);

  // 7.8.3. Vector Indexed Segment Loads and Stores
  case RISCV::VLUXSEG2EI8_V:
  case RISCV::VLUXSEG3EI8_V:
  case RISCV::VLUXSEG4EI8_V:
  case RISCV::VLUXSEG5EI8_V:
  case RISCV::VLUXSEG6EI8_V:
  case RISCV::VLUXSEG7EI8_V:
  case RISCV::VLUXSEG8EI8_V:
  case RISCV::VLOXSEG2EI8_V:
  case RISCV::VLOXSEG3EI8_V:
  case RISCV::VLOXSEG4EI8_V:
  case RISCV::VLOXSEG5EI8_V:
  case RISCV::VLOXSEG6EI8_V:
  case RISCV::VLOXSEG7EI8_V:
  case RISCV::VLOXSEG8EI8_V:
    return getIndexSegmentLoadStoreOperandInfo(3, MI, MO, /* IsLoad */ true);
  case RISCV::VSUXSEG2EI8_V:
  case RISCV::VSUXSEG3EI8_V:
  case RISCV::VSUXSEG4EI8_V:
  case RISCV::VSUXSEG5EI8_V:
  case RISCV::VSUXSEG6EI8_V:
  case RISCV::VSUXSEG7EI8_V:
  case RISCV::VSUXSEG8EI8_V:
  case RISCV::VSOXSEG2EI8_V:
  case RISCV::VSOXSEG3EI8_V:
  case RISCV::VSOXSEG4EI8_V:
  case RISCV::VSOXSEG5EI8_V:
  case RISCV::VSOXSEG6EI8_V:
  case RISCV::VSOXSEG7EI8_V:
  case RISCV::VSOXSEG8EI8_V:
    return getIndexSegmentLoadStoreOperandInfo(3, MI, MO, /* IsLoad */ false);
  case RISCV::VLUXSEG2EI16_V:
  case RISCV::VLUXSEG3EI16_V:
  case RISCV::VLUXSEG4EI16_V:
  case RISCV::VLUXSEG5EI16_V:
  case RISCV::VLUXSEG6EI16_V:
  case RISCV::VLUXSEG7EI16_V:
  case RISCV::VLUXSEG8EI16_V:
  case RISCV::VLOXSEG2EI16_V:
  case RISCV::VLOXSEG3EI16_V:
  case RISCV::VLOXSEG4EI16_V:
  case RISCV::VLOXSEG5EI16_V:
  case RISCV::VLOXSEG6EI16_V:
  case RISCV::VLOXSEG7EI16_V:
  case RISCV::VLOXSEG8EI16_V:
    return getIndexSegmentLoadStoreOperandInfo(4, MI, MO, /* IsLoad */ true);
  case RISCV::VSUXSEG2EI16_V:
  case RISCV::VSUXSEG3EI16_V:
  case RISCV::VSUXSEG4EI16_V:
  case RISCV::VSUXSEG5EI16_V:
  case RISCV::VSUXSEG6EI16_V:
  case RISCV::VSUXSEG7EI16_V:
  case RISCV::VSUXSEG8EI16_V:
  case RISCV::VSOXSEG2EI16_V:
  case RISCV::VSOXSEG3EI16_V:
  case RISCV::VSOXSEG4EI16_V:
  case RISCV::VSOXSEG5EI16_V:
  case RISCV::VSOXSEG6EI16_V:
  case RISCV::VSOXSEG7EI16_V:
  case RISCV::VSOXSEG8EI16_V:
    return getIndexSegmentLoadStoreOperandInfo(4, MI, MO, /* IsLoad */ false);
  case RISCV::VLUXSEG2EI32_V:
  case RISCV::VLUXSEG3EI32_V:
  case RISCV::VLUXSEG4EI32_V:
  case RISCV::VLUXSEG5EI32_V:
  case RISCV::VLUXSEG6EI32_V:
  case RISCV::VLUXSEG7EI32_V:
  case RISCV::VLUXSEG8EI32_V:
  case RISCV::VLOXSEG2EI32_V:
  case RISCV::VLOXSEG3EI32_V:
  case RISCV::VLOXSEG4EI32_V:
  case RISCV::VLOXSEG5EI32_V:
  case RISCV::VLOXSEG6EI32_V:
  case RISCV::VLOXSEG7EI32_V:
  case RISCV::VLOXSEG8EI32_V:
    return getIndexSegmentLoadStoreOperandInfo(5, MI, MO, /* IsLoad */ true);
  case RISCV::VSUXSEG2EI32_V:
  case RISCV::VSUXSEG3EI32_V:
  case RISCV::VSUXSEG4EI32_V:
  case RISCV::VSUXSEG5EI32_V:
  case RISCV::VSUXSEG6EI32_V:
  case RISCV::VSUXSEG7EI32_V:
  case RISCV::VSUXSEG8EI32_V:
  case RISCV::VSOXSEG2EI32_V:
  case RISCV::VSOXSEG3EI32_V:
  case RISCV::VSOXSEG4EI32_V:
  case RISCV::VSOXSEG5EI32_V:
  case RISCV::VSOXSEG6EI32_V:
  case RISCV::VSOXSEG7EI32_V:
  case RISCV::VSOXSEG8EI32_V:
    return getIndexSegmentLoadStoreOperandInfo(5, MI, MO, /* IsLoad */ false);
  case RISCV::VLUXSEG2EI64_V:
  case RISCV::VLUXSEG3EI64_V:
  case RISCV::VLUXSEG4EI64_V:
  case RISCV::VLUXSEG5EI64_V:
  case RISCV::VLUXSEG6EI64_V:
  case RISCV::VLUXSEG7EI64_V:
  case RISCV::VLUXSEG8EI64_V:
  case RISCV::VLOXSEG2EI64_V:
  case RISCV::VLOXSEG3EI64_V:
  case RISCV::VLOXSEG4EI64_V:
  case RISCV::VLOXSEG5EI64_V:
  case RISCV::VLOXSEG6EI64_V:
  case RISCV::VLOXSEG7EI64_V:
  case RISCV::VLOXSEG8EI64_V:
    return getIndexSegmentLoadStoreOperandInfo(6, MI, MO, /* IsLoad */ true);
  case RISCV::VSUXSEG2EI64_V:
  case RISCV::VSUXSEG3EI64_V:
  case RISCV::VSUXSEG4EI64_V:
  case RISCV::VSUXSEG5EI64_V:
  case RISCV::VSUXSEG6EI64_V:
  case RISCV::VSUXSEG7EI64_V:
  case RISCV::VSUXSEG8EI64_V:
  case RISCV::VSOXSEG2EI64_V:
  case RISCV::VSOXSEG3EI64_V:
  case RISCV::VSOXSEG4EI64_V:
  case RISCV::VSOXSEG5EI64_V:
  case RISCV::VSOXSEG6EI64_V:
  case RISCV::VSOXSEG7EI64_V:
  case RISCV::VSOXSEG8EI64_V:
    return getIndexSegmentLoadStoreOperandInfo(6, MI, MO, /* IsLoad */ false);

  // 7.9. Vector Load/Store Whole Register Instructions
  // EMUL=nr. EEW=eew. Since in-register byte layouts are idential to in-memory
  // byte layouts, the same data is writen to destination register regardless
  // of EEW. eew is just a hint to the hardware and has not functional impact.
  // Therefore, it is be okay if we ignore eew and always use the same EEW to
  // create more optimization opportunities.
  // FIXME: Instead of using any SEW, we really ought to return the SEW in the
  // instruction and add a field to OperandInfo that says the SEW is just a hint
  // so that this optimization can use any sew to construct a ratio.
  case RISCV::VL1RE8_V:
  case RISCV::VL1RE16_V:
  case RISCV::VL1RE32_V:
  case RISCV::VL1RE64_V:
  case RISCV::VS1R_V:
    return OperandInfo(RISCVII::VLMUL::LMUL_1, 0);
  case RISCV::VL2RE8_V:
  case RISCV::VL2RE16_V:
  case RISCV::VL2RE32_V:
  case RISCV::VL2RE64_V:
  case RISCV::VS2R_V:
    return OperandInfo(RISCVII::VLMUL::LMUL_2, 0);
  case RISCV::VL4RE8_V:
  case RISCV::VL4RE16_V:
  case RISCV::VL4RE32_V:
  case RISCV::VL4RE64_V:
  case RISCV::VS4R_V:
    return OperandInfo(RISCVII::VLMUL::LMUL_4, 0);
  case RISCV::VL8RE8_V:
  case RISCV::VL8RE16_V:
  case RISCV::VL8RE32_V:
  case RISCV::VL8RE64_V:
  case RISCV::VS8R_V:
    return OperandInfo(RISCVII::VLMUL::LMUL_8, 0);

  // 11. Vector Integer Arithmetic Instructions
  // 11.1. Vector Single-Width Integer Add and Subtract
  case RISCV::VADD_VI:
  case RISCV::VADD_VV:
  case RISCV::VADD_VX:
  case RISCV::VSUB_VV:
  case RISCV::VSUB_VX:
  case RISCV::VRSUB_VI:
  case RISCV::VRSUB_VX:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.2. Vector Widening Integer Add/Subtract
  // Def uses EEW=2*SEW and EMUL=2*LMUL. Operands use EEW=SEW and EMUL=LMUL.
  case RISCV::VWADDU_VV:
  case RISCV::VWADDU_VX:
  case RISCV::VWSUBU_VV:
  case RISCV::VWSUBU_VX:
  case RISCV::VWADD_VV:
  case RISCV::VWADD_VX:
  case RISCV::VWSUB_VV:
  case RISCV::VWSUB_VX:
  case RISCV::VWSLL_VI: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
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
    bool TwoTimes = IsMODef || IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = TwoTimes ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 11.3. Vector Integer Extension
  case RISCV::VZEXT_VF2:
  case RISCV::VSEXT_VF2:
    return getIntegerExtensionOperandInfo(2, MI, MO);
  case RISCV::VZEXT_VF4:
  case RISCV::VSEXT_VF4:
    return getIntegerExtensionOperandInfo(4, MI, MO);
  case RISCV::VZEXT_VF8:
  case RISCV::VSEXT_VF8:
    return getIntegerExtensionOperandInfo(8, MI, MO);

  // 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  // EEW=SEW and EMUL=LMUL. Mask Operand EEW=1 and EMUL=(EEW/SEW)*LMUL
  case RISCV::VADC_VIM:
  case RISCV::VADC_VVM:
  case RISCV::VADC_VXM:
  case RISCV::VSBC_VVM:
  case RISCV::VSBC_VXM:
    return MO.getOperandNo() == 3
               ? OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0)
               : OperandInfo(MIVLMul, MILog2SEW);
  // Dest EEW=1 and EMUL=(EEW/SEW)*LMUL. Source EEW=SEW and EMUL=LMUL. Mask
  // operand EEW=1 and EMUL=(EEW/SEW)*LMUL
  case RISCV::VMADC_VIM:
  case RISCV::VMADC_VVM:
  case RISCV::VMADC_VXM:
  case RISCV::VMSBC_VVM:
  case RISCV::VMSBC_VXM:
    return IsMODef || MO.getOperandNo() == 3
               ? OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0)
               : OperandInfo(MIVLMul, MILog2SEW);
  // Dest EEW=1 and EMUL=(EEW/SEW)*LMUL. Source EEW=SEW and EMUL=LMUL.
  case RISCV::VMADC_VV:
  case RISCV::VMADC_VI:
  case RISCV::VMADC_VX:
  case RISCV::VMSBC_VV:
  case RISCV::VMSBC_VX:
    return IsMODef ? OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0)
                   : OperandInfo(MIVLMul, MILog2SEW);

  // 11.5. Vector Bitwise Logical Instructions
  // 11.6. Vector Single-Width Shift Instructions
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
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.7. Vector Narrowing Integer Right Shift Instructions
  // Destination EEW=SEW and EMUL=LMUL, Op 1 has EEW=2*SEW EMUL=2*LMUL. Op2 has
  // EEW=SEW EMUL=LMUL.
  case RISCV::VNSRL_WX:
  case RISCV::VNSRL_WI:
  case RISCV::VNSRL_WV:
  case RISCV::VNSRA_WI:
  case RISCV::VNSRA_WV:
  case RISCV::VNSRA_WX: {
    bool TwoTimes = IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = TwoTimes ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 11.8. Vector Integer Compare Instructions
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
    if (IsMODef)
      return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.9. Vector Integer Min/Max Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VMINU_VV:
  case RISCV::VMINU_VX:
  case RISCV::VMIN_VV:
  case RISCV::VMIN_VX:
  case RISCV::VMAXU_VV:
  case RISCV::VMAXU_VX:
  case RISCV::VMAX_VV:
  case RISCV::VMAX_VX:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.10. Vector Single-Width Integer Multiply Instructions
  // Source and Dest EEW=SEW and EMUL=LMUL.
  case RISCV::VMUL_VV:
  case RISCV::VMUL_VX:
  case RISCV::VMULH_VV:
  case RISCV::VMULH_VX:
  case RISCV::VMULHU_VV:
  case RISCV::VMULHU_VX:
  case RISCV::VMULHSU_VV:
  case RISCV::VMULHSU_VX: {
    return OperandInfo(MIVLMul, MILog2SEW);
  }
  // 11.11. Vector Integer Divide Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VDIVU_VV:
  case RISCV::VDIVU_VX:
  case RISCV::VDIV_VV:
  case RISCV::VDIV_VX:
  case RISCV::VREMU_VV:
  case RISCV::VREMU_VX:
  case RISCV::VREM_VV:
  case RISCV::VREM_VX:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.12. Vector Widening Integer Multiply Instructions
  // Source and Destination EMUL=LMUL. Destination EEW=2*SEW. Source EEW=SEW.
  case RISCV::VWMUL_VV:
  case RISCV::VWMUL_VX:
  case RISCV::VWMULSU_VV:
  case RISCV::VWMULSU_VX:
  case RISCV::VWMULU_VV:
  case RISCV::VWMULU_VX: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 11.13. Vector Single-Width Integer Multiply-Add Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VMACC_VV:
  case RISCV::VMACC_VX:
  case RISCV::VNMSAC_VV:
  case RISCV::VNMSAC_VX:
  case RISCV::VMADD_VV:
  case RISCV::VMADD_VX:
  case RISCV::VNMSUB_VV:
  case RISCV::VNMSUB_VX:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.14. Vector Widening Integer Multiply-Add Instructions
  // Destination EEW=2*SEW and EMUL=2*LMUL. Source EEW=SEW and EMUL=LMUL.
  // Even though the add is a 2*SEW addition, the operands of the add are the
  // Dest which is 2*SEW and the result of the multiply which is 2*SEW.
  case RISCV::VWMACCU_VV:
  case RISCV::VWMACCU_VX:
  case RISCV::VWMACC_VV:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCSU_VV:
  case RISCV::VWMACCSU_VX:
  case RISCV::VWMACCUS_VX: {
    // Operand 0 is destination as a def and Operand 1 is destination as a use
    // due to SSA.
    bool TwoTimes = IsMODef || IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = TwoTimes ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 11.15. Vector Integer Merge Instructions
  // EEW=SEW and EMUL=LMUL, except the mask operand has EEW=1 and EMUL=
  // (EEW/SEW)*LMUL.
  case RISCV::VMERGE_VIM:
  case RISCV::VMERGE_VVM:
  case RISCV::VMERGE_VXM:
    if (MO.getOperandNo() == 3)
      return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
    return OperandInfo(MIVLMul, MILog2SEW);

  // 11.16. Vector Integer Move Instructions
  // 12. Vector Fixed-Point Arithmetic Instructions
  // 12.1. Vector Single-Width Saturating Add and Subtract
  // 12.2. Vector Single-Width Averaging Add and Subtract
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
    return OperandInfo(MIVLMul, MILog2SEW);

  // 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
  // Destination EEW=2*SEW and EMUL=2*EMUL. Source EEW=SEW and EMUL=LMUL.
  case RISCV::VSMUL_VV:
  case RISCV::VSMUL_VX: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 12.4. Vector Single-Width Scaling Shift Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VSSRL_VI:
  case RISCV::VSSRL_VV:
  case RISCV::VSSRL_VX:
  case RISCV::VSSRA_VI:
  case RISCV::VSSRA_VV:
  case RISCV::VSSRA_VX:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 12.5. Vector Narrowing Fixed-Point Clip Instructions
  // Destination and Op1 EEW=SEW and EMUL=LMUL. Op2 EEW=2*SEW and EMUL=2*LMUL
  case RISCV::VNCLIPU_WI:
  case RISCV::VNCLIPU_WV:
  case RISCV::VNCLIPU_WX:
  case RISCV::VNCLIP_WI:
  case RISCV::VNCLIP_WV:
  case RISCV::VNCLIP_WX: {
    bool TwoTimes = !IsMODef && IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = TwoTimes ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 13. Vector Floating-Point Instructions
  // 13.2. Vector Single-Width Floating-Point Add/Subtract Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VFADD_VF:
  case RISCV::VFADD_VV:
  case RISCV::VFSUB_VF:
  case RISCV::VFSUB_VV:
  case RISCV::VFRSUB_VF:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 13.3. Vector Widening Floating-Point Add/Subtract Instructions
  // Dest EEW=2*SEW and EMUL=2*LMUL. Source EEW=SEW and EMUL=LMUL.
  case RISCV::VFWADD_VV:
  case RISCV::VFWADD_VF:
  case RISCV::VFWSUB_VV:
  case RISCV::VFWSUB_VF: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // Dest and Op1 EEW=2*SEW and EMUL=2*LMUL. Op2 EEW=SEW and EMUL=LMUL.
  case RISCV::VFWADD_WF:
  case RISCV::VFWADD_WV:
  case RISCV::VFWSUB_WF:
  case RISCV::VFWSUB_WV: {
    bool TwoTimes = IsMODef || IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = TwoTimes ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 13.4. Vector Single-Width Floating-Point Multiply/Divide Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VFMUL_VF:
  case RISCV::VFMUL_VV:
  case RISCV::VFDIV_VF:
  case RISCV::VFDIV_VV:
  case RISCV::VFRDIV_VF:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 13.5. Vector Widening Floating-Point Multiply
  case RISCV::VFWMUL_VF:
  case RISCV::VFWMUL_VV:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 13.6. Vector Single-Width Floating-Point Fused Multiply-Add Instructions
  // EEW=SEW. EMUL=LMUL.
  // TODO: FMA instructions reads 3 registers but MC layer only reads 2
  // registers since its missing that the output operand should be part of the
  // input operand list.
  case RISCV::VFMACC_VF:
  case RISCV::VFMACC_VV:
  case RISCV::VFNMACC_VF:
  case RISCV::VFNMACC_VV:
  case RISCV::VFMSAC_VF:
  case RISCV::VFMSAC_VV:
  case RISCV::VFNMSAC_VF:
  case RISCV::VFNMSAC_VV:
  case RISCV::VFMADD_VF:
  case RISCV::VFMADD_VV:
  case RISCV::VFNMADD_VF:
  case RISCV::VFNMADD_VV:
  case RISCV::VFMSUB_VF:
  case RISCV::VFMSUB_VV:
  case RISCV::VFNMSUB_VF:
  case RISCV::VFNMSUB_VV:
    return OperandInfo(OperandInfo::State::Unknown);

  // 13.7. Vector Widening Floating-Point Fused Multiply-Add Instructions
  // Dest EEW=2*SEW and EMUL=2*LMUL. Source EEW=SEW EMUL=LMUL.
  case RISCV::VFWMACC_VF:
  case RISCV::VFWMACC_VV:
  case RISCV::VFWNMACC_VF:
  case RISCV::VFWNMACC_VV:
  case RISCV::VFWMSAC_VF:
  case RISCV::VFWMSAC_VV:
  case RISCV::VFWNMSAC_VF:
  case RISCV::VFWNMSAC_VV: {
    // Operand 0 is destination as a def and Operand 1 is destination as a use
    // due to SSA.
    bool TwoTimes = IsMODef || IsOp1;
    unsigned Log2EEW = TwoTimes ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = TwoTimes ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 13.8. Vector Floating-Point Square-Root Instruction
  // 13.9. Vector Floating-Point Reciprocal Square-Root Estimate Instruction
  // 13.10. Vector Floating-Point Reciprocal Estimate Instruction
  // 13.11. Vector Floating-Point MIN/MAX Instructions
  // 13.12. Vector Floating-Point Sign-Injection Instructions
  // 13.14. Vector Floating-Point Classify Instruction
  // 13.16. Vector Floating-Point Move Instruction
  // 13.17. Single-Width Floating-Point/Integer Type-Convert Instructions
  // EEW=SEW. EMUL=LMUL.
  case RISCV::VFSQRT_V:
  case RISCV::VFRSQRT7_V:
  case RISCV::VFREC7_V:
  case RISCV::VFMIN_VF:
  case RISCV::VFMIN_VV:
  case RISCV::VFMAX_VF:
  case RISCV::VFMAX_VV:
  case RISCV::VFSGNJ_VF:
  case RISCV::VFSGNJ_VV:
  case RISCV::VFSGNJN_VV:
  case RISCV::VFSGNJN_VF:
  case RISCV::VFSGNJX_VF:
  case RISCV::VFSGNJX_VV:
  case RISCV::VFCLASS_V:
  case RISCV::VFMV_V_F:
  case RISCV::VFCVT_XU_F_V:
  case RISCV::VFCVT_X_F_V:
  case RISCV::VFCVT_RTZ_XU_F_V:
  case RISCV::VFCVT_RTZ_X_F_V:
  case RISCV::VFCVT_F_XU_V:
  case RISCV::VFCVT_F_X_V:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 13.13. Vector Floating-Point Compare Instructions
  // Dest EEW=1 and EMUL=(EEW/SEW)*LMUL. Source EEW=SEW EMUL=LMUL.
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
    if (IsMODef)
      return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
    return OperandInfo(MIVLMul, MILog2SEW);

  // 13.15. Vector Floating-Point Merge Instruction
  // EEW=SEW and EMUL=LMUL, except the mask operand has EEW=1 and EMUL=
  // (EEW/SEW)*LMUL.
  case RISCV::VFMERGE_VFM:
    if (IsOp3)
      return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
    return OperandInfo(MIVLMul, MILog2SEW);

  // 13.18. Widening Floating-Point/Integer Type-Convert Instructions
  // Dest EEW=2*SEW and EMUL=2*LMUL. Source EEW=SEW and EMUL=LMUL.
  case RISCV::VFWCVT_XU_F_V:
  case RISCV::VFWCVT_X_F_V:
  case RISCV::VFWCVT_RTZ_XU_F_V:
  case RISCV::VFWCVT_RTZ_X_F_V:
  case RISCV::VFWCVT_F_XU_V:
  case RISCV::VFWCVT_F_X_V:
  case RISCV::VFWCVT_F_F_V: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 13.19. Narrowing Floating-Point/Integer Type-Convert Instructions
  // EMUL=LMUL. Dest EEW=SEW/2. Source EEW=SEW EMUL=LMUL.
  case RISCV::VFNCVT_XU_F_W:
  case RISCV::VFNCVT_X_F_W:
  case RISCV::VFNCVT_RTZ_XU_F_W:
  case RISCV::VFNCVT_RTZ_X_F_W:
  case RISCV::VFNCVT_F_XU_W:
  case RISCV::VFNCVT_F_X_W:
  case RISCV::VFNCVT_F_F_W:
  case RISCV::VFNCVT_ROD_F_F_W: {
    unsigned Log2EEW = IsMODef ? MILog2SEW - 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? halfVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 14. Vector Reduction Operations
  // 14.1. Vector Single-Width Integer Reduction Instructions
  // We need to return Unknown since only element 0 of reduction is valid but it
  // was generated by reducing over all of the input elements. There are 3
  // vector sources for reductions. One for scalar, one for tail value, and one
  // for the elements to reduce over. Only the one with the elements to reduce
  // over obeys VL. The other two only read element 0 from the register.
  case RISCV::VREDAND_VS:
  case RISCV::VREDMAX_VS:
  case RISCV::VREDMAXU_VS:
  case RISCV::VREDMIN_VS:
  case RISCV::VREDMINU_VS:
  case RISCV::VREDOR_VS:
  case RISCV::VREDSUM_VS:
  case RISCV::VREDXOR_VS:
    return OperandInfo(OperandInfo::State::Unknown);

  // 14.2. Vector Widening Integer Reduction Instructions
  // Dest EEW=2*SEW and EMUL=2*LMUL. Source EEW=SEW EMUL=LMUL. Source is zero
  // extended to 2*SEW in order to generate 2*SEW Dest.
  case RISCV::VWREDSUM_VS:
  case RISCV::VWREDSUMU_VS: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }
  // 14.3. Vector Single-Width Floating-Point Reduction Instructions
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VFREDMAX_VS:
  case RISCV::VFREDMIN_VS:
  case RISCV::VFREDOSUM_VS:
  case RISCV::VFREDUSUM_VS:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 14.4. Vector Widening Floating-Point Reduction Instructions
  // Source EEW=SEW and EMUL=LMUL. Dest EEW=2*SEW and EMUL=2*LMUL.
  case RISCV::VFWREDOSUM_VS:
  case RISCV::VFWREDUSUM_VS: {
    unsigned Log2EEW = IsMODef ? MILog2SEW + 1 : MILog2SEW;
    RISCVII::VLMUL EMUL = IsMODef ? twoTimesVLMUL(MIVLMul) : MIVLMul;
    return OperandInfo(EMUL, Log2EEW);
  }

  // 15. Vector Mask Instructions
  // 15.2. Vector count population in mask vcpop.m
  // 15.3. vfirst find-first-set mask bit
  // 15.4. vmsbf.m set-before-first mask bit
  // 15.6. vmsof.m set-only-first mask bit
  // EEW=1 and EMUL= (EEW/SEW)*LMUL
  case RISCV::VMAND_MM:
  case RISCV::VMNAND_MM:
  case RISCV::VMANDN_MM:
  case RISCV::VMXOR_MM:
  case RISCV::VMOR_MM:
  case RISCV::VMNOR_MM:
  case RISCV::VMORN_MM:
  case RISCV::VMXNOR_MM:
  case RISCV::VCPOP_M:
  case RISCV::VFIRST_M:
  case RISCV::VMSBF_M:
  case RISCV::VMSIF_M:
  case RISCV::VMSOF_M: {
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(0, MI), 0);
  }
  // 15.8. Vector Iota Instruction
  // Dest and Op1 EEW=SEW and EMUL=LMUL. Op2 EEW=1 and EMUL(EEW/SEW)*LMUL.
  case RISCV::VIOTA_M: {
    bool IsDefOrOp1 = IsMODef || IsOp1;
    unsigned Log2EEW = IsDefOrOp1 ? 0 : MILog2SEW;
    if (IsDefOrOp1)
      return OperandInfo(MIVLMul, Log2EEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(MILog2SEW, MI), Log2EEW);
  }
  // 15.9. Vector Element Index Instruction
  // Dest EEW=SEW EMUL=LMUL. Mask Operand EEW=1 and EMUL(EEW/SEW)*LMUL.
  case RISCV::VID_V: {
    unsigned Log2EEW = IsMODef ? MILog2SEW : 0;
    if (IsMODef)
      return OperandInfo(MIVLMul, Log2EEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(Log2EEW, MI), Log2EEW);
  }
  // 16. Vector Permutation Instructions
  // 16.1. Integer Scalar Move Instructions
  // 16.2. Floating-Point Scalar Move Instructions
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VMV_X_S:
  case RISCV::VMV_S_X:
  case RISCV::VFMV_F_S:
  case RISCV::VFMV_S_F:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 16.3. Vector Slide Instructions
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VSLIDEUP_VI:
  case RISCV::VSLIDEUP_VX:
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDE1UP_VX:
  case RISCV::VFSLIDE1UP_VF:
  case RISCV::VSLIDE1DOWN_VX:
  case RISCV::VFSLIDE1DOWN_VF:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 16.4. Vector Register Gather Instructions
  // EMUL=LMUL. EEW=SEW. For mask operand, EMUL=1 and EEW=1.
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
    return OperandInfo(MIVLMul, MILog2SEW);
  // Destination EMUL=LMUL and EEW=SEW. Op2 EEW=SEW and EMUL=LMUL. Op1 EEW=16
  // and EMUL=(16/SEW)*LMUL.
  case RISCV::VRGATHEREI16_VV: {
    if (IsMODef || IsOp2)
      return OperandInfo(MIVLMul, MILog2SEW);
    return OperandInfo(getEMULEqualsEEWDivSEWTimesLMUL(4, MI), 4);
  }
  // 16.5. Vector Compress Instruction
  // EMUL=LMUL. EEW=SEW.
  case RISCV::VCOMPRESS_VM:
    return OperandInfo(MIVLMul, MILog2SEW);

  // 16.6. Whole Vector Register Move
  case RISCV::VMV1R_V:
  case RISCV::VMV2R_V:
  case RISCV::VMV4R_V:
  case RISCV::VMV8R_V:
    llvm_unreachable("These instructions don't have pseudo versions so they "
                     "don't have an SEW operand.");

  default:
    return OperandInfo(OperandInfo::State::Unknown);
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
  case RISCV::VADD_VI:
  case RISCV::VADD_VV:
  case RISCV::VADD_VX:
  case RISCV::VMUL_VV:
  case RISCV::VMUL_VX:
  case RISCV::VSLL_VI:
  case RISCV::VSEXT_VF2:
  case RISCV::VSEXT_VF4:
  case RISCV::VSEXT_VF8:
  case RISCV::VZEXT_VF2:
  case RISCV::VZEXT_VF4:
  case RISCV::VZEXT_VF8:
  case RISCV::VMV_V_I:
  case RISCV::VMV_V_X:
  case RISCV::VNSRL_WI:
  case RISCV::VWADD_VV:
  case RISCV::VWADDU_VV:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCU_VX:
  case RISCV::VWSLL_VI:
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
  case RISCV::VFWREDUSUM_VS: {
    return isOpN(MO, 2);
  }
  default:
    return false;
  }
}

static bool safeToPropgateVL(const MachineInstr &MI) {
  const RISCVVPseudosTable::PseudoInfo *RVV =
      RISCVVPseudosTable::getPseudoInfo(MI.getOpcode());
  if (!RVV)
    return false;

  switch (RVV->BaseInstr) {
  // vslidedown instructions may use the higher part of the input operand beyond
  // the VL.
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDE1DOWN_VX:
  case RISCV::VFSLIDE1DOWN_VF:

  // vrgather instructions may index beyond the VL.
  case RISCV::VRGATHER_VI:
  case RISCV::VRGATHER_VV:
  case RISCV::VRGATHER_VX:
  case RISCV::VRGATHEREI16_VV:
    return false;

  default:
    return true;
  }
}

bool RISCVVLOptimizer::isCandidate(const MachineInstr &MI) const {

  LLVM_DEBUG(
      dbgs() << "Check whether the instruction is a candidate for reducing VL:"
             << MI << "\n");

  const MCInstrDesc &Desc = MI.getDesc();
  if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags)) {
    LLVM_DEBUG(dbgs() << "  Not a candidate due to lack of vl op or sew op\n");
    return false;
  }

  if (MI.getNumDefs() != 1) {
    LLVM_DEBUG(dbgs() << " Not a candidate due to it def more than one\n");
    return false;
  }
  unsigned VLOpNum = RISCVII::getVLOpNum(Desc);
  const MachineOperand &VLOp = MI.getOperand(VLOpNum);
  if (!VLOp.isImm() || VLOp.getImm() != RISCV::VLMaxSentinel) {
    LLVM_DEBUG(dbgs() << "  Not a candidate due to VL is not VLMAX\n");
    return false;
  }

  // Some instructions that produce vectors have semantics that make it more
  // difficult to determine whether the VL can be reduced. For example, some
  // instructions, such as reductions, may write lanes past VL to a scalar
  // register. Other instructions, such as some loads or stores, may write
  // lower lanes using data from higher lanes. There may be other complex
  // semantics not mentioned here that make it hard to determine whether
  // the VL can be optimized. As a result, a white-list of supported
  // instructions is used. Over time, more instructions cam be supported
  // upon careful examination of their semantics under the logic in this
  // optimization.
  // TODO: Use a better approach than a white-list, such as adding
  // properties to instructions using something like TSFlags.
  if (!isSupportedInstr(MI)) {
    LLVM_DEBUG(dbgs() << "  Not a candidate due to unsupported instruction\n");
    return false;
  }

  return true;
}

bool RISCVVLOptimizer::tryReduceVL(MachineInstr &OrigMI) {
  SetVector<MachineInstr *> Worklist;
  Worklist.insert(&OrigMI);

  bool MadeChange = false;
  while (!Worklist.empty()) {
    MachineInstr &MI = *Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "Try reduce VL for " << MI << "\n");
    std::optional<Register> CommonVL;
    bool CanReduceVL = true;
    for (auto &UserOp : MRI->use_operands(MI.getOperand(0).getReg())) {
      const MachineInstr &UserMI = *UserOp.getParent();
      LLVM_DEBUG(dbgs() << "  Check user: " << UserMI << "\n");

      // Instructions like reductions may use a vector register as a scalar
      // register. In this case, we should treat it like a scalar register which
      // does not impact the decision on whether to optimize VL.
      if (isVectorOpUsedAsScalarOp(UserOp)) {
        [[maybe_unused]] Register R = UserOp.getReg();
        [[maybe_unused]] const TargetRegisterClass *RC = MRI->getRegClass(R);
        assert(RISCV::VRRegClass.hasSubClassEq(RC) &&
               "Expect LMUL 1 register class for vector as scalar operands!");
        LLVM_DEBUG(dbgs() << "    Use this operand as a scalar operand\n");
        continue;
      }

      if (!safeToPropgateVL(UserMI)) {
        LLVM_DEBUG(dbgs() << "    Abort due to used by unsafe instruction\n");
        CanReduceVL = false;
        break;
      }

      // Tied operands might pass through.
      if (UserOp.isTied()) {
        LLVM_DEBUG(dbgs() << "    Abort due to user use it as tied operand\n");
        CanReduceVL = false;
        break;
      }

      const MCInstrDesc &Desc = UserMI.getDesc();
      if (!RISCVII::hasVLOp(Desc.TSFlags) || !RISCVII::hasSEWOp(Desc.TSFlags)) {
        LLVM_DEBUG(dbgs() << "    Abort due to lack of VL or SEW, assume that"
                             " use VLMAX.\n");
        CanReduceVL = false;
        break;
      }

      unsigned VLOpNum = RISCVII::getVLOpNum(Desc);
      const MachineOperand &VLOp = UserMI.getOperand(VLOpNum);
      // Looking for a register VL that isn't X0.
      if (!VLOp.isReg() || VLOp.getReg() == RISCV::X0) {
        LLVM_DEBUG(dbgs() << "    Abort due to user use X0 as VL.\n");
        CanReduceVL = false;
        break;
      }

      if (!CommonVL) {
        CommonVL = VLOp.getReg();
      } else if (*CommonVL != VLOp.getReg()) {
        LLVM_DEBUG(dbgs() << "    Abort due to users have different VL!\n");
        CanReduceVL = false;
        break;
      }

      // The SEW and LMUL of destination and source registers need to match.

      // If the produced Dest is not a vector register, then it has no EEW or
      // EMUL, so there is no need to check that producer and consumer LMUL and
      // SEW match. We've already checked above that UserOp is a vector
      // register.
      if (!isVectorRegClass(MI.getOperand(0).getReg(), MRI)) {
        LLVM_DEBUG(dbgs() << "    Abort due to register class mismatch between "
                             "USE and DEF\n");
        continue;
      }

      OperandInfo ConsumerInfo = getOperandInfo(UserMI, UserOp, MRI);
      OperandInfo ProducerInfo = getOperandInfo(MI, MI.getOperand(0), MRI);
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

    if (!CanReduceVL || !CommonVL)
      continue;

    if (!CommonVL->isVirtual()) {
      LLVM_DEBUG(
          dbgs() << "    Abort due to new VL is not virtual register.\n");
      continue;
    }

    const MachineInstr *VLMI = MRI->getVRegDef(*CommonVL);
    if (!MDT->dominates(VLMI, &MI))
      continue;

    // All our checks passed. We can reduce VL.
    unsigned VLOpNum = RISCVII::getVLOpNum(MI.getDesc());
    MachineOperand &VLOp = MI.getOperand(VLOpNum);
    VLOp.ChangeToRegister(*CommonVL, false);
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
