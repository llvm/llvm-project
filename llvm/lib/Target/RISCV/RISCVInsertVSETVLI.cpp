//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed and expands the vl outputs of VLEFF/VLSEGFF to PseudoReadVL
// instructions.
//
// This pass consists of 3 phases:
//
// Phase 1 collects how each basic block affects VL/VTYPE.
//
// Phase 2 uses the information from phase 1 to do a data flow analysis to
// propagate the VL/VTYPE changes through the function. This gives us the
// VL/VTYPE at the start of each basic block.
//
// Phase 3 inserts VSETVLI instructions in each basic block. Information from
// phase 2 is used to prevent inserting a VSETVLI before the first vector
// instruction in the block if possible.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-vsetvli"
#define RISCV_INSERT_VSETVLI_NAME "RISCV Insert VSETVLI pass"

static cl::opt<bool> DisableInsertVSETVLPHIOpt(
    "riscv-disable-insert-vsetvl-phi-opt", cl::init(false), cl::Hidden,
    cl::desc("Disable looking through phis when inserting vsetvlis."));

static cl::opt<bool> UseStrictAsserts(
    "riscv-insert-vsetvl-strict-asserts", cl::init(true), cl::Hidden,
    cl::desc("Enable strict assertion checking for the dataflow algorithm"));

namespace {

static unsigned getVLOpNum(const MachineInstr &MI) {
  return RISCVII::getVLOpNum(MI.getDesc());
}

static unsigned getSEWOpNum(const MachineInstr &MI) {
  return RISCVII::getSEWOpNum(MI.getDesc());
}

static bool isScalarMoveInstr(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return false;
  case RISCV::PseudoVMV_S_X_M1:
  case RISCV::PseudoVMV_S_X_M2:
  case RISCV::PseudoVMV_S_X_M4:
  case RISCV::PseudoVMV_S_X_M8:
  case RISCV::PseudoVMV_S_X_MF2:
  case RISCV::PseudoVMV_S_X_MF4:
  case RISCV::PseudoVMV_S_X_MF8:
  case RISCV::PseudoVFMV_S_F16_M1:
  case RISCV::PseudoVFMV_S_F16_M2:
  case RISCV::PseudoVFMV_S_F16_M4:
  case RISCV::PseudoVFMV_S_F16_M8:
  case RISCV::PseudoVFMV_S_F16_MF2:
  case RISCV::PseudoVFMV_S_F16_MF4:
  case RISCV::PseudoVFMV_S_F32_M1:
  case RISCV::PseudoVFMV_S_F32_M2:
  case RISCV::PseudoVFMV_S_F32_M4:
  case RISCV::PseudoVFMV_S_F32_M8:
  case RISCV::PseudoVFMV_S_F32_MF2:
  case RISCV::PseudoVFMV_S_F64_M1:
  case RISCV::PseudoVFMV_S_F64_M2:
  case RISCV::PseudoVFMV_S_F64_M4:
  case RISCV::PseudoVFMV_S_F64_M8:
    return true;
  }
}

/// Get the EEW for a load or store instruction.  Return None if MI is not
/// a load or store which ignores SEW.
static Optional<unsigned> getEEWForLoadStore(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  default:
    return None;
  case RISCV::PseudoVLE8_V_M1:
  case RISCV::PseudoVLE8_V_M1_MASK:
  case RISCV::PseudoVLE8_V_M2:
  case RISCV::PseudoVLE8_V_M2_MASK:
  case RISCV::PseudoVLE8_V_M4:
  case RISCV::PseudoVLE8_V_M4_MASK:
  case RISCV::PseudoVLE8_V_M8:
  case RISCV::PseudoVLE8_V_M8_MASK:
  case RISCV::PseudoVLE8_V_MF2:
  case RISCV::PseudoVLE8_V_MF2_MASK:
  case RISCV::PseudoVLE8_V_MF4:
  case RISCV::PseudoVLE8_V_MF4_MASK:
  case RISCV::PseudoVLE8_V_MF8:
  case RISCV::PseudoVLE8_V_MF8_MASK:
  case RISCV::PseudoVLSE8_V_M1:
  case RISCV::PseudoVLSE8_V_M1_MASK:
  case RISCV::PseudoVLSE8_V_M2:
  case RISCV::PseudoVLSE8_V_M2_MASK:
  case RISCV::PseudoVLSE8_V_M4:
  case RISCV::PseudoVLSE8_V_M4_MASK:
  case RISCV::PseudoVLSE8_V_M8:
  case RISCV::PseudoVLSE8_V_M8_MASK:
  case RISCV::PseudoVLSE8_V_MF2:
  case RISCV::PseudoVLSE8_V_MF2_MASK:
  case RISCV::PseudoVLSE8_V_MF4:
  case RISCV::PseudoVLSE8_V_MF4_MASK:
  case RISCV::PseudoVLSE8_V_MF8:
  case RISCV::PseudoVLSE8_V_MF8_MASK:
  case RISCV::PseudoVSE8_V_M1:
  case RISCV::PseudoVSE8_V_M1_MASK:
  case RISCV::PseudoVSE8_V_M2:
  case RISCV::PseudoVSE8_V_M2_MASK:
  case RISCV::PseudoVSE8_V_M4:
  case RISCV::PseudoVSE8_V_M4_MASK:
  case RISCV::PseudoVSE8_V_M8:
  case RISCV::PseudoVSE8_V_M8_MASK:
  case RISCV::PseudoVSE8_V_MF2:
  case RISCV::PseudoVSE8_V_MF2_MASK:
  case RISCV::PseudoVSE8_V_MF4:
  case RISCV::PseudoVSE8_V_MF4_MASK:
  case RISCV::PseudoVSE8_V_MF8:
  case RISCV::PseudoVSE8_V_MF8_MASK:
  case RISCV::PseudoVSSE8_V_M1:
  case RISCV::PseudoVSSE8_V_M1_MASK:
  case RISCV::PseudoVSSE8_V_M2:
  case RISCV::PseudoVSSE8_V_M2_MASK:
  case RISCV::PseudoVSSE8_V_M4:
  case RISCV::PseudoVSSE8_V_M4_MASK:
  case RISCV::PseudoVSSE8_V_M8:
  case RISCV::PseudoVSSE8_V_M8_MASK:
  case RISCV::PseudoVSSE8_V_MF2:
  case RISCV::PseudoVSSE8_V_MF2_MASK:
  case RISCV::PseudoVSSE8_V_MF4:
  case RISCV::PseudoVSSE8_V_MF4_MASK:
  case RISCV::PseudoVSSE8_V_MF8:
  case RISCV::PseudoVSSE8_V_MF8_MASK:
    return 8;
  case RISCV::PseudoVLE16_V_M1:
  case RISCV::PseudoVLE16_V_M1_MASK:
  case RISCV::PseudoVLE16_V_M2:
  case RISCV::PseudoVLE16_V_M2_MASK:
  case RISCV::PseudoVLE16_V_M4:
  case RISCV::PseudoVLE16_V_M4_MASK:
  case RISCV::PseudoVLE16_V_M8:
  case RISCV::PseudoVLE16_V_M8_MASK:
  case RISCV::PseudoVLE16_V_MF2:
  case RISCV::PseudoVLE16_V_MF2_MASK:
  case RISCV::PseudoVLE16_V_MF4:
  case RISCV::PseudoVLE16_V_MF4_MASK:
  case RISCV::PseudoVLSE16_V_M1:
  case RISCV::PseudoVLSE16_V_M1_MASK:
  case RISCV::PseudoVLSE16_V_M2:
  case RISCV::PseudoVLSE16_V_M2_MASK:
  case RISCV::PseudoVLSE16_V_M4:
  case RISCV::PseudoVLSE16_V_M4_MASK:
  case RISCV::PseudoVLSE16_V_M8:
  case RISCV::PseudoVLSE16_V_M8_MASK:
  case RISCV::PseudoVLSE16_V_MF2:
  case RISCV::PseudoVLSE16_V_MF2_MASK:
  case RISCV::PseudoVLSE16_V_MF4:
  case RISCV::PseudoVLSE16_V_MF4_MASK:
  case RISCV::PseudoVSE16_V_M1:
  case RISCV::PseudoVSE16_V_M1_MASK:
  case RISCV::PseudoVSE16_V_M2:
  case RISCV::PseudoVSE16_V_M2_MASK:
  case RISCV::PseudoVSE16_V_M4:
  case RISCV::PseudoVSE16_V_M4_MASK:
  case RISCV::PseudoVSE16_V_M8:
  case RISCV::PseudoVSE16_V_M8_MASK:
  case RISCV::PseudoVSE16_V_MF2:
  case RISCV::PseudoVSE16_V_MF2_MASK:
  case RISCV::PseudoVSE16_V_MF4:
  case RISCV::PseudoVSE16_V_MF4_MASK:
  case RISCV::PseudoVSSE16_V_M1:
  case RISCV::PseudoVSSE16_V_M1_MASK:
  case RISCV::PseudoVSSE16_V_M2:
  case RISCV::PseudoVSSE16_V_M2_MASK:
  case RISCV::PseudoVSSE16_V_M4:
  case RISCV::PseudoVSSE16_V_M4_MASK:
  case RISCV::PseudoVSSE16_V_M8:
  case RISCV::PseudoVSSE16_V_M8_MASK:
  case RISCV::PseudoVSSE16_V_MF2:
  case RISCV::PseudoVSSE16_V_MF2_MASK:
  case RISCV::PseudoVSSE16_V_MF4:
  case RISCV::PseudoVSSE16_V_MF4_MASK:
    return 16;
  case RISCV::PseudoVLE32_V_M1:
  case RISCV::PseudoVLE32_V_M1_MASK:
  case RISCV::PseudoVLE32_V_M2:
  case RISCV::PseudoVLE32_V_M2_MASK:
  case RISCV::PseudoVLE32_V_M4:
  case RISCV::PseudoVLE32_V_M4_MASK:
  case RISCV::PseudoVLE32_V_M8:
  case RISCV::PseudoVLE32_V_M8_MASK:
  case RISCV::PseudoVLE32_V_MF2:
  case RISCV::PseudoVLE32_V_MF2_MASK:
  case RISCV::PseudoVLSE32_V_M1:
  case RISCV::PseudoVLSE32_V_M1_MASK:
  case RISCV::PseudoVLSE32_V_M2:
  case RISCV::PseudoVLSE32_V_M2_MASK:
  case RISCV::PseudoVLSE32_V_M4:
  case RISCV::PseudoVLSE32_V_M4_MASK:
  case RISCV::PseudoVLSE32_V_M8:
  case RISCV::PseudoVLSE32_V_M8_MASK:
  case RISCV::PseudoVLSE32_V_MF2:
  case RISCV::PseudoVLSE32_V_MF2_MASK:
  case RISCV::PseudoVSE32_V_M1:
  case RISCV::PseudoVSE32_V_M1_MASK:
  case RISCV::PseudoVSE32_V_M2:
  case RISCV::PseudoVSE32_V_M2_MASK:
  case RISCV::PseudoVSE32_V_M4:
  case RISCV::PseudoVSE32_V_M4_MASK:
  case RISCV::PseudoVSE32_V_M8:
  case RISCV::PseudoVSE32_V_M8_MASK:
  case RISCV::PseudoVSE32_V_MF2:
  case RISCV::PseudoVSE32_V_MF2_MASK:
  case RISCV::PseudoVSSE32_V_M1:
  case RISCV::PseudoVSSE32_V_M1_MASK:
  case RISCV::PseudoVSSE32_V_M2:
  case RISCV::PseudoVSSE32_V_M2_MASK:
  case RISCV::PseudoVSSE32_V_M4:
  case RISCV::PseudoVSSE32_V_M4_MASK:
  case RISCV::PseudoVSSE32_V_M8:
  case RISCV::PseudoVSSE32_V_M8_MASK:
  case RISCV::PseudoVSSE32_V_MF2:
  case RISCV::PseudoVSSE32_V_MF2_MASK:
    return 32;
  case RISCV::PseudoVLE64_V_M1:
  case RISCV::PseudoVLE64_V_M1_MASK:
  case RISCV::PseudoVLE64_V_M2:
  case RISCV::PseudoVLE64_V_M2_MASK:
  case RISCV::PseudoVLE64_V_M4:
  case RISCV::PseudoVLE64_V_M4_MASK:
  case RISCV::PseudoVLE64_V_M8:
  case RISCV::PseudoVLE64_V_M8_MASK:
  case RISCV::PseudoVLSE64_V_M1:
  case RISCV::PseudoVLSE64_V_M1_MASK:
  case RISCV::PseudoVLSE64_V_M2:
  case RISCV::PseudoVLSE64_V_M2_MASK:
  case RISCV::PseudoVLSE64_V_M4:
  case RISCV::PseudoVLSE64_V_M4_MASK:
  case RISCV::PseudoVLSE64_V_M8:
  case RISCV::PseudoVLSE64_V_M8_MASK:
  case RISCV::PseudoVSE64_V_M1:
  case RISCV::PseudoVSE64_V_M1_MASK:
  case RISCV::PseudoVSE64_V_M2:
  case RISCV::PseudoVSE64_V_M2_MASK:
  case RISCV::PseudoVSE64_V_M4:
  case RISCV::PseudoVSE64_V_M4_MASK:
  case RISCV::PseudoVSE64_V_M8:
  case RISCV::PseudoVSE64_V_M8_MASK:
  case RISCV::PseudoVSSE64_V_M1:
  case RISCV::PseudoVSSE64_V_M1_MASK:
  case RISCV::PseudoVSSE64_V_M2:
  case RISCV::PseudoVSSE64_V_M2_MASK:
  case RISCV::PseudoVSSE64_V_M4:
  case RISCV::PseudoVSSE64_V_M4_MASK:
  case RISCV::PseudoVSSE64_V_M8:
  case RISCV::PseudoVSSE64_V_M8_MASK:
    return 64;
  }
}

/// Return true if this is an operation on mask registers.  Note that
/// this includes both arithmetic/logical ops and load/store (vlm/vsm).
static bool isMaskRegOp(const MachineInstr &MI) {
  if (RISCVII::hasSEWOp(MI.getDesc().TSFlags)) {
    const unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
    // A Log2SEW of 0 is an operation on mask registers only.
    return Log2SEW == 0;
  }
  return false;
}

static unsigned getSEWLMULRatio(unsigned SEW, RISCVII::VLMUL VLMul) {
  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = RISCVVType::decodeVLMUL(VLMul);

  // Convert LMul to a fixed point value with 3 fractional bits.
  LMul = Fractional ? (8 / LMul) : (LMul * 8);

  assert(SEW >= 8 && "Unexpected SEW value");
  return (SEW * 8) / LMul;
}

/// Which subfields of VL or VTYPE have values we need to preserve?
struct DemandedFields {
  bool VL = false;
  bool SEW = false;
  bool LMUL = false;
  bool SEWLMULRatio = false;
  bool TailPolicy = false;
  bool MaskPolicy = false;

  // Return true if any part of VTYPE was used
  bool usedVTYPE() {
    return SEW || LMUL || SEWLMULRatio || TailPolicy || MaskPolicy;
  }

  // Mark all VTYPE subfields and properties as demanded
  void demandVTYPE() {
    SEW = true;
    LMUL = true;
    SEWLMULRatio = true;
    TailPolicy = true;
    MaskPolicy = true;
  }
};

/// Return true if the two values of the VTYPE register provided are
/// indistinguishable from the perspective of an instruction (or set of
/// instructions) which use only the Used subfields and properties.
static bool areCompatibleVTYPEs(uint64_t VType1,
                                uint64_t VType2,
                                const DemandedFields &Used) {
  if (Used.SEW &&
      RISCVVType::getSEW(VType1) != RISCVVType::getSEW(VType2))
    return false;

  if (Used.LMUL &&
      RISCVVType::getVLMUL(VType1) != RISCVVType::getVLMUL(VType2))
    return false;

  if (Used.SEWLMULRatio) {
    auto Ratio1 = getSEWLMULRatio(RISCVVType::getSEW(VType1),
                                  RISCVVType::getVLMUL(VType1));
    auto Ratio2 = getSEWLMULRatio(RISCVVType::getSEW(VType2),
                                  RISCVVType::getVLMUL(VType2));
    if (Ratio1 != Ratio2)
      return false;
  }

  if (Used.TailPolicy &&
      RISCVVType::isTailAgnostic(VType1) != RISCVVType::isTailAgnostic(VType2))
    return false;
  if (Used.MaskPolicy &&
      RISCVVType::isMaskAgnostic(VType1) != RISCVVType::isMaskAgnostic(VType2))
    return false;
  return true;
}

/// Return the fields and properties demanded by the provided instruction.
static DemandedFields getDemanded(const MachineInstr &MI) {
  // Warning: This function has to work on both the lowered (i.e. post
  // emitVSETVLIs) and pre-lowering forms.  The main implication of this is
  // that it can't use the value of a SEW, VL, or Policy operand as they might
  // be stale after lowering.

  // Most instructions don't use any of these subfeilds.
  DemandedFields Res;
  // Start conservative if registers are used
  if (MI.isCall() || MI.isInlineAsm() || MI.readsRegister(RISCV::VL))
    Res.VL = true;
  if (MI.isCall() || MI.isInlineAsm() || MI.readsRegister(RISCV::VTYPE))
    Res.demandVTYPE();
  // Start conservative on the unlowered form too
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (RISCVII::hasSEWOp(TSFlags)) {
    Res.demandVTYPE();
    if (RISCVII::hasVLOp(TSFlags))
      Res.VL = true;
  }

  // Loads and stores with implicit EEW do not demand SEW or LMUL directly.
  // They instead demand the ratio of the two which is used in computing
  // EMUL, but which allows us the flexibility to change SEW and LMUL
  // provided we don't change the ratio.
  // Note: We assume that the instructions initial SEW is the EEW encoded
  // in the opcode.  This is asserted when constructing the VSETVLIInfo.
  if (getEEWForLoadStore(MI)) {
    Res.SEW = false;
    Res.LMUL = false;
  }

  // Store instructions don't use the policy fields.
  if (RISCVII::hasSEWOp(TSFlags) && MI.getNumExplicitDefs() == 0) {
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  // If this is a mask reg operation, it only cares about VLMAX.
  // TODO: Possible extensions to this logic
  // * Probably ok if available VLMax is larger than demanded
  // * The policy bits can probably be ignored..
  if (isMaskRegOp(MI)) {
    Res.SEW = false;
    Res.LMUL = false;
  }

  return Res;
}

/// Defines the abstract state with which the forward dataflow models the
/// values of the VL and VTYPE registers after insertion.
class VSETVLIInfo {
  union {
    Register AVLReg;
    unsigned AVLImm;
  };

  enum : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    Unknown,
  } State = Uninitialized;

  // Fields from VTYPE.
  RISCVII::VLMUL VLMul = RISCVII::LMUL_1;
  uint8_t SEW = 0;
  uint8_t TailAgnostic : 1;
  uint8_t MaskAgnostic : 1;
  uint8_t SEWLMULRatioOnly : 1;

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false),
        SEWLMULRatioOnly(false) {}

  static VSETVLIInfo getUnknown() {
    VSETVLIInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  void setAVLReg(Register Reg) {
    AVLReg = Reg;
    State = AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLIsImm;
  }

  bool hasAVLImm() const { return State == AVLIsImm; }
  bool hasAVLReg() const { return State == AVLIsReg; }
  Register getAVLReg() const {
    assert(hasAVLReg());
    return AVLReg;
  }
  unsigned getAVLImm() const {
    assert(hasAVLImm());
    return AVLImm;
  }

  unsigned getSEW() const { return SEW; }
  RISCVII::VLMUL getVLMUL() const { return VLMul; }

  bool hasNonZeroAVL() const {
    if (hasAVLImm())
      return getAVLImm() > 0;
    if (hasAVLReg())
      return getAVLReg() == RISCV::X0;
    return false;
  }

  bool hasSameAVL(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare AVL in unknown state");
    if (hasAVLReg() && Other.hasAVLReg())
      return getAVLReg() == Other.getAVLReg();

    if (hasAVLImm() && Other.hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    return false;
  }

  void setVTYPE(unsigned VType) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = RISCVVType::getVLMUL(VType);
    SEW = RISCVVType::getSEW(VType);
    TailAgnostic = RISCVVType::isTailAgnostic(VType);
    MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
  }
  void setVTYPE(RISCVII::VLMUL L, unsigned S, bool TA, bool MA) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
  }

  unsigned encodeVTYPE() const {
    assert(isValid() && !isUnknown() && !SEWLMULRatioOnly &&
           "Can't encode VTYPE for uninitialized or unknown");
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);
  }

  bool hasSEWLMULRatioOnly() const { return SEWLMULRatioOnly; }

  bool hasSameSEW(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
           "Can't compare when only LMUL/SEW ratio is valid.");
    return SEW == Other.SEW;
  }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
           "Can't compare when only LMUL/SEW ratio is valid.");
    return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic) ==
           std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                    Other.MaskAgnostic);
  }

  unsigned getSEWLMULRatio() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return ::getSEWLMULRatio(SEW, VLMul);
  }

  // Check if the VTYPE for these two VSETVLIInfos produce the same VLMAX.
  // Note that having the same VLMAX ensures that both share the same
  // function from AVL to VL; that is, they must produce the same VL value
  // for any given AVL value.
  bool hasSameVLMAX(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return getSEWLMULRatio() == Other.getSEWLMULRatio();
  }

  bool hasSamePolicy(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return TailAgnostic == Other.TailAgnostic &&
           MaskAgnostic == Other.MaskAgnostic;
  }

  bool hasCompatibleVTYPE(const MachineInstr &MI,
                          const VSETVLIInfo &Require) const {
    const DemandedFields Used = getDemanded(MI);
    return areCompatibleVTYPEs(encodeVTYPE(), Require.encodeVTYPE(), Used);
  }

  // Determine whether the vector instructions requirements represented by
  // Require are compatible with the previous vsetvli instruction represented
  // by this.  MI is the instruction whose requirements we're considering.
  bool isCompatible(const MachineInstr &MI, const VSETVLIInfo &Require) const {
    assert(isValid() && Require.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!Require.SEWLMULRatioOnly &&
           "Expected a valid VTYPE for instruction!");
    // Nothing is compatible with Unknown.
    if (isUnknown() || Require.isUnknown())
      return false;

    // If only our VLMAX ratio is valid, then this isn't compatible.
    if (SEWLMULRatioOnly)
      return false;

    // If the instruction doesn't need an AVLReg and the SEW matches, consider
    // it compatible.
    if (Require.hasAVLReg() && Require.AVLReg == RISCV::NoRegister)
      if (SEW == Require.SEW)
        return true;

    return hasSameAVL(Require) && hasCompatibleVTYPE(MI, Require);
  }

  bool operator==(const VSETVLIInfo &Other) const {
    // Uninitialized is only equal to another Uninitialized.
    if (!isValid())
      return !Other.isValid();
    if (!Other.isValid())
      return !isValid();

    // Unknown is only equal to another Unknown.
    if (isUnknown())
      return Other.isUnknown();
    if (Other.isUnknown())
      return isUnknown();

    if (!hasSameAVL(Other))
      return false;

    // If the SEWLMULRatioOnly bits are different, then they aren't equal.
    if (SEWLMULRatioOnly != Other.SEWLMULRatioOnly)
      return false;

    // If only the VLMAX is valid, check that it is the same.
    if (SEWLMULRatioOnly)
      return hasSameVLMAX(Other);

    // If the full VTYPE is valid, check that it is the same.
    return hasSameVTYPE(Other);
  }

  bool operator!=(const VSETVLIInfo &Other) const {
    return !(*this == Other);
  }

  // Calculate the VSETVLIInfo visible to a block assuming this and Other are
  // both predecessors.
  VSETVLIInfo intersect(const VSETVLIInfo &Other) const {
    // If the new value isn't valid, ignore it.
    if (!Other.isValid())
      return *this;

    // If this value isn't valid, this must be the first predecessor, use it.
    if (!isValid())
      return Other;

    // If either is unknown, the result is unknown.
    if (isUnknown() || Other.isUnknown())
      return VSETVLIInfo::getUnknown();

    // If we have an exact, match return this.
    if (*this == Other)
      return *this;

    // Not an exact match, but maybe the AVL and VLMAX are the same. If so,
    // return an SEW/LMUL ratio only value.
    if (hasSameAVL(Other) && hasSameVLMAX(Other)) {
      VSETVLIInfo MergeInfo = *this;
      MergeInfo.SEWLMULRatioOnly = true;
      return MergeInfo;
    }

    // Otherwise the result is unknown.
    return VSETVLIInfo::getUnknown();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  /// @{
  void print(raw_ostream &OS) const {
    OS << "{";
    if (!isValid())
      OS << "Uninitialized";
    if (isUnknown())
      OS << "unknown";
    if (hasAVLReg())
      OS << "AVLReg=" << (unsigned)AVLReg;
    if (hasAVLImm())
      OS << "AVLImm=" << (unsigned)AVLImm;
    OS << ", "
       << "VLMul=" << (unsigned)VLMul << ", "
       << "SEW=" << (unsigned)SEW << ", "
       << "TailAgnostic=" << (bool)TailAgnostic << ", "
       << "MaskAgnostic=" << (bool)MaskAgnostic << ", "
       << "SEWLMULRatioOnly=" << (bool)SEWLMULRatioOnly << "}";
  }
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VSETVLIInfo &V) {
  V.print(OS);
  return OS;
}
#endif

struct BlockData {
  // The VSETVLIInfo that represents the net changes to the VL/VTYPE registers
  // made by this block. Calculated in Phase 1.
  VSETVLIInfo Change;

  // The VSETVLIInfo that represents the VL/VTYPE settings on exit from this
  // block. Calculated in Phase 2.
  VSETVLIInfo Exit;

  // The VSETVLIInfo that represents the VL/VTYPE settings from all predecessor
  // blocks. Calculated in Phase 2, and used by Phase 3.
  VSETVLIInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue = false;

  BlockData() = default;
};

class RISCVInsertVSETVLI : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;

  std::vector<BlockData> BlockInfo;
  std::queue<const MachineBasicBlock *> WorkList;

public:
  static char ID;

  RISCVInsertVSETVLI() : MachineFunctionPass(ID) {
    initializeRISCVInsertVSETVLIPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }

private:
  bool needVSETVLI(const MachineInstr &MI, const VSETVLIInfo &Require,
                   const VSETVLIInfo &CurInfo) const;
  bool needVSETVLIPHI(const VSETVLIInfo &Require,
                      const MachineBasicBlock &MBB) const;
  void insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);
  void insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);

  void transferBefore(VSETVLIInfo &Info, const MachineInstr &MI);
  void transferAfter(VSETVLIInfo &Info, const MachineInstr &MI);
  bool computeVLVTYPEChanges(const MachineBasicBlock &MBB);
  void computeIncomingVLVTYPE(const MachineBasicBlock &MBB);
  void emitVSETVLIs(MachineBasicBlock &MBB);
  void doLocalPostpass(MachineBasicBlock &MBB);
  void doPRE(MachineBasicBlock &MBB);
  void insertReadVL(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;

INITIALIZE_PASS(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

static bool isVectorConfigInstr(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoVSETVLI ||
         MI.getOpcode() == RISCV::PseudoVSETVLIX0 ||
         MI.getOpcode() == RISCV::PseudoVSETIVLI;
}

/// Return true if this is 'vsetvli x0, x0, vtype' which preserves
/// VL and only sets VTYPE.
static bool isVLPreservingConfig(const MachineInstr &MI) {
  if (MI.getOpcode() != RISCV::PseudoVSETVLIX0)
    return false;
  assert(RISCV::X0 == MI.getOperand(1).getReg());
  return RISCV::X0 == MI.getOperand(0).getReg();
}

static VSETVLIInfo computeInfoForInstr(const MachineInstr &MI, uint64_t TSFlags,
                                       const MachineRegisterInfo *MRI) {
  VSETVLIInfo InstrInfo;

  // If the instruction has policy argument, use the argument.
  // If there is no policy argument, default to tail agnostic unless the
  // destination is tied to a source. Unless the source is undef. In that case
  // the user would have some control over the policy values.
  bool TailAgnostic = true;
  bool UsesMaskPolicy = RISCVII::usesMaskPolicy(TSFlags);
  // FIXME: Could we look at the above or below instructions to choose the
  // matched mask policy to reduce vsetvli instructions? Default mask policy is
  // agnostic if instructions use mask policy, otherwise is undisturbed. Because
  // most mask operations are mask undisturbed, so we could possibly reduce the
  // vsetvli between mask and nomasked instruction sequence.
  bool MaskAgnostic = UsesMaskPolicy;
  unsigned UseOpIdx;
  if (RISCVII::hasVecPolicyOp(TSFlags)) {
    const MachineOperand &Op = MI.getOperand(MI.getNumExplicitOperands() - 1);
    uint64_t Policy = Op.getImm();
    assert(Policy <= (RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC) &&
           "Invalid Policy Value");
    // Although in some cases, mismatched passthru/maskedoff with policy value
    // does not make sense (ex. tied operand is IMPLICIT_DEF with non-TAMA
    // policy, or tied operand is not IMPLICIT_DEF with TAMA policy), but users
    // have set the policy value explicitly, so compiler would not fix it.
    TailAgnostic = Policy & RISCVII::TAIL_AGNOSTIC;
    MaskAgnostic = Policy & RISCVII::MASK_AGNOSTIC;
  } else if (MI.isRegTiedToUseOperand(0, &UseOpIdx)) {
    TailAgnostic = false;
    if (UsesMaskPolicy)
      MaskAgnostic = false;
    // If the tied operand is an IMPLICIT_DEF we can keep TailAgnostic.
    const MachineOperand &UseMO = MI.getOperand(UseOpIdx);
    MachineInstr *UseMI = MRI->getVRegDef(UseMO.getReg());
    if (UseMI && UseMI->isImplicitDef()) {
      TailAgnostic = true;
      if (UsesMaskPolicy)
        MaskAgnostic = true;
    }
    // Some pseudo instructions force a tail agnostic policy despite having a
    // tied def.
    if (RISCVII::doesForceTailAgnostic(TSFlags))
      TailAgnostic = true;
  }

  RISCVII::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    if (VLOp.isImm()) {
      int64_t Imm = VLOp.getImm();
      // Conver the VLMax sentintel to X0 register.
      if (Imm == RISCV::VLMaxSentinel)
        InstrInfo.setAVLReg(RISCV::X0);
      else
        InstrInfo.setAVLImm(Imm);
    } else {
      InstrInfo.setAVLReg(VLOp.getReg());
    }
  } else {
    InstrInfo.setAVLReg(RISCV::NoRegister);
  }
#ifndef NDEBUG
  if (Optional<unsigned> EEW = getEEWForLoadStore(MI)) {
    assert(SEW == EEW && "Initial SEW doesn't match expected EEW");
  }
#endif
  InstrInfo.setVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);

  return InstrInfo;
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                                       const VSETVLIInfo &Info,
                                       const VSETVLIInfo &PrevInfo) {
  DebugLoc DL = MI.getDebugLoc();
  insertVSETVLI(MBB, MachineBasicBlock::iterator(&MI), DL, Info, PrevInfo);
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo) {

  // Use X0, X0 form if the AVL is the same and the SEW+LMUL gives the same
  // VLMAX.
  if (PrevInfo.isValid() && !PrevInfo.isUnknown() &&
      Info.hasSameAVL(PrevInfo) && Info.hasSameVLMAX(PrevInfo)) {
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addReg(RISCV::X0, RegState::Kill)
        .addImm(Info.encodeVTYPE())
        .addReg(RISCV::VL, RegState::Implicit);
    return;
  }

  if (Info.hasAVLImm()) {
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addImm(Info.getAVLImm())
        .addImm(Info.encodeVTYPE());
    return;
  }

  Register AVLReg = Info.getAVLReg();
  if (AVLReg == RISCV::NoRegister) {
    // We can only use x0, x0 if there's no chance of the vtype change causing
    // the previous vl to become invalid.
    if (PrevInfo.isValid() && !PrevInfo.isUnknown() &&
        Info.hasSameVLMAX(PrevInfo)) {
      BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
          .addReg(RISCV::X0, RegState::Define | RegState::Dead)
          .addReg(RISCV::X0, RegState::Kill)
          .addImm(Info.encodeVTYPE())
          .addReg(RISCV::VL, RegState::Implicit);
      return;
    }
    // Otherwise use an AVL of 0 to avoid depending on previous vl.
    BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
        .addImm(0)
        .addImm(Info.encodeVTYPE());
    return;
  }

  if (AVLReg.isVirtual())
    MRI->constrainRegClass(AVLReg, &RISCV::GPRNoX0RegClass);

  // Use X0 as the DestReg unless AVLReg is X0. We also need to change the
  // opcode if the AVLReg is X0 as they have different register classes for
  // the AVL operand.
  Register DestReg = RISCV::X0;
  unsigned Opcode = RISCV::PseudoVSETVLI;
  if (AVLReg == RISCV::X0) {
    DestReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    Opcode = RISCV::PseudoVSETVLIX0;
  }
  BuildMI(MBB, InsertPt, DL, TII->get(Opcode))
      .addReg(DestReg, RegState::Define | RegState::Dead)
      .addReg(AVLReg)
      .addImm(Info.encodeVTYPE());
}

// Return a VSETVLIInfo representing the changes made by this VSETVLI or
// VSETIVLI instruction.
static VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETIVLI) {
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI ||
           MI.getOpcode() == RISCV::PseudoVSETVLIX0);
    Register AVLReg = MI.getOperand(1).getReg();
    assert((AVLReg != RISCV::X0 || MI.getOperand(0).getReg() != RISCV::X0) &&
           "Can't handle X0, X0 vsetvli yet");
    NewInfo.setAVLReg(AVLReg);
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  return NewInfo;
}

/// Return true if a VSETVLI is required to transition from CurInfo to Require
/// before MI.
bool RISCVInsertVSETVLI::needVSETVLI(const MachineInstr &MI,
                                     const VSETVLIInfo &Require,
                                     const VSETVLIInfo &CurInfo) const {
  assert(Require == computeInfoForInstr(MI, MI.getDesc().TSFlags, MRI));

  if (CurInfo.isCompatible(MI, Require))
    return false;

  if (!CurInfo.isValid() || CurInfo.isUnknown() || CurInfo.hasSEWLMULRatioOnly())
    return true;

  // For vmv.s.x and vfmv.s.f, there is only two behaviors, VL = 0 and VL > 0.
  // VL=0 is uninteresting (as it should have been deleted already), so it is
  // compatible if we can prove both are non-zero.  Additionally, if writing
  // to an implicit_def operand, we don't need to preserve any other bits and
  // are thus compatible with any larger etype, and can disregard policy bits.
  if (isScalarMoveInstr(MI) &&
      CurInfo.hasNonZeroAVL() && Require.hasNonZeroAVL()) {
    auto *VRegDef = MRI->getVRegDef(MI.getOperand(1).getReg());
    if (VRegDef && VRegDef->isImplicitDef() &&
        CurInfo.getSEW() >= Require.getSEW())
      return false;
    if (CurInfo.hasSameSEW(Require) && CurInfo.hasSamePolicy(Require))
      return false;
  }

  // We didn't find a compatible value. If our AVL is a virtual register,
  // it might be defined by a VSET(I)VLI. If it has the same VLMAX we need
  // and the last VL/VTYPE we observed is the same, we don't need a
  // VSETVLI here.
  if (Require.hasAVLReg() && Require.getAVLReg().isVirtual() &&
      CurInfo.hasCompatibleVTYPE(MI, Require)) {
    if (MachineInstr *DefMI = MRI->getVRegDef(Require.getAVLReg())) {
      if (isVectorConfigInstr(*DefMI)) {
        VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
        if (DefInfo.hasSameAVL(CurInfo) && DefInfo.hasSameVLMAX(CurInfo))
          return false;
      }
    }
  }

  return true;
}

// Given an incoming state reaching MI, modifies that state so that it is minimally
// compatible with MI.  The resulting state is guaranteed to be semantically legal
// for MI, but may not be the state requested by MI.
void RISCVInsertVSETVLI::transferBefore(VSETVLIInfo &Info, const MachineInstr &MI) {
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (!RISCVII::hasSEWOp(TSFlags))
    return;

  const VSETVLIInfo NewInfo = computeInfoForInstr(MI, TSFlags, MRI);
  if (Info.isValid() && !needVSETVLI(MI, NewInfo, Info))
    return;

  const VSETVLIInfo PrevInfo = Info;
  Info = NewInfo;

  if (!RISCVII::hasVLOp(TSFlags))
    return;

  // For vmv.s.x and vfmv.s.f, there are only two behaviors, VL = 0 and
  // VL > 0. We can discard the user requested AVL and just use the last
  // one if we can prove it equally zero.  This removes a vsetvli entirely
  // if the types match or allows use of cheaper avl preserving variant
  // if VLMAX doesn't change.  If VLMAX might change, we couldn't use
  // the 'vsetvli x0, x0, vtype" variant, so we avoid the transform to
  // prevent extending live range of an avl register operand.
  // TODO: We can probably relax this for immediates.
  if (isScalarMoveInstr(MI) && PrevInfo.isValid() &&
      PrevInfo.hasNonZeroAVL() && Info.hasNonZeroAVL() &&
      Info.hasSameVLMAX(PrevInfo)) {
    if (PrevInfo.hasAVLImm())
      Info.setAVLImm(PrevInfo.getAVLImm());
    else
      Info.setAVLReg(PrevInfo.getAVLReg());
    return;
  }

  // Two cases involving an AVL resulting from a previous vsetvli.
  // 1) If the AVL is the result of a previous vsetvli which has the
  //    same AVL and VLMAX as our current state, we can reuse the AVL
  //    from the current state for the new one.  This allows us to
  //    generate 'vsetvli x0, x0, vtype" or possible skip the transition
  //    entirely.
  // 2) If AVL is defined by a vsetvli with the same VLMAX, we can
  //    replace the AVL operand with the AVL of the defining vsetvli.
  //    We avoid general register AVLs to avoid extending live ranges
  //    without being sure we can kill the original source reg entirely.
  if (!Info.hasAVLReg() || !Info.getAVLReg().isVirtual())
    return;
  MachineInstr *DefMI = MRI->getVRegDef(Info.getAVLReg());
  if (!DefMI || !isVectorConfigInstr(*DefMI))
    return;

  VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
  // case 1
  if (PrevInfo.isValid() && !PrevInfo.isUnknown() &&
      DefInfo.hasSameAVL(PrevInfo) &&
      DefInfo.hasSameVLMAX(PrevInfo)) {
    if (PrevInfo.hasAVLImm())
      Info.setAVLImm(PrevInfo.getAVLImm());
    else
      Info.setAVLReg(PrevInfo.getAVLReg());
    return;
  }
  // case 2
  if (DefInfo.hasSameVLMAX(Info) &&
      (DefInfo.hasAVLImm() || DefInfo.getAVLReg() == RISCV::X0)) {
    if (DefInfo.hasAVLImm())
      Info.setAVLImm(DefInfo.getAVLImm());
    else
      Info.setAVLReg(DefInfo.getAVLReg());
    return;
  }
}

// Given a state with which we evaluated MI (see transferBefore above for why
// this might be different that the state MI requested), modify the state to
// reflect the changes MI might make.
void RISCVInsertVSETVLI::transferAfter(VSETVLIInfo &Info, const MachineInstr &MI) {
  if (isVectorConfigInstr(MI)) {
    Info = getInfoForVSETVLI(MI);
    return;
  }

  if (RISCV::isFaultFirstLoad(MI)) {
    // Update AVL to vl-output of the fault first load.
    Info.setAVLReg(MI.getOperand(1).getReg());
    return;
  }

  // If this is something that updates VL/VTYPE that we don't know about, set
  // the state to unknown.
  if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VL) ||
      MI.modifiesRegister(RISCV::VTYPE))
    Info = VSETVLIInfo::getUnknown();
}

bool RISCVInsertVSETVLI::computeVLVTYPEChanges(const MachineBasicBlock &MBB) {
  bool HadVectorOp = false;

  BlockData &BBInfo = BlockInfo[MBB.getNumber()];
  BBInfo.Change = BBInfo.Pred;
  for (const MachineInstr &MI : MBB) {
    transferBefore(BBInfo.Change, MI);

    if (isVectorConfigInstr(MI) || RISCVII::hasSEWOp(MI.getDesc().TSFlags))
      HadVectorOp = true;

    transferAfter(BBInfo.Change, MI);
  }

  return HadVectorOp;
}

void RISCVInsertVSETVLI::computeIncomingVLVTYPE(const MachineBasicBlock &MBB) {

  BlockData &BBInfo = BlockInfo[MBB.getNumber()];

  BBInfo.InQueue = false;

  VSETVLIInfo InInfo;
  if (MBB.pred_empty()) {
    // There are no predecessors, so use the default starting status.
    InInfo.setUnknown();
  } else {
    for (MachineBasicBlock *P : MBB.predecessors())
      InInfo = InInfo.intersect(BlockInfo[P->getNumber()].Exit);
  }

  // If we don't have any valid predecessor value, wait until we do.
  if (!InInfo.isValid())
    return;

  // If no change, no need to rerun block
  if (InInfo == BBInfo.Pred)
    return;

  BBInfo.Pred = InInfo;
  LLVM_DEBUG(dbgs() << "Entry state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Pred << "\n");

  // Note: It's tempting to cache the state changes here, but due to the
  // compatibility checks performed a blocks output state can change based on
  // the input state.  To cache, we'd have to add logic for finding
  // never-compatible state changes.
  computeVLVTYPEChanges(MBB);
  VSETVLIInfo TmpStatus = BBInfo.Change;

  // If the new exit value matches the old exit value, we don't need to revisit
  // any blocks.
  if (BBInfo.Exit == TmpStatus)
    return;

  BBInfo.Exit = TmpStatus;
  LLVM_DEBUG(dbgs() << "Exit state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Exit << "\n");

  // Add the successors to the work list so we can propagate the changed exit
  // status.
  for (MachineBasicBlock *S : MBB.successors())
    if (!BlockInfo[S->getNumber()].InQueue)
      WorkList.push(S);
}

// If we weren't able to prove a vsetvli was directly unneeded, it might still
// be unneeded if the AVL is a phi node where all incoming values are VL
// outputs from the last VSETVLI in their respective basic blocks.
bool RISCVInsertVSETVLI::needVSETVLIPHI(const VSETVLIInfo &Require,
                                        const MachineBasicBlock &MBB) const {
  if (DisableInsertVSETVLPHIOpt)
    return true;

  if (!Require.hasAVLReg())
    return true;

  Register AVLReg = Require.getAVLReg();
  if (!AVLReg.isVirtual())
    return true;

  // We need the AVL to be produce by a PHI node in this basic block.
  MachineInstr *PHI = MRI->getVRegDef(AVLReg);
  if (!PHI || PHI->getOpcode() != RISCV::PHI || PHI->getParent() != &MBB)
    return true;

  for (unsigned PHIOp = 1, NumOps = PHI->getNumOperands(); PHIOp != NumOps;
       PHIOp += 2) {
    Register InReg = PHI->getOperand(PHIOp).getReg();
    MachineBasicBlock *PBB = PHI->getOperand(PHIOp + 1).getMBB();
    const BlockData &PBBInfo = BlockInfo[PBB->getNumber()];
    // If the exit from the predecessor has the VTYPE we are looking for
    // we might be able to avoid a VSETVLI.
    if (PBBInfo.Exit.isUnknown() || !PBBInfo.Exit.hasSameVTYPE(Require))
      return true;

    // We need the PHI input to the be the output of a VSET(I)VLI.
    MachineInstr *DefMI = MRI->getVRegDef(InReg);
    if (!DefMI || !isVectorConfigInstr(*DefMI))
      return true;

    // We found a VSET(I)VLI make sure it matches the output of the
    // predecessor block.
    VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
    if (!DefInfo.hasSameAVL(PBBInfo.Exit) ||
        !DefInfo.hasSameVTYPE(PBBInfo.Exit))
      return true;
  }

  // If all the incoming values to the PHI checked out, we don't need
  // to insert a VSETVLI.
  return false;
}

void RISCVInsertVSETVLI::emitVSETVLIs(MachineBasicBlock &MBB) {
  VSETVLIInfo CurInfo = BlockInfo[MBB.getNumber()].Pred;
  // Track whether the prefix of the block we've scanned is transparent
  // (meaning has not yet changed the abstract state).
  bool PrefixTransparent = true;
  for (MachineInstr &MI : MBB) {
    const VSETVLIInfo PrevInfo = CurInfo;
    transferBefore(CurInfo, MI);

    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (isVectorConfigInstr(MI)) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      PrefixTransparent = false;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      if (PrevInfo != CurInfo) {
        // If this is the first implicit state change, and the state change
        // requested can be proven to produce the same register contents, we
        // can skip emitting the actual state change and continue as if we
        // had since we know the GPR result of the implicit state change
        // wouldn't be used and VL/VTYPE registers are correct.  Note that
        // we *do* need to model the state as if it changed as while the
        // register contents are unchanged, the abstract model can change.
        if (!PrefixTransparent || needVSETVLIPHI(CurInfo, MBB))
          insertVSETVLI(MBB, MI, CurInfo, PrevInfo);
        PrefixTransparent = false;
      }

      if (RISCVII::hasVLOp(TSFlags)) {
        MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
        if (VLOp.isReg()) {
          // Erase the AVL operand from the instruction.
          VLOp.setReg(RISCV::NoRegister);
          VLOp.setIsKill(false);
        }
        MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ false,
                                                /*isImp*/ true));
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));
    }

    if (MI.isCall() || MI.isInlineAsm() || MI.modifiesRegister(RISCV::VL) ||
        MI.modifiesRegister(RISCV::VTYPE))
      PrefixTransparent = false;

    transferAfter(CurInfo, MI);
  }

  // If we reach the end of the block and our current info doesn't match the
  // expected info, insert a vsetvli to correct.
  if (!UseStrictAsserts) {
    const VSETVLIInfo &ExitInfo = BlockInfo[MBB.getNumber()].Exit;
    if (CurInfo.isValid() && ExitInfo.isValid() && !ExitInfo.isUnknown() &&
        CurInfo != ExitInfo) {
      // Note there's an implicit assumption here that terminators never use
      // or modify VL or VTYPE.  Also, fallthrough will return end().
      auto InsertPt = MBB.getFirstInstrTerminator();
      insertVSETVLI(MBB, InsertPt, MBB.findDebugLoc(InsertPt), ExitInfo,
                    CurInfo);
      CurInfo = ExitInfo;
    }
  }

  if (UseStrictAsserts && CurInfo.isValid()) {
    const auto &Info = BlockInfo[MBB.getNumber()];
    if (CurInfo != Info.Exit) {
      LLVM_DEBUG(dbgs() << "in block " << printMBBReference(MBB) << "\n");
      LLVM_DEBUG(dbgs() << "  begin        state: " << Info.Pred << "\n");
      LLVM_DEBUG(dbgs() << "  expected end state: " << Info.Exit << "\n");
      LLVM_DEBUG(dbgs() << "  actual   end state: " << CurInfo << "\n");
    }
    assert(CurInfo == Info.Exit &&
           "InsertVSETVLI dataflow invariant violated");
  }
}

/// Return true if the VL value configured must be equal to the requested one.
static bool hasFixedResult(const VSETVLIInfo &Info, const RISCVSubtarget &ST) {
  if (!Info.hasAVLImm())
    // VLMAX is always the same value.
    // TODO: Could extend to other registers by looking at the associated vreg
    // def placement.
    return RISCV::X0 == Info.getAVLReg();

  unsigned AVL = Info.getAVLImm();
  unsigned SEW = Info.getSEW();
  unsigned AVLInBits = AVL * SEW;

  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = RISCVVType::decodeVLMUL(Info.getVLMUL());

  if (Fractional)
    return ST.getRealMinVLen() / LMul >= AVLInBits;
  return ST.getRealMinVLen() * LMul >= AVLInBits;
}

/// Perform simple partial redundancy elimination of the VSETVLI instructions
/// we're about to insert by looking for cases where we can PRE from the
/// beginning of one block to the end of one of its predecessors.  Specifically,
/// this is geared to catch the common case of a fixed length vsetvl in a single
/// block loop when it could execute once in the preheader instead.
void RISCVInsertVSETVLI::doPRE(MachineBasicBlock &MBB) {
  const MachineFunction &MF = *MBB.getParent();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();

  if (!BlockInfo[MBB.getNumber()].Pred.isUnknown())
    return;

  MachineBasicBlock *UnavailablePred = nullptr;
  VSETVLIInfo AvailableInfo;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    const VSETVLIInfo &PredInfo = BlockInfo[P->getNumber()].Exit;
    if (PredInfo.isUnknown()) {
      if (UnavailablePred)
        return;
      UnavailablePred = P;
    } else if (!AvailableInfo.isValid()) {
      AvailableInfo = PredInfo;
    } else if (AvailableInfo != PredInfo) {
      return;
    }
  }

  // Unreachable, single pred, or full redundancy. Note that FRE is handled by
  // phase 3.
  if (!UnavailablePred || !AvailableInfo.isValid())
    return;

  // Critical edge - TODO: consider splitting?
  if (UnavailablePred->succ_size() != 1)
    return;

  // If VL can be less than AVL, then we can't reduce the frequency of exec.
  if (!hasFixedResult(AvailableInfo, ST))
    return;

  // Does it actually let us remove an implicit transition in MBB?
  bool Found = false;
  for (auto &MI : MBB) {
    if (isVectorConfigInstr(MI))
      return;

    const uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      if (AvailableInfo != computeInfoForInstr(MI, TSFlags, MRI))
        return;
      Found = true;
      break;
    }
  }
  if (!Found)
    return;

  // Finally, update both data flow state and insert the actual vsetvli.
  // Doing both keeps the code in sync with the dataflow results, which
  // is critical for correctness of phase 3.
  auto OldInfo = BlockInfo[UnavailablePred->getNumber()].Exit;
  LLVM_DEBUG(dbgs() << "PRE VSETVLI from " << MBB.getName() << " to "
                    << UnavailablePred->getName() << " with state "
                    << AvailableInfo << "\n");
  BlockInfo[UnavailablePred->getNumber()].Exit = AvailableInfo;
  BlockInfo[MBB.getNumber()].Pred = AvailableInfo;

  // Note there's an implicit assumption here that terminators never use
  // or modify VL or VTYPE.  Also, fallthrough will return end().
  auto InsertPt = UnavailablePred->getFirstInstrTerminator();
  insertVSETVLI(*UnavailablePred, InsertPt,
                UnavailablePred->findDebugLoc(InsertPt),
                AvailableInfo, OldInfo);
}

static void doUnion(DemandedFields &A, DemandedFields B) {
  A.VL |= B.VL;
  A.SEW |= B.SEW;
  A.LMUL |= B.LMUL;
  A.SEWLMULRatio |= B.SEWLMULRatio;
  A.TailPolicy |= B.TailPolicy;
  A.MaskPolicy |= B.MaskPolicy;
}

// Return true if we can mutate PrevMI's VTYPE to match MI's
// without changing any the fields which have been used.
// TODO: Restructure code to allow code reuse between this and isCompatible
// above.
static bool canMutatePriorConfig(const MachineInstr &PrevMI,
                                 const MachineInstr &MI,
                                 const DemandedFields &Used) {
  // TODO: Extend this to handle cases where VL does change, but VL
  // has not been used.  (e.g. over a vmv.x.s)
  if (!isVLPreservingConfig(MI))
    // Note: `vsetvli x0, x0, vtype' is the canonical instruction
    // for this case.  If you find yourself wanting to add other forms
    // to this "unused VTYPE" case, we're probably missing a
    // canonicalization earlier.
    return false;

  if (!PrevMI.getOperand(2).isImm() || !MI.getOperand(2).isImm())
    return false;

  auto PriorVType = PrevMI.getOperand(2).getImm();
  auto VType = MI.getOperand(2).getImm();
  return areCompatibleVTYPEs(PriorVType, VType, Used);
}

void RISCVInsertVSETVLI::doLocalPostpass(MachineBasicBlock &MBB) {
  MachineInstr *PrevMI = nullptr;
  DemandedFields Used;
  SmallVector<MachineInstr*> ToDelete;
  for (MachineInstr &MI : MBB) {
    // Note: Must be *before* vsetvli handling to account for config cases
    // which only change some subfields.
    doUnion(Used, getDemanded(MI));

    if (!isVectorConfigInstr(MI))
      continue;

    if (PrevMI) {
      if (!Used.VL && !Used.usedVTYPE()) {
        ToDelete.push_back(PrevMI);
        // fallthrough
      } else if (canMutatePriorConfig(*PrevMI, MI, Used)) {
        PrevMI->getOperand(2).setImm(MI.getOperand(2).getImm());
        ToDelete.push_back(&MI);
        // Leave PrevMI unchanged
        continue;
      }
    }
    PrevMI = &MI;
    Used = getDemanded(MI);
    Register VRegDef = MI.getOperand(0).getReg();
    if (VRegDef != RISCV::X0 &&
        !(VRegDef.isVirtual() && MRI->use_nodbg_empty(VRegDef)))
      Used.VL = true;
  }

  for (auto *MI : ToDelete)
    MI->eraseFromParent();
}

void RISCVInsertVSETVLI::insertReadVL(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;
    if (RISCV::isFaultFirstLoad(MI)) {
      Register VLOutput = MI.getOperand(1).getReg();
      if (!MRI->use_nodbg_empty(VLOutput))
        BuildMI(MBB, I, MI.getDebugLoc(), TII->get(RISCV::PseudoReadVL),
                VLOutput);
      // We don't use the vl output of the VLEFF/VLSEGFF anymore.
      MI.getOperand(1).setReg(RISCV::X0);
    }
  }
}

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  LLVM_DEBUG(dbgs() << "Entering InsertVSETVLI for " << MF.getName() << "\n");

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();

  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  bool HaveVectorOp = false;

  // Phase 1 - determine how VL/VTYPE are affected by the each block.
  for (const MachineBasicBlock &MBB : MF) {
    HaveVectorOp |= computeVLVTYPEChanges(MBB);
    // Initial exit state is whatever change we found in the block.
    BlockData &BBInfo = BlockInfo[MBB.getNumber()];
    BBInfo.Exit = BBInfo.Change;
    LLVM_DEBUG(dbgs() << "Initial exit state of " << printMBBReference(MBB)
                      << " is " << BBInfo.Exit << "\n");

  }

  // If we didn't find any instructions that need VSETVLI, we're done.
  if (!HaveVectorOp) {
    BlockInfo.clear();
    return false;
  }

  // Phase 2 - determine the exit VL/VTYPE from each block. We add all
  // blocks to the list here, but will also add any that need to be revisited
  // during Phase 2 processing.
  for (const MachineBasicBlock &MBB : MF) {
    WorkList.push(&MBB);
    BlockInfo[MBB.getNumber()].InQueue = true;
  }
  while (!WorkList.empty()) {
    const MachineBasicBlock &MBB = *WorkList.front();
    WorkList.pop();
    computeIncomingVLVTYPE(MBB);
  }

  // Perform partial redundancy elimination of vsetvli transitions.
  for (MachineBasicBlock &MBB : MF)
    doPRE(MBB);

  // Phase 3 - add any vsetvli instructions needed in the block. Use the
  // Phase 2 information to avoid adding vsetvlis before the first vector
  // instruction in the block if the VL/VTYPE is satisfied by its
  // predecessors.
  for (MachineBasicBlock &MBB : MF)
    emitVSETVLIs(MBB);

  // Now that all vsetvlis are explicit, go through and do block local
  // DSE and peephole based demanded fields based transforms.  Note that
  // this *must* be done outside the main dataflow so long as we allow
  // any cross block analysis within the dataflow.  We can't have both
  // demanded fields based mutation and non-local analysis in the
  // dataflow at the same time without introducing inconsistencies.
  for (MachineBasicBlock &MBB : MF)
    doLocalPostpass(MBB);

  // Once we're fully done rewriting all the instructions, do a final pass
  // through to check for VSETVLIs which write to an unused destination.
  // For the non X0, X0 variant, we can replace the destination register
  // with X0 to reduce register pressure.  This is really a generic
  // optimization which can be applied to any dead def (TODO: generalize).
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == RISCV::PseudoVSETVLI ||
          MI.getOpcode() == RISCV::PseudoVSETIVLI) {
        Register VRegDef = MI.getOperand(0).getReg();
        if (VRegDef != RISCV::X0 && MRI->use_nodbg_empty(VRegDef))
          MI.getOperand(0).setReg(RISCV::X0);
      }
    }
  }

  // Insert PseudoReadVL after VLEFF/VLSEGFF and replace it with the vl output
  // of VLEFF/VLSEGFF.
  for (MachineBasicBlock &MBB : MF)
    insertReadVL(MBB);

  BlockInfo.clear();
  return HaveVectorOp;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
