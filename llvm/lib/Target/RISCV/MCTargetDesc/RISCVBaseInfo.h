//===-- RISCVBaseInfo.h - Top level definitions for RISC-V MC ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone enum definitions for the RISC-V target
// useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVBASEINFO_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVBASEINFO_H

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/SubtargetFeature.h"

namespace llvm {

// RISCVII - This namespace holds all of the target specific flags that
// instruction info tracks. All definitions must match RISCVInstrFormats.td.
namespace RISCVII {
enum {
  InstFormatPseudo = 0,
  InstFormatR = 1,
  InstFormatR4 = 2,
  InstFormatI = 3,
  InstFormatS = 4,
  InstFormatB = 5,
  InstFormatU = 6,
  InstFormatJ = 7,
  InstFormatCR = 8,
  InstFormatCI = 9,
  InstFormatCSS = 10,
  InstFormatCIW = 11,
  InstFormatCL = 12,
  InstFormatCS = 13,
  InstFormatCA = 14,
  InstFormatCB = 15,
  InstFormatCJ = 16,
  InstFormatCU = 17,
  InstFormatCLB = 18,
  InstFormatCLH = 19,
  InstFormatCSB = 20,
  InstFormatCSH = 21,
  InstFormatQC_EAI = 22,
  InstFormatQC_EI = 23,
  InstFormatQC_EB = 24,
  InstFormatQC_EJ = 25,
  InstFormatQC_ES = 26,
  InstFormatNDS_BRANCH_10 = 27,
  InstFormatOther = 31,

  InstFormatMask = 31,
  InstFormatShift = 0,

  ConstraintShift = InstFormatShift + 5,
  VS2Constraint = 0b001 << ConstraintShift,
  VS1Constraint = 0b010 << ConstraintShift,
  VMConstraint = 0b100 << ConstraintShift,
  ConstraintMask = 0b111 << ConstraintShift,

  VLMulShift = ConstraintShift + 3,
  VLMulMask = 0b111 << VLMulShift,

  // Is this a _TIED vector pseudo instruction. For these instructions we
  // shouldn't skip the tied operand when converting to MC instructions.
  IsTiedPseudoShift = VLMulShift + 3,
  IsTiedPseudoMask = 1 << IsTiedPseudoShift,

  // Does this instruction have a SEW operand. It will be the last explicit
  // operand unless there is a vector policy operand. Used by RVV Pseudos.
  HasSEWOpShift = IsTiedPseudoShift + 1,
  HasSEWOpMask = 1 << HasSEWOpShift,

  // Does this instruction have a VL operand. It will be the second to last
  // explicit operand unless there is a vector policy operand. Used by RVV
  // Pseudos.
  HasVLOpShift = HasSEWOpShift + 1,
  HasVLOpMask = 1 << HasVLOpShift,

  // Does this instruction have a vector policy operand. It will be the last
  // explicit operand. Used by RVV Pseudos.
  HasVecPolicyOpShift = HasVLOpShift + 1,
  HasVecPolicyOpMask = 1 << HasVecPolicyOpShift,

  // Is this instruction a vector widening reduction instruction. Used by RVV
  // Pseudos.
  IsRVVWideningReductionShift = HasVecPolicyOpShift + 1,
  IsRVVWideningReductionMask = 1 << IsRVVWideningReductionShift,

  // Does this instruction care about mask policy. If it is not, the mask policy
  // could be either agnostic or undisturbed. For example, unmasked, store, and
  // reduction operations result would not be affected by mask policy, so
  // compiler has free to select either one.
  UsesMaskPolicyShift = IsRVVWideningReductionShift + 1,
  UsesMaskPolicyMask = 1 << UsesMaskPolicyShift,

  // Indicates that the result can be considered sign extended from bit 31. Some
  // instructions with this flag aren't W instructions, but are either sign
  // extended from a smaller size, always outputs a small integer, or put zeros
  // in bits 63:31. Used by the SExtWRemoval pass.
  IsSignExtendingOpWShift = UsesMaskPolicyShift + 1,
  IsSignExtendingOpWMask = 1ULL << IsSignExtendingOpWShift,

  HasRoundModeOpShift = IsSignExtendingOpWShift + 1,
  HasRoundModeOpMask = 1 << HasRoundModeOpShift,

  UsesVXRMShift = HasRoundModeOpShift + 1,
  UsesVXRMMask = 1 << UsesVXRMShift,

  // Indicates whether these instructions can partially overlap between source
  // registers and destination registers according to the vector spec.
  // 0 -> not a vector pseudo
  // 1 -> default value for vector pseudos. not widening or narrowing.
  // 2 -> narrowing case
  // 3 -> widening case
  TargetOverlapConstraintTypeShift = UsesVXRMShift + 1,
  TargetOverlapConstraintTypeMask = 3ULL << TargetOverlapConstraintTypeShift,

  ElementsDependOnVLShift = TargetOverlapConstraintTypeShift + 2,
  ElementsDependOnVLMask = 1ULL << ElementsDependOnVLShift,

  ElementsDependOnMaskShift = ElementsDependOnVLShift + 1,
  ElementsDependOnMaskMask = 1ULL << ElementsDependOnMaskShift,

  // Indicates the EEW of a vector instruction's destination operand.
  // 0 -> 1
  // 1 -> SEW
  // 2 -> SEW * 2
  // 3 -> SEW * 4
  DestEEWShift = ElementsDependOnMaskShift + 1,
  DestEEWMask = 3ULL << DestEEWShift,

  ReadsPastVLShift = DestEEWShift + 2,
  ReadsPastVLMask = 1ULL << ReadsPastVLShift,
};

// Helper functions to read TSFlags.
/// \returns the format of the instruction.
static inline unsigned getFormat(uint64_t TSFlags) {
  return (TSFlags & InstFormatMask) >> InstFormatShift;
}
/// \returns the LMUL for the instruction.
static inline RISCVVType::VLMUL getLMul(uint64_t TSFlags) {
  return static_cast<RISCVVType::VLMUL>((TSFlags & VLMulMask) >> VLMulShift);
}
/// \returns true if this a _TIED pseudo.
static inline bool isTiedPseudo(uint64_t TSFlags) {
  return TSFlags & IsTiedPseudoMask;
}
/// \returns true if there is a SEW operand for the instruction.
static inline bool hasSEWOp(uint64_t TSFlags) {
  return TSFlags & HasSEWOpMask;
}
/// \returns true if there is a VL operand for the instruction.
static inline bool hasVLOp(uint64_t TSFlags) {
  return TSFlags & HasVLOpMask;
}
/// \returns true if there is a vector policy operand for this instruction.
static inline bool hasVecPolicyOp(uint64_t TSFlags) {
  return TSFlags & HasVecPolicyOpMask;
}
/// \returns true if it is a vector widening reduction instruction.
static inline bool isRVVWideningReduction(uint64_t TSFlags) {
  return TSFlags & IsRVVWideningReductionMask;
}
/// \returns true if mask policy is valid for the instruction.
static inline bool usesMaskPolicy(uint64_t TSFlags) {
  return TSFlags & UsesMaskPolicyMask;
}

/// \returns true if there is a rounding mode operand for this instruction
static inline bool hasRoundModeOp(uint64_t TSFlags) {
  return TSFlags & HasRoundModeOpMask;
}

/// \returns true if this instruction uses vxrm
static inline bool usesVXRM(uint64_t TSFlags) { return TSFlags & UsesVXRMMask; }

/// \returns true if the elements in the body are affected by VL,
/// e.g. vslide1down.vx/vredsum.vs/viota.m
static inline bool elementsDependOnVL(uint64_t TSFlags) {
  return TSFlags & ElementsDependOnVLMask;
}

/// \returns true if the elements in the body are affected by the mask,
/// e.g. vredsum.vs/viota.m
static inline bool elementsDependOnMask(uint64_t TSFlags) {
  return TSFlags & ElementsDependOnMaskMask;
}

/// \returns true if the instruction may read elements past VL, e.g.
/// vslidedown/vrgather
static inline bool readsPastVL(uint64_t TSFlags) {
  return TSFlags & ReadsPastVLMask;
}

static inline unsigned getVLOpNum(const MCInstrDesc &Desc) {
  const uint64_t TSFlags = Desc.TSFlags;
  // This method is only called if we expect to have a VL operand, and all
  // instructions with VL also have SEW.
  assert(hasSEWOp(TSFlags) && hasVLOp(TSFlags));
  unsigned Offset = 2;
  if (hasVecPolicyOp(TSFlags))
    Offset = 3;
  return Desc.getNumOperands() - Offset;
}

static inline MCRegister
getTailExpandUseRegNo(const FeatureBitset &FeatureBits) {
  // For Zicfilp, PseudoTAIL should be expanded to a software guarded branch.
  // It means to use t2(x7) as rs1 of JALR to expand PseudoTAIL.
  return FeatureBits[RISCV::FeatureStdExtZicfilp] ? RISCV::X7 : RISCV::X6;
}

static inline unsigned getSEWOpNum(const MCInstrDesc &Desc) {
  const uint64_t TSFlags = Desc.TSFlags;
  assert(hasSEWOp(TSFlags));
  unsigned Offset = 1;
  if (hasVecPolicyOp(TSFlags))
    Offset = 2;
  return Desc.getNumOperands() - Offset;
}

static inline unsigned getVecPolicyOpNum(const MCInstrDesc &Desc) {
  assert(hasVecPolicyOp(Desc.TSFlags));
  return Desc.getNumOperands() - 1;
}

/// \returns  the index to the rounding mode immediate value if any, otherwise
/// returns -1.
static inline int getFRMOpNum(const MCInstrDesc &Desc) {
  const uint64_t TSFlags = Desc.TSFlags;
  if (!hasRoundModeOp(TSFlags) || usesVXRM(TSFlags))
    return -1;

  // The operand order
  // --------------------------------------
  // | n-1 (if any)   | n-2  | n-3 | n-4 |
  // | policy         | sew  | vl  | frm |
  // --------------------------------------
  return getVLOpNum(Desc) - 1;
}

/// \returns  the index to the rounding mode immediate value if any, otherwise
/// returns -1.
static inline int getVXRMOpNum(const MCInstrDesc &Desc) {
  const uint64_t TSFlags = Desc.TSFlags;
  if (!hasRoundModeOp(TSFlags) || !usesVXRM(TSFlags))
    return -1;
  // The operand order
  // --------------------------------------
  // | n-1 (if any)   | n-2  | n-3 | n-4  |
  // | policy         | sew  | vl  | vxrm |
  // --------------------------------------
  return getVLOpNum(Desc) - 1;
}

// Is the first def operand tied to the first use operand. This is true for
// vector pseudo instructions that have a merge operand for tail/mask
// undisturbed. It's also true for vector FMA instructions where one of the
// operands is also the destination register.
static inline bool isFirstDefTiedToFirstUse(const MCInstrDesc &Desc) {
  return Desc.getNumDefs() < Desc.getNumOperands() &&
         Desc.getOperandConstraint(Desc.getNumDefs(), MCOI::TIED_TO) == 0;
}

// RISC-V Specific Machine Operand Flags
enum {
  MO_None = 0,
  MO_CALL = 1,
  MO_LO = 3,
  MO_HI = 4,
  MO_PCREL_LO = 5,
  MO_PCREL_HI = 6,
  MO_GOT_HI = 7,
  MO_TPREL_LO = 8,
  MO_TPREL_HI = 9,
  MO_TPREL_ADD = 10,
  MO_TLS_GOT_HI = 11,
  MO_TLS_GD_HI = 12,
  MO_TLSDESC_HI = 13,
  MO_TLSDESC_LOAD_LO = 14,
  MO_TLSDESC_ADD_LO = 15,
  MO_TLSDESC_CALL = 16,

  // Used to differentiate between target-specific "direct" flags and "bitmask"
  // flags. A machine operand can only have one "direct" flag, but can have
  // multiple "bitmask" flags.
  MO_DIRECT_FLAG_MASK = 31
};
} // namespace RISCVII

namespace RISCVOp {
enum OperandType : unsigned {
  OPERAND_FIRST_RISCV_IMM = MCOI::OPERAND_FIRST_TARGET,
  OPERAND_UIMM1 = OPERAND_FIRST_RISCV_IMM,
  OPERAND_UIMM2,
  OPERAND_UIMM2_LSB0,
  OPERAND_UIMM3,
  OPERAND_UIMM4,
  OPERAND_UIMM5,
  OPERAND_UIMM5_NONZERO,
  OPERAND_UIMM5_GT3,
  OPERAND_UIMM5_PLUS1,
  OPERAND_UIMM5_GE6_PLUS1,
  OPERAND_UIMM5_LSB0,
  OPERAND_UIMM5_SLIST,
  OPERAND_UIMM6,
  OPERAND_UIMM6_LSB0,
  OPERAND_UIMM7,
  OPERAND_UIMM7_LSB00,
  OPERAND_UIMM7_LSB000,
  OPERAND_UIMM8_LSB00,
  OPERAND_UIMM8,
  OPERAND_UIMM8_LSB000,
  OPERAND_UIMM8_GE32,
  OPERAND_UIMM9_LSB000,
  OPERAND_UIMM9,
  OPERAND_UIMM10,
  OPERAND_UIMM10_LSB00_NONZERO,
  OPERAND_UIMM11,
  OPERAND_UIMM12,
  OPERAND_UIMM14_LSB00,
  OPERAND_UIMM16,
  OPERAND_UIMM16_NONZERO,
  OPERAND_UIMM20,
  OPERAND_UIMMLOG2XLEN,
  OPERAND_UIMMLOG2XLEN_NONZERO,
  OPERAND_UIMM32,
  OPERAND_UIMM48,
  OPERAND_UIMM64,
  OPERAND_THREE,
  OPERAND_FOUR,
  OPERAND_SIMM5,
  OPERAND_SIMM5_NONZERO,
  OPERAND_SIMM5_PLUS1,
  OPERAND_SIMM6,
  OPERAND_SIMM6_NONZERO,
  OPERAND_SIMM8,
  OPERAND_SIMM10,
  OPERAND_SIMM10_LSB0000_NONZERO,
  OPERAND_SIMM11,
  OPERAND_SIMM12,
  OPERAND_SIMM12_LSB00000,
  OPERAND_SIMM16,
  OPERAND_SIMM16_NONZERO,
  OPERAND_SIMM20_LI,
  OPERAND_SIMM26,
  OPERAND_BARE_SIMM32,
  OPERAND_CLUI_IMM,
  OPERAND_VTYPEI10,
  OPERAND_VTYPEI11,
  OPERAND_RVKRNUM,
  OPERAND_RVKRNUM_0_7,
  OPERAND_RVKRNUM_1_10,
  OPERAND_RVKRNUM_2_14,
  OPERAND_RLIST,
  OPERAND_RLIST_S0,
  OPERAND_STACKADJ,
  // Operand is a 3-bit rounding mode, '111' indicates FRM register.
  // Represents 'frm' argument passing to floating-point operations.
  OPERAND_FRMARG,
  // Operand is a 3-bit rounding mode where only RTZ is valid.
  OPERAND_RTZARG,
  // Condition code used by select and short forward branch pseudos.
  OPERAND_COND_CODE,
  // Vector policy operand.
  OPERAND_VEC_POLICY,
  // Vector SEW operand. Stores in log2(SEW).
  OPERAND_SEW,
  // Special SEW for mask only instructions. Always 0.
  OPERAND_SEW_MASK,
  // Vector rounding mode for VXRM or FRM.
  OPERAND_VEC_RM,
  OPERAND_LAST_RISCV_IMM = OPERAND_VEC_RM,
  // Operand is either a register or uimm5, this is used by V extension pseudo
  // instructions to represent a value that be passed as AVL to either vsetvli
  // or vsetivli.
  OPERAND_AVL,
};
} // namespace RISCVOp

// Describes the predecessor/successor bits used in the FENCE instruction.
namespace RISCVFenceField {
enum FenceField {
  I = 8,
  O = 4,
  R = 2,
  W = 1
};
}

// Describes the supported floating point rounding mode encodings.
namespace RISCVFPRndMode {
enum RoundingMode {
  RNE = 0,
  RTZ = 1,
  RDN = 2,
  RUP = 3,
  RMM = 4,
  DYN = 7,
  Invalid
};

inline static StringRef roundingModeToString(RoundingMode RndMode) {
  switch (RndMode) {
  default:
    llvm_unreachable("Unknown floating point rounding mode");
  case RISCVFPRndMode::RNE:
    return "rne";
  case RISCVFPRndMode::RTZ:
    return "rtz";
  case RISCVFPRndMode::RDN:
    return "rdn";
  case RISCVFPRndMode::RUP:
    return "rup";
  case RISCVFPRndMode::RMM:
    return "rmm";
  case RISCVFPRndMode::DYN:
    return "dyn";
  }
}

inline static RoundingMode stringToRoundingMode(StringRef Str) {
  return StringSwitch<RoundingMode>(Str)
      .Case("rne", RISCVFPRndMode::RNE)
      .Case("rtz", RISCVFPRndMode::RTZ)
      .Case("rdn", RISCVFPRndMode::RDN)
      .Case("rup", RISCVFPRndMode::RUP)
      .Case("rmm", RISCVFPRndMode::RMM)
      .Case("dyn", RISCVFPRndMode::DYN)
      .Default(RISCVFPRndMode::Invalid);
}

inline static bool isValidRoundingMode(unsigned Mode) {
  switch (Mode) {
  default:
    return false;
  case RISCVFPRndMode::RNE:
  case RISCVFPRndMode::RTZ:
  case RISCVFPRndMode::RDN:
  case RISCVFPRndMode::RUP:
  case RISCVFPRndMode::RMM:
  case RISCVFPRndMode::DYN:
    return true;
  }
}
} // namespace RISCVFPRndMode

namespace RISCVVXRndMode {
enum RoundingMode {
  RNU = 0,
  RNE = 1,
  RDN = 2,
  ROD = 3,
  Invalid
};

inline static StringRef roundingModeToString(RoundingMode RndMode) {
  switch (RndMode) {
  default:
    llvm_unreachable("Unknown vector fixed-point rounding mode");
  case RISCVVXRndMode::RNU:
    return "rnu";
  case RISCVVXRndMode::RNE:
    return "rne";
  case RISCVVXRndMode::RDN:
    return "rdn";
  case RISCVVXRndMode::ROD:
    return "rod";
  }
}

inline static RoundingMode stringToRoundingMode(StringRef Str) {
  return StringSwitch<RoundingMode>(Str)
      .Case("rnu", RISCVVXRndMode::RNU)
      .Case("rne", RISCVVXRndMode::RNE)
      .Case("rdn", RISCVVXRndMode::RDN)
      .Case("rod", RISCVVXRndMode::ROD)
      .Default(RISCVVXRndMode::Invalid);
}

inline static bool isValidRoundingMode(unsigned Mode) {
  switch (Mode) {
  default:
    return false;
  case RISCVVXRndMode::RNU:
  case RISCVVXRndMode::RNE:
  case RISCVVXRndMode::RDN:
  case RISCVVXRndMode::ROD:
    return true;
  }
}
} // namespace RISCVVXRndMode

namespace RISCVExceptFlags {
enum ExceptionFlag {
  NX = 0x01, // Inexact
  UF = 0x02, // Underflow
  OF = 0x04, // Overflow
  DZ = 0x08, // Divide by zero
  NV = 0x10, // Invalid operation
  ALL = 0x1F // Mask for all accrued exception flags
};
}

//===----------------------------------------------------------------------===//
// Floating-point Immediates
//

namespace RISCVLoadFPImm {
float getFPImm(unsigned Imm);

/// getLoadFPImm - Return a 5-bit binary encoding of the floating-point
/// immediate value. If the value cannot be represented as a 5-bit binary
/// encoding, then return -1.
int getLoadFPImm(APFloat FPImm);
} // namespace RISCVLoadFPImm

namespace RISCVSysReg {
struct SysReg {
  const char Name[32];
  unsigned Encoding;
  // FIXME: add these additional fields when needed.
  // Privilege Access: Read, Write, Read-Only.
  // unsigned ReadWrite;
  // Privilege Mode: User, System or Machine.
  // unsigned Mode;
  // Check field name.
  // unsigned Extra;
  // Register number without the privilege bits.
  // unsigned Number;
  FeatureBitset FeaturesRequired;
  bool IsRV32Only;
  bool IsAltName;
  bool IsDeprecatedName;

  bool haveRequiredFeatures(const FeatureBitset &ActiveFeatures) const {
    // Not in 32-bit mode.
    if (IsRV32Only && ActiveFeatures[RISCV::Feature64Bit])
      return false;
    // No required feature associated with the system register.
    if (FeaturesRequired.none())
      return true;
    return (FeaturesRequired & ActiveFeatures) == FeaturesRequired;
  }
};

#define GET_SysRegEncodings_DECL
#define GET_SysRegsList_DECL
#include "RISCVGenSearchableTables.inc"
} // end namespace RISCVSysReg

namespace RISCVInsnOpcode {
struct RISCVOpcode {
  char Name[10];
  uint8_t Value;
};

#define GET_RISCVOpcodesList_DECL
#include "RISCVGenSearchableTables.inc"
} // end namespace RISCVInsnOpcode

namespace RISCVABI {

enum ABI {
  ABI_ILP32,
  ABI_ILP32F,
  ABI_ILP32D,
  ABI_ILP32E,
  ABI_LP64,
  ABI_LP64F,
  ABI_LP64D,
  ABI_LP64E,
  ABI_Unknown
};

// Returns the target ABI, or else a StringError if the requested ABIName is
// not supported for the given TT and FeatureBits combination.
ABI computeTargetABI(const Triple &TT, const FeatureBitset &FeatureBits,
                     StringRef ABIName);

ABI getTargetABI(StringRef ABIName);

// Returns the register used to hold the stack pointer after realignment.
MCRegister getBPReg();

// Returns the register holding shadow call stack pointer.
MCRegister getSCSPReg();

} // namespace RISCVABI

namespace RISCVFeatures {

// Validates if the given combination of features are valid for the target
// triple. Exits with report_fatal_error if not.
void validate(const Triple &TT, const FeatureBitset &FeatureBits);

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
parseFeatureBits(bool IsRV64, const FeatureBitset &FeatureBits);

} // namespace RISCVFeatures

namespace RISCVRVC {
bool compress(MCInst &OutInst, const MCInst &MI, const MCSubtargetInfo &STI);
bool uncompress(MCInst &OutInst, const MCInst &MI, const MCSubtargetInfo &STI);
} // namespace RISCVRVC

namespace RISCVZC {
enum RLISTENCODE {
  RA = 4,
  RA_S0,
  RA_S0_S1,
  RA_S0_S2,
  RA_S0_S3,
  RA_S0_S4,
  RA_S0_S5,
  RA_S0_S6,
  RA_S0_S7,
  RA_S0_S8,
  RA_S0_S9,
  // note - to include s10, s11 must also be included
  RA_S0_S11,
  INVALID_RLIST,
};

inline unsigned encodeRegList(MCRegister EndReg, bool IsRVE = false) {
  assert((!IsRVE || EndReg <= RISCV::X9) && "Invalid Rlist for RV32E");
  switch (EndReg) {
  case RISCV::X1:
    return RLISTENCODE::RA;
  case RISCV::X8:
    return RLISTENCODE::RA_S0;
  case RISCV::X9:
    return RLISTENCODE::RA_S0_S1;
  case RISCV::X18:
    return RLISTENCODE::RA_S0_S2;
  case RISCV::X19:
    return RLISTENCODE::RA_S0_S3;
  case RISCV::X20:
    return RLISTENCODE::RA_S0_S4;
  case RISCV::X21:
    return RLISTENCODE::RA_S0_S5;
  case RISCV::X22:
    return RLISTENCODE::RA_S0_S6;
  case RISCV::X23:
    return RLISTENCODE::RA_S0_S7;
  case RISCV::X24:
    return RLISTENCODE::RA_S0_S8;
  case RISCV::X25:
    return RLISTENCODE::RA_S0_S9;
  case RISCV::X27:
    return RLISTENCODE::RA_S0_S11;
  default:
    llvm_unreachable("Undefined input.");
  }
}

inline static unsigned encodeRegListNumRegs(unsigned NumRegs) {
  assert(NumRegs > 0 && NumRegs < 14 && NumRegs != 12 &&
         "Unexpected number of registers");
  if (NumRegs == 13)
    return RLISTENCODE::RA_S0_S11;

  return RLISTENCODE::RA + (NumRegs - 1);
}

inline static unsigned getStackAdjBase(unsigned RlistVal, bool IsRV64) {
  assert(RlistVal >= RLISTENCODE::RA && RlistVal <= RLISTENCODE::RA_S0_S11 &&
         "Invalid Rlist");
  unsigned NumRegs = (RlistVal - RLISTENCODE::RA) + 1;
  // s10 and s11 are saved together.
  if (RlistVal == RLISTENCODE::RA_S0_S11)
    ++NumRegs;

  unsigned RegSize = IsRV64 ? 8 : 4;
  return alignTo(NumRegs * RegSize, 16);
}

void printRegList(unsigned RlistEncode, raw_ostream &OS);
} // namespace RISCVZC

namespace RISCVVInversePseudosTable {
struct PseudoInfo {
  uint16_t Pseudo;
  uint16_t BaseInstr;
  uint8_t VLMul;
  uint8_t SEW;
};

#define GET_RISCVVInversePseudosTable_DECL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCVVInversePseudosTable

namespace RISCV {
struct VLSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VSSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSXSEGPseudo {
  uint16_t NF : 4;
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct VLEPseudo {
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t FF : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VSEPseudo {
  uint16_t Masked : 1;
  uint16_t Strided : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

struct VLX_VSXPseudo {
  uint16_t Masked : 1;
  uint16_t Ordered : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t IndexLMUL : 3;
  uint16_t Pseudo;
};

struct NDSVLNPseudo {
  uint16_t Masked : 1;
  uint16_t Unsigned : 1;
  uint16_t Log2SEW : 3;
  uint16_t LMUL : 3;
  uint16_t Pseudo;
};

#define GET_RISCVVSSEGTable_DECL
#define GET_RISCVVLSEGTable_DECL
#define GET_RISCVVLXSEGTable_DECL
#define GET_RISCVVSXSEGTable_DECL
#define GET_RISCVVLETable_DECL
#define GET_RISCVVSETable_DECL
#define GET_RISCVVLXTable_DECL
#define GET_RISCVVSXTable_DECL
#define GET_RISCVNDSVLNTable_DECL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCV

} // namespace llvm

#endif
