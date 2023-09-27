//===-- RISCVBaseInfo.cpp - Top level definitions for RISC-V MC -----------===//
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

#include "RISCVBaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/RISCVISAInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

extern const SubtargetFeatureKV RISCVFeatureKV[RISCV::NumSubtargetFeatures];

namespace RISCVSysReg {
#define GET_SysRegsList_IMPL
#define GET_SiFiveRegsList_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCVSysReg

namespace RISCVInsnOpcode {
#define GET_RISCVOpcodesList_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCVInsnOpcode

namespace RISCVABI {
ABI computeTargetABI(const Triple &TT, const FeatureBitset &FeatureBits,
                     StringRef ABIName) {
  auto TargetABI = getTargetABI(ABIName);
  bool IsRV64 = TT.isArch64Bit();
  bool IsRVE = FeatureBits[RISCV::FeatureRVE];

  if (!ABIName.empty() && TargetABI == ABI_Unknown) {
    errs()
        << "'" << ABIName
        << "' is not a recognized ABI for this target (ignoring target-abi)\n";
  } else if (ABIName.startswith("ilp32") && IsRV64) {
    errs() << "32-bit ABIs are not supported for 64-bit targets (ignoring "
              "target-abi)\n";
    TargetABI = ABI_Unknown;
  } else if (ABIName.startswith("lp64") && !IsRV64) {
    errs() << "64-bit ABIs are not supported for 32-bit targets (ignoring "
              "target-abi)\n";
    TargetABI = ABI_Unknown;
  } else if (!IsRV64 && IsRVE && TargetABI != ABI_ILP32E &&
             TargetABI != ABI_Unknown) {
    // TODO: move this checking to RISCVTargetLowering and RISCVAsmParser
    errs()
        << "Only the ilp32e ABI is supported for RV32E (ignoring target-abi)\n";
    TargetABI = ABI_Unknown;
  } else if (IsRV64 && IsRVE && TargetABI != ABI_LP64E &&
             TargetABI != ABI_Unknown) {
    // TODO: move this checking to RISCVTargetLowering and RISCVAsmParser
    errs()
        << "Only the lp64e ABI is supported for RV64E (ignoring target-abi)\n";
    TargetABI = ABI_Unknown;
  }

  if (TargetABI != ABI_Unknown)
    return TargetABI;

  // If no explicit ABI is given, try to compute the default ABI.
  auto ISAInfo = RISCVFeatures::parseFeatureBits(IsRV64, FeatureBits);
  if (!ISAInfo)
    report_fatal_error(ISAInfo.takeError());
  return getTargetABI((*ISAInfo)->computeDefaultABI());
}

ABI getTargetABI(StringRef ABIName) {
  auto TargetABI = StringSwitch<ABI>(ABIName)
                       .Case("ilp32", ABI_ILP32)
                       .Case("ilp32f", ABI_ILP32F)
                       .Case("ilp32d", ABI_ILP32D)
                       .Case("ilp32e", ABI_ILP32E)
                       .Case("lp64", ABI_LP64)
                       .Case("lp64f", ABI_LP64F)
                       .Case("lp64d", ABI_LP64D)
                       .Case("lp64e", ABI_LP64E)
                       .Default(ABI_Unknown);
  return TargetABI;
}

// To avoid the BP value clobbered by a function call, we need to choose a
// callee saved register to save the value. RV32E only has X8 and X9 as callee
// saved registers and X8 will be used as fp. So we choose X9 as bp.
MCRegister getBPReg() { return RISCV::X9; }

// Returns the register holding shadow call stack pointer.
MCRegister getSCSPReg() { return RISCV::X3; }

} // namespace RISCVABI

namespace RISCVFeatures {

void validate(const Triple &TT, const FeatureBitset &FeatureBits) {
  if (TT.isArch64Bit() && !FeatureBits[RISCV::Feature64Bit])
    report_fatal_error("RV64 target requires an RV64 CPU");
  if (!TT.isArch64Bit() && !FeatureBits[RISCV::Feature32Bit])
    report_fatal_error("RV32 target requires an RV32 CPU");
  if (FeatureBits[RISCV::Feature32Bit] &&
      FeatureBits[RISCV::Feature64Bit])
    report_fatal_error("RV32 and RV64 can't be combined");
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
parseFeatureBits(bool IsRV64, const FeatureBitset &FeatureBits) {
  unsigned XLen = IsRV64 ? 64 : 32;
  std::vector<std::string> FeatureVector;
  // Convert FeatureBitset to FeatureVector.
  for (auto Feature : RISCVFeatureKV) {
    if (FeatureBits[Feature.Value] &&
        llvm::RISCVISAInfo::isSupportedExtensionFeature(Feature.Key))
      FeatureVector.push_back(std::string("+") + Feature.Key);
  }
  return llvm::RISCVISAInfo::parseFeatures(XLen, FeatureVector);
}

} // namespace RISCVFeatures

bool RISCVII::vectorInstUsesNBitsOfScalarOp(uint16_t Opcode, unsigned Bits,
                                            unsigned Log2SEW) {
  // TODO: Handle Zvbb instructions
  switch (Opcode) {
  default:
    return false;

  // 11.6. Vector Single-Width Shift Instructions
  case RISCV::VSLL_VX:
  case RISCV::VSRL_VX:
  case RISCV::VSRA_VX:
  // 12.4. Vector Single-Width Scaling Shift Instructions
  case RISCV::VSSRL_VX:
  case RISCV::VSSRA_VX:
    // Only the low lg2(SEW) bits of the shift-amount value are used.
    return Log2SEW <= Bits;

  // 11.7 Vector Narrowing Integer Right Shift Instructions
  case RISCV::VNSRL_WX:
  case RISCV::VNSRA_WX:
  // 12.5. Vector Narrowing Fixed-Point Clip Instructions
  case RISCV::VNCLIPU_WX:
  case RISCV::VNCLIP_WX:
    // Only the low lg2(2*SEW) bits of the shift-amount value are used.
    return (Log2SEW + 1) <= Bits;

  // 11.1. Vector Single-Width Integer Add and Subtract
  case RISCV::VADD_VX:
  case RISCV::VSUB_VX:
  case RISCV::VRSUB_VX:
  // 11.2. Vector Widening Integer Add/Subtract
  case RISCV::VWADDU_VX:
  case RISCV::VWSUBU_VX:
  case RISCV::VWADD_VX:
  case RISCV::VWSUB_VX:
  case RISCV::VWADDU_WX:
  case RISCV::VWSUBU_WX:
  case RISCV::VWADD_WX:
  case RISCV::VWSUB_WX:
  // 11.4. Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions
  case RISCV::VADC_VXM:
  case RISCV::VADC_VIM:
  case RISCV::VMADC_VXM:
  case RISCV::VMADC_VIM:
  case RISCV::VMADC_VX:
  case RISCV::VSBC_VXM:
  case RISCV::VMSBC_VXM:
  case RISCV::VMSBC_VX:
  // 11.5 Vector Bitwise Logical Instructions
  case RISCV::VAND_VX:
  case RISCV::VOR_VX:
  case RISCV::VXOR_VX:
  // 11.8. Vector Integer Compare Instructions
  case RISCV::VMSEQ_VX:
  case RISCV::VMSNE_VX:
  case RISCV::VMSLTU_VX:
  case RISCV::VMSLT_VX:
  case RISCV::VMSLEU_VX:
  case RISCV::VMSLE_VX:
  case RISCV::VMSGTU_VX:
  case RISCV::VMSGT_VX:
  // 11.9. Vector Integer Min/Max Instructions
  case RISCV::VMINU_VX:
  case RISCV::VMIN_VX:
  case RISCV::VMAXU_VX:
  case RISCV::VMAX_VX:
  // 11.10. Vector Single-Width Integer Multiply Instructions
  case RISCV::VMUL_VX:
  case RISCV::VMULH_VX:
  case RISCV::VMULHU_VX:
  case RISCV::VMULHSU_VX:
  // 11.11. Vector Integer Divide Instructions
  case RISCV::VDIVU_VX:
  case RISCV::VDIV_VX:
  case RISCV::VREMU_VX:
  case RISCV::VREM_VX:
  // 11.12. Vector Widening Integer Multiply Instructions
  case RISCV::VWMUL_VX:
  case RISCV::VWMULU_VX:
  case RISCV::VWMULSU_VX:
  // 11.13. Vector Single-Width Integer Multiply-Add Instructions
  case RISCV::VMACC_VX:
  case RISCV::VNMSAC_VX:
  case RISCV::VMADD_VX:
  case RISCV::VNMSUB_VX:
  // 11.14. Vector Widening Integer Multiply-Add Instructions
  case RISCV::VWMACCU_VX:
  case RISCV::VWMACC_VX:
  case RISCV::VWMACCSU_VX:
  case RISCV::VWMACCUS_VX:
  // 11.15. Vector Integer Merge Instructions
  case RISCV::VMERGE_VXM:
  // 11.16. Vector Integer Move Instructions
  case RISCV::VMV_V_X:
  // 12.1. Vector Single-Width Saturating Add and Subtract
  case RISCV::VSADDU_VX:
  case RISCV::VSADD_VX:
  case RISCV::VSSUBU_VX:
  case RISCV::VSSUB_VX:
  // 12.2. Vector Single-Width Averaging Add and Subtract
  case RISCV::VAADDU_VX:
  case RISCV::VAADD_VX:
  case RISCV::VASUBU_VX:
  case RISCV::VASUB_VX:
  // 12.3. Vector Single-Width Fractional Multiply with Rounding and Saturation
  case RISCV::VSMUL_VX:
  // 16.1. Integer Scalar Move Instructions
  case RISCV::VMV_S_X:
    return (1 << Log2SEW) <= Bits;
  }
}

// Encode VTYPE into the binary format used by the the VSETVLI instruction which
// is used by our MC layer representation.
//
// Bits | Name       | Description
// -----+------------+------------------------------------------------
// 7    | vma        | Vector mask agnostic
// 6    | vta        | Vector tail agnostic
// 5:3  | vsew[2:0]  | Standard element width (SEW) setting
// 2:0  | vlmul[2:0] | Vector register group multiplier (LMUL) setting
unsigned RISCVVType::encodeVTYPE(RISCVII::VLMUL VLMUL, unsigned SEW,
                                 bool TailAgnostic, bool MaskAgnostic) {
  assert(isValidSEW(SEW) && "Invalid SEW");
  unsigned VLMULBits = static_cast<unsigned>(VLMUL);
  unsigned VSEWBits = encodeSEW(SEW);
  unsigned VTypeI = (VSEWBits << 3) | (VLMULBits & 0x7);
  if (TailAgnostic)
    VTypeI |= 0x40;
  if (MaskAgnostic)
    VTypeI |= 0x80;

  return VTypeI;
}

std::pair<unsigned, bool> RISCVVType::decodeVLMUL(RISCVII::VLMUL VLMUL) {
  switch (VLMUL) {
  default:
    llvm_unreachable("Unexpected LMUL value!");
  case RISCVII::VLMUL::LMUL_1:
  case RISCVII::VLMUL::LMUL_2:
  case RISCVII::VLMUL::LMUL_4:
  case RISCVII::VLMUL::LMUL_8:
    return std::make_pair(1 << static_cast<unsigned>(VLMUL), false);
  case RISCVII::VLMUL::LMUL_F2:
  case RISCVII::VLMUL::LMUL_F4:
  case RISCVII::VLMUL::LMUL_F8:
    return std::make_pair(1 << (8 - static_cast<unsigned>(VLMUL)), true);
  }
}

void RISCVVType::printVType(unsigned VType, raw_ostream &OS) {
  unsigned Sew = getSEW(VType);
  OS << "e" << Sew;

  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(getVLMUL(VType));

  if (Fractional)
    OS << ", mf";
  else
    OS << ", m";
  OS << LMul;

  if (isTailAgnostic(VType))
    OS << ", ta";
  else
    OS << ", tu";

  if (isMaskAgnostic(VType))
    OS << ", ma";
  else
    OS << ", mu";
}

unsigned RISCVVType::getSEWLMULRatio(unsigned SEW, RISCVII::VLMUL VLMul) {
  unsigned LMul;
  bool Fractional;
  std::tie(LMul, Fractional) = decodeVLMUL(VLMul);

  // Convert LMul to a fixed point value with 3 fractional bits.
  LMul = Fractional ? (8 / LMul) : (LMul * 8);

  assert(SEW >= 8 && "Unexpected SEW value");
  return (SEW * 8) / LMul;
}

// Include the auto-generated portion of the compress emitter.
#define GEN_UNCOMPRESS_INSTR
#define GEN_COMPRESS_INSTR
#include "RISCVGenCompressInstEmitter.inc"

bool RISCVRVC::compress(MCInst &OutInst, const MCInst &MI,
                        const MCSubtargetInfo &STI) {
  return compressInst(OutInst, MI, STI);
}

bool RISCVRVC::uncompress(MCInst &OutInst, const MCInst &MI,
                          const MCSubtargetInfo &STI) {
  return uncompressInst(OutInst, MI, STI);
}

// Lookup table for fli.s for entries 2-31.
static constexpr std::pair<uint8_t, uint8_t> LoadFP32ImmArr[] = {
    {0b01101111, 0b00}, {0b01110000, 0b00}, {0b01110111, 0b00},
    {0b01111000, 0b00}, {0b01111011, 0b00}, {0b01111100, 0b00},
    {0b01111101, 0b00}, {0b01111101, 0b01}, {0b01111101, 0b10},
    {0b01111101, 0b11}, {0b01111110, 0b00}, {0b01111110, 0b01},
    {0b01111110, 0b10}, {0b01111110, 0b11}, {0b01111111, 0b00},
    {0b01111111, 0b01}, {0b01111111, 0b10}, {0b01111111, 0b11},
    {0b10000000, 0b00}, {0b10000000, 0b01}, {0b10000000, 0b10},
    {0b10000001, 0b00}, {0b10000010, 0b00}, {0b10000011, 0b00},
    {0b10000110, 0b00}, {0b10000111, 0b00}, {0b10001110, 0b00},
    {0b10001111, 0b00}, {0b11111111, 0b00}, {0b11111111, 0b10},
};

int RISCVLoadFPImm::getLoadFPImm(APFloat FPImm) {
  assert((&FPImm.getSemantics() == &APFloat::IEEEsingle() ||
          &FPImm.getSemantics() == &APFloat::IEEEdouble() ||
          &FPImm.getSemantics() == &APFloat::IEEEhalf()) &&
         "Unexpected semantics");

  // Handle the minimum normalized value which is different for each type.
  if (FPImm.isSmallestNormalized())
    return 1;

  // Convert to single precision to use its lookup table.
  bool LosesInfo;
  APFloat::opStatus Status = FPImm.convert(
      APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &LosesInfo);
  if (Status != APFloat::opOK || LosesInfo)
    return -1;

  APInt Imm = FPImm.bitcastToAPInt();

  if (Imm.extractBitsAsZExtValue(21, 0) != 0)
    return -1;

  bool Sign = Imm.extractBitsAsZExtValue(1, 31);
  uint8_t Mantissa = Imm.extractBitsAsZExtValue(2, 21);
  uint8_t Exp = Imm.extractBitsAsZExtValue(8, 23);

  auto EMI = llvm::lower_bound(LoadFP32ImmArr, std::make_pair(Exp, Mantissa));
  if (EMI == std::end(LoadFP32ImmArr) || EMI->first != Exp ||
      EMI->second != Mantissa)
    return -1;

  // Table doesn't have entry 0 or 1.
  int Entry = std::distance(std::begin(LoadFP32ImmArr), EMI) + 2;

  // The only legal negative value is -1.0(entry 0). 1.0 is entry 16.
  if (Sign) {
    if (Entry == 16)
      return 0;
    return false;
  }

  return Entry;
}

float RISCVLoadFPImm::getFPImm(unsigned Imm) {
  assert(Imm != 1 && Imm != 30 && Imm != 31 && "Unsupported immediate");

  // Entry 0 is -1.0, the only negative value. Entry 16 is 1.0.
  uint32_t Sign = 0;
  if (Imm == 0) {
    Sign = 0b1;
    Imm = 16;
  }

  uint32_t Exp = LoadFP32ImmArr[Imm - 2].first;
  uint32_t Mantissa = LoadFP32ImmArr[Imm - 2].second;

  uint32_t I = Sign << 31 | Exp << 23 | Mantissa << 21;
  return bit_cast<float>(I);
}

void RISCVZC::printRlist(unsigned SlistEncode, raw_ostream &OS) {
  OS << "{ra";
  if (SlistEncode > 4) {
    OS << ", s0";
    if (SlistEncode == 15)
      OS << "-s11";
    else if (SlistEncode > 5 && SlistEncode <= 14)
      OS << "-s" << (SlistEncode - 5);
  }
  OS << "}";
}

void RISCVZC::printSpimm(int64_t Spimm, raw_ostream &OS) { OS << Spimm; }

} // namespace llvm
