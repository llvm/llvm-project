//===-- AMDGPUBaseInfo.cpp - AMDGPU Base encoding information--------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "AMDGPUBaseInfo.h"
#include "AMDGPU.h"
#include "SIDefines.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_ENUM

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"
#undef GET_REGINFO_ENUM

namespace llvm {
namespace AMDGPU {

IsaVersion getIsaVersion(const FeatureBitset &Features) {

  if (Features.test(FeatureISAVersion7_0_0))
    return {7, 0, 0};

  if (Features.test(FeatureISAVersion7_0_1))
    return {7, 0, 1};

  if (Features.test(FeatureISAVersion8_0_0))
    return {8, 0, 0};

  if (Features.test(FeatureISAVersion8_0_1))
    return {8, 0, 1};

  if (Features.test(FeatureISAVersion8_0_3))
    return {8, 0, 3};

  return {0, 0, 0};
}

void initDefaultAMDKernelCodeT(amd_kernel_code_t &Header,
                               const FeatureBitset &Features) {

  IsaVersion ISA = getIsaVersion(Features);

  memset(&Header, 0, sizeof(Header));

  Header.amd_kernel_code_version_major = 1;
  Header.amd_kernel_code_version_minor = 0;
  Header.amd_machine_kind = 1; // AMD_MACHINE_KIND_AMDGPU
  Header.amd_machine_version_major = ISA.Major;
  Header.amd_machine_version_minor = ISA.Minor;
  Header.amd_machine_version_stepping = ISA.Stepping;
  Header.kernel_code_entry_byte_offset = sizeof(Header);
  // wavefront_size is specified as a power of 2: 2^6 = 64 threads.
  Header.wavefront_size = 6;
  // These alignment values are specified in powers of two, so alignment =
  // 2^n.  The minimum alignment is 2^4 = 16.
  Header.kernarg_segment_alignment = 4;
  Header.group_segment_alignment = 4;
  Header.private_segment_alignment = 4;
}

MCSection *getHSATextSection(MCContext &Ctx) {
  return Ctx.getELFSection(".hsatext", ELF::SHT_PROGBITS,
                           ELF::SHF_ALLOC | ELF::SHF_WRITE |
                           ELF::SHF_EXECINSTR |
                           ELF::SHF_AMDGPU_HSA_AGENT |
                           ELF::SHF_AMDGPU_HSA_CODE);
}

MCSection *getHSADataGlobalAgentSection(MCContext &Ctx) {
  return Ctx.getELFSection(".hsadata_global_agent", ELF::SHT_PROGBITS,
                           ELF::SHF_ALLOC | ELF::SHF_WRITE |
                           ELF::SHF_AMDGPU_HSA_GLOBAL |
                           ELF::SHF_AMDGPU_HSA_AGENT);
}

MCSection *getHSADataGlobalProgramSection(MCContext &Ctx) {
  return  Ctx.getELFSection(".hsadata_global_program", ELF::SHT_PROGBITS,
                            ELF::SHF_ALLOC | ELF::SHF_WRITE |
                            ELF::SHF_AMDGPU_HSA_GLOBAL);
}

MCSection *getHSARodataReadonlyAgentSection(MCContext &Ctx) {
  return Ctx.getELFSection(".hsarodata_readonly_agent", ELF::SHT_PROGBITS,
                           ELF::SHF_ALLOC | ELF::SHF_AMDGPU_HSA_READONLY |
                           ELF::SHF_AMDGPU_HSA_AGENT);
}

bool isGroupSegment(const GlobalValue *GV) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS;
}

bool isGlobalSegment(const GlobalValue *GV) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS;
}

bool isReadOnlySegment(const GlobalValue *GV) {
  return GV->getType()->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS;
}

int getIntegerAttribute(const Function &F, StringRef Name, int Default) {
  Attribute A = F.getFnAttribute(Name);
  int Result = Default;

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, Result)) {
      LLVMContext &Ctx = F.getContext();
      Ctx.emitError("can't parse integer attribute " + Name);
    }
  }

  return Result;
}

std::pair<int, int> getIntegerPairAttribute(const Function &F,
                                            StringRef Name,
                                            std::pair<int, int> Default,
                                            bool OnlyFirstRequired) {
  Attribute A = F.getFnAttribute(Name);
  if (!A.isStringAttribute())
    return Default;

  LLVMContext &Ctx = F.getContext();
  std::pair<int, int> Ints = Default;
  std::pair<StringRef, StringRef> Strs = A.getValueAsString().split(',');
  if (Strs.first.trim().getAsInteger(0, Ints.first)) {
    Ctx.emitError("can't parse first integer attribute " + Name);
    return Default;
  }
  if (Strs.second.trim().getAsInteger(0, Ints.second)) {
    if (!OnlyFirstRequired || Strs.second.trim().size()) {
      Ctx.emitError("can't parse second integer attribute " + Name);
      return Default;
    }
  }

  return Ints;
}

unsigned getInitialPSInputAddr(const Function &F) {
  return getIntegerAttribute(F, "InitialPSInputAddr", 0);
}

bool isShader(CallingConv::ID cc) {
  switch(cc) {
    case CallingConv::AMDGPU_VS:
    case CallingConv::AMDGPU_GS:
    case CallingConv::AMDGPU_PS:
    case CallingConv::AMDGPU_CS:
      return true;
    default:
      return false;
  }
}

bool isCompute(CallingConv::ID cc) {
  return !isShader(cc) || cc == CallingConv::AMDGPU_CS;
}

bool isSI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSouthernIslands];
}

bool isCI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureSeaIslands];
}

bool isVI(const MCSubtargetInfo &STI) {
  return STI.getFeatureBits()[AMDGPU::FeatureVolcanicIslands];
}

unsigned getMCReg(unsigned Reg, const MCSubtargetInfo &STI) {

  switch(Reg) {
  default: break;
  case AMDGPU::FLAT_SCR:
    assert(!isSI(STI));
    return isCI(STI) ? AMDGPU::FLAT_SCR_ci : AMDGPU::FLAT_SCR_vi;

  case AMDGPU::FLAT_SCR_LO:
    assert(!isSI(STI));
    return isCI(STI) ? AMDGPU::FLAT_SCR_LO_ci : AMDGPU::FLAT_SCR_LO_vi;

  case AMDGPU::FLAT_SCR_HI:
    assert(!isSI(STI));
    return isCI(STI) ? AMDGPU::FLAT_SCR_HI_ci : AMDGPU::FLAT_SCR_HI_vi;
  }
  return Reg;
}

bool isSISrcOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;

  return OpType == AMDGPU::OPERAND_REG_IMM32_INT ||
         OpType == AMDGPU::OPERAND_REG_IMM32_FP ||
         OpType == AMDGPU::OPERAND_REG_INLINE_C_INT ||
         OpType == AMDGPU::OPERAND_REG_INLINE_C_FP;
}

bool isSISrcFPOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;

  return OpType == AMDGPU::OPERAND_REG_IMM32_FP ||
         OpType == AMDGPU::OPERAND_REG_INLINE_C_FP;
}

bool isSISrcInlinableOperand(const MCInstrDesc &Desc, unsigned OpNo) {
  unsigned OpType = Desc.OpInfo[OpNo].OperandType;

  return OpType == AMDGPU::OPERAND_REG_INLINE_C_INT ||
         OpType == AMDGPU::OPERAND_REG_INLINE_C_FP;
}

unsigned getRegOperandSize(const MCRegisterInfo *MRI, const MCInstrDesc &Desc,
                           unsigned OpNo) {
  int RCID = Desc.OpInfo[OpNo].RegClass;
  const MCRegisterClass &RC = MRI->getRegClass(RCID);
  return RC.getSize();
}

bool isInlinableLiteral64(int64_t Literal, bool IsVI) {
  if (Literal >= -16 && Literal <= 64)
    return true;

  double D = BitsToDouble(Literal);

  if (D == 0.5 || D == -0.5 ||
      D == 1.0 || D == -1.0 ||
      D == 2.0 || D == -2.0 ||
      D == 4.0 || D == -4.0)
    return true;

  if (IsVI && Literal == 0x3fc45f306dc9c882)
    return true;

  return false;
}

bool isInlinableLiteral32(int32_t Literal, bool IsVI) {
  if (Literal >= -16 && Literal <= 64)
    return true;

  float F = BitsToFloat(Literal);

  if (F == 0.5 || F == -0.5 ||
      F == 1.0 || F == -1.0 ||
      F == 2.0 || F == -2.0 ||
      F == 4.0 || F == -4.0)
    return true;

  if (IsVI && Literal == 0x3e22f983)
    return true;

  return false;
}


} // End namespace AMDGPU
} // End namespace llvm
