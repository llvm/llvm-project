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
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/MCContext.h"
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

static unsigned getIntegerAttribute(const Function &F, const char *Name,
                                    unsigned Default) {
  Attribute A = F.getFnAttribute(Name);
  unsigned Result = Default;

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, Result)) {
      LLVMContext &Ctx = F.getContext();
      Ctx.emitError("can't parse shader type");
    }
  }
  return Result;
}

unsigned getShaderType(const Function &F) {
  return getIntegerAttribute(F, "ShaderType", ShaderType::COMPUTE);
}

unsigned getInitialPSInputAddr(const Function &F) {
  return getIntegerAttribute(F, "InitialPSInputAddr", 0);
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

} // End namespace AMDGPU
} // End namespace llvm
