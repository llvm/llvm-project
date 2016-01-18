//===-- AMDGPUBaseInfo.cpp - AMDGPU Base encoding information--------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "AMDGPUBaseInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/SubtargetFeature.h"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"
#undef GET_SUBTARGETINFO_ENUM

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

} // End namespace AMDGPU
} // End namespace llvm
