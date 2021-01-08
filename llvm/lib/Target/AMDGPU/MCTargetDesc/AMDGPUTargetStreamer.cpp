//===-- AMDGPUTargetStreamer.cpp - Mips Target Streamer Methods -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides AMDGPU specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetStreamer.h"
#include "AMDGPUPTNote.h"
#include "AMDKernelCodeT.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace llvm::AMDGPU;

//===----------------------------------------------------------------------===//
// AMDGPUTargetStreamer
//===----------------------------------------------------------------------===//

bool AMDGPUTargetStreamer::EmitHSAMetadataV2(StringRef HSAMetadataString) {
  HSAMD::Metadata HSAMetadata;
  if (HSAMD::fromString(std::string(HSAMetadataString), HSAMetadata))
    return false;

  return EmitHSAMetadata(HSAMetadata);
}

bool AMDGPUTargetStreamer::EmitHSAMetadataV3(StringRef HSAMetadataString) {
  msgpack::Document HSAMetadataDoc;
  if (!HSAMetadataDoc.fromYAML(HSAMetadataString))
    return false;
  return EmitHSAMetadata(HSAMetadataDoc, false);
}

StringRef AMDGPUTargetStreamer::getArchNameFromElfMach(unsigned ElfMach) {
  AMDGPU::GPUKind AK;

  switch (ElfMach) {
  default: llvm_unreachable("Unhandled ELF::EF_AMDGPU type");
  case ELF::EF_AMDGPU_MACH_R600_R600:      AK = GK_R600;    break;
  case ELF::EF_AMDGPU_MACH_R600_R630:      AK = GK_R630;    break;
  case ELF::EF_AMDGPU_MACH_R600_RS880:     AK = GK_RS880;   break;
  case ELF::EF_AMDGPU_MACH_R600_RV670:     AK = GK_RV670;   break;
  case ELF::EF_AMDGPU_MACH_R600_RV710:     AK = GK_RV710;   break;
  case ELF::EF_AMDGPU_MACH_R600_RV730:     AK = GK_RV730;   break;
  case ELF::EF_AMDGPU_MACH_R600_RV770:     AK = GK_RV770;   break;
  case ELF::EF_AMDGPU_MACH_R600_CEDAR:     AK = GK_CEDAR;   break;
  case ELF::EF_AMDGPU_MACH_R600_CYPRESS:   AK = GK_CYPRESS; break;
  case ELF::EF_AMDGPU_MACH_R600_JUNIPER:   AK = GK_JUNIPER; break;
  case ELF::EF_AMDGPU_MACH_R600_REDWOOD:   AK = GK_REDWOOD; break;
  case ELF::EF_AMDGPU_MACH_R600_SUMO:      AK = GK_SUMO;    break;
  case ELF::EF_AMDGPU_MACH_R600_BARTS:     AK = GK_BARTS;   break;
  case ELF::EF_AMDGPU_MACH_R600_CAICOS:    AK = GK_CAICOS;  break;
  case ELF::EF_AMDGPU_MACH_R600_CAYMAN:    AK = GK_CAYMAN;  break;
  case ELF::EF_AMDGPU_MACH_R600_TURKS:     AK = GK_TURKS;   break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX600:  AK = GK_GFX600;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX601:  AK = GK_GFX601;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX602:  AK = GK_GFX602;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX700:  AK = GK_GFX700;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX701:  AK = GK_GFX701;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX702:  AK = GK_GFX702;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX703:  AK = GK_GFX703;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX704:  AK = GK_GFX704;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX705:  AK = GK_GFX705;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX801:  AK = GK_GFX801;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX802:  AK = GK_GFX802;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX803:  AK = GK_GFX803;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX805:  AK = GK_GFX805;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX810:  AK = GK_GFX810;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX900:  AK = GK_GFX900;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX902:  AK = GK_GFX902;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX904:  AK = GK_GFX904;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX906:  AK = GK_GFX906;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX908:  AK = GK_GFX908;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX909:  AK = GK_GFX909;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C:  AK = GK_GFX90C;  break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1010: AK = GK_GFX1010; break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1011: AK = GK_GFX1011; break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1012: AK = GK_GFX1012; break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030: AK = GK_GFX1030; break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1031: AK = GK_GFX1031; break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1032: AK = GK_GFX1032; break;
  case ELF::EF_AMDGPU_MACH_AMDGCN_GFX1033: AK = GK_GFX1033; break;
  case ELF::EF_AMDGPU_MACH_NONE:           AK = GK_NONE;    break;
  }

  StringRef GPUName = getArchNameAMDGCN(AK);
  if (GPUName != "")
    return GPUName;
  return getArchNameR600(AK);
}

unsigned AMDGPUTargetStreamer::getElfMach(StringRef GPU) {
  AMDGPU::GPUKind AK = parseArchAMDGCN(GPU);
  if (AK == AMDGPU::GPUKind::GK_NONE)
    AK = parseArchR600(GPU);

  switch (AK) {
  case GK_R600:    return ELF::EF_AMDGPU_MACH_R600_R600;
  case GK_R630:    return ELF::EF_AMDGPU_MACH_R600_R630;
  case GK_RS880:   return ELF::EF_AMDGPU_MACH_R600_RS880;
  case GK_RV670:   return ELF::EF_AMDGPU_MACH_R600_RV670;
  case GK_RV710:   return ELF::EF_AMDGPU_MACH_R600_RV710;
  case GK_RV730:   return ELF::EF_AMDGPU_MACH_R600_RV730;
  case GK_RV770:   return ELF::EF_AMDGPU_MACH_R600_RV770;
  case GK_CEDAR:   return ELF::EF_AMDGPU_MACH_R600_CEDAR;
  case GK_CYPRESS: return ELF::EF_AMDGPU_MACH_R600_CYPRESS;
  case GK_JUNIPER: return ELF::EF_AMDGPU_MACH_R600_JUNIPER;
  case GK_REDWOOD: return ELF::EF_AMDGPU_MACH_R600_REDWOOD;
  case GK_SUMO:    return ELF::EF_AMDGPU_MACH_R600_SUMO;
  case GK_BARTS:   return ELF::EF_AMDGPU_MACH_R600_BARTS;
  case GK_CAICOS:  return ELF::EF_AMDGPU_MACH_R600_CAICOS;
  case GK_CAYMAN:  return ELF::EF_AMDGPU_MACH_R600_CAYMAN;
  case GK_TURKS:   return ELF::EF_AMDGPU_MACH_R600_TURKS;
  case GK_GFX600:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX600;
  case GK_GFX601:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX601;
  case GK_GFX602:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX602;
  case GK_GFX700:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX700;
  case GK_GFX701:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX701;
  case GK_GFX702:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX702;
  case GK_GFX703:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX703;
  case GK_GFX704:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX704;
  case GK_GFX705:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX705;
  case GK_GFX801:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX801;
  case GK_GFX802:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX802;
  case GK_GFX803:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX803;
  case GK_GFX805:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX805;
  case GK_GFX810:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX810;
  case GK_GFX900:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX900;
  case GK_GFX902:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX902;
  case GK_GFX904:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX904;
  case GK_GFX906:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX906;
  case GK_GFX908:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX908;
  case GK_GFX909:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX909;
  case GK_GFX90C:  return ELF::EF_AMDGPU_MACH_AMDGCN_GFX90C;
  case GK_GFX1010: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1010;
  case GK_GFX1011: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1011;
  case GK_GFX1012: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1012;
  case GK_GFX1030: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030;
  case GK_GFX1031: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1031;
  case GK_GFX1032: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1032;
  case GK_GFX1033: return ELF::EF_AMDGPU_MACH_AMDGCN_GFX1033;
  case GK_NONE:    return ELF::EF_AMDGPU_MACH_NONE;
  }

  llvm_unreachable("unknown GPU");
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetAsmStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetAsmStreamer::AMDGPUTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : AMDGPUTargetStreamer(S), OS(OS) { }

// A hook for emitting stuff at the end.
// We use it for emitting the accumulated PAL metadata as directives.
// The PAL metadata is reset after it is emitted.
void AMDGPUTargetAsmStreamer::finish() {
  std::string S;
  getPALMetadata()->toString(S);
  OS << S;

  // Reset the pal metadata so its data will not affect a compilation that
  // reuses this object.
  getPALMetadata()->reset();
}

void AMDGPUTargetAsmStreamer::EmitDirectiveAMDGCNTarget(StringRef Target) {
  OS << "\t.amdgcn_target \"" << Target << "\"\n";
}

void AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectVersion(
    uint32_t Major, uint32_t Minor) {
  OS << "\t.hsa_code_object_version " <<
        Twine(Major) << "," << Twine(Minor) << '\n';
}

void
AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  OS << "\t.hsa_code_object_isa " <<
        Twine(Major) << "," << Twine(Minor) << "," << Twine(Stepping) <<
        ",\"" << VendorName << "\",\"" << ArchName << "\"\n";

}

void
AMDGPUTargetAsmStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {
  OS << "\t.amd_kernel_code_t\n";
  dumpAmdKernelCode(&Header, OS, "\t\t");
  OS << "\t.end_amd_kernel_code_t\n";
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  switch (Type) {
    default: llvm_unreachable("Invalid AMDGPU symbol type");
    case ELF::STT_AMDGPU_HSA_KERNEL:
      OS << "\t.amdgpu_hsa_kernel " << SymbolName << '\n' ;
      break;
  }
}

void AMDGPUTargetAsmStreamer::emitAMDGPULDS(MCSymbol *Symbol, unsigned Size,
                                            Align Alignment) {
  OS << "\t.amdgpu_lds " << Symbol->getName() << ", " << Size << ", "
     << Alignment.value() << '\n';
}

bool AMDGPUTargetAsmStreamer::EmitISAVersion(StringRef IsaVersionString) {
  OS << "\t.amd_amdgpu_isa \"" << IsaVersionString << "\"\n";
  return true;
}

bool AMDGPUTargetAsmStreamer::EmitHSAMetadata(
    const AMDGPU::HSAMD::Metadata &HSAMetadata) {
  std::string HSAMetadataString;
  if (HSAMD::toString(HSAMetadata, HSAMetadataString))
    return false;

  OS << '\t' << HSAMD::AssemblerDirectiveBegin << '\n';
  OS << HSAMetadataString << '\n';
  OS << '\t' << HSAMD::AssemblerDirectiveEnd << '\n';
  return true;
}

bool AMDGPUTargetAsmStreamer::EmitHSAMetadata(
    msgpack::Document &HSAMetadataDoc, bool Strict) {
  HSAMD::V3::MetadataVerifier Verifier(Strict);
  if (!Verifier.verify(HSAMetadataDoc.getRoot()))
    return false;

  std::string HSAMetadataString;
  raw_string_ostream StrOS(HSAMetadataString);
  HSAMetadataDoc.toYAML(StrOS);

  OS << '\t' << HSAMD::V3::AssemblerDirectiveBegin << '\n';
  OS << StrOS.str() << '\n';
  OS << '\t' << HSAMD::V3::AssemblerDirectiveEnd << '\n';
  return true;
}

bool AMDGPUTargetAsmStreamer::EmitCodeEnd() {
  const uint32_t Encoded_s_code_end = 0xbf9f0000;
  OS << "\t.p2alignl 6, " << Encoded_s_code_end << '\n';
  OS << "\t.fill 48, 4, " << Encoded_s_code_end << '\n';
  return true;
}

void AMDGPUTargetAsmStreamer::EmitAmdhsaKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const amdhsa::kernel_descriptor_t &KD, uint64_t NextVGPR, uint64_t NextSGPR,
    bool ReserveVCC, bool ReserveFlatScr, bool ReserveXNACK) {
  IsaVersion IVersion = getIsaVersion(STI.getCPU());

  OS << "\t.amdhsa_kernel " << KernelName << '\n';

#define PRINT_FIELD(STREAM, DIRECTIVE, KERNEL_DESC, MEMBER_NAME, FIELD_NAME)   \
  STREAM << "\t\t" << DIRECTIVE << " "                                         \
         << AMDHSA_BITS_GET(KERNEL_DESC.MEMBER_NAME, FIELD_NAME) << '\n';

  OS << "\t\t.amdhsa_group_segment_fixed_size " << KD.group_segment_fixed_size
     << '\n';
  OS << "\t\t.amdhsa_private_segment_fixed_size "
     << KD.private_segment_fixed_size << '\n';

  PRINT_FIELD(OS, ".amdhsa_user_sgpr_private_segment_buffer", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER);
  PRINT_FIELD(OS, ".amdhsa_user_sgpr_dispatch_ptr", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR);
  PRINT_FIELD(OS, ".amdhsa_user_sgpr_queue_ptr", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR);
  PRINT_FIELD(OS, ".amdhsa_user_sgpr_kernarg_segment_ptr", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  PRINT_FIELD(OS, ".amdhsa_user_sgpr_dispatch_id", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID);
  PRINT_FIELD(OS, ".amdhsa_user_sgpr_flat_scratch_init", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT);
  PRINT_FIELD(OS, ".amdhsa_user_sgpr_private_segment_size", KD,
              kernel_code_properties,
              amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE);
  if (IVersion.Major >= 10)
    PRINT_FIELD(OS, ".amdhsa_wavefront_size32", KD,
                kernel_code_properties,
                amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);
  PRINT_FIELD(
      OS, ".amdhsa_system_sgpr_private_segment_wavefront_offset", KD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT);
  PRINT_FIELD(OS, ".amdhsa_system_sgpr_workgroup_id_x", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  PRINT_FIELD(OS, ".amdhsa_system_sgpr_workgroup_id_y", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  PRINT_FIELD(OS, ".amdhsa_system_sgpr_workgroup_id_z", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  PRINT_FIELD(OS, ".amdhsa_system_sgpr_workgroup_info", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO);
  PRINT_FIELD(OS, ".amdhsa_system_vgpr_workitem_id", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID);

  // These directives are required.
  OS << "\t\t.amdhsa_next_free_vgpr " << NextVGPR << '\n';
  OS << "\t\t.amdhsa_next_free_sgpr " << NextSGPR << '\n';

  if (!ReserveVCC)
    OS << "\t\t.amdhsa_reserve_vcc " << ReserveVCC << '\n';
  if (IVersion.Major >= 7 && !ReserveFlatScr)
    OS << "\t\t.amdhsa_reserve_flat_scratch " << ReserveFlatScr << '\n';
  if (IVersion.Major >= 8 && ReserveXNACK != hasXNACK(STI))
    OS << "\t\t.amdhsa_reserve_xnack_mask " << ReserveXNACK << '\n';

  PRINT_FIELD(OS, ".amdhsa_float_round_mode_32", KD,
              compute_pgm_rsrc1,
              amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  PRINT_FIELD(OS, ".amdhsa_float_round_mode_16_64", KD,
              compute_pgm_rsrc1,
              amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  PRINT_FIELD(OS, ".amdhsa_float_denorm_mode_32", KD,
              compute_pgm_rsrc1,
              amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  PRINT_FIELD(OS, ".amdhsa_float_denorm_mode_16_64", KD,
              compute_pgm_rsrc1,
              amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  PRINT_FIELD(OS, ".amdhsa_dx10_clamp", KD,
              compute_pgm_rsrc1,
              amdhsa::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP);
  PRINT_FIELD(OS, ".amdhsa_ieee_mode", KD,
              compute_pgm_rsrc1,
              amdhsa::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE);
  if (IVersion.Major >= 9)
    PRINT_FIELD(OS, ".amdhsa_fp16_overflow", KD,
                compute_pgm_rsrc1,
                amdhsa::COMPUTE_PGM_RSRC1_FP16_OVFL);
  if (IVersion.Major >= 10) {
    PRINT_FIELD(OS, ".amdhsa_workgroup_processor_mode", KD,
                compute_pgm_rsrc1,
                amdhsa::COMPUTE_PGM_RSRC1_WGP_MODE);
    PRINT_FIELD(OS, ".amdhsa_memory_ordered", KD,
                compute_pgm_rsrc1,
                amdhsa::COMPUTE_PGM_RSRC1_MEM_ORDERED);
    PRINT_FIELD(OS, ".amdhsa_forward_progress", KD,
                compute_pgm_rsrc1,
                amdhsa::COMPUTE_PGM_RSRC1_FWD_PROGRESS);
  }
  PRINT_FIELD(
      OS, ".amdhsa_exception_fp_ieee_invalid_op", KD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION);
  PRINT_FIELD(OS, ".amdhsa_exception_fp_denorm_src", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE);
  PRINT_FIELD(
      OS, ".amdhsa_exception_fp_ieee_div_zero", KD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO);
  PRINT_FIELD(OS, ".amdhsa_exception_fp_ieee_overflow", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW);
  PRINT_FIELD(OS, ".amdhsa_exception_fp_ieee_underflow", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW);
  PRINT_FIELD(OS, ".amdhsa_exception_fp_ieee_inexact", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT);
  PRINT_FIELD(OS, ".amdhsa_exception_int_div_zero", KD,
              compute_pgm_rsrc2,
              amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO);
#undef PRINT_FIELD

  OS << "\t.end_amdhsa_kernel\n";
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetELFStreamer::AMDGPUTargetELFStreamer(MCStreamer &S,
                                                 const MCSubtargetInfo &STI)
    : AMDGPUTargetStreamer(S), Streamer(S), Os(STI.getTargetTriple().getOS()) {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned EFlags = MCA.getELFHeaderEFlags();

  EFlags &= ~ELF::EF_AMDGPU_MACH;
  EFlags |= getElfMach(STI.getCPU());

  EFlags &= ~ELF::EF_AMDGPU_XNACK;
  if (AMDGPU::hasXNACK(STI))
    EFlags |= ELF::EF_AMDGPU_XNACK;

  EFlags &= ~ELF::EF_AMDGPU_SRAM_ECC;
  if (AMDGPU::hasSRAMECC(STI))
    EFlags |= ELF::EF_AMDGPU_SRAM_ECC;

  MCA.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &AMDGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

// A hook for emitting stuff at the end.
// We use it for emitting the accumulated PAL metadata as a .note record.
// The PAL metadata is reset after it is emitted.
void AMDGPUTargetELFStreamer::finish() {
  std::string Blob;
  const char *Vendor = getPALMetadata()->getVendor();
  unsigned Type = getPALMetadata()->getType();
  getPALMetadata()->toBlob(Type, Blob);
  if (Blob.empty())
    return;
  EmitNote(Vendor, MCConstantExpr::create(Blob.size(), getContext()), Type,
           [&](MCELFStreamer &OS) { OS.emitBytes(Blob); });

  // Reset the pal metadata so its data will not affect a compilation that
  // reuses this object.
  getPALMetadata()->reset();
}

void AMDGPUTargetELFStreamer::EmitNote(
    StringRef Name, const MCExpr *DescSZ, unsigned NoteType,
    function_ref<void(MCELFStreamer &)> EmitDesc) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = Name.size() + 1;

  unsigned NoteFlags = 0;
  // TODO Apparently, this is currently needed for OpenCL as mentioned in
  // https://reviews.llvm.org/D74995
  if (Os == Triple::AMDHSA)
    NoteFlags = ELF::SHF_ALLOC;

  S.PushSection();
  S.SwitchSection(
      Context.getELFSection(ElfNote::SectionName, ELF::SHT_NOTE, NoteFlags));
  S.emitInt32(NameSZ);                                        // namesz
  S.emitValue(DescSZ, 4);                                     // descz
  S.emitInt32(NoteType);                                      // type
  S.emitBytes(Name);                                          // name
  S.emitValueToAlignment(4, 0, 1, 0);                         // padding 0
  EmitDesc(S);                                                // desc
  S.emitValueToAlignment(4, 0, 1, 0);                         // padding 0
  S.PopSection();
}

void AMDGPUTargetELFStreamer::EmitDirectiveAMDGCNTarget(StringRef Target) {}

void AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(
    uint32_t Major, uint32_t Minor) {

  EmitNote(ElfNote::NoteNameV2, MCConstantExpr::create(8, getContext()),
           ElfNote::NT_AMDGPU_HSA_CODE_OBJECT_VERSION, [&](MCELFStreamer &OS) {
             OS.emitInt32(Major);
             OS.emitInt32(Minor);
           });
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;

  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
    VendorNameSize + ArchNameSize;

  EmitNote(ElfNote::NoteNameV2, MCConstantExpr::create(DescSZ, getContext()),
           ElfNote::NT_AMDGPU_HSA_ISA, [&](MCELFStreamer &OS) {
             OS.emitInt16(VendorNameSize);
             OS.emitInt16(ArchNameSize);
             OS.emitInt32(Major);
             OS.emitInt32(Minor);
             OS.emitInt32(Stepping);
             OS.emitBytes(VendorName);
             OS.emitInt8(0); // NULL terminate VendorName
             OS.emitBytes(ArchName);
             OS.emitInt8(0); // NULL terminte ArchName
           });
}

void
AMDGPUTargetELFStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.emitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.PopSection();
}

void AMDGPUTargetELFStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(Type);
}

void AMDGPUTargetELFStreamer::emitAMDGPULDS(MCSymbol *Symbol, unsigned Size,
                                            Align Alignment) {
  MCSymbolELF *SymbolELF = cast<MCSymbolELF>(Symbol);
  SymbolELF->setType(ELF::STT_OBJECT);

  if (!SymbolELF->isBindingSet()) {
    SymbolELF->setBinding(ELF::STB_GLOBAL);
    SymbolELF->setExternal(true);
  }

  if (SymbolELF->declareCommon(Size, Alignment.value(), true)) {
    report_fatal_error("Symbol: " + Symbol->getName() +
                       " redeclared as different type");
  }

  SymbolELF->setIndex(ELF::SHN_AMDGPU_LDS);
  SymbolELF->setSize(MCConstantExpr::create(Size, getContext()));
}

bool AMDGPUTargetELFStreamer::EmitISAVersion(StringRef IsaVersionString) {
  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV2, DescSZ, ELF::NT_AMD_AMDGPU_ISA,
           [&](MCELFStreamer &OS) {
             OS.emitLabel(DescBegin);
             OS.emitBytes(IsaVersionString);
             OS.emitLabel(DescEnd);
           });
  return true;
}

bool AMDGPUTargetELFStreamer::EmitHSAMetadata(msgpack::Document &HSAMetadataDoc,
                                              bool Strict) {
  HSAMD::V3::MetadataVerifier Verifier(Strict);
  if (!Verifier.verify(HSAMetadataDoc.getRoot()))
    return false;

  std::string HSAMetadataString;
  HSAMetadataDoc.writeToBlob(HSAMetadataString);

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(DescEnd, Context),
      MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV3, DescSZ, ELF::NT_AMDGPU_METADATA,
           [&](MCELFStreamer &OS) {
             OS.emitLabel(DescBegin);
             OS.emitBytes(HSAMetadataString);
             OS.emitLabel(DescEnd);
           });
  return true;
}

bool AMDGPUTargetELFStreamer::EmitHSAMetadata(
    const AMDGPU::HSAMD::Metadata &HSAMetadata) {
  std::string HSAMetadataString;
  if (HSAMD::toString(HSAMetadata, HSAMetadataString))
    return false;

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV2, DescSZ, ELF::NT_AMD_AMDGPU_HSA_METADATA,
           [&](MCELFStreamer &OS) {
             OS.emitLabel(DescBegin);
             OS.emitBytes(HSAMetadataString);
             OS.emitLabel(DescEnd);
           });
  return true;
}

bool AMDGPUTargetELFStreamer::EmitCodeEnd() {
  const uint32_t Encoded_s_code_end = 0xbf9f0000;

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.emitValueToAlignment(64, Encoded_s_code_end, 4);
  for (unsigned I = 0; I < 48; ++I)
    OS.emitInt32(Encoded_s_code_end);
  OS.PopSection();
  return true;
}

void AMDGPUTargetELFStreamer::EmitAmdhsaKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const amdhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
    uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
    bool ReserveXNACK) {
  auto &Streamer = getStreamer();
  auto &Context = Streamer.getContext();

  MCSymbolELF *KernelCodeSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName)));
  MCSymbolELF *KernelDescriptorSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName) + Twine(".kd")));

  // Copy kernel descriptor symbol's binding, other and visibility from the
  // kernel code symbol.
  KernelDescriptorSymbol->setBinding(KernelCodeSymbol->getBinding());
  KernelDescriptorSymbol->setOther(KernelCodeSymbol->getOther());
  KernelDescriptorSymbol->setVisibility(KernelCodeSymbol->getVisibility());
  // Kernel descriptor symbol's type and size are fixed.
  KernelDescriptorSymbol->setType(ELF::STT_OBJECT);
  KernelDescriptorSymbol->setSize(
      MCConstantExpr::create(sizeof(KernelDescriptor), Context));

  // The visibility of the kernel code symbol must be protected or less to allow
  // static relocations from the kernel descriptor to be used.
  if (KernelCodeSymbol->getVisibility() == ELF::STV_DEFAULT)
    KernelCodeSymbol->setVisibility(ELF::STV_PROTECTED);

  Streamer.emitLabel(KernelDescriptorSymbol);
  Streamer.emitInt32(KernelDescriptor.group_segment_fixed_size);
  Streamer.emitInt32(KernelDescriptor.private_segment_fixed_size);
  for (uint8_t Res : KernelDescriptor.reserved0)
    Streamer.emitInt8(Res);
  // FIXME: Remove the use of VK_AMDGPU_REL64 in the expression below. The
  // expression being created is:
  //   (start of kernel code) - (start of kernel descriptor)
  // It implies R_AMDGPU_REL64, but ends up being R_AMDGPU_ABS64.
  Streamer.emitValue(MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(
          KernelCodeSymbol, MCSymbolRefExpr::VK_AMDGPU_REL64, Context),
      MCSymbolRefExpr::create(
          KernelDescriptorSymbol, MCSymbolRefExpr::VK_None, Context),
      Context),
      sizeof(KernelDescriptor.kernel_code_entry_byte_offset));
  for (uint8_t Res : KernelDescriptor.reserved1)
    Streamer.emitInt8(Res);
  Streamer.emitInt32(KernelDescriptor.compute_pgm_rsrc3);
  Streamer.emitInt32(KernelDescriptor.compute_pgm_rsrc1);
  Streamer.emitInt32(KernelDescriptor.compute_pgm_rsrc2);
  Streamer.emitInt16(KernelDescriptor.kernel_code_properties);
  for (uint8_t Res : KernelDescriptor.reserved2)
    Streamer.emitInt8(Res);
}
