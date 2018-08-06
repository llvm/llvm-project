//===-- AMDGPUTargetStreamer.cpp - Mips Target Streamer Methods -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AMDGPU specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUTargetStreamer.h"
#include "AMDGPU.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDKernelCodeTUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
#include "AMDGPUPTNote.h"
}

using namespace llvm;
using namespace llvm::AMDGPU;

//===----------------------------------------------------------------------===//
// AMDGPUTargetStreamer
//===----------------------------------------------------------------------===//

static const struct {
  const char *Name;
  unsigned Mach;
} MachTable[] = {
      // Radeon HD 2000/3000 Series (R600).
      { "r600", ELF::EF_AMDGPU_MACH_R600_R600 },
      { "r630", ELF::EF_AMDGPU_MACH_R600_R630 },
      { "rs880", ELF::EF_AMDGPU_MACH_R600_RS880 },
      { "rv670", ELF::EF_AMDGPU_MACH_R600_RV670 },
      // Radeon HD 4000 Series (R700).
      { "rv710", ELF::EF_AMDGPU_MACH_R600_RV710 },
      { "rv730", ELF::EF_AMDGPU_MACH_R600_RV730 },
      { "rv770", ELF::EF_AMDGPU_MACH_R600_RV770 },
      // Radeon HD 5000 Series (Evergreen).
      { "cedar", ELF::EF_AMDGPU_MACH_R600_CEDAR },
      { "cypress", ELF::EF_AMDGPU_MACH_R600_CYPRESS },
      { "juniper", ELF::EF_AMDGPU_MACH_R600_JUNIPER },
      { "redwood", ELF::EF_AMDGPU_MACH_R600_REDWOOD },
      { "sumo", ELF::EF_AMDGPU_MACH_R600_SUMO },
      // Radeon HD 6000 Series (Northern Islands).
      { "barts", ELF::EF_AMDGPU_MACH_R600_BARTS },
      { "caicos", ELF::EF_AMDGPU_MACH_R600_CAICOS },
      { "cayman", ELF::EF_AMDGPU_MACH_R600_CAYMAN },
      { "turks", ELF::EF_AMDGPU_MACH_R600_TURKS },
      // AMDGCN GFX6.
      { "gfx600", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600 },
      { "tahiti", ELF::EF_AMDGPU_MACH_AMDGCN_GFX600 },
      { "gfx601", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "hainan", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "oland", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "pitcairn", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      { "verde", ELF::EF_AMDGPU_MACH_AMDGCN_GFX601 },
      // AMDGCN GFX7.
      { "gfx700", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700 },
      { "kaveri", ELF::EF_AMDGPU_MACH_AMDGCN_GFX700 },
      { "gfx701", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701 },
      { "hawaii", ELF::EF_AMDGPU_MACH_AMDGCN_GFX701 },
      { "gfx702", ELF::EF_AMDGPU_MACH_AMDGCN_GFX702 },
      { "gfx703", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703 },
      { "kabini", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703 },
      { "mullins", ELF::EF_AMDGPU_MACH_AMDGCN_GFX703 },
      { "gfx704", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704 },
      { "bonaire", ELF::EF_AMDGPU_MACH_AMDGCN_GFX704 },
      // AMDGCN GFX8.
      { "gfx801", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801 },
      { "carrizo", ELF::EF_AMDGPU_MACH_AMDGCN_GFX801 },
      { "gfx802", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802 },
      { "iceland", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802 },
      { "tonga", ELF::EF_AMDGPU_MACH_AMDGCN_GFX802 },
      { "gfx803", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "fiji", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "polaris10", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "polaris11", ELF::EF_AMDGPU_MACH_AMDGCN_GFX803 },
      { "gfx810", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810 },
      { "stoney", ELF::EF_AMDGPU_MACH_AMDGCN_GFX810 },
      // AMDGCN GFX9.
      { "gfx900", ELF::EF_AMDGPU_MACH_AMDGCN_GFX900 },
      { "gfx902", ELF::EF_AMDGPU_MACH_AMDGCN_GFX902 },
      { "gfx904", ELF::EF_AMDGPU_MACH_AMDGCN_GFX904 },
      { "gfx906", ELF::EF_AMDGPU_MACH_AMDGCN_GFX906 },
      // Not specified processor.
      { nullptr, ELF::EF_AMDGPU_MACH_NONE }
};

unsigned AMDGPUTargetStreamer::getMACH(StringRef GPU) const {
  auto Entry = MachTable;
  for (; Entry->Name && GPU != Entry->Name; ++Entry)
    ;
  return Entry->Mach;
}

const char *AMDGPUTargetStreamer::getMachName(unsigned Mach) {
  auto Entry = MachTable;
  for (; Entry->Name && Mach != Entry->Mach; ++Entry)
    ;
  return Entry->Name;
}

bool AMDGPUTargetStreamer::EmitHSAMetadata(StringRef HSAMetadataString) {
  HSAMD::Metadata HSAMetadata;
  if (HSAMD::fromString(HSAMetadataString, HSAMetadata))
    return false;

  return EmitHSAMetadata(HSAMetadata);
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetAsmStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetAsmStreamer::AMDGPUTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : AMDGPUTargetStreamer(S), OS(OS) { }

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

bool AMDGPUTargetAsmStreamer::EmitPALMetadata(
    const PALMD::Metadata &PALMetadata) {
  std::string PALMetadataString;
  if (PALMD::toString(PALMetadata, PALMetadataString))
    return false;

  OS << '\t' << PALMD::AssemblerDirective << PALMetadataString << '\n';
  return true;
}

void AMDGPUTargetAsmStreamer::EmitAmdhsaKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const amdhsa::kernel_descriptor_t &KD, uint64_t NextVGPR, uint64_t NextSGPR,
    bool ReserveVCC, bool ReserveFlatScr, bool ReserveXNACK) {
  amdhsa::kernel_descriptor_t DefaultKD = getDefaultAmdhsaKernelDescriptor();

  IsaInfo::IsaVersion IVersion = IsaInfo::getIsaVersion(STI.getFeatureBits());

  OS << "\t.amdhsa_kernel " << KernelName << '\n';

#define PRINT_IF_NOT_DEFAULT(STREAM, DIRECTIVE, KERNEL_DESC,                   \
                             DEFAULT_KERNEL_DESC, MEMBER_NAME, FIELD_NAME)     \
  if (AMDHSA_BITS_GET(KERNEL_DESC.MEMBER_NAME, FIELD_NAME) !=                  \
      AMDHSA_BITS_GET(DEFAULT_KERNEL_DESC.MEMBER_NAME, FIELD_NAME))            \
    STREAM << "\t\t" << DIRECTIVE << " "                                       \
           << AMDHSA_BITS_GET(KERNEL_DESC.MEMBER_NAME, FIELD_NAME) << '\n';

  if (KD.group_segment_fixed_size != DefaultKD.group_segment_fixed_size)
    OS << "\t\t.amdhsa_group_segment_fixed_size " << KD.group_segment_fixed_size
       << '\n';
  if (KD.private_segment_fixed_size != DefaultKD.private_segment_fixed_size)
    OS << "\t\t.amdhsa_private_segment_fixed_size "
       << KD.private_segment_fixed_size << '\n';

  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_user_sgpr_private_segment_buffer", KD, DefaultKD,
      kernel_code_properties,
      amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_user_sgpr_dispatch_ptr", KD, DefaultKD,
                       kernel_code_properties,
                       amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_user_sgpr_queue_ptr", KD, DefaultKD,
                       kernel_code_properties,
                       amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_user_sgpr_kernarg_segment_ptr", KD, DefaultKD,
      kernel_code_properties,
      amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_user_sgpr_dispatch_id", KD, DefaultKD,
                       kernel_code_properties,
                       amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_user_sgpr_flat_scratch_init", KD, DefaultKD,
      kernel_code_properties,
      amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_user_sgpr_private_segment_size", KD, DefaultKD,
      kernel_code_properties,
      amdhsa::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_system_sgpr_private_segment_wavefront_offset", KD, DefaultKD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_PRIVATE_SEGMENT_WAVEFRONT_OFFSET);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_system_sgpr_workgroup_id_x", KD, DefaultKD,
                       compute_pgm_rsrc2,
                       amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_system_sgpr_workgroup_id_y", KD, DefaultKD,
                       compute_pgm_rsrc2,
                       amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_system_sgpr_workgroup_id_z", KD, DefaultKD,
                       compute_pgm_rsrc2,
                       amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_system_sgpr_workgroup_info", KD, DefaultKD,
                       compute_pgm_rsrc2,
                       amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_system_vgpr_workitem_id", KD, DefaultKD,
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

  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_float_round_mode_32", KD, DefaultKD,
                       compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_float_round_mode_16_64", KD, DefaultKD,
                       compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_float_denorm_mode_32", KD, DefaultKD,
                       compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_float_denorm_mode_16_64", KD, DefaultKD,
                       compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_dx10_clamp", KD, DefaultKD,
                       compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP);
  PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_ieee_mode", KD, DefaultKD,
                       compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE);
  if (IVersion.Major >= 9)
    PRINT_IF_NOT_DEFAULT(OS, ".amdhsa_fp16_overflow", KD, DefaultKD,
                         compute_pgm_rsrc1,
                         amdhsa::COMPUTE_PGM_RSRC1_FP16_OVFL);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_fp_ieee_invalid_op", KD, DefaultKD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_fp_denorm_src", KD, DefaultKD, compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_fp_ieee_div_zero", KD, DefaultKD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_fp_ieee_overflow", KD, DefaultKD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_fp_ieee_underflow", KD, DefaultKD,
      compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_fp_ieee_inexact", KD, DefaultKD, compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT);
  PRINT_IF_NOT_DEFAULT(
      OS, ".amdhsa_exception_int_div_zero", KD, DefaultKD, compute_pgm_rsrc2,
      amdhsa::COMPUTE_PGM_RSRC2_ENABLE_EXCEPTION_INT_DIVIDE_BY_ZERO);
#undef PRINT_IF_NOT_DEFAULT

  OS << "\t.end_amdhsa_kernel\n";
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetELFStreamer::AMDGPUTargetELFStreamer(
    MCStreamer &S, const MCSubtargetInfo &STI)
    : AMDGPUTargetStreamer(S), Streamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();
  unsigned EFlags = MCA.getELFHeaderEFlags();

  EFlags &= ~ELF::EF_AMDGPU_MACH;
  EFlags |= getMACH(STI.getCPU());

  EFlags &= ~ELF::EF_AMDGPU_XNACK;
  if (AMDGPU::hasXNACK(STI))
    EFlags |= ELF::EF_AMDGPU_XNACK;

  MCA.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &AMDGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUNote(
    const MCExpr *DescSZ, unsigned NoteType,
    function_ref<void(MCELFStreamer &)> EmitDesc) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = sizeof(ElfNote::NoteName);

  S.PushSection();
  S.SwitchSection(Context.getELFSection(
    ElfNote::SectionName, ELF::SHT_NOTE, ELF::SHF_ALLOC));
  S.EmitIntValue(NameSZ, 4);                                  // namesz
  S.EmitValue(DescSZ, 4);                                     // descz
  S.EmitIntValue(NoteType, 4);                                // type
  S.EmitBytes(StringRef(ElfNote::NoteName, NameSZ));          // name
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  EmitDesc(S);                                                // desc
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  S.PopSection();
}

void AMDGPUTargetELFStreamer::EmitDirectiveAMDGCNTarget(StringRef Target) {}

void AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(
    uint32_t Major, uint32_t Minor) {

  EmitAMDGPUNote(
    MCConstantExpr::create(8, getContext()),
    ElfNote::NT_AMDGPU_HSA_CODE_OBJECT_VERSION,
    [&](MCELFStreamer &OS){
      OS.EmitIntValue(Major, 4);
      OS.EmitIntValue(Minor, 4);
    }
  );
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

  EmitAMDGPUNote(
    MCConstantExpr::create(DescSZ, getContext()),
    ElfNote::NT_AMDGPU_HSA_ISA,
    [&](MCELFStreamer &OS) {
      OS.EmitIntValue(VendorNameSize, 2);
      OS.EmitIntValue(ArchNameSize, 2);
      OS.EmitIntValue(Major, 4);
      OS.EmitIntValue(Minor, 4);
      OS.EmitIntValue(Stepping, 4);
      OS.EmitBytes(VendorName);
      OS.EmitIntValue(0, 1); // NULL terminate VendorName
      OS.EmitBytes(ArchName);
      OS.EmitIntValue(0, 1); // NULL terminte ArchName
    }
  );
}

void
AMDGPUTargetELFStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.EmitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.PopSection();
}

void AMDGPUTargetELFStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(Type);
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

  EmitAMDGPUNote(
    DescSZ,
    ELF::NT_AMD_AMDGPU_ISA,
    [&](MCELFStreamer &OS) {
      OS.EmitLabel(DescBegin);
      OS.EmitBytes(IsaVersionString);
      OS.EmitLabel(DescEnd);
    }
  );
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

  EmitAMDGPUNote(
    DescSZ,
    ELF::NT_AMD_AMDGPU_HSA_METADATA,
    [&](MCELFStreamer &OS) {
      OS.EmitLabel(DescBegin);
      OS.EmitBytes(HSAMetadataString);
      OS.EmitLabel(DescEnd);
    }
  );
  return true;
}

bool AMDGPUTargetELFStreamer::EmitPALMetadata(
    const PALMD::Metadata &PALMetadata) {
  EmitAMDGPUNote(
    MCConstantExpr::create(PALMetadata.size() * sizeof(uint32_t), getContext()),
    ELF::NT_AMD_AMDGPU_PAL_METADATA,
    [&](MCELFStreamer &OS){
      for (auto I : PALMetadata)
        OS.EmitIntValue(I, sizeof(uint32_t));
    }
  );
  return true;
}

void AMDGPUTargetELFStreamer::EmitAmdhsaKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const amdhsa::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
    uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
    bool ReserveXNACK) {
  auto &Streamer = getStreamer();
  auto &Context = Streamer.getContext();

  MCSymbolELF *KernelDescriptorSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName) + Twine(".kd")));
  KernelDescriptorSymbol->setBinding(ELF::STB_GLOBAL);
  KernelDescriptorSymbol->setType(ELF::STT_OBJECT);
  KernelDescriptorSymbol->setSize(
      MCConstantExpr::create(sizeof(KernelDescriptor), Context));

  MCSymbolELF *KernelCodeSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName)));
  KernelCodeSymbol->setBinding(ELF::STB_LOCAL);

  Streamer.EmitLabel(KernelDescriptorSymbol);
  Streamer.EmitBytes(StringRef(
      (const char*)&(KernelDescriptor),
      offsetof(amdhsa::kernel_descriptor_t, kernel_code_entry_byte_offset)));
  // FIXME: Remove the use of VK_AMDGPU_REL64 in the expression below. The
  // expression being created is:
  //   (start of kernel code) - (start of kernel descriptor)
  // It implies R_AMDGPU_REL64, but ends up being R_AMDGPU_ABS64.
  Streamer.EmitValue(MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(
          KernelCodeSymbol, MCSymbolRefExpr::VK_AMDGPU_REL64, Context),
      MCSymbolRefExpr::create(
          KernelDescriptorSymbol, MCSymbolRefExpr::VK_None, Context),
      Context),
      sizeof(KernelDescriptor.kernel_code_entry_byte_offset));
  Streamer.EmitBytes(StringRef(
      (const char*)&(KernelDescriptor) +
          offsetof(amdhsa::kernel_descriptor_t, kernel_code_entry_byte_offset) +
          sizeof(KernelDescriptor.kernel_code_entry_byte_offset),
      sizeof(KernelDescriptor) -
          offsetof(amdhsa::kernel_descriptor_t, kernel_code_entry_byte_offset) -
          sizeof(KernelDescriptor.kernel_code_entry_byte_offset)));
}
