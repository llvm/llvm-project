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
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

AMDGPUTargetStreamer::AMDGPUTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) { }

//===----------------------------------------------------------------------===//
// AMDGPUTargetAsmStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetAsmStreamer::AMDGPUTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : AMDGPUTargetStreamer(S), OS(OS) { }

void
AMDGPUTargetAsmStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {
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
  uint64_t ComputePgmRsrc2 = (Header.compute_pgm_resource_registers >> 32);
  bool EnableSGPRPrivateSegmentBuffer = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER);
  bool EnableSGPRDispatchPtr = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR);
  bool EnableSGPRQueuePtr = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR);
  bool EnableSGPRKernargSegmentPtr = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR);
  bool EnableSGPRDispatchID = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID);
  bool EnableSGPRFlatScratchInit = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT);
  bool EnableSGPRPrivateSegmentSize = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE);
  bool EnableSGPRGridWorkgroupCountX = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X);
  bool EnableSGPRGridWorkgroupCountY = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y);
  bool EnableSGPRGridWorkgroupCountZ = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z);
  bool EnableOrderedAppendGDS = (Header.code_properties &
      AMD_CODE_PROPERTY_ENABLE_ORDERED_APPEND_GDS);
  uint32_t PrivateElementSize = (Header.code_properties &
      AMD_CODE_PROPERTY_PRIVATE_ELEMENT_SIZE) >>
          AMD_CODE_PROPERTY_PRIVATE_ELEMENT_SIZE_SHIFT;
  bool IsPtr64 = (Header.code_properties & AMD_CODE_PROPERTY_IS_PTR64);
  bool IsDynamicCallstack = (Header.code_properties &
      AMD_CODE_PROPERTY_IS_DYNAMIC_CALLSTACK);
  bool IsDebugEnabled = (Header.code_properties &
      AMD_CODE_PROPERTY_IS_DEBUG_SUPPORTED);
  bool IsXNackEnabled = (Header.code_properties &
      AMD_CODE_PROPERTY_IS_XNACK_SUPPORTED);

  OS << "\t.amd_kernel_code_t\n" <<
    "\t\tkernel_code_version_major = " <<
        Header.amd_kernel_code_version_major << '\n' <<
    "\t\tkernel_code_version_minor = " <<
        Header.amd_kernel_code_version_minor << '\n' <<
    "\t\tmachine_kind = " <<
        Header.amd_machine_kind << '\n' <<
    "\t\tmachine_version_major = " <<
        Header.amd_machine_version_major << '\n' <<
    "\t\tmachine_version_minor = " <<
        Header.amd_machine_version_minor << '\n' <<
    "\t\tmachine_version_stepping = " <<
        Header.amd_machine_version_stepping << '\n' <<
    "\t\tkernel_code_entry_byte_offset = " <<
        Header.kernel_code_entry_byte_offset << '\n' <<
    "\t\tkernel_code_prefetch_byte_size = " <<
        Header.kernel_code_prefetch_byte_size << '\n' <<
    "\t\tmax_scratch_backing_memory_byte_size = " <<
        Header.max_scratch_backing_memory_byte_size << '\n' <<
    "\t\tcompute_pgm_rsrc1_vgprs = " <<
        G_00B848_VGPRS(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_sgprs = " <<
        G_00B848_SGPRS(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_priority = " <<
        G_00B848_PRIORITY(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_float_mode = " <<
        G_00B848_FLOAT_MODE(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_priv = " <<
        G_00B848_PRIV(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_dx10_clamp = " <<
        G_00B848_DX10_CLAMP(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_debug_mode = " <<
        G_00B848_DEBUG_MODE(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc1_ieee_mode = " <<
        G_00B848_IEEE_MODE(Header.compute_pgm_resource_registers) << '\n' <<
    "\t\tcompute_pgm_rsrc2_scratch_en = " <<
        G_00B84C_SCRATCH_EN(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_user_sgpr = " <<
        G_00B84C_USER_SGPR(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_tgid_x_en = " <<
        G_00B84C_TGID_X_EN(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_tgid_y_en = " <<
        G_00B84C_TGID_Y_EN(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_tgid_z_en = " <<
        G_00B84C_TGID_Z_EN(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_tg_size_en = " <<
        G_00B84C_TG_SIZE_EN(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_tidig_comp_cnt = " <<
        G_00B84C_TIDIG_COMP_CNT(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_excp_en_msb = " <<
        G_00B84C_EXCP_EN_MSB(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_lds_size = " <<
        G_00B84C_LDS_SIZE(ComputePgmRsrc2) << '\n' <<
    "\t\tcompute_pgm_rsrc2_excp_en = " <<
        G_00B84C_EXCP_EN(ComputePgmRsrc2) << '\n' <<

    "\t\tenable_sgpr_private_segment_buffer = " <<
        EnableSGPRPrivateSegmentBuffer << '\n' <<
    "\t\tenable_sgpr_dispatch_ptr = " <<
        EnableSGPRDispatchPtr << '\n' <<
    "\t\tenable_sgpr_queue_ptr = " <<
        EnableSGPRQueuePtr << '\n' <<
    "\t\tenable_sgpr_kernarg_segment_ptr = " <<
        EnableSGPRKernargSegmentPtr << '\n' <<
    "\t\tenable_sgpr_dispatch_id = " <<
        EnableSGPRDispatchID << '\n' <<
    "\t\tenable_sgpr_flat_scratch_init = " <<
        EnableSGPRFlatScratchInit << '\n' <<
    "\t\tenable_sgpr_private_segment_size = " <<
        EnableSGPRPrivateSegmentSize << '\n' <<
    "\t\tenable_sgpr_grid_workgroup_count_x = " <<
        EnableSGPRGridWorkgroupCountX << '\n' <<
    "\t\tenable_sgpr_grid_workgroup_count_y = " <<
        EnableSGPRGridWorkgroupCountY << '\n' <<
    "\t\tenable_sgpr_grid_workgroup_count_z = " <<
        EnableSGPRGridWorkgroupCountZ << '\n' <<
    "\t\tenable_ordered_append_gds = " <<
        EnableOrderedAppendGDS << '\n' <<
    "\t\tprivate_element_size = " <<
        PrivateElementSize << '\n' <<
    "\t\tis_ptr64 = " <<
        IsPtr64 << '\n' <<
    "\t\tis_dynamic_callstack = " <<
        IsDynamicCallstack << '\n' <<
    "\t\tis_debug_enabled = " <<
        IsDebugEnabled << '\n' <<
    "\t\tis_xnack_enabled = " <<
        IsXNackEnabled << '\n' <<
    "\t\tworkitem_private_segment_byte_size = " <<
        Header.workitem_private_segment_byte_size << '\n' <<
    "\t\tworkgroup_group_segment_byte_size = " <<
        Header.workgroup_group_segment_byte_size << '\n' <<
    "\t\tgds_segment_byte_size = " <<
        Header.gds_segment_byte_size << '\n' <<
    "\t\tkernarg_segment_byte_size = " <<
        Header.kernarg_segment_byte_size << '\n' <<
    "\t\tworkgroup_fbarrier_count = " <<
        Header.workgroup_fbarrier_count << '\n' <<
    "\t\twavefront_sgpr_count = " <<
        Header.wavefront_sgpr_count << '\n' <<
    "\t\tworkitem_vgpr_count = " <<
        Header.workitem_vgpr_count << '\n' <<
    "\t\treserved_vgpr_first = " <<
        Header.reserved_vgpr_first << '\n' <<
    "\t\treserved_vgpr_count = " <<
        Header.reserved_vgpr_count << '\n' <<
    "\t\treserved_sgpr_first = " <<
        Header.reserved_sgpr_first << '\n' <<
    "\t\treserved_sgpr_count = " <<
        Header.reserved_sgpr_count << '\n' <<
    "\t\tdebug_wavefront_private_segment_offset_sgpr = " <<
        Header.debug_wavefront_private_segment_offset_sgpr << '\n' <<
    "\t\tdebug_private_segment_buffer_sgpr = " <<
        Header.debug_private_segment_buffer_sgpr << '\n' <<
    "\t\tkernarg_segment_alignment = " <<
        (uint32_t)Header.kernarg_segment_alignment << '\n' <<
    "\t\tgroup_segment_alignment = " <<
        (uint32_t)Header.group_segment_alignment << '\n' <<
    "\t\tprivate_segment_alignment = " <<
        (uint32_t)Header.private_segment_alignment << '\n' <<
    "\t\twavefront_size = " <<
        (uint32_t)Header.wavefront_size << '\n' <<
    "\t\tcall_convention = " <<
        Header.call_convention << '\n' <<
    "\t\truntime_loader_kernel_symbol = " <<
        Header.runtime_loader_kernel_symbol << '\n' <<
    // TODO: control_directives
    "\t.end_amd_kernel_code_t\n";

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

void AMDGPUTargetAsmStreamer::EmitAMDGPUHsaModuleScopeGlobal(
    StringRef GlobalName) {
  OS << "\t.amdgpu_hsa_module_global " << GlobalName << '\n';
}

void AMDGPUTargetAsmStreamer::EmitAMDGPUHsaProgramScopeGlobal(
    StringRef GlobalName) {
  OS << "\t.amdgpu_hsa_program_global " << GlobalName << '\n';
}

//===----------------------------------------------------------------------===//
// AMDGPUTargetELFStreamer
//===----------------------------------------------------------------------===//

AMDGPUTargetELFStreamer::AMDGPUTargetELFStreamer(MCStreamer &S)
    : AMDGPUTargetStreamer(S), Streamer(S) { }

MCELFStreamer &AMDGPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(uint32_t Major,
                                                           uint32_t Minor) {
  MCStreamer &OS = getStreamer();
  MCSectionELF *Note = OS.getContext().getELFSection(".note", ELF::SHT_NOTE, 0);

  unsigned NameSZ = 4;

  OS.PushSection();
  OS.SwitchSection(Note);
  OS.EmitIntValue(NameSZ, 4);                            // namesz
  OS.EmitIntValue(8, 4);                                 // descz
  OS.EmitIntValue(NT_AMDGPU_HSA_CODE_OBJECT_VERSION, 4); // type
  OS.EmitBytes(StringRef("AMD", NameSZ));                // name
  OS.EmitIntValue(Major, 4);                             // desc
  OS.EmitIntValue(Minor, 4);
  OS.EmitValueToAlignment(4);
  OS.PopSection();
}

void
AMDGPUTargetELFStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  MCStreamer &OS = getStreamer();
  MCSectionELF *Note = OS.getContext().getELFSection(".note", ELF::SHT_NOTE, 0);

  unsigned NameSZ = 4;
  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;
  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
                    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
                    VendorNameSize + ArchNameSize;

  OS.PushSection();
  OS.SwitchSection(Note);
  OS.EmitIntValue(NameSZ, 4);                            // namesz
  OS.EmitIntValue(DescSZ, 4);                            // descsz
  OS.EmitIntValue(NT_AMDGPU_HSA_ISA, 4);                 // type
  OS.EmitBytes(StringRef("AMD", 4));                     // name
  OS.EmitIntValue(VendorNameSize, 2);                    // desc
  OS.EmitIntValue(ArchNameSize, 2);
  OS.EmitIntValue(Major, 4);
  OS.EmitIntValue(Minor, 4);
  OS.EmitIntValue(Stepping, 4);
  OS.EmitBytes(VendorName);
  OS.EmitIntValue(0, 1); // NULL terminate VendorName
  OS.EmitBytes(ArchName);
  OS.EmitIntValue(0, 1); // NULL terminte ArchName
  OS.EmitValueToAlignment(4);
  OS.PopSection();
}

void
AMDGPUTargetELFStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  // The MCObjectFileInfo that is available to the assembler is a generic
  // implementation and not AMDGPUHSATargetObjectFile, so we can't use
  // MCObjectFileInfo::getTextSection() here for fetching the HSATextSection.
  OS.SwitchSection(AMDGPU::getHSATextSection(OS.getContext()));
  OS.EmitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.PopSection();
}

void AMDGPUTargetELFStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(ELF::STT_AMDGPU_HSA_KERNEL);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUHsaModuleScopeGlobal(
    StringRef GlobalName) {

  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(GlobalName));
  Symbol->setType(ELF::STT_OBJECT);
  Symbol->setBinding(ELF::STB_LOCAL);
}

void AMDGPUTargetELFStreamer::EmitAMDGPUHsaProgramScopeGlobal(
    StringRef GlobalName) {

  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(GlobalName));
  Symbol->setType(ELF::STT_OBJECT);
  Symbol->setBinding(ELF::STB_GLOBAL);
}
