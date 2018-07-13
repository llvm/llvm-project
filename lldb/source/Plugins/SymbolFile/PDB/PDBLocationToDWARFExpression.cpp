//===-- PDBLocationToDWARFExpression.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PDBLocationToDWARFExpression.h"

#include "lldb/Core/Section.h"
#include "lldb/Core/StreamBuffer.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Utility/DataBufferHeap.h"

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/PDB/PDBSymbolData.h"

#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm::pdb;

namespace {
const uint32_t g_code_view_to_lldb_registers_x86[] = {
    LLDB_INVALID_REGNUM, // CVRegNONE
    lldb_al_i386,        // CVRegAL
    lldb_cl_i386,        // CVRegCL
    lldb_dl_i386,        // CVRegDL
    lldb_bl_i386,        // CVRegBL
    lldb_ah_i386,        // CVRegAH
    lldb_ch_i386,        // CVRegCH
    lldb_dh_i386,        // CVRegDH
    lldb_bh_i386,        // CVRegBH
    lldb_ax_i386,        // CVRegAX
    lldb_cx_i386,        // CVRegCX
    lldb_dx_i386,        // CVRegDX
    lldb_bx_i386,        // CVRegBX
    lldb_sp_i386,        // CVRegSP
    lldb_bp_i386,        // CVRegBP
    lldb_si_i386,        // CVRegSI
    lldb_di_i386,        // CVRegDI
    lldb_eax_i386,       // CVRegEAX
    lldb_ecx_i386,       // CVRegECX
    lldb_edx_i386,       // CVRegEDX
    lldb_ebx_i386,       // CVRegEBX
    lldb_esp_i386,       // CVRegESP
    lldb_ebp_i386,       // CVRegEBP
    lldb_esi_i386,       // CVRegESI
    lldb_edi_i386,       // CVRegEDI
    lldb_es_i386,        // CVRegES
    lldb_cs_i386,        // CVRegCS
    lldb_ss_i386,        // CVRegSS
    lldb_ds_i386,        // CVRegDS
    lldb_fs_i386,        // CVRegFS
    lldb_gs_i386,        // CVRegGS
    LLDB_INVALID_REGNUM, // CVRegIP
    LLDB_INVALID_REGNUM, // CVRegFLAGS
    lldb_eip_i386,       // CVRegEIP
    lldb_eflags_i386,    // CVRegEFLAGS
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegTEMP
    LLDB_INVALID_REGNUM, // CVRegTEMPH
    LLDB_INVALID_REGNUM, // CVRegQUOTE
    LLDB_INVALID_REGNUM, // CVRegPCDR3
    LLDB_INVALID_REGNUM, // CVRegPCDR4
    LLDB_INVALID_REGNUM, // CVRegPCDR5
    LLDB_INVALID_REGNUM, // CVRegPCDR6
    LLDB_INVALID_REGNUM, // CVRegPCDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegCR0
    LLDB_INVALID_REGNUM, // CVRegCR1
    LLDB_INVALID_REGNUM, // CVRegCR2
    LLDB_INVALID_REGNUM, // CVRegCR3
    LLDB_INVALID_REGNUM, // CVRegCR4
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_dr0_i386, // CVRegDR0
    lldb_dr1_i386, // CVRegDR1
    lldb_dr2_i386, // CVRegDR2
    lldb_dr3_i386, // CVRegDR3
    lldb_dr4_i386, // CVRegDR4
    lldb_dr5_i386, // CVRegDR5
    lldb_dr6_i386, // CVRegDR6
    lldb_dr7_i386, // CVRegDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegGDTR
    LLDB_INVALID_REGNUM, // CVRegGDTL
    LLDB_INVALID_REGNUM, // CVRegIDTR
    LLDB_INVALID_REGNUM, // CVRegIDTL
    LLDB_INVALID_REGNUM, // CVRegLDTR
    LLDB_INVALID_REGNUM, // CVRegTR
    LLDB_INVALID_REGNUM, // CVRegPSEUDO1
    LLDB_INVALID_REGNUM, // CVRegPSEUDO2
    LLDB_INVALID_REGNUM, // CVRegPSEUDO3
    LLDB_INVALID_REGNUM, // CVRegPSEUDO4
    LLDB_INVALID_REGNUM, // CVRegPSEUDO5
    LLDB_INVALID_REGNUM, // CVRegPSEUDO6
    LLDB_INVALID_REGNUM, // CVRegPSEUDO7
    LLDB_INVALID_REGNUM, // CVRegPSEUDO8
    LLDB_INVALID_REGNUM, // CVRegPSEUDO9
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_st0_i386,       // CVRegST0
    lldb_st1_i386,       // CVRegST1
    lldb_st2_i386,       // CVRegST2
    lldb_st3_i386,       // CVRegST3
    lldb_st4_i386,       // CVRegST4
    lldb_st5_i386,       // CVRegST5
    lldb_st6_i386,       // CVRegST6
    lldb_st7_i386,       // CVRegST7
    LLDB_INVALID_REGNUM, // CVRegCTRL
    LLDB_INVALID_REGNUM, // CVRegSTAT
    LLDB_INVALID_REGNUM, // CVRegTAG
    LLDB_INVALID_REGNUM, // CVRegFPIP
    LLDB_INVALID_REGNUM, // CVRegFPCS
    LLDB_INVALID_REGNUM, // CVRegFPDO
    LLDB_INVALID_REGNUM, // CVRegFPDS
    LLDB_INVALID_REGNUM, // CVRegISEM
    LLDB_INVALID_REGNUM, // CVRegFPEIP
    LLDB_INVALID_REGNUM, // CVRegFPEDO
    lldb_mm0_i386,       // CVRegMM0
    lldb_mm1_i386,       // CVRegMM1
    lldb_mm2_i386,       // CVRegMM2
    lldb_mm3_i386,       // CVRegMM3
    lldb_mm4_i386,       // CVRegMM4
    lldb_mm5_i386,       // CVRegMM5
    lldb_mm6_i386,       // CVRegMM6
    lldb_mm7_i386,       // CVRegMM7
    lldb_xmm0_i386,      // CVRegXMM0
    lldb_xmm1_i386,      // CVRegXMM1
    lldb_xmm2_i386,      // CVRegXMM2
    lldb_xmm3_i386,      // CVRegXMM3
    lldb_xmm4_i386,      // CVRegXMM4
    lldb_xmm5_i386,      // CVRegXMM5
    lldb_xmm6_i386,      // CVRegXMM6
    lldb_xmm7_i386       // CVRegXMM7
};

const uint32_t g_code_view_to_lldb_registers_x86_64[] = {
    LLDB_INVALID_REGNUM, // CVRegNONE
    lldb_al_x86_64,      // CVRegAL
    lldb_cl_x86_64,      // CVRegCL
    lldb_dl_x86_64,      // CVRegDL
    lldb_bl_x86_64,      // CVRegBL
    lldb_ah_x86_64,      // CVRegAH
    lldb_ch_x86_64,      // CVRegCH
    lldb_dh_x86_64,      // CVRegDH
    lldb_bh_x86_64,      // CVRegBH
    lldb_ax_x86_64,      // CVRegAX
    lldb_cx_x86_64,      // CVRegCX
    lldb_dx_x86_64,      // CVRegDX
    lldb_bx_x86_64,      // CVRegBX
    lldb_sp_x86_64,      // CVRegSP
    lldb_bp_x86_64,      // CVRegBP
    lldb_si_x86_64,      // CVRegSI
    lldb_di_x86_64,      // CVRegDI
    lldb_eax_x86_64,     // CVRegEAX
    lldb_ecx_x86_64,     // CVRegECX
    lldb_edx_x86_64,     // CVRegEDX
    lldb_ebx_x86_64,     // CVRegEBX
    lldb_esp_x86_64,     // CVRegESP
    lldb_ebp_x86_64,     // CVRegEBP
    lldb_esi_x86_64,     // CVRegESI
    lldb_edi_x86_64,     // CVRegEDI
    lldb_es_x86_64,      // CVRegES
    lldb_cs_x86_64,      // CVRegCS
    lldb_ss_x86_64,      // CVRegSS
    lldb_ds_x86_64,      // CVRegDS
    lldb_fs_x86_64,      // CVRegFS
    lldb_gs_x86_64,      // CVRegGS
    LLDB_INVALID_REGNUM, // CVRegIP
    LLDB_INVALID_REGNUM, // CVRegFLAGS
    LLDB_INVALID_REGNUM, // CVRegEIP
    LLDB_INVALID_REGNUM, // CVRegEFLAGS
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegTEMP
    LLDB_INVALID_REGNUM, // CVRegTEMPH
    LLDB_INVALID_REGNUM, // CVRegQUOTE
    LLDB_INVALID_REGNUM, // CVRegPCDR3
    LLDB_INVALID_REGNUM, // CVRegPCDR4
    LLDB_INVALID_REGNUM, // CVRegPCDR5
    LLDB_INVALID_REGNUM, // CVRegPCDR6
    LLDB_INVALID_REGNUM, // CVRegPCDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegCR0
    LLDB_INVALID_REGNUM, // CVRegCR1
    LLDB_INVALID_REGNUM, // CVRegCR2
    LLDB_INVALID_REGNUM, // CVRegCR3
    LLDB_INVALID_REGNUM, // CVRegCR4
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_dr0_x86_64, // CVRegDR0
    lldb_dr1_x86_64, // CVRegDR1
    lldb_dr2_x86_64, // CVRegDR2
    lldb_dr3_x86_64, // CVRegDR3
    lldb_dr4_x86_64, // CVRegDR4
    lldb_dr5_x86_64, // CVRegDR5
    lldb_dr6_x86_64, // CVRegDR6
    lldb_dr7_x86_64, // CVRegDR7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegGDTR
    LLDB_INVALID_REGNUM, // CVRegGDTL
    LLDB_INVALID_REGNUM, // CVRegIDTR
    LLDB_INVALID_REGNUM, // CVRegIDTL
    LLDB_INVALID_REGNUM, // CVRegLDTR
    LLDB_INVALID_REGNUM, // CVRegTR
    LLDB_INVALID_REGNUM, // CVRegPSEUDO1
    LLDB_INVALID_REGNUM, // CVRegPSEUDO2
    LLDB_INVALID_REGNUM, // CVRegPSEUDO3
    LLDB_INVALID_REGNUM, // CVRegPSEUDO4
    LLDB_INVALID_REGNUM, // CVRegPSEUDO5
    LLDB_INVALID_REGNUM, // CVRegPSEUDO6
    LLDB_INVALID_REGNUM, // CVRegPSEUDO7
    LLDB_INVALID_REGNUM, // CVRegPSEUDO8
    LLDB_INVALID_REGNUM, // CVRegPSEUDO9
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_st0_x86_64,     // CVRegST0
    lldb_st1_x86_64,     // CVRegST1
    lldb_st2_x86_64,     // CVRegST2
    lldb_st3_x86_64,     // CVRegST3
    lldb_st4_x86_64,     // CVRegST4
    lldb_st5_x86_64,     // CVRegST5
    lldb_st6_x86_64,     // CVRegST6
    lldb_st7_x86_64,     // CVRegST7
    LLDB_INVALID_REGNUM, // CVRegCTRL
    LLDB_INVALID_REGNUM, // CVRegSTAT
    LLDB_INVALID_REGNUM, // CVRegTAG
    LLDB_INVALID_REGNUM, // CVRegFPIP
    LLDB_INVALID_REGNUM, // CVRegFPCS
    LLDB_INVALID_REGNUM, // CVRegFPDO
    LLDB_INVALID_REGNUM, // CVRegFPDS
    LLDB_INVALID_REGNUM, // CVRegISEM
    LLDB_INVALID_REGNUM, // CVRegFPEIP
    LLDB_INVALID_REGNUM, // CVRegFPEDO
    lldb_mm0_x86_64,     // CVRegMM0
    lldb_mm1_x86_64,     // CVRegMM1
    lldb_mm2_x86_64,     // CVRegMM2
    lldb_mm3_x86_64,     // CVRegMM3
    lldb_mm4_x86_64,     // CVRegMM4
    lldb_mm5_x86_64,     // CVRegMM5
    lldb_mm6_x86_64,     // CVRegMM6
    lldb_mm7_x86_64,     // CVRegMM7
    lldb_xmm0_x86_64,    // CVRegXMM0
    lldb_xmm1_x86_64,    // CVRegXMM1
    lldb_xmm2_x86_64,    // CVRegXMM2
    lldb_xmm3_x86_64,    // CVRegXMM3
    lldb_xmm4_x86_64,    // CVRegXMM4
    lldb_xmm5_x86_64,    // CVRegXMM5
    lldb_xmm6_x86_64,    // CVRegXMM6
    lldb_xmm7_x86_64,    // CVRegXMM7
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    lldb_mxcsr_x86_64,   // CVRegMXCSR
    LLDB_INVALID_REGNUM, // CVRegEDXEAX
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, // CVRegEMM0L
    LLDB_INVALID_REGNUM, // CVRegEMM1L
    LLDB_INVALID_REGNUM, // CVRegEMM2L
    LLDB_INVALID_REGNUM, // CVRegEMM3L
    LLDB_INVALID_REGNUM, // CVRegEMM4L
    LLDB_INVALID_REGNUM, // CVRegEMM5L
    LLDB_INVALID_REGNUM, // CVRegEMM6L
    LLDB_INVALID_REGNUM, // CVRegEMM7L
    LLDB_INVALID_REGNUM, // CVRegEMM0H
    LLDB_INVALID_REGNUM, // CVRegEMM1H
    LLDB_INVALID_REGNUM, // CVRegEMM2H
    LLDB_INVALID_REGNUM, // CVRegEMM3H
    LLDB_INVALID_REGNUM, // CVRegEMM4H
    LLDB_INVALID_REGNUM, // CVRegEMM5H
    LLDB_INVALID_REGNUM, // CVRegEMM6H
    LLDB_INVALID_REGNUM, // CVRegEMM7H
    LLDB_INVALID_REGNUM, // CVRegMM00
    LLDB_INVALID_REGNUM, // CVRegMM01
    LLDB_INVALID_REGNUM, // CVRegMM10
    LLDB_INVALID_REGNUM, // CVRegMM11
    LLDB_INVALID_REGNUM, // CVRegMM20
    LLDB_INVALID_REGNUM, // CVRegMM21
    LLDB_INVALID_REGNUM, // CVRegMM30
    LLDB_INVALID_REGNUM, // CVRegMM31
    LLDB_INVALID_REGNUM, // CVRegMM40
    LLDB_INVALID_REGNUM, // CVRegMM41
    LLDB_INVALID_REGNUM, // CVRegMM50
    LLDB_INVALID_REGNUM, // CVRegMM51
    LLDB_INVALID_REGNUM, // CVRegMM60
    LLDB_INVALID_REGNUM, // CVRegMM61
    LLDB_INVALID_REGNUM, // CVRegMM70
    LLDB_INVALID_REGNUM, // CVRegMM71
    lldb_xmm8_x86_64,    // CVRegXMM8
    lldb_xmm9_x86_64,    // CVRegXMM9
    lldb_xmm10_x86_64,   // CVRegXMM10
    lldb_xmm11_x86_64,   // CVRegXMM11
    lldb_xmm12_x86_64,   // CVRegXMM12
    lldb_xmm13_x86_64,   // CVRegXMM13
    lldb_xmm14_x86_64,   // CVRegXMM14
    lldb_xmm15_x86_64,   // CVRegXMM15
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM,
    lldb_sil_x86_64,   // CVRegSIL
    lldb_dil_x86_64,   // CVRegDIL
    lldb_bpl_x86_64,   // CVRegBPL
    lldb_spl_x86_64,   // CVRegSPL
    lldb_rax_x86_64,   // CVRegRAX
    lldb_rbx_x86_64,   // CVRegRBX
    lldb_rcx_x86_64,   // CVRegRCX
    lldb_rdx_x86_64,   // CVRegRDX
    lldb_rsi_x86_64,   // CVRegRSI
    lldb_rdi_x86_64,   // CVRegRDI
    lldb_rbp_x86_64,   // CVRegRBP
    lldb_rsp_x86_64,   // CVRegRSP
    lldb_r8_x86_64,    // CVRegR8
    lldb_r9_x86_64,    // CVRegR9
    lldb_r10_x86_64,   // CVRegR10
    lldb_r11_x86_64,   // CVRegR11
    lldb_r12_x86_64,   // CVRegR12
    lldb_r13_x86_64,   // CVRegR13
    lldb_r14_x86_64,   // CVRegR14
    lldb_r15_x86_64,   // CVRegR15
    lldb_r8l_x86_64,   // CVRegR8B
    lldb_r9l_x86_64,   // CVRegR9B
    lldb_r10l_x86_64,  // CVRegR10B
    lldb_r11l_x86_64,  // CVRegR11B
    lldb_r12l_x86_64,  // CVRegR12B
    lldb_r13l_x86_64,  // CVRegR13B
    lldb_r14l_x86_64,  // CVRegR14B
    lldb_r15l_x86_64,  // CVRegR15B
    lldb_r8w_x86_64,   // CVRegR8W
    lldb_r9w_x86_64,   // CVRegR9W
    lldb_r10w_x86_64,  // CVRegR10W
    lldb_r11w_x86_64,  // CVRegR11W
    lldb_r12w_x86_64,  // CVRegR12W
    lldb_r13w_x86_64,  // CVRegR13W
    lldb_r14w_x86_64,  // CVRegR14W
    lldb_r15w_x86_64,  // CVRegR15W
    lldb_r8d_x86_64,   // CVRegR8D
    lldb_r9d_x86_64,   // CVRegR9D
    lldb_r10d_x86_64,  // CVRegR10D
    lldb_r11d_x86_64,  // CVRegR11D
    lldb_r12d_x86_64,  // CVRegR12D
    lldb_r13d_x86_64,  // CVRegR13D
    lldb_r14d_x86_64,  // CVRegR14D
    lldb_r15d_x86_64,  // CVRegR15D
    lldb_ymm0_x86_64,  // CVRegAMD64_YMM0
    lldb_ymm1_x86_64,  // CVRegAMD64_YMM1
    lldb_ymm2_x86_64,  // CVRegAMD64_YMM2
    lldb_ymm3_x86_64,  // CVRegAMD64_YMM3
    lldb_ymm4_x86_64,  // CVRegAMD64_YMM4
    lldb_ymm5_x86_64,  // CVRegAMD64_YMM5
    lldb_ymm6_x86_64,  // CVRegAMD64_YMM6
    lldb_ymm7_x86_64,  // CVRegAMD64_YMM7
    lldb_ymm8_x86_64,  // CVRegAMD64_YMM8
    lldb_ymm9_x86_64,  // CVRegAMD64_YMM9
    lldb_ymm10_x86_64, // CVRegAMD64_YMM10
    lldb_ymm11_x86_64, // CVRegAMD64_YMM11
    lldb_ymm12_x86_64, // CVRegAMD64_YMM12
    lldb_ymm13_x86_64, // CVRegAMD64_YMM13
    lldb_ymm14_x86_64, // CVRegAMD64_YMM14
    lldb_ymm15_x86_64, // CVRegAMD64_YMM15
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
    lldb_bnd0_x86_64, // CVRegBND0
    lldb_bnd1_x86_64, // CVRegBND1
    lldb_bnd2_x86_64  // CVRegBND2
};

uint32_t GetLLDBRegisterNumber(llvm::Triple::ArchType arch_type,
                               llvm::codeview::RegisterId register_id) {
  switch (arch_type) {
  case llvm::Triple::x86:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_x86) /
            sizeof(g_code_view_to_lldb_registers_x86[0]))
      return g_code_view_to_lldb_registers_x86[static_cast<uint16_t>(
          register_id)];

    switch (register_id) {
    case llvm::codeview::RegisterId::CVRegMXCSR:
      return lldb_mxcsr_i386;
    case llvm::codeview::RegisterId::CVRegBND0:
      return lldb_bnd0_i386;
    case llvm::codeview::RegisterId::CVRegBND1:
      return lldb_bnd1_i386;
    case llvm::codeview::RegisterId::CVRegBND2:
      return lldb_bnd2_i386;
    default:
      return LLDB_INVALID_REGNUM;
    }
  case llvm::Triple::x86_64:
    if (static_cast<uint16_t>(register_id) <
        sizeof(g_code_view_to_lldb_registers_x86_64) /
            sizeof(g_code_view_to_lldb_registers_x86_64[0]))
      return g_code_view_to_lldb_registers_x86_64[static_cast<uint16_t>(
          register_id)];

    return LLDB_INVALID_REGNUM;
  default:
    return LLDB_INVALID_REGNUM;
  }
}

uint32_t GetGenericRegisterNumber(llvm::codeview::RegisterId register_id) {
  if (register_id == llvm::codeview::RegisterId::CVRegVFRAME)
    return LLDB_REGNUM_GENERIC_FP;

  return LLDB_INVALID_REGNUM;
}

uint32_t GetRegisterNumber(llvm::Triple::ArchType arch_type,
                           llvm::codeview::RegisterId register_id,
                           RegisterKind &register_kind) {
  register_kind = eRegisterKindLLDB;
  uint32_t reg_num = GetLLDBRegisterNumber(arch_type, register_id);
  if (reg_num != LLDB_INVALID_REGNUM)
    return reg_num;

  register_kind = eRegisterKindGeneric;
  return GetGenericRegisterNumber(register_id);
}
} // namespace

DWARFExpression ConvertPDBLocationToDWARFExpression(ModuleSP module,
                                                    const PDBSymbolData &symbol,
                                                    bool &is_constant) {
  is_constant = true;

  if (!module)
    return DWARFExpression(nullptr);

  const ArchSpec &architecture = module->GetArchitecture();
  llvm::Triple::ArchType arch_type = architecture.GetMachine();
  ByteOrder byte_order = architecture.GetByteOrder();
  uint32_t address_size = architecture.GetAddressByteSize();
  uint32_t byte_size = architecture.GetDataByteSize();
  if (byte_order == eByteOrderInvalid || address_size == 0)
    return DWARFExpression(nullptr);

  RegisterKind register_kind = eRegisterKindDWARF;
  StreamBuffer<32> stream(Stream::eBinary, address_size, byte_order);
  switch (symbol.getLocationType()) {
  case PDB_LocType::Static:
  case PDB_LocType::TLS: {
    stream.PutHex8(DW_OP_addr);

    SectionList *section_list = module->GetSectionList();
    if (!section_list)
      return DWARFExpression(nullptr);

    uint32_t section_idx = symbol.getAddressSection() - 1;
    if (section_idx >= section_list->GetSize())
      return DWARFExpression(nullptr);

    auto section = section_list->GetSectionAtIndex(section_idx);
    if (!section)
      return DWARFExpression(nullptr);

    uint32_t offset = symbol.getAddressOffset();
    stream.PutMaxHex64(section->GetFileAddress() + offset, address_size,
                       byte_order);

    is_constant = false;

    break;
  }
  case PDB_LocType::RegRel: {
    uint32_t reg_num =
        GetRegisterNumber(arch_type, symbol.getRegisterId(), register_kind);
    if (reg_num == LLDB_INVALID_REGNUM)
      return DWARFExpression(nullptr);

    if (reg_num > 31) {
      stream.PutHex8(DW_OP_bregx);
      stream.PutULEB128(reg_num);
    } else
      stream.PutHex8(DW_OP_breg0 + reg_num);

    int32_t offset = symbol.getOffset();
    stream.PutSLEB128(offset);

    is_constant = false;

    break;
  }
  case PDB_LocType::Enregistered: {
    uint32_t reg_num =
        GetRegisterNumber(arch_type, symbol.getRegisterId(), register_kind);
    if (reg_num == LLDB_INVALID_REGNUM)
      return DWARFExpression(nullptr);

    if (reg_num > 31) {
      stream.PutHex8(DW_OP_regx);
      stream.PutULEB128(reg_num);
    } else
      stream.PutHex8(DW_OP_reg0 + reg_num);

    is_constant = false;

    break;
  }
  case PDB_LocType::Constant: {
    Variant value = symbol.getValue();
    stream.PutRawBytes(&value.Value, sizeof(value.Value),
                       endian::InlHostByteOrder());
    break;
  }
  default:
    return DWARFExpression(nullptr);
  }

  DataBufferSP buffer =
      std::make_shared<DataBufferHeap>(stream.GetData(), stream.GetSize());
  DataExtractor extractor(buffer, byte_order, address_size, byte_size);
  DWARFExpression result(module, extractor, nullptr, 0, buffer->GetByteSize());
  result.SetRegisterKind(register_kind);

  return result;
}
