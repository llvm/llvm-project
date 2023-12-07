//===-- AMDGPUAsmUtils.cpp - AsmParser/InstPrinter common -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AMDGPUAsmUtils.h"
#include "AMDGPUBaseInfo.h"
#include "SIDefines.h"

namespace llvm {
namespace AMDGPU {

namespace DepCtr {

// NOLINTBEGIN
const CustomOperandVal DepCtrInfo[] = {
  // Name               max dflt offset width  constraint
  {{"depctr_hold_cnt"},  1,   1,    7,    1,   isGFX10_BEncoding},
  {{"depctr_sa_sdst"},   1,   1,    0,    1},
  {{"depctr_va_vdst"},  15,  15,   12,    4},
  {{"depctr_va_sdst"},   7,   7,    9,    3},
  {{"depctr_va_ssrc"},   1,   1,    8,    1},
  {{"depctr_va_vcc"},    1,   1,    1,    1},
  {{"depctr_vm_vsrc"},   7,   7,    2,    3},
};
// NOLINTEND

const int DEP_CTR_SIZE =
    static_cast<int>(sizeof(DepCtrInfo) / sizeof(CustomOperandVal));

} // namespace DepCtr

namespace SendMsg {

// Disable lint checking for this block since it makes the table unreadable.
// NOLINTBEGIN
const CustomOperand<const MCSubtargetInfo &> Msg[] = {
  {{""}},
  {{"MSG_INTERRUPT"},           ID_INTERRUPT},
  {{"MSG_GS"},                  ID_GS_PreGFX11,             isNotGFX11Plus},
  {{"MSG_GS_DONE"},             ID_GS_DONE_PreGFX11,        isNotGFX11Plus},
  {{"MSG_SAVEWAVE"},            ID_SAVEWAVE,                isGFX8_GFX9_GFX10},
  {{"MSG_STALL_WAVE_GEN"},      ID_STALL_WAVE_GEN,          isGFX9Plus},
  {{"MSG_HALT_WAVES"},          ID_HALT_WAVES,              isGFX9Plus},
  {{"MSG_ORDERED_PS_DONE"},     ID_ORDERED_PS_DONE,         isGFX9_GFX10},
  {{"MSG_EARLY_PRIM_DEALLOC"},  ID_EARLY_PRIM_DEALLOC,      isGFX9_GFX10},
  {{"MSG_GS_ALLOC_REQ"},        ID_GS_ALLOC_REQ,            isGFX9Plus},
  {{"MSG_GET_DOORBELL"},        ID_GET_DOORBELL,            isGFX9_GFX10},
  {{"MSG_GET_DDID"},            ID_GET_DDID,                isGFX10},
  {{"MSG_HS_TESSFACTOR"},       ID_HS_TESSFACTOR_GFX11Plus, isGFX11Plus},
  {{"MSG_DEALLOC_VGPRS"},       ID_DEALLOC_VGPRS_GFX11Plus, isGFX11Plus},
  {{""}},
  {{"MSG_SYSMSG"},              ID_SYSMSG},
  {{"MSG_RTN_GET_DOORBELL"},    ID_RTN_GET_DOORBELL,        isGFX11Plus},
  {{"MSG_RTN_GET_DDID"},        ID_RTN_GET_DDID,            isGFX11Plus},
  {{"MSG_RTN_GET_TMA"},         ID_RTN_GET_TMA,             isGFX11Plus},
  {{"MSG_RTN_GET_REALTIME"},    ID_RTN_GET_REALTIME,        isGFX11Plus},
  {{"MSG_RTN_SAVE_WAVE"},       ID_RTN_SAVE_WAVE,           isGFX11Plus},
  {{"MSG_RTN_GET_TBA"},         ID_RTN_GET_TBA,             isGFX11Plus},
};
// NOLINTEND

const int MSG_SIZE = static_cast<int>(
    sizeof(Msg) / sizeof(CustomOperand<const MCSubtargetInfo &>));

// These two must be in sync with llvm::AMDGPU::SendMsg::Op enum members, see SIDefines.h.
const char *const OpSysSymbolic[OP_SYS_LAST_] = {
  nullptr,
  "SYSMSG_OP_ECC_ERR_INTERRUPT",
  "SYSMSG_OP_REG_RD",
  "SYSMSG_OP_HOST_TRAP_ACK",
  "SYSMSG_OP_TTRACE_PC"
};

const char *const OpGsSymbolic[OP_GS_LAST_] = {
  "GS_OP_NOP",
  "GS_OP_CUT",
  "GS_OP_EMIT",
  "GS_OP_EMIT_CUT"
};

} // namespace SendMsg

namespace Hwreg {

// Disable lint checking for this block since it makes the table unreadable.
// NOLINTBEGIN
const CustomOperand<const MCSubtargetInfo &> Opr[] = {
  {{""}},
  {{"HW_REG_MODE"},          ID_MODE},
  {{"HW_REG_STATUS"},        ID_STATUS},
  {{"HW_REG_TRAPSTS"},       ID_TRAPSTS},
  {{"HW_REG_HW_ID"},         ID_HW_ID,       isNotGFX10Plus},
  {{"HW_REG_GPR_ALLOC"},     ID_GPR_ALLOC},
  {{"HW_REG_LDS_ALLOC"},     ID_LDS_ALLOC},
  {{"HW_REG_IB_STS"},        ID_IB_STS},
  {{""}},
  {{""}},
  {{""}},
  {{""}},
  {{""}},
  {{""}},
  {{""}},
  {{"HW_REG_SH_MEM_BASES"},  ID_MEM_BASES,   isGFX9Plus},
  {{"HW_REG_TBA_LO"},        ID_TBA_LO,      isGFX9_GFX10},
  {{"HW_REG_TBA_HI"},        ID_TBA_HI,      isGFX9_GFX10},
  {{"HW_REG_TMA_LO"},        ID_TMA_LO,      isGFX9_GFX10},
  {{"HW_REG_TMA_HI"},        ID_TMA_HI,      isGFX9_GFX10},
  {{"HW_REG_FLAT_SCR_LO"},   ID_FLAT_SCR_LO, isGFX10Plus},
  {{"HW_REG_FLAT_SCR_HI"},   ID_FLAT_SCR_HI, isGFX10Plus},
  {{"HW_REG_XNACK_MASK"},    ID_XNACK_MASK,  isGFX10Before1030},
  {{"HW_REG_HW_ID1"},        ID_HW_ID1,      isGFX10Plus},
  {{"HW_REG_HW_ID2"},        ID_HW_ID2,      isGFX10Plus},
  {{"HW_REG_POPS_PACKER"},   ID_POPS_PACKER, isGFX10},
  {{""}},
  {{"HW_REG_PERF_SNAPSHOT_DATA"}, ID_PERF_SNAPSHOT_DATA, isGFX11Plus},
  {{""}},
  {{"HW_REG_SHADER_CYCLES"}, ID_SHADER_CYCLES, isGFX10_BEncoding},

  // Register numbers reused in GFX11+
  {{"HW_REG_PERF_SNAPSHOT_PC_LO"}, ID_PERF_SNAPSHOT_PC_LO, isGFX11Plus},
  {{"HW_REG_PERF_SNAPSHOT_PC_HI"}, ID_PERF_SNAPSHOT_PC_HI, isGFX11Plus},

  // GFX940 specific registers
  {{"HW_REG_XCC_ID"},                 ID_XCC_ID,                 isGFX940},
  {{"HW_REG_SQ_PERF_SNAPSHOT_DATA"},  ID_SQ_PERF_SNAPSHOT_DATA,  isGFX940},
  {{"HW_REG_SQ_PERF_SNAPSHOT_DATA1"}, ID_SQ_PERF_SNAPSHOT_DATA1, isGFX940},
  {{"HW_REG_SQ_PERF_SNAPSHOT_PC_LO"}, ID_SQ_PERF_SNAPSHOT_PC_LO, isGFX940},
  {{"HW_REG_SQ_PERF_SNAPSHOT_PC_HI"}, ID_SQ_PERF_SNAPSHOT_PC_HI, isGFX940},

  // Aliases
  {{"HW_REG_HW_ID"},                  ID_HW_ID1,                 isGFX10},
};
// NOLINTEND

const int OPR_SIZE = static_cast<int>(
    sizeof(Opr) / sizeof(CustomOperand<const MCSubtargetInfo &>));

} // namespace Hwreg

namespace MTBUFFormat {

StringLiteral const DfmtSymbolic[] = {
  "BUF_DATA_FORMAT_INVALID",
  "BUF_DATA_FORMAT_8",
  "BUF_DATA_FORMAT_16",
  "BUF_DATA_FORMAT_8_8",
  "BUF_DATA_FORMAT_32",
  "BUF_DATA_FORMAT_16_16",
  "BUF_DATA_FORMAT_10_11_11",
  "BUF_DATA_FORMAT_11_11_10",
  "BUF_DATA_FORMAT_10_10_10_2",
  "BUF_DATA_FORMAT_2_10_10_10",
  "BUF_DATA_FORMAT_8_8_8_8",
  "BUF_DATA_FORMAT_32_32",
  "BUF_DATA_FORMAT_16_16_16_16",
  "BUF_DATA_FORMAT_32_32_32",
  "BUF_DATA_FORMAT_32_32_32_32",
  "BUF_DATA_FORMAT_RESERVED_15"
};

StringLiteral const NfmtSymbolicGFX10[] = {
  "BUF_NUM_FORMAT_UNORM",
  "BUF_NUM_FORMAT_SNORM",
  "BUF_NUM_FORMAT_USCALED",
  "BUF_NUM_FORMAT_SSCALED",
  "BUF_NUM_FORMAT_UINT",
  "BUF_NUM_FORMAT_SINT",
  "",
  "BUF_NUM_FORMAT_FLOAT"
};

StringLiteral const NfmtSymbolicSICI[] = {
  "BUF_NUM_FORMAT_UNORM",
  "BUF_NUM_FORMAT_SNORM",
  "BUF_NUM_FORMAT_USCALED",
  "BUF_NUM_FORMAT_SSCALED",
  "BUF_NUM_FORMAT_UINT",
  "BUF_NUM_FORMAT_SINT",
  "BUF_NUM_FORMAT_SNORM_OGL",
  "BUF_NUM_FORMAT_FLOAT"
};

StringLiteral const NfmtSymbolicVI[] = {    // VI and GFX9
  "BUF_NUM_FORMAT_UNORM",
  "BUF_NUM_FORMAT_SNORM",
  "BUF_NUM_FORMAT_USCALED",
  "BUF_NUM_FORMAT_SSCALED",
  "BUF_NUM_FORMAT_UINT",
  "BUF_NUM_FORMAT_SINT",
  "BUF_NUM_FORMAT_RESERVED_6",
  "BUF_NUM_FORMAT_FLOAT"
};

StringLiteral const UfmtSymbolicGFX10[] = {
  "BUF_FMT_INVALID",

  "BUF_FMT_8_UNORM",
  "BUF_FMT_8_SNORM",
  "BUF_FMT_8_USCALED",
  "BUF_FMT_8_SSCALED",
  "BUF_FMT_8_UINT",
  "BUF_FMT_8_SINT",

  "BUF_FMT_16_UNORM",
  "BUF_FMT_16_SNORM",
  "BUF_FMT_16_USCALED",
  "BUF_FMT_16_SSCALED",
  "BUF_FMT_16_UINT",
  "BUF_FMT_16_SINT",
  "BUF_FMT_16_FLOAT",

  "BUF_FMT_8_8_UNORM",
  "BUF_FMT_8_8_SNORM",
  "BUF_FMT_8_8_USCALED",
  "BUF_FMT_8_8_SSCALED",
  "BUF_FMT_8_8_UINT",
  "BUF_FMT_8_8_SINT",

  "BUF_FMT_32_UINT",
  "BUF_FMT_32_SINT",
  "BUF_FMT_32_FLOAT",

  "BUF_FMT_16_16_UNORM",
  "BUF_FMT_16_16_SNORM",
  "BUF_FMT_16_16_USCALED",
  "BUF_FMT_16_16_SSCALED",
  "BUF_FMT_16_16_UINT",
  "BUF_FMT_16_16_SINT",
  "BUF_FMT_16_16_FLOAT",

  "BUF_FMT_10_11_11_UNORM",
  "BUF_FMT_10_11_11_SNORM",
  "BUF_FMT_10_11_11_USCALED",
  "BUF_FMT_10_11_11_SSCALED",
  "BUF_FMT_10_11_11_UINT",
  "BUF_FMT_10_11_11_SINT",
  "BUF_FMT_10_11_11_FLOAT",

  "BUF_FMT_11_11_10_UNORM",
  "BUF_FMT_11_11_10_SNORM",
  "BUF_FMT_11_11_10_USCALED",
  "BUF_FMT_11_11_10_SSCALED",
  "BUF_FMT_11_11_10_UINT",
  "BUF_FMT_11_11_10_SINT",
  "BUF_FMT_11_11_10_FLOAT",

  "BUF_FMT_10_10_10_2_UNORM",
  "BUF_FMT_10_10_10_2_SNORM",
  "BUF_FMT_10_10_10_2_USCALED",
  "BUF_FMT_10_10_10_2_SSCALED",
  "BUF_FMT_10_10_10_2_UINT",
  "BUF_FMT_10_10_10_2_SINT",

  "BUF_FMT_2_10_10_10_UNORM",
  "BUF_FMT_2_10_10_10_SNORM",
  "BUF_FMT_2_10_10_10_USCALED",
  "BUF_FMT_2_10_10_10_SSCALED",
  "BUF_FMT_2_10_10_10_UINT",
  "BUF_FMT_2_10_10_10_SINT",

  "BUF_FMT_8_8_8_8_UNORM",
  "BUF_FMT_8_8_8_8_SNORM",
  "BUF_FMT_8_8_8_8_USCALED",
  "BUF_FMT_8_8_8_8_SSCALED",
  "BUF_FMT_8_8_8_8_UINT",
  "BUF_FMT_8_8_8_8_SINT",

  "BUF_FMT_32_32_UINT",
  "BUF_FMT_32_32_SINT",
  "BUF_FMT_32_32_FLOAT",

  "BUF_FMT_16_16_16_16_UNORM",
  "BUF_FMT_16_16_16_16_SNORM",
  "BUF_FMT_16_16_16_16_USCALED",
  "BUF_FMT_16_16_16_16_SSCALED",
  "BUF_FMT_16_16_16_16_UINT",
  "BUF_FMT_16_16_16_16_SINT",
  "BUF_FMT_16_16_16_16_FLOAT",

  "BUF_FMT_32_32_32_UINT",
  "BUF_FMT_32_32_32_SINT",
  "BUF_FMT_32_32_32_FLOAT",
  "BUF_FMT_32_32_32_32_UINT",
  "BUF_FMT_32_32_32_32_SINT",
  "BUF_FMT_32_32_32_32_FLOAT"
};

unsigned const DfmtNfmt2UFmtGFX10[] = {
  DFMT_INVALID     | (NFMT_UNORM   << NFMT_SHIFT),

  DFMT_8           | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_8           | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_8           | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_8           | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_8           | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_8           | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_16          | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_16          | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_16          | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_16          | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_16          | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_16          | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_16          | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_8_8         | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_32          | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32          | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32          | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_16_16       | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_10_11_11    | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_10_11_11    | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_10_11_11    | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_10_11_11    | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_10_11_11    | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_10_11_11    | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_10_11_11    | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_11_11_10    | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_11_11_10    | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_11_11_10    | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_11_11_10    | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_11_11_10    | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_11_11_10    | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_11_11_10    | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_10_10_10_2  | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_2_10_10_10  | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_8_8_8_8     | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_32_32       | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32_32       | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32_32       | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_16_16_16_16 | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_32_32_32    | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32_32_32    | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32_32_32    | (NFMT_FLOAT   << NFMT_SHIFT),
  DFMT_32_32_32_32 | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32_32_32_32 | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32_32_32_32 | (NFMT_FLOAT   << NFMT_SHIFT)
};

StringLiteral const UfmtSymbolicGFX11[] = {
  "BUF_FMT_INVALID",

  "BUF_FMT_8_UNORM",
  "BUF_FMT_8_SNORM",
  "BUF_FMT_8_USCALED",
  "BUF_FMT_8_SSCALED",
  "BUF_FMT_8_UINT",
  "BUF_FMT_8_SINT",

  "BUF_FMT_16_UNORM",
  "BUF_FMT_16_SNORM",
  "BUF_FMT_16_USCALED",
  "BUF_FMT_16_SSCALED",
  "BUF_FMT_16_UINT",
  "BUF_FMT_16_SINT",
  "BUF_FMT_16_FLOAT",

  "BUF_FMT_8_8_UNORM",
  "BUF_FMT_8_8_SNORM",
  "BUF_FMT_8_8_USCALED",
  "BUF_FMT_8_8_SSCALED",
  "BUF_FMT_8_8_UINT",
  "BUF_FMT_8_8_SINT",

  "BUF_FMT_32_UINT",
  "BUF_FMT_32_SINT",
  "BUF_FMT_32_FLOAT",

  "BUF_FMT_16_16_UNORM",
  "BUF_FMT_16_16_SNORM",
  "BUF_FMT_16_16_USCALED",
  "BUF_FMT_16_16_SSCALED",
  "BUF_FMT_16_16_UINT",
  "BUF_FMT_16_16_SINT",
  "BUF_FMT_16_16_FLOAT",

  "BUF_FMT_10_11_11_FLOAT",

  "BUF_FMT_11_11_10_FLOAT",

  "BUF_FMT_10_10_10_2_UNORM",
  "BUF_FMT_10_10_10_2_SNORM",
  "BUF_FMT_10_10_10_2_UINT",
  "BUF_FMT_10_10_10_2_SINT",

  "BUF_FMT_2_10_10_10_UNORM",
  "BUF_FMT_2_10_10_10_SNORM",
  "BUF_FMT_2_10_10_10_USCALED",
  "BUF_FMT_2_10_10_10_SSCALED",
  "BUF_FMT_2_10_10_10_UINT",
  "BUF_FMT_2_10_10_10_SINT",

  "BUF_FMT_8_8_8_8_UNORM",
  "BUF_FMT_8_8_8_8_SNORM",
  "BUF_FMT_8_8_8_8_USCALED",
  "BUF_FMT_8_8_8_8_SSCALED",
  "BUF_FMT_8_8_8_8_UINT",
  "BUF_FMT_8_8_8_8_SINT",

  "BUF_FMT_32_32_UINT",
  "BUF_FMT_32_32_SINT",
  "BUF_FMT_32_32_FLOAT",

  "BUF_FMT_16_16_16_16_UNORM",
  "BUF_FMT_16_16_16_16_SNORM",
  "BUF_FMT_16_16_16_16_USCALED",
  "BUF_FMT_16_16_16_16_SSCALED",
  "BUF_FMT_16_16_16_16_UINT",
  "BUF_FMT_16_16_16_16_SINT",
  "BUF_FMT_16_16_16_16_FLOAT",

  "BUF_FMT_32_32_32_UINT",
  "BUF_FMT_32_32_32_SINT",
  "BUF_FMT_32_32_32_FLOAT",
  "BUF_FMT_32_32_32_32_UINT",
  "BUF_FMT_32_32_32_32_SINT",
  "BUF_FMT_32_32_32_32_FLOAT"
};

unsigned const DfmtNfmt2UFmtGFX11[] = {
  DFMT_INVALID     | (NFMT_UNORM   << NFMT_SHIFT),

  DFMT_8           | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_8           | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_8           | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_8           | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_8           | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_8           | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_16          | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_16          | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_16          | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_16          | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_16          | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_16          | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_16          | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_8_8         | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_8_8         | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_32          | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32          | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32          | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_16_16       | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_16_16       | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_10_11_11    | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_11_11_10    | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_10_10_10_2  | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_10_10_10_2  | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_2_10_10_10  | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_2_10_10_10  | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_8_8_8_8     | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_8_8_8_8     | (NFMT_SINT    << NFMT_SHIFT),

  DFMT_32_32       | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32_32       | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32_32       | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_16_16_16_16 | (NFMT_UNORM   << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_SNORM   << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_USCALED << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_SSCALED << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_16_16_16_16 | (NFMT_FLOAT   << NFMT_SHIFT),

  DFMT_32_32_32    | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32_32_32    | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32_32_32    | (NFMT_FLOAT   << NFMT_SHIFT),
  DFMT_32_32_32_32 | (NFMT_UINT    << NFMT_SHIFT),
  DFMT_32_32_32_32 | (NFMT_SINT    << NFMT_SHIFT),
  DFMT_32_32_32_32 | (NFMT_FLOAT   << NFMT_SHIFT)
};

} // namespace MTBUFFormat

namespace Swizzle {

// This must be in sync with llvm::AMDGPU::Swizzle::Id enum members, see SIDefines.h.
const char* const IdSymbolic[] = {
  "QUAD_PERM",
  "BITMASK_PERM",
  "SWAP",
  "REVERSE",
  "BROADCAST",
};

} // namespace Swizzle

namespace VGPRIndexMode {

// This must be in sync with llvm::AMDGPU::VGPRIndexMode::Id enum members, see SIDefines.h.
const char* const IdSymbolic[] = {
  "SRC0",
  "SRC1",
  "SRC2",
  "DST",
};

} // namespace VGPRIndexMode

} // namespace AMDGPU
} // namespace llvm
