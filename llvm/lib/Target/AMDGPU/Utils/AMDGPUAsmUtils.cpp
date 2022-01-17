//===-- AMDGPUAsmUtils.cpp - AsmParser/InstPrinter common -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AMDGPUAsmUtils.h"
#include "SIDefines.h"

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace AMDGPU {
namespace SendMsg {

// This must be in sync with llvm::AMDGPU::SendMsg::Id enum members, see SIDefines.h.
const char *const IdSymbolic[ID_GAPS_LAST_] = {
  nullptr,
  "MSG_INTERRUPT",
  "MSG_GS",
  "MSG_GS_DONE",
  "MSG_SAVEWAVE",
  "MSG_STALL_WAVE_GEN",
  "MSG_HALT_WAVES",
  "MSG_ORDERED_PS_DONE",
  "MSG_EARLY_PRIM_DEALLOC",
  "MSG_GS_ALLOC_REQ",
  "MSG_GET_DOORBELL",
  "MSG_GET_DDID",
  nullptr,
  nullptr,
  nullptr,
  "MSG_SYSMSG"
};

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

// This must be in sync with llvm::AMDGPU::Hwreg::ID_SYMBOLIC_FIRST_/LAST_, see SIDefines.h.
const char* const IdSymbolic[] = {
  nullptr,
  "HW_REG_MODE",
  "HW_REG_STATUS",
  "HW_REG_TRAPSTS",
  "HW_REG_HW_ID",
  "HW_REG_GPR_ALLOC",
  "HW_REG_LDS_ALLOC",
  "HW_REG_IB_STS",
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  "HW_REG_SH_MEM_BASES",
  "HW_REG_TBA_LO",
  "HW_REG_TBA_HI",
  "HW_REG_TMA_LO",
  "HW_REG_TMA_HI",
  "HW_REG_FLAT_SCR_LO",
  "HW_REG_FLAT_SCR_HI",
  "HW_REG_XNACK_MASK",
  "HW_REG_HW_ID1",
  "HW_REG_HW_ID2",
  "HW_REG_POPS_PACKER",
  nullptr,
  nullptr,
  nullptr,
  "HW_REG_SHADER_CYCLES"
};

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

StringLiteral const UfmtSymbolic[] = {
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

unsigned const DfmtNfmt2UFmt[] = {
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
