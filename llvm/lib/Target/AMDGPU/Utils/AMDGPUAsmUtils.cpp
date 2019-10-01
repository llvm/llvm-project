//===-- AMDGPUAsmUtils.cpp - AsmParser/InstPrinter common -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "AMDGPUAsmUtils.h"

namespace llvm {
namespace AMDGPU {
namespace SendMsg {

// This must be in sync with llvm::AMDGPU::SendMsg::Id enum members, see SIDefines.h.
const char* const IdSymbolic[] = {
  nullptr,
  "MSG_INTERRUPT",
  "MSG_GS",
  "MSG_GS_DONE",
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  "MSG_GS_ALLOC_REQ",
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  "MSG_SYSMSG"
};

// These two must be in sync with llvm::AMDGPU::SendMsg::Op enum members, see SIDefines.h.
const char* const OpSysSymbolic[] = {
  nullptr,
  "SYSMSG_OP_ECC_ERR_INTERRUPT",
  "SYSMSG_OP_REG_RD",
  "SYSMSG_OP_HOST_TRAP_ACK",
  "SYSMSG_OP_TTRACE_PC"
};

const char* const OpGsSymbolic[] = {
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
  nullptr, // HW_ID1, no predictable values
  nullptr, // HW_ID2, no predictable values
  "HW_REG_POPS_PACKER"
};

} // namespace Hwreg

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
