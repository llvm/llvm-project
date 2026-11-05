//===-- RegisterContextDarwin_arm64.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterContextDarwin_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-ehframe-register-enums.h"

#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-types.h"
#include "lldb/lldb-private.h"
#include <cstddef>
#include <cstdint>

#include "RegisterContextDarwinConstants.h"

#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include "Plugins/Process/Utility/InstructionUtils.h"

#include <memory>

#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"

namespace dwarf_regs = lldb_private::registers::arm64::dwarf;
namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb;
using namespace lldb_private;

#define GPR_OFFSET(idx) ((idx)*8)
#define GPR_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_arm64::GPR, reg))

#define FPU_OFFSET(idx) ((idx)*16 + sizeof(RegisterContextDarwin_arm64::GPR))
#define FPU_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_arm64::FPU, reg))

#define EXC_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_arm64::EXC, reg) +            \
   sizeof(RegisterContextDarwin_arm64::GPR) +                                  \
   sizeof(RegisterContextDarwin_arm64::FPU))
#define DBG_OFFSET_NAME(reg)                                                   \
  (LLVM_EXTENSION offsetof(RegisterContextDarwin_arm64::DBG, reg) +            \
   sizeof(RegisterContextDarwin_arm64::GPR) +                                  \
   sizeof(RegisterContextDarwin_arm64::FPU) +                                  \
   sizeof(RegisterContextDarwin_arm64::EXC))

#define DEFINE_DBG(reg, i)                                                     \
  #reg, NULL,                                                                  \
      sizeof(((RegisterContextDarwin_arm64::DBG *) NULL)->reg[i]),             \
              DBG_OFFSET_NAME(reg[i]), eEncodingUint, eFormatHex,              \
                              {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       \
                               LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       \
                               LLDB_INVALID_REGNUM },                          \
                               NULL, NULL, NULL
#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_arm64::GPR) +                                  \
   sizeof(RegisterContextDarwin_arm64::FPU) +                                  \
   sizeof(RegisterContextDarwin_arm64::EXC))

// Include RegisterInfos_arm64 to declare our g_register_infos_arm64 structure.
#define DECLARE_REGISTER_INFOS_ARM64_STRUCT
#include "Register/Arch/RegisterInfos_arm64.h"
#undef DECLARE_REGISTER_INFOS_ARM64_STRUCT

// General purpose registers
static uint32_t g_gpr_regnums[] = {
    reg_ids::gpr_x0,  reg_ids::gpr_x1,  reg_ids::gpr_x2,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,  reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_x8,  reg_ids::gpr_x9,  reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13, reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17, reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21, reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25, reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_fp,  reg_ids::gpr_lr,  reg_ids::gpr_sp,
    reg_ids::gpr_pc,  reg_ids::gpr_cpsr};

// Floating point registers
static uint32_t g_fpu_regnums[] = {
    reg_ids::fpu_v0,   reg_ids::fpu_v1,  reg_ids::fpu_v2,  reg_ids::fpu_v3,
    reg_ids::fpu_v4,   reg_ids::fpu_v5,  reg_ids::fpu_v6,  reg_ids::fpu_v7,
    reg_ids::fpu_v8,   reg_ids::fpu_v9,  reg_ids::fpu_v10, reg_ids::fpu_v11,
    reg_ids::fpu_v12,  reg_ids::fpu_v13, reg_ids::fpu_v14, reg_ids::fpu_v15,
    reg_ids::fpu_v16,  reg_ids::fpu_v17, reg_ids::fpu_v18, reg_ids::fpu_v19,
    reg_ids::fpu_v20,  reg_ids::fpu_v21, reg_ids::fpu_v22, reg_ids::fpu_v23,
    reg_ids::fpu_v24,  reg_ids::fpu_v25, reg_ids::fpu_v26, reg_ids::fpu_v27,
    reg_ids::fpu_v28,  reg_ids::fpu_v29, reg_ids::fpu_v30, reg_ids::fpu_v31,
    reg_ids::fpu_fpsr, reg_ids::fpu_fpcr};

// Exception registers

static uint32_t g_exc_regnums[] = {reg_ids::exc_far, reg_ids::exc_esr,
                                   reg_ids::exc_exception};

static size_t k_num_register_infos = std::size(g_register_infos_arm64_le);

RegisterContextDarwin_arm64::RegisterContextDarwin_arm64(
    Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContext(thread, concrete_frame_idx), gpr(), fpu(), exc(), dbg() {
  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpu_errs[i] = -1;
    exc_errs[i] = -1;
  }
}

RegisterContextDarwin_arm64::~RegisterContextDarwin_arm64() = default;

void RegisterContextDarwin_arm64::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

size_t RegisterContextDarwin_arm64::GetRegisterCount() {
  assert(k_num_register_infos == reg_ids::k_num_registers);
  return reg_ids::k_num_registers;
}

const RegisterInfo *
RegisterContextDarwin_arm64::GetRegisterInfoAtIndex(size_t reg) {
  assert(k_num_register_infos == reg_ids::k_num_registers);
  if (reg < reg_ids::k_num_registers)
    return &g_register_infos_arm64_le[reg];
  return nullptr;
}

size_t RegisterContextDarwin_arm64::GetRegisterInfosCount() {
  return k_num_register_infos;
}

const RegisterInfo *RegisterContextDarwin_arm64::GetRegisterInfos() {
  return g_register_infos_arm64_le;
}

// Number of registers in each register set
const size_t k_num_gpr_registers = std::size(g_gpr_regnums);
const size_t k_num_fpu_registers = std::size(g_fpu_regnums);
const size_t k_num_exc_registers = std::size(g_exc_regnums);

// Register set definitions. The first definitions at register set index of
// zero is for all registers, followed by other registers sets. The register
// information for the all register set need not be filled in.
static const RegisterSet g_reg_sets[] = {
    {
        "General Purpose Registers", "gpr", k_num_gpr_registers, g_gpr_regnums,
    },
    {"Floating Point Registers", "fpu", k_num_fpu_registers, g_fpu_regnums},
    {"Exception State Registers", "exc", k_num_exc_registers, g_exc_regnums}};

const size_t k_num_regsets = std::size(g_reg_sets);

size_t RegisterContextDarwin_arm64::GetRegisterSetCount() {
  return k_num_regsets;
}

const RegisterSet *RegisterContextDarwin_arm64::GetRegisterSet(size_t reg_set) {
  if (reg_set < k_num_regsets)
    return &g_reg_sets[reg_set];
  return nullptr;
}

// Register information definitions for arm64
int RegisterContextDarwin_arm64::GetSetForNativeRegNum(uint32_t reg) {
  if (reg < reg_ids::fpu_v0)
    return GPRRegSet;
  else if (reg < reg_ids::exc_far)
    return FPURegSet;
  else if (reg < reg_ids::k_num_registers)
    return EXCRegSet;
  return -1;
}

int RegisterContextDarwin_arm64::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(GPRRegSet, Read);
}

int RegisterContextDarwin_arm64::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
  }
  return GetError(FPURegSet, Read);
}

int RegisterContextDarwin_arm64::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(EXCRegSet, Read);
}

int RegisterContextDarwin_arm64::ReadDBG(bool force) {
  int set = DBGRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadDBG(GetThreadID(), set, dbg));
  }
  return GetError(DBGRegSet, Read);
}

int RegisterContextDarwin_arm64::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(GPRRegSet, Write);
}

int RegisterContextDarwin_arm64::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpu));
  SetError(set, Read, -1);
  return GetError(FPURegSet, Write);
}

int RegisterContextDarwin_arm64::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(EXCRegSet, Write);
}

int RegisterContextDarwin_arm64::WriteDBG() {
  int set = DBGRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return KERN_INVALID_ARGUMENT;
  }
  SetError(set, Write, DoWriteDBG(GetThreadID(), set, dbg));
  SetError(set, Read, -1);
  return GetError(DBGRegSet, Write);
}

int RegisterContextDarwin_arm64::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  case DBGRegSet:
    return ReadDBG(force);
  default:
    break;
  }
  return KERN_INVALID_ARGUMENT;
}

int RegisterContextDarwin_arm64::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case FPURegSet:
      return WriteFPU();
    case EXCRegSet:
      return WriteEXC();
    case DBGRegSet:
      return WriteDBG();
    default:
      break;
    }
  }
  return KERN_INVALID_ARGUMENT;
}

void RegisterContextDarwin_arm64::LogDBGRegisters(Log *log, const DBG &dbg) {
  if (log) {
    for (uint32_t i = 0; i < 16; i++)
      LLDB_LOGF(log,
                "BVR%-2u/BCR%-2u = { 0x%8.8" PRIu64 ", 0x%8.8" PRIu64
                " } WVR%-2u/WCR%-2u "
                "= { 0x%8.8" PRIu64 ", 0x%8.8" PRIu64 " }",
                i, i, dbg.bvr[i], dbg.bcr[i], i, i, dbg.wvr[i], dbg.wcr[i]);
  }
}

bool RegisterContextDarwin_arm64::ReadRegister(const RegisterInfo *reg_info,
                                               RegisterValue &value) {
  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  int set = RegisterContextDarwin_arm64::GetSetForNativeRegNum(reg);

  if (set == -1)
    return false;

  if (ReadRegisterSet(set, false) != KERN_SUCCESS)
    return false;

  switch (reg) {
  case reg_ids::gpr_x0:
  case reg_ids::gpr_x1:
  case reg_ids::gpr_x2:
  case reg_ids::gpr_x3:
  case reg_ids::gpr_x4:
  case reg_ids::gpr_x5:
  case reg_ids::gpr_x6:
  case reg_ids::gpr_x7:
  case reg_ids::gpr_x8:
  case reg_ids::gpr_x9:
  case reg_ids::gpr_x10:
  case reg_ids::gpr_x11:
  case reg_ids::gpr_x12:
  case reg_ids::gpr_x13:
  case reg_ids::gpr_x14:
  case reg_ids::gpr_x15:
  case reg_ids::gpr_x16:
  case reg_ids::gpr_x17:
  case reg_ids::gpr_x18:
  case reg_ids::gpr_x19:
  case reg_ids::gpr_x20:
  case reg_ids::gpr_x21:
  case reg_ids::gpr_x22:
  case reg_ids::gpr_x23:
  case reg_ids::gpr_x24:
  case reg_ids::gpr_x25:
  case reg_ids::gpr_x26:
  case reg_ids::gpr_x27:
  case reg_ids::gpr_x28:
    value.SetUInt64(gpr.x[reg - reg_ids::gpr_x0]);
    break;
  case reg_ids::gpr_fp:
    value.SetUInt64(gpr.fp);
    break;
  case reg_ids::gpr_sp:
    value.SetUInt64(gpr.sp);
    break;
  case reg_ids::gpr_lr:
    value.SetUInt64(gpr.lr);
    break;
  case reg_ids::gpr_pc:
    value.SetUInt64(gpr.pc);
    break;
  case reg_ids::gpr_cpsr:
    value.SetUInt64(gpr.cpsr);
    break;

  case reg_ids::gpr_w0:
  case reg_ids::gpr_w1:
  case reg_ids::gpr_w2:
  case reg_ids::gpr_w3:
  case reg_ids::gpr_w4:
  case reg_ids::gpr_w5:
  case reg_ids::gpr_w6:
  case reg_ids::gpr_w7:
  case reg_ids::gpr_w8:
  case reg_ids::gpr_w9:
  case reg_ids::gpr_w10:
  case reg_ids::gpr_w11:
  case reg_ids::gpr_w12:
  case reg_ids::gpr_w13:
  case reg_ids::gpr_w14:
  case reg_ids::gpr_w15:
  case reg_ids::gpr_w16:
  case reg_ids::gpr_w17:
  case reg_ids::gpr_w18:
  case reg_ids::gpr_w19:
  case reg_ids::gpr_w20:
  case reg_ids::gpr_w21:
  case reg_ids::gpr_w22:
  case reg_ids::gpr_w23:
  case reg_ids::gpr_w24:
  case reg_ids::gpr_w25:
  case reg_ids::gpr_w26:
  case reg_ids::gpr_w27:
  case reg_ids::gpr_w28: {
    ProcessSP process_sp(m_thread.GetProcess());
    if (process_sp.get()) {
      DataExtractor regdata(&gpr.x[reg - reg_ids::gpr_w0], 8,
                            process_sp->GetByteOrder(),
                            process_sp->GetAddressByteSize());
      offset_t offset = 0;
      uint64_t retval = regdata.GetMaxU64(&offset, 8);
      uint32_t retval_lower32 = static_cast<uint32_t>(retval & 0xffffffff);
      value.SetUInt32(retval_lower32);
    }
  } break;

  case reg_ids::fpu_v0:
  case reg_ids::fpu_v1:
  case reg_ids::fpu_v2:
  case reg_ids::fpu_v3:
  case reg_ids::fpu_v4:
  case reg_ids::fpu_v5:
  case reg_ids::fpu_v6:
  case reg_ids::fpu_v7:
  case reg_ids::fpu_v8:
  case reg_ids::fpu_v9:
  case reg_ids::fpu_v10:
  case reg_ids::fpu_v11:
  case reg_ids::fpu_v12:
  case reg_ids::fpu_v13:
  case reg_ids::fpu_v14:
  case reg_ids::fpu_v15:
  case reg_ids::fpu_v16:
  case reg_ids::fpu_v17:
  case reg_ids::fpu_v18:
  case reg_ids::fpu_v19:
  case reg_ids::fpu_v20:
  case reg_ids::fpu_v21:
  case reg_ids::fpu_v22:
  case reg_ids::fpu_v23:
  case reg_ids::fpu_v24:
  case reg_ids::fpu_v25:
  case reg_ids::fpu_v26:
  case reg_ids::fpu_v27:
  case reg_ids::fpu_v28:
  case reg_ids::fpu_v29:
  case reg_ids::fpu_v30:
  case reg_ids::fpu_v31:
    value.SetBytes(fpu.v[reg - reg_ids::fpu_v0].bytes, reg_info->byte_size,
                   endian::InlHostByteOrder());
    break;

  case reg_ids::fpu_s0:
  case reg_ids::fpu_s1:
  case reg_ids::fpu_s2:
  case reg_ids::fpu_s3:
  case reg_ids::fpu_s4:
  case reg_ids::fpu_s5:
  case reg_ids::fpu_s6:
  case reg_ids::fpu_s7:
  case reg_ids::fpu_s8:
  case reg_ids::fpu_s9:
  case reg_ids::fpu_s10:
  case reg_ids::fpu_s11:
  case reg_ids::fpu_s12:
  case reg_ids::fpu_s13:
  case reg_ids::fpu_s14:
  case reg_ids::fpu_s15:
  case reg_ids::fpu_s16:
  case reg_ids::fpu_s17:
  case reg_ids::fpu_s18:
  case reg_ids::fpu_s19:
  case reg_ids::fpu_s20:
  case reg_ids::fpu_s21:
  case reg_ids::fpu_s22:
  case reg_ids::fpu_s23:
  case reg_ids::fpu_s24:
  case reg_ids::fpu_s25:
  case reg_ids::fpu_s26:
  case reg_ids::fpu_s27:
  case reg_ids::fpu_s28:
  case reg_ids::fpu_s29:
  case reg_ids::fpu_s30:
  case reg_ids::fpu_s31: {
    ProcessSP process_sp(m_thread.GetProcess());
    if (process_sp.get()) {
      DataExtractor regdata(&fpu.v[reg - reg_ids::fpu_s0], 4,
                            process_sp->GetByteOrder(),
                            process_sp->GetAddressByteSize());
      offset_t offset = 0;
      value.SetFloat(regdata.GetFloat(&offset));
    }
  } break;

  case reg_ids::fpu_d0:
  case reg_ids::fpu_d1:
  case reg_ids::fpu_d2:
  case reg_ids::fpu_d3:
  case reg_ids::fpu_d4:
  case reg_ids::fpu_d5:
  case reg_ids::fpu_d6:
  case reg_ids::fpu_d7:
  case reg_ids::fpu_d8:
  case reg_ids::fpu_d9:
  case reg_ids::fpu_d10:
  case reg_ids::fpu_d11:
  case reg_ids::fpu_d12:
  case reg_ids::fpu_d13:
  case reg_ids::fpu_d14:
  case reg_ids::fpu_d15:
  case reg_ids::fpu_d16:
  case reg_ids::fpu_d17:
  case reg_ids::fpu_d18:
  case reg_ids::fpu_d19:
  case reg_ids::fpu_d20:
  case reg_ids::fpu_d21:
  case reg_ids::fpu_d22:
  case reg_ids::fpu_d23:
  case reg_ids::fpu_d24:
  case reg_ids::fpu_d25:
  case reg_ids::fpu_d26:
  case reg_ids::fpu_d27:
  case reg_ids::fpu_d28:
  case reg_ids::fpu_d29:
  case reg_ids::fpu_d30:
  case reg_ids::fpu_d31: {
    ProcessSP process_sp(m_thread.GetProcess());
    if (process_sp.get()) {
      DataExtractor regdata(&fpu.v[reg - reg_ids::fpu_d0], 8,
                            process_sp->GetByteOrder(),
                            process_sp->GetAddressByteSize());
      offset_t offset = 0;
      value.SetDouble(regdata.GetDouble(&offset));
    }
  } break;

  case reg_ids::fpu_fpsr:
    value.SetUInt32(fpu.fpsr);
    break;

  case reg_ids::fpu_fpcr:
    value.SetUInt32(fpu.fpcr);
    break;

  case reg_ids::exc_exception:
    value.SetUInt32(exc.exception);
    break;
  case reg_ids::exc_esr:
    value.SetUInt32(exc.esr);
    break;
  case reg_ids::exc_far:
    value.SetUInt64(exc.far);
    break;

  default:
    value.SetValueToInvalid();
    return false;
  }
  return true;
}

bool RegisterContextDarwin_arm64::WriteRegister(const RegisterInfo *reg_info,
                                                const RegisterValue &value) {
  const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
  int set = GetSetForNativeRegNum(reg);

  if (set == -1)
    return false;

  if (ReadRegisterSet(set, false) != KERN_SUCCESS)
    return false;

  switch (reg) {
  case reg_ids::gpr_x0:
  case reg_ids::gpr_x1:
  case reg_ids::gpr_x2:
  case reg_ids::gpr_x3:
  case reg_ids::gpr_x4:
  case reg_ids::gpr_x5:
  case reg_ids::gpr_x6:
  case reg_ids::gpr_x7:
  case reg_ids::gpr_x8:
  case reg_ids::gpr_x9:
  case reg_ids::gpr_x10:
  case reg_ids::gpr_x11:
  case reg_ids::gpr_x12:
  case reg_ids::gpr_x13:
  case reg_ids::gpr_x14:
  case reg_ids::gpr_x15:
  case reg_ids::gpr_x16:
  case reg_ids::gpr_x17:
  case reg_ids::gpr_x18:
  case reg_ids::gpr_x19:
  case reg_ids::gpr_x20:
  case reg_ids::gpr_x21:
  case reg_ids::gpr_x22:
  case reg_ids::gpr_x23:
  case reg_ids::gpr_x24:
  case reg_ids::gpr_x25:
  case reg_ids::gpr_x26:
  case reg_ids::gpr_x27:
  case reg_ids::gpr_x28:
  case reg_ids::gpr_fp:
  case reg_ids::gpr_sp:
  case reg_ids::gpr_lr:
  case reg_ids::gpr_pc:
  case reg_ids::gpr_cpsr:
    gpr.x[reg - reg_ids::gpr_x0] = value.GetAsUInt64();
    break;

  case reg_ids::fpu_v0:
  case reg_ids::fpu_v1:
  case reg_ids::fpu_v2:
  case reg_ids::fpu_v3:
  case reg_ids::fpu_v4:
  case reg_ids::fpu_v5:
  case reg_ids::fpu_v6:
  case reg_ids::fpu_v7:
  case reg_ids::fpu_v8:
  case reg_ids::fpu_v9:
  case reg_ids::fpu_v10:
  case reg_ids::fpu_v11:
  case reg_ids::fpu_v12:
  case reg_ids::fpu_v13:
  case reg_ids::fpu_v14:
  case reg_ids::fpu_v15:
  case reg_ids::fpu_v16:
  case reg_ids::fpu_v17:
  case reg_ids::fpu_v18:
  case reg_ids::fpu_v19:
  case reg_ids::fpu_v20:
  case reg_ids::fpu_v21:
  case reg_ids::fpu_v22:
  case reg_ids::fpu_v23:
  case reg_ids::fpu_v24:
  case reg_ids::fpu_v25:
  case reg_ids::fpu_v26:
  case reg_ids::fpu_v27:
  case reg_ids::fpu_v28:
  case reg_ids::fpu_v29:
  case reg_ids::fpu_v30:
  case reg_ids::fpu_v31:
    ::memcpy(fpu.v[reg - reg_ids::fpu_v0].bytes, value.GetBytes(),
             value.GetByteSize());
    break;

  case reg_ids::fpu_fpsr:
    fpu.fpsr = value.GetAsUInt32();
    break;

  case reg_ids::fpu_fpcr:
    fpu.fpcr = value.GetAsUInt32();
    break;

  case reg_ids::exc_exception:
    exc.exception = value.GetAsUInt32();
    break;
  case reg_ids::exc_esr:
    exc.esr = value.GetAsUInt32();
    break;
  case reg_ids::exc_far:
    exc.far = value.GetAsUInt64();
    break;

  default:
    return false;
  }
  return WriteRegisterSet(set) == KERN_SUCCESS;
}

bool RegisterContextDarwin_arm64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == KERN_SUCCESS && ReadFPU(false) == KERN_SUCCESS &&
      ReadEXC(false) == KERN_SUCCESS) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpu, sizeof(fpu));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_arm64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpu, src, sizeof(fpu));
    src += sizeof(gpr);

    ::memcpy(&exc, src, sizeof(exc));
    uint32_t success_count = 0;
    if (WriteGPR() == KERN_SUCCESS)
      ++success_count;
    if (WriteFPU() == KERN_SUCCESS)
      ++success_count;
    if (WriteEXC() == KERN_SUCCESS)
      ++success_count;
    return success_count == 3;
  }
  return false;
}

uint32_t RegisterContextDarwin_arm64::ConvertRegisterKindToRegisterNumber(
    RegisterKind kind, uint32_t reg) {
  if (kind == eRegisterKindGeneric) {
    switch (reg) {
    case LLDB_REGNUM_GENERIC_PC:
      return reg_ids::gpr_pc;
    case LLDB_REGNUM_GENERIC_SP:
      return reg_ids::gpr_sp;
    case LLDB_REGNUM_GENERIC_FP:
      return reg_ids::gpr_fp;
    case LLDB_REGNUM_GENERIC_RA:
      return reg_ids::gpr_lr;
    case LLDB_REGNUM_GENERIC_FLAGS:
      return reg_ids::gpr_cpsr;
    default:
      break;
    }
  } else if (kind == eRegisterKindDWARF) {
    switch (reg) {
    case dwarf_regs::x0:
      return reg_ids::gpr_x0;
    case dwarf_regs::x1:
      return reg_ids::gpr_x1;
    case dwarf_regs::x2:
      return reg_ids::gpr_x2;
    case dwarf_regs::x3:
      return reg_ids::gpr_x3;
    case dwarf_regs::x4:
      return reg_ids::gpr_x4;
    case dwarf_regs::x5:
      return reg_ids::gpr_x5;
    case dwarf_regs::x6:
      return reg_ids::gpr_x6;
    case dwarf_regs::x7:
      return reg_ids::gpr_x7;
    case dwarf_regs::x8:
      return reg_ids::gpr_x8;
    case dwarf_regs::x9:
      return reg_ids::gpr_x9;
    case dwarf_regs::x10:
      return reg_ids::gpr_x10;
    case dwarf_regs::x11:
      return reg_ids::gpr_x11;
    case dwarf_regs::x12:
      return reg_ids::gpr_x12;
    case dwarf_regs::x13:
      return reg_ids::gpr_x13;
    case dwarf_regs::x14:
      return reg_ids::gpr_x14;
    case dwarf_regs::x15:
      return reg_ids::gpr_x15;
    case dwarf_regs::x16:
      return reg_ids::gpr_x16;
    case dwarf_regs::x17:
      return reg_ids::gpr_x17;
    case dwarf_regs::x18:
      return reg_ids::gpr_x18;
    case dwarf_regs::x19:
      return reg_ids::gpr_x19;
    case dwarf_regs::x20:
      return reg_ids::gpr_x20;
    case dwarf_regs::x21:
      return reg_ids::gpr_x21;
    case dwarf_regs::x22:
      return reg_ids::gpr_x22;
    case dwarf_regs::x23:
      return reg_ids::gpr_x23;
    case dwarf_regs::x24:
      return reg_ids::gpr_x24;
    case dwarf_regs::x25:
      return reg_ids::gpr_x25;
    case dwarf_regs::x26:
      return reg_ids::gpr_x26;
    case dwarf_regs::x27:
      return reg_ids::gpr_x27;
    case dwarf_regs::x28:
      return reg_ids::gpr_x28;

    case dwarf_regs::fp:
      return reg_ids::gpr_fp;
    case dwarf_regs::sp:
      return reg_ids::gpr_sp;
    case dwarf_regs::lr:
      return reg_ids::gpr_lr;
    case dwarf_regs::pc:
      return reg_ids::gpr_pc;

    case dwarf_regs::v0:
      return reg_ids::fpu_v0;
    case dwarf_regs::v1:
      return reg_ids::fpu_v1;
    case dwarf_regs::v2:
      return reg_ids::fpu_v2;
    case dwarf_regs::v3:
      return reg_ids::fpu_v3;
    case dwarf_regs::v4:
      return reg_ids::fpu_v4;
    case dwarf_regs::v5:
      return reg_ids::fpu_v5;
    case dwarf_regs::v6:
      return reg_ids::fpu_v6;
    case dwarf_regs::v7:
      return reg_ids::fpu_v7;
    case dwarf_regs::v8:
      return reg_ids::fpu_v8;
    case dwarf_regs::v9:
      return reg_ids::fpu_v9;
    case dwarf_regs::v10:
      return reg_ids::fpu_v10;
    case dwarf_regs::v11:
      return reg_ids::fpu_v11;
    case dwarf_regs::v12:
      return reg_ids::fpu_v12;
    case dwarf_regs::v13:
      return reg_ids::fpu_v13;
    case dwarf_regs::v14:
      return reg_ids::fpu_v14;
    case dwarf_regs::v15:
      return reg_ids::fpu_v15;
    case dwarf_regs::v16:
      return reg_ids::fpu_v16;
    case dwarf_regs::v17:
      return reg_ids::fpu_v17;
    case dwarf_regs::v18:
      return reg_ids::fpu_v18;
    case dwarf_regs::v19:
      return reg_ids::fpu_v19;
    case dwarf_regs::v20:
      return reg_ids::fpu_v20;
    case dwarf_regs::v21:
      return reg_ids::fpu_v21;
    case dwarf_regs::v22:
      return reg_ids::fpu_v22;
    case dwarf_regs::v23:
      return reg_ids::fpu_v23;
    case dwarf_regs::v24:
      return reg_ids::fpu_v24;
    case dwarf_regs::v25:
      return reg_ids::fpu_v25;
    case dwarf_regs::v26:
      return reg_ids::fpu_v26;
    case dwarf_regs::v27:
      return reg_ids::fpu_v27;
    case dwarf_regs::v28:
      return reg_ids::fpu_v28;
    case dwarf_regs::v29:
      return reg_ids::fpu_v29;
    case dwarf_regs::v30:
      return reg_ids::fpu_v30;
    case dwarf_regs::v31:
      return reg_ids::fpu_v31;

    default:
      break;
    }
  } else if (kind == eRegisterKindEHFrame) {
    switch (reg) {
    case dwarf_regs::x0:
      return reg_ids::gpr_x0;
    case dwarf_regs::x1:
      return reg_ids::gpr_x1;
    case dwarf_regs::x2:
      return reg_ids::gpr_x2;
    case dwarf_regs::x3:
      return reg_ids::gpr_x3;
    case dwarf_regs::x4:
      return reg_ids::gpr_x4;
    case dwarf_regs::x5:
      return reg_ids::gpr_x5;
    case dwarf_regs::x6:
      return reg_ids::gpr_x6;
    case dwarf_regs::x7:
      return reg_ids::gpr_x7;
    case dwarf_regs::x8:
      return reg_ids::gpr_x8;
    case dwarf_regs::x9:
      return reg_ids::gpr_x9;
    case dwarf_regs::x10:
      return reg_ids::gpr_x10;
    case dwarf_regs::x11:
      return reg_ids::gpr_x11;
    case dwarf_regs::x12:
      return reg_ids::gpr_x12;
    case dwarf_regs::x13:
      return reg_ids::gpr_x13;
    case dwarf_regs::x14:
      return reg_ids::gpr_x14;
    case dwarf_regs::x15:
      return reg_ids::gpr_x15;
    case dwarf_regs::x16:
      return reg_ids::gpr_x16;
    case dwarf_regs::x17:
      return reg_ids::gpr_x17;
    case dwarf_regs::x18:
      return reg_ids::gpr_x18;
    case dwarf_regs::x19:
      return reg_ids::gpr_x19;
    case dwarf_regs::x20:
      return reg_ids::gpr_x20;
    case dwarf_regs::x21:
      return reg_ids::gpr_x21;
    case dwarf_regs::x22:
      return reg_ids::gpr_x22;
    case dwarf_regs::x23:
      return reg_ids::gpr_x23;
    case dwarf_regs::x24:
      return reg_ids::gpr_x24;
    case dwarf_regs::x25:
      return reg_ids::gpr_x25;
    case dwarf_regs::x26:
      return reg_ids::gpr_x26;
    case dwarf_regs::x27:
      return reg_ids::gpr_x27;
    case dwarf_regs::x28:
      return reg_ids::gpr_x28;
    case dwarf_regs::fp:
      return reg_ids::gpr_fp;
    case dwarf_regs::sp:
      return reg_ids::gpr_sp;
    case dwarf_regs::lr:
      return reg_ids::gpr_lr;
    case dwarf_regs::pc:
      return reg_ids::gpr_pc;
    }
  } else if (kind == eRegisterKindLLDB) {
    return reg;
  }
  return LLDB_INVALID_REGNUM;
}

uint32_t RegisterContextDarwin_arm64::NumSupportedHardwareWatchpoints() {
#if defined(__APPLE__) && (defined(__arm64__) || defined(__aarch64__))
  // autodetect how many watchpoints are supported dynamically...
  static uint32_t g_num_supported_hw_watchpoints = UINT32_MAX;
  if (g_num_supported_hw_watchpoints == UINT32_MAX) {
    size_t len;
    uint32_t n = 0;
    len = sizeof(n);
    if (::sysctlbyname("hw.optional.watchpoint", &n, &len, NULL, 0) == 0) {
      g_num_supported_hw_watchpoints = n;
    }
  }
  return g_num_supported_hw_watchpoints;
#else
  // TODO: figure out remote case here!
  return 2;
#endif
}

uint32_t RegisterContextDarwin_arm64::SetHardwareWatchpoint(lldb::addr_t addr,
                                                            size_t size,
                                                            bool read,
                                                            bool write) {
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint(addr = %8.8p,
  //    size = %u, read = %u, write = %u)", addr, size, read, write);

  const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

  // Can't watch zero bytes
  if (size == 0)
    return LLDB_INVALID_INDEX32;

  // We must watch for either read or write
  if (!read && !write)
    return LLDB_INVALID_INDEX32;

  // Can't watch more than 4 bytes per WVR/WCR pair
  if (size > 4)
    return LLDB_INVALID_INDEX32;

  // We can only watch up to four bytes that follow a 4 byte aligned address
  // per watchpoint register pair. Since we have at most so we can only watch
  // until the next 4 byte boundary and we need to make sure we can properly
  // encode this.
  uint32_t addr_word_offset = addr % 4;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() -
  //    addr_word_offset = 0x%8.8x", addr_word_offset);

  uint32_t byte_mask = ((1u << size) - 1u) << addr_word_offset;
  //    if (log) log->Printf
  //    ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() - byte_mask =
  //    0x%8.8x", byte_mask);
  if (byte_mask > 0xfu)
    return LLDB_INVALID_INDEX32;

  // Read the debug state
  int kret = ReadDBG(false);

  if (kret == KERN_SUCCESS) {
    // Check to make sure we have the needed hardware support
    uint32_t i = 0;

    for (i = 0; i < num_hw_watchpoints; ++i) {
      if ((dbg.wcr[i] & WCR_ENABLE) == 0)
        break; // We found an available hw breakpoint slot (in i)
    }

    // See if we found an available hw breakpoint slot above
    if (i < num_hw_watchpoints) {
      // Make the byte_mask into a valid Byte Address Select mask
      uint32_t byte_address_select = byte_mask << 5;
      // Make sure bits 1:0 are clear in our address
      dbg.wvr[i] = addr & ~((lldb::addr_t)3);
      dbg.wcr[i] = byte_address_select |     // Which bytes that follow the IMVA
                                             // that we will watch
                   S_USER |                  // Stop only in user mode
                   (read ? WCR_LOAD : 0) |   // Stop on read access?
                   (write ? WCR_STORE : 0) | // Stop on write access?
                   WCR_ENABLE;               // Enable this watchpoint;

      kret = WriteDBG();
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint()
      //            WriteDBG() => 0x%8.8x.", kret);

      if (kret == KERN_SUCCESS)
        return i;
    } else {
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint():
      //            All hardware resources (%u) are in use.",
      //            num_hw_watchpoints);
    }
  }
  return LLDB_INVALID_INDEX32;
}

bool RegisterContextDarwin_arm64::ClearHardwareWatchpoint(uint32_t hw_index) {
  int kret = ReadDBG(false);

  const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
  if (kret == KERN_SUCCESS) {
    if (hw_index < num_hw_points) {
      dbg.wcr[hw_index] = 0;
      //            if (log) log->Printf
      //            ("RegisterContextDarwin_arm64::ClearHardwareWatchpoint( %u )
      //            - WVR%u = 0x%8.8x  WCR%u = 0x%8.8x",
      //                    hw_index,
      //                    hw_index,
      //                    dbg.wvr[hw_index],
      //                    hw_index,
      //                    dbg.wcr[hw_index]);

      kret = WriteDBG();

      if (kret == KERN_SUCCESS)
        return true;
    }
  }
  return false;
}
