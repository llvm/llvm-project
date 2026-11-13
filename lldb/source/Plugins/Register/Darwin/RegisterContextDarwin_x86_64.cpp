//===-- RegisterContextDarwin_x86_64.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cinttypes>
#include <cstdarg>
#include <cstddef>

#include <memory>

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Register/Common/RegisterInfo_x86.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include "RegisterContextDarwin_x86_64.h"

using namespace lldb;
using namespace lldb_private;

enum {
  gpr_rax = 0,
  gpr_rbx,
  gpr_rcx,
  gpr_rdx,
  gpr_rdi,
  gpr_rsi,
  gpr_rbp,
  gpr_rsp,
  gpr_r8,
  gpr_r9,
  gpr_r10,
  gpr_r11,
  gpr_r12,
  gpr_r13,
  gpr_r14,
  gpr_r15,
  gpr_rip,
  gpr_rflags,
  gpr_cs,
  gpr_fs,
  gpr_gs,

  fpu_fcw,
  fpu_fsw,
  fpu_ftw,
  fpu_fop,
  fpu_ip,
  fpu_cs,
  fpu_dp,
  fpu_ds,
  fpu_mxcsr,
  fpu_mxcsrmask,
  fpu_stmm0,
  fpu_stmm1,
  fpu_stmm2,
  fpu_stmm3,
  fpu_stmm4,
  fpu_stmm5,
  fpu_stmm6,
  fpu_stmm7,
  fpu_xmm0,
  fpu_xmm1,
  fpu_xmm2,
  fpu_xmm3,
  fpu_xmm4,
  fpu_xmm5,
  fpu_xmm6,
  fpu_xmm7,
  fpu_xmm8,
  fpu_xmm9,
  fpu_xmm10,
  fpu_xmm11,
  fpu_xmm12,
  fpu_xmm13,
  fpu_xmm14,
  fpu_xmm15,

  exc_trapno,
  exc_err,
  exc_faultvaddr,

  k_num_registers,

  // Aliases
  fpu_fctrl = fpu_fcw,
  fpu_fstat = fpu_fsw,
  fpu_ftag = fpu_ftw,
  fpu_fiseg = fpu_cs,
  fpu_fioff = fpu_ip,
  fpu_foseg = fpu_ds,
  fpu_fooff = fpu_dp
};

#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_x86_64::GPR) +                                 \
   sizeof(RegisterContextDarwin_x86_64::FPU) +                                 \
   sizeof(RegisterContextDarwin_x86_64::EXC))

// General purpose registers for 64 bit
static std::shared_ptr<const RegisterDescription> CreateContextDescription();

RegisterContextDarwin_x86_64::RegisterContextDarwin_x86_64(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpu(), exc() {
  SetRegisterDescription(CreateContextDescription());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpu_errs[i] = -1;
    exc_errs[i] = -1;
  }
}

RegisterContextDarwin_x86_64::~RegisterContextDarwin_x86_64() = default;

void RegisterContextDarwin_x86_64::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

static uint32_t g_gpr_regnums[] = {
    gpr_rax, gpr_rbx, gpr_rcx, gpr_rdx,    gpr_rdi, gpr_rsi, gpr_rbp,
    gpr_rsp, gpr_r8,  gpr_r9,  gpr_r10,    gpr_r11, gpr_r12, gpr_r13,
    gpr_r14, gpr_r15, gpr_rip, gpr_rflags, gpr_cs,  gpr_fs,  gpr_gs};

static uint32_t g_fpu_regnums[] = {
    fpu_fcw,   fpu_fsw,   fpu_ftw,   fpu_fop,       fpu_ip,    fpu_cs,
    fpu_dp,    fpu_ds,    fpu_mxcsr, fpu_mxcsrmask, fpu_stmm0, fpu_stmm1,
    fpu_stmm2, fpu_stmm3, fpu_stmm4, fpu_stmm5,     fpu_stmm6, fpu_stmm7,
    fpu_xmm0,  fpu_xmm1,  fpu_xmm2,  fpu_xmm3,      fpu_xmm4,  fpu_xmm5,
    fpu_xmm6,  fpu_xmm7,  fpu_xmm8,  fpu_xmm9,      fpu_xmm10, fpu_xmm11,
    fpu_xmm12, fpu_xmm13, fpu_xmm14, fpu_xmm15};

static uint32_t g_exc_regnums[] = {exc_trapno, exc_err, exc_faultvaddr};

// Number of registers in each register set
const size_t k_num_gpr_registers = std::size(g_gpr_regnums);
const size_t k_num_fpu_registers = std::size(g_fpu_regnums);
const size_t k_num_exc_registers = std::size(g_exc_regnums);

// Register set definitions. The first definitions at register set index of
// zero is for all registers, followed by other registers sets. The register
// information for the all register set need not be filled in.
static const RegisterGroup g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums,
                             g_gpr_regnums + k_num_gpr_registers)},
    {"Floating Point Registers", "fpu",
     std::vector<RegisterID>(g_fpu_regnums,
                             g_fpu_regnums + k_num_fpu_registers)},
    {"Exception State Registers", "exc",
     std::vector<RegisterID>(g_exc_regnums,
                             g_exc_regnums + k_num_exc_registers)}};

static std::shared_ptr<const RegisterDescription> CreateContextDescription() {
  static const auto description = [] {
    auto cpu = RegisterInfo_x86::CreateDescription(
        ArchSpec("x86_64-apple-darwin"));
    std::vector<RegisterDescriptor> registers;
    // Mach selectors are an explicit projection of the CPU catalog. These
    // contexts expose complete independent values, without the catalog's
    // unexposed subregister aliases.
    auto append = [&](uint32_t selector, const RegisterDescriptor &definition)
        -> RegisterDescriptor & {
      auto reg = definition;
      reg.id = selector;
      reg.numbers[eRegisterKindLLDB] = selector;
      reg.numbers[eRegisterKindProcessPlugin] = LLDB_INVALID_REGNUM;
      reg.value_registers.clear();
      reg.invalidate_registers.clear();
      reg.value_byte_offset = 0;
      reg.zero_extend_parent = LLDB_INVALID_REGNUM;
      registers.push_back(std::move(reg));
      return registers.back();
    };
    for (const char *name : {"rax", "rbx", "rcx", "rdx", "rdi", "rsi",
                             "rbp", "rsp", "r8", "r9", "r10", "r11",
                             "r12", "r13", "r14", "r15", "rip", "rflags",
                             "cs", "fs", "gs"}) {
      auto &reg = append(registers.size(), *cpu->Find(name));
      if (reg.id == gpr_rbp)
        reg.alt_name = "fp";
      else if (reg.id == gpr_rsp)
        reg.alt_name = "sp";
      else if (reg.id == gpr_rip)
        reg.alt_name = "pc";
      else if (reg.id == gpr_rflags)
        reg.alt_name = "flags";
      uint32_t &generic = reg.numbers[eRegisterKindGeneric];
      if (generic >= LLDB_REGNUM_GENERIC_ARG1 &&
          generic <= LLDB_REGNUM_GENERIC_ARG8)
        generic = LLDB_INVALID_REGNUM;
      if (reg.id >= gpr_rflags) {
        reg.numbers[eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
        reg.numbers[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
      }
    }
    // Preserve the Mach spelling, widths and absent external numbers for the
    // x87 control words. Their logical definitions come from the CPU catalog.
    for (auto names : {std::make_pair("fctrl", "fcw"),
                       std::make_pair("fstat", "fsw"),
                       std::make_pair("ftag", "ftw"),
                       std::make_pair("fop", "fop"),
                       std::make_pair("fioff", "ip"),
                       std::make_pair("fiseg", "cs"),
                       std::make_pair("fooff", "dp"),
                       std::make_pair("foseg", "ds"),
                       std::make_pair("mxcsr", "mxcsr"),
                       std::make_pair("mxcsrmask", "mxcsrmask")}) {
      auto &reg = append(registers.size(), *cpu->Find(names.first));
      reg.name = names.second;
      if (reg.id == fpu_ftw)
        reg.byte_size = sizeof(RegisterContextDarwin_x86_64::FPU::ftw);
      else if (reg.id == fpu_cs || reg.id == fpu_ds)
        reg.byte_size = sizeof(RegisterContextDarwin_x86_64::FPU::cs);
      reg.numbers[eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
      reg.numbers[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    }
    for (uint32_t i = 0; i < 8; ++i) {
      auto &reg = append(fpu_stmm0 + i, *cpu->Find("st" + std::to_string(i)));
      reg.name = "stmm" + std::to_string(i);
    }
    for (uint32_t i = 0; i < 16; ++i)
      append(fpu_xmm0 + i, *cpu->Find("xmm" + std::to_string(i)));
    for (auto entry : {std::make_pair(exc_trapno, "trapno"),
                       std::make_pair(exc_err, "err"),
                       std::make_pair(exc_faultvaddr, "faultvaddr")})
      registers.emplace_back(
          entry.first, entry.second, nullptr,
          entry.first == exc_faultvaddr ? 8 : 4, eEncodingUint, eFormatHex,
          RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       static_cast<uint32_t>(entry.first)});
    return std::make_shared<RegisterDescription>(
        llvm::cantFail(RegisterDescription::Create(
            std::move(registers),
            {std::begin(g_reg_sets), std::end(g_reg_sets)},
            {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
  }();
  return description;
}

int RegisterContextDarwin_x86_64::GetSetForNativeRegNum(int reg_num) {
  if (reg_num < fpu_fcw)
    return GPRRegSet;
  else if (reg_num < exc_trapno)
    return FPURegSet;
  else if (reg_num < k_num_registers)
    return EXCRegSet;
  return -1;
}

int RegisterContextDarwin_x86_64::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(GPRRegSet, Read);
}

int RegisterContextDarwin_x86_64::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
  }
  return GetError(FPURegSet, Read);
}

int RegisterContextDarwin_x86_64::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(EXCRegSet, Read);
}

int RegisterContextDarwin_x86_64::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_x86_64::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpu));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_x86_64::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_x86_64::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  default:
    break;
  }
  return -1;
}

int RegisterContextDarwin_x86_64::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  switch (set) {
  case GPRRegSet:
    return WriteGPR();
  case FPURegSet:
    return WriteFPU();
  case EXCRegSet:
    return WriteEXC();
  default:
    break;
  }
  return -1;
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_x86_64::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  int set = RegisterContextDarwin_x86_64::GetSetForNativeRegNum(reg);
  if (set == -1)
    return llvm::createStringError("register value is unavailable");

  if (ReadRegisterSet(set, false) != 0)
    return llvm::createStringError("register value is unavailable");

  switch (reg) {
  case gpr_rax:
  case gpr_rbx:
  case gpr_rcx:
  case gpr_rdx:
  case gpr_rdi:
  case gpr_rsi:
  case gpr_rbp:
  case gpr_rsp:
  case gpr_r8:
  case gpr_r9:
  case gpr_r10:
  case gpr_r11:
  case gpr_r12:
  case gpr_r13:
  case gpr_r14:
  case gpr_r15:
  case gpr_rip:
  case gpr_rflags:
  case gpr_cs:
  case gpr_fs:
  case gpr_gs:
    value = (&gpr.rax)[reg - gpr_rax];
    break;

  case fpu_fcw:
    value = fpu.fcw;
    break;

  case fpu_fsw:
    value = fpu.fsw;
    break;

  case fpu_ftw:
    value = fpu.ftw;
    break;

  case fpu_fop:
    value = fpu.fop;
    break;

  case fpu_ip:
    value = fpu.ip;
    break;

  case fpu_cs:
    value = fpu.cs;
    break;

  case fpu_dp:
    value = fpu.dp;
    break;

  case fpu_ds:
    value = fpu.ds;
    break;

  case fpu_mxcsr:
    value = fpu.mxcsr;
    break;

  case fpu_mxcsrmask:
    value = fpu.mxcsrmask;
    break;

  case fpu_stmm0:
  case fpu_stmm1:
  case fpu_stmm2:
  case fpu_stmm3:
  case fpu_stmm4:
  case fpu_stmm5:
  case fpu_stmm6:
  case fpu_stmm7:
    value.SetBytes(fpu.stmm[reg - fpu_stmm0].bytes, reg_info->byte_size,
                   endian::InlHostByteOrder());
    break;

  case fpu_xmm0:
  case fpu_xmm1:
  case fpu_xmm2:
  case fpu_xmm3:
  case fpu_xmm4:
  case fpu_xmm5:
  case fpu_xmm6:
  case fpu_xmm7:
  case fpu_xmm8:
  case fpu_xmm9:
  case fpu_xmm10:
  case fpu_xmm11:
  case fpu_xmm12:
  case fpu_xmm13:
  case fpu_xmm14:
  case fpu_xmm15:
    value.SetBytes(fpu.xmm[reg - fpu_xmm0].bytes, reg_info->byte_size,
                   endian::InlHostByteOrder());
    break;

  case exc_trapno:
    value = exc.trapno;
    break;

  case exc_err:
    value = exc.err;
    break;

  case exc_faultvaddr:
    value = exc.faultvaddr;
    break;

  default:
    return llvm::createStringError("register value is unavailable");
  }
  return value;
}

llvm::Error
RegisterContextDarwin_x86_64::WriteRegister(RegisterID id,
                                                 const RegisterValue &value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  int set = RegisterContextDarwin_x86_64::GetSetForNativeRegNum(reg);

  if (set == -1)
    return llvm::createStringError("register write is unavailable");

  if (ReadRegisterSet(set, false) != 0)
    return llvm::createStringError("register write is unavailable");

  switch (reg) {
  case gpr_rax:
  case gpr_rbx:
  case gpr_rcx:
  case gpr_rdx:
  case gpr_rdi:
  case gpr_rsi:
  case gpr_rbp:
  case gpr_rsp:
  case gpr_r8:
  case gpr_r9:
  case gpr_r10:
  case gpr_r11:
  case gpr_r12:
  case gpr_r13:
  case gpr_r14:
  case gpr_r15:
  case gpr_rip:
  case gpr_rflags:
  case gpr_cs:
  case gpr_fs:
  case gpr_gs:
    (&gpr.rax)[reg - gpr_rax] = value.GetAsUInt64();
    break;

  case fpu_fcw:
    fpu.fcw = value.GetAsUInt16();
    break;

  case fpu_fsw:
    fpu.fsw = value.GetAsUInt16();
    break;

  case fpu_ftw:
    fpu.ftw = value.GetAsUInt8();
    break;

  case fpu_fop:
    fpu.fop = value.GetAsUInt16();
    break;

  case fpu_ip:
    fpu.ip = value.GetAsUInt32();
    break;

  case fpu_cs:
    fpu.cs = value.GetAsUInt16();
    break;

  case fpu_dp:
    fpu.dp = value.GetAsUInt32();
    break;

  case fpu_ds:
    fpu.ds = value.GetAsUInt16();
    break;

  case fpu_mxcsr:
    fpu.mxcsr = value.GetAsUInt32();
    break;

  case fpu_mxcsrmask:
    fpu.mxcsrmask = value.GetAsUInt32();
    break;

  case fpu_stmm0:
  case fpu_stmm1:
  case fpu_stmm2:
  case fpu_stmm3:
  case fpu_stmm4:
  case fpu_stmm5:
  case fpu_stmm6:
  case fpu_stmm7:
    ::memcpy(fpu.stmm[reg - fpu_stmm0].bytes, value.GetBytes(),
             value.GetByteSize());
    break;

  case fpu_xmm0:
  case fpu_xmm1:
  case fpu_xmm2:
  case fpu_xmm3:
  case fpu_xmm4:
  case fpu_xmm5:
  case fpu_xmm6:
  case fpu_xmm7:
  case fpu_xmm8:
  case fpu_xmm9:
  case fpu_xmm10:
  case fpu_xmm11:
  case fpu_xmm12:
  case fpu_xmm13:
  case fpu_xmm14:
  case fpu_xmm15:
    ::memcpy(fpu.xmm[reg - fpu_xmm0].bytes, value.GetBytes(),
             value.GetByteSize());
    return llvm::createStringError("register write is unavailable");

  case exc_trapno:
    exc.trapno = value.GetAsUInt32();
    break;

  case exc_err:
    exc.err = value.GetAsUInt32();
    break;

  case exc_faultvaddr:
    exc.faultvaddr = value.GetAsUInt64();
    break;

  default:
    return llvm::createStringError("register write is unavailable");
  }
  {
    if (!(WriteRegisterSet(set) == 0))
      return llvm::createStringError("register write failed");
    return llvm::Error::success();
  }
}

bool RegisterContextDarwin_x86_64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == 0 && ReadFPU(false) == 0 && ReadEXC(false) == 0) {
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

bool RegisterContextDarwin_x86_64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpu, src, sizeof(fpu));
    src += sizeof(gpr);

    ::memcpy(&exc, src, sizeof(exc));
    uint32_t success_count = 0;
    if (WriteGPR() == 0)
      ++success_count;
    if (WriteFPU() == 0)
      ++success_count;
    if (WriteEXC() == 0)
      ++success_count;
    return success_count == 3;
  }
  return false;
}

bool RegisterContextDarwin_x86_64::HardwareSingleStep(bool enable) {
  if (ReadGPR(true) != 0)
    return false;

  const uint64_t trace_bit = 0x100ull;
  if (enable) {

    if (gpr.rflags & trace_bit)
      return true; // trace bit is already set, there is nothing to do
    else
      gpr.rflags |= trace_bit;
  } else {
    if (gpr.rflags & trace_bit)
      gpr.rflags &= ~trace_bit;
    else
      return true; // trace bit is clear, there is nothing to do
  }

  return WriteGPR() == 0;
}
