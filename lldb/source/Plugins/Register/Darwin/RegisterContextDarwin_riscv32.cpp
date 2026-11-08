//===-- RegisterContextDarwin_riscv32.cpp
//------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Log.h"
#include "lldb/Register/RegisterValue.h"
#include "lldb/Register/Common/RegisterInfo_riscv32.h"
#include "lldb/Utility/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include <cstddef>

#include <memory>

#include "RegisterContextDarwin_riscv32.h"

using namespace lldb;
using namespace lldb_private;

enum {
  gpr_x0 = 0,
  gpr_x1,
  gpr_x2,
  gpr_x3,
  gpr_x4,
  gpr_x5,
  gpr_x6,
  gpr_x7,
  gpr_x8,
  gpr_x9,
  gpr_x10,
  gpr_x11,
  gpr_x12,
  gpr_x13,
  gpr_x14,
  gpr_x15,
  gpr_x16,
  gpr_x17,
  gpr_x18,
  gpr_x19,
  gpr_x20,
  gpr_x21,
  gpr_x22,
  gpr_x23,
  gpr_x24,
  gpr_x25,
  gpr_x26,
  gpr_x27,
  gpr_x28,
  gpr_x29,
  gpr_x30,
  gpr_x31,
  gpr_pc,

  fpr_f0,
  fpr_f1,
  fpr_f2,
  fpr_f3,
  fpr_f4,
  fpr_f5,
  fpr_f6,
  fpr_f7,
  fpr_f8,
  fpr_f9,
  fpr_f10,
  fpr_f11,
  fpr_f12,
  fpr_f13,
  fpr_f14,
  fpr_f15,
  fpr_f16,
  fpr_f17,
  fpr_f18,
  fpr_f19,
  fpr_f20,
  fpr_f21,
  fpr_f22,
  fpr_f23,
  fpr_f24,
  fpr_f25,
  fpr_f26,
  fpr_f27,
  fpr_f28,
  fpr_f29,
  fpr_f30,
  fpr_f31,
  fpr_fcsr,

  exc_exception,
  exc_fsr,
  exc_far,

  csr_bank,

  k_num_registers
};

/* clang-format off */
#define REG_CONTEXT_SIZE                                                       \
  (sizeof(RegisterContextDarwin_riscv32::GPR) +                                \
   sizeof(RegisterContextDarwin_riscv32::FPU) +                                \
   sizeof(RegisterContextDarwin_riscv32::EXC) +                                \
   sizeof(RegisterContextDarwin_riscv32::CSR))
/* clang-format on */

static std::shared_ptr<const RegisterDescription> CreateContextDescription();

RegisterContextDarwin_riscv32::RegisterContextDarwin_riscv32(
    Thread &thread, uint32_t concrete_frame_idx)
    : ThreadRegisterContext(thread, concrete_frame_idx), gpr(), fpr(), exc() {
  SetRegisterDescription(CreateContextDescription());

  uint32_t i;
  for (i = 0; i < kNumErrors; i++) {
    gpr_errs[i] = -1;
    fpr_errs[i] = -1;
    exc_errs[i] = -1;
    csr_errs[i] = -1;
  }
}

RegisterContextDarwin_riscv32::~RegisterContextDarwin_riscv32() = default;

void RegisterContextDarwin_riscv32::InvalidateAllRegisters() {
  InvalidateAllRegisterStates();
}

// General purpose registers
static uint32_t g_gpr_regnums[] = {
    gpr_x0,  gpr_x1,  gpr_x2,  gpr_x3,  gpr_x4,  gpr_x5,  gpr_x6,
    gpr_x7,  gpr_x8,  gpr_x9,  gpr_x10, gpr_x11, gpr_x12, gpr_x13,
    gpr_x14, gpr_x15, gpr_x16, gpr_x17, gpr_x18, gpr_x19, gpr_x20,
    gpr_x21, gpr_x22, gpr_x23, gpr_x24, gpr_x25, gpr_x26, gpr_x27,
    gpr_x28, gpr_x29, gpr_x30, gpr_x31, gpr_pc};

// Floating point registers
static uint32_t g_fpr_regnums[] = {
    fpr_f0,  fpr_f1,  fpr_f2,  fpr_f3,  fpr_f4,  fpr_f5,  fpr_f6,
    fpr_f7,  fpr_f8,  fpr_f9,  fpr_f10, fpr_f11, fpr_f12, fpr_f13,
    fpr_f14, fpr_f15, fpr_f16, fpr_f17, fpr_f18, fpr_f19, fpr_f20,
    fpr_f21, fpr_f22, fpr_f23, fpr_f24, fpr_f25, fpr_f26, fpr_f27,
    fpr_f28, fpr_f29, fpr_f30, fpr_f31, fpr_fcsr};

// Exception registers

static uint32_t g_exc_regnums[] = {exc_exception, exc_fsr, exc_far};

// CSR bank registers
static uint32_t g_csr_regnums[] = {csr_bank};

// Number of registers in each register set
const size_t k_num_gpr_registers = std::size(g_gpr_regnums);
const size_t k_num_fpr_registers = std::size(g_fpr_regnums);
const size_t k_num_exc_registers = std::size(g_exc_regnums);
const size_t k_num_csr_registers = std::size(g_csr_regnums);

// Register set definitions. The first definitions at register set index of
// zero is for all registers, followed by other registers sets. The register
// information for the all register set need not be filled in.
static const RegisterGroup g_reg_sets[] = {
    {"General Purpose Registers", "gpr",
     std::vector<RegisterID>(g_gpr_regnums,
                             g_gpr_regnums + k_num_gpr_registers)},
    {"Floating Point Registers", "fpr",
     std::vector<RegisterID>(g_fpr_regnums,
                             g_fpr_regnums + k_num_fpr_registers)},
    {"Exception State Registers", "exc",
     std::vector<RegisterID>(g_exc_regnums,
                             g_exc_regnums + k_num_exc_registers)},
    {"CSR register bank", "csr",
     std::vector<RegisterID>(g_csr_regnums,
                             g_csr_regnums + k_num_csr_registers)}};

static std::shared_ptr<const RegisterDescription> CreateContextDescription() {
  static const auto description = [] {
    auto cpu = RegisterInfo_riscv32::CreateDescription();
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
    for (uint32_t i = 0; i < 32; ++i)
      append(gpr_x0 + i, *cpu->FindByNumber(eRegisterKindDWARF, i));
    append(gpr_pc, *cpu->Find("pc")).alt_name.clear();
    for (uint32_t i = 0; i < 32; ++i) {
      auto &reg = append(fpr_f0 + i,
                         *cpu->FindByNumber(eRegisterKindDWARF, 32 + i));
      // This Mach flavor carries 32-bit integer-encoded FP words.
      reg.byte_size = sizeof(RegisterContextDarwin_riscv32::FPU::f0);
      reg.encoding = eEncodingUint;
    }
    auto &fcsr = append(fpr_fcsr, *cpu->Find("fcsr"));
    fcsr.encoding = eEncodingUint;
    fcsr.alt_name.clear();
    fcsr.numbers[eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
    fcsr.numbers[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    for (auto entry : {std::make_pair(exc_exception, "exception"),
                       std::make_pair(exc_fsr, "fsr"),
                       std::make_pair(exc_far, "far")})
      registers.emplace_back(
          entry.first, entry.second, nullptr, 4, eEncodingUint, eFormatHex,
          RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       LLDB_INVALID_REGNUM,
                                       static_cast<uint32_t>(entry.first)});
    registers.emplace_back(
        csr_bank, "csr", nullptr, sizeof(RegisterContextDarwin_riscv32::CSR),
        eEncodingVector, eFormatVectorOfUInt32,
        RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                     LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                     csr_bank});
    return std::make_shared<RegisterDescription>(
        llvm::cantFail(RegisterDescription::Create(
            std::move(registers),
            {std::begin(g_reg_sets), std::end(g_reg_sets)},
            {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
  }();
  return description;
}

// Register information definitions for 32 bit riscv32.
int RegisterContextDarwin_riscv32::GetSetForNativeRegNum(int reg_num) {
  if (reg_num < fpr_f0)
    return GPRRegSet;
  else if (reg_num < exc_exception)
    return FPURegSet;
  else if (reg_num < csr_bank)
    return EXCRegSet;
  else if (reg_num < k_num_registers)
    return CSRRegSet;
  return -1;
}

void RegisterContextDarwin_riscv32::LogGPR(Log *log, const char *title) {
  if (log) {
    if (title)
      LLDB_LOGF(log, "%s", title);
    for (uint32_t i = 0; i < k_num_gpr_registers; i++) {
      uint32_t reg = gpr_x0 + i;
      LLDB_LOGF(log, "%12s = 0x%4.4x", GetRegisterAtIndex(reg)->name.c_str(),
                (&gpr.x0)[reg]);
    }
  }
}

int RegisterContextDarwin_riscv32::ReadGPR(bool force) {
  int set = GPRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadFPU(bool force) {
  int set = FPURegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadFPU(GetThreadID(), set, fpr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadEXC(bool force) {
  int set = EXCRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::ReadCSR(bool force) {
  int set = CSRRegSet;
  if (force || !RegisterSetIsCached(set)) {
    SetError(set, Read, DoReadCSR(GetThreadID(), set, csr));
  }
  return GetError(set, Read);
}

int RegisterContextDarwin_riscv32::WriteGPR() {
  int set = GPRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteGPR(GetThreadID(), set, gpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteFPU() {
  int set = FPURegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteFPU(GetThreadID(), set, fpr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteEXC() {
  int set = EXCRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteEXC(GetThreadID(), set, exc));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::WriteCSR() {
  int set = CSRRegSet;
  if (!RegisterSetIsCached(set)) {
    SetError(set, Write, -1);
    return -1;
  }
  SetError(set, Write, DoWriteCSR(GetThreadID(), set, csr));
  SetError(set, Read, -1);
  return GetError(set, Write);
}

int RegisterContextDarwin_riscv32::ReadRegisterSet(uint32_t set, bool force) {
  switch (set) {
  case GPRRegSet:
    return ReadGPR(force);
  case FPURegSet:
    return ReadFPU(force);
  case EXCRegSet:
    return ReadEXC(force);
  case CSRRegSet:
    return ReadCSR(force);
  default:
    break;
  }
  return -1;
}

int RegisterContextDarwin_riscv32::WriteRegisterSet(uint32_t set) {
  // Make sure we have a valid context to set.
  if (RegisterSetIsCached(set)) {
    switch (set) {
    case GPRRegSet:
      return WriteGPR();
    case FPURegSet:
      return WriteFPU();
    case EXCRegSet:
      return WriteEXC();
    case CSRRegSet:
      return WriteCSR();
    default:
      break;
    }
  }
  return -1;
}

llvm::Expected<RegisterValue>
RegisterContextDarwin_riscv32::ReadRegister(RegisterID id) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  RegisterValue value;
  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  int set = RegisterContextDarwin_riscv32::GetSetForNativeRegNum(reg);

  if (set == -1)
    return llvm::createStringError("register value is unavailable");

  if (ReadRegisterSet(set, false) != 0)
    return llvm::createStringError("register value is unavailable");

  switch (reg) {
  case gpr_x0:
  case gpr_x1:
  case gpr_x2:
  case gpr_x3:
  case gpr_x4:
  case gpr_x5:
  case gpr_x6:
  case gpr_x7:
  case gpr_x8:
  case gpr_x9:
  case gpr_x10:
  case gpr_x11:
  case gpr_x12:
  case gpr_x13:
  case gpr_x14:
  case gpr_x15:
  case gpr_x16:
  case gpr_x17:
  case gpr_x18:
  case gpr_x19:
  case gpr_x20:
  case gpr_x21:
  case gpr_x22:
  case gpr_x23:
  case gpr_x24:
  case gpr_x25:
  case gpr_x26:
  case gpr_x27:
  case gpr_x28:
  case gpr_x29:
  case gpr_x30:
  case gpr_x31:
  case gpr_pc:
    value = (&gpr.x0)[reg - gpr_x0];
    break;

  case fpr_f0:
  case fpr_f1:
  case fpr_f2:
  case fpr_f3:
  case fpr_f4:
  case fpr_f5:
  case fpr_f6:
  case fpr_f7:
  case fpr_f8:
  case fpr_f9:
  case fpr_f10:
  case fpr_f11:
  case fpr_f12:
  case fpr_f13:
  case fpr_f14:
  case fpr_f15:
  case fpr_f16:
  case fpr_f17:
  case fpr_f18:
  case fpr_f19:
  case fpr_f20:
  case fpr_f21:
  case fpr_f22:
  case fpr_f23:
  case fpr_f24:
  case fpr_f25:
  case fpr_f26:
  case fpr_f27:
  case fpr_f28:
  case fpr_f29:
  case fpr_f30:
  case fpr_f31:
  case fpr_fcsr:
    value = (&fpr.f0)[reg - fpr_f0];
    break;

  case exc_exception:
    value = exc.exception;
    break;

  case exc_fsr:
    value = exc.fsr;
    break;

  case exc_far:
    value = exc.far;
    break;

  case csr_bank:
    // These values don't fit into scalar types,
    // ThreadRegisterContext::ReadRegisterBytes() must be used for these
    // registers
    //::memcpy (reg_value.value.vector.uint8, fpu.stmm[reg - fpu_stmm0].bytes,
    // 10);

    // AArch64 copies NEON registers with
    // value.SetBytes(csr.bytes, reg_info->byte_size,
    //                endian::InlHostByteOrder());
    return llvm::createStringError("register value is unavailable");

  default:
    return llvm::createStringError("register value is unavailable");
  }
  return value;
}

llvm::Error
RegisterContextDarwin_riscv32::WriteRegister(RegisterID id,
                                                  const RegisterValue &value) {
  auto description = GetDescription();
  const auto *reg_info = description ? description->Find(id) : nullptr;
  if (!reg_info)
    return llvm::createStringError("unknown register ID");
  const uint32_t reg = reg_info->numbers[eRegisterKindLLDB];
  int set = GetSetForNativeRegNum(reg);

  if (set == -1)
    return llvm::createStringError("register write is unavailable");

  if (ReadRegisterSet(set, false) != 0)
    return llvm::createStringError("register write is unavailable");

  switch (reg) {
  case gpr_x0:
  case gpr_x1:
  case gpr_x2:
  case gpr_x3:
  case gpr_x4:
  case gpr_x5:
  case gpr_x6:
  case gpr_x7:
  case gpr_x8:
  case gpr_x9:
  case gpr_x10:
  case gpr_x11:
  case gpr_x12:
  case gpr_x13:
  case gpr_x14:
  case gpr_x15:
  case gpr_x16:
  case gpr_x17:
  case gpr_x18:
  case gpr_x19:
  case gpr_x20:
  case gpr_x21:
  case gpr_x22:
  case gpr_x23:
  case gpr_x24:
  case gpr_x25:
  case gpr_x26:
  case gpr_x27:
  case gpr_x28:
  case gpr_x29:
  case gpr_x30:
  case gpr_x31:
  case gpr_pc:
    (&gpr.x0)[reg - gpr_x0] = value.GetAsUInt32();
    break;

  case fpr_f0:
  case fpr_f1:
  case fpr_f2:
  case fpr_f3:
  case fpr_f4:
  case fpr_f5:
  case fpr_f6:
  case fpr_f7:
  case fpr_f8:
  case fpr_f9:
  case fpr_f10:
  case fpr_f11:
  case fpr_f12:
  case fpr_f13:
  case fpr_f14:
  case fpr_f15:
  case fpr_f16:
  case fpr_f17:
  case fpr_f18:
  case fpr_f19:
  case fpr_f20:
  case fpr_f21:
  case fpr_f22:
  case fpr_f23:
  case fpr_f24:
  case fpr_f25:
  case fpr_f26:
  case fpr_f27:
  case fpr_f28:
  case fpr_f29:
  case fpr_f30:
  case fpr_f31:
  case fpr_fcsr:
    (&fpr.f0)[reg - fpr_f0] = value.GetAsUInt32();
    break;

  case exc_exception:
    exc.exception = value.GetAsUInt32();
    break;

  case exc_fsr:
    exc.fsr = value.GetAsUInt32();
    break;

  case exc_far:
    exc.far = value.GetAsUInt32();
    break;

  case csr_bank:
    // These values don't fit into scalar types,
    // ThreadRegisterContext::ReadRegisterBytes() must be used for these
    // registers
    //::memcpy(csr.bytes, value.GetBytes(),
    // value.GetByteSize());
    return llvm::createStringError("register write is unavailable");

  default:
    return llvm::createStringError("register write is unavailable");
  }
  {
    if (!(WriteRegisterSet(set) == 0))
      return llvm::createStringError("register write failed");
    return llvm::Error::success();
  }
}

bool RegisterContextDarwin_riscv32::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  data_sp = std::make_shared<DataBufferHeap>(REG_CONTEXT_SIZE, 0);
  if (ReadGPR(false) == 0 && ReadFPU(false) == 0 && ReadEXC(false) == 0 &&
      ReadCSR(false) == 0) {
    uint8_t *dst = data_sp->GetBytes();
    ::memcpy(dst, &gpr, sizeof(gpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &fpr, sizeof(fpr));
    dst += sizeof(gpr);

    ::memcpy(dst, &exc, sizeof(exc));
    return true;

    ::memcpy(dst, &csr, sizeof(csr));
    return true;
  }
  return false;
}

bool RegisterContextDarwin_riscv32::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE) {
    const uint8_t *src = data_sp->GetBytes();
    ::memcpy(&gpr, src, sizeof(gpr));
    src += sizeof(gpr);

    ::memcpy(&fpr, src, sizeof(fpr));
    src += sizeof(fpr);

    ::memcpy(&exc, src, sizeof(exc));
    src += sizeof(exc);

    ::memcpy(&csr, src, sizeof(csr));
    uint32_t success_count = 0;

    if (WriteGPR() == 0)
      ++success_count;
    if (WriteFPU() == 0)
      ++success_count;
    if (WriteEXC() == 0)
      ++success_count;
    if (WriteCSR() == 0)
      ++success_count;
    return success_count == 3;
  }
  return false;
}
