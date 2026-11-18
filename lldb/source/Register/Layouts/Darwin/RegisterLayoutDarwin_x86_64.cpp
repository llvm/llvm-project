//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_x86_64.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <cstddef>

using namespace lldb;
using namespace lldb_private;

namespace {
using Layout = RegisterLayoutDarwin_x86_64;

namespace reg_ids = lldb_private::registers::x86_64::id;
namespace darwin_ids = lldb_private::registers::x86_64::darwin;

constexpr RegisterID GPRRegisters[] = {
    reg_ids::rax, reg_ids::rbx, reg_ids::rcx,    reg_ids::rdx, reg_ids::rdi,
    reg_ids::rsi, reg_ids::rbp, reg_ids::rsp,    reg_ids::r8,  reg_ids::r9,
    reg_ids::r10, reg_ids::r11, reg_ids::r12,    reg_ids::r13, reg_ids::r14,
    reg_ids::r15, reg_ids::rip, reg_ids::rflags, reg_ids::cs,  reg_ids::fs,
    reg_ids::gs};

constexpr RegisterID FPRRegisters[] = {
    reg_ids::fctrl, reg_ids::fstat,     reg_ids::ftag,  reg_ids::fop,
    reg_ids::fioff, reg_ids::fiseg,     reg_ids::fooff, reg_ids::foseg,
    reg_ids::mxcsr, reg_ids::mxcsrmask, reg_ids::st0,   reg_ids::st1,
    reg_ids::st2,   reg_ids::st3,       reg_ids::st4,   reg_ids::st5,
    reg_ids::st6,   reg_ids::st7,       reg_ids::xmm0,  reg_ids::xmm1,
    reg_ids::xmm2,  reg_ids::xmm3,      reg_ids::xmm4,  reg_ids::xmm5,
    reg_ids::xmm6,  reg_ids::xmm7,      reg_ids::xmm8,  reg_ids::xmm9,
    reg_ids::xmm10, reg_ids::xmm11,     reg_ids::xmm12, reg_ids::xmm13,
    reg_ids::xmm14, reg_ids::xmm15};

constexpr RegisterID ExceptionRegisters[] = {
    darwin_ids::exc_trapno, darwin_ids::exc_err, darwin_ids::exc_faultvaddr};

std::shared_ptr<const RegisterDescription> CreateDescription() {
  auto cpu = std::make_shared<RegisterDescription_x86_64>(
      RegisterDescription_x86_64::Config{});
  std::vector<RegisterDescriptor> registers;

  // Keep the CPU IDs while assigning the register numbers used by Mach.
  auto append = [&](RegisterID id) -> RegisterDescriptor & {
    auto reg = *cpu->Find(id);
    reg.numbers[eRegisterKindLLDB] = registers.size();
    reg.numbers[eRegisterKindProcessPlugin] = LLDB_INVALID_REGNUM;
    reg.value_registers.clear();
    reg.invalidate_registers.clear();
    reg.value_byte_offset = 0;
    reg.zero_extend_parent = LLDB_INVALID_REGNUM;
    registers.push_back(std::move(reg));
    return registers.back();
  };

  for (RegisterID id : GPRRegisters) {
    auto &reg = append(id);
    if (id == reg_ids::rbp)
      reg.alt_name = "fp";
    else if (id == reg_ids::rsp)
      reg.alt_name = "sp";
    else if (id == reg_ids::rip)
      reg.alt_name = "pc";
    else if (id == reg_ids::rflags)
      reg.alt_name = "flags";

    auto &generic = reg.numbers[eRegisterKindGeneric];
    if (generic >= LLDB_REGNUM_GENERIC_ARG1 &&
        generic <= LLDB_REGNUM_GENERIC_ARG8)
      generic = LLDB_INVALID_REGNUM;

    if (id >= reg_ids::rflags) {
      reg.numbers[eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
      reg.numbers[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    }
  }

  // Mach names these saved fields differently from the CPU catalog.
  for (RegisterID id : FPRRegisters) {
    auto &reg = append(id);
    switch (id) {
    case reg_ids::fctrl:
      reg.name = "fcw";
      break;
    case reg_ids::fstat:
      reg.name = "fsw";
      break;
    case reg_ids::ftag:
      reg.name = "ftw";
      reg.byte_size = sizeof(Layout::FPU::ftw);
      break;
    case reg_ids::fioff:
      reg.name = "ip";
      break;
    case reg_ids::fiseg:
      reg.name = "cs";
      reg.byte_size = sizeof(Layout::FPU::cs);
      break;
    case reg_ids::fooff:
      reg.name = "dp";
      break;
    case reg_ids::foseg:
      reg.name = "ds";
      reg.byte_size = sizeof(Layout::FPU::ds);
      break;
    default:
      if (id >= reg_ids::st0 && id <= reg_ids::st7)
        reg.name = "stmm" + std::to_string(id - reg_ids::st0);
      break;
    }
    if (id < reg_ids::st0) {
      reg.numbers[eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
      reg.numbers[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    }
  }

  for (auto entry : {std::make_pair(darwin_ids::exc_trapno, "trapno"),
                     std::make_pair(darwin_ids::exc_err, "err"),
                     std::make_pair(darwin_ids::exc_faultvaddr, "faultvaddr")})
    registers.emplace_back(
        entry.first, entry.second, nullptr,
        entry.first == darwin_ids::exc_faultvaddr ? 8 : 4, eEncodingUint,
        eFormatHex,
        RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                      static_cast<uint32_t>(registers.size())});

  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(registers),
          {MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters),
           MakeRegisterGroup("Floating Point Registers", "fpu", FPRRegisters),
           MakeRegisterGroup("Exception State Registers", "exc",
                             ExceptionRegisters)},
          {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
}

std::optional<RegisterLocation>
GetRegisterLocation(const RegisterDescriptor &reg) {
  const auto id = reg.id;
  const auto stored = [&](RegisterBankID bank, uint64_t offset) {
    return RegisterLocation::StoredAt(bank, offset, reg.byte_size);
  };

  if (id >= reg_ids::rax && id <= reg_ids::gs)
    return stored(Layout::GPRRegSet,
                  offsetof(Layout::GPR, rax) +
                      (id - reg_ids::rax) * sizeof(uint64_t));
  if (id >= reg_ids::st0 && id <= reg_ids::st7)
    return stored(Layout::FPURegSet,
                  offsetof(Layout::FPU, stmm) +
                      (id - reg_ids::st0) * sizeof(Layout::MMSReg));
  if (id >= reg_ids::xmm0 && id <= reg_ids::xmm15)
    return stored(Layout::FPURegSet,
                  offsetof(Layout::FPU, xmm) +
                      (id - reg_ids::xmm0) * sizeof(Layout::XMMReg));
  switch (id) {
  case reg_ids::fctrl:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, fcw));
  case reg_ids::fstat:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, fsw));
  case reg_ids::ftag:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, ftw));
  case reg_ids::fop:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, fop));
  case reg_ids::fioff:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, ip));
  case reg_ids::fiseg:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, cs));
  case reg_ids::fooff:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, dp));
  case reg_ids::foseg:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, ds));
  case reg_ids::mxcsr:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, mxcsr));
  case reg_ids::mxcsrmask:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, mxcsrmask));
  case darwin_ids::exc_trapno:
    return stored(Layout::EXCRegSet, offsetof(Layout::EXC, trapno));
  case darwin_ids::exc_err:
    return stored(Layout::EXCRegSet, offsetof(Layout::EXC, err));
  case darwin_ids::exc_faultvaddr:
    return stored(Layout::EXCRegSet, offsetof(Layout::EXC, faultvaddr));
  default:
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutDarwin_x86_64::RegisterLayoutDarwin_x86_64() {
  auto description = CreateDescription();
  llvm::cantFail(AddBank(GPRRegSet, sizeof(GPR)));
  llvm::cantFail(AddBank(FPURegSet, sizeof(FPU)));
  llvm::cantFail(AddBank(EXCRegSet, sizeof(EXC)));

  llvm::cantFail(AddRegisterLocations(*description, GetRegisterLocation));
  llvm::cantFail(Finalize(std::move(description), eByteOrderLittle));
}

const RegisterLayoutDarwin_x86_64 &RegisterLayoutDarwin_x86_64::GetShared() {
  static const RegisterLayoutDarwin_x86_64 layout;
  return layout;
}

bool RegisterLayoutDarwin_x86_64::IsXMM(RegisterID id) {
  return id >= reg_ids::xmm0 && id <= reg_ids::xmm15;
}
