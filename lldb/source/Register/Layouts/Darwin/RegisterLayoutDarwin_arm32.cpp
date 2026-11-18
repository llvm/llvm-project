//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_arm32.h"
#include "lldb/Register/Arch/lldb-arm32-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm32.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <cstddef>

using namespace lldb;
using namespace lldb_private;

namespace {
using Layout = RegisterLayoutDarwin_arm32;

namespace reg_ids = lldb_private::registers::arm32::id;

constexpr RegisterID GPRRegisters[] = {
    reg_ids::gpr_r0,  reg_ids::gpr_r1, reg_ids::gpr_r2,  reg_ids::gpr_r3,
    reg_ids::gpr_r4,  reg_ids::gpr_r5, reg_ids::gpr_r6,  reg_ids::gpr_r7,
    reg_ids::gpr_r8,  reg_ids::gpr_r9, reg_ids::gpr_r10, reg_ids::gpr_r11,
    reg_ids::gpr_r12, reg_ids::gpr_sp, reg_ids::gpr_lr,  reg_ids::gpr_pc,
    reg_ids::gpr_cpsr};

constexpr RegisterID FPRRegisters[] = {
    reg_ids::fpu_s0,   reg_ids::fpu_s1,  reg_ids::fpu_s2,  reg_ids::fpu_s3,
    reg_ids::fpu_s4,   reg_ids::fpu_s5,  reg_ids::fpu_s6,  reg_ids::fpu_s7,
    reg_ids::fpu_s8,   reg_ids::fpu_s9,  reg_ids::fpu_s10, reg_ids::fpu_s11,
    reg_ids::fpu_s12,  reg_ids::fpu_s13, reg_ids::fpu_s14, reg_ids::fpu_s15,
    reg_ids::fpu_s16,  reg_ids::fpu_s17, reg_ids::fpu_s18, reg_ids::fpu_s19,
    reg_ids::fpu_s20,  reg_ids::fpu_s21, reg_ids::fpu_s22, reg_ids::fpu_s23,
    reg_ids::fpu_s24,  reg_ids::fpu_s25, reg_ids::fpu_s26, reg_ids::fpu_s27,
    reg_ids::fpu_s28,  reg_ids::fpu_s29, reg_ids::fpu_s30, reg_ids::fpu_s31,
    reg_ids::fpu_fpscr};

constexpr RegisterID ExceptionRegisters[] = {
    reg_ids::exc_exception, reg_ids::exc_fsr, reg_ids::exc_far};

std::shared_ptr<const RegisterDescription> CreateDescription() {
  auto cpu = RegisterDescription_arm32::GetShared();
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
    // Mach uses r7 as its frame pointer and does not expose argument roles.
    switch (id) {
    case reg_ids::gpr_r7:
      reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FP;
      break;
    case reg_ids::gpr_sp:
      reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
      break;
    case reg_ids::gpr_lr:
      reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_RA;
      break;
    case reg_ids::gpr_pc:
      reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_PC;
      break;
    case reg_ids::gpr_cpsr:
      reg.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_FLAGS;
      break;
    default:
      reg.numbers[eRegisterKindGeneric] = LLDB_INVALID_REGNUM;
      break;
    }
  }

  for (RegisterID id : FPRRegisters)
    append(id);

  for (auto entry : {std::make_pair(reg_ids::exc_exception, "exception"),
                     std::make_pair(reg_ids::exc_fsr, "fsr"),
                     std::make_pair(reg_ids::exc_far, "far")})
    registers.emplace_back(
        entry.first, entry.second, nullptr, 4, eEncodingUint, eFormatHex,
        RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                      static_cast<uint32_t>(registers.size())});

  for (auto entry : {std::make_pair(reg_ids::dbg_bvr0, "bvr"),
                     std::make_pair(reg_ids::dbg_bcr0, "bcr"),
                     std::make_pair(reg_ids::dbg_wvr0, "wvr"),
                     std::make_pair(reg_ids::dbg_wcr0, "wcr")})
    for (uint32_t i = 0; i < 16; ++i)
      registers.emplace_back(
          entry.first + i, entry.second, nullptr, 4, eEncodingUint, eFormatHex,
          RegisterDescriptor::Numbering{
              LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
              LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM});

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
  if (id <= reg_ids::gpr_pc)
    return RegisterLocation::StoredAt(
        Layout::GPRRegSet,
        offsetof(Layout::GPR, r) + (id - reg_ids::gpr_r0) * sizeof(uint32_t),
        reg.byte_size);
  if (id == reg_ids::gpr_cpsr)
    return RegisterLocation::StoredAt(
        Layout::GPRRegSet, offsetof(Layout::GPR, cpsr), reg.byte_size);
  if (id >= reg_ids::fpu_s0 && id <= reg_ids::fpu_s31)
    return RegisterLocation::StoredAt(Layout::FPURegSet,
                                      offsetof(Layout::FPU, floats.s) +
                                          (id - reg_ids::fpu_s0) *
                                              sizeof(uint32_t),
                                      reg.byte_size);
  if (id == reg_ids::fpu_fpscr)
    return RegisterLocation::StoredAt(
        Layout::FPURegSet, offsetof(Layout::FPU, fpscr), reg.byte_size);
  switch (id) {
  case reg_ids::exc_exception:
    return RegisterLocation::StoredAt(
        Layout::EXCRegSet, offsetof(Layout::EXC, exception), reg.byte_size);
  case reg_ids::exc_fsr:
    return RegisterLocation::StoredAt(
        Layout::EXCRegSet, offsetof(Layout::EXC, fsr), reg.byte_size);
  case reg_ids::exc_far:
    return RegisterLocation::StoredAt(
        Layout::EXCRegSet, offsetof(Layout::EXC, far), reg.byte_size);
  default:
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutDarwin_arm32::RegisterLayoutDarwin_arm32() {
  auto description = CreateDescription();
  llvm::cantFail(AddBank(GPRRegSet, sizeof(GPR)));
  llvm::cantFail(AddBank(FPURegSet, sizeof(FPU)));
  llvm::cantFail(AddBank(EXCRegSet, sizeof(EXC)));
  llvm::cantFail(AddBank(DBGRegSet, sizeof(DBG)));

  llvm::cantFail(AddRegisterLocations(*description, GetRegisterLocation));
  llvm::cantFail(Finalize(std::move(description), eByteOrderLittle));
}

const RegisterLayoutDarwin_arm32 &RegisterLayoutDarwin_arm32::GetShared() {
  static const RegisterLayoutDarwin_arm32 layout;
  return layout;
}
