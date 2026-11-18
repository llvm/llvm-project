//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_riscv32.h"
#include "lldb/Register/Arch/lldb-riscv32-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_riscv32.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <cstddef>

using namespace lldb;
using namespace lldb_private;

namespace {
using Layout = RegisterLayoutDarwin_riscv32;

namespace reg_ids = lldb_private::registers::riscv32::id;
namespace darwin_ids = lldb_private::registers::riscv32::darwin;

constexpr RegisterID GPRRegisters[] = {
    reg_ids::gpr_x0,  reg_ids::gpr_x1,  reg_ids::gpr_x2,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,  reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_x8,  reg_ids::gpr_x9,  reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13, reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17, reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21, reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25, reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_x29, reg_ids::gpr_x30, reg_ids::gpr_x31,
    reg_ids::gpr_pc};

constexpr RegisterID FPRRegisters[] = {
    reg_ids::fpr_f0,  reg_ids::fpr_f1,  reg_ids::fpr_f2,  reg_ids::fpr_f3,
    reg_ids::fpr_f4,  reg_ids::fpr_f5,  reg_ids::fpr_f6,  reg_ids::fpr_f7,
    reg_ids::fpr_f8,  reg_ids::fpr_f9,  reg_ids::fpr_f10, reg_ids::fpr_f11,
    reg_ids::fpr_f12, reg_ids::fpr_f13, reg_ids::fpr_f14, reg_ids::fpr_f15,
    reg_ids::fpr_f16, reg_ids::fpr_f17, reg_ids::fpr_f18, reg_ids::fpr_f19,
    reg_ids::fpr_f20, reg_ids::fpr_f21, reg_ids::fpr_f22, reg_ids::fpr_f23,
    reg_ids::fpr_f24, reg_ids::fpr_f25, reg_ids::fpr_f26, reg_ids::fpr_f27,
    reg_ids::fpr_f28, reg_ids::fpr_f29, reg_ids::fpr_f30, reg_ids::fpr_f31,
    reg_ids::fpr_fcsr};

constexpr RegisterID ExceptionRegisters[] = {
    darwin_ids::exc_exception, darwin_ids::exc_fsr, darwin_ids::exc_far};

constexpr RegisterID CSRRegisters[] = {darwin_ids::csr_bank};

std::shared_ptr<const RegisterDescription> CreateDescription() {
  auto cpu = RegisterDescription_riscv32::GetShared({});
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

  for (RegisterID id : GPRRegisters)
    append(id);

  // Mach thread state carries an explicit x0 slot.
  registers.front().constant_value.clear();
  registers.back().alt_name.clear(); // pc has no alternate name in Mach.

  for (RegisterID id : FPRRegisters) {
    auto &reg = append(id);
    // This Mach flavor carries 32-bit integer-encoded FP words.
    reg.byte_size = sizeof(Layout::FPU::f0);
    reg.encoding = eEncodingUint;
    if (id == reg_ids::fpr_fcsr) {
      reg.alt_name.clear();
      reg.numbers[eRegisterKindEHFrame] = LLDB_INVALID_REGNUM;
      reg.numbers[eRegisterKindDWARF] = LLDB_INVALID_REGNUM;
    }
  }

  for (auto entry : {std::make_pair(darwin_ids::exc_exception, "exception"),
                     std::make_pair(darwin_ids::exc_fsr, "fsr"),
                     std::make_pair(darwin_ids::exc_far, "far")})
    registers.emplace_back(
        entry.first, entry.second, nullptr, 4, eEncodingUint, eFormatHex,
        RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                      LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                      static_cast<uint32_t>(registers.size())});

  registers.emplace_back(
      darwin_ids::csr_bank, "csr", nullptr,
      sizeof(RegisterLayoutDarwin_riscv32::CSR), eEncodingVector,
      eFormatVectorOfUInt32,
      RegisterDescriptor::Numbering{LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
                                    static_cast<uint32_t>(registers.size())});

  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(registers),
          {MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters),
           MakeRegisterGroup("Floating Point Registers", "fpr", FPRRegisters),
           MakeRegisterGroup("Exception State Registers", "exc",
                             ExceptionRegisters),
           MakeRegisterGroup("CSR register bank", "csr", CSRRegisters)},
          {cpu->GetOwnedTypes().begin(), cpu->GetOwnedTypes().end()})));
}

std::optional<RegisterLocation>
GetRegisterLocation(const RegisterDescriptor &reg) {
  const auto id = reg.id;
  if (id == reg_ids::gpr_x0)
    return RegisterLocation::StoredAt(Layout::GPRRegSet,
                                      offsetof(Layout::GPR, x0), reg.byte_size);
  if (id >= reg_ids::gpr_x1 && id <= reg_ids::gpr_x31)
    return RegisterLocation::StoredAt(
        Layout::GPRRegSet,
        offsetof(Layout::GPR, x1) + (id - reg_ids::gpr_x1) * sizeof(uint32_t),
        reg.byte_size);
  if (id == reg_ids::gpr_pc)
    return RegisterLocation::StoredAt(Layout::GPRRegSet,
                                      offsetof(Layout::GPR, pc), reg.byte_size);
  if (id >= reg_ids::fpr_f0 && id <= reg_ids::fpr_fcsr)
    return RegisterLocation::StoredAt(
        Layout::FPURegSet,
        offsetof(Layout::FPU, f0) + (id - reg_ids::fpr_f0) * sizeof(uint32_t),
        reg.byte_size);

  switch (id) {
  case darwin_ids::exc_exception:
    return RegisterLocation::StoredAt(
        Layout::EXCRegSet, offsetof(Layout::EXC, exception), reg.byte_size);
  case darwin_ids::exc_fsr:
    return RegisterLocation::StoredAt(
        Layout::EXCRegSet, offsetof(Layout::EXC, fsr), reg.byte_size);
  case darwin_ids::exc_far:
    return RegisterLocation::StoredAt(
        Layout::EXCRegSet, offsetof(Layout::EXC, far), reg.byte_size);
  default:
    // The aggregate CSR state is available through the checkpoint path only.
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutDarwin_riscv32::RegisterLayoutDarwin_riscv32() {
  auto description = CreateDescription();
  llvm::cantFail(AddBank(GPRRegSet, sizeof(GPR)));
  llvm::cantFail(AddBank(FPURegSet, sizeof(FPU)));
  llvm::cantFail(AddBank(EXCRegSet, sizeof(EXC)));
  llvm::cantFail(AddBank(CSRRegSet, sizeof(CSR)));

  llvm::cantFail(AddRegisterLocations(*description, GetRegisterLocation));
  llvm::cantFail(Finalize(std::move(description), eByteOrderLittle));
}

const RegisterLayoutDarwin_riscv32 &RegisterLayoutDarwin_riscv32::GetShared() {
  static const RegisterLayoutDarwin_riscv32 layout;
  return layout;
}
