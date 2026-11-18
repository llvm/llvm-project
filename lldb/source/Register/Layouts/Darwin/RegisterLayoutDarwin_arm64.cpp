//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <cstddef>

using namespace lldb;
using namespace lldb_private;

namespace {
using Layout = RegisterLayoutDarwin_arm64;

namespace reg_ids = lldb_private::registers::arm64::id;

constexpr RegisterID GPRRegisters[] = {
    reg_ids::gpr_x0,  reg_ids::gpr_x1,  reg_ids::gpr_x2,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,  reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_x8,  reg_ids::gpr_x9,  reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13, reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17, reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21, reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25, reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_fp,  reg_ids::gpr_lr,  reg_ids::gpr_sp,
    reg_ids::gpr_pc,  reg_ids::gpr_cpsr};

constexpr RegisterID FPRRegisters[] = {
    reg_ids::fpu_v0,   reg_ids::fpu_v1,  reg_ids::fpu_v2,  reg_ids::fpu_v3,
    reg_ids::fpu_v4,   reg_ids::fpu_v5,  reg_ids::fpu_v6,  reg_ids::fpu_v7,
    reg_ids::fpu_v8,   reg_ids::fpu_v9,  reg_ids::fpu_v10, reg_ids::fpu_v11,
    reg_ids::fpu_v12,  reg_ids::fpu_v13, reg_ids::fpu_v14, reg_ids::fpu_v15,
    reg_ids::fpu_v16,  reg_ids::fpu_v17, reg_ids::fpu_v18, reg_ids::fpu_v19,
    reg_ids::fpu_v20,  reg_ids::fpu_v21, reg_ids::fpu_v22, reg_ids::fpu_v23,
    reg_ids::fpu_v24,  reg_ids::fpu_v25, reg_ids::fpu_v26, reg_ids::fpu_v27,
    reg_ids::fpu_v28,  reg_ids::fpu_v29, reg_ids::fpu_v30, reg_ids::fpu_v31,
    reg_ids::fpu_fpsr, reg_ids::fpu_fpcr};

constexpr RegisterID ExceptionRegisters[] = {reg_ids::exc_far, reg_ids::exc_esr,
                                             reg_ids::exc_exception};

std::shared_ptr<const RegisterDescription> CreateDescription() {
  auto cpu = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{0});
  std::vector<RegisterDescriptor> regs(cpu->GetRegisters().begin(),
                                       cpu->GetRegisters().end());
  auto extra = [&](std::string name, RegisterID id, uint32_t size) {
    RegisterDescriptor reg;
    reg.id = id;
    reg.name = std::move(name);
    reg.byte_size = size;
    reg.numbers[eRegisterKindLLDB] = id;
    regs.push_back(std::move(reg));
  };

  extra("far", reg_ids::exc_far, 8);
  extra("esr", reg_ids::exc_esr, 4);
  extra("exception", reg_ids::exc_exception, 4);
  for (auto entry : {std::make_pair("bvr", reg_ids::dbg_bvr0),
                     std::make_pair("bcr", reg_ids::dbg_bcr0),
                     std::make_pair("wvr", reg_ids::dbg_wvr0),
                     std::make_pair("wcr", reg_ids::dbg_wcr0)})
    for (uint32_t i = 0; i < 16; ++i)
      extra(entry.first + std::to_string(i), entry.second + i, 8);

  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(regs),
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

  if (id <= reg_ids::gpr_cpsr)
    return stored(Layout::GPRRegSet, id * sizeof(uint64_t));
  if (id >= reg_ids::gpr_w0 && id <= reg_ids::gpr_w28)
    return stored(Layout::GPRRegSet, (id - reg_ids::gpr_w0) * sizeof(uint64_t));
  if (id >= reg_ids::fpu_v0 && id <= reg_ids::fpu_v31)
    return stored(Layout::FPURegSet,
                  (id - reg_ids::fpu_v0) * sizeof(Layout::VReg));
  if (id >= reg_ids::fpu_s0 && id <= reg_ids::fpu_s31)
    return stored(Layout::FPURegSet,
                  (id - reg_ids::fpu_s0) * sizeof(Layout::VReg));
  if (id >= reg_ids::fpu_d0 && id <= reg_ids::fpu_d31)
    return stored(Layout::FPURegSet,
                  (id - reg_ids::fpu_d0) * sizeof(Layout::VReg));
  if (id >= reg_ids::dbg_bvr0 && id < reg_ids::dbg_bvr0 + 16)
    return stored(Layout::DBGRegSet,
                  offsetof(Layout::DBG, bvr) +
                      (id - reg_ids::dbg_bvr0) * sizeof(uint64_t));
  if (id >= reg_ids::dbg_bcr0 && id < reg_ids::dbg_bcr0 + 16)
    return stored(Layout::DBGRegSet,
                  offsetof(Layout::DBG, bcr) +
                      (id - reg_ids::dbg_bcr0) * sizeof(uint64_t));
  if (id >= reg_ids::dbg_wvr0 && id < reg_ids::dbg_wvr0 + 16)
    return stored(Layout::DBGRegSet,
                  offsetof(Layout::DBG, wvr) +
                      (id - reg_ids::dbg_wvr0) * sizeof(uint64_t));
  if (id >= reg_ids::dbg_wcr0 && id < reg_ids::dbg_wcr0 + 16)
    return stored(Layout::DBGRegSet,
                  offsetof(Layout::DBG, wcr) +
                      (id - reg_ids::dbg_wcr0) * sizeof(uint64_t));

  switch (id) {
  case reg_ids::fpu_fpsr:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, fpsr));
  case reg_ids::fpu_fpcr:
    return stored(Layout::FPURegSet, offsetof(Layout::FPU, fpcr));
  case reg_ids::exc_far:
    return stored(Layout::EXCRegSet, offsetof(Layout::EXC, far));
  case reg_ids::exc_esr:
    return stored(Layout::EXCRegSet, offsetof(Layout::EXC, esr));
  case reg_ids::exc_exception:
    return stored(Layout::EXCRegSet, offsetof(Layout::EXC, exception));
  default:
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutDarwin_arm64::RegisterLayoutDarwin_arm64() {
  auto description = CreateDescription();
  llvm::cantFail(AddBank(GPRRegSet, sizeof(GPR)));
  llvm::cantFail(AddBank(FPURegSet, sizeof(FPU)));
  llvm::cantFail(AddBank(EXCRegSet, sizeof(EXC)));
  llvm::cantFail(AddBank(DBGRegSet, sizeof(DBG)));

  llvm::cantFail(AddRegisterLocations(*description, GetRegisterLocation));
  llvm::cantFail(Finalize(std::move(description), eByteOrderLittle));
}

const RegisterLayoutDarwin_arm64 &RegisterLayoutDarwin_arm64::GetShared() {
  static const RegisterLayoutDarwin_arm64 layout;
  return layout;
}
