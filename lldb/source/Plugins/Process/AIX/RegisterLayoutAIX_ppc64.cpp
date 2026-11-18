//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterLayoutAIX_ppc64.h"
#include "lldb/Register/Arch/lldb-ppc64be-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64be-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64be.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
using namespace lldb_private;

namespace {
using OSLayout = RegisterLayoutAIX_ppc64;
} // namespace

namespace {

std::shared_ptr<const RegisterDescription>
CreateAIXDescription(const ArchSpec &arch) {
  auto cpu = RegisterDescription_ppc64be::GetShared(
      RegisterDescription_ppc64be::Numbering::AIX);
  if (arch.GetMachine() != llvm::Triple::ppc)
    return cpu;

  // AIX exposes a 32-bit view of the same hardware register banks.
  std::vector<RegisterDescriptor> values(cpu->GetRegisters().begin(),
                                         cpu->GetRegisters().end());

  for (auto &reg : values)
    if (cpu->IsInGroup(reg.id, "gpr"))
      reg.byte_size = 4;

  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(values),
          {cpu->GetGroups().begin(), cpu->GetGroups().end()})));
}

template <typename GP>
std::optional<RegisterLocation>
GetRegisterLocation(const RegisterDescriptor &reg) {
  namespace reg_ids = registers::ppc64be::id;
  const auto id = reg.id;
  const auto stored = [&](RegisterBankID bank, uint64_t offset) {
    return RegisterLocation::StoredAt(bank, offset, reg.byte_size);
  };
  if (id >= reg_ids::gpr_r0 && id <= reg_ids::gpr_r31)
    return stored(0,
                  offsetof(GP, r0) + (id - reg_ids::gpr_r0) * sizeof(GP::r0));
  if (id >= reg_ids::fpr_f0 && id <= reg_ids::fpr_f31)
    return stored(1,
                  offsetof(OSLayout::FPR_PPC64, f0) +
                      (id - reg_ids::fpr_f0) * sizeof(OSLayout::FPR_PPC64::f0));
  if (id >= reg_ids::vmx_vr0 && id <= reg_ids::vmx_vr31)
    return stored(2, offsetof(OSLayout::VMX_PPC64, vr0) +
                         (id - reg_ids::vmx_vr0) *
                             sizeof(OSLayout::VMX_PPC64::vr0));
  if (id >= reg_ids::vsx_vs0 && id <= reg_ids::vsx_vs63)
    return stored(3, offsetof(OSLayout::VSX_PPC64, vs0) +
                         (id - reg_ids::vsx_vs0) *
                             sizeof(OSLayout::VSX_PPC64::vs0));

  switch (id) {
  case reg_ids::gpr_cr:
    return stored(0, offsetof(GP, cr));
  case reg_ids::gpr_msr:
    return stored(0, offsetof(GP, msr));
  case reg_ids::gpr_xer:
    return stored(0, offsetof(GP, xer));
  case reg_ids::gpr_lr:
    return stored(0, offsetof(GP, lr));
  case reg_ids::gpr_ctr:
    return stored(0, offsetof(GP, ctr));
  case reg_ids::gpr_pc:
    return stored(0, offsetof(GP, pc));
  case reg_ids::fpr_fpscr:
    return stored(1, offsetof(OSLayout::FPR_PPC64, fpscr));
  case reg_ids::vmx_vscr:
    return stored(2, offsetof(OSLayout::VMX_PPC64, vscr));
  case reg_ids::vmx_vrsave:
    return stored(2, offsetof(OSLayout::VMX_PPC64, vrsave));
  default:
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutAIX_ppc64::RegisterLayoutAIX_ppc64(const ArchSpec &arch) {
  auto description = CreateAIXDescription(arch);
  const bool is_32_bit = arch.GetMachine() == llvm::Triple::ppc;
  llvm::cantFail(AddBank(0, is_32_bit ? sizeof(GPR_PPC) : sizeof(GPR_PPC64)));
  llvm::cantFail(AddBank(1, sizeof(FPR_PPC64)));
  llvm::cantFail(AddBank(2, sizeof(VMX_PPC64)));
  llvm::cantFail(AddBank(3, sizeof(VSX_PPC64)));

  const auto get_location = [&](const RegisterDescriptor &reg) {
    return is_32_bit ? GetRegisterLocation<GPR_PPC>(reg)
                     : GetRegisterLocation<GPR_PPC64>(reg);
  };
  llvm::cantFail(AddRegisterLocations(*description, get_location));
  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
