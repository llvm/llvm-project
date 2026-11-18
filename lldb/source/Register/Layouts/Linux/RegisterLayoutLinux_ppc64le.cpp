//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_ppc64le.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64le.h"
#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace dwarf_regs = lldb_private::registers::ppc64le::dwarf;
namespace ehframe_regs = lldb_private::registers::ppc64le::ehframe;
namespace reg_ids = lldb_private::registers::ppc64le::id;

using namespace lldb_private;

namespace {
using OSLayout = RegisterLayoutLinux_ppc64le;
} // namespace

namespace {

std::shared_ptr<const RegisterDescription> LinuxDescription() {
  static const auto description = [] {
    auto cpu = RegisterDescription_ppc64le::GetShared();
    std::vector<RegisterDescriptor> registers(cpu->GetRegisters().begin(),
                                              cpu->GetRegisters().end());
    std::vector<RegisterGroup> groups(cpu->GetGroups().begin(),
                                      cpu->GetGroups().end());
    RegisterID next = 0;

    for (const auto &reg : registers)
      next = std::max(next, reg.id + 1);

    {
      RegisterDescriptor extra;
      extra.id = next++;
      extra.name = "origr3";
      extra.byte_size = 8;
      extra.numbers[lldb::eRegisterKindLLDB] = reg_ids::gpr_origr3;
      extra.numbers[lldb::eRegisterKindDWARF] = dwarf_regs::origr3;
      extra.numbers[lldb::eRegisterKindEHFrame] = ehframe_regs::origr3;
      registers.push_back(std::move(extra));
    }

    {
      RegisterDescriptor extra;
      extra.id = next++;
      extra.name = "softe";
      extra.byte_size = 8;
      extra.numbers[lldb::eRegisterKindLLDB] = reg_ids::gpr_softe;
      extra.numbers[lldb::eRegisterKindDWARF] = dwarf_regs::softe;
      extra.numbers[lldb::eRegisterKindEHFrame] = ehframe_regs::softe;
      registers.push_back(std::move(extra));
    }

    {
      RegisterDescriptor extra;
      extra.id = next++;
      extra.name = "trap";
      extra.byte_size = 8;
      extra.numbers[lldb::eRegisterKindLLDB] = reg_ids::gpr_trap;
      extra.numbers[lldb::eRegisterKindDWARF] = dwarf_regs::trap;
      extra.numbers[lldb::eRegisterKindEHFrame] = ehframe_regs::trap;
      registers.push_back(std::move(extra));
    }

    llvm::sort(registers, [](const auto &a, const auto &b) {
      return a.numbers[lldb::eRegisterKindLLDB] <
             b.numbers[lldb::eRegisterKindLLDB];
    });
    groups[0].registers.clear();

    for (const auto &reg : registers)
      if (reg.numbers[lldb::eRegisterKindLLDB] <= reg_ids::k_last_gpr)
        groups[0].registers.push_back(reg.id);

    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(registers), std::move(groups))));
  }();

  return description;
}

} // namespace

RegisterLayoutLinux_ppc64le::RegisterLayoutLinux_ppc64le(const ArchSpec &arch) {
  auto description = LinuxDescription();
  llvm::cantFail(AddBank(0, sizeof(OSLayout::GPR)));
  llvm::cantFail(AddBank(1, sizeof(OSLayout::FPR)));
  llvm::cantFail(AddBank(2, sizeof(OSLayout::VMX)));
  llvm::cantFail(AddBank(3, sizeof(OSLayout::VSX)));

  const auto get_location =
      [](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
    const auto id = reg.id;
    const auto stored = [&](RegisterBankID bank, uint64_t offset) {
      return RegisterLocation::StoredAt(bank, offset, reg.byte_size);
    };
    if (id >= reg_ids::gpr_r0 && id <= reg_ids::gpr_r31)
      return stored(0, offsetof(OSLayout::GPR, r0) +
                           (id - reg_ids::gpr_r0) * sizeof(OSLayout::GPR::r0));
    if (id >= reg_ids::fpr_f0 && id <= reg_ids::fpr_f31)
      return stored(1, offsetof(OSLayout::FPR, f0) +
                           (id - reg_ids::fpr_f0) * sizeof(OSLayout::FPR::f0));
    if (id >= reg_ids::vmx_vr0 && id <= reg_ids::vmx_vr31)
      return stored(2,
                    offsetof(OSLayout::VMX, vr0) +
                        (id - reg_ids::vmx_vr0) * sizeof(OSLayout::VMX::vr0));
    if (id >= reg_ids::vsx_vs0 && id <= reg_ids::vsx_vs63) {
      const auto i = id - reg_ids::vsx_vs0;
      if (i >= 32)
        return RegisterLocation::StoredAt(2, (i - 32) * 16, 16);

      RegisterLocation location;
      location.kind = RegisterLocation::Stored;
      location.pieces = {{3, i * 8, 8, 0}, {1, i * 8, 8, 8}};
      return location;
    }

    // Linux adds these registers with IDs outside the architectural catalog.
    if (reg.name == "origr3")
      return stored(0, offsetof(OSLayout::GPR, origr3));
    if (reg.name == "softe")
      return stored(0, offsetof(OSLayout::GPR, softe));
    if (reg.name == "trap")
      return stored(0, offsetof(OSLayout::GPR, trap));

    switch (id) {
    case reg_ids::gpr_pc:
      return stored(0, offsetof(OSLayout::GPR, pc));
    case reg_ids::gpr_msr:
      return stored(0, offsetof(OSLayout::GPR, msr));
    case reg_ids::gpr_ctr:
      return stored(0, offsetof(OSLayout::GPR, ctr));
    case reg_ids::gpr_lr:
      return stored(0, offsetof(OSLayout::GPR, lr));
    case reg_ids::gpr_xer:
      return stored(0, offsetof(OSLayout::GPR, xer));
    case reg_ids::gpr_cr:
      return stored(0, offsetof(OSLayout::GPR, cr));
    case reg_ids::fpr_fpscr:
      return stored(1, offsetof(OSLayout::FPR, fpscr));
    case reg_ids::vmx_vscr:
      return stored(2, offsetof(OSLayout::VMX, vscr));
    case reg_ids::vmx_vrsave:
      return stored(2, offsetof(OSLayout::VMX, vrsave));
    default:
      return std::nullopt;
    }
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));
  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
