//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_loongarch64.h"
#include "lldb/Register/Arch/lldb-loongarch-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_loongarch64.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <iterator>

namespace dwarf_regs = lldb_private::registers::loongarch::dwarf;
namespace ehframe_regs = lldb_private::registers::loongarch::ehframe;
namespace reg_ids = lldb_private::registers::loongarch::id;

using namespace lldb_private;

static std::shared_ptr<const RegisterDescription> LinuxDescription() {
  static const auto description = [] {
    auto arch = RegisterDescription_loongarch64::GetShared();
    std::vector<RegisterDescriptor> registers(arch->GetRegisters().begin(),
                                              arch->GetRegisters().end());
    std::vector<RegisterGroup> groups(arch->GetGroups().begin(),
                                      arch->GetGroups().end());

    // Linux syscall state and padding slots are extensions of the CPU catalog.
    struct Extra {
      RegisterID id;
      const char *name;
      uint32_t dwarf;
      uint32_t ehframe;
    };

    const Extra extras[] = {
        {reg_ids::gpr_orig_a0, "orig_a0", dwarf_regs::gpr_orig_a0,
         ehframe_regs::gpr_orig_a0},
        {reg_ids::gpr_reserved0, "reserved0", dwarf_regs::gpr_reserved0,
         ehframe_regs::gpr_reserved0},
        {reg_ids::gpr_reserved1, "reserved1", dwarf_regs::gpr_reserved1,
         ehframe_regs::gpr_reserved1},
        {reg_ids::gpr_reserved2, "reserved2", dwarf_regs::gpr_reserved2,
         ehframe_regs::gpr_reserved2},
        {reg_ids::gpr_reserved3, "reserved3", dwarf_regs::gpr_reserved3,
         ehframe_regs::gpr_reserved3},
        {reg_ids::gpr_reserved4, "reserved4", dwarf_regs::gpr_reserved4,
         ehframe_regs::gpr_reserved4},
        {reg_ids::gpr_reserved5, "reserved5", dwarf_regs::gpr_reserved5,
         ehframe_regs::gpr_reserved5},
        {reg_ids::gpr_reserved6, "reserved6", dwarf_regs::gpr_reserved6,
         ehframe_regs::gpr_reserved6},
        {reg_ids::gpr_reserved7, "reserved7", dwarf_regs::gpr_reserved7,
         ehframe_regs::gpr_reserved7},
        {reg_ids::gpr_reserved8, "reserved8", dwarf_regs::gpr_reserved8,
         ehframe_regs::gpr_reserved8},
        {reg_ids::gpr_reserved9, "reserved9", dwarf_regs::gpr_reserved9,
         ehframe_regs::gpr_reserved9}};

    for (const auto &extra : extras) {
      RegisterDescriptor reg;
      reg.id = extra.id;
      reg.name = extra.name;
      reg.byte_size = 8;
      reg.numbers[lldb::eRegisterKindLLDB] = extra.id;
      reg.numbers[lldb::eRegisterKindDWARF] = extra.dwarf;
      reg.numbers[lldb::eRegisterKindEHFrame] = extra.ehframe;

      registers.push_back(std::move(reg));
      groups[0].registers.push_back(extra.id);
    }

    llvm::sort(registers,
               [](const auto &a, const auto &b) { return a.id < b.id; });
    llvm::sort(groups[0].registers);

    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(registers), std::move(groups))));
  }();

  return description;
}

RegisterLayoutLinux_loongarch64::RegisterLayoutLinux_loongarch64(
    const ArchSpec &arch) {
  auto description = LinuxDescription();
  const uint32_t sizes[] = {360, 272, 512, 1024};

  for (uint32_t bank = 0; bank != 4; ++bank)
    llvm::cantFail(AddBank(bank, sizes[bank]));

  const auto get_location =
      [&](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
    const RegisterID id = reg.id;
    if (id > reg_ids::gpr_r0 && id <= reg_ids::gpr_reserved9)
      return RegisterLocation::StoredAt(0, (id - reg_ids::gpr_r0) * 8, 8);
    if (id >= reg_ids::fpr_f0 && id <= reg_ids::fpr_f31)
      return RegisterLocation::StoredAt(1, (id - reg_ids::fpr_f0) * 8, 8);
    if (id >= reg_ids::fpr_fcc0 && id <= reg_ids::fpr_fcc7)
      return RegisterLocation::StoredAt(1, 256 + id - reg_ids::fpr_fcc0, 1);
    if (id == reg_ids::fpr_fcsr)
      return RegisterLocation::StoredAt(1, 264, 4);
    if (id >= reg_ids::lsx_vr0 && id <= reg_ids::lsx_vr31)
      return RegisterLocation::StoredAt(2, (id - reg_ids::lsx_vr0) * 16, 16);
    if (id >= reg_ids::lasx_xr0 && id <= reg_ids::lasx_xr31)
      return RegisterLocation::StoredAt(3, (id - reg_ids::lasx_xr0) * 32, 32);

    return std::nullopt;
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));

  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
