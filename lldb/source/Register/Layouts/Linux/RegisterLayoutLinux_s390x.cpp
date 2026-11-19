//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_s390x.h"
#include "lldb/Register/Descriptions/RegisterDescription_s390x.h"
#include "lldb/Register/Arch/lldb-s390x-register-enums.h"
#include "lldb/Utility/ArchSpec.h"

#include <cstddef>

namespace reg_ids = lldb_private::registers::s390x::id;

using namespace lldb_private;

namespace {
using OSLayout = RegisterLayoutLinux_s390x;
} // namespace

namespace {

std::shared_ptr<const RegisterDescription> LinuxDescription() {
  static const auto result = [] {
    auto cpu = RegisterDescription_s390x::GetShared();
    std::vector<RegisterDescriptor> regs(cpu->GetRegisters().begin(),
                                         cpu->GetRegisters().end());
    std::vector<RegisterGroup> groups(cpu->GetGroups().begin(),
                                      cpu->GetGroups().end());
    RegisterGroup linux_group{"Linux Operating System Data", "linux", {}};

    for (auto entry : {std::pair<const char *, uint32_t>{"orig_r2", 8},
                       {"last_break", 8},
                       {"system_call", 4}}) {
      RegisterDescriptor reg;
      reg.id = reg_ids::orig_r2 + linux_group.registers.size();
      reg.name = entry.first;
      reg.byte_size = entry.second;
      reg.encoding = lldb::eEncodingUint;
      reg.format = lldb::eFormatHex;
      reg.numbers.fill(LLDB_INVALID_REGNUM);
      reg.numbers[lldb::eRegisterKindLLDB] = reg.id;

      linux_group.registers.push_back(reg.id);
      regs.push_back(std::move(reg));
    }

    groups.push_back(std::move(linux_group));

    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(regs), std::move(groups))));
  }();

  return result;
}

} // namespace

RegisterLayoutLinux_s390x::RegisterLayoutLinux_s390x(const ArchSpec &arch) {
  auto description = LinuxDescription();
  llvm::cantFail(AddBank(0, sizeof(OSLayout::GPR)));
  llvm::cantFail(AddBank(1, sizeof(OSLayout::FPR)));
  llvm::cantFail(AddBank(2, sizeof(OSLayout::LastBreak)));
  llvm::cantFail(AddBank(3, sizeof(OSLayout::SystemCall)));

  const auto get_location =
      [&](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
    const RegisterID id = reg.id;
    if (id >= reg_ids::r0 && id <= reg_ids::r15)
      return RegisterLocation::StoredAt(
          0, offsetof(OSLayout::GPR, gpr) + (id - reg_ids::r0) * 8, 8);
    if (id >= reg_ids::acr0 && id <= reg_ids::acr15)
      return RegisterLocation::StoredAt(
          0, offsetof(OSLayout::GPR, acr) + (id - reg_ids::acr0) * 4, 4);
    if (id >= reg_ids::f0 && id <= reg_ids::f15)
      return RegisterLocation::StoredAt(
          1, offsetof(OSLayout::FPR, fpr) + (id - reg_ids::f0) * 8, 8);

    switch (id) {
    case reg_ids::pswm:
      return RegisterLocation::StoredAt(0, offsetof(OSLayout::GPR, pswm), 8);
    case reg_ids::pswa:
      return RegisterLocation::StoredAt(0, offsetof(OSLayout::GPR, pswa), 8);
    case reg_ids::fpc:
      return RegisterLocation::StoredAt(1, offsetof(OSLayout::FPR, fpc), 4);
    case reg_ids::orig_r2:
      return RegisterLocation::StoredAt(0, offsetof(OSLayout::GPR, orig_r2), 8);
    case reg_ids::last_break:
      return RegisterLocation::StoredAt(2, 0, 8);
    case reg_ids::system_call:
      return RegisterLocation::StoredAt(3, 0, 4);
    default:
      return std::nullopt;
    }
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));

  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
