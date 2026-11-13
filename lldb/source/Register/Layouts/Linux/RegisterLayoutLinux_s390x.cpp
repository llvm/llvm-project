//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_s390x.h"
#include "lldb/Register/Arch/lldb-s390x-register-enums.h"
#include "lldb/Utility/ArchSpec.h"
#include <cstddef>

namespace reg_ids = lldb_private::registers::s390x::id;
using namespace lldb_private;
using namespace lldb_private::linux_s390x;
namespace {
std::shared_ptr<const RegisterDescription> LinuxDescription() {
  static const auto result = [] {
    auto cpu = RegisterInfo_s390x::CreateDescription();
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
RegisterLayout Layout(const ArchSpec &arch) {
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(GPR)));
  llvm::cantFail(layout.AddBank(1, sizeof(FPR)));
  llvm::cantFail(layout.AddBank(2, sizeof(LastBreak)));
  llvm::cantFail(layout.AddBank(3, sizeof(SystemCall)));
  auto add = [&](RegisterID id, RegisterBankID bank, uint32_t offset,
                 uint32_t size) {
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, size});

    llvm::cantFail(layout.Add(id, std::move(location)));
  };
  for (uint32_t i = 0; i < 16; ++i) {
    add(reg_ids::r0 + i, 0, offsetof(GPR, gpr) + 8 * i, 8);
    add(reg_ids::acr0 + i, 0, offsetof(GPR, acr) + 4 * i, 4);
    add(reg_ids::f0 + i, 1, offsetof(FPR, fpr) + 8 * i, 8);
  }
  add(reg_ids::pswm, 0, offsetof(GPR, pswm), 8);
  add(reg_ids::pswa, 0, offsetof(GPR, pswa), 8);
  add(reg_ids::fpc, 1, offsetof(FPR, fpc), 4);
  add(reg_ids::orig_r2, 0, offsetof(GPR, orig_r2), 8);
  add(reg_ids::last_break, 2, 0, 8);
  add(reg_ids::system_call, 3, 0, 4);
  return layout;
}
} // namespace
RegisterLayout
lldb_private::CreateLinuxS390XRegisterLayout(const ArchSpec &arch) {
  return llvm::cantFail(
      RegisterLayout::Create(LinuxDescription(), Layout(arch)));
}
