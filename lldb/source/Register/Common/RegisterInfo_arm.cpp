//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_arm.h"
#include "lldb/Register/Arch/lldb-arm-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-arm-register-enums.h"
#include "lldb/Register/Arch/lldb-arm-dwarf-register-enums.h"
#include <iterator>


namespace reg_ids = lldb_private::registers::arm::id;
using namespace lldb_private;
std::shared_ptr<const RegisterDescription>
RegisterInfo_arm::CreateDescription(bool tls) {
#include "Register/Arch/RegisterDescriptions_arm.inc"

  auto make = [](bool tls) {
    const size_t count = tls ? reg_ids::tls_tpidruro + 1 : reg_ids::fpu_q15 + 1;
    std::vector<RegisterDescriptor> regs(std::begin(g_register_infos_arm),
                                         std::begin(g_register_infos_arm) +
                                             count);
    std::vector<RegisterGroup> groups = {
        {"General Purpose Registers", "gpr", {}},
        {"Floating Point Registers", "fpu", {}}};
    if (tls)
      groups.push_back({"Thread Local Storage Registers", "tls", {}});
    for (auto &reg : regs) {
      if (reg.id >= reg_ids::fpu_s0 && reg.id <= reg_ids::fpu_s31)
        reg.value_byte_offset = ((reg.id - reg_ids::fpu_s0) % 4) * 4;
      if (reg.id >= reg_ids::fpu_d0 && reg.id <= reg_ids::fpu_d31)
        reg.value_byte_offset = ((reg.id - reg_ids::fpu_d0) % 2) * 8;
      groups[reg.id <= reg_ids::gpr_cpsr  ? GPRegSet
             : reg.id <= reg_ids::fpu_q15 ? FPRegSet
                                     : TLSRegSet]
          .registers.push_back(reg.id);
    }
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(regs), std::move(groups))));
  };
  static const auto base = make(false);
  static const auto with_tls = make(true);
  return tls ? with_tls : base;
}
