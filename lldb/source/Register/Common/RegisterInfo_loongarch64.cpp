//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Register/Arch/lldb-loongarch-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-dwarf-register-enums.h"
#include "lldb/lldb-private-types.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>
#include <iterator>

namespace reg_ids = lldb_private::registers::loongarch::id;
using namespace lldb_private;

std::shared_ptr<const RegisterDescription>
RegisterInfo_loongarch64::CreateDescription() {
#define DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT
#include "Register/Arch/RegisterDescriptions_loongarch64.inc"
#undef DECLARE_REGISTER_INFOS_LOONGARCH64_STRUCT

  static const auto description = [] {
    std::vector<RegisterDescriptor> registers(
        std::begin(g_register_infos_loongarch64),
        std::end(g_register_infos_loongarch64));
    std::vector<RegisterGroup> groups = {
        {"General Purpose Registers", "gpr", {}},
        {"Floating Point Registers", "fpr", {}},
        {"LSX Vector Registers", "lsx", {}},
        {"LASX Vector Registers", "lasx", {}}};
    for (auto &reg : registers) {
      if (reg.id == reg_ids::gpr_r0)
        reg.constant_value.resize(reg.byte_size, 0);
      size_t group = reg.id <= reg_ids::gpr_last   ? GPRegSet
                     : reg.id <= reg_ids::fpr_last ? FPRegSet
                     : reg.id <= reg_ids::lsx_last ? LSXRegSet
                                                    : LASXRegSet;
      groups[group].registers.push_back(reg.id);
    }
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(registers), std::move(groups))));
  }();
  return description;
}
