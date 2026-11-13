//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_riscv64.h"
#include "lldb/Register/Arch/lldb-riscv-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv-dwarf-register-enums.h"
#include "lldb/lldb-private-types.h"
#include "llvm/ADT/STLExtras.h"
#include <cstddef>
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <iterator>

namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

// Number of entries in the general-purpose register table.
enum { k_num_gpr_registers = reg_ids::gpr_last - reg_ids::gpr_first + 1 };

// RISC-V64 general purpose registers.
static const uint32_t g_gpr_regnums_riscv64[] = {
    reg_ids::gpr_pc,  reg_ids::gpr_ra,  reg_ids::gpr_sp,  reg_ids::gpr_x3,
    reg_ids::gpr_x4,  reg_ids::gpr_x5,  reg_ids::gpr_x6,  reg_ids::gpr_x7,
    reg_ids::gpr_fp,  reg_ids::gpr_x9,  reg_ids::gpr_x10, reg_ids::gpr_x11,
    reg_ids::gpr_x12, reg_ids::gpr_x13, reg_ids::gpr_x14, reg_ids::gpr_x15,
    reg_ids::gpr_x16, reg_ids::gpr_x17, reg_ids::gpr_x18, reg_ids::gpr_x19,
    reg_ids::gpr_x20, reg_ids::gpr_x21, reg_ids::gpr_x22, reg_ids::gpr_x23,
    reg_ids::gpr_x24, reg_ids::gpr_x25, reg_ids::gpr_x26, reg_ids::gpr_x27,
    reg_ids::gpr_x28, reg_ids::gpr_x29, reg_ids::gpr_x30, reg_ids::gpr_x31,
    reg_ids::gpr_x0};

static_assert((sizeof g_gpr_regnums_riscv64 /
               sizeof g_gpr_regnums_riscv64[0]) == k_num_gpr_registers,
              "g_gpr_regnums_riscv64 has wrong number of register infos");

std::shared_ptr<const RegisterDescription>
RegisterInfo_riscv64::CreateDescription(Flags features) {
// These tables describe values, not positions in an operating-system buffer.
#define DECLARE_REGISTER_INFOS_RISCV64_STRUCT
#include "Register/Arch/RegisterDescriptions_riscv64.inc"
#undef DECLARE_REGISTER_INFOS_RISCV64_STRUCT

  auto build = [](bool has_fp) {
    std::vector<RegisterDescriptor> values(
        std::begin(g_register_infos_riscv64_gpr),
        std::end(g_register_infos_riscv64_gpr));
    std::vector<RegisterGroup> groups = {
        {"General Purpose Registers", "gpr", {}}};
    for (uint32_t id : llvm::ArrayRef(g_gpr_regnums_riscv64))
      groups.front().registers.push_back(id);
    if (has_fp) {
      RegisterGroup group{"Floating Point Registers", "fpr", {}};
      for (const auto &reg : g_register_infos_riscv64_fpr) {
        group.registers.push_back(reg.id);
        values.push_back(reg);
      }
      groups.push_back(std::move(group));
    }
    for (auto &reg : values)
      if (reg.id == reg_ids::gpr_x0)
        reg.constant_value.resize(reg.byte_size, 0);
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(values), std::move(groups))));
  };
  // Emulators can return compatibility records by value; their names retain
  // this shared architecture lifetime after a temporary provider is destroyed.
  static const auto gp = build(false);
  static const auto gp_fp = build(true);
  return features.AnySet(eRegsetMaskFP) ? gp_fp : gp;
}
