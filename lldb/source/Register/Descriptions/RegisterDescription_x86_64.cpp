//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "lldb/Register/Arch/lldb-x86_64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <cassert>

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::x86_64::id;
namespace dwarf_regs = lldb_private::registers::x86_64::dwarf;
namespace ehframe_regs = lldb_private::registers::x86_64::ehframe;

#include "RegisterDescription_x86_64.inc"

} // namespace

RegisterDescription_x86_64::RegisterDescription_x86_64(
    const Config &config,
    llvm::ArrayRef<std::pair<RegisterID, uint32_t>> byte_sizes) {
  const auto &features = config.features;
  std::vector<RegisterDescriptor> catalog(std::begin(GetDefinitions()),
                                          std::end(GetDefinitions()));

  if (!config.segment_bases)
    llvm::erase_if(catalog, [](const auto &reg) {
      return reg.id == reg_ids::fs_base || reg.id == reg_ids::gs_base;
    });

  // LLDB numbers follow the complete catalog's historical presentation order.
  // Logical IDs and relationships do not change when optional bases are absent.
  for (size_t i = 0; i != catalog.size(); ++i)
    catalog[i].numbers[lldb::eRegisterKindLLDB] = i;

  if (config.abi == Config::ABI::Windows) {
    for (auto &reg : catalog) {
      auto &role = reg.numbers[lldb::eRegisterKindGeneric];

      if (role >= LLDB_REGNUM_GENERIC_ARG1 && role <= LLDB_REGNUM_GENERIC_ARG8)
        role = LLDB_INVALID_REGNUM;

      if (reg.id == reg_ids::rcx)
        role = LLDB_REGNUM_GENERIC_ARG1;
      else if (reg.id == reg_ids::rdx)
        role = LLDB_REGNUM_GENERIC_ARG2;
      else if (reg.id == reg_ids::r8)
        role = LLDB_REGNUM_GENERIC_ARG3;
      else if (reg.id == reg_ids::r9)
        role = LLDB_REGNUM_GENERIC_ARG4;
    }
  }

  if (config.abi == Config::ABI::Windows) {
    for (auto &reg : catalog)
      if (llvm::is_contained(SIMDRegisters, reg.id)) {
        reg.encoding = lldb::eEncodingUint;
        reg.format = lldb::eFormatVectorOfUInt64;
      }
  }

  auto gpr =
      MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters);
  llvm::erase_if(gpr.registers, [&](RegisterID id) {
    auto reg =
        llvm::find_if(catalog, [&](const auto &reg) { return reg.id == id; });

    return reg == catalog.end() ||
           (!features.AnySet(Config::eRegsetMaskSegments) &&
            llvm::is_contained(SegmentRegisters, id)) ||
           (!features.AnySet(Config::eRegsetMaskAliases) &&
            !reg->value_registers.empty());
  });

  auto fpr = MakeRegisterGroup("Floating Point Registers", "fpu", FPRRegisters);
  llvm::erase_if(fpr.registers, [&](RegisterID id) {
    return !llvm::is_contained(SIMDRegisters, id) &&
           !features.AnySet(Config::eRegsetMaskX87);
  });

  std::vector<RegisterGroup> groups;
  if (!gpr.registers.empty())
    groups.push_back(std::move(gpr));

  groups.push_back(std::move(fpr));

  if (features.AnySet(Config::eRegsetMaskAVX))
    groups.push_back(
        MakeRegisterGroup("Advanced Vector Extensions", "avx", AVXRegisters));

  if (features.AnySet(Config::eRegsetMaskMPX))
    groups.push_back(
        MakeRegisterGroup("Memory Protection Extensions", "mpx", MPXRegisters));

  if (features.AnySet(Config::eRegsetMaskDebug))
    groups.push_back(
        MakeRegisterGroup("Hardware Debug Registers", "dbr", DebugRegisters));

  auto values = SelectRegisterDefinitions(catalog, groups);
  SetRegisterByteSizes(values, byte_sizes);
  auto description = llvm::cantFail(
      RegisterDescription::Create(std::move(values), std::move(groups)));

  // Selected debug registers remain available by name and group, outside the
  // default user-visible register list.
  RegisterDescription::operator=(
      HideUserRegisters(description, DebugRegisters));
}
