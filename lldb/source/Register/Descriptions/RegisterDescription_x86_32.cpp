//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_x86_32.h"
#include "lldb/Register/Arch/lldb-x86_32-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_32-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_32-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::x86_32::id;
namespace dwarf_regs = lldb_private::registers::x86_32::dwarf;
namespace ehframe_regs = lldb_private::registers::x86_32::ehframe;

#include "RegisterDescription_x86_32.inc"

} // namespace

RegisterDescription_x86_32::RegisterDescription_x86_32(
    const Config &config,
    llvm::ArrayRef<std::pair<RegisterID, uint32_t>> byte_sizes) {
  const auto &features = config.features;
  std::vector<RegisterDescriptor> catalog(std::begin(GetDefinitions()),
                                          std::end(GetDefinitions()));

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
    return llvm::is_contained(SIMDRegisters, id)
               ? !features.AnySet(Config::eRegsetMaskSIMD)
               : !features.AnySet(Config::eRegsetMaskX87);
  });

  std::vector<RegisterGroup> groups;
  if (!gpr.registers.empty())
    groups.push_back(std::move(gpr));

  if (!fpr.registers.empty())
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
