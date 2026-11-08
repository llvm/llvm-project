//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_ppc64.h"
#include "lldb/Register/Arch/lldb-ppc64-ehframe-register-enums.h"
#include "RegisterInfo_powerpc_internal.h"
#include <cassert>
#include <cstddef>
#include "lldb/Register/Arch/lldb-ppc64-register-enums.h"
#include "llvm/ADT/STLExtras.h"
#include "lldb/Register/Arch/lldb-ppc64-dwarf-register-enums.h"
#include "lldb/lldb-private-types.h"
#include "llvm/Support/Compiler.h"
#include <iterator>

using namespace lldb_private;

static llvm::ArrayRef<RegisterDescriptor> Values4() {
  constexpr uint32_t word_size = 4;
#include "Register/Arch/RegisterDescriptions_ppc64.inc"
  return values;
}

static llvm::ArrayRef<RegisterDescriptor> Values8() {
  constexpr uint32_t word_size = 8;
#include "Register/Arch/RegisterDescriptions_ppc64.inc"
  return values;
}

std::shared_ptr<const RegisterDescription>
RegisterInfo_ppc64::CreateDescription(const ArchSpec &arch) {
  static const auto description4 = std::make_shared<RegisterDescription>(
      MakePowerPCDescription(Values4(), false));
  static const auto description8 = std::make_shared<RegisterDescription>(
      MakePowerPCDescription(Values8(), false));
  return arch.GetMachine() == llvm::Triple::ppc ? description4 : description8;
}

RegisterDescription
lldb_private::MakePowerPCDescription(llvm::ArrayRef<RegisterDescriptor> values,
                                     bool normalize_identities) {
  std::vector<RegisterDescriptor> registers(values.begin(), values.end());
  std::shared_ptr<const RegisterDescription> master;
  if (normalize_identities)
    master = RegisterInfo_ppc64::CreateDescription(
        ArchSpec("powerpc64-unknown-unknown"));
  std::map<RegisterID, RegisterID> identities;
  for (auto &reg : registers) {
    std::string canonical = reg.name;
    if (canonical.size() > 1 && canonical[0] == 'v' && canonical[1] >= '0' &&
        canonical[1] <= '9')
      canonical.insert(1, "r");
    const auto *definition = master ? master->Find(canonical) : &reg;
    assert(definition && "unknown PowerPC architecture identity");
    identities[reg.id] = definition->id;
    reg.id = definition->id;
  }
  std::vector<RegisterGroup> groups = {{"General Purpose Registers", "gpr", {}},
                                       {"Floating Point Registers", "fpr", {}},
                                       {"Altivec/VMX Registers", "vmx", {}},
                                       {"VSX Registers", "vsx", {}}};
  for (auto &reg : registers) {
    for (auto *refs : {&reg.value_registers, &reg.invalidate_registers})
      for (auto &id : *refs)
        id = identities.at(id);
    llvm::StringRef name(reg.name);
    size_t group = 0;
    if (name.starts_with("f"))
      group = 1;
    else if (name.starts_with("v")) {
      group = name.starts_with("vs") && name != "vscr" ? 3 : 2;
    }
    groups[group].registers.push_back(reg.id);
  }
  llvm::erase_if(groups, [](const auto &g) { return g.registers.empty(); });
  return llvm::cantFail(
      RegisterDescription::Create(std::move(registers), std::move(groups)));
}
