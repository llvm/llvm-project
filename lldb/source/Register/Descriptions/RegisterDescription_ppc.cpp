//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_ppc.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <cassert>

using namespace lldb_private;

RegisterDescription RegisterDescription_ppc::CreateDescription(
    llvm::ArrayRef<RegisterDescriptor> definitions,
    llvm::ArrayRef<RegisterID> gpr, llvm::ArrayRef<RegisterID> fpr,
    llvm::ArrayRef<RegisterID> vmx, llvm::ArrayRef<RegisterID> vsx,
    NumberingMap numbering) {
  std::vector<RegisterGroup> groups = {
      MakeRegisterGroup("General Purpose Registers", "gpr", gpr),
      MakeRegisterGroup("Floating Point Registers", "fpr", fpr),
      MakeRegisterGroup("Altivec/VMX Registers", "vmx", vmx),
  };

  if (!vsx.empty())
    groups.push_back(MakeRegisterGroup("VSX Registers", "vsx", vsx));

  auto values = SelectRegisterDefinitions(definitions, groups);

  if (!numbering.empty()) {
    for (auto &reg : values) {
      auto mapping = llvm::find_if(
          numbering, [&](const auto &entry) { return entry.first == reg.id; });
      assert(mapping != numbering.end());
      reg.numbers = mapping->second;
    }
  }

  return llvm::cantFail(
      RegisterDescription::Create(std::move(values), std::move(groups)));
}
