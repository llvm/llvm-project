//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_arm32.h"
#include "lldb/Register/Arch/lldb-arm32-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-arm32-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-arm32-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::arm32::id;
namespace dwarf_regs = lldb_private::registers::arm32::dwarf;
namespace ehframe_regs = lldb_private::registers::arm32::ehframe;

#include "RegisterDescription_arm32.inc"

} // namespace

RegisterDescription_arm32::RegisterDescription_arm32(bool tls) {
  std::vector<RegisterGroup> groups = {
      MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters),
      MakeRegisterGroup("Floating Point Registers", "fpu", FPRRegisters),
  };

  if (tls)
    groups.push_back(MakeRegisterGroup("Thread Local Storage Registers", "tls",
                                       TLSRegisters));

  auto values = SelectRegisterDefinitions(GetDefinitions(), groups);

  RegisterDescription::operator=(llvm::cantFail(
      RegisterDescription::Create(std::move(values), std::move(groups))));
}

std::shared_ptr<const RegisterDescription>
RegisterDescription_arm32::GetShared(bool tls) {
  static const auto base = std::make_shared<RegisterDescription_arm32>(false);
  static const auto with_tls =
      std::make_shared<RegisterDescription_arm32>(true);
  return tls ? with_tls : base;
}
