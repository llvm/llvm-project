//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_ppc32be.h"
#include "lldb/Register/Arch/lldb-ppc32be-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc32be-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc32be-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::ppc32be::id;
namespace dwarf_regs = lldb_private::registers::ppc32be::dwarf;
namespace ehframe_regs = lldb_private::registers::ppc32be::ehframe;

#include "RegisterDescription_ppc32be.inc"

} // namespace

RegisterDescription_ppc32be::RegisterDescription_ppc32be() {
  RegisterDescription::operator=(CreateDescription(
      GetDefinitions(), GPRRegisters, FPRRegisters, VMXRegisters));
}

std::shared_ptr<const RegisterDescription>
RegisterDescription_ppc32be::GetShared() {
  static const auto description =
      std::make_shared<RegisterDescription_ppc32be>();
  return description;
}
