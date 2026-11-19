//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_ppc64le.h"
#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::ppc64le::id;
namespace dwarf_regs = lldb_private::registers::ppc64le::dwarf;
namespace ehframe_regs = lldb_private::registers::ppc64le::ehframe;

#include "RegisterDescription_ppc64le.inc"

} // namespace

RegisterDescription_ppc64le::RegisterDescription_ppc64le() {
  RegisterDescription::operator=(CreateDescription(GetDefinitions(),
                                                   GPRRegisters, FPRRegisters,
                                                   VMXRegisters, VSXRegisters));
}

std::shared_ptr<const RegisterDescription>
RegisterDescription_ppc64le::GetShared() {
  static const auto description =
      std::make_shared<RegisterDescription_ppc64le>();
  return description;
}
