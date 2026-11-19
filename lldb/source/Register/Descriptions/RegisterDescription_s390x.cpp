//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_s390x.h"
#include "lldb/Register/Arch/lldb-s390x-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-s390x-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-s390x-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::s390x::id;
namespace dwarf_regs = lldb_private::registers::s390x::dwarf;
namespace ehframe_regs = lldb_private::registers::s390x::ehframe;

#include "RegisterDescription_s390x.inc"

} // namespace

RegisterDescription_s390x::RegisterDescription_s390x() {
  std::vector<RegisterGroup> groups = {
      MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters),
      MakeRegisterGroup("Floating Point Registers", "fpr", FPRRegisters),
  };

  auto values = SelectRegisterDefinitions(GetDefinitions(), groups);

  RegisterDescription::operator=(llvm::cantFail(
      RegisterDescription::Create(std::move(values), std::move(groups))));
}

std::shared_ptr<const RegisterDescription>
RegisterDescription_s390x::GetShared() {
  static const auto description = std::make_shared<RegisterDescription_s390x>();
  return description;
}
