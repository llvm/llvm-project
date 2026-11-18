//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_loongarch64.h"
#include "lldb/Register/Arch/lldb-loongarch-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::loongarch::id;
namespace dwarf_regs = lldb_private::registers::loongarch::dwarf;
namespace ehframe_regs = lldb_private::registers::loongarch::ehframe;

#include "RegisterDescription_loongarch64.inc"

} // namespace

RegisterDescription_loongarch64::RegisterDescription_loongarch64() {
  std::vector<RegisterGroup> groups = {
      MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters),
      MakeRegisterGroup("Floating Point Registers", "fpr", FPRRegisters),
      MakeRegisterGroup("LSX Vector Registers", "lsx", LSXRegisters),
      MakeRegisterGroup("LASX Vector Registers", "lasx", LASXRegisters),
  };

  auto values = SelectRegisterDefinitions(GetDefinitions(), groups);

  RegisterDescription::operator=(llvm::cantFail(
      RegisterDescription::Create(std::move(values), std::move(groups))));
}

std::shared_ptr<const RegisterDescription>
RegisterDescription_loongarch64::GetShared() {
  static const auto description =
      std::make_shared<RegisterDescription_loongarch64>();
  return description;
}
