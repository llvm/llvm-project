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
// Keep the legacy metadata's numbering until its consumers migrate.
namespace dwarf_regs {
using namespace lldb_private::registers::ppc32be::dwarf;

enum : uint32_t {
  cr = 68,
  fpscr = LLDB_INVALID_REGNUM,
  msr = LLDB_INVALID_REGNUM,
  vscr = 110,
  xer = 76,
  lr = 65,
  ctr = 66,
  pc = LLDB_INVALID_REGNUM,
  vrsave = 109,
  v0 = 77, v1, v2, v3, v4, v5, v6, v7,
  v8, v9, v10, v11, v12, v13, v14, v15,
  v16, v17, v18, v19, v20, v21, v22, v23,
  v24, v25, v26, v27, v28, v29, v30, v31,
};
} // namespace dwarf_regs
namespace ehframe_regs = dwarf_regs;

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
