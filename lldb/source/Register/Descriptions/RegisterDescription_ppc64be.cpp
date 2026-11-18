//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_ppc64be.h"
#include "lldb/Register/Arch/lldb-ppc64be-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64be-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64be-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include "llvm/Support/ErrorHandling.h"

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::ppc64be::id;
namespace dwarf_regs = lldb_private::registers::ppc64be::dwarf;
namespace ehframe_regs = lldb_private::registers::ppc64be::ehframe;

#include "RegisterDescription_ppc64be.inc"

} // namespace

RegisterDescription_ppc64be::RegisterDescription_ppc64be(Numbering numbering) {
  const auto mapping = [](Numbering numbering) -> NumberingMap {
    switch (numbering) {
    case Numbering::AIX:
      return {};
    case Numbering::ELF:
      return ELFNumbering;
    }
    llvm_unreachable("invalid PPC64 register numbering");
  }(numbering);

  RegisterDescription::operator=(
      CreateDescription(GetDefinitions(), GPRRegisters, FPRRegisters,
                        VMXRegisters, VSXRegisters, mapping));
}

std::shared_ptr<const RegisterDescription>
RegisterDescription_ppc64be::GetShared(Numbering numbering) {
  switch (numbering) {
  case Numbering::AIX: {
    static const auto aix =
        std::make_shared<RegisterDescription_ppc64be>(Numbering::AIX);
    return aix;
  }
  case Numbering::ELF: {
    static const auto elf =
        std::make_shared<RegisterDescription_ppc64be>(Numbering::ELF);
    return elf;
  }
  }
  llvm_unreachable("invalid PPC64 register numbering");
}
