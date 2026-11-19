//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_riscv32.h"
#include "lldb/Register/Arch/lldb-riscv32-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv32-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv32-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include <array>

using namespace lldb_private;

namespace {

namespace reg_ids = lldb_private::registers::riscv32::id;
namespace dwarf_regs = lldb_private::registers::riscv32::dwarf;
namespace ehframe_regs = lldb_private::registers::riscv32::ehframe;

#include "RegisterDescription_riscv32.inc"

} // namespace

RegisterDescription_riscv32::RegisterDescription_riscv32(const Config &config)
    : RegisterDescription_riscv(GetDefinitions(), GPRRegisters, FPRRegisters,
                                VectorRegisters, config.features,
                                config.csr_addresses, 4, config.extensions) {}

std::shared_ptr<const RegisterDescription>
RegisterDescription_riscv32::GetShared(const Config &config) {
  if (!config.csr_addresses.empty())
    return std::make_shared<RegisterDescription_riscv32>(config);

  static const std::array<std::shared_ptr<const RegisterDescription>, 4>
      descriptions = {
          std::make_shared<RegisterDescription_riscv32>(
              Config{eRegsetMaskDefault}),
          std::make_shared<RegisterDescription_riscv32>(Config{eRegsetMaskFP}),
          std::make_shared<RegisterDescription_riscv32>(
              Config{eRegsetMaskVector}),
          std::make_shared<RegisterDescription_riscv32>(
              Config{eRegsetMaskFP | eRegsetMaskVector})};

  return descriptions[config.features.Get() &
                      (eRegsetMaskFP | eRegsetMaskVector)];
}
