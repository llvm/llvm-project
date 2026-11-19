//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_riscv64.h"
#include "lldb/Register/Arch/lldb-riscv64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_riscv64.h"
#include "lldb/Utility/ArchSpec.h"

#include <cstddef>

namespace reg_ids = lldb_private::registers::riscv64::id;

using namespace lldb_private;

RegisterLayoutLinux_riscv64::RegisterLayoutLinux_riscv64(const ArchSpec &arch,
                                                         Flags features) {
  auto description = RegisterDescription_riscv64::GetShared({features});
  llvm::cantFail(AddBank(0, sizeof(RegisterLayoutLinux_riscv64::GPR)));

  if (description->Find(reg_ids::fpr_f0))
    llvm::cantFail(AddBank(1, sizeof(FPR)));

  const auto get_location =
      [&](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
    const RegisterID id = reg.id;
    // The kernel saves pc in slot 0, followed by x1 through x31.
    if (id >= reg_ids::gpr_pc && id <= reg_ids::gpr_x31)
      return RegisterLocation::StoredAt(0, (id - reg_ids::gpr_pc) * 8, 8);
    if (id >= reg_ids::fpr_f0 && id <= reg_ids::fpr_f31)
      return RegisterLocation::StoredAt(1, (id - reg_ids::fpr_f0) * 8, 8);
    if (id == reg_ids::fpr_fcsr)
      return RegisterLocation::StoredAt(1, offsetof(FPR, fcsr), 4);

    return std::nullopt;
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));

  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
