//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm32.h"
#include "lldb/Register/Arch/lldb-arm32-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm32.h"

#include <cstddef>

namespace reg_ids = lldb_private::registers::arm32::id;

using namespace lldb_private;

RegisterLayoutFreeBSD_arm32::RegisterLayoutFreeBSD_arm32(const ArchSpec &arch,
                                                         bool tls) {
  auto description = RegisterDescription_arm32::GetShared(tls);
  llvm::cantFail(AddBank(0, sizeof(RegisterLayoutFreeBSD_arm32::GPR)));
  llvm::cantFail(AddBank(1, sizeof(RegisterLayoutFreeBSD_arm32::FPU)));

  if (tls)
    llvm::cantFail(AddBank(2, sizeof(RegisterLayoutFreeBSD_arm32::TLS)));

  const auto get_location =
      [&](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
    RegisterBankID bank = 0;
    uint32_t offset = 0;

    if (reg.id <= reg_ids::gpr_cpsr)
      offset = 4 * reg.id;
    else if (reg.id <= reg_ids::fpu_q15) {
      bank = 1;

      if (reg.id == reg_ids::fpu_fpscr)
        offset = offsetof(RegisterLayoutFreeBSD_arm32::FPU, fpscr);
      else if (reg.id < reg_ids::fpu_fpscr)
        offset = 4 * (reg.id - reg_ids::fpu_s0);
      else if (reg.id < reg_ids::fpu_q0)
        offset = 8 * (reg.id - reg_ids::fpu_d0);
      else
        offset = 16 * (reg.id - reg_ids::fpu_q0);
    } else {
      bank = 2;
    }

    return RegisterLocation::StoredAt(bank, offset, reg.byte_size);
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));

  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
