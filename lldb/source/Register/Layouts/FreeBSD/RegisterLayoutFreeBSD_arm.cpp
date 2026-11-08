//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm.h"
#include "lldb/Register/Arch/lldb-arm-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"
#include <cstddef>

namespace reg_ids = lldb_private::registers::arm::id;
using namespace lldb_private;
RegisterLayout
lldb_private::CreateFreeBSDArmRegisterLayout(const ArchSpec &arch, bool tls) {
  auto description = RegisterInfo_arm::CreateDescription(tls);
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(freebsd_arm::GPR)));
  llvm::cantFail(layout.AddBank(1, sizeof(freebsd_arm::FPU)));
  if (tls)
    llvm::cantFail(layout.AddBank(2, sizeof(freebsd_arm::TLS)));
  for (const auto &reg : description->GetRegisters()) {
    RegisterBankID bank = 0;
    uint32_t offset = 0;
    if (reg.id <= reg_ids::gpr_cpsr)
      offset = 4 * reg.id;
    else if (reg.id <= reg_ids::fpu_q15) {
      bank = 1;

      if (reg.id == reg_ids::fpu_fpscr)
        offset = offsetof(freebsd_arm::FPU, fpscr);
      else if (reg.id < reg_ids::fpu_fpscr)
        offset = 4 * (reg.id - reg_ids::fpu_s0);
      else if (reg.id < reg_ids::fpu_q0)
        offset = 8 * (reg.id - reg_ids::fpu_d0);
      else
        offset = 16 * (reg.id - reg_ids::fpu_q0);
    } else {
      bank = 2;
    }
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, reg.byte_size});

    llvm::cantFail(layout.Add(reg.id, std::move(location)));
  }
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
