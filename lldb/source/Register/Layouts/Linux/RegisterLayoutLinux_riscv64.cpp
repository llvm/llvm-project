//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_riscv64.h"
#include "lldb/Register/Arch/lldb-riscv-register-enums.h"
#include "lldb/Utility/ArchSpec.h"
#include <iterator>


namespace reg_ids = lldb_private::registers::riscv::id;
using namespace lldb_private;

static RegisterLayout CreateLayout(const RegisterDescription &description,
                                   lldb::ByteOrder byte_order) {
  RegisterLayout layout(byte_order);
  llvm::cantFail(layout.AddBank(0, sizeof(linux_riscv64::GPR)));
  auto add = [&](RegisterID id, RegisterBankID bank, uint32_t offset,
                 uint32_t size) {
    if (!description.Find(id))
      return;
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, size});

    llvm::cantFail(layout.Add(id, std::move(location)));
  };
  const RegisterID gpr_slots[] = {
      reg_ids::gpr_pc,  reg_ids::gpr_x1,  reg_ids::gpr_x2,  reg_ids::gpr_x3,
      reg_ids::gpr_x4,  reg_ids::gpr_x5,  reg_ids::gpr_x6,  reg_ids::gpr_x7,
      reg_ids::gpr_x8,  reg_ids::gpr_x9,  reg_ids::gpr_x10, reg_ids::gpr_x11,
      reg_ids::gpr_x12, reg_ids::gpr_x13, reg_ids::gpr_x14, reg_ids::gpr_x15,
      reg_ids::gpr_x16, reg_ids::gpr_x17, reg_ids::gpr_x18, reg_ids::gpr_x19,
      reg_ids::gpr_x20, reg_ids::gpr_x21, reg_ids::gpr_x22, reg_ids::gpr_x23,
      reg_ids::gpr_x24, reg_ids::gpr_x25, reg_ids::gpr_x26, reg_ids::gpr_x27,
      reg_ids::gpr_x28, reg_ids::gpr_x29, reg_ids::gpr_x30, reg_ids::gpr_x31};
  for (size_t slot = 0; slot != std::size(gpr_slots); ++slot)
    add(gpr_slots[slot], 0, slot * 8, 8);
  if (description.Find(reg_ids::fpr_f0)) {
    llvm::cantFail(layout.AddBank(1, sizeof(linux_riscv64::FPR)));
    const RegisterID fp_slots[] = {
        reg_ids::fpr_f0,  reg_ids::fpr_f1,  reg_ids::fpr_f2,  reg_ids::fpr_f3,
        reg_ids::fpr_f4,  reg_ids::fpr_f5,  reg_ids::fpr_f6,  reg_ids::fpr_f7,
        reg_ids::fpr_f8,  reg_ids::fpr_f9,  reg_ids::fpr_f10, reg_ids::fpr_f11,
        reg_ids::fpr_f12, reg_ids::fpr_f13, reg_ids::fpr_f14, reg_ids::fpr_f15,
        reg_ids::fpr_f16, reg_ids::fpr_f17, reg_ids::fpr_f18, reg_ids::fpr_f19,
        reg_ids::fpr_f20, reg_ids::fpr_f21, reg_ids::fpr_f22, reg_ids::fpr_f23,
        reg_ids::fpr_f24, reg_ids::fpr_f25, reg_ids::fpr_f26, reg_ids::fpr_f27,
        reg_ids::fpr_f28, reg_ids::fpr_f29, reg_ids::fpr_f30, reg_ids::fpr_f31};
    for (size_t slot = 0; slot != std::size(fp_slots); ++slot)
      add(fp_slots[slot], 1, slot * 8, 8);
    add(reg_ids::fpr_fcsr, 1, 256, 4);
  }
  return layout;
}

RegisterLayout
lldb_private::CreateLinuxRISCV64RegisterLayout(const ArchSpec &arch,
                                               Flags features) {
  auto description = RegisterInfo_riscv64::CreateDescription(features);
  auto layout = CreateLayout(*description, arch.GetByteOrder());
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
