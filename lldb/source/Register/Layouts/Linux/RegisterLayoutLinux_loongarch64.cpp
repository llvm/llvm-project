//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_loongarch64.h"
#include "lldb/Register/Arch/lldb-loongarch-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-loongarch-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_loongarch64.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iterator>

namespace dwarf_regs = lldb_private::registers::loongarch::dwarf;
namespace ehframe_regs = lldb_private::registers::loongarch::ehframe;
namespace reg_ids = lldb_private::registers::loongarch::id;
using namespace lldb_private;

static std::shared_ptr<const RegisterDescription> LinuxDescription() {
  static const auto description = [] {
    auto arch = RegisterInfo_loongarch64::CreateDescription();
    std::vector<RegisterDescriptor> registers(arch->GetRegisters().begin(),
                                              arch->GetRegisters().end());
    std::vector<RegisterGroup> groups(arch->GetGroups().begin(),
                                      arch->GetGroups().end());
    // Linux syscall state and padding slots are extensions of the CPU catalog.
    struct Extra {
      RegisterID id;
      const char *name;
      uint32_t dwarf;
      uint32_t ehframe;
    };
    const Extra extras[] = {
        {reg_ids::gpr_orig_a0, "orig_a0", dwarf_regs::gpr_orig_a0,
         ehframe_regs::gpr_orig_a0},
        {reg_ids::gpr_reserved0, "reserved0", dwarf_regs::gpr_reserved0,
         ehframe_regs::gpr_reserved0},
        {reg_ids::gpr_reserved1, "reserved1", dwarf_regs::gpr_reserved1,
         ehframe_regs::gpr_reserved1},
        {reg_ids::gpr_reserved2, "reserved2", dwarf_regs::gpr_reserved2,
         ehframe_regs::gpr_reserved2},
        {reg_ids::gpr_reserved3, "reserved3", dwarf_regs::gpr_reserved3,
         ehframe_regs::gpr_reserved3},
        {reg_ids::gpr_reserved4, "reserved4", dwarf_regs::gpr_reserved4,
         ehframe_regs::gpr_reserved4},
        {reg_ids::gpr_reserved5, "reserved5", dwarf_regs::gpr_reserved5,
         ehframe_regs::gpr_reserved5},
        {reg_ids::gpr_reserved6, "reserved6", dwarf_regs::gpr_reserved6,
         ehframe_regs::gpr_reserved6},
        {reg_ids::gpr_reserved7, "reserved7", dwarf_regs::gpr_reserved7,
         ehframe_regs::gpr_reserved7},
        {reg_ids::gpr_reserved8, "reserved8", dwarf_regs::gpr_reserved8,
         ehframe_regs::gpr_reserved8},
        {reg_ids::gpr_reserved9, "reserved9", dwarf_regs::gpr_reserved9,
         ehframe_regs::gpr_reserved9}};
    for (const auto &extra : extras) {
      RegisterDescriptor reg;
      reg.id = extra.id;
      reg.name = extra.name;
      reg.byte_size = 8;
      reg.numbers[lldb::eRegisterKindLLDB] = extra.id;
      reg.numbers[lldb::eRegisterKindDWARF] = extra.dwarf;
      reg.numbers[lldb::eRegisterKindEHFrame] = extra.ehframe;
      registers.push_back(std::move(reg));
      groups[0].registers.push_back(extra.id);
    }
    llvm::sort(registers,
               [](const auto &a, const auto &b) { return a.id < b.id; });
    llvm::sort(groups[0].registers);
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(registers), std::move(groups))));
  }();
  return description;
}

static RegisterLayout CreateLayout(const RegisterDescription &description,
                                   lldb::ByteOrder byte_order) {
  RegisterLayout layout(byte_order);
  const uint32_t sizes[] = {360, 272, 512, 1024};
  for (uint32_t bank = 0; bank != 4; ++bank)
    llvm::cantFail(layout.AddBank(bank, sizes[bank]));
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
      reg_ids::gpr_r0,        reg_ids::gpr_r1,        reg_ids::gpr_r2,
      reg_ids::gpr_r3,        reg_ids::gpr_r4,        reg_ids::gpr_r5,
      reg_ids::gpr_r6,        reg_ids::gpr_r7,        reg_ids::gpr_r8,
      reg_ids::gpr_r9,        reg_ids::gpr_r10,       reg_ids::gpr_r11,
      reg_ids::gpr_r12,       reg_ids::gpr_r13,       reg_ids::gpr_r14,
      reg_ids::gpr_r15,       reg_ids::gpr_r16,       reg_ids::gpr_r17,
      reg_ids::gpr_r18,       reg_ids::gpr_r19,       reg_ids::gpr_r20,
      reg_ids::gpr_r21,       reg_ids::gpr_r22,       reg_ids::gpr_r23,
      reg_ids::gpr_r24,       reg_ids::gpr_r25,       reg_ids::gpr_r26,
      reg_ids::gpr_r27,       reg_ids::gpr_r28,       reg_ids::gpr_r29,
      reg_ids::gpr_r30,       reg_ids::gpr_r31,       reg_ids::gpr_orig_a0,
      reg_ids::gpr_pc,        reg_ids::gpr_badv,      reg_ids::gpr_reserved0,
      reg_ids::gpr_reserved1, reg_ids::gpr_reserved2, reg_ids::gpr_reserved3,
      reg_ids::gpr_reserved4, reg_ids::gpr_reserved5, reg_ids::gpr_reserved6,
      reg_ids::gpr_reserved7, reg_ids::gpr_reserved8, reg_ids::gpr_reserved9};
  for (size_t slot = 0; slot != std::size(gpr_slots); ++slot) {
    if (gpr_slots[slot] == reg_ids::gpr_r0)
      continue;
    add(gpr_slots[slot], 0, slot * 8, 8);
  }
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
  const RegisterID fcc_slots[] = {reg_ids::fpr_fcc0, reg_ids::fpr_fcc1,
                                  reg_ids::fpr_fcc2, reg_ids::fpr_fcc3,
                                  reg_ids::fpr_fcc4, reg_ids::fpr_fcc5,
                                  reg_ids::fpr_fcc6, reg_ids::fpr_fcc7};
  for (size_t slot = 0; slot != std::size(fcc_slots); ++slot)
    add(fcc_slots[slot], 1, 256 + slot, 1);
  add(reg_ids::fpr_fcsr, 1, 264, 4);
  const RegisterID lsx_slots[] = {
      reg_ids::lsx_vr0,  reg_ids::lsx_vr1,  reg_ids::lsx_vr2,
      reg_ids::lsx_vr3,  reg_ids::lsx_vr4,  reg_ids::lsx_vr5,
      reg_ids::lsx_vr6,  reg_ids::lsx_vr7,  reg_ids::lsx_vr8,
      reg_ids::lsx_vr9,  reg_ids::lsx_vr10, reg_ids::lsx_vr11,
      reg_ids::lsx_vr12, reg_ids::lsx_vr13, reg_ids::lsx_vr14,
      reg_ids::lsx_vr15, reg_ids::lsx_vr16, reg_ids::lsx_vr17,
      reg_ids::lsx_vr18, reg_ids::lsx_vr19, reg_ids::lsx_vr20,
      reg_ids::lsx_vr21, reg_ids::lsx_vr22, reg_ids::lsx_vr23,
      reg_ids::lsx_vr24, reg_ids::lsx_vr25, reg_ids::lsx_vr26,
      reg_ids::lsx_vr27, reg_ids::lsx_vr28, reg_ids::lsx_vr29,
      reg_ids::lsx_vr30, reg_ids::lsx_vr31};
  for (size_t slot = 0; slot != std::size(lsx_slots); ++slot)
    add(lsx_slots[slot], 2, slot * 16, 16);
  const RegisterID lasx_slots[] = {
      reg_ids::lasx_xr0,  reg_ids::lasx_xr1,  reg_ids::lasx_xr2,
      reg_ids::lasx_xr3,  reg_ids::lasx_xr4,  reg_ids::lasx_xr5,
      reg_ids::lasx_xr6,  reg_ids::lasx_xr7,  reg_ids::lasx_xr8,
      reg_ids::lasx_xr9,  reg_ids::lasx_xr10, reg_ids::lasx_xr11,
      reg_ids::lasx_xr12, reg_ids::lasx_xr13, reg_ids::lasx_xr14,
      reg_ids::lasx_xr15, reg_ids::lasx_xr16, reg_ids::lasx_xr17,
      reg_ids::lasx_xr18, reg_ids::lasx_xr19, reg_ids::lasx_xr20,
      reg_ids::lasx_xr21, reg_ids::lasx_xr22, reg_ids::lasx_xr23,
      reg_ids::lasx_xr24, reg_ids::lasx_xr25, reg_ids::lasx_xr26,
      reg_ids::lasx_xr27, reg_ids::lasx_xr28, reg_ids::lasx_xr29,
      reg_ids::lasx_xr30, reg_ids::lasx_xr31};
  for (size_t slot = 0; slot != std::size(lasx_slots); ++slot)
    add(lasx_slots[slot], 3, slot * 32, 32);
  return layout;
}

RegisterLayout
lldb_private::CreateLinuxLoongArch64RegisterLayout(const ArchSpec &arch) {
  auto description = LinuxDescription();
  auto layout = CreateLayout(*description, arch.GetByteOrder());
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
