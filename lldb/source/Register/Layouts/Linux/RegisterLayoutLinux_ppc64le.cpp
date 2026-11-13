//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_ppc64le.h"
#include "lldb/Register/Arch/lldb-ppc64le-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64le-register-enums.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstddef>
#include <iterator>

namespace dwarf_regs = lldb_private::registers::ppc64le::dwarf;
namespace ehframe_regs = lldb_private::registers::ppc64le::ehframe;
namespace reg_ids = lldb_private::registers::ppc64le::id;
using namespace lldb_private;
using namespace lldb_private::linux_ppc64le;

namespace {
std::shared_ptr<const RegisterDescription> Description(const ArchSpec &) {
  static const auto description = [] {
    auto cpu = RegisterInfo_ppc64le::CreateDescription(
        ArchSpec("powerpc64le-unknown-linux"));
    std::vector<RegisterDescriptor> registers(cpu->GetRegisters().begin(),
                                              cpu->GetRegisters().end());
    std::vector<RegisterGroup> groups(cpu->GetGroups().begin(),
                                      cpu->GetGroups().end());
    RegisterID next = 0;
    for (const auto &reg : registers)
      next = std::max(next, reg.id + 1);
    {
      RegisterDescriptor extra;
      extra.id = next++;
      extra.name = "origr3";
      extra.byte_size = 8;
      extra.numbers[lldb::eRegisterKindLLDB] = reg_ids::gpr_origr3;
      extra.numbers[lldb::eRegisterKindDWARF] =
          dwarf_regs::origr3;
      extra.numbers[lldb::eRegisterKindEHFrame] =
          ehframe_regs::origr3;
      registers.push_back(std::move(extra));
    }
    {
      RegisterDescriptor extra;
      extra.id = next++;
      extra.name = "softe";
      extra.byte_size = 8;
      extra.numbers[lldb::eRegisterKindLLDB] = reg_ids::gpr_softe;
      extra.numbers[lldb::eRegisterKindDWARF] =
          dwarf_regs::softe;
      extra.numbers[lldb::eRegisterKindEHFrame] =
          ehframe_regs::softe;
      registers.push_back(std::move(extra));
    }
    {
      RegisterDescriptor extra;
      extra.id = next++;
      extra.name = "trap";
      extra.byte_size = 8;
      extra.numbers[lldb::eRegisterKindLLDB] = reg_ids::gpr_trap;
      extra.numbers[lldb::eRegisterKindDWARF] =
          dwarf_regs::trap;
      extra.numbers[lldb::eRegisterKindEHFrame] =
          ehframe_regs::trap;
      registers.push_back(std::move(extra));
    }
    llvm::sort(registers, [](const auto &a, const auto &b) {
      return a.numbers[lldb::eRegisterKindLLDB] <
             b.numbers[lldb::eRegisterKindLLDB];
    });
    groups[0].registers.clear();
    for (const auto &reg : registers)
      if (reg.numbers[lldb::eRegisterKindLLDB] <= reg_ids::k_last_gpr)
        groups[0].registers.push_back(reg.id);
    return std::make_shared<RegisterDescription>(llvm::cantFail(
        RegisterDescription::Create(std::move(registers), std::move(groups))));
  }();
  return description;
}

RegisterLayout Build(const ArchSpec &arch,
                     const RegisterDescription &description_value) {
  const auto *description = &description_value;
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(GPR)));
  llvm::cantFail(layout.AddBank(1, sizeof(FPR)));
  llvm::cantFail(layout.AddBank(2, sizeof(VMX)));
  llvm::cantFail(layout.AddBank(3, sizeof(VSX)));
  auto add = [&](const char *name, RegisterBankID bank, uint32_t offset) {
    const auto *reg = description->Find(name);
    if (!reg)
      return;
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, reg->byte_size});

    llvm::cantFail(layout.Add(reg->id, std::move(location)));
  };
  add("r0", 0, offsetof(GPR, r0));
  add("r1", 0, offsetof(GPR, r1));
  add("r2", 0, offsetof(GPR, r2));
  add("r3", 0, offsetof(GPR, r3));
  add("r4", 0, offsetof(GPR, r4));
  add("r5", 0, offsetof(GPR, r5));
  add("r6", 0, offsetof(GPR, r6));
  add("r7", 0, offsetof(GPR, r7));
  add("r8", 0, offsetof(GPR, r8));
  add("r9", 0, offsetof(GPR, r9));
  add("r10", 0, offsetof(GPR, r10));
  add("r11", 0, offsetof(GPR, r11));
  add("r12", 0, offsetof(GPR, r12));
  add("r13", 0, offsetof(GPR, r13));
  add("r14", 0, offsetof(GPR, r14));
  add("r15", 0, offsetof(GPR, r15));
  add("r16", 0, offsetof(GPR, r16));
  add("r17", 0, offsetof(GPR, r17));
  add("r18", 0, offsetof(GPR, r18));
  add("r19", 0, offsetof(GPR, r19));
  add("r20", 0, offsetof(GPR, r20));
  add("r21", 0, offsetof(GPR, r21));
  add("r22", 0, offsetof(GPR, r22));
  add("r23", 0, offsetof(GPR, r23));
  add("r24", 0, offsetof(GPR, r24));
  add("r25", 0, offsetof(GPR, r25));
  add("r26", 0, offsetof(GPR, r26));
  add("r27", 0, offsetof(GPR, r27));
  add("r28", 0, offsetof(GPR, r28));
  add("r29", 0, offsetof(GPR, r29));
  add("r30", 0, offsetof(GPR, r30));
  add("r31", 0, offsetof(GPR, r31));
  add("pc", 0, offsetof(GPR, pc));
  add("msr", 0, offsetof(GPR, msr));
  add("origr3", 0, offsetof(GPR, origr3));
  add("ctr", 0, offsetof(GPR, ctr));
  add("lr", 0, offsetof(GPR, lr));
  add("xer", 0, offsetof(GPR, xer));
  add("cr", 0, offsetof(GPR, cr));
  add("softe", 0, offsetof(GPR, softe));
  add("trap", 0, offsetof(GPR, trap));
  add("f0", 1, offsetof(FPR, f0));
  add("f1", 1, offsetof(FPR, f1));
  add("f2", 1, offsetof(FPR, f2));
  add("f3", 1, offsetof(FPR, f3));
  add("f4", 1, offsetof(FPR, f4));
  add("f5", 1, offsetof(FPR, f5));
  add("f6", 1, offsetof(FPR, f6));
  add("f7", 1, offsetof(FPR, f7));
  add("f8", 1, offsetof(FPR, f8));
  add("f9", 1, offsetof(FPR, f9));
  add("f10", 1, offsetof(FPR, f10));
  add("f11", 1, offsetof(FPR, f11));
  add("f12", 1, offsetof(FPR, f12));
  add("f13", 1, offsetof(FPR, f13));
  add("f14", 1, offsetof(FPR, f14));
  add("f15", 1, offsetof(FPR, f15));
  add("f16", 1, offsetof(FPR, f16));
  add("f17", 1, offsetof(FPR, f17));
  add("f18", 1, offsetof(FPR, f18));
  add("f19", 1, offsetof(FPR, f19));
  add("f20", 1, offsetof(FPR, f20));
  add("f21", 1, offsetof(FPR, f21));
  add("f22", 1, offsetof(FPR, f22));
  add("f23", 1, offsetof(FPR, f23));
  add("f24", 1, offsetof(FPR, f24));
  add("f25", 1, offsetof(FPR, f25));
  add("f26", 1, offsetof(FPR, f26));
  add("f27", 1, offsetof(FPR, f27));
  add("f28", 1, offsetof(FPR, f28));
  add("f29", 1, offsetof(FPR, f29));
  add("f30", 1, offsetof(FPR, f30));
  add("f31", 1, offsetof(FPR, f31));
  add("fpscr", 1, offsetof(FPR, fpscr));
  add("vr0", 2, offsetof(VMX, vr0));
  add("vr1", 2, offsetof(VMX, vr1));
  add("vr2", 2, offsetof(VMX, vr2));
  add("vr3", 2, offsetof(VMX, vr3));
  add("vr4", 2, offsetof(VMX, vr4));
  add("vr5", 2, offsetof(VMX, vr5));
  add("vr6", 2, offsetof(VMX, vr6));
  add("vr7", 2, offsetof(VMX, vr7));
  add("vr8", 2, offsetof(VMX, vr8));
  add("vr9", 2, offsetof(VMX, vr9));
  add("vr10", 2, offsetof(VMX, vr10));
  add("vr11", 2, offsetof(VMX, vr11));
  add("vr12", 2, offsetof(VMX, vr12));
  add("vr13", 2, offsetof(VMX, vr13));
  add("vr14", 2, offsetof(VMX, vr14));
  add("vr15", 2, offsetof(VMX, vr15));
  add("vr16", 2, offsetof(VMX, vr16));
  add("vr17", 2, offsetof(VMX, vr17));
  add("vr18", 2, offsetof(VMX, vr18));
  add("vr19", 2, offsetof(VMX, vr19));
  add("vr20", 2, offsetof(VMX, vr20));
  add("vr21", 2, offsetof(VMX, vr21));
  add("vr22", 2, offsetof(VMX, vr22));
  add("vr23", 2, offsetof(VMX, vr23));
  add("vr24", 2, offsetof(VMX, vr24));
  add("vr25", 2, offsetof(VMX, vr25));
  add("vr26", 2, offsetof(VMX, vr26));
  add("vr27", 2, offsetof(VMX, vr27));
  add("vr28", 2, offsetof(VMX, vr28));
  add("vr29", 2, offsetof(VMX, vr29));
  add("vr30", 2, offsetof(VMX, vr30));
  add("vr31", 2, offsetof(VMX, vr31));
  add("vscr", 2, offsetof(VMX, vscr));
  add("vrsave", 2, offsetof(VMX, vrsave));
  for (uint32_t i = 0; i < 64; ++i) {
    const auto *reg = description->Find("vs" + std::to_string(i));
    if (!reg)
      continue;
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    if (i < 32) {
      location.pieces.push_back({3, i * 8, 8, 0});
      location.pieces.push_back({1, i * 8, 8, 8});
    } else
      location.pieces.push_back({2, (i - 32) * 16, 16, 0});
    llvm::cantFail(layout.Add(reg->id, std::move(location)));
  }
  return layout;
}
} // namespace

RegisterLayout
lldb_private::CreateLinuxPPC64LERegisterLayout(const ArchSpec &arch) {
  auto description = Description(arch);
  auto layout = Build(arch, *description);
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
