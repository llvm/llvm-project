//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterLayoutAIX_ppc64.h"
#include "lldb/Register/Arch/lldb-ppc64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-ppc64-register-enums.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
using namespace lldb_private;
using namespace lldb_private::aix_ppc64;

namespace {
template <typename GP> RegisterLayout Build(const ArchSpec &arch) {
  auto description = RegisterInfo_ppc64::CreateDescription(arch);
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(GP)));
  llvm::cantFail(layout.AddBank(1, sizeof(FPR_PPC64)));
  llvm::cantFail(layout.AddBank(2, sizeof(VMX_PPC64)));
  llvm::cantFail(layout.AddBank(3, sizeof(VSX_PPC64)));
  auto add = [&](const char *name, RegisterBankID bank, uint32_t offset) {
    const auto *reg = description->Find(name);
    if (!reg)
      return;
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset, reg->byte_size});

    llvm::cantFail(layout.Add(reg->id, std::move(location)));
  };
  add("r0", 0, offsetof(GP, r0));
  add("r1", 0, offsetof(GP, r1));
  add("r2", 0, offsetof(GP, r2));
  add("r3", 0, offsetof(GP, r3));
  add("r4", 0, offsetof(GP, r4));
  add("r5", 0, offsetof(GP, r5));
  add("r6", 0, offsetof(GP, r6));
  add("r7", 0, offsetof(GP, r7));
  add("r8", 0, offsetof(GP, r8));
  add("r9", 0, offsetof(GP, r9));
  add("r10", 0, offsetof(GP, r10));
  add("r11", 0, offsetof(GP, r11));
  add("r12", 0, offsetof(GP, r12));
  add("r13", 0, offsetof(GP, r13));
  add("r14", 0, offsetof(GP, r14));
  add("r15", 0, offsetof(GP, r15));
  add("r16", 0, offsetof(GP, r16));
  add("r17", 0, offsetof(GP, r17));
  add("r18", 0, offsetof(GP, r18));
  add("r19", 0, offsetof(GP, r19));
  add("r20", 0, offsetof(GP, r20));
  add("r21", 0, offsetof(GP, r21));
  add("r22", 0, offsetof(GP, r22));
  add("r23", 0, offsetof(GP, r23));
  add("r24", 0, offsetof(GP, r24));
  add("r25", 0, offsetof(GP, r25));
  add("r26", 0, offsetof(GP, r26));
  add("r27", 0, offsetof(GP, r27));
  add("r28", 0, offsetof(GP, r28));
  add("r29", 0, offsetof(GP, r29));
  add("r30", 0, offsetof(GP, r30));
  add("r31", 0, offsetof(GP, r31));
  add("cr", 0, offsetof(GP, cr));
  add("msr", 0, offsetof(GP, msr));
  add("xer", 0, offsetof(GP, xer));
  add("lr", 0, offsetof(GP, lr));
  add("ctr", 0, offsetof(GP, ctr));
  add("pc", 0, offsetof(GP, pc));
  add("f0", 1, offsetof(FPR_PPC64, f0));
  add("f1", 1, offsetof(FPR_PPC64, f1));
  add("f2", 1, offsetof(FPR_PPC64, f2));
  add("f3", 1, offsetof(FPR_PPC64, f3));
  add("f4", 1, offsetof(FPR_PPC64, f4));
  add("f5", 1, offsetof(FPR_PPC64, f5));
  add("f6", 1, offsetof(FPR_PPC64, f6));
  add("f7", 1, offsetof(FPR_PPC64, f7));
  add("f8", 1, offsetof(FPR_PPC64, f8));
  add("f9", 1, offsetof(FPR_PPC64, f9));
  add("f10", 1, offsetof(FPR_PPC64, f10));
  add("f11", 1, offsetof(FPR_PPC64, f11));
  add("f12", 1, offsetof(FPR_PPC64, f12));
  add("f13", 1, offsetof(FPR_PPC64, f13));
  add("f14", 1, offsetof(FPR_PPC64, f14));
  add("f15", 1, offsetof(FPR_PPC64, f15));
  add("f16", 1, offsetof(FPR_PPC64, f16));
  add("f17", 1, offsetof(FPR_PPC64, f17));
  add("f18", 1, offsetof(FPR_PPC64, f18));
  add("f19", 1, offsetof(FPR_PPC64, f19));
  add("f20", 1, offsetof(FPR_PPC64, f20));
  add("f21", 1, offsetof(FPR_PPC64, f21));
  add("f22", 1, offsetof(FPR_PPC64, f22));
  add("f23", 1, offsetof(FPR_PPC64, f23));
  add("f24", 1, offsetof(FPR_PPC64, f24));
  add("f25", 1, offsetof(FPR_PPC64, f25));
  add("f26", 1, offsetof(FPR_PPC64, f26));
  add("f27", 1, offsetof(FPR_PPC64, f27));
  add("f28", 1, offsetof(FPR_PPC64, f28));
  add("f29", 1, offsetof(FPR_PPC64, f29));
  add("f30", 1, offsetof(FPR_PPC64, f30));
  add("f31", 1, offsetof(FPR_PPC64, f31));
  add("fpscr", 1, offsetof(FPR_PPC64, fpscr));
  add("vr0", 2, offsetof(VMX_PPC64, vr0));
  add("vr1", 2, offsetof(VMX_PPC64, vr1));
  add("vr2", 2, offsetof(VMX_PPC64, vr2));
  add("vr3", 2, offsetof(VMX_PPC64, vr3));
  add("vr4", 2, offsetof(VMX_PPC64, vr4));
  add("vr5", 2, offsetof(VMX_PPC64, vr5));
  add("vr6", 2, offsetof(VMX_PPC64, vr6));
  add("vr7", 2, offsetof(VMX_PPC64, vr7));
  add("vr8", 2, offsetof(VMX_PPC64, vr8));
  add("vr9", 2, offsetof(VMX_PPC64, vr9));
  add("vr10", 2, offsetof(VMX_PPC64, vr10));
  add("vr11", 2, offsetof(VMX_PPC64, vr11));
  add("vr12", 2, offsetof(VMX_PPC64, vr12));
  add("vr13", 2, offsetof(VMX_PPC64, vr13));
  add("vr14", 2, offsetof(VMX_PPC64, vr14));
  add("vr15", 2, offsetof(VMX_PPC64, vr15));
  add("vr16", 2, offsetof(VMX_PPC64, vr16));
  add("vr17", 2, offsetof(VMX_PPC64, vr17));
  add("vr18", 2, offsetof(VMX_PPC64, vr18));
  add("vr19", 2, offsetof(VMX_PPC64, vr19));
  add("vr20", 2, offsetof(VMX_PPC64, vr20));
  add("vr21", 2, offsetof(VMX_PPC64, vr21));
  add("vr22", 2, offsetof(VMX_PPC64, vr22));
  add("vr23", 2, offsetof(VMX_PPC64, vr23));
  add("vr24", 2, offsetof(VMX_PPC64, vr24));
  add("vr25", 2, offsetof(VMX_PPC64, vr25));
  add("vr26", 2, offsetof(VMX_PPC64, vr26));
  add("vr27", 2, offsetof(VMX_PPC64, vr27));
  add("vr28", 2, offsetof(VMX_PPC64, vr28));
  add("vr29", 2, offsetof(VMX_PPC64, vr29));
  add("vr30", 2, offsetof(VMX_PPC64, vr30));
  add("vr31", 2, offsetof(VMX_PPC64, vr31));
  add("vscr", 2, offsetof(VMX_PPC64, vscr));
  add("vrsave", 2, offsetof(VMX_PPC64, vrsave));
  add("vs0", 3, offsetof(VSX_PPC64, vs0));
  add("vs1", 3, offsetof(VSX_PPC64, vs1));
  add("vs2", 3, offsetof(VSX_PPC64, vs2));
  add("vs3", 3, offsetof(VSX_PPC64, vs3));
  add("vs4", 3, offsetof(VSX_PPC64, vs4));
  add("vs5", 3, offsetof(VSX_PPC64, vs5));
  add("vs6", 3, offsetof(VSX_PPC64, vs6));
  add("vs7", 3, offsetof(VSX_PPC64, vs7));
  add("vs8", 3, offsetof(VSX_PPC64, vs8));
  add("vs9", 3, offsetof(VSX_PPC64, vs9));
  add("vs10", 3, offsetof(VSX_PPC64, vs10));
  add("vs11", 3, offsetof(VSX_PPC64, vs11));
  add("vs12", 3, offsetof(VSX_PPC64, vs12));
  add("vs13", 3, offsetof(VSX_PPC64, vs13));
  add("vs14", 3, offsetof(VSX_PPC64, vs14));
  add("vs15", 3, offsetof(VSX_PPC64, vs15));
  add("vs16", 3, offsetof(VSX_PPC64, vs16));
  add("vs17", 3, offsetof(VSX_PPC64, vs17));
  add("vs18", 3, offsetof(VSX_PPC64, vs18));
  add("vs19", 3, offsetof(VSX_PPC64, vs19));
  add("vs20", 3, offsetof(VSX_PPC64, vs20));
  add("vs21", 3, offsetof(VSX_PPC64, vs21));
  add("vs22", 3, offsetof(VSX_PPC64, vs22));
  add("vs23", 3, offsetof(VSX_PPC64, vs23));
  add("vs24", 3, offsetof(VSX_PPC64, vs24));
  add("vs25", 3, offsetof(VSX_PPC64, vs25));
  add("vs26", 3, offsetof(VSX_PPC64, vs26));
  add("vs27", 3, offsetof(VSX_PPC64, vs27));
  add("vs28", 3, offsetof(VSX_PPC64, vs28));
  add("vs29", 3, offsetof(VSX_PPC64, vs29));
  add("vs30", 3, offsetof(VSX_PPC64, vs30));
  add("vs31", 3, offsetof(VSX_PPC64, vs31));
  add("vs32", 3, offsetof(VSX_PPC64, vs32));
  add("vs33", 3, offsetof(VSX_PPC64, vs33));
  add("vs34", 3, offsetof(VSX_PPC64, vs34));
  add("vs35", 3, offsetof(VSX_PPC64, vs35));
  add("vs36", 3, offsetof(VSX_PPC64, vs36));
  add("vs37", 3, offsetof(VSX_PPC64, vs37));
  add("vs38", 3, offsetof(VSX_PPC64, vs38));
  add("vs39", 3, offsetof(VSX_PPC64, vs39));
  add("vs40", 3, offsetof(VSX_PPC64, vs40));
  add("vs41", 3, offsetof(VSX_PPC64, vs41));
  add("vs42", 3, offsetof(VSX_PPC64, vs42));
  add("vs43", 3, offsetof(VSX_PPC64, vs43));
  add("vs44", 3, offsetof(VSX_PPC64, vs44));
  add("vs45", 3, offsetof(VSX_PPC64, vs45));
  add("vs46", 3, offsetof(VSX_PPC64, vs46));
  add("vs47", 3, offsetof(VSX_PPC64, vs47));
  add("vs48", 3, offsetof(VSX_PPC64, vs48));
  add("vs49", 3, offsetof(VSX_PPC64, vs49));
  add("vs50", 3, offsetof(VSX_PPC64, vs50));
  add("vs51", 3, offsetof(VSX_PPC64, vs51));
  add("vs52", 3, offsetof(VSX_PPC64, vs52));
  add("vs53", 3, offsetof(VSX_PPC64, vs53));
  add("vs54", 3, offsetof(VSX_PPC64, vs54));
  add("vs55", 3, offsetof(VSX_PPC64, vs55));
  add("vs56", 3, offsetof(VSX_PPC64, vs56));
  add("vs57", 3, offsetof(VSX_PPC64, vs57));
  add("vs58", 3, offsetof(VSX_PPC64, vs58));
  add("vs59", 3, offsetof(VSX_PPC64, vs59));
  add("vs60", 3, offsetof(VSX_PPC64, vs60));
  add("vs61", 3, offsetof(VSX_PPC64, vs61));
  add("vs62", 3, offsetof(VSX_PPC64, vs62));
  add("vs63", 3, offsetof(VSX_PPC64, vs63));
  return layout;
}
} // namespace

RegisterLayout
lldb_private::CreateAIXPPC64RegisterLayout(const ArchSpec &arch) {
  auto layout = arch.GetMachine() == llvm::Triple::ppc ? Build<GPR_PPC>(arch)
                                                       : Build<GPR_PPC64>(arch);
  return llvm::cantFail(RegisterLayout::Create(
      RegisterInfo_ppc64::CreateDescription(arch), std::move(layout)));
}
