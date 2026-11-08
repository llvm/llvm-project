//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_powerpc.h"
#include "lldb/Utility/ArchSpec.h"
#include <cstddef>
#include <cstdint>
using namespace lldb_private;
namespace {
// Mirrors FreeBSD's machine/reg.h; native SDK checks remain in native code.
typedef struct _GPR64 {
  uint64_t r0;
  uint64_t r1;
  uint64_t r2;
  uint64_t r3;
  uint64_t r4;
  uint64_t r5;
  uint64_t r6;
  uint64_t r7;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t r16;
  uint64_t r17;
  uint64_t r18;
  uint64_t r19;
  uint64_t r20;
  uint64_t r21;
  uint64_t r22;
  uint64_t r23;
  uint64_t r24;
  uint64_t r25;
  uint64_t r26;
  uint64_t r27;
  uint64_t r28;
  uint64_t r29;
  uint64_t r30;
  uint64_t r31;
  uint64_t lr;
  uint64_t cr;
  uint64_t xer;
  uint64_t ctr;
  uint64_t pc;
} GPR64;

typedef struct _GPR32 {
  uint32_t r0;
  uint32_t r1;
  uint32_t r2;
  uint32_t r3;
  uint32_t r4;
  uint32_t r5;
  uint32_t r6;
  uint32_t r7;
  uint32_t r8;
  uint32_t r9;
  uint32_t r10;
  uint32_t r11;
  uint32_t r12;
  uint32_t r13;
  uint32_t r14;
  uint32_t r15;
  uint32_t r16;
  uint32_t r17;
  uint32_t r18;
  uint32_t r19;
  uint32_t r20;
  uint32_t r21;
  uint32_t r22;
  uint32_t r23;
  uint32_t r24;
  uint32_t r25;
  uint32_t r26;
  uint32_t r27;
  uint32_t r28;
  uint32_t r29;
  uint32_t r30;
  uint32_t r31;
  uint32_t lr;
  uint32_t cr;
  uint32_t xer;
  uint32_t ctr;
  uint32_t pc;
} GPR32;

typedef struct _FPR {
  uint64_t f0;
  uint64_t f1;
  uint64_t f2;
  uint64_t f3;
  uint64_t f4;
  uint64_t f5;
  uint64_t f6;
  uint64_t f7;
  uint64_t f8;
  uint64_t f9;
  uint64_t f10;
  uint64_t f11;
  uint64_t f12;
  uint64_t f13;
  uint64_t f14;
  uint64_t f15;
  uint64_t f16;
  uint64_t f17;
  uint64_t f18;
  uint64_t f19;
  uint64_t f20;
  uint64_t f21;
  uint64_t f22;
  uint64_t f23;
  uint64_t f24;
  uint64_t f25;
  uint64_t f26;
  uint64_t f27;
  uint64_t f28;
  uint64_t f29;
  uint64_t f30;
  uint64_t f31;
  uint64_t fpscr;
} FPR;

typedef struct _VMX {
  uint32_t v0[4];
  uint32_t v1[4];
  uint32_t v2[4];
  uint32_t v3[4];
  uint32_t v4[4];
  uint32_t v5[4];
  uint32_t v6[4];
  uint32_t v7[4];
  uint32_t v8[4];
  uint32_t v9[4];
  uint32_t v10[4];
  uint32_t v11[4];
  uint32_t v12[4];
  uint32_t v13[4];
  uint32_t v14[4];
  uint32_t v15[4];
  uint32_t v16[4];
  uint32_t v17[4];
  uint32_t v18[4];
  uint32_t v19[4];
  uint32_t v20[4];
  uint32_t v21[4];
  uint32_t v22[4];
  uint32_t v23[4];
  uint32_t v24[4];
  uint32_t v25[4];
  uint32_t v26[4];
  uint32_t v27[4];
  uint32_t v28[4];
  uint32_t v29[4];
  uint32_t v30[4];
  uint32_t v31[4];
  uint32_t pad[2];
  uint32_t vrsave;
  uint32_t vscr;
} VMX;

template <typename GP>
RegisterLayout Layout(const ArchSpec &arch, uint32_t value_bytes) {
  auto description = RegisterInfo_powerpc::CreateDescription(value_bytes);
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(GP)));
  llvm::cantFail(layout.AddBank(1, sizeof(FPR)));
  llvm::cantFail(layout.AddBank(2, sizeof(VMX)));
  auto add = [&](const char *name, RegisterBankID bank, uint32_t offset) {
    const auto *reg = description->Find(name);
    if (!reg)
      return;
    // A 32-bit value in a saved 64-bit word occupies its low half.
    uint32_t adjust = 0;
    if (bank == 0 && sizeof(((GP *)nullptr)->r0) > reg->byte_size &&
        arch.GetByteOrder() == lldb::eByteOrderBig)
      adjust = sizeof(((GP *)nullptr)->r0) - reg->byte_size;
    RegisterLocation location;
    location.kind = RegisterLocation::Stored;
    location.pieces.push_back({bank, offset + adjust, reg->byte_size});

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
  add("lr", 0, offsetof(GP, lr));
  add("cr", 0, offsetof(GP, cr));
  add("xer", 0, offsetof(GP, xer));
  add("ctr", 0, offsetof(GP, ctr));
  add("pc", 0, offsetof(GP, pc));
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
  add("v0", 2, offsetof(VMX, v0));
  add("v1", 2, offsetof(VMX, v1));
  add("v2", 2, offsetof(VMX, v2));
  add("v3", 2, offsetof(VMX, v3));
  add("v4", 2, offsetof(VMX, v4));
  add("v5", 2, offsetof(VMX, v5));
  add("v6", 2, offsetof(VMX, v6));
  add("v7", 2, offsetof(VMX, v7));
  add("v8", 2, offsetof(VMX, v8));
  add("v9", 2, offsetof(VMX, v9));
  add("v10", 2, offsetof(VMX, v10));
  add("v11", 2, offsetof(VMX, v11));
  add("v12", 2, offsetof(VMX, v12));
  add("v13", 2, offsetof(VMX, v13));
  add("v14", 2, offsetof(VMX, v14));
  add("v15", 2, offsetof(VMX, v15));
  add("v16", 2, offsetof(VMX, v16));
  add("v17", 2, offsetof(VMX, v17));
  add("v18", 2, offsetof(VMX, v18));
  add("v19", 2, offsetof(VMX, v19));
  add("v20", 2, offsetof(VMX, v20));
  add("v21", 2, offsetof(VMX, v21));
  add("v22", 2, offsetof(VMX, v22));
  add("v23", 2, offsetof(VMX, v23));
  add("v24", 2, offsetof(VMX, v24));
  add("v25", 2, offsetof(VMX, v25));
  add("v26", 2, offsetof(VMX, v26));
  add("v27", 2, offsetof(VMX, v27));
  add("v28", 2, offsetof(VMX, v28));
  add("v29", 2, offsetof(VMX, v29));
  add("v30", 2, offsetof(VMX, v30));
  add("v31", 2, offsetof(VMX, v31));
  add("vrsave", 2, offsetof(VMX, vrsave));
  add("vscr", 2, offsetof(VMX, vscr));
  return layout;
}
} // namespace

RegisterLayout
lldb_private::CreateFreeBSDPowerPC32RegisterLayout(const ArchSpec &arch) {
  return llvm::cantFail(RegisterLayout::Create(
      RegisterInfo_powerpc::CreateDescription(4), Layout<GPR32>(arch, 4)));
}
RegisterLayout
lldb_private::CreateFreeBSDPowerPC64RegisterLayout(const ArchSpec &arch) {
  const uint32_t value_bytes = arch.GetMachine() == llvm::Triple::ppc ? 4 : 8;
  return llvm::cantFail(RegisterLayout::Create(
      RegisterInfo_powerpc::CreateDescription(value_bytes),
      Layout<GPR64>(arch, value_bytes)));
}
