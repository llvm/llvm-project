//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_ppc.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc32be.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64be.h"
#include "lldb/Register/Descriptions/RegisterDescription_ppc64le.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>

using namespace lldb_private;

namespace {

// Mirrors FreeBSD's machine/reg.h; native SDK checks remain in native code.
template <typename Word> struct GPR {
  Word r0;
  Word r1;
  Word r2;
  Word r3;
  Word r4;
  Word r5;
  Word r6;
  Word r7;
  Word r8;
  Word r9;
  Word r10;
  Word r11;
  Word r12;
  Word r13;
  Word r14;
  Word r15;
  Word r16;
  Word r17;
  Word r18;
  Word r19;
  Word r20;
  Word r21;
  Word r22;
  Word r23;
  Word r24;
  Word r25;
  Word r26;
  Word r27;
  Word r28;
  Word r29;
  Word r30;
  Word r31;
  Word lr;
  Word cr;
  Word xer;
  Word ctr;
  Word pc;
};

using GPR32 = GPR<uint32_t>;
using GPR64 = GPR<uint64_t>;

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
std::optional<RegisterLocation>
GetRegisterLocation(const RegisterDescriptor &reg, lldb::ByteOrder byte_order,
                    llvm::StringRef vector_prefix, bool has_vsx) {
  const auto index = [&](llvm::StringRef prefix,
                         uint32_t count) -> std::optional<uint32_t> {
    llvm::StringRef name = reg.name;
    uint32_t number;
    if (name.consume_front(prefix) && !name.getAsInteger(10, number) &&
        number < count)
      return number;
    return std::nullopt;
  };
  const auto gpr = [&](uint64_t offset) {
    // A 32-bit value in a saved 64-bit word occupies its low half.
    if (sizeof(GP::r0) > reg.byte_size && byte_order == lldb::eByteOrderBig)
      offset += sizeof(GP::r0) - reg.byte_size;
    return RegisterLocation::StoredAt(0, offset, reg.byte_size);
  };

  if (auto i = index("r", 32))
    return gpr(offsetof(GP, r0) + *i * sizeof(GP::r0));
  if (auto offset = llvm::StringSwitch<std::optional<uint64_t>>(reg.name)
                        .Case("lr", offsetof(GP, lr))
                        .Case("cr", offsetof(GP, cr))
                        .Case("xer", offsetof(GP, xer))
                        .Case("ctr", offsetof(GP, ctr))
                        .Case("pc", offsetof(GP, pc))
                        .Default(std::nullopt))
    return gpr(*offset);

  if (auto i = index("f", 32))
    return RegisterLocation::StoredAt(
        1, offsetof(FPR, f0) + *i * sizeof(FPR::f0), reg.byte_size);
  if (reg.name == "fpscr")
    return RegisterLocation::StoredAt(1, offsetof(FPR, fpscr), reg.byte_size);
  if (auto i = index(vector_prefix, 32))
    return RegisterLocation::StoredAt(
        2, offsetof(VMX, v0) + *i * sizeof(VMX::v0), reg.byte_size);
  if (reg.name == "vrsave")
    return RegisterLocation::StoredAt(2, offsetof(VMX, vrsave), reg.byte_size);
  if (reg.name == "vscr")
    return RegisterLocation::StoredAt(2, offsetof(VMX, vscr), reg.byte_size);

  const auto vsr = index("vs", 64);
  if (!has_vsx || !vsr)
    return std::nullopt;

  // PT_GETVSRREGS contains only doubleword 1. Doubleword 0 is in
  // PT_GETFPREGS, and VSR32-63 are the complete VMX registers.
  if (*vsr >= 32)
    return RegisterLocation::StoredAt(2, (*vsr - 32) * 16, 16);

  const bool big = byte_order == lldb::eByteOrderBig;
  RegisterLocation location;
  location.kind = RegisterLocation::Stored;
  location.pieces = {{1, *vsr * 8, 8, big ? 0u : 8u},
                     {3, *vsr * 8, 8, big ? 8u : 0u}};
  return location;
}

} // namespace

RegisterLayoutFreeBSD_ppc::RegisterLayoutFreeBSD_ppc(
    const ArchSpec &arch, bool has_vsx, std::optional<uint32_t> gpr_slot_size) {
  const auto slot_size = gpr_slot_size.value_or(arch.GetAddressByteSize());
  std::shared_ptr<const RegisterDescription> description;
  llvm::StringRef vector_prefix;

  switch (arch.GetMachine()) {
  case llvm::Triple::ppc:
    assert(slot_size == 4 || slot_size == 8);
    description = RegisterDescription_ppc32be::GetShared();
    vector_prefix = "v";
    break;
  case llvm::Triple::ppc64:
    assert(slot_size == 8);
    description = RegisterDescription_ppc64be::GetShared(
        RegisterDescription_ppc64be::Numbering::ELF);
    vector_prefix = "vr";
    break;
  case llvm::Triple::ppc64le:
    assert(slot_size == 8);
    description = RegisterDescription_ppc64le::GetShared();
    vector_prefix = "vr";
    break;
  default:
    llvm_unreachable("unsupported FreeBSD PPC architecture");
  }

  llvm::cantFail(AddBank(0, slot_size == 4 ? sizeof(GPR32) : sizeof(GPR64)));
  llvm::cantFail(AddBank(1, sizeof(FPR)));
  llvm::cantFail(AddBank(2, sizeof(VMX)));
  if (has_vsx)
    llvm::cantFail(AddBank(3, sizeof(VSX)));

  const auto get_location = [&](const RegisterDescriptor &reg) {
    if (slot_size == 4)
      return GetRegisterLocation<GPR32>(reg, arch.GetByteOrder(), vector_prefix,
                                        has_vsx);
    return GetRegisterLocation<GPR64>(reg, arch.GetByteOrder(), vector_prefix,
                                      has_vsx);
  };
  llvm::cantFail(AddRegisterLocations(*description, get_location));
  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
