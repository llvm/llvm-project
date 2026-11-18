//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/NetBSD/RegisterLayoutNetBSD_x86_64.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "lldb/Utility/ArchSpec.h"

#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>

namespace reg_ids = lldb_private::registers::x86_64::id;

using namespace lldb_private;

namespace {

struct GPR {
  uint64_t rdi;    /*  0 */
  uint64_t rsi;    /*  1 */
  uint64_t rdx;    /*  2 */
  uint64_t rcx;    /*  3 */
  uint64_t r8;     /*  4 */
  uint64_t r9;     /*  5 */
  uint64_t r10;    /*  6 */
  uint64_t r11;    /*  7 */
  uint64_t r12;    /*  8 */
  uint64_t r13;    /*  9 */
  uint64_t r14;    /* 10 */
  uint64_t r15;    /* 11 */
  uint64_t rbp;    /* 12 */
  uint64_t rbx;    /* 13 */
  uint64_t rax;    /* 14 */
  uint64_t gs;     /* 15 */
  uint64_t fs;     /* 16 */
  uint64_t es;     /* 17 */
  uint64_t ds;     /* 18 */
  uint64_t trapno; /* 19 */
  uint64_t err;    /* 20 */
  uint64_t rip;    /* 21 */
  uint64_t cs;     /* 22 */
  uint64_t rflags; /* 23 */
  uint64_t rsp;    /* 24 */
  uint64_t ss;     /* 25 */
};

} // namespace

std::optional<RegisterPiece>
RegisterLayoutNetBSD_x86_64::GetGPRField(RegisterID id) {
  switch (id) {
  case reg_ids::rdi:
    return RegisterPiece{GP, offsetof(GPR, rdi), sizeof(GPR::rdi)};
  case reg_ids::rsi:
    return RegisterPiece{GP, offsetof(GPR, rsi), sizeof(GPR::rsi)};
  case reg_ids::rdx:
    return RegisterPiece{GP, offsetof(GPR, rdx), sizeof(GPR::rdx)};
  case reg_ids::rcx:
    return RegisterPiece{GP, offsetof(GPR, rcx), sizeof(GPR::rcx)};
  case reg_ids::r8:
    return RegisterPiece{GP, offsetof(GPR, r8), sizeof(GPR::r8)};
  case reg_ids::r9:
    return RegisterPiece{GP, offsetof(GPR, r9), sizeof(GPR::r9)};
  case reg_ids::r10:
    return RegisterPiece{GP, offsetof(GPR, r10), sizeof(GPR::r10)};
  case reg_ids::r11:
    return RegisterPiece{GP, offsetof(GPR, r11), sizeof(GPR::r11)};
  case reg_ids::r12:
    return RegisterPiece{GP, offsetof(GPR, r12), sizeof(GPR::r12)};
  case reg_ids::r13:
    return RegisterPiece{GP, offsetof(GPR, r13), sizeof(GPR::r13)};
  case reg_ids::r14:
    return RegisterPiece{GP, offsetof(GPR, r14), sizeof(GPR::r14)};
  case reg_ids::r15:
    return RegisterPiece{GP, offsetof(GPR, r15), sizeof(GPR::r15)};
  case reg_ids::rbp:
    return RegisterPiece{GP, offsetof(GPR, rbp), sizeof(GPR::rbp)};
  case reg_ids::rbx:
    return RegisterPiece{GP, offsetof(GPR, rbx), sizeof(GPR::rbx)};
  case reg_ids::rax:
    return RegisterPiece{GP, offsetof(GPR, rax), sizeof(GPR::rax)};
  case reg_ids::gs:
    return RegisterPiece{GP, offsetof(GPR, gs), sizeof(GPR::gs)};
  case reg_ids::fs:
    return RegisterPiece{GP, offsetof(GPR, fs), sizeof(GPR::fs)};
  case reg_ids::es:
    return RegisterPiece{GP, offsetof(GPR, es), sizeof(GPR::es)};
  case reg_ids::ds:
    return RegisterPiece{GP, offsetof(GPR, ds), sizeof(GPR::ds)};
  case reg_ids::rip:
    return RegisterPiece{GP, offsetof(GPR, rip), sizeof(GPR::rip)};
  case reg_ids::cs:
    return RegisterPiece{GP, offsetof(GPR, cs), sizeof(GPR::cs)};
  case reg_ids::rflags:
    return RegisterPiece{GP, offsetof(GPR, rflags), sizeof(GPR::rflags)};
  case reg_ids::rsp:
    return RegisterPiece{GP, offsetof(GPR, rsp), sizeof(GPR::rsp)};
  case reg_ids::ss:
    return RegisterPiece{GP, offsetof(GPR, ss), sizeof(GPR::ss)};
  default:
    return std::nullopt;
  }
}

RegisterLayoutNetBSD_x86_64::RegisterLayoutNetBSD_x86_64(
    const ArchSpec &target_arch, TagWordView tag_word) {
  const uint32_t debug_slot_size = 8;
  auto description = std::make_shared<RegisterDescription_x86_64>(
      RegisterDescription_x86_64::Config{},
      GetGPRWidths(reg_ids::k_num_registers, GetGPRField));

  llvm::cantFail(AddBank(GP, sizeof(GPR)));
  for (const auto &bank : GetHardwareBanks(debug_slot_size))
    llvm::cantFail(AddBank(bank.id, bank.byte_size));

  const auto get_location = [&](const RegisterDescriptor &reg) {
    return GetRegisterLocation(reg, GetGPRField, true, debug_slot_size,
                               tag_word);
  };
  llvm::cantFail(AddRegisterLocations(*description, get_location));
  llvm::cantFail(Finalize(std::move(description), target_arch.GetByteOrder()));
}

uint32_t RegisterLayoutNetBSD_x86_64::GetGPRSize() { return sizeof(GPR); }
