//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterLayoutOpenBSD_x86_64.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_64.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
namespace reg_ids = lldb_private::registers::x86_64::id;

using namespace lldb_private;
namespace {

typedef struct _GPR {
  uint64_t rdi;
  uint64_t rsi;
  uint64_t rdx;
  uint64_t rcx;
  uint64_t r8;
  uint64_t r9;
  uint64_t r10;
  uint64_t r11;
  uint64_t r12;
  uint64_t r13;
  uint64_t r14;
  uint64_t r15;
  uint64_t rbp;
  uint64_t rbx;
  uint64_t rax;
  uint64_t rsp;
  uint64_t rip;
  uint64_t rflags;
  uint64_t cs;
  uint64_t ss;
  uint64_t ds;
  uint64_t es;
  uint64_t fs;
  uint64_t gs;
} GPR;
std::optional<RegisterPiece> GetGPRField(RegisterID id) {
  switch (id) {
  case reg_ids::rdi:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rdi),
                         sizeof(GPR::rdi)};
  case reg_ids::rsi:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rsi),
                         sizeof(GPR::rsi)};
  case reg_ids::rdx:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rdx),
                         sizeof(GPR::rdx)};
  case reg_ids::rcx:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rcx),
                         sizeof(GPR::rcx)};
  case reg_ids::r8:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r8),
                         sizeof(GPR::r8)};
  case reg_ids::r9:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r9),
                         sizeof(GPR::r9)};
  case reg_ids::r10:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r10),
                         sizeof(GPR::r10)};
  case reg_ids::r11:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r11),
                         sizeof(GPR::r11)};
  case reg_ids::r12:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r12),
                         sizeof(GPR::r12)};
  case reg_ids::r13:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r13),
                         sizeof(GPR::r13)};
  case reg_ids::r14:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r14),
                         sizeof(GPR::r14)};
  case reg_ids::r15:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, r15),
                         sizeof(GPR::r15)};
  case reg_ids::rbp:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rbp),
                         sizeof(GPR::rbp)};
  case reg_ids::rbx:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rbx),
                         sizeof(GPR::rbx)};
  case reg_ids::rax:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rax),
                         sizeof(GPR::rax)};
  case reg_ids::rsp:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rsp),
                         sizeof(GPR::rsp)};
  case reg_ids::rip:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rip),
                         sizeof(GPR::rip)};
  case reg_ids::rflags:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, rflags),
                         sizeof(GPR::rflags)};
  case reg_ids::cs:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, cs),
                         sizeof(GPR::cs)};
  case reg_ids::ss:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, ss),
                         sizeof(GPR::ss)};
  case reg_ids::ds:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, ds),
                         sizeof(GPR::ds)};
  case reg_ids::es:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, es),
                         sizeof(GPR::es)};
  case reg_ids::fs:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, fs),
                         sizeof(GPR::fs)};
  case reg_ids::gs:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, gs),
                         sizeof(GPR::gs)};
  default:
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutOpenBSD_x86_64::RegisterLayoutOpenBSD_x86_64(
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
