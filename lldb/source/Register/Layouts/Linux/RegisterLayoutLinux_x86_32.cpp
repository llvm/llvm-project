//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_32.h"
#include "lldb/Register/Descriptions/RegisterDescription_x86_32.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_x86_64.h"
#include "lldb/Utility/ArchSpec.h"

#include "lldb/Register/Arch/lldb-x86_32-register-enums.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>

namespace reg_ids = lldb_private::registers::x86_32::id;

using namespace lldb_private;

namespace {

struct GPR {
  uint32_t ebx;
  uint32_t ecx;
  uint32_t edx;
  uint32_t esi;
  uint32_t edi;
  uint32_t ebp;
  uint32_t eax;
  uint32_t ds;
  uint32_t es;
  uint32_t fs;
  uint32_t gs;
  uint32_t orig_eax;
  uint32_t eip;
  uint32_t cs;
  uint32_t eflags;
  uint32_t esp;
  uint32_t ss;
};

std::optional<RegisterPiece> GetGPRField(RegisterID id) {
  switch (id) {
  case reg_ids::ebx:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, ebx),
                         sizeof(GPR::ebx)};
  case reg_ids::ecx:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, ecx),
                         sizeof(GPR::ecx)};
  case reg_ids::edx:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, edx),
                         sizeof(GPR::edx)};
  case reg_ids::esi:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, esi),
                         sizeof(GPR::esi)};
  case reg_ids::edi:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, edi),
                         sizeof(GPR::edi)};
  case reg_ids::ebp:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, ebp),
                         sizeof(GPR::ebp)};
  case reg_ids::eax:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, eax),
                         sizeof(GPR::eax)};
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
  case reg_ids::eip:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, eip),
                         sizeof(GPR::eip)};
  case reg_ids::cs:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, cs),
                         sizeof(GPR::cs)};
  case reg_ids::eflags:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, eflags),
                         sizeof(GPR::eflags)};
  case reg_ids::esp:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, esp),
                         sizeof(GPR::esp)};
  case reg_ids::ss:
    return RegisterPiece{RegisterLayout_x86::GP, offsetof(GPR, ss),
                         sizeof(GPR::ss)};
  default:
    return std::nullopt;
  }
}

} // namespace

RegisterLayoutLinux_x86_32::RegisterLayoutLinux_x86_32(
    const ArchSpec &target_arch, TagWordView tag_word)
    : RegisterLayoutLinux_x86_32(target_arch, GPRMode::Native, tag_word) {}

RegisterLayoutLinux_x86_32::RegisterLayoutLinux_x86_32(
    const ArchSpec &target_arch, GPRMode mode, TagWordView tag_word) {
  const bool compat = mode == GPRMode::Compat64;
  const auto get_field = [compat](RegisterID id) {
    return compat ? RegisterLayoutLinux_x86_64::GetGPRField(GetX86_64GPR(id))
                  : GetGPRField(id);
  };
  const uint32_t gpr_size =
      compat ? RegisterLayoutLinux_x86_64::GetGPRSize() : sizeof(GPR);
  const uint32_t debug_slot_size = compat ? 8 : 4;
  auto description = std::make_shared<RegisterDescription_x86_32>(
      RegisterDescription_x86_32::Config{},
      GetGPRWidths(reg_ids::k_num_registers, get_field));

  llvm::cantFail(AddBank(GP, gpr_size));
  for (const auto &bank : GetHardwareBanks(debug_slot_size))
    llvm::cantFail(AddBank(bank.id, bank.byte_size));

  const auto get_location = [&](const RegisterDescriptor &reg) {
    return GetRegisterLocation(reg, get_field, false, debug_slot_size,
                               tag_word);
  };
  llvm::cantFail(AddRegisterLocations(*description, get_location));
  llvm::cantFail(Finalize(std::move(description), target_arch.GetByteOrder()));
}

uint32_t RegisterLayoutLinux_x86_32::GetOriginalSyscallNumberOffset() {
  return offsetof(GPR, orig_eax);
}
