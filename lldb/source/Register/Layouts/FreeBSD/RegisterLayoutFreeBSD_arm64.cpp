//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include <cstddef>

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;
static RegisterLayout CreateLayout(const ArchSpec &arch) {
  auto description = RegisterInfo_arm64::CreateDescription(0);
  RegisterLayout layout(arch.GetByteOrder());
  llvm::cantFail(layout.AddBank(0, sizeof(freebsd_arm64::GPR)));
  llvm::cantFail(layout.AddBank(1, sizeof(freebsd_arm64::FPR)));
  for (const auto &reg : description->GetRegisters()) {
    RegisterBankID bank = reg.id <= reg_ids::gpr_w28 ? 0 : 1;
    uint32_t offset;
    if (reg.id <= reg_ids::gpr_cpsr)
      offset = reg.id * 8 + (reg.id == reg_ids::gpr_cpsr &&
                                     arch.GetByteOrder() == lldb::eByteOrderBig
                                 ? 4
                                 : 0);
    else if (reg.id <= reg_ids::gpr_w28)
      offset = (reg.id - reg_ids::gpr_w0) * 8 +
               (arch.GetByteOrder() == lldb::eByteOrderBig ? 4 : 0);
    else if (reg.id == reg_ids::fpu_fpsr)
      offset = offsetof(freebsd_arm64::FPR, fpsr);
    else if (reg.id == reg_ids::fpu_fpcr)
      offset = offsetof(freebsd_arm64::FPR, fpcr);
    else {
      uint32_t i = reg.id >= reg_ids::fpu_d0   ? reg.id - reg_ids::fpu_d0
                   : reg.id >= reg_ids::fpu_s0 ? reg.id - reg_ids::fpu_s0
                                               : reg.id - reg_ids::fpu_v0;
      offset =
          i * 16 +
          (arch.GetByteOrder() == lldb::eByteOrderBig ? 16 - reg.byte_size : 0);
    }
    RegisterLocation loc;
    loc.kind = RegisterLocation::Stored;
    loc.pieces.push_back({bank, offset, reg.byte_size});
    llvm::cantFail(layout.Add(reg.id, std::move(loc)));
  }
  return layout;
}

// FreeBSD's arm64 HWCAP ABI explicitly assigns the same bit positions as Linux:
// freebsd-src/sys/arm64/include/elf.h. Keep decoding at the OS layout boundary.
Arm64RegisterFeatures
lldb_private::DecodeFreeBSDArm64RegisterFeatures(uint64_t hwcap,
                                                 uint64_t hwcap2) {
  Arm64RegisterFeatures features;
  features.fp16 = (hwcap & (1ULL << 9)) && (hwcap & (1ULL << 10));
  features.dit = hwcap & (1ULL << 24);
  features.ssbs = hwcap & (1ULL << 28);
  features.gcs = hwcap & (1ULL << 32);
  features.bti = hwcap2 & (1ULL << 17);
  features.mte = hwcap2 & (1ULL << 18);
  features.afp = hwcap2 & (1ULL << 20);
  features.sme = hwcap2 & (1ULL << 23);
  features.ebf16 = hwcap2 & (1ULL << 32);
  features.fpmr = hwcap2 & (1ULL << 48);
  features.poe = hwcap2 & (1ULL << 63);
  return features;
}

RegisterLayout lldb_private::CreateFreeBSDArm64RegisterLayout(
    const ArchSpec &arch, uint64_t hwcap, uint64_t hwcap2) {
  auto features = DecodeFreeBSDArm64RegisterFeatures(hwcap, hwcap2);
  return llvm::cantFail(RegisterLayout::Create(
      RegisterInfo_arm64::CreateDescription(0, 0, 0, &features),
      CreateLayout(arch)));
}
