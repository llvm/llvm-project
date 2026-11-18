//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"

#include <cstddef>

namespace reg_ids = lldb_private::registers::arm64::id;

using namespace lldb_private;

// FreeBSD's arm64 HWCAP ABI explicitly assigns the same bit positions as Linux:
// freebsd-src/sys/arm64/include/elf.h. Keep decoding at the OS layout boundary.
RegisterDescription_arm64::TypeFeatures
lldb_private::RegisterLayoutFreeBSD_arm64::DecodeRegisterFeatures(
    uint64_t hwcap, uint64_t hwcap2) {
  RegisterDescription_arm64::TypeFeatures features;
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

RegisterLayoutFreeBSD_arm64::RegisterLayoutFreeBSD_arm64(const ArchSpec &arch,
                                                         uint64_t hwcap,
                                                         uint64_t hwcap2) {
  auto features = DecodeRegisterFeatures(hwcap, hwcap2);
  auto description = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{0, 0, 0, &features});

  llvm::cantFail(AddBank(0, sizeof(RegisterLayoutFreeBSD_arm64::GPR)));
  llvm::cantFail(AddBank(1, sizeof(RegisterLayoutFreeBSD_arm64::FPR)));

  const auto get_location =
      [&](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
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
      offset = offsetof(RegisterLayoutFreeBSD_arm64::FPR, fpsr);
    else if (reg.id == reg_ids::fpu_fpcr)
      offset = offsetof(RegisterLayoutFreeBSD_arm64::FPR, fpcr);
    else {
      uint32_t i = reg.id >= reg_ids::fpu_d0   ? reg.id - reg_ids::fpu_d0
                   : reg.id >= reg_ids::fpu_s0 ? reg.id - reg_ids::fpu_s0
                                               : reg.id - reg_ids::fpu_v0;
      offset =
          i * 16 +
          (arch.GetByteOrder() == lldb::eByteOrderBig ? 16 - reg.byte_size : 0);
    }

    return RegisterLocation::StoredAt(bank, offset, reg.byte_size);
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));

  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
