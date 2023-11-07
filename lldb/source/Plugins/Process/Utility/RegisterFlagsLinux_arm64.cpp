//===-- RegisterFlagsLinux_arm64.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RegisterFlagsLinux_arm64.h"
#include "lldb/lldb-private-types.h"

// This file is built on all systems because it is used by native processes and
// core files, so we manually define the needed HWCAP values here.

#define HWCAP_DIT (1 << 24)
#define HWCAP_SSBS (1 << 28)

#define HWCAP2_BTI (1 << 17)
#define HWCAP2_MTE (1 << 18)

using namespace lldb_private;

LinuxArm64RegisterFlags::Fields
LinuxArm64RegisterFlags::DetectCPSRFields(uint64_t hwcap, uint64_t hwcap2) {
  // The fields here are a combination of the Arm manual's SPSR_EL1,
  // plus a few changes where Linux has decided not to make use of them at all,
  // or at least not from userspace.

  // Status bits that are always present.
  std::vector<RegisterFlags::Field> cpsr_fields{
      {"N", 31}, {"Z", 30}, {"C", 29}, {"V", 28},
      // Bits 27-26 reserved.
  };

  if (hwcap2 & HWCAP2_MTE)
    cpsr_fields.push_back({"TCO", 25});
  if (hwcap & HWCAP_DIT)
    cpsr_fields.push_back({"DIT", 24});

  // UAO and PAN are bits 23 and 22 and have no meaning for userspace so
  // are treated as reserved by the kernel.

  cpsr_fields.push_back({"SS", 21});
  cpsr_fields.push_back({"IL", 20});
  // Bits 19-14 reserved.

  // Bit 13, ALLINT, requires FEAT_NMI that isn't relevant to userspace, and we
  // can't detect either, don't show this field.
  if (hwcap & HWCAP_SSBS)
    cpsr_fields.push_back({"SSBS", 12});
  if (hwcap2 & HWCAP2_BTI)
    cpsr_fields.push_back({"BTYPE", 10, 11});

  cpsr_fields.push_back({"D", 9});
  cpsr_fields.push_back({"A", 8});
  cpsr_fields.push_back({"I", 7});
  cpsr_fields.push_back({"F", 6});
  // Bit 5 reserved
  // Called "M" in the ARMARM.
  cpsr_fields.push_back({"nRW", 4});
  // This is a 4 bit field M[3:0] in the ARMARM, we split it into parts.
  cpsr_fields.push_back({"EL", 2, 3});
  // Bit 1 is unused and expected to be 0.
  cpsr_fields.push_back({"SP", 0});

  return cpsr_fields;
}

void LinuxArm64RegisterFlags::DetectFields(uint64_t hwcap, uint64_t hwcap2) {
  for (auto &reg : m_registers)
    reg.m_flags.SetFields(reg.m_detector(hwcap, hwcap2));
  m_has_detected = true;
}

void LinuxArm64RegisterFlags::UpdateRegisterInfo(const RegisterInfo *reg_info,
                                                 uint32_t num_regs) {
  assert(m_has_detected &&
         "Must call DetectFields before updating register info.");

  // Register names will not be duplicated, so we do not want to compare against
  // one if it has already been found. Each time we find one, we erase it from
  // this list.
  std::vector<std::pair<llvm::StringRef, const RegisterFlags *>> search_registers;
  for (const auto &reg : m_registers) {
    // It is possible that a register is all extension dependent fields, and
    // none of them are present.
    if (reg.m_flags.GetFields().size())
      search_registers.push_back({reg.m_name, &reg.m_flags});
  }

  // Walk register information while there are registers we know need
  // to be updated. Example:
  // Register information: [a, b, c, d]
  // To be patched: [b, c]
  // * a != b, a != c, do nothing and move on.
  // * b == b, patch b, new patch list is [c], move on.
  // * c == c, patch c, patch list is empty, exit early without looking at d.
  for (uint32_t idx = 0; idx < num_regs && search_registers.size();
       ++idx, ++reg_info) {
    auto end = search_registers.cend();
    for (auto it = search_registers.cbegin(); it != end; ++it) {
      // If this register is one that must be patched.
      if (reg_info->name == it->first) {
        // Attach the field information.
        reg_info->flags_type = it->second;
        // We do not expect to see this name again, so don't look for it.
        search_registers.erase(it);
        break;
      }
    }
  }
}