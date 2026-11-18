//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cassert>
#include <iterator>

namespace reg_ids = lldb_private::registers::arm64::id;

using namespace lldb_private;

namespace {
using OSLayout = RegisterLayoutLinux_arm64;
using SVEFormat = OSLayout::SVEFormat;
} // namespace

RegisterDescription_arm64::TypeFeatures
lldb_private::RegisterLayoutLinux_arm64::DecodeRegisterFeatures(
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

namespace {

std::shared_ptr<const RegisterDescription>
LinuxDescription(Flags features, uint32_t sve_vq, uint32_t za_vq,
                 const RegisterDescription_arm64::TypeFeatures *type_features,
                 uint64_t hwcap3,
                 const RegisterDescription *previous = nullptr) {
  auto cpu = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{features, sve_vq, za_vq,
                                        type_features});
  std::vector<RegisterDescriptor> regs(cpu->GetRegisters().begin(),
                                       cpu->GetRegisters().end());
  std::vector<RegisterGroup> groups(cpu->GetGroups().begin(),
                                    cpu->GetGroups().end());
  RegisterDescription::OwnedTypes types(cpu->GetOwnedTypes().begin(),
                                        cpu->GetOwnedTypes().end());

  auto add = [&](RegisterID id, const char *name, const char *group_name,
                 const char *short_name, const RegisterType *type = nullptr) {
    RegisterDescriptor reg;
    reg.id = id;
    reg.name = name;
    reg.byte_size = 8;
    reg.encoding = lldb::eEncodingUint;
    reg.format = lldb::eFormatHex;
    reg.type = type;

    auto group = llvm::find_if(groups, [&](const auto &group) {
      return group.short_name == short_name;
    });
    if (group == groups.end()) {
      groups.push_back({group_name, short_name, {}});
      group = std::prev(groups.end());
    }

    group->registers.push_back(id);
    regs.push_back(std::move(reg));
  };

  if (features.AnySet(RegisterDescription_arm64::eRegsetMaskPAuth)) {
    add(0x10000, "data_mask", "Pointer Authentication Registers", "pauth");
    add(0x10001, "code_mask", "Pointer Authentication Registers", "pauth");
  }

  if (features.AnySet(RegisterDescription_arm64::eRegsetMaskMTE)) {
    const RegisterType *type = nullptr;

    if (type_features && type_features->mte) {
      auto tcf = std::make_shared<RegisterTypeEnum>(
          "tcf_enum", RegisterTypeEnum::Enumerators{{0, "TCF_NONE"},
                                                    {1, "TCF_SYNC"},
                                                    {2, "TCF_ASYNC"},
                                                    {3, "TCF_ASYMM"}});
      std::vector<RegisterTypeFlags::Field> fields;
      if (hwcap3 & (1ULL << 1))
        fields.push_back({"STORE_ONLY", 19});

      fields.insert(fields.end(), {{"TAGS", 3, 18},
                                   {"TCF", 1, 2, tcf.get()},
                                   {"TAGGED_ADDR_ENABLE", 0}});

      auto control =
          std::make_shared<RegisterTypeFlags>("mte_ctrl_flags", 8, fields);
      type = control.get();
      types.push_back(std::move(tcf));
      types.push_back(std::move(control));
    }

    add(0x10010, "mte_ctrl", "MTE Control Register", "mte", type);
  }

  if (features.AnySet(RegisterDescription_arm64::eRegsetMaskGCS)) {
    const RegisterType *type = nullptr;

    if (type_features && type_features->gcs) {
      auto control = std::make_shared<RegisterTypeFlags>(
          "gcs_features_flags", 8,
          std::vector<RegisterTypeFlags::Field>{
              {"PUSH", 2}, {"WRITE", 1}, {"ENABLE", 0}});
      type = control.get();
      types.push_back(std::move(control));
    }

    add(0x10050, "gcs_features_enabled", "Guarded Control Stack Registers",
        "gcs", type);
    add(0x10051, "gcs_features_locked", "Guarded Control Stack Registers",
        "gcs", type);
  }

  // Keep the established Linux presentation order while identities remain
  // stable.
  llvm::sort(regs, [](const auto &a, const auto &b) { return a.id < b.id; });

  for (size_t i = 0; i < regs.size(); ++i)
    regs[i].numbers[lldb::eRegisterKindLLDB] = i;

  for (auto &group : groups)
    llvm::sort(group.registers);

  llvm::sort(groups, [](const auto &a, const auto &b) {
    return a.registers.front() < b.registers.front();
  });

  if (previous) {
    // Vector-length changes preserve immutable scalar type nodes and their
    // UIDs.
    types.assign(previous->GetOwnedTypes().begin(),
                 previous->GetOwnedTypes().end());

    for (auto &reg : regs)
      if (const auto *old = previous->Find(reg.id))
        reg.type = old->type;
  }

  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(regs), std::move(groups), std::move(types))));
}

} // namespace

RegisterLayoutLinux_arm64::RegisterLayoutLinux_arm64(
    const ArchSpec &arch, const OSLayout::Config &config,
    const RegisterDescription *previous) {
  assert(RegisterDescription_arm64::Config::VectorSizeIsValid(config.sve_vq));
  assert(RegisterDescription_arm64::Config::VectorSizeIsValid(config.za_vq));

  auto type_features = DecodeRegisterFeatures(config.hwcap, config.hwcap2);
  auto description =
      LinuxDescription(config.features, config.sve_vq, config.za_vq,
                       &type_features, config.hwcap3, previous);
  const SVEState state =
      config.features.AnySet(RegisterDescription_arm64::eRegsetMaskSVE |
                             RegisterDescription_arm64::eRegsetMaskSSVE)
          ? config.sve_state
          : SVEState::Disabled;
  const uint32_t sve_vq = std::max(1u, config.sve_vq);

  const bool full = state == OSLayout::SVEState::Full ||
                    state == OSLayout::SVEState::Streaming;
  const bool sve_fp = state == OSLayout::SVEState::FPSIMD;

  for (auto entry :
       {RegisterBank{OSLayout::GP, sizeof(OSLayout::GPR)},
        {OSLayout::FP, sizeof(OSLayout::FPU)},
        {OSLayout::SVE,
         SVEFormat::PTraceSize(sve_vq, full ? SVEFormat::ptrace_regs_sve
                                            : SVEFormat::ptrace_regs_fpsimd)},
        {OSLayout::PAC, sizeof(OSLayout::PAuthMasks)},
        {OSLayout::MTE, sizeof(OSLayout::MTEControl)},
        {OSLayout::TLS, sizeof(OSLayout::TLSRegisters)},
        {OSLayout::SMEControl, sizeof(OSLayout::SMERegisters)},
        {OSLayout::ZT, sizeof(OSLayout::ZTRegister)},
        {OSLayout::FPMR, sizeof(OSLayout::FPMRRegister)},
        {OSLayout::GCS, sizeof(OSLayout::GCSRegisters)},
        {OSLayout::POE, sizeof(OSLayout::POERegister)}})
    llvm::cantFail(AddBank(entry.id, entry.byte_size));

  if (auto *za = description->Find("za"))
    llvm::cantFail(AddBank(OSLayout::ZA,
                           sizeof(SVEFormat::user_za_header) + za->byte_size));

  const auto get_location =
      [&](const RegisterDescriptor &reg) -> std::optional<RegisterLocation> {
    const auto stored = [&](RegisterBankID bank, uint32_t offset,
                            uint32_t size = 0) {
      auto location =
          RegisterLocation::StoredAt(bank, offset, size ? size : reg.byte_size);
      location.zero_fill = size && size < reg.byte_size;
      return location;
    };
    const auto zero = [&] {
      RegisterLocation loc;
      loc.kind = RegisterLocation::Constant;
      loc.constant.resize(reg.byte_size, 0);
      return loc;
    };

    if (reg.id <= reg_ids::gpr_cpsr)
      return stored(OSLayout::GP, reg.id * 8);
    if (reg.id <= reg_ids::gpr_w28)
      return stored(OSLayout::GP,
                    (reg.id - reg_ids::gpr_w0) * 8 +
                        (arch.GetByteOrder() == lldb::eByteOrderBig ? 4 : 0));
    if (reg.id <= reg_ids::fpu_fpcr) {
      RegisterBankID bank = full || sve_fp ? OSLayout::SVE : OSLayout::FP;
      uint32_t offset;
      if (reg.id == reg_ids::fpu_fpsr)
        offset = full ? SVEFormat::PTraceFPSROffset(sve_vq)
                      : (sve_fp ? 16 : 0) + 512;
      else if (reg.id == reg_ids::fpu_fpcr)
        offset = full ? SVEFormat::PTraceFPCROffset(sve_vq)
                      : (sve_fp ? 16 : 0) + 516;
      else {
        uint32_t i = reg.id >= reg_ids::fpu_d0   ? reg.id - reg_ids::fpu_d0
                     : reg.id >= reg_ids::fpu_s0 ? reg.id - reg_ids::fpu_s0
                                                 : reg.id - reg_ids::fpu_v0;
        offset = full ? SVEFormat::PTraceZRegOffset(sve_vq, i)
                      : (sve_fp ? 16 : 0) + i * 16;
        if (!full && arch.GetByteOrder() == lldb::eByteOrderBig)
          offset += 16 - reg.byte_size;
      }

      return stored(bank, offset);
    }
    if (reg.id == reg_ids::sve_vg) {
      RegisterLocation loc;
      loc.kind = RegisterLocation::Constant;
      loc.constant.resize(8, 0);

      uint64_t vg = state == OSLayout::SVEState::Disabled ? 0 : sve_vq * 2;

      for (uint32_t i = 0; i < 8; ++i)
        loc.constant[arch.GetByteOrder() == lldb::eByteOrderBig ? 7 - i : i] =
            vg >> (8 * i);

      return loc;
    }
    if (reg.id <= reg_ids::sve_z31) {
      uint32_t i = reg.id - reg_ids::sve_z0;
      return stored(full || sve_fp ? OSLayout::SVE : OSLayout::FP,
                    full ? SVEFormat::PTraceZRegOffset(sve_vq, i)
                         : (sve_fp ? 16 : 0) + i * 16,
                    full ? 0 : 16);
    }
    if (reg.id <= reg_ids::sve_ffr) {
      if (full)
        return stored(OSLayout::SVE,
                      reg.id == reg_ids::sve_ffr
                          ? SVEFormat::PTraceFFROffset(sve_vq)
                          : SVEFormat::PTracePRegOffset(
                                sve_vq, reg.id - reg_ids::sve_p0));
      return zero();
    }
    if (reg.name == "data_mask")
      return stored(OSLayout::PAC, 0);
    if (reg.name == "code_mask")
      return stored(OSLayout::PAC, 8);
    if (reg.name == "mte_ctrl")
      return stored(OSLayout::MTE, 0);
    if (reg.name == "tpidr")
      return stored(OSLayout::TLS, 0);
    if (reg.name == "tpidr2")
      return stored(OSLayout::TLS, 8);
    if (reg.name == "svcr")
      return stored(OSLayout::SMEControl, 0);
    if (reg.name == "svg")
      return stored(OSLayout::SMEControl, 8);
    if (reg.name == "za")
      return stored(OSLayout::ZA, sizeof(SVEFormat::user_za_header));
    if (reg.name == "zt0")
      return stored(OSLayout::ZT, 0);
    if (reg.name == "fpmr")
      return stored(OSLayout::FPMR, 0);
    if (reg.name == "gcs_features_enabled")
      return stored(OSLayout::GCS, 0);
    if (reg.name == "gcs_features_locked")
      return stored(OSLayout::GCS, 8);
    if (reg.name == "gcspr_el0")
      return stored(OSLayout::GCS, 16);
    if (reg.name == "por_el0")
      return stored(OSLayout::POE, 0);

    return std::nullopt;
  };

  llvm::cantFail(AddRegisterLocations(*description, get_location));

  llvm::cantFail(Finalize(std::move(description), arch.GetByteOrder()));
}
