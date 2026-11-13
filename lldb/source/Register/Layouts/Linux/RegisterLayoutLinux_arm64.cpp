//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>
#include <iterator>

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;
using namespace lldb_private::linux_arm64;
Arm64RegisterFeatures
lldb_private::DecodeLinuxArm64RegisterFeatures(uint64_t hwcap,
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

namespace {
std::shared_ptr<const RegisterDescription>
LinuxDescription(Flags features, uint32_t sve_vq, uint32_t za_vq,
                 const Arm64RegisterFeatures *type_features, uint64_t hwcap3,
                 const RegisterDescription *previous = nullptr) {
  auto cpu = RegisterInfo_arm64::CreateDescription(features, sve_vq, za_vq,
                                                   type_features);
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
  if (features.AnySet(RegisterInfo_arm64::eRegsetMaskPAuth)) {
    add(0x10000, "data_mask", "Pointer Authentication Registers", "pauth");
    add(0x10001, "code_mask", "Pointer Authentication Registers", "pauth");
  }
  if (features.AnySet(RegisterInfo_arm64::eRegsetMaskMTE)) {
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
  if (features.AnySet(RegisterInfo_arm64::eRegsetMaskGCS)) {
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
static RegisterLayout CreateLayout(const ArchSpec &arch,
                                   const RegisterDescription &description,
                                   SVEState state, uint32_t sve_vq) {
  RegisterLayout layout(arch.GetByteOrder());
  sve_vq = std::max(1u, sve_vq);
  const bool full = state == SVEState::Full || state == SVEState::Streaming;
  const bool sve_fp = state == SVEState::FPSIMD;
  for (auto entry :
       {RegisterBank{GP, sizeof(GPR)},
        {FP, sizeof(FPU)},
        {SVE, sve::PTraceSize(sve_vq, full ? sve::ptrace_regs_sve
                                           : sve::ptrace_regs_fpsimd)},
        {PAC, sizeof(PAuthMasks)},
        {MTE, sizeof(MTEControl)},
        {TLS, sizeof(TLSRegisters)},
        {SMEControl, sizeof(SMERegisters)},
        {ZT, sizeof(ZTRegister)},
        {FPMR, sizeof(FPMRRegister)},
        {GCS, sizeof(GCSRegisters)},
        {POE, sizeof(POERegister)}})
    llvm::cantFail(layout.AddBank(entry.id, entry.byte_size));
  if (auto *za = description.Find("za"))
    llvm::cantFail(
        layout.AddBank(ZA, sizeof(sve::user_za_header) + za->byte_size));
  auto add = [&](const RegisterDescriptor &reg, RegisterBankID bank,
                 uint32_t offset, uint32_t size = 0) {
    RegisterLocation loc;
    loc.kind = RegisterLocation::Stored;
    loc.pieces.push_back({bank, offset, size ? size : reg.byte_size});
    loc.zero_fill = size && size < reg.byte_size;
    llvm::cantFail(layout.Add(reg.id, std::move(loc)));
  };
  auto zero = [&](const RegisterDescriptor &reg) {
    RegisterLocation loc;
    loc.kind = RegisterLocation::Constant;
    loc.constant.resize(reg.byte_size, 0);
    llvm::cantFail(layout.Add(reg.id, std::move(loc)));
  };
  for (const auto &reg : description.GetRegisters()) {
    if (reg.id <= reg_ids::gpr_cpsr)
      add(reg, GP, reg.id * 8);
    else if (reg.id <= reg_ids::gpr_w28)
      add(reg, GP,
          (reg.id - reg_ids::gpr_w0) * 8 +
              (arch.GetByteOrder() == lldb::eByteOrderBig ? 4 : 0));
    else if (reg.id <= reg_ids::fpu_fpcr) {
      RegisterBankID bank = full || sve_fp ? SVE : FP;
      uint32_t offset;
      if (reg.id == reg_ids::fpu_fpsr)
        offset = full ? sve::PTraceFPSROffset(sve_vq) : (sve_fp ? 16 : 0) + 512;
      else if (reg.id == reg_ids::fpu_fpcr)
        offset = full ? sve::PTraceFPCROffset(sve_vq) : (sve_fp ? 16 : 0) + 516;
      else {
        uint32_t i = reg.id >= reg_ids::fpu_d0   ? reg.id - reg_ids::fpu_d0
                     : reg.id >= reg_ids::fpu_s0 ? reg.id - reg_ids::fpu_s0
                                                 : reg.id - reg_ids::fpu_v0;
        offset = full ? sve::PTraceZRegOffset(sve_vq, i)
                      : (sve_fp ? 16 : 0) + i * 16;
        if (!full && arch.GetByteOrder() == lldb::eByteOrderBig)
          offset += 16 - reg.byte_size;
      }
      add(reg, bank, offset);
    } else if (reg.id == reg_ids::sve_vg) {
      RegisterLocation loc;
      loc.kind = RegisterLocation::Constant;
      loc.constant.resize(8, 0);
      uint64_t vg = state == SVEState::Disabled ? 0 : sve_vq * 2;
      for (uint32_t i = 0; i < 8; ++i)
        loc.constant[arch.GetByteOrder() == lldb::eByteOrderBig ? 7 - i : i] =
            vg >> (8 * i);
      llvm::cantFail(layout.Add(reg.id, std::move(loc)));
    } else if (reg.id <= reg_ids::sve_z31) {
      uint32_t i = reg.id - reg_ids::sve_z0;
      add(reg, full || sve_fp ? SVE : FP,
          full ? sve::PTraceZRegOffset(sve_vq, i) : (sve_fp ? 16 : 0) + i * 16,
          full ? 0 : 16);
    } else if (reg.id <= reg_ids::sve_ffr) {
      if (full)
        add(reg, SVE,
            reg.id == reg_ids::sve_ffr
                ? sve::PTraceFFROffset(sve_vq)
                : sve::PTracePRegOffset(sve_vq, reg.id - reg_ids::sve_p0));
      else
        zero(reg);
    } else if (reg.name == "data_mask")
      add(reg, PAC, 0);
    else if (reg.name == "code_mask")
      add(reg, PAC, 8);
    else if (reg.name == "mte_ctrl")
      add(reg, MTE, 0);
    else if (reg.name == "tpidr")
      add(reg, TLS, 0);
    else if (reg.name == "tpidr2")
      add(reg, TLS, 8);
    else if (reg.name == "svcr")
      add(reg, SMEControl, 0);
    else if (reg.name == "svg")
      add(reg, SMEControl, 8);
    else if (reg.name == "za")
      add(reg, ZA, sizeof(sve::user_za_header));
    else if (reg.name == "zt0")
      add(reg, ZT, 0);
    else if (reg.name == "fpmr")
      add(reg, FPMR, 0);
    else if (reg.name == "gcs_features_enabled")
      add(reg, GCS, 0);
    else if (reg.name == "gcs_features_locked")
      add(reg, GCS, 8);
    else if (reg.name == "gcspr_el0")
      add(reg, GCS, 16);
    else if (reg.name == "por_el0")
      add(reg, POE, 0);
  }
  return layout;
}
RegisterLayout lldb_private::CreateLinuxArm64RegisterLayout(
    const ArchSpec &arch, const linux_arm64::RegisterConfig &config,
    const RegisterDescription *previous) {
  assert(RegisterInfo_arm64::VectorSizeIsValid(config.sve_vq));
  assert(RegisterInfo_arm64::VectorSizeIsValid(config.za_vq));
  auto type_features =
      DecodeLinuxArm64RegisterFeatures(config.hwcap, config.hwcap2);
  auto description =
      LinuxDescription(config.features, config.sve_vq, config.za_vq,
                       &type_features, config.hwcap3, previous);
  const SVEState state =
      config.features.AnySet(RegisterInfo_arm64::eRegsetMaskSVE |
                             RegisterInfo_arm64::eRegsetMaskSSVE)
          ? config.sve_state
          : SVEState::Disabled;
  auto layout = CreateLayout(arch, *description, state, config.sve_vq);
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}
