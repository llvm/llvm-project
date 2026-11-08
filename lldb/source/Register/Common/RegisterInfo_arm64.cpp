//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Common/RegisterInfo_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;
namespace {
struct RegisterTypes {
  RegisterDescription::OwnedTypes owned;
  template <typename T, typename... Args> const T *MakeType(Args &&...args) {
    auto type = std::make_shared<T>(std::forward<Args>(args)...);
    const T *result = type.get();
    owned.push_back(std::move(type));
    return result;
  }
  const RegisterType *BuildPOREL0Type(const Arm64RegisterFeatures &features);
  const RegisterType *BuildFPMRType(const Arm64RegisterFeatures &features);
  const RegisterType *BuildSVCRType(const Arm64RegisterFeatures &features);
  const RegisterType *BuildFPCRType(const Arm64RegisterFeatures &features);
  const RegisterType *BuildFPSRType(const Arm64RegisterFeatures &features);
  const RegisterType *BuildCPSRType(const Arm64RegisterFeatures &features);
};
const RegisterType *
RegisterTypes::BuildPOREL0Type(const Arm64RegisterFeatures &features) {

  if (!features.poe)
    return nullptr;

  const RegisterTypeEnum *por_el0_perm_enum = MakeType<RegisterTypeEnum>(
      "por_el0_perm_enum", RegisterTypeEnum::Enumerators{
                               {0b0000, "No Access"},
                               {0b0001, "Read"},
                               {0b0010, "Execute"},
                               {0b0011, "Read, Execute"},
                               {0b0100, "Write"},
                               {0b0101, "Write, Read"},
                               {0b0110, "Write, Execute"},
                               {0b0111, "Read, Write, Execute"},
                           });

  return MakeType<RegisterTypeFlags>("por_el0_flags", 8,
                                     std::vector<RegisterTypeFlags::Field>{
                                         {"Perm15", 60, 63, por_el0_perm_enum},
                                         {"Perm14", 56, 59, por_el0_perm_enum},
                                         {"Perm13", 52, 55, por_el0_perm_enum},
                                         {"Perm12", 48, 51, por_el0_perm_enum},
                                         {"Perm11", 44, 47, por_el0_perm_enum},
                                         {"Perm10", 40, 43, por_el0_perm_enum},
                                         {"Perm9", 36, 39, por_el0_perm_enum},
                                         {"Perm8", 32, 35, por_el0_perm_enum},
                                         {"Perm7", 28, 31, por_el0_perm_enum},
                                         {"Perm6", 24, 27, por_el0_perm_enum},
                                         {"Perm5", 20, 23, por_el0_perm_enum},
                                         {"Perm4", 16, 19, por_el0_perm_enum},
                                         {"Perm3", 12, 15, por_el0_perm_enum},
                                         {"Perm2", 8, 11, por_el0_perm_enum},
                                         {"Perm1", 4, 7, por_el0_perm_enum},
                                         {"Perm0", 0, 3, por_el0_perm_enum},
                                     });
}

const RegisterType *
RegisterTypes::BuildFPMRType(const Arm64RegisterFeatures &features) {

  if (!features.fpmr)
    return nullptr;

  const RegisterTypeEnum *fp8_format_enum = MakeType<RegisterTypeEnum>(
      "fp8_format_enum", RegisterTypeEnum::Enumerators{
                             {0, "FP8_E5M2"},
                             {1, "FP8_E4M3"},
                         });

  return MakeType<RegisterTypeFlags>(
      "fpmr_flags", 8,
      std::vector<RegisterTypeFlags::Field>{{"LSCALE2", 32, 37},
                                            {"NSCALE", 24, 31},
                                            {"LSCALE", 16, 22},
                                            {"OSC", 15},
                                            {"OSM", 14},
                                            {"F8D", 6, 8, fp8_format_enum},
                                            {"F8S2", 3, 5, fp8_format_enum},
                                            {"F8S1", 0, 2, fp8_format_enum}});
}

const RegisterType *
RegisterTypes::BuildSVCRType(const Arm64RegisterFeatures &features) {

  if (!features.sme)
    return nullptr;

  // Represents the pseudo register that lldb-server builds, which itself
  // matches the architectural register SCVR. The fields match SVCR in the Arm
  // manual.
  return MakeType<RegisterTypeFlags>(
      "svcr_flags", 8,
      std::vector<RegisterTypeFlags::Field>{{"ZA", 1}, {"SM", 0}});
}

const RegisterType *
RegisterTypes::BuildFPCRType(const Arm64RegisterFeatures &features) {

  const RegisterTypeEnum *rmode_enum = MakeType<RegisterTypeEnum>(
      "rmode_enum", RegisterTypeEnum::Enumerators{
                        {0, "RN"}, {1, "RP"}, {2, "RM"}, {3, "RZ"}});

  std::vector<RegisterTypeFlags::Field> fpcr_fields{
      {"AHP", 26},
      {"DN", 25},
      {"FZ", 24},
      {"RMode", 22, 23, rmode_enum},
      // Bits 21-20 are "Stride" which is unused in AArch64 state.
  };

  // FEAT_FP16 is indicated by the presence of FPHP (floating point half
  // precision) and ASIMDHP (Advanced SIMD half precision) features.
  if (features.fp16)
    fpcr_fields.push_back({"FZ16", 19});

  // Bits 18-16 are "Len" which is unused in AArch64 state.

  fpcr_fields.push_back({"IDE", 15});

  // Bit 14 is unused.
  if (features.ebf16)
    fpcr_fields.push_back({"EBF", 13});

  fpcr_fields.push_back({"IXE", 12});
  fpcr_fields.push_back({"UFE", 11});
  fpcr_fields.push_back({"OFE", 10});
  fpcr_fields.push_back({"DZE", 9});
  fpcr_fields.push_back({"IOE", 8});
  // Bits 7-3 reserved.

  if (features.afp) {
    fpcr_fields.push_back({"NEP", 2});
    fpcr_fields.push_back({"AH", 1});
    fpcr_fields.push_back({"FIZ", 0});
  }

  return MakeType<RegisterTypeFlags>("fpcr_flags", 4, fpcr_fields);
}

const RegisterType *
RegisterTypes::BuildFPSRType(const Arm64RegisterFeatures &features) {
  // fpsr's contents are constant.
  (void)features;

  return MakeType<RegisterTypeFlags>(
      "fpsr_flags", 4,
      std::vector<RegisterTypeFlags::Field>{
          // Bits 31-28 are N/Z/C/V, only used by AArch32.
          {"QC", 27},
          // Bits 26-8 reserved.
          {"IDC", 7},
          // Bits 6-5 reserved.
          {"IXC", 4},
          {"UFC", 3},
          {"OFC", 2},
          {"DZC", 1},
          {"IOC", 0},
      });
}

const RegisterType *
RegisterTypes::BuildCPSRType(const Arm64RegisterFeatures &features) {

  // Architectural SPSR fields. The caller's feature profile determines which
  // fields are meaningful for the saved execution state.

  // Status bits that are always present.
  std::vector<RegisterTypeFlags::Field> cpsr_fields{
      {"N", 31},
      {"Z", 30},
      {"C", 29},
      {"V", 28},
      // Bits 27-26 reserved.
  };

  if (features.mte)
    cpsr_fields.push_back({"TCO", 25});
  if (features.dit)
    cpsr_fields.push_back({"DIT", 24});

  if (features.uao)
    cpsr_fields.push_back({"UAO", 23});
  if (features.pan)
    cpsr_fields.push_back({"PAN", 22});

  cpsr_fields.push_back({"SS", 21});
  cpsr_fields.push_back({"IL", 20});
  // Bits 19-14 reserved.

  if (features.nmi)
    cpsr_fields.push_back({"ALLINT", 13});
  if (features.ssbs)
    cpsr_fields.push_back({"SSBS", 12});
  if (features.bti)
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

  return MakeType<RegisterTypeFlags>("cpsr_flags", 4, cpsr_fields);
}

} // namespace

std::shared_ptr<const RegisterDescription>
RegisterInfo_arm64::CreateDescription(
    Flags features, uint32_t sve_vq, uint32_t za_vq,
    const Arm64RegisterFeatures *type_features) {
#include "Register/Arch/RegisterDescriptions_arm64.inc"
#include "Register/Arch/RegisterDescriptions_arm64_sve.inc"
  static const lldb_private::RegisterDescriptor g_register_infos_tls[] = {
      DEFINE_EXTENSION_REG_GENERIC(tpidr, LLDB_REGNUM_GENERIC_TP),
      // Only present when SME is present
      DEFINE_EXTENSION_REG(tpidr2)};

  static const lldb_private::RegisterDescriptor g_register_infos_sme[] = {
      DEFINE_EXTENSION_REG(svcr), DEFINE_EXTENSION_REG(svg),
      // 16 is a default size we will change later.
      RegisterDescriptor(LLDB_INVALID_REGNUM, "za", nullptr, 16,
                         lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,
                         KIND_ALL_INVALID, {}, {}, nullptr)};

  static const lldb_private::RegisterDescriptor g_register_infos_fpmr[] = {
      DEFINE_EXTENSION_REG(fpmr)};

  static const lldb_private::RegisterDescriptor g_register_infos_gcs[] = {
      DEFINE_EXTENSION_REG(gcspr_el0)};

  static const lldb_private::RegisterDescriptor g_register_infos_poe[] = {
      DEFINE_EXTENSION_REG(por_el0)};

#undef MISC_KIND
#undef LLDB_KIND
#undef GENERIC_KIND
#undef KIND_ALL_INVALID
#undef GPR64_KIND
#undef VREG_KIND
#undef DEFINE_GPR64
#undef DEFINE_GPR64_ALT
#undef DEFINE_GPR32
#undef DEFINE_VREG
#undef DEFINE_FPU_PSEUDO
#undef DEFINE_GPR_CPSR
#undef DEFINE_MISC_REGS
#undef DEFINE_MISC_LLDB_REGS
#undef DEFINE_EXTENSION_REG
#undef DEFINE_EXTENSION_REG_GENERIC
#undef LLDB_REGISTER_INFOS_ARM64_KEEP_MACROS

  assert(VectorSizeIsValid(sve_vq) && VectorSizeIsValid(za_vq));
  bool sve = features.AnySet(eRegsetMaskSVE | eRegsetMaskSSVE);
  auto values = sve ? llvm::ArrayRef(g_register_infos_arm64_sve_le)
                    : llvm::ArrayRef(g_register_infos_arm64_le);
  std::vector<RegisterDescriptor> regs(values.begin(), values.end());
  std::vector<RegisterGroup> groups = {{"General Purpose Registers", "gpr", {}},
                                       {"Floating Point Registers", "fpu", {}}};
  if (sve)
    groups.push_back({"Scalable Vector Extension Registers", "sve", {}});
  for (auto &reg : regs) {
    groups[reg.id <= reg_ids::gpr_w28    ? 0
           : reg.id <= reg_ids::fpu_fpcr ? 1
                                      : 2]
        .registers.push_back(reg.id);
    if (reg.id >= reg_ids::gpr_w0 && reg.id <= reg_ids::gpr_w28 &&
        !reg.value_registers.empty())
      reg.zero_extend_parent = reg.value_registers.front();
    if (sve && reg.id >= reg_ids::sve_z0) {
      uint32_t vq = std::max(1u, sve_vq);
      reg.byte_size = reg.id <= reg_ids::sve_z31 ? vq * 16 : vq * 2;
    }
  }
  auto add = [&](llvm::ArrayRef<RegisterDescriptor> values, const char *name,
                 const char *short_name, RegisterID first_id) {
    RegisterGroup group{name, short_name, {}};
    for (const auto &value : values) {
      RegisterDescriptor reg = value;
      reg.id = first_id++;
      reg.numbers[lldb::eRegisterKindLLDB] = regs.size();
      if (reg.name == "za")
        reg.byte_size = za_vq ? za_vq * za_vq * 256 : 16;
      group.registers.push_back(reg.id);
      regs.push_back(std::move(reg));
    }
    groups.push_back(std::move(group));
  };
  // Stable identities are independent of which optional sets precede them.
  if (features.AnySet(eRegsetMaskTLS))
    add(llvm::ArrayRef(g_register_infos_tls)
            .take_front(features.AnySet(eRegsetMaskSSVE) ? 2 : 1),
        "Thread Local Storage Registers", "tls", 0x10020);
  if (features.AnySet(eRegsetMaskSSVE)) {
    add(g_register_infos_sme, "Scalable Matrix Extension Registers", "sme",
        0x10030);
    if (features.AnySet(eRegsetMaskZT)) {
      RegisterDescriptor zt;
      zt.id = 0x10033;
      zt.name = "zt0";
      zt.byte_size = 64;
      zt.encoding = lldb::eEncodingVector;
      zt.format = lldb::eFormatVectorOfUInt8;
      zt.numbers[lldb::eRegisterKindLLDB] = regs.size();
      groups.back().registers.push_back(zt.id);
      regs.push_back(std::move(zt));
    }
    for (auto &reg : regs)
      if (reg.id == reg_ids::sve_vg)
        reg.invalidate_registers.push_back(0x10031); // svg
  }
  if (features.AnySet(eRegsetMaskFPMR))
    add(g_register_infos_fpmr, "Floating Point Mode Register", "fpmr", 0x10040);
  if (features.AnySet(eRegsetMaskGCS))
    add(g_register_infos_gcs, "Guarded Control Stack Registers", "gcs",
        0x10052);
  if (features.AnySet(eRegsetMaskPOE))
    add(g_register_infos_poe, "Permission Overlay Registers", "poe", 0x10060);
  RegisterTypes types;
  if (type_features) {
    const std::pair<llvm::StringRef, const RegisterType *> fields[] = {
        {"cpsr", types.BuildCPSRType(*type_features)},
        {"fpsr", types.BuildFPSRType(*type_features)},
        {"fpcr", types.BuildFPCRType(*type_features)},
        {"svcr", types.BuildSVCRType(*type_features)},
        {"fpmr", types.BuildFPMRType(*type_features)},
        {"por_el0", types.BuildPOREL0Type(*type_features)}};
    for (auto &reg : regs)
      for (const auto &[name, type] : fields)
        if (name == reg.name)
          reg.type = type;
  }
  return std::make_shared<RegisterDescription>(
      llvm::cantFail(RegisterDescription::Create(
          std::move(regs), std::move(groups), std::move(types.owned))));
}
bool RegisterInfo_arm64::InGroup(const RegisterDescription &description,
                                 unsigned number, llvm::StringRef group) {
  const auto *reg = description.FindByNumber(lldb::eRegisterKindLLDB, number);
  return reg && description.IsInGroup(reg->id, group);
}
uint32_t RegisterInfo_arm64::Number(const RegisterDescription &description,
                                    llvm::StringRef name) {
  const auto *reg = description.Find(name);
  return reg ? reg->numbers[lldb::eRegisterKindLLDB] : LLDB_INVALID_REGNUM;
}
bool RegisterInfo_arm64::IsGPR(const RegisterDescription &description,
                               unsigned number) {
  return InGroup(description, number, "gpr");
}
bool RegisterInfo_arm64::IsFPR(const RegisterDescription &description,
                               unsigned number) {
  return InGroup(description, number, "fpu");
}
bool RegisterInfo_arm64::IsPAuthReg(const RegisterDescription &description,
                                    unsigned number) {
  return InGroup(description, number, "pauth");
}
bool RegisterInfo_arm64::IsMTEReg(const RegisterDescription &description,
                                  unsigned number) {
  return InGroup(description, number, "mte");
}
bool RegisterInfo_arm64::IsTLSReg(const RegisterDescription &description,
                                  unsigned number) {
  return InGroup(description, number, "tls");
}
bool RegisterInfo_arm64::IsSMEReg(const RegisterDescription &description,
                                  unsigned number) {
  return InGroup(description, number, "sme");
}
bool RegisterInfo_arm64::IsFPMRReg(const RegisterDescription &description,
                                   unsigned number) {
  return InGroup(description, number, "fpmr");
}
bool RegisterInfo_arm64::IsGCSReg(const RegisterDescription &description,
                                  unsigned number) {
  return InGroup(description, number, "gcs");
}
bool RegisterInfo_arm64::IsPOEReg(const RegisterDescription &description,
                                  unsigned number) {
  return InGroup(description, number, "poe");
}
bool RegisterInfo_arm64::IsSVEReg(const RegisterDescription &description,
                                  unsigned number) {
  return InGroup(description, number, "sve");
}
bool RegisterInfo_arm64::IsSVEZReg(unsigned number) {
  return reg_ids::sve_z0 <= number && number <= reg_ids::sve_z31;
}
bool RegisterInfo_arm64::IsSVEPReg(unsigned number) {
  return reg_ids::sve_p0 <= number && number <= reg_ids::sve_p15;
}
bool RegisterInfo_arm64::IsSVERegVG(unsigned number) {
  return number == reg_ids::sve_vg;
}
bool RegisterInfo_arm64::IsSVERegFFR(unsigned number) {
  return number == reg_ids::sve_ffr;
}
bool RegisterInfo_arm64::IsSMERegZA(const RegisterDescription &description,
                                    unsigned number) {
  return number == Number(description, "za");
}
bool RegisterInfo_arm64::IsSMERegZT(const RegisterDescription &description,
                                    unsigned number) {
  return number == Number(description, "zt0");
}
uint32_t RegisterInfo_arm64::GetRegNumSVEZ0() { return reg_ids::sve_z0; }
uint32_t RegisterInfo_arm64::GetRegNumSVEFFR() { return reg_ids::sve_ffr; }
uint32_t RegisterInfo_arm64::GetRegNumFPCR() { return reg_ids::fpu_fpcr; }
uint32_t RegisterInfo_arm64::GetRegNumFPSR() { return reg_ids::fpu_fpsr; }
uint32_t RegisterInfo_arm64::GetRegNumFPV0() { return reg_ids::fpu_v0; }
uint32_t RegisterInfo_arm64::GetRegNumSVEVG() { return reg_ids::sve_vg; }
uint32_t
RegisterInfo_arm64::GetRegNumSMESVG(const RegisterDescription &description) {
  return Number(description, "svg");
}
