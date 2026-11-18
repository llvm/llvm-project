//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"
#include "lldb/Register/Arch/lldb-arm64-dwarf-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-ehframe-register-enums.h"
#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/RegisterDescriptionBuilder.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "llvm/ADT/STLExtras.h"

#include <utility>

namespace reg_ids = lldb_private::registers::arm64::id;

using namespace lldb_private;

namespace {

using TypeFeatures = RegisterDescription_arm64::TypeFeatures;

namespace dwarf_regs = lldb_private::registers::arm64::dwarf;
namespace ehframe_regs = lldb_private::registers::arm64::ehframe;

#include "RegisterDescription_arm64.inc"
#include "RegisterDescription_arm64_sve.inc"

struct RegisterTypes {
  RegisterDescription::OwnedTypes owned;

  template <typename T, typename... Args> const T *MakeType(Args &&...args) {
    auto type = std::make_shared<T>(std::forward<Args>(args)...);
    const T *result = type.get();

    owned.push_back(std::move(type));

    return result;
  }

  const RegisterType *BuildPOREL0Type(const TypeFeatures &features);

  const RegisterType *BuildFPMRType(const TypeFeatures &features);

  const RegisterType *BuildSVCRType(const TypeFeatures &features);

  const RegisterType *BuildFPCRType(const TypeFeatures &features);

  const RegisterType *BuildFPSRType(const TypeFeatures &features);

  const RegisterType *BuildCPSRType(const TypeFeatures &features);
};

const RegisterType *
RegisterTypes::BuildPOREL0Type(const TypeFeatures &features) {

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

const RegisterType *RegisterTypes::BuildFPMRType(const TypeFeatures &features) {

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

const RegisterType *RegisterTypes::BuildSVCRType(const TypeFeatures &features) {

  if (!features.sme)
    return nullptr;

  // Represents the pseudo register that lldb-server builds, which itself
  // matches the architectural register SCVR. The fields match SVCR in the Arm
  // manual.
  return MakeType<RegisterTypeFlags>(
      "svcr_flags", 8,
      std::vector<RegisterTypeFlags::Field>{{"ZA", 1}, {"SM", 0}});
}

const RegisterType *RegisterTypes::BuildFPCRType(const TypeFeatures &features) {

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

const RegisterType *RegisterTypes::BuildFPSRType(const TypeFeatures &features) {
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

const RegisterType *RegisterTypes::BuildCPSRType(const TypeFeatures &features) {

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

RegisterDescription_arm64::RegisterDescription_arm64(const Config &config) {
  const auto &[features, sve_vq, za_vq, type_features] = config;
  assert(RegisterDescription_arm64::Config::VectorSizeIsValid(sve_vq) &&
         RegisterDescription_arm64::Config::VectorSizeIsValid(za_vq));

  const bool sve = features.AnySet(eRegsetMaskSVE | eRegsetMaskSSVE);
  auto definitions = sve ? llvm::ArrayRef(GetSVEDefinitions())
                         : llvm::ArrayRef(GetDefinitions());
  std::vector<RegisterDescriptor> catalog(definitions.begin(),
                                          definitions.end());
  catalog.insert(catalog.end(), std::begin(GetExtensionDefinitions()),
                 std::end(GetExtensionDefinitions()));

  std::vector<RegisterGroup> groups = {
      MakeRegisterGroup("General Purpose Registers", "gpr", GPRRegisters),
      MakeRegisterGroup("Floating Point Registers", "fpu", FPRRegisters)};

  if (sve)
    groups.push_back(MakeRegisterGroup("Scalable Vector Extension Registers",
                                       "sve", SVERegisters));

  if (features.AnySet(eRegsetMaskTLS))
    groups.push_back(MakeRegisterGroup(
        "Thread Local Storage Registers", "tls",
        llvm::ArrayRef(TLSRegisters)
            .take_front(features.AnySet(eRegsetMaskSSVE) ? 2 : 1)));

  if (features.AnySet(eRegsetMaskSSVE)) {
    groups.push_back(MakeRegisterGroup("Scalable Matrix Extension Registers",
                                       "sme", SMERegisters));
    if (features.AnySet(eRegsetMaskZT))
      groups.back().registers.insert(groups.back().registers.end(),
                                     std::begin(ZTRegisters),
                                     std::end(ZTRegisters));
  }

  if (features.AnySet(eRegsetMaskFPMR))
    groups.push_back(MakeRegisterGroup("Floating Point Mode Register", "fpmr",
                                       FPMRRegisters));

  if (features.AnySet(eRegsetMaskGCS))
    groups.push_back(MakeRegisterGroup("Guarded Control Stack Registers", "gcs",
                                       GCSRegisters));

  if (features.AnySet(eRegsetMaskPOE))
    groups.push_back(
        MakeRegisterGroup("Permission Overlay Registers", "poe", POERegisters));

  auto regs = SelectRegisterDefinitions(catalog, groups);
  for (size_t i = 0; i != regs.size(); ++i) {
    auto &reg = regs[i];
    reg.numbers[lldb::eRegisterKindLLDB] = i;

    if (llvm::is_contained(SVEVectorRegisters, reg.id))
      reg.byte_size = std::max(1u, sve_vq) * 16;
    else if (llvm::is_contained(SVEPredicateRegisters, reg.id))
      reg.byte_size = std::max(1u, sve_vq) * 2;
    else if (reg.id == reg_ids::sme_za)
      reg.byte_size = za_vq ? za_vq * za_vq * 256 : 16;
  }

  RegisterTypes types;
  if (type_features) {
    const std::pair<RegisterID, const RegisterType *> fields[] = {
        {reg_ids::gpr_cpsr, types.BuildCPSRType(*type_features)},
        {reg_ids::fpu_fpsr, types.BuildFPSRType(*type_features)},
        {reg_ids::fpu_fpcr, types.BuildFPCRType(*type_features)},
        {reg_ids::sme_svcr, types.BuildSVCRType(*type_features)},
        {reg_ids::fpmr_fpmr, types.BuildFPMRType(*type_features)},
        {reg_ids::poe_por_el0, types.BuildPOREL0Type(*type_features)}};

    for (auto &reg : regs)
      for (const auto &[id, type] : fields)
        if (id == reg.id)
          reg.type = type;
  }

  RegisterDescription::operator=(llvm::cantFail(RegisterDescription::Create(
      std::move(regs), std::move(groups), std::move(types.owned))));
}
