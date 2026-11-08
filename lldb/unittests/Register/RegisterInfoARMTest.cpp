//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Common/RegisterInfo_arm.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;
static void Rebuild(RegisterLayout &info,
                    const linux_arm64::RegisterConfig &config) {
  info = CreateLinuxArm64RegisterLayout(ArchSpec("aarch64-unknown-linux"),
                                        config, info.GetDescription().get());
}
TEST(RegisterInfoARMTest, MetadataDoesNotRequireSavedState) {
  auto base = RegisterInfo_arm::CreateDescription();
  EXPECT_EQ(base->Find("tpidruro"), nullptr);
  EXPECT_EQ(base->Find("exception"), nullptr);
  EXPECT_EQ(base->Find("bvr0"), nullptr);
  auto extended = RegisterInfo_arm::CreateDescription(true);
  ASSERT_NE(extended->Find("tpidruro"), nullptr);
  EXPECT_EQ(base->Find("pc")->id, extended->Find("pc")->id);
  ArchSpec arch("armv7-unknown-freebsd");
  auto saved = CreateFreeBSDArmRegisterLayout(arch);
  EXPECT_EQ(saved.GetBankSize(0), 68u);
  auto *fpscr = saved.GetDescription()->Find("fpscr");
  ASSERT_NE(fpscr, nullptr);
  EXPECT_EQ(saved.Find(fpscr->id)->pieces[0].byte_offset, 256u);
}

TEST(RegisterInfoARMTest, ARM64FeatureIdentitiesDoNotDependOnOtherSets) {
  auto tls =
      RegisterInfo_arm64::CreateDescription(RegisterInfo_arm64::eRegsetMaskTLS);
  auto extended = RegisterInfo_arm64::CreateDescription(
      RegisterInfo_arm64::eRegsetMaskTLS | RegisterInfo_arm64::eRegsetMaskSVE);
  ASSERT_NE(tls->Find("tpidr"), nullptr);
  ASSERT_NE(extended->Find("tpidr"), nullptr);
  EXPECT_EQ(tls->Find("tpidr")->id, extended->Find("tpidr")->id);
  EXPECT_NE(tls->Find("tpidr")->numbers[lldb::eRegisterKindLLDB],
            extended->Find("tpidr")->numbers[lldb::eRegisterKindLLDB]);
  EXPECT_TRUE(tls->IsInGroup(tls->Find("tpidr")->id, "tls"));
  EXPECT_TRUE(extended->IsInGroup(extended->Find("tpidr")->id, "tls"));
  EXPECT_FALSE(tls->IsInGroup(tls->Find("tpidr")->id, "gpr"));
  EXPECT_TRUE(RegisterInfo_arm64::IsTLSReg(
      *tls, tls->Find("tpidr")->numbers[lldb::eRegisterKindLLDB]));
  EXPECT_TRUE(RegisterInfo_arm64::IsTLSReg(
      *extended, extended->Find("tpidr")->numbers[lldb::eRegisterKindLLDB]));
}
TEST(RegisterInfoARMTest, ARM64RebuildsBothVectorDescriptionsAndLayouts) {
  ArchSpec arch("aarch64-unknown-linux");
  linux_arm64::RegisterConfig config{RegisterInfo_arm64::eRegsetMaskSSVE |
                                     RegisterInfo_arm64::eRegsetMaskZA |
                                     RegisterInfo_arm64::eRegsetMaskZT};
  auto info = CreateLinuxArm64RegisterLayout(arch, config);
  auto previous = info.GetDescription();
  const auto *original_z0 = previous->Find("z0");
  config.sve_vq = 4;
  Rebuild(info, config);
  EXPECT_EQ(original_z0->byte_size, 16u);
  EXPECT_EQ(info.GetDescription()->Find("z0")->id, original_z0->id);
  auto *z0 = info.GetDescription()->Find("z0");
  ASSERT_NE(z0, nullptr);
  EXPECT_EQ(z0->byte_size, 64u);
  EXPECT_EQ(info.Find(z0->id)->pieces[0].byte_size, 64u);
  config.za_vq = 2;
  Rebuild(info, config);
  auto *za = info.GetDescription()->Find("za");
  auto *zt = info.GetDescription()->Find("zt0");
  ASSERT_NE(za, nullptr);
  ASSERT_NE(zt, nullptr);
  EXPECT_EQ(za->byte_size, 1024u);
  EXPECT_EQ(info.Find(zt->id)->pieces[0].bank, linux_arm64::ZT);
  EXPECT_EQ(info.Find(zt->id)->pieces[0].byte_offset, 0u);
  EXPECT_FALSE(RegisterInfo_arm64::VectorSizeIsValid(257));
  EXPECT_EQ(info.GetDescription()->Find("z0")->byte_size, 64u);
  config.sve_vq = 2;
  Rebuild(info, config);
  EXPECT_EQ(info.GetDescription()->Find("za")->byte_size, 1024u);
  EXPECT_EQ(info.GetDescription()->Find("z0")->byte_size, 32u);
}

TEST(RegisterInfoARMTest, ARM64TypeAttachmentsSurviveReconfiguration) {
  ArchSpec arch("aarch64-unknown-linux");
  linux_arm64::RegisterConfig config{RegisterInfo_arm64::eRegsetMaskSVE |
                                         RegisterInfo_arm64::eRegsetMaskMTE,
                                     0, 1ULL << 18, 1ULL << 1};
  auto info = CreateLinuxArm64RegisterLayout(arch, config);
  auto *before = info.GetDescription()
                     ->FindByNumber(lldb::eRegisterKindLLDB, reg_ids::fpu_fpcr)
                     ->type;
  ASSERT_NE(before, nullptr);
  config.sve_vq = 4;
  Rebuild(info, config);
  EXPECT_EQ(info.GetDescription()
                ->FindByNumber(lldb::eRegisterKindLLDB, reg_ids::fpu_fpcr)
                ->type,
            before);
  auto *control = info.GetDescription()->Find("mte_ctrl");
  ASSERT_NE(control, nullptr);
  EXPECT_NE(control->type, nullptr);
}

TEST(RegisterInfoARMTest, ARM64CommonDescriptionsExcludeOSControls) {
  auto description = RegisterInfo_arm64::CreateDescription(
      RegisterInfo_arm64::eRegsetMaskPAuth |
      RegisterInfo_arm64::eRegsetMaskMTE | RegisterInfo_arm64::eRegsetMaskGCS);
  EXPECT_EQ(description->Find("data_mask"), nullptr);
  EXPECT_EQ(description->Find("mte_ctrl"), nullptr);
  EXPECT_EQ(description->Find("gcs_features_enabled"), nullptr);
  ASSERT_NE(description->Find("gcspr_el0"), nullptr);
}
TEST(RegisterInfoARMTest, ARM64ArchitectureTypesUseExplicitFeatures) {
  Arm64RegisterFeatures features;
  features.pan = features.uao = features.nmi = true;
  auto description = RegisterInfo_arm64::CreateDescription(0, 0, 0, &features);
  auto *type = llvm::cast<RegisterTypeFlags>(description->Find("cpsr")->type);
  for (auto name : {"PAN", "UAO", "ALLINT"})
    EXPECT_TRUE(llvm::any_of(type->GetFields(), [&](const auto &field) {
      return field.GetName() == name;
    }));
}
TEST(RegisterInfoARMTest, ARM64OSDecodersBuildOwnedTypes) {
  const uint64_t hwcap = (1ULL << 9) | (1ULL << 10) | (1ULL << 32);
  const uint64_t hwcap2 = (1ULL << 18) | (1ULL << 20);
  auto fbsd = DecodeFreeBSDArm64RegisterFeatures(hwcap, hwcap2);
  auto linux_features = DecodeLinuxArm64RegisterFeatures(hwcap, hwcap2);
  EXPECT_EQ(fbsd.fp16, linux_features.fp16);
  EXPECT_TRUE(fbsd.fp16);
  EXPECT_TRUE(fbsd.afp);
  auto model = CreateFreeBSDArm64RegisterLayout(
      ArchSpec("aarch64-unknown-freebsd"), hwcap, hwcap2);
  EXPECT_NE(model.GetDescription()->Find("fpcr")->type, nullptr);
  EXPECT_EQ(model.GetDescription()->Find("mte_ctrl"), nullptr);
  linux_arm64::RegisterConfig config{RegisterInfo_arm64::eRegsetMaskGCS |
                                         RegisterInfo_arm64::eRegsetMaskMTE,
                                     hwcap, hwcap2, 1ULL << 1};
  auto info =
      CreateLinuxArm64RegisterLayout(ArchSpec("aarch64-unknown-linux"), config);
  auto *control = info.GetDescription()->Find("mte_ctrl");
  ASSERT_NE(control, nullptr);
  auto *type = llvm::cast<RegisterTypeFlags>(control->type);
  EXPECT_TRUE(llvm::any_of(type->GetFields(), [](const auto &field) {
    return field.GetName() == "STORE_ONLY";
  }));
  EXPECT_NE(info.GetDescription()->Find("gcs_features_enabled")->type, nullptr);
}

TEST(RegisterInfoARMTest, ARM64LinuxControlNumberingPreservesPresentation) {
  const uint32_t first =
      RegisterInfo_arm64::CreateDescription(0)->GetRegisters().size();
  linux_arm64::RegisterConfig config{RegisterInfo_arm64::eRegsetMaskTLS |
                                     RegisterInfo_arm64::eRegsetMaskPAuth |
                                     RegisterInfo_arm64::eRegsetMaskMTE |
                                     RegisterInfo_arm64::eRegsetMaskGCS};
  auto info =
      CreateLinuxArm64RegisterLayout(ArchSpec("aarch64-unknown-linux"), config);
  const auto &description = *info.GetDescription();
  uint32_t number = first;
  for (auto name :
       {"data_mask", "code_mask", "mte_ctrl", "tpidr", "gcs_features_enabled",
        "gcs_features_locked", "gcspr_el0"}) {
    const auto *reg = description.Find(name);
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->numbers[lldb::eRegisterKindLLDB], number++);
  }
  EXPECT_EQ(description.GetRegisters().size(), number);
}
