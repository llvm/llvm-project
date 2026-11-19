//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-arm64-register-enums.h"
#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"
#include "lldb/Register/Layouts/FreeBSD/RegisterLayoutFreeBSD_arm64.h"
#include "lldb/Register/Layouts/Linux/RegisterLayoutLinux_arm64.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

namespace reg_ids = lldb_private::registers::arm64::id;
using namespace lldb_private;

static void Rebuild(RegisterLayout &info,
                    const RegisterLayoutLinux_arm64::Config &config) {
  info = RegisterLayoutLinux_arm64(ArchSpec("aarch64-unknown-linux"), config,
                                   info.GetDescription().get());
}

TEST(RegisterLayout_arm64Test, RebuildsBothVectorDescriptionsAndLayouts) {
  ArchSpec arch("aarch64-unknown-linux");
  RegisterLayoutLinux_arm64::Config config{
      RegisterDescription_arm64::eRegsetMaskSSVE |
      RegisterDescription_arm64::eRegsetMaskZA |
      RegisterDescription_arm64::eRegsetMaskZT};
  auto info = RegisterLayoutLinux_arm64(arch, config);
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
  EXPECT_EQ(info.Find(zt->id)->pieces[0].bank, RegisterLayoutLinux_arm64::ZT);
  EXPECT_EQ(info.Find(zt->id)->pieces[0].byte_offset, 0u);
  EXPECT_FALSE(RegisterDescription_arm64::Config::VectorSizeIsValid(257));
  EXPECT_EQ(info.GetDescription()->Find("z0")->byte_size, 64u);
  config.sve_vq = 2;
  Rebuild(info, config);
  EXPECT_EQ(info.GetDescription()->Find("za")->byte_size, 1024u);
  EXPECT_EQ(info.GetDescription()->Find("z0")->byte_size, 32u);
}

TEST(RegisterLayout_arm64Test, TypeAttachmentsSurviveReconfiguration) {
  ArchSpec arch("aarch64-unknown-linux");
  RegisterLayoutLinux_arm64::Config config{
      RegisterDescription_arm64::eRegsetMaskSVE |
          RegisterDescription_arm64::eRegsetMaskMTE,
      0, 1ULL << 18, 1ULL << 1};
  auto info = RegisterLayoutLinux_arm64(arch, config);
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

TEST(RegisterLayout_arm64Test, OSDecodersBuildOwnedTypes) {
  const uint64_t hwcap = (1ULL << 9) | (1ULL << 10) | (1ULL << 32);
  const uint64_t hwcap2 = (1ULL << 18) | (1ULL << 20);
  auto fbsd =
      RegisterLayoutFreeBSD_arm64::DecodeRegisterFeatures(hwcap, hwcap2);
  auto linux_features =
      RegisterLayoutLinux_arm64::DecodeRegisterFeatures(hwcap, hwcap2);
  EXPECT_EQ(fbsd.fp16, linux_features.fp16);
  EXPECT_TRUE(fbsd.fp16);
  EXPECT_TRUE(fbsd.afp);
  auto model = RegisterLayoutFreeBSD_arm64(ArchSpec("aarch64-unknown-freebsd"),
                                           hwcap, hwcap2);
  EXPECT_NE(model.GetDescription()->Find("fpcr")->type, nullptr);
  EXPECT_EQ(model.GetDescription()->Find("mte_ctrl"), nullptr);
  RegisterLayoutLinux_arm64::Config config{
      RegisterDescription_arm64::eRegsetMaskGCS |
          RegisterDescription_arm64::eRegsetMaskMTE,
      hwcap, hwcap2, 1ULL << 1};
  auto info =
      RegisterLayoutLinux_arm64(ArchSpec("aarch64-unknown-linux"), config);
  auto *control = info.GetDescription()->Find("mte_ctrl");
  ASSERT_NE(control, nullptr);
  auto *type = llvm::cast<RegisterTypeFlags>(control->type);
  EXPECT_TRUE(llvm::any_of(type->GetFields(), [](const auto &field) {
    return field.GetName() == "STORE_ONLY";
  }));
  EXPECT_NE(info.GetDescription()->Find("gcs_features_enabled")->type, nullptr);
}

TEST(RegisterLayout_arm64Test, LinuxControlNumberingPreservesPresentation) {
  const uint32_t first = std::make_shared<RegisterDescription_arm64>(
                             RegisterDescription_arm64::Config{0})
                             ->GetRegisters()
                             .size();
  RegisterLayoutLinux_arm64::Config config{
      RegisterDescription_arm64::eRegsetMaskTLS |
      RegisterDescription_arm64::eRegsetMaskPAuth |
      RegisterDescription_arm64::eRegsetMaskMTE |
      RegisterDescription_arm64::eRegsetMaskGCS};
  auto info =
      RegisterLayoutLinux_arm64(ArchSpec("aarch64-unknown-linux"), config);
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
