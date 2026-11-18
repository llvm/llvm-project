//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Descriptions/RegisterDescription_arm64.h"
#include "lldb/Register/RegisterTypeFlags.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(RegisterDescription_arm64Test, FeatureIdentitiesDoNotDependOnOtherGroups) {
  auto tls = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{
          RegisterDescription_arm64::eRegsetMaskTLS});
  auto extended = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{
          RegisterDescription_arm64::eRegsetMaskTLS |
          RegisterDescription_arm64::eRegsetMaskSVE});
  ASSERT_NE(tls->Find("tpidr"), nullptr);
  ASSERT_NE(extended->Find("tpidr"), nullptr);
  EXPECT_EQ(tls->Find("tpidr")->id, extended->Find("tpidr")->id);
  EXPECT_NE(tls->Find("tpidr")->numbers[lldb::eRegisterKindLLDB],
            extended->Find("tpidr")->numbers[lldb::eRegisterKindLLDB]);
  EXPECT_TRUE(tls->IsInGroup(tls->Find("tpidr")->id, "tls"));
  EXPECT_TRUE(extended->IsInGroup(extended->Find("tpidr")->id, "tls"));
  EXPECT_FALSE(tls->IsInGroup(tls->Find("tpidr")->id, "gpr"));
}

TEST(RegisterDescription_arm64Test, ExcludesOSControls) {
  auto description = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{
          RegisterDescription_arm64::eRegsetMaskPAuth |
          RegisterDescription_arm64::eRegsetMaskMTE |
          RegisterDescription_arm64::eRegsetMaskGCS});
  EXPECT_EQ(description->Find("data_mask"), nullptr);
  EXPECT_EQ(description->Find("mte_ctrl"), nullptr);
  EXPECT_EQ(description->Find("gcs_features_enabled"), nullptr);
  ASSERT_NE(description->Find("gcspr_el0"), nullptr);
}

TEST(RegisterDescription_arm64Test, UsesExplicitFeaturesForRegisterTypes) {
  RegisterDescription_arm64::TypeFeatures features;
  features.pan = features.uao = features.nmi = true;
  auto description = std::make_shared<RegisterDescription_arm64>(
      RegisterDescription_arm64::Config{0, 0, 0, &features});
  auto *type = llvm::cast<RegisterTypeFlags>(description->Find("cpsr")->type);
  for (auto name : {"PAN", "UAO", "ALLINT"})
    EXPECT_TRUE(llvm::any_of(type->GetFields(), [&](const auto &field) {
      return field.GetName() == name;
    }));
}
