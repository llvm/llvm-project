//===- unittest/IR/OpenMPContextTest.cpp - OpenMP Context handling tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPContext.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace omp;

namespace {

class OpenMPContextTest : public testing::Test {
protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(OpenMPContextTest, RoundTripAndAssociation) {
#define OMP_TRAIT_SET(Enum, Str)                                               \
  EXPECT_EQ(TraitSet::Enum,                                                    \
            getOpenMPContextTraitSetKind(                                      \
                getOpenMPContextTraitSetName(TraitSet::Enum)));                \
  EXPECT_EQ(Str,                                                               \
            getOpenMPContextTraitSetName(getOpenMPContextTraitSetKind(Str)));
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, RequiresProperty)          \
  EXPECT_EQ(TraitSelector::Enum,                                               \
            getOpenMPContextTraitSelectorKind(                                 \
                getOpenMPContextTraitSelectorName(TraitSelector::Enum)));      \
  EXPECT_EQ(Str, getOpenMPContextTraitSelectorName(                            \
                     getOpenMPContextTraitSelectorKind(Str)));
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  EXPECT_EQ(TraitProperty::Enum,                                               \
            getOpenMPContextTraitPropertyKind(                                 \
                TraitSet::TraitSetEnum, TraitSelector::TraitSelectorEnum,      \
                getOpenMPContextTraitPropertyName(TraitProperty::Enum, Str))); \
  EXPECT_EQ(Str, getOpenMPContextTraitPropertyName(                            \
                     getOpenMPContextTraitPropertyKind(                        \
                         TraitSet::TraitSetEnum,                               \
                         TraitSelector::TraitSelectorEnum, Str),               \
                     Str));                                                    \
  EXPECT_EQ(TraitSet::TraitSetEnum,                                            \
            getOpenMPContextTraitSetForProperty(TraitProperty::Enum));         \
  EXPECT_EQ(TraitSelector::TraitSelectorEnum,                                  \
            getOpenMPContextTraitSelectorForProperty(TraitProperty::Enum));
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}

TEST_F(OpenMPContextTest, ValidNesting) {
  bool AllowsTraitScore, ReqProperty;
#define OMP_TRAIT_SELECTOR(Enum, TraitSetEnum, Str, RequiresProperty)          \
  EXPECT_TRUE(isValidTraitSelectorForTraitSet(TraitSelector::Enum,             \
                                              TraitSet::TraitSetEnum,          \
                                              AllowsTraitScore, ReqProperty)); \
  EXPECT_EQ(RequiresProperty, ReqProperty);
#define OMP_TRAIT_PROPERTY(Enum, TraitSetEnum, TraitSelectorEnum, Str)         \
  EXPECT_TRUE(isValidTraitPropertyForTraitSetAndSelector(                      \
      TraitProperty::Enum, TraitSelector::TraitSelectorEnum,                   \
      TraitSet::TraitSetEnum));
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}

TEST_F(OpenMPContextTest, ApplicabilityNonConstruct) {
  OMPContext HostLinux(false, Triple("x86_64-unknown-linux"));
  OMPContext DeviceLinux(true, Triple("x86_64-unknown-linux"));
  OMPContext HostNVPTX(false, Triple("nvptx64-nvidia-cuda"));
  OMPContext DeviceNVPTX(true, Triple("nvptx64-nvidia-cuda"));

  VariantMatchInfo Empty;
  EXPECT_TRUE(isVariantApplicableInContext(Empty, HostLinux));
  EXPECT_TRUE(isVariantApplicableInContext(Empty, DeviceLinux));
  EXPECT_TRUE(isVariantApplicableInContext(Empty, HostNVPTX));
  EXPECT_TRUE(isVariantApplicableInContext(Empty, DeviceNVPTX));

  VariantMatchInfo UserCondFalse;
  UserCondFalse.addTrait(TraitProperty::user_condition_false, "");
  EXPECT_FALSE(isVariantApplicableInContext(UserCondFalse, HostLinux));
  EXPECT_FALSE(isVariantApplicableInContext(UserCondFalse, DeviceLinux));
  EXPECT_FALSE(isVariantApplicableInContext(UserCondFalse, HostNVPTX));
  EXPECT_FALSE(isVariantApplicableInContext(UserCondFalse, DeviceNVPTX));

  VariantMatchInfo DeviceArchArm;
  DeviceArchArm.addTrait(TraitProperty::device_arch_arm, "");
  EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArm, HostLinux));
  EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArm, DeviceLinux));
  EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArm, HostNVPTX));
  EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArm, DeviceNVPTX));

  VariantMatchInfo LLVMHostUserCondTrue;
  LLVMHostUserCondTrue.addTrait(TraitProperty::implementation_vendor_llvm, "");
  LLVMHostUserCondTrue.addTrait(TraitProperty::device_kind_host, "");
  LLVMHostUserCondTrue.addTrait(TraitProperty::device_kind_any, "");
  LLVMHostUserCondTrue.addTrait(TraitProperty::user_condition_true, "");
  EXPECT_TRUE(isVariantApplicableInContext(LLVMHostUserCondTrue, HostLinux));
  EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrue, DeviceLinux));
  EXPECT_TRUE(isVariantApplicableInContext(LLVMHostUserCondTrue, HostNVPTX));
  EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrue, DeviceNVPTX));

  VariantMatchInfo LLVMHostUserCondTrueCPU = LLVMHostUserCondTrue;
  LLVMHostUserCondTrueCPU.addTrait(TraitProperty::device_kind_cpu, "");
  EXPECT_TRUE(isVariantApplicableInContext(LLVMHostUserCondTrueCPU, HostLinux));
  EXPECT_FALSE(
      isVariantApplicableInContext(LLVMHostUserCondTrueCPU, DeviceLinux));
  EXPECT_FALSE(
      isVariantApplicableInContext(LLVMHostUserCondTrueCPU, HostNVPTX));
  EXPECT_FALSE(
      isVariantApplicableInContext(LLVMHostUserCondTrueCPU, DeviceNVPTX));

  VariantMatchInfo GPU;
  GPU.addTrait(TraitProperty::device_kind_gpu, "");
  EXPECT_FALSE(isVariantApplicableInContext(GPU, HostLinux));
  EXPECT_FALSE(isVariantApplicableInContext(GPU, DeviceLinux));
  EXPECT_TRUE(isVariantApplicableInContext(GPU, HostNVPTX));
  EXPECT_TRUE(isVariantApplicableInContext(GPU, DeviceNVPTX));

  VariantMatchInfo NoHost;
  NoHost.addTrait(TraitProperty::device_kind_nohost, "");
  EXPECT_FALSE(isVariantApplicableInContext(NoHost, HostLinux));
  EXPECT_TRUE(isVariantApplicableInContext(NoHost, DeviceLinux));
  EXPECT_FALSE(isVariantApplicableInContext(NoHost, HostNVPTX));
  EXPECT_TRUE(isVariantApplicableInContext(NoHost, DeviceNVPTX));
}

TEST_F(OpenMPContextTest, ApplicabilityAllTraits) {
  OMPContext HostLinuxParallelParallel(false, Triple("x86_64-unknown-linux"));
  HostLinuxParallelParallel.addTrait(
      TraitProperty::construct_parallel_parallel);
  HostLinuxParallelParallel.addTrait(
      TraitProperty::construct_parallel_parallel);
  OMPContext DeviceLinuxTargetParallel(true, Triple("x86_64-unknown-linux"));
  DeviceLinuxTargetParallel.addTrait(TraitProperty::construct_target_target);
  DeviceLinuxTargetParallel.addTrait(
      TraitProperty::construct_parallel_parallel);
  OMPContext HostNVPTXFor(false, Triple("nvptx64-nvidia-cuda"));
  HostNVPTXFor.addTrait(TraitProperty::construct_for_for);
  OMPContext DeviceNVPTXTargetTeamsParallel(true,
                                            Triple("nvptx64-nvidia-cuda"));
  DeviceNVPTXTargetTeamsParallel.addTrait(
      TraitProperty::construct_target_target);
  DeviceNVPTXTargetTeamsParallel.addTrait(TraitProperty::construct_teams_teams);
  DeviceNVPTXTargetTeamsParallel.addTrait(
      TraitProperty::construct_parallel_parallel);

  { // non-construct variants
    VariantMatchInfo Empty;
    EXPECT_TRUE(isVariantApplicableInContext(Empty, HostLinuxParallelParallel));
    EXPECT_TRUE(isVariantApplicableInContext(Empty, DeviceLinuxTargetParallel));
    EXPECT_TRUE(isVariantApplicableInContext(Empty, HostNVPTXFor));
    EXPECT_TRUE(
        isVariantApplicableInContext(Empty, DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo UserCondFalse;
    UserCondFalse.addTrait(TraitProperty::user_condition_false, "");
    EXPECT_FALSE(
        isVariantApplicableInContext(UserCondFalse, HostLinuxParallelParallel));
    EXPECT_FALSE(
        isVariantApplicableInContext(UserCondFalse, DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(UserCondFalse, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(UserCondFalse,
                                              DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo DeviceArchArm;
    DeviceArchArm.addTrait(TraitProperty::device_arch_arm, "");
    EXPECT_FALSE(
        isVariantApplicableInContext(DeviceArchArm, HostLinuxParallelParallel));
    EXPECT_FALSE(
        isVariantApplicableInContext(DeviceArchArm, DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArm, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArm,
                                              DeviceNVPTXTargetTeamsParallel));

    APInt Score(32, 1000);
    VariantMatchInfo LLVMHostUserCondTrue;
    LLVMHostUserCondTrue.addTrait(TraitProperty::implementation_vendor_llvm,
                                  "");
    LLVMHostUserCondTrue.addTrait(TraitProperty::device_kind_host, "");
    LLVMHostUserCondTrue.addTrait(TraitProperty::device_kind_any, "");
    LLVMHostUserCondTrue.addTrait(TraitProperty::user_condition_true, "",
                                  &Score);
    EXPECT_TRUE(isVariantApplicableInContext(LLVMHostUserCondTrue,
                                             HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrue,
                                              DeviceLinuxTargetParallel));
    EXPECT_TRUE(
        isVariantApplicableInContext(LLVMHostUserCondTrue, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrue,
                                              DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo LLVMHostUserCondTrueCPU = LLVMHostUserCondTrue;
    LLVMHostUserCondTrueCPU.addTrait(TraitProperty::device_kind_cpu, "");
    EXPECT_TRUE(isVariantApplicableInContext(LLVMHostUserCondTrueCPU,
                                             HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrueCPU,
                                              DeviceLinuxTargetParallel));
    EXPECT_FALSE(
        isVariantApplicableInContext(LLVMHostUserCondTrueCPU, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrueCPU,
                                              DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo GPU;
    GPU.addTrait(TraitProperty::device_kind_gpu, "");
    EXPECT_FALSE(isVariantApplicableInContext(GPU, HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(GPU, DeviceLinuxTargetParallel));
    EXPECT_TRUE(isVariantApplicableInContext(GPU, HostNVPTXFor));
    EXPECT_TRUE(
        isVariantApplicableInContext(GPU, DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo NoHost;
    NoHost.addTrait(TraitProperty::device_kind_nohost, "");
    EXPECT_FALSE(
        isVariantApplicableInContext(NoHost, HostLinuxParallelParallel));
    EXPECT_TRUE(
        isVariantApplicableInContext(NoHost, DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(NoHost, HostNVPTXFor));
    EXPECT_TRUE(
        isVariantApplicableInContext(NoHost, DeviceNVPTXTargetTeamsParallel));
  }
  { // variants with all sets
    VariantMatchInfo DeviceArchArmParallel;
    DeviceArchArmParallel.addTrait(TraitProperty::construct_parallel_parallel,
                                   "");
    DeviceArchArmParallel.addTrait(TraitProperty::device_arch_arm, "");
    EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArmParallel,
                                              HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArmParallel,
                                              DeviceLinuxTargetParallel));
    EXPECT_FALSE(
        isVariantApplicableInContext(DeviceArchArmParallel, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(DeviceArchArmParallel,
                                              DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo LLVMHostUserCondTrueParallel;
    LLVMHostUserCondTrueParallel.addTrait(
        TraitProperty::implementation_vendor_llvm, "");
    LLVMHostUserCondTrueParallel.addTrait(TraitProperty::device_kind_host, "");
    LLVMHostUserCondTrueParallel.addTrait(TraitProperty::device_kind_any, "");
    LLVMHostUserCondTrueParallel.addTrait(TraitProperty::user_condition_true,
                                          "");
    LLVMHostUserCondTrueParallel.addTrait(
        TraitProperty::construct_parallel_parallel, "");
    EXPECT_TRUE(isVariantApplicableInContext(LLVMHostUserCondTrueParallel,
                                             HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrueParallel,
                                              DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrueParallel,
                                              HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(LLVMHostUserCondTrueParallel,
                                              DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo LLVMHostUserCondTrueParallelParallel =
        LLVMHostUserCondTrueParallel;
    LLVMHostUserCondTrueParallelParallel.addTrait(
        TraitProperty::construct_parallel_parallel, "");
    EXPECT_TRUE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallel, HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallel, DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallel, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallel, DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo LLVMHostUserCondTrueParallelParallelParallel =
        LLVMHostUserCondTrueParallelParallel;
    LLVMHostUserCondTrueParallelParallelParallel.addTrait(
        TraitProperty::construct_parallel_parallel, "");
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallelParallel,
        HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallelParallel,
        DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallelParallel, HostNVPTXFor));
    EXPECT_FALSE(isVariantApplicableInContext(
        LLVMHostUserCondTrueParallelParallelParallel,
        DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo GPUTargetTeams;
    GPUTargetTeams.addTrait(TraitProperty::construct_target_target, "");
    GPUTargetTeams.addTrait(TraitProperty::construct_teams_teams, "");
    GPUTargetTeams.addTrait(TraitProperty::device_kind_gpu, "");
    EXPECT_FALSE(isVariantApplicableInContext(GPUTargetTeams,
                                              HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(GPUTargetTeams,
                                              DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(GPUTargetTeams, HostNVPTXFor));
    EXPECT_TRUE(isVariantApplicableInContext(GPUTargetTeams,
                                             DeviceNVPTXTargetTeamsParallel));

    VariantMatchInfo GPUTargetParallel;
    GPUTargetParallel.addTrait(TraitProperty::construct_target_target, "");
    GPUTargetParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
    GPUTargetParallel.addTrait(TraitProperty::device_kind_gpu, "");
    EXPECT_FALSE(isVariantApplicableInContext(GPUTargetParallel,
                                              HostLinuxParallelParallel));
    EXPECT_FALSE(isVariantApplicableInContext(GPUTargetParallel,
                                              DeviceLinuxTargetParallel));
    EXPECT_FALSE(isVariantApplicableInContext(GPUTargetParallel, HostNVPTXFor));
    EXPECT_TRUE(isVariantApplicableInContext(GPUTargetParallel,
                                             DeviceNVPTXTargetTeamsParallel));
  }
}

TEST_F(OpenMPContextTest, ScoringSimple) {
  // TODO: Add scoring tests (via getBestVariantMatchForContext).
}

} // namespace
