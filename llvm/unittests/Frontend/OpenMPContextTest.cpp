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
                getOpenMPContextTraitSelectorName(TraitSelector::Enum),        \
                TraitSet::TraitSetEnum));                                      \
  EXPECT_EQ(Str, getOpenMPContextTraitSelectorName(                            \
                     getOpenMPContextTraitSelectorKind(                        \
                         Str, TraitSet::TraitSetEnum)));
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
  OMPContext HostLinux(false, Triple("x86_64-unknown-linux"), Triple(), -1);
  OMPContext DeviceLinux(true, Triple("x86_64-unknown-linux"), Triple(), -1);
  OMPContext HostNVPTX(false, Triple("nvptx64-nvidia-cuda"), Triple(), -1);
  OMPContext DeviceNVPTX(true, Triple("nvptx64-nvidia-cuda"), Triple(), -1);

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
  OMPContext HostLinuxParallelParallel(false, Triple("x86_64-unknown-linux"),
                                       Triple(), -1);
  HostLinuxParallelParallel.addTrait(
      TraitProperty::construct_parallel_parallel);
  HostLinuxParallelParallel.addTrait(
      TraitProperty::construct_parallel_parallel);
  OMPContext DeviceLinuxTargetParallel(true, Triple("x86_64-unknown-linux"),
                                       Triple(), -1);
  DeviceLinuxTargetParallel.addTrait(TraitProperty::construct_target_target);
  DeviceLinuxTargetParallel.addTrait(
      TraitProperty::construct_parallel_parallel);
  OMPContext HostNVPTXFor(false, Triple("nvptx64-nvidia-cuda"), Triple(), -1);
  HostNVPTXFor.addTrait(TraitProperty::construct_for_for);
  OMPContext DeviceNVPTXTargetTeamsParallel(true, Triple("nvptx64-nvidia-cuda"),
                                            Triple(), -1);
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

TEST_F(OpenMPContextTest, ApplicabilityISATraits) {
  struct SSE2Context : OMPContext {
    using OMPContext::OMPContext;
    bool matchesISATrait(StringRef ISA) const override { return ISA == "sse2"; }
  };
  SSE2Context Context(false, Triple("x86_64-unknown-linux"), Triple(), -1);

  // An unsupported ISA in one selector must not fail the other selector.
  for (auto [Supported, Unsupported] :
       {std::pair(TraitProperty::device_isa___ANY,
                  TraitProperty::target_device_isa___ANY),
        std::pair(TraitProperty::target_device_isa___ANY,
                  TraitProperty::device_isa___ANY)}) {
    VariantMatchInfo VMI;
    VMI.addTrait(Supported, "sse2");
    VMI.addTrait(Unsupported, "bogus_isa");
    EXPECT_FALSE(isVariantApplicableInContext(VMI, Context));
    VariantMatchInfo Any = VMI, None = VMI;
    Any.addTrait(TraitProperty::implementation_extension_match_any, "");
    EXPECT_TRUE(isVariantApplicableInContext(Any, Context));
    None.addTrait(TraitProperty::implementation_extension_match_none, "");
    EXPECT_FALSE(isVariantApplicableInContext(None, Context));
  }

  // Every name in one selector is still required.
  VariantMatchInfo TwoNames;
  TwoNames.addTrait(TraitProperty::device_isa___ANY, "sse2");
  TwoNames.addTrait(TraitProperty::device_isa___ANY, "bogus_isa");
  TwoNames.addTrait(TraitProperty::implementation_extension_match_any, "");
  EXPECT_FALSE(isVariantApplicableInContext(TwoNames, Context));
}

TEST_F(OpenMPContextTest, ScoringSimple) {
  // TODO: Add scoring tests (via getBestVariantMatchForContext).
}

TEST_F(OpenMPContextTest, ConstructContextStartsAtTarget) {
  OMPContext ParallelTarget(false, Triple("x86_64-unknown-linux"), Triple(),
                            -1);
  ParallelTarget.addTrait(TraitProperty::construct_parallel_parallel);
  ParallelTarget.addTrait(TraitProperty::construct_target_target);
  OMPContext ParallelTargetParallel(false, Triple("x86_64-unknown-linux"),
                                    Triple(), -1);
  ParallelTargetParallel.addTrait(TraitProperty::construct_parallel_parallel);
  ParallelTargetParallel.addTrait(TraitProperty::construct_target_target);
  ParallelTargetParallel.addTrait(TraitProperty::construct_parallel_parallel);

  EXPECT_FALSE(ParallelTarget.ActiveTraits.test(
      unsigned(TraitProperty::construct_parallel_parallel)));
  EXPECT_TRUE(ParallelTargetParallel.ActiveTraits.test(
      unsigned(TraitProperty::construct_parallel_parallel)));

  VariantMatchInfo Parallel, ParallelParallel, TargetParallel;
  Parallel.addTrait(TraitProperty::construct_parallel_parallel, "");
  ParallelParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
  ParallelParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
  TargetParallel.addTrait(TraitProperty::construct_target_target, "");
  TargetParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
  EXPECT_FALSE(isVariantApplicableInContext(Parallel, ParallelTarget));
  EXPECT_TRUE(isVariantApplicableInContext(Parallel, ParallelTargetParallel));
  EXPECT_FALSE(
      isVariantApplicableInContext(ParallelParallel, ParallelTargetParallel));
  EXPECT_TRUE(
      isVariantApplicableInContext(TargetParallel, ParallelTargetParallel));
}

TEST_F(OpenMPContextTest, StrictSubsetWithRepeatedConstruct) {
  OMPContext Context(false, Triple("x86_64-unknown-linux"), Triple(), -1);
  Context.addTrait(TraitProperty::construct_parallel_parallel);
  Context.addTrait(TraitProperty::construct_for_for);
  Context.addTrait(TraitProperty::construct_parallel_parallel);

  APInt Four(64, 4), One(64, 1);
  VariantMatchInfo ParallelFor, ParallelForParallel, ForParallel;
  ParallelFor.addTrait(TraitProperty::implementation_vendor_llvm, "", &Four);
  ParallelFor.addTrait(TraitProperty::construct_parallel_parallel, "");
  ParallelFor.addTrait(TraitProperty::construct_for_for, "");
  ParallelForParallel.addTrait(TraitProperty::implementation_vendor_llvm, "");
  ParallelForParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
  ParallelForParallel.addTrait(TraitProperty::construct_for_for, "");
  ParallelForParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
  ForParallel.addTrait(TraitProperty::implementation_vendor_llvm, "", &One);
  ForParallel.addTrait(TraitProperty::construct_for_for, "");
  ForParallel.addTrait(TraitProperty::construct_parallel_parallel, "");

  // All score 8 and require the same properties. Only the repeated construct
  // makes ParallelFor a strict subset of ParallelForParallel.
  SmallVector<VariantMatchInfo, 2> Candidates{ParallelFor, ParallelForParallel};
  EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 1);
  std::swap(Candidates[0], Candidates[1]);
  EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);

  // Sequences of the same length are not strict subsets of each other.
  Candidates = {ParallelFor, ForParallel};
  EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
  std::swap(Candidates[0], Candidates[1]);
  EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
}

TEST_F(OpenMPContextTest, RepeatedConstructWithConditionOrISA) {
  struct ISAContext : OMPContext {
    using OMPContext::OMPContext;
    bool matchesISATrait(StringRef) const override { return true; }
  } Context(false, Triple("x86_64-unknown-linux"), Triple(), -1);
  Context.addTrait(TraitProperty::construct_parallel_parallel);
  Context.addTrait(TraitProperty::construct_for_for);
  Context.addTrait(TraitProperty::construct_parallel_parallel);

  // Each pair ties and requires the same traits, but the conditions or ISA
  // names may differ, so clause order decides.
  APInt Four(64, 4), Twenty(64, 20), ThirtySix(64, 36);
  for (auto [Property, Score] :
       {std::pair(TraitProperty::user_condition_true, &Four),
        std::pair(TraitProperty::device_isa___ANY, &Twenty)}) {
    VariantMatchInfo ParallelFor, ParallelForParallel;
    ParallelFor.addTrait(TraitProperty::implementation_vendor_llvm, "", Score);
    ParallelFor.addTrait(Property, "sse");
    ParallelFor.addTrait(TraitProperty::construct_parallel_parallel, "");
    ParallelFor.addTrait(TraitProperty::construct_for_for, "");
    ParallelForParallel.addTrait(TraitProperty::implementation_vendor_llvm, "");
    ParallelForParallel.addTrait(Property, "sse2");
    ParallelForParallel.addTrait(TraitProperty::construct_parallel_parallel,
                                 "");
    ParallelForParallel.addTrait(TraitProperty::construct_for_for, "");
    ParallelForParallel.addTrait(TraitProperty::construct_parallel_parallel,
                                 "");
    SmallVector<VariantMatchInfo, 2> Candidates{ParallelFor,
                                                ParallelForParallel};
    EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
    std::swap(Candidates[0], Candidates[1]);
    EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);

    // With all of the shorter selector's ISA names, the longer one wins.
    if (Property == TraitProperty::device_isa___ANY) {
      Candidates[0].addTrait(Property, "sse");
      EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
      std::swap(Candidates[0], Candidates[1]);
      EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 1);

      // The same ISA names in swapped selectors are not subsets of each other.
      ParallelFor.addTrait(TraitProperty::implementation_vendor_llvm, "",
                           &ThirtySix);
      ParallelFor.addTrait(TraitProperty::target_device_isa___ANY, "sse2");
      ParallelForParallel.addTrait(TraitProperty::target_device_isa___ANY,
                                   "sse");
      Candidates = {ParallelFor, ParallelForParallel};
      EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
      std::swap(Candidates[0], Candidates[1]);
      EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
    }
  }
}

TEST_F(OpenMPContextTest, ConstructOrderInStrictSubset) {
  APInt Five(64, 5), Nineteen(64, 19);
  {
    OMPContext Context(false, Triple("x86_64-unknown-linux"), Triple(), -1);
    Context.addTrait(TraitProperty::construct_parallel_parallel);
    Context.addTrait(TraitProperty::construct_for_for);
    Context.addTrait(TraitProperty::construct_parallel_parallel);
    Context.addTrait(TraitProperty::construct_for_for);
    Context.addTrait(TraitProperty::construct_simd_simd);

    VariantMatchInfo ParallelFor, ForParallelSimd;
    ParallelFor.addTrait(TraitProperty::implementation_vendor_llvm, "",
                         &Nineteen);
    ParallelFor.addTrait(TraitProperty::construct_parallel_parallel, "");
    ParallelFor.addTrait(TraitProperty::construct_for_for, "");
    ForParallelSimd.addTrait(TraitProperty::implementation_vendor_llvm, "");
    ForParallelSimd.addTrait(TraitProperty::user_condition_true, "");
    ForParallelSimd.addTrait(TraitProperty::construct_for_for, "");
    ForParallelSimd.addTrait(TraitProperty::construct_parallel_parallel, "");
    ForParallelSimd.addTrait(TraitProperty::construct_simd_simd, "");

    // Both score 23. Each property of ParallelFor is also required by
    // ForParallelSimd, but its construct sequence is not a subsequence.
    SmallVector<VariantMatchInfo, 2> Candidates{ParallelFor, ForParallelSimd};
    EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
    std::swap(Candidates[0], Candidates[1]);
    EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
  }
  {
    OMPContext Context(false, Triple("x86_64-unknown-linux"), Triple(), -1);
    Context.addTrait(TraitProperty::construct_parallel_parallel);
    Context.addTrait(TraitProperty::construct_parallel_parallel);
    Context.addTrait(TraitProperty::construct_for_for);
    Context.addTrait(TraitProperty::construct_parallel_parallel);

    VariantMatchInfo ForParallel, ParallelParallelFor;
    ForParallel.addTrait(TraitProperty::implementation_vendor_llvm, "");
    ForParallel.addTrait(TraitProperty::construct_for_for, "");
    ForParallel.addTrait(TraitProperty::construct_parallel_parallel, "");
    ParallelParallelFor.addTrait(TraitProperty::implementation_vendor_llvm, "",
                                 &Five);
    ParallelParallelFor.addTrait(TraitProperty::construct_parallel_parallel,
                                 "");
    ParallelParallelFor.addTrait(TraitProperty::construct_parallel_parallel,
                                 "");
    ParallelParallelFor.addTrait(TraitProperty::construct_for_for, "");

    // Both score 13 and require the same properties. The shorter sequence is
    // not a subsequence of the longer one.
    SmallVector<VariantMatchInfo, 2> Candidates{ForParallel,
                                                ParallelParallelFor};
    EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
    std::swap(Candidates[0], Candidates[1]);
    EXPECT_EQ(getBestVariantMatchForContext(Candidates, Context), 0);
  }
}

} // namespace
