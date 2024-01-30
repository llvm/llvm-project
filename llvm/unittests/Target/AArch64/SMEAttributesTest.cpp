#include "Utils/AArch64SMEAttributes.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;
using SA = SMEAttrs;

std::unique_ptr<Module> parseIR(const char *IR) {
  static LLVMContext C;
  SMDiagnostic Err;
  return parseAssemblyString(IR, Err, C);
}

TEST(SMEAttributes, Constructors) {
  LLVMContext Context;

  ASSERT_TRUE(SA(*parseIR("declare void @foo()")->getFunction("foo"))
                  .hasNonStreamingInterfaceAndBody());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_sm_body\"")
                      ->getFunction("foo"))
                  .hasNonStreamingInterface());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_sm_enabled\"")
                      ->getFunction("foo"))
                  .hasStreamingInterface());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_sm_body\"")
                      ->getFunction("foo"))
                  .hasStreamingBody());

  ASSERT_TRUE(
      SA(*parseIR("declare void @foo() \"aarch64_pstate_sm_compatible\"")
              ->getFunction("foo"))
          .hasStreamingCompatibleInterface());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_za_shared\"")
                      ->getFunction("foo"))
                  .sharesZA());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_za_shared\"")
                      ->getFunction("foo"))
                  .hasSharedZAInterface());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_za_new\"")
                      ->getFunction("foo"))
                  .hasNewZABody());

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_pstate_za_preserved\"")
                      ->getFunction("foo"))
                  .preservesZA());

  ASSERT_TRUE(
      SA(*parseIR("declare void @foo() \"aarch64_in_zt0\"")->getFunction("foo"))
          .isInZT0());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_out_zt0\"")
                      ->getFunction("foo"))
                  .isOutZT0());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_inout_zt0\"")
                      ->getFunction("foo"))
                  .isInOutZT0());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_preserves_zt0\"")
                      ->getFunction("foo"))
                  .isPreservesZT0());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_new_zt0\"")
                      ->getFunction("foo"))
                  .isNewZT0());

  // Invalid combinations.
  EXPECT_DEBUG_DEATH(SA(SA::SM_Enabled | SA::SM_Compatible),
                     "SM_Enabled and SM_Compatible are mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZA_New | SA::ZA_Shared),
                     "ZA_New and ZA_Shared are mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZA_New | SA::ZA_Preserved),
                     "ZA_New and ZA_Preserved are mutually exclusive");

  // Test that the set() methods equally check validity.
  EXPECT_DEBUG_DEATH(SA(SA::SM_Enabled).set(SA::SM_Compatible),
                     "SM_Enabled and SM_Compatible are mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::SM_Compatible).set(SA::SM_Enabled),
                     "SM_Enabled and SM_Compatible are mutually exclusive");
}

TEST(SMEAttributes, Basics) {
  // Test PSTATE.SM interfaces.
  ASSERT_TRUE(SA(SA::Normal).hasNonStreamingInterfaceAndBody());
  ASSERT_TRUE(SA(SA::SM_Enabled).hasStreamingInterface());
  ASSERT_TRUE(SA(SA::SM_Body).hasStreamingBody());
  ASSERT_TRUE(SA(SA::SM_Body).hasNonStreamingInterface());
  ASSERT_FALSE(SA(SA::SM_Body).hasNonStreamingInterfaceAndBody());
  ASSERT_FALSE(SA(SA::SM_Body).hasStreamingInterface());
  ASSERT_TRUE(SA(SA::SM_Compatible).hasStreamingCompatibleInterface());
  ASSERT_TRUE(
      SA(SA::SM_Compatible | SA::SM_Body).hasStreamingCompatibleInterface());
  ASSERT_TRUE(SA(SA::SM_Compatible | SA::SM_Body).hasStreamingBody());
  ASSERT_FALSE(SA(SA::SM_Compatible | SA::SM_Body).hasNonStreamingInterface());

  // Test PSTATE.ZA interfaces.
  ASSERT_FALSE(SA(SA::ZA_Shared).hasPrivateZAInterface());
  ASSERT_TRUE(SA(SA::ZA_Shared).hasSharedZAInterface());
  ASSERT_TRUE(SA(SA::ZA_Shared).sharesZA());
  ASSERT_TRUE(SA(SA::ZA_Shared).hasZAState());
  ASSERT_FALSE(SA(SA::ZA_Shared).preservesZA());
  ASSERT_TRUE(SA(SA::ZA_Shared | SA::ZA_Preserved).preservesZA());
  ASSERT_FALSE(SA(SA::ZA_Shared).sharesZT0());
  ASSERT_FALSE(SA(SA::ZA_Shared).hasZT0State());

  ASSERT_TRUE(SA(SA::ZA_New).hasPrivateZAInterface());
  ASSERT_FALSE(SA(SA::ZA_New).hasSharedZAInterface());
  ASSERT_TRUE(SA(SA::ZA_New).hasNewZABody());
  ASSERT_TRUE(SA(SA::ZA_New).hasZAState());
  ASSERT_FALSE(SA(SA::ZA_New).preservesZA());
  ASSERT_FALSE(SA(SA::ZA_New).sharesZT0());
  ASSERT_FALSE(SA(SA::ZA_New).hasZT0State());

  ASSERT_TRUE(SA(SA::Normal).hasPrivateZAInterface());
  ASSERT_FALSE(SA(SA::Normal).hasSharedZAInterface());
  ASSERT_FALSE(SA(SA::Normal).hasNewZABody());
  ASSERT_FALSE(SA(SA::Normal).hasZAState());
  ASSERT_FALSE(SA(SA::Normal).preservesZA());

  // Test ZT0 State interfaces
  SA ZT0_In = SA(SA::encodeZT0State(SA::StateValue::In));
  ASSERT_TRUE(ZT0_In.isInZT0());
  ASSERT_FALSE(ZT0_In.isOutZT0());
  ASSERT_FALSE(ZT0_In.isInOutZT0());
  ASSERT_FALSE(ZT0_In.isPreservesZT0());
  ASSERT_FALSE(ZT0_In.isNewZT0());
  ASSERT_TRUE(ZT0_In.sharesZT0());
  ASSERT_TRUE(ZT0_In.hasZT0State());
  ASSERT_TRUE(ZT0_In.hasSharedZAInterface());
  ASSERT_FALSE(ZT0_In.hasPrivateZAInterface());

  SA ZT0_Out = SA(SA::encodeZT0State(SA::StateValue::Out));
  ASSERT_TRUE(ZT0_Out.isOutZT0());
  ASSERT_FALSE(ZT0_Out.isInZT0());
  ASSERT_FALSE(ZT0_Out.isInOutZT0());
  ASSERT_FALSE(ZT0_Out.isPreservesZT0());
  ASSERT_FALSE(ZT0_Out.isNewZT0());
  ASSERT_TRUE(ZT0_Out.sharesZT0());
  ASSERT_TRUE(ZT0_Out.hasZT0State());
  ASSERT_TRUE(ZT0_Out.hasSharedZAInterface());
  ASSERT_FALSE(ZT0_Out.hasPrivateZAInterface());

  SA ZT0_InOut = SA(SA::encodeZT0State(SA::StateValue::InOut));
  ASSERT_TRUE(ZT0_InOut.isInOutZT0());
  ASSERT_FALSE(ZT0_InOut.isInZT0());
  ASSERT_FALSE(ZT0_InOut.isOutZT0());
  ASSERT_FALSE(ZT0_InOut.isPreservesZT0());
  ASSERT_FALSE(ZT0_InOut.isNewZT0());
  ASSERT_TRUE(ZT0_InOut.sharesZT0());
  ASSERT_TRUE(ZT0_InOut.hasZT0State());
  ASSERT_TRUE(ZT0_InOut.hasSharedZAInterface());
  ASSERT_FALSE(ZT0_InOut.hasPrivateZAInterface());

  SA ZT0_Preserved = SA(SA::encodeZT0State(SA::StateValue::Preserved));
  ASSERT_TRUE(ZT0_Preserved.isPreservesZT0());
  ASSERT_FALSE(ZT0_Preserved.isInZT0());
  ASSERT_FALSE(ZT0_Preserved.isOutZT0());
  ASSERT_FALSE(ZT0_Preserved.isInOutZT0());
  ASSERT_FALSE(ZT0_Preserved.isNewZT0());
  ASSERT_TRUE(ZT0_Preserved.sharesZT0());
  ASSERT_TRUE(ZT0_Preserved.hasZT0State());
  ASSERT_TRUE(ZT0_Preserved.hasSharedZAInterface());
  ASSERT_FALSE(ZT0_Preserved.hasPrivateZAInterface());

  SA ZT0_New = SA(SA::encodeZT0State(SA::StateValue::New));
  ASSERT_TRUE(ZT0_New.isNewZT0());
  ASSERT_FALSE(ZT0_New.isInZT0());
  ASSERT_FALSE(ZT0_New.isOutZT0());
  ASSERT_FALSE(ZT0_New.isInOutZT0());
  ASSERT_FALSE(ZT0_New.isPreservesZT0());
  ASSERT_FALSE(ZT0_New.sharesZT0());
  ASSERT_TRUE(ZT0_New.hasZT0State());
  ASSERT_FALSE(ZT0_New.hasSharedZAInterface());
  ASSERT_TRUE(ZT0_New.hasPrivateZAInterface());

  ASSERT_FALSE(SA(SA::Normal).isInZT0());
  ASSERT_FALSE(SA(SA::Normal).isOutZT0());
  ASSERT_FALSE(SA(SA::Normal).isInOutZT0());
  ASSERT_FALSE(SA(SA::Normal).isPreservesZT0());
  ASSERT_FALSE(SA(SA::Normal).isNewZT0());
  ASSERT_FALSE(SA(SA::Normal).sharesZT0());
  ASSERT_FALSE(SA(SA::Normal).hasZT0State());
}

TEST(SMEAttributes, Transitions) {
  // Normal -> Normal
  ASSERT_FALSE(SA(SA::Normal).requiresSMChange(SA(SA::Normal)));
  // Normal -> Normal + LocallyStreaming
  ASSERT_FALSE(SA(SA::Normal).requiresSMChange(SA(SA::Normal | SA::SM_Body)));
  ASSERT_EQ(*SA(SA::Normal)
                 .requiresSMChange(SA(SA::Normal | SA::SM_Body),
                                   /*BodyOverridesInterface=*/true),
            true);

  // Normal -> Streaming
  ASSERT_EQ(*SA(SA::Normal).requiresSMChange(SA(SA::SM_Enabled)), true);
  // Normal -> Streaming + LocallyStreaming
  ASSERT_EQ(*SA(SA::Normal).requiresSMChange(SA(SA::SM_Enabled | SA::SM_Body)),
            true);
  ASSERT_EQ(*SA(SA::Normal)
                 .requiresSMChange(SA(SA::SM_Enabled | SA::SM_Body),
                                   /*BodyOverridesInterface=*/true),
            true);

  // Normal -> Streaming-compatible
  ASSERT_FALSE(SA(SA::Normal).requiresSMChange(SA(SA::SM_Compatible)));
  // Normal -> Streaming-compatible + LocallyStreaming
  ASSERT_FALSE(
      SA(SA::Normal).requiresSMChange(SA(SA::SM_Compatible | SA::SM_Body)));
  ASSERT_EQ(*SA(SA::Normal)
                 .requiresSMChange(SA(SA::SM_Compatible | SA::SM_Body),
                                   /*BodyOverridesInterface=*/true),
            true);

  // Streaming -> Normal
  ASSERT_EQ(*SA(SA::SM_Enabled).requiresSMChange(SA(SA::Normal)), false);
  // Streaming -> Normal + LocallyStreaming
  ASSERT_EQ(*SA(SA::SM_Enabled).requiresSMChange(SA(SA::Normal | SA::SM_Body)),
            false);
  ASSERT_FALSE(SA(SA::SM_Enabled)
                   .requiresSMChange(SA(SA::Normal | SA::SM_Body),
                                     /*BodyOverridesInterface=*/true));

  // Streaming -> Streaming
  ASSERT_FALSE(SA(SA::SM_Enabled).requiresSMChange(SA(SA::SM_Enabled)));
  // Streaming -> Streaming + LocallyStreaming
  ASSERT_FALSE(
      SA(SA::SM_Enabled).requiresSMChange(SA(SA::SM_Enabled | SA::SM_Body)));
  ASSERT_FALSE(SA(SA::SM_Enabled)
                   .requiresSMChange(SA(SA::SM_Enabled | SA::SM_Body),
                                     /*BodyOverridesInterface=*/true));

  // Streaming -> Streaming-compatible
  ASSERT_FALSE(SA(SA::SM_Enabled).requiresSMChange(SA(SA::SM_Compatible)));
  // Streaming -> Streaming-compatible + LocallyStreaming
  ASSERT_FALSE(
      SA(SA::SM_Enabled).requiresSMChange(SA(SA::SM_Compatible | SA::SM_Body)));
  ASSERT_FALSE(SA(SA::SM_Enabled)
                   .requiresSMChange(SA(SA::SM_Compatible | SA::SM_Body),
                                     /*BodyOverridesInterface=*/true));

  // Streaming-compatible -> Normal
  ASSERT_EQ(*SA(SA::SM_Compatible).requiresSMChange(SA(SA::Normal)), false);
  ASSERT_EQ(
      *SA(SA::SM_Compatible).requiresSMChange(SA(SA::Normal | SA::SM_Body)),
      false);
  ASSERT_EQ(*SA(SA::SM_Compatible)
                 .requiresSMChange(SA(SA::Normal | SA::SM_Body),
                                   /*BodyOverridesInterface=*/true),
            true);

  // Streaming-compatible -> Streaming
  ASSERT_EQ(*SA(SA::SM_Compatible).requiresSMChange(SA(SA::SM_Enabled)), true);
  // Streaming-compatible -> Streaming + LocallyStreaming
  ASSERT_EQ(
      *SA(SA::SM_Compatible).requiresSMChange(SA(SA::SM_Enabled | SA::SM_Body)),
      true);
  ASSERT_EQ(*SA(SA::SM_Compatible)
                 .requiresSMChange(SA(SA::SM_Enabled | SA::SM_Body),
                                   /*BodyOverridesInterface=*/true),
            true);

  // Streaming-compatible -> Streaming-compatible
  ASSERT_FALSE(SA(SA::SM_Compatible).requiresSMChange(SA(SA::SM_Compatible)));
  // Streaming-compatible -> Streaming-compatible + LocallyStreaming
  ASSERT_FALSE(SA(SA::SM_Compatible)
                   .requiresSMChange(SA(SA::SM_Compatible | SA::SM_Body)));
  ASSERT_EQ(*SA(SA::SM_Compatible)
                 .requiresSMChange(SA(SA::SM_Compatible | SA::SM_Body),
                                   /*BodyOverridesInterface=*/true),
            true);
}
