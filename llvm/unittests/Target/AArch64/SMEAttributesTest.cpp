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

  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_sme_zt0_in\"")
                      ->getFunction("foo"))
                  .isZT0In());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_sme_zt0_out\"")
                      ->getFunction("foo"))
                  .isZT0Out());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_sme_zt0_inout\"")
                      ->getFunction("foo"))
                  .isZT0InOut());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_sme_zt0_preserved\"")
                      ->getFunction("foo"))
                  .preservesZT0());
  ASSERT_TRUE(SA(*parseIR("declare void @foo() \"aarch64_sme_zt0_new\"")
                      ->getFunction("foo"))
                  .hasNewZT0Body());

  // Invalid combinations.
  EXPECT_DEBUG_DEATH(SA(SA::SM_Enabled | SA::SM_Compatible),
                     "SM_Enabled and SM_Compatible are mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZA_New | SA::ZA_Shared),
                     "ZA_New and ZA_Shared are mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZA_New | SA::ZA_Preserved),
                     "ZA_New and ZA_Preserved are mutually exclusive");

  EXPECT_DEBUG_DEATH(SA(SA::ZT0_New | SA::ZT0_In),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_New | SA::ZT0_Out),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_New | SA::ZT0_InOut),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_New | SA::ZT0_Preserved),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");

  EXPECT_DEBUG_DEATH(SA(SA::ZT0_In | SA::ZT0_Out),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_In | SA::ZT0_InOut),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_Out | SA::ZT0_InOut),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");

  EXPECT_DEBUG_DEATH(SA(SA::ZT0_Preserved | SA::ZT0_In),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_Preserved | SA::ZT0_Out),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");
  EXPECT_DEBUG_DEATH(SA(SA::ZT0_Preserved | SA::ZT0_InOut),
                     "ZT0_New, ZT0_In, ZT0_Out, ZT0_InOut and ZT0_Preserved "
                     "are all \" \"mutually exclusive");

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

  ASSERT_TRUE(SA(SA::ZA_New).hasPrivateZAInterface());
  ASSERT_FALSE(SA(SA::ZA_New).hasSharedZAInterface());
  ASSERT_TRUE(SA(SA::ZA_New).hasNewZABody());
  ASSERT_TRUE(SA(SA::ZA_New).hasZAState());
  ASSERT_FALSE(SA(SA::ZA_New).preservesZA());

  ASSERT_TRUE(SA(SA::Normal).hasPrivateZAInterface());
  ASSERT_FALSE(SA(SA::Normal).hasSharedZAInterface());
  ASSERT_FALSE(SA(SA::Normal).hasNewZABody());
  ASSERT_FALSE(SA(SA::Normal).hasZAState());
  ASSERT_FALSE(SA(SA::Normal).preservesZA());

  // Test ZT0 State interfaces
  ASSERT_TRUE(SA(SA::ZT0_In).isZT0In());
  ASSERT_FALSE(SA(SA::ZT0_In).isZT0Out());
  ASSERT_FALSE(SA(SA::ZT0_In).isZT0InOut());
  ASSERT_FALSE(SA(SA::ZT0_In).preservesZT0());
  ASSERT_FALSE(SA(SA::ZT0_In).hasNewZT0Body());
  ASSERT_TRUE(SA(SA::ZT0_In).sharesZT0());
  ASSERT_TRUE(SA(SA::ZT0_In).hasZT0State());
  ASSERT_TRUE(SA(SA::ZT0_In).hasSharedZAInterface());
  ASSERT_FALSE(SA(SA::ZT0_In).hasPrivateZAInterface());

  ASSERT_TRUE(SA(SA::ZT0_Out).isZT0Out());
  ASSERT_FALSE(SA(SA::ZT0_Out).isZT0In());
  ASSERT_FALSE(SA(SA::ZT0_Out).isZT0InOut());
  ASSERT_FALSE(SA(SA::ZT0_Out).preservesZT0());
  ASSERT_FALSE(SA(SA::ZT0_Out).hasNewZT0Body());
  ASSERT_TRUE(SA(SA::ZT0_Out).sharesZT0());
  ASSERT_TRUE(SA(SA::ZT0_Out).hasZT0State());
  ASSERT_TRUE(SA(SA::ZT0_Out).hasSharedZAInterface());
  ASSERT_FALSE(SA(SA::ZT0_Out).hasPrivateZAInterface());

  ASSERT_TRUE(SA(SA::ZT0_InOut).isZT0InOut());
  ASSERT_FALSE(SA(SA::ZT0_InOut).isZT0In());
  ASSERT_FALSE(SA(SA::ZT0_InOut).isZT0Out());
  ASSERT_FALSE(SA(SA::ZT0_InOut).preservesZT0());
  ASSERT_FALSE(SA(SA::ZT0_InOut).hasNewZT0Body());
  ASSERT_TRUE(SA(SA::ZT0_InOut).sharesZT0());
  ASSERT_TRUE(SA(SA::ZT0_InOut).hasZT0State());
  ASSERT_TRUE(SA(SA::ZT0_InOut).hasSharedZAInterface());
  ASSERT_FALSE(SA(SA::ZT0_InOut).hasPrivateZAInterface());

  ASSERT_TRUE(SA(SA::ZT0_Preserved).preservesZT0());
  ASSERT_FALSE(SA(SA::ZT0_Preserved).isZT0In());
  ASSERT_FALSE(SA(SA::ZT0_Preserved).isZT0Out());
  ASSERT_FALSE(SA(SA::ZT0_Preserved).isZT0InOut());
  ASSERT_FALSE(SA(SA::ZT0_Preserved).hasNewZT0Body());
  ASSERT_TRUE(SA(SA::ZT0_Preserved).sharesZT0());
  ASSERT_TRUE(SA(SA::ZT0_Preserved).hasZT0State());
  ASSERT_TRUE(SA(SA::ZT0_Preserved).hasSharedZAInterface());
  ASSERT_FALSE(SA(SA::ZT0_Preserved).hasPrivateZAInterface());

  ASSERT_TRUE(SA(SA::ZT0_New).hasNewZT0Body());
  ASSERT_FALSE(SA(SA::ZT0_New).isZT0In());
  ASSERT_FALSE(SA(SA::ZT0_New).isZT0Out());
  ASSERT_FALSE(SA(SA::ZT0_New).isZT0InOut());
  ASSERT_FALSE(SA(SA::ZT0_New).preservesZT0());
  ASSERT_FALSE(SA(SA::ZT0_New).sharesZT0());
  ASSERT_TRUE(SA(SA::ZT0_New).hasZT0State());
  ASSERT_FALSE(SA(SA::ZT0_New).hasSharedZAInterface());
  ASSERT_TRUE(SA(SA::ZT0_New).hasPrivateZAInterface());

  ASSERT_FALSE(SA(SA::Normal).isZT0In());
  ASSERT_FALSE(SA(SA::Normal).isZT0Out());
  ASSERT_FALSE(SA(SA::Normal).isZT0InOut());
  ASSERT_FALSE(SA(SA::Normal).preservesZT0());
  ASSERT_FALSE(SA(SA::Normal).hasNewZT0Body());
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
