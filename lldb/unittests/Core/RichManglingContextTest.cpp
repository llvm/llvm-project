//===-- RichManglingContextTest.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/RichManglingContext.h"

#include "lldb/Utility/ConstString.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

TEST(RichManglingContextTest, Basic) {
  RichManglingContext RMC;
  ConstString mangled("_ZN3foo3barEv");
  EXPECT_TRUE(RMC.FromItaniumName(mangled));

  EXPECT_TRUE(RMC.IsFunction());
  EXPECT_FALSE(RMC.IsCtorOrDtor());

  RMC.ParseFunctionDeclContextName();
  EXPECT_EQ("foo", RMC.GetBufferRef());

  RMC.ParseFunctionBaseName();
  EXPECT_EQ("bar", RMC.GetBufferRef());

  RMC.ParseFullName();
  EXPECT_EQ("foo::bar()", RMC.GetBufferRef());
}

TEST(RichManglingContextTest, FromCxxMethodName) {
  RichManglingContext ItaniumRMC;
  ConstString mangled("_ZN3foo3barEv");
  EXPECT_TRUE(ItaniumRMC.FromItaniumName(mangled));

  RichManglingContext CxxMethodRMC;
  ConstString demangled("foo::bar()");
  EXPECT_TRUE(CxxMethodRMC.FromCxxMethodName(demangled));

  EXPECT_TRUE(ItaniumRMC.IsFunction() == CxxMethodRMC.IsFunction());
  EXPECT_TRUE(ItaniumRMC.IsCtorOrDtor() == CxxMethodRMC.IsCtorOrDtor());

  ItaniumRMC.ParseFunctionDeclContextName();
  CxxMethodRMC.ParseFunctionDeclContextName();
  EXPECT_TRUE(ItaniumRMC.GetBufferRef() == CxxMethodRMC.GetBufferRef());

  ItaniumRMC.ParseFunctionBaseName();
  CxxMethodRMC.ParseFunctionBaseName();
  EXPECT_TRUE(ItaniumRMC.GetBufferRef() == CxxMethodRMC.GetBufferRef());

  ItaniumRMC.ParseFullName();
  CxxMethodRMC.ParseFullName();
  EXPECT_TRUE(ItaniumRMC.GetBufferRef() == CxxMethodRMC.GetBufferRef());

  // Construct with a random name.
  {
    RichManglingContext CxxMethodRMC;
    EXPECT_TRUE(CxxMethodRMC.FromCxxMethodName(ConstString("X")));

    // We expect it is not a function.
    EXPECT_FALSE(CxxMethodRMC.IsFunction());
  }

  // Construct with a function without a context.
  {
    RichManglingContext CxxMethodRMC;
    EXPECT_TRUE(CxxMethodRMC.FromCxxMethodName(
        ConstString("void * operator new(unsigned __int64)")));

    // We expect it is a function.
    EXPECT_TRUE(CxxMethodRMC.IsFunction());

    // We expect its context is empty.
    CxxMethodRMC.ParseFunctionDeclContextName();
    EXPECT_TRUE(CxxMethodRMC.GetBufferRef().empty());
  }
}

TEST(RichManglingContextTest, SwitchProvider) {
  RichManglingContext RMC;
  llvm::StringRef mangled = "_ZN3foo3barEv";
  llvm::StringRef demangled = "foo::bar()";

  EXPECT_TRUE(RMC.FromItaniumName(ConstString(mangled)));
  RMC.ParseFullName();
  EXPECT_EQ("foo::bar()", RMC.GetBufferRef());

  EXPECT_TRUE(RMC.FromCxxMethodName(ConstString(demangled)));
  RMC.ParseFullName();
  EXPECT_EQ("foo::bar()", RMC.GetBufferRef());

  EXPECT_TRUE(RMC.FromItaniumName(ConstString(mangled)));
  RMC.ParseFullName();
  EXPECT_EQ("foo::bar()", RMC.GetBufferRef());
}

TEST(RichManglingContextTest, IPDRealloc) {
  // The demangled name should fit into the Itanium default buffer.
  const char *ShortMangled = "_ZN3foo3barEv";

  // The demangled name for this will certainly not fit into the default buffer.
  const char *LongMangled =
      "_ZNK3shk6detail17CallbackPublisherIZNS_5ThrowERKNSt15__exception_"
      "ptr13exception_ptrEEUlOT_E_E9SubscribeINS0_9ConcatMapINS0_"
      "18CallbackSubscriberIZNS_6GetAllIiNS1_IZZNS_9ConcatMapIZNS_6ConcatIJNS1_"
      "IZZNS_3MapIZZNS_7IfEmptyIS9_EEDaS7_ENKUlS6_E_clINS1_IZZNS_4TakeIiEESI_"
      "S7_ENKUlS6_E_clINS1_IZZNS_6FilterIZNS_9ElementAtEmEUlS7_E_EESI_S7_"
      "ENKUlS6_E_clINS1_IZZNSL_ImEESI_S7_ENKUlS6_E_clINS1_IZNS_4FromINS0_"
      "22InfiniteRangeContainerIiEEEESI_S7_EUlS7_E_EEEESI_S6_EUlS7_E_EEEESI_S6_"
      "EUlS7_E_EEEESI_S6_EUlS7_E_EEEESI_S6_EUlS7_E_EESI_S7_ENKUlS6_E_clIS14_"
      "EESI_S6_EUlS7_E_EERNS1_IZZNSH_IS9_EESI_S7_ENKSK_IS14_EESI_S6_EUlS7_E0_"
      "EEEEESI_DpOT_EUlS7_E_EESI_S7_ENKUlS6_E_clINS1_IZNS_5StartIJZNS_"
      "4JustIJS19_S1C_EEESI_S1F_EUlvE_ZNS1K_IJS19_S1C_EEESI_S1F_EUlvE0_EEESI_"
      "S1F_EUlS7_E_EEEESI_S6_EUlS7_E_EEEESt6vectorIS6_SaIS6_EERKT0_NS_"
      "12ElementCountEbEUlS7_E_ZNSD_IiS1Q_EES1T_S1W_S1X_bEUlOS3_E_ZNSD_IiS1Q_"
      "EES1T_S1W_S1X_bEUlvE_EES1G_S1O_E25ConcatMapValuesSubscriberEEEDaS7_";

  RichManglingContext RMC;

  // Demangle the short one.
  EXPECT_TRUE(RMC.FromItaniumName(ConstString(ShortMangled)));
  RMC.ParseFullName();
  const char *ShortDemangled = RMC.GetBufferRef().data();

  // Demangle the long one.
  EXPECT_TRUE(RMC.FromItaniumName(ConstString(LongMangled)));
  RMC.ParseFullName();
  const char *LongDemangled = RMC.GetBufferRef().data();

  // Make sure a new buffer was allocated or the default buffer was extended.
  bool AllocatedNewBuffer = (ShortDemangled != LongDemangled);
  bool ExtendedExistingBuffer = (strlen(LongDemangled) > 2048);
  EXPECT_TRUE(AllocatedNewBuffer || ExtendedExistingBuffer);
}
