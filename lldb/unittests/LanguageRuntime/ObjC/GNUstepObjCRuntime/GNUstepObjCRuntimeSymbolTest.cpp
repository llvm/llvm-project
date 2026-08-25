//===-- GNUstepObjCRuntimeSymbolTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/LanguageRuntime/ObjC/GNUstepObjCRuntime/GNUstepObjCRuntime.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct Parsed {
  bool matched;
  std::string class_name;
  std::string ivar_name;
};

Parsed Parse(llvm::StringRef symbol) {
  llvm::StringRef class_name, ivar_name;
  const bool matched =
      GNUstepObjCRuntime::ParseIvarOffsetSymbol(symbol, class_name, ivar_name);
  return {matched, class_name.str(), ivar_name.str()};
}
} // namespace

TEST(GNUstepObjCRuntimeSymbolTest, SplitsClassAndIvar) {
  const Parsed p = Parse("__objc_ivar_offset_Holder.value.i");
  EXPECT_TRUE(p.matched);
  EXPECT_EQ(p.class_name, "Holder");
  EXPECT_EQ(p.ivar_name, "value");
}

TEST(GNUstepObjCRuntimeSymbolTest, IgnoresTheTypeEncoding) {
  // The encoding is mangled differently per object format - '@' becomes \1 and
  // '=' becomes \2 - and a struct encoding can be arbitrarily long. None of it
  // identifies the ivar, so all of it is ignored.
  for (llvm::StringRef encoding :
       {"i", "d", "\1", "{_NSRange\2QQ}", "^v", "[16c]"}) {
    const std::string symbol =
        ("__objc_ivar_offset_Widget.field." + encoding).str();
    const Parsed p = Parse(symbol);
    EXPECT_TRUE(p.matched) << symbol;
    EXPECT_EQ(p.class_name, "Widget") << symbol;
    EXPECT_EQ(p.ivar_name, "field") << symbol;
  }
}

TEST(GNUstepObjCRuntimeSymbolTest, RejectsOtherSymbols) {
  // Apple's spelling must not be claimed by this runtime, and neither should
  // the class symbol, which is handled separately.
  for (llvm::StringRef symbol :
       {"OBJC_IVAR_$_Holder.value", "._OBJC_CLASS_Holder",
        "$_OBJC_CLASS_Holder", "__objc_ivar_offset_", "objc_msg_lookup", ""}) {
    const Parsed p = Parse(symbol);
    EXPECT_FALSE(p.matched) << symbol;
  }
}

TEST(GNUstepObjCRuntimeSymbolTest, RejectsAMissingComponent) {
  // The gnustep-2.x spelling always carries an encoding. A two-part name is
  // therefore some other ABI's - the v1 form, or __objc_ivar_offset_value_ -
  // and reading it as class "Holder", ivar "value" would resolve to an
  // address this runtime has no business supplying.
  EXPECT_FALSE(Parse("__objc_ivar_offset_Holder.value").matched);
  EXPECT_FALSE(Parse("__objc_ivar_offset_Holder").matched);
  EXPECT_FALSE(Parse("__objc_ivar_offset_.value.i").matched);
  EXPECT_FALSE(Parse("__objc_ivar_offset_Holder..i").matched);
}
