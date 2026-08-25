//===-- GNUstepObjCDeclVendorTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/LanguageRuntime/ObjC/GNUstepObjCRuntime/GNUstepObjCDeclVendor.h"
#include "gtest/gtest.h"

#include <vector>

using namespace lldb_private;

namespace {
/// Return type followed by every argument, so a whole split is one literal.
std::vector<std::string> Split(llvm::StringRef types) {
  MethodTypeSplitter splitter(types);
  if (!splitter.IsValid())
    return {};
  std::vector<std::string> out{splitter.GetReturnType().str()};
  for (size_t i = 0; i < splitter.GetNumArguments(); ++i)
    out.push_back(splitter.GetArgumentType(i).str());
  return out;
}
} // namespace

TEST(GNUstepObjCDeclVendorTest, DropsArgumentFrameOffsets) {
  // "v16@0:8": void return, self at 0, _cmd at 8, no declared parameters.
  EXPECT_EQ(Split("v16@0:8"), (std::vector<std::string>{"v"}));
  EXPECT_EQ(Split("i24@0:8i16"), (std::vector<std::string>{"i", "i"}));
  EXPECT_EQ(Split("v32@0:8i16f24"), (std::vector<std::string>{"v", "i", "f"}));
}

TEST(GNUstepObjCDeclVendorTest, KeepsDigitsInsideAClassName) {
  // clang's extended encoding spells an object parameter with its class name
  // (setEncodeClassNames), and a digit in that name is not an offset. Reading
  // it as one truncates the type and the method is silently dropped.
  EXPECT_EQ(Split("i24@0:8@\"SHA256\"16"),
            (std::vector<std::string>{"i", "@\"SHA256\""}));
  EXPECT_EQ(Split("v24@0:8@\"OAuth2Client\"16"),
            (std::vector<std::string>{"v", "@\"OAuth2Client\""}));
  // Two of them, so a mis-split would also shift the argument order.
  EXPECT_EQ(Split("v32@0:8@\"MD5\"16@\"Base64\"24"),
            (std::vector<std::string>{"v", "@\"MD5\"", "@\"Base64\""}));
}

TEST(GNUstepObjCDeclVendorTest, KeepsDigitsInsideAProtocolQualifier) {
  EXPECT_EQ(Split("v24@0:8@\"<NSCoding>\"16"),
            (std::vector<std::string>{"v", "@\"<NSCoding>\""}));
  EXPECT_EQ(Split("v24@0:8@\"<OAuth2>\"16"),
            (std::vector<std::string>{"v", "@\"<OAuth2>\""}));
}

TEST(GNUstepObjCDeclVendorTest, KeepsDigitsInsideAggregates) {
  EXPECT_EQ(Split("v24@0:8{CGPoint=dd}16"),
            (std::vector<std::string>{"v", "{CGPoint=dd}"}));
  // An array's element count is part of the type.
  EXPECT_EQ(Split("v24@0:8[16c]16"), (std::vector<std::string>{"v", "[16c]"}));
  // Nested, and with a class name inside carrying its own digits.
  EXPECT_EQ(Split("v24@0:8{Outer={Inner=i}@\"SHA256\"}16"),
            (std::vector<std::string>{"v", "{Outer={Inner=i}@\"SHA256\"}"}));
}

TEST(GNUstepObjCDeclVendorTest, KeepsABlockSignatureTogether) {
  // The extended encoding writes a block's own signature inline as `@?<...>`
  // (setEncodeBlockParameters); it is one type however many digits it holds.
  EXPECT_EQ(Split("v24@0:8@?<v@?@\"SHA256\">16"),
            (std::vector<std::string>{"v", "@?<v@?@\"SHA256\">"}));
}

TEST(GNUstepObjCDeclVendorTest, RejectsAnUnterminatedQuote) {
  // Truncated or corrupt metadata: decline rather than report a class name
  // that runs to the end of the buffer.
  MethodTypeSplitter splitter("i24@0:8@\"SHA256");
  EXPECT_FALSE(splitter.IsValid());
}

TEST(GNUstepObjCDeclVendorTest, RejectsTooFewComponents) {
  // Anything shorter than return, self and _cmd is not a method encoding.
  EXPECT_FALSE(MethodTypeSplitter("").IsValid());
  EXPECT_FALSE(MethodTypeSplitter("v").IsValid());
  EXPECT_FALSE(MethodTypeSplitter("v16@0").IsValid());
}
