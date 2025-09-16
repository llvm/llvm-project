//===- GlobalObjectTest.cpp - Global object unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/GlobalObject.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
using namespace llvm;
namespace {
using testing::Eq;
using testing::Optional;
using testing::StrEq;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("GlobalObjectTests", errs());
  return Mod;
}

static LLVMContext C;
static std::unique_ptr<Module> M;

class GlobalObjectTest : public testing::Test {
public:
  static void SetUpTestSuite() {
    M = parseIR(C, R"(
@foo = global i32 3, !section_prefix !0
@bar = global i32 0

!0 = !{!"section_prefix", !"hot"}
)");
  }
};

TEST_F(GlobalObjectTest, SectionPrefix) {
  GlobalVariable *Foo = M->getGlobalVariable("foo");

  // Initial section prefix is hot.
  ASSERT_NE(Foo, nullptr);
  ASSERT_THAT(Foo->getSectionPrefix(), Optional(StrEq("hot")));

  // Test that set method returns false since existing section prefix is hot.
  EXPECT_FALSE(Foo->setSectionPrefix("hot"));

  // Set prefix from hot to unlikely.
  Foo->setSectionPrefix("unlikely");
  EXPECT_THAT(Foo->getSectionPrefix(), Optional(StrEq("unlikely")));

  // Set prefix to empty is the same as clear.
  Foo->setSectionPrefix("");
  // Test that section prefix is cleared.
  EXPECT_THAT(Foo->getSectionPrefix(), Eq(std::nullopt));

  GlobalVariable *Bar = M->getGlobalVariable("bar");

  // Initial section prefix is empty.
  ASSERT_NE(Bar, nullptr);
  ASSERT_THAT(Bar->getSectionPrefix(), Eq(std::nullopt));

  // Test that set method returns false since Bar doesn't have prefix metadata.
  EXPECT_FALSE(Bar->setSectionPrefix(""));

  // Set from empty to hot.
  EXPECT_TRUE(Bar->setSectionPrefix("hot"));
  EXPECT_THAT(Bar->getSectionPrefix(), Optional(StrEq("hot")));

  // Test that set method returns true and section prefix is cleared.
  EXPECT_TRUE(Bar->setSectionPrefix(""));
  EXPECT_THAT(Bar->getSectionPrefix(), Eq(std::nullopt));
}
} // namespace
