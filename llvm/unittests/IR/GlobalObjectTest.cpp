//===- GlobalObjectTest.cpp - Global Object unit tests
//-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("GlobalObjectTest", errs());
  return Mod;
}

} // namespace

TEST(GlobalObjectTest, updateSectionPrefix) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(C, R"(
@foo = global i32 1, !section_prefix !0
@bar = global i32 2
@baz = global i32 0, !section_prefix !1

!0 = !{!"section_prefix", !"hot"}
!1 = !{!"section_prefix", !"unlikely"}
)");

  GlobalVariable *Foo = M->getGlobalVariable("foo");
  EXPECT_EQ(Foo->getSectionPrefix(), "hot");
  Foo->updateSectionPrefix("unlikely", std::make_optional(StringRef("hot")));
  EXPECT_EQ(Foo->getSectionPrefix(), "hot");

  GlobalVariable *Bar = M->getGlobalVariable("bar");
  EXPECT_EQ(Bar->getSectionPrefix(), std::nullopt);
  Bar->setSectionPrefix("unlikely");
  EXPECT_EQ(Bar->getSectionPrefix(), "unlikely");

  GlobalVariable *Baz = M->getGlobalVariable("baz");
  EXPECT_EQ(Baz->getSectionPrefix(), "unlikely");
  Baz->updateSectionPrefix("hot");
  EXPECT_EQ(Baz->getSectionPrefix(), "hot");
}
