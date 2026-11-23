//===- SPIRVUtilsTests.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

// SyncScope IDs are context-specific and depend on registration order. Check
// that getMemScope uses the IDs from the supplied context.
TEST(SPIRVUtilsTest, getMemScopeAcrossContexts) {
  Triple TT("spirv64-unknown-unknown");
  auto Check = [&](ArrayRef<StringRef> Order) {
    LLVMContext Ctx;
    for (StringRef Name : Order)
      Ctx.getOrInsertSyncScopeID(Name);
    auto Scope = [&](StringRef Name) {
      return getMemScope(TT, Ctx, Ctx.getOrInsertSyncScopeID(Name));
    };
    EXPECT_EQ(Scope("subgroup"), SPIRV::Scope::Subgroup);
    EXPECT_EQ(Scope("workgroup"), SPIRV::Scope::Workgroup);
    EXPECT_EQ(Scope("device"), SPIRV::Scope::Device);
  };
  Check({"device", "workgroup", "subgroup"});
  Check({"subgroup", "workgroup", "device"});
}
