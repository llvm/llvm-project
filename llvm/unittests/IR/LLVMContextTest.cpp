//===- LLVMContextTest.cpp - LLVMContext unit tests -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(LLVMContextTest, ensureDITypeMap) {
  LLVMContext Context;
  EXPECT_FALSE(Context.hasDITypeMap());
  Context.ensureDITypeMap();
  EXPECT_TRUE(Context.hasDITypeMap());
  Context.destroyDITypeMap();
  EXPECT_FALSE(Context.hasDITypeMap());
}

TEST(LLVMContextTest, getOrInsertDITypeMapping) {
  LLVMContext Context;
  const MDString &S = *MDString::get(Context, "string");

  // Without a type map, this should return null.
  EXPECT_FALSE(Context.getOrInsertDITypeMapping(S));

  // Get the mapping.
  Context.ensureDITypeMap();
  DIType **Mapping = Context.getOrInsertDITypeMapping(S);
  ASSERT_TRUE(Mapping);

  // Create some type and add it to the mapping.
  auto &BT =
      *DIBasicType::get(Context, dwarf::DW_TAG_unspecified_type, S.getString());
  *Mapping = &BT;

  // Check that we get it back.
  Mapping = Context.getOrInsertDITypeMapping(S);
  ASSERT_TRUE(Mapping);
  EXPECT_EQ(&BT, *Mapping);

  // Check that it's discarded with the type map.
  Context.destroyDITypeMap();
  EXPECT_FALSE(Context.getOrInsertDITypeMapping(S));

  // And it shouldn't magically reappear...
  Context.ensureDITypeMap();
  EXPECT_FALSE(*Context.getOrInsertDITypeMapping(S));
}

} // end namespace
