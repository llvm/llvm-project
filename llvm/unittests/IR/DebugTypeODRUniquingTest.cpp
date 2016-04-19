//===- DebugTypeODRUniquingTest.cpp - Debug type ODR uniquing tests -------===//
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

TEST(DebugTypeODRUniquingTest, enableDebugTypeODRUniquing) {
  LLVMContext Context;
  EXPECT_FALSE(Context.isODRUniquingDebugTypes());
  Context.enableDebugTypeODRUniquing();
  EXPECT_TRUE(Context.isODRUniquingDebugTypes());
  Context.disableDebugTypeODRUniquing();
  EXPECT_FALSE(Context.isODRUniquingDebugTypes());
}

TEST(DebugTypeODRUniquingTest, getODRType) {
  LLVMContext Context;
  MDString &UUID = *MDString::get(Context, "string");

  // Without a type map, this should return null.
  EXPECT_FALSE(DICompositeType::getODRType(
      Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0, 0, nullptr, 0, nullptr, nullptr));

  // Enable the mapping.  There still shouldn't be a type.
  Context.enableDebugTypeODRUniquing();
  EXPECT_FALSE(DICompositeType::getODRTypeIfExists(Context, UUID));

  // Create some ODR-uniqued type.
  auto &CT = *DICompositeType::getODRType(
      Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0, 0, nullptr, 0, nullptr, nullptr);
  EXPECT_EQ(UUID.getString(), CT.getIdentifier());

  // Check that we get it back, even if we change a field.
  EXPECT_EQ(&CT, DICompositeType::getODRTypeIfExists(Context, UUID));
  EXPECT_EQ(
      &CT, DICompositeType::getODRType(Context, UUID, dwarf::DW_TAG_class_type,
                                       nullptr, nullptr, 0, nullptr, nullptr, 0,
                                       0, 0, 0, nullptr, 0, nullptr, nullptr));
  EXPECT_EQ(&CT, DICompositeType::getODRType(
                     Context, UUID, dwarf::DW_TAG_class_type,
                     MDString::get(Context, "name"), nullptr, 0, nullptr,
                     nullptr, 0, 0, 0, 0, nullptr, 0, nullptr, nullptr));

  // Check that it's discarded with the type map.
  Context.disableDebugTypeODRUniquing();
  EXPECT_FALSE(DICompositeType::getODRTypeIfExists(Context, UUID));

  // And it shouldn't magically reappear...
  Context.enableDebugTypeODRUniquing();
  EXPECT_FALSE(DICompositeType::getODRTypeIfExists(Context, UUID));
}

} // end namespace
