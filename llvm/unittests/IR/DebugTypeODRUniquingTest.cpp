//===- DebugTypeODRUniquingTest.cpp - Debug type ODR uniquing tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LLVMContext.h"
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
      nullptr, 0, 0, 0, DINode::FlagZero, nullptr, 0, nullptr, nullptr, nullptr,
      nullptr));

  // Enable the mapping.  There still shouldn't be a type.
  Context.enableDebugTypeODRUniquing();
  EXPECT_FALSE(DICompositeType::getODRTypeIfExists(Context, UUID));

  // Create some ODR-uniqued type.
  auto &CT = *DICompositeType::getODRType(
      Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0, DINode::FlagZero, nullptr, 0, nullptr, nullptr, nullptr,
      nullptr);
  EXPECT_EQ(UUID.getString(), CT.getIdentifier());

  // Check that we get it back, even if we change a field.
  EXPECT_EQ(&CT, DICompositeType::getODRTypeIfExists(Context, UUID));
  EXPECT_EQ(&CT, DICompositeType::getODRType(
                     Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr,
                     0, nullptr, nullptr, 0, 0, 0, DINode::FlagZero, nullptr, 0,
                     nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(&CT, DICompositeType::getODRType(
                     Context, UUID, dwarf::DW_TAG_class_type,
                     MDString::get(Context, "name"), nullptr, 0, nullptr,
                     nullptr, 0, 0, 0, DINode::FlagZero, nullptr, 0, nullptr,
                     nullptr, nullptr, nullptr));

  // Check that it's discarded with the type map.
  Context.disableDebugTypeODRUniquing();
  EXPECT_FALSE(DICompositeType::getODRTypeIfExists(Context, UUID));

  // And it shouldn't magically reappear...
  Context.enableDebugTypeODRUniquing();
  EXPECT_FALSE(DICompositeType::getODRTypeIfExists(Context, UUID));
}

TEST(DebugTypeODRUniquingTest, buildODRType) {
  LLVMContext Context;
  Context.enableDebugTypeODRUniquing();

  // Build an ODR type that's a forward decl.
  MDString &UUID = *MDString::get(Context, "Type");
  auto &CT = *DICompositeType::buildODRType(
      Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0, DINode::FlagFwdDecl, nullptr, 0, nullptr, nullptr,
      nullptr, nullptr);
  EXPECT_EQ(&CT, DICompositeType::getODRTypeIfExists(Context, UUID));
  EXPECT_EQ(dwarf::DW_TAG_class_type, CT.getTag());

  // Update with another forward decl.  This should be a no-op.
  EXPECT_EQ(&CT, DICompositeType::buildODRType(
                     Context, UUID, dwarf::DW_TAG_structure_type, nullptr,
                     nullptr, 0, nullptr, nullptr, 0, 0, 0, DINode::FlagFwdDecl,
                     nullptr, 0, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(dwarf::DW_TAG_class_type, CT.getTag());

  // Update with a definition.  This time we should see a change.
  EXPECT_EQ(&CT, DICompositeType::buildODRType(
                     Context, UUID, dwarf::DW_TAG_structure_type, nullptr,
                     nullptr, 0, nullptr, nullptr, 0, 0, 0, DINode::FlagZero,
                     nullptr, 0, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(dwarf::DW_TAG_structure_type, CT.getTag());

  // Further updates should be ignored.
  EXPECT_EQ(&CT, DICompositeType::buildODRType(
                     Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr,
                     0, nullptr, nullptr, 0, 0, 0, DINode::FlagFwdDecl, nullptr,
                     0, nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(dwarf::DW_TAG_structure_type, CT.getTag());
  EXPECT_EQ(&CT, DICompositeType::buildODRType(
                     Context, UUID, dwarf::DW_TAG_class_type, nullptr, nullptr,
                     0, nullptr, nullptr, 0, 0, 0, DINode::FlagZero, nullptr, 0,
                     nullptr, nullptr, nullptr, nullptr));
  EXPECT_EQ(dwarf::DW_TAG_structure_type, CT.getTag());
}

TEST(DebugTypeODRUniquingTest, buildODRTypeFields) {
  LLVMContext Context;
  Context.enableDebugTypeODRUniquing();

  // Build an ODR type that's a forward decl with no other fields set.
  MDString &UUID = *MDString::get(Context, "UUID");
  auto &CT = *DICompositeType::buildODRType(
      Context, UUID, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, 0, 0,
      DINode::FlagFwdDecl, nullptr, 0, nullptr, nullptr, nullptr, nullptr);

// Create macros for running through all the fields except Identifier and Flags.
#define FOR_EACH_MDFIELD()                                                     \
  DO_FOR_FIELD(Name)                                                           \
  DO_FOR_FIELD(File)                                                           \
  DO_FOR_FIELD(Scope)                                                          \
  DO_FOR_FIELD(BaseType)                                                       \
  DO_FOR_FIELD(Elements)                                                       \
  DO_FOR_FIELD(VTableHolder)                                                   \
  DO_FOR_FIELD(TemplateParams)
#define FOR_EACH_INLINEFIELD()                                                 \
  DO_FOR_FIELD(Tag)                                                            \
  DO_FOR_FIELD(Line)                                                           \
  DO_FOR_FIELD(SizeInBits)                                                     \
  DO_FOR_FIELD(AlignInBits)                                                    \
  DO_FOR_FIELD(OffsetInBits)                                                   \
  DO_FOR_FIELD(RuntimeLang)

// Create all the fields.
#define DO_FOR_FIELD(X) auto *X = MDString::get(Context, #X);
  FOR_EACH_MDFIELD();
#undef DO_FOR_FIELD
  unsigned NonZeroInit = 0;
#define DO_FOR_FIELD(X) auto X = ++NonZeroInit;
  FOR_EACH_INLINEFIELD();
#undef DO_FOR_FIELD

  // Replace all the fields with new values that are distinct from each other.
  EXPECT_EQ(&CT, DICompositeType::buildODRType(
                     Context, UUID, Tag, Name, File, Line, Scope, BaseType,
                     SizeInBits, AlignInBits, OffsetInBits,
                     DINode::FlagArtificial, Elements, RuntimeLang,
                     VTableHolder, TemplateParams, nullptr, nullptr));

  // Confirm that all the right fields got updated.
#define DO_FOR_FIELD(X) EXPECT_EQ(X, CT.getRaw##X());
  FOR_EACH_MDFIELD();
#undef DO_FOR_FIELD
#undef FOR_EACH_MDFIELD
#define DO_FOR_FIELD(X) EXPECT_EQ(X, CT.get##X());
  FOR_EACH_INLINEFIELD();
#undef DO_FOR_FIELD
#undef FOR_EACH_INLINEFIELD
  EXPECT_EQ(DINode::FlagArtificial, CT.getFlags());
  EXPECT_EQ(&UUID, CT.getRawIdentifier());
}

} // end namespace
