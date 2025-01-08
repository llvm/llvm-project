//===- llvm/unittests/IR/MemoryModelRelaxationAnnotationsTest.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

void checkMMRA(const MMRAMetadata &MMRA,
               ArrayRef<MMRAMetadata::TagT> Expected) {
  EXPECT_EQ(MMRA.size(), Expected.size());
  for (const auto &E : Expected)
    EXPECT_TRUE(MMRA.hasTag(E.first, E.second));
}

MMRAMetadata createFromMD(LLVMContext &Ctx,
                          ArrayRef<MMRAMetadata::TagT> Expected) {
  SmallVector<Metadata *> MD;
  for (const auto &Tag : Expected)
    MD.push_back(MMRAMetadata::getTagMD(Ctx, Tag));
  return MDTuple::get(Ctx, MD);
}

TEST(MMRATest, MDParse) {
  LLVMContext Ctx;

  // No nesting:
  // !{!"foo", "!bar"}
  MDNode *FooBar =
      MDTuple::get(Ctx, {MDString::get(Ctx, "foo"), MDString::get(Ctx, "bar")});
  MMRAMetadata FooBarMMRA(FooBar);

  checkMMRA(FooBarMMRA, {{"foo", "bar"}});

  // Nested:
  // !{!{!"foo", "!bar"}, !{!"bux", !"qux"}}
  MDNode *BuxQux =
      MDTuple::get(Ctx, {MDString::get(Ctx, "bux"), MDString::get(Ctx, "qux")});
  MDNode *Nested = MDTuple::get(Ctx, {FooBar, BuxQux});
  MMRAMetadata NestedMMRA(Nested);

  checkMMRA(NestedMMRA, {{"foo", "bar"}, {"bux", "qux"}});
}

TEST(MMRATest, GetMD) {
  LLVMContext Ctx;

  EXPECT_EQ(MMRAMetadata::getMD(Ctx, {}), nullptr);

  MDTuple *SingleMD = MMRAMetadata::getMD(Ctx, {{"foo", "bar"}});
  EXPECT_EQ(SingleMD->getNumOperands(), 2u);
  EXPECT_EQ(cast<MDString>(SingleMD->getOperand(0))->getString(), "foo");
  EXPECT_EQ(cast<MDString>(SingleMD->getOperand(1))->getString(), "bar");

  MDTuple *MultiMD = MMRAMetadata::getMD(Ctx, {{"foo", "bar"}, {"bux", "qux"}});
  EXPECT_EQ(MultiMD->getNumOperands(), 2u);

  MDTuple *FooBar = cast<MDTuple>(MultiMD->getOperand(0));
  EXPECT_EQ(cast<MDString>(FooBar->getOperand(0))->getString(), "foo");
  EXPECT_EQ(cast<MDString>(FooBar->getOperand(1))->getString(), "bar");
  MDTuple *BuxQux = cast<MDTuple>(MultiMD->getOperand(1));
  EXPECT_EQ(cast<MDString>(BuxQux->getOperand(0))->getString(), "bux");
  EXPECT_EQ(cast<MDString>(BuxQux->getOperand(1))->getString(), "qux");
}

TEST(MMRATest, Utility) {
  LLVMContext Ctx;
  MMRAMetadata MMRA =
      createFromMD(Ctx, {{"foo", "0"}, {"foo", "1"}, {"bar", "x"}});

  EXPECT_TRUE(MMRA.hasTagWithPrefix("foo"));
  EXPECT_TRUE(MMRA.hasTagWithPrefix("bar"));
  EXPECT_FALSE(MMRA.hasTagWithPrefix("x"));

  EXPECT_TRUE(MMRA.hasTag("foo", "0"));
  EXPECT_TRUE(MMRA.hasTag("foo", "1"));
  EXPECT_TRUE(MMRA.hasTag("bar", "x"));
}

TEST(MMRATest, Operators) {
  LLVMContext Ctx;

  MMRAMetadata A = createFromMD(Ctx, {{"foo", "0"}, {"bar", "x"}});
  MMRAMetadata B = createFromMD(Ctx, {{"foo", "0"}, {"bar", "y"}});

  // ensure we have different objects by creating copies.
  EXPECT_EQ(MMRAMetadata(A), MMRAMetadata(A));
  EXPECT_TRUE((bool)A);

  EXPECT_EQ(MMRAMetadata(B), MMRAMetadata(B));
  EXPECT_TRUE((bool)B);

  EXPECT_NE(A, B);

  EXPECT_EQ(MMRAMetadata(), MMRAMetadata());
  EXPECT_NE(A, MMRAMetadata());
  EXPECT_NE(B, MMRAMetadata());

  MMRAMetadata Empty;
  EXPECT_FALSE((bool)Empty);
}

TEST(MMRATest, Compatibility) {
  LLVMContext Ctx;

  MMRAMetadata Foo0 = createFromMD(Ctx, {{"foo", "0"}});
  MMRAMetadata Foo1 = createFromMD(Ctx, {{"foo", "1"}});
  MMRAMetadata Foo10 = createFromMD(Ctx, {{"foo", "0"}, {"foo", "1"}});

  MMRAMetadata Bar = createFromMD(Ctx, {{"bar", "y"}});

  MMRAMetadata Empty;

  // Other set has no tag with same prefix
  EXPECT_TRUE(Foo0.isCompatibleWith(Bar));
  EXPECT_TRUE(Bar.isCompatibleWith(Foo0));

  EXPECT_TRUE(Foo0.isCompatibleWith(Empty));
  EXPECT_TRUE(Empty.isCompatibleWith(Foo0));

  EXPECT_TRUE(Empty.isCompatibleWith(MMRAMetadata()));
  EXPECT_TRUE(MMRAMetadata().isCompatibleWith(Empty));

  // Other set has conflicting tags.
  EXPECT_FALSE(Foo1.isCompatibleWith(Foo0));
  EXPECT_FALSE(Foo0.isCompatibleWith(Foo1));

  // Both have common tags.
  EXPECT_TRUE(Foo0.isCompatibleWith(Foo0));
  EXPECT_TRUE(Foo0.isCompatibleWith(Foo10));
  EXPECT_TRUE(Foo10.isCompatibleWith(Foo0));

  EXPECT_TRUE(Foo1.isCompatibleWith(Foo1));
  EXPECT_TRUE(Foo1.isCompatibleWith(Foo10));
  EXPECT_TRUE(Foo10.isCompatibleWith(Foo1));

  // Try with more prefixes now:
  MMRAMetadata Multiple0 =
      createFromMD(Ctx, {{"foo", "y"}, {"foo", "x"}, {"bar", "z"}});
  MMRAMetadata Multiple1 =
      createFromMD(Ctx, {{"foo", "z"}, {"foo", "x"}, {"bar", "y"}});
  MMRAMetadata Multiple2 =
      createFromMD(Ctx, {{"foo", "z"}, {"foo", "x"}, {"bux", "y"}});

  // Multiple0 and Multiple1 are not compatible because "bar" is getting in the
  // way.
  EXPECT_FALSE(Multiple0.isCompatibleWith(Multiple1));
  EXPECT_FALSE(Multiple1.isCompatibleWith(Multiple0));

  EXPECT_TRUE(Multiple0.isCompatibleWith(Empty));
  EXPECT_TRUE(Empty.isCompatibleWith(Multiple0));
  EXPECT_TRUE(Multiple1.isCompatibleWith(Empty));
  EXPECT_TRUE(Empty.isCompatibleWith(Multiple1));

  // Multiple2 is compatible with both 1/0 because there is always "foo:x" in
  // common, and the other prefixes are unique to each set.
  EXPECT_TRUE(Multiple2.isCompatibleWith(Multiple0));
  EXPECT_TRUE(Multiple0.isCompatibleWith(Multiple2));
  EXPECT_TRUE(Multiple2.isCompatibleWith(Multiple1));
  EXPECT_TRUE(Multiple1.isCompatibleWith(Multiple2));
}

TEST(MMRATest, Combine) {
  LLVMContext Ctx;

  MMRAMetadata Foo0 = createFromMD(Ctx, {{"foo", "0"}});
  MMRAMetadata Foo10 = createFromMD(Ctx, {{"foo", "0"}, {"foo", "1"}});
  MMRAMetadata Bar0 = createFromMD(Ctx, {{"bar", "0"}});
  MMRAMetadata BarFoo0 = createFromMD(Ctx, {{"bar", "0"}, {"foo", "0"}});

  {
    // foo is common to both sets
    MMRAMetadata Combined = MMRAMetadata::combine(Ctx, Foo0, Foo10);
    EXPECT_EQ(Combined, Foo10);
  }

  {
    // nothing is common
    MMRAMetadata Combined = MMRAMetadata::combine(Ctx, Foo0, Bar0);
    EXPECT_TRUE(Combined.empty());
  }

  {
    // only foo is common.
    MMRAMetadata Combined = MMRAMetadata::combine(Ctx, BarFoo0, Foo0);
    EXPECT_EQ(Combined, Foo0);
  }

  {
    // only bar is common.
    MMRAMetadata Combined = MMRAMetadata::combine(Ctx, BarFoo0, Bar0);
    EXPECT_EQ(Combined, Bar0);
  }

  {
    // only foo is common
    MMRAMetadata Combined = MMRAMetadata::combine(Ctx, BarFoo0, Foo10);
    EXPECT_EQ(Combined, Foo10);
  }
}

} // namespace
