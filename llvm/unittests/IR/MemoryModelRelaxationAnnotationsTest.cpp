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

TEST(MMRATest, Order) {
  MMRAMetadata A, B;

  std::array<MMRAMetadata::TagT, 5> Tags{{{"opencl-fence-mem", "local"},
                                          {"opencl-fence-mem", "global"},
                                          {"foo", "0"},
                                          {"foo", "2"},
                                          {"foo", "4"}}};

  // Test that ordering does not matter.
  for (unsigned K : {0, 2, 3, 1, 4})
    A.addTag(Tags[K]);
  for (unsigned K : {2, 3, 0, 4, 1})
    B.addTag(Tags[K]);

  EXPECT_EQ(A, B);
}

TEST(MMRATest, MDParse) {
  LLVMContext Ctx;

  // No nesting:
  // !{!"foo", "!bar"}
  MDNode *FooBar =
      MDTuple::get(Ctx, {MDString::get(Ctx, "foo"), MDString::get(Ctx, "bar")});
  MMRAMetadata FooBarMMRA(FooBar);

  EXPECT_EQ(FooBarMMRA.size(), 1u);
  EXPECT_EQ(FooBarMMRA, MMRAMetadata().addTag("foo", "bar"));

  // Nested:
  // !{!{!"foo", "!bar"}, !{!"bux", !"qux"}}
  MDNode *BuxQux =
      MDTuple::get(Ctx, {MDString::get(Ctx, "bux"), MDString::get(Ctx, "qux")});
  MDNode *Nested = MDTuple::get(Ctx, {FooBar, BuxQux});
  MMRAMetadata NestedMMRA(Nested);

  EXPECT_EQ(NestedMMRA.size(), 2u);
  EXPECT_EQ(NestedMMRA,
            MMRAMetadata().addTag("foo", "bar").addTag("bux", "qux"));
}

TEST(MMRATest, MDEmit) {
  LLVMContext Ctx;

  // Simple MD.
  // !{!"foo", "!bar"}
  {
    MMRAMetadata FooBarMMRA = MMRAMetadata().addTag("foo", "bar");
    MDTuple *FooBar = FooBarMMRA.getAsMD(Ctx);

    ASSERT_NE(FooBar, nullptr);
    ASSERT_EQ(FooBar->getNumOperands(), 2u);
    MDString *Foo = dyn_cast<MDString>(FooBar->getOperand(0));
    MDString *Bar = dyn_cast<MDString>(FooBar->getOperand(1));
    ASSERT_NE(Foo, nullptr);
    ASSERT_NE(Bar, nullptr);
    EXPECT_EQ(Foo->getString(), std::string("foo"));
    EXPECT_EQ(Bar->getString(), std::string("bar"));
  }

  // Nested MD
  // !{!{!"foo", "!bar"}, !{!"bux", !"qux"}}
  {
    MMRAMetadata NestedMMRA =
        MMRAMetadata().addTag("foo", "bar").addTag("bux", "qux");
    MDTuple *Nested = NestedMMRA.getAsMD(Ctx);

    ASSERT_NE(Nested, nullptr);
    ASSERT_EQ(Nested->getNumOperands(), 2u);
    MDTuple *BuxQux = dyn_cast<MDTuple>(Nested->getOperand(0));
    MDTuple *FooBar = dyn_cast<MDTuple>(Nested->getOperand(1));
    ASSERT_NE(FooBar, nullptr);
    ASSERT_NE(BuxQux, nullptr);
    ASSERT_EQ(FooBar->getNumOperands(), 2u);
    ASSERT_EQ(BuxQux->getNumOperands(), 2u);

    MDString *Foo = dyn_cast<MDString>(FooBar->getOperand(0));
    MDString *Bar = dyn_cast<MDString>(FooBar->getOperand(1));
    MDString *Bux = dyn_cast<MDString>(BuxQux->getOperand(0));
    MDString *Qux = dyn_cast<MDString>(BuxQux->getOperand(1));

    EXPECT_EQ(Foo->getString(), std::string("foo"));
    EXPECT_EQ(Bar->getString(), std::string("bar"));
    EXPECT_EQ(Bux->getString(), std::string("bux"));
    EXPECT_EQ(Qux->getString(), std::string("qux"));
  }
}

TEST(MMRATest, Utility) {
  using TagT = MMRAMetadata::TagT;

  MMRAMetadata MMRA;
  MMRA.addTag("foo", "0");
  MMRA.addTag("foo", "1");
  MMRA.addTag("bar", "x");

  EXPECT_TRUE(MMRA.hasTagWithPrefix("foo"));
  EXPECT_TRUE(MMRA.hasTagWithPrefix("bar"));
  EXPECT_FALSE(MMRA.hasTagWithPrefix("x"));

  EXPECT_TRUE(MMRA.hasTag("foo", "0"));
  EXPECT_TRUE(MMRA.hasTag(TagT("foo", "0")));
  EXPECT_TRUE(MMRA.hasTag("foo", "1"));
  EXPECT_TRUE(MMRA.hasTag(TagT("foo", "1")));
  EXPECT_TRUE(MMRA.hasTag("bar", "x"));
  EXPECT_TRUE(MMRA.hasTag(TagT("bar", "x")));

  auto AllFoo = MMRA.getAllTagsWithPrefix("foo");
  auto AllBar = MMRA.getAllTagsWithPrefix("bar");
  auto AllX = MMRA.getAllTagsWithPrefix("x");

  EXPECT_EQ(AllFoo.size(), 2u);
  EXPECT_EQ(AllBar.size(), 1u);
  EXPECT_EQ(AllX.size(), 0u);

  EXPECT_TRUE(is_contained(AllFoo, TagT("foo", "0")));
  EXPECT_TRUE(is_contained(AllFoo, TagT("foo", "1")));
  EXPECT_TRUE(is_contained(AllBar, TagT("bar", "x")));
}

TEST(MMRATest, Operators) {
  MMRAMetadata A;
  A.addTag("foo", "0");
  A.addTag("bar", "x");

  MMRAMetadata B;
  B.addTag("foo", "0");
  B.addTag("bar", "y");

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
  MMRAMetadata Foo0;
  Foo0.addTag("foo", "0");

  MMRAMetadata Foo1;
  Foo1.addTag("foo", "1");

  MMRAMetadata Foo10;
  Foo10.addTag("foo", "0");
  Foo10.addTag("foo", "1");

  MMRAMetadata Bar;
  Bar.addTag("bar", "y");

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
  MMRAMetadata Multiple0;
  Multiple0.addTag("foo", "y");
  Multiple0.addTag("foo", "x");
  Multiple0.addTag("bar", "z");

  MMRAMetadata Multiple1;
  Multiple1.addTag("foo", "z");
  Multiple1.addTag("foo", "x");
  Multiple1.addTag("bar", "y");

  MMRAMetadata Multiple2;
  Multiple2.addTag("foo", "z");
  Multiple2.addTag("foo", "x");
  Multiple2.addTag("bux", "y");

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
  MMRAMetadata Foo0;
  Foo0.addTag("foo", "0");

  MMRAMetadata Foo10;
  Foo10.addTag("foo", "0");
  Foo10.addTag("foo", "1");

  MMRAMetadata Bar0;
  Bar0.addTag("bar", "0");

  MMRAMetadata BarFoo0;
  BarFoo0.addTag("bar", "0");
  BarFoo0.addTag("foo", "0");

  {
    // foo is common to both sets
    MMRAMetadata Combined = Foo0.combine(Foo10);
    EXPECT_EQ(Combined, Foo10);
  }

  {
    // nothing is common
    MMRAMetadata Combined = Foo0.combine(Bar0);
    EXPECT_TRUE(Combined.empty());
  }

  {
    // only foo is common.
    MMRAMetadata Combined = BarFoo0.combine(Foo0);
    EXPECT_EQ(Combined, Foo0);
  }

  {
    // only bar is common.
    MMRAMetadata Combined = BarFoo0.combine(Bar0);
    EXPECT_EQ(Combined, Bar0);
  }

  {
    // only foo is common
    MMRAMetadata Combined = BarFoo0.combine(Foo10);
    EXPECT_EQ(Combined, Foo10);
  }
}

} // namespace
