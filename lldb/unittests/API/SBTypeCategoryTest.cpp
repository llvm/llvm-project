//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"

#include "SBTestingSupport/SBTestUtilities.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

static const char *const g_category_name = "sb-type-category-test";

class SBTypeCategoryTest : public SBDebuggerFixture {
protected:
  void SetUp() override {
    SBDebuggerFixture::SetUp();
    m_category = m_debugger.CreateCategory(g_category_name);
    ASSERT_TRUE(m_category.IsValid());
  }

  // Categories are process-wide and outlive the debugger.
  void TearDown() override {
    m_debugger.DeleteCategory(g_category_name);
    SBDebuggerFixture::TearDown();
  }

  SBTypeCategory m_category;
  SBTypeNameSpecifier m_spec = SBTypeNameSpecifier("Foo");
};

TEST_F(SBTypeCategoryTest, FilterAtIndex) {
  SBTypeFilter filter(eTypeOptionNone);
  filter.AppendExpressionPath("member");
  ASSERT_TRUE(m_category.AddTypeFilter(m_spec, filter));

  SBTypeFilter at_index = m_category.GetFilterAtIndex(0);
  ASSERT_EQ(at_index.GetNumberOfExpressionPaths(), 1u);
  EXPECT_STREQ(at_index.GetExpressionPathAtIndex(0), "member");
  EXPECT_FALSE(m_category.GetFilterAtIndex(1).IsValid());
}
