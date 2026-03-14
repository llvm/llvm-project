#include "gtest/gtest.h"

#include "TestingSupport/MockTildeExpressionResolver.h"
#include "lldb/Utility/TildeExpressionResolver.h"

#include "llvm/ADT/SmallString.h"

using namespace llvm;
using namespace lldb_private;

TEST(TildeExpressionResolver, ResolveFullPath) {
  MockTildeExpressionResolver Resolver("James", "/james");
  Resolver.AddKnownUser("Kirk", "/kirk");
  Resolver.AddKnownUser("Lars", "/lars");
  Resolver.AddKnownUser("Jason", "/jason");
  Resolver.AddKnownUser("Larry", "/larry");

  SmallString<32> Result;
  ASSERT_TRUE(Resolver.ResolveFullPath("~", Result));
  EXPECT_EQ("/james", Result);
  ASSERT_TRUE(Resolver.ResolveFullPath("~/", Result));
  EXPECT_EQ("/james/", Result);

  ASSERT_TRUE(Resolver.ResolveFullPath("~James/bar/baz", Result));
  EXPECT_EQ("/james/bar/baz", Result);

  ASSERT_TRUE(Resolver.ResolveFullPath("~Jason/", Result));
  EXPECT_EQ("/jason/", Result);

  ASSERT_TRUE(Resolver.ResolveFullPath("~Lars", Result));
  EXPECT_EQ("/lars", Result);

  ASSERT_FALSE(Resolver.ResolveFullPath("~Jaso", Result));
  EXPECT_EQ("~Jaso", Result);
  ASSERT_FALSE(Resolver.ResolveFullPath("", Result));
  EXPECT_EQ("", Result);
  ASSERT_FALSE(Resolver.ResolveFullPath("Jason", Result));
  EXPECT_EQ("Jason", Result);
}
