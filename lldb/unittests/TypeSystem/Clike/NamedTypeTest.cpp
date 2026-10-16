//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Clike/NamedType.h"
#include "Plugins/TypeSystem/Clike/Identifier.h"

#include "gtest/gtest.h"

using namespace lldb_private::clike;

TEST(NamedTypeTest, StoresName) {
  IdentifierMap ids;
  NamedType<Type> type(ids.get("Foo"));
  EXPECT_EQ(type.GetName().GetName(), "Foo");
  // A NamedType has no scope, so the unqualified name is the same as the name.
  EXPECT_EQ(type.GetUnqualifiedName().GetName(), "Foo");
  EXPECT_EQ(type.GetDeclContext(), nullptr);
}
