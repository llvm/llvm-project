//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/TypeSystem/Clike/ByteSizedType.h"
#include "Plugins/TypeSystem/Clike/Type.h"

#include "gtest/gtest.h"

using namespace lldb_private::clike;

TEST(ByteSizedTypeTest, StoresSize) {
  EXPECT_EQ(ByteSizedType<Type>(8).GetByteSize(), std::optional<uint64_t>(8));
  // Zero is used for empty C structs.
  EXPECT_EQ(ByteSizedType<Type>(0).GetByteSize(), std::optional<uint64_t>(0));
  EXPECT_FALSE(ByteSizedType<Type>(std::nullopt).GetByteSize().has_value());
}
