//===-- UniqueCStringMapTest.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/UniqueCStringMap.h"
#include "gmock/gmock.h"

using namespace lldb_private;

namespace {
struct NoDefault {
  int x;

  NoDefault(int x) : x(x) {}
  NoDefault() = delete;

  friend bool operator==(NoDefault lhs, NoDefault rhs) {
    return lhs.x == rhs.x;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       NoDefault x) {
    return OS << "NoDefault{" << x.x << "}";
  }
};
} // namespace

TEST(UniqueCStringMap, NoDefaultConstructor) {
  using MapT = UniqueCStringMap<NoDefault>;
  using EntryT = MapT::Entry;

  MapT Map;
  ConstString Foo("foo"), Bar("bar");

  Map.Append(Foo, NoDefault(42));
  EXPECT_THAT(Map.Find(Foo, NoDefault(47)), NoDefault(42));
  EXPECT_THAT(Map.Find(Bar, NoDefault(47)), NoDefault(47));
  EXPECT_THAT(Map.FindFirstValueForName(Foo),
              testing::Pointee(testing::Field(&EntryT::value, NoDefault(42))));
  EXPECT_THAT(Map.FindFirstValueForName(Bar), nullptr);

  std::vector<NoDefault> Values;
  EXPECT_THAT(Map.GetValues(Foo, Values), 1);
  EXPECT_THAT(Values, testing::ElementsAre(NoDefault(42)));

  Values.clear();
  EXPECT_THAT(Map.GetValues(Bar, Values), 0);
  EXPECT_THAT(Values, testing::IsEmpty());
}
