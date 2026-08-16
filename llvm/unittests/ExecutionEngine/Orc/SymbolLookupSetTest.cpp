//===- SymbolLookupSetTest.cpp - Test SymbolLookupSet --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"

#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class SymbolLookupSetTest : public testing::Test {
protected:
  static constexpr SymbolLookupFlags Required =
      SymbolLookupFlags::RequiredSymbol;
  static constexpr SymbolLookupFlags Weak =
      SymbolLookupFlags::WeaklyReferencedSymbol;

  std::shared_ptr<SymbolStringPool> SSP = std::make_shared<SymbolStringPool>();

  SymbolStringPtr intern(StringRef S) { return SSP->intern(S); }

  /// Collect a lookup set into a name-sorted vector of (name, flags).
  ///
  /// A vector rather than a map so that a name surviving more than once is
  /// visible, and name-sorted because mergeEntries sorts by pointer value,
  /// leaving an order that depends on allocation and must not be asserted on.
  using Entries = std::vector<std::pair<std::string, SymbolLookupFlags>>;

  static Entries contents(const SymbolLookupSet &LS) {
    Entries Result;
    for (const auto &[Name, Flags] : LS)
      Result.emplace_back(std::string(*Name), Flags);
    llvm::sort(Result);
    return Result;
  }
};

} // namespace

// A set that is already duplicate-free is left alone, flags included.
TEST_F(SymbolLookupSetTest, MergeEntriesNoDuplicates) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Required}}));
}

// Duplicates that agree on flags collapse to a single entry.
TEST_F(SymbolLookupSetTest, MergeEntriesSameFlags) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);
  LS.add(intern("bar"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Required}}));
}

// A name requested both ways merges to RequiredSymbol: if any requester needs
// the symbol then a missing definition must fail the lookup.
//
// Both insertion orders are checked because mergeEntries sorts by pointer
// value, so which of the two entries is seen first is not under our control.
TEST_F(SymbolLookupSetTest, MergeEntriesRequiredWinsWeakFirst) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Weak);
  LS.add(intern("foo"), Required);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"foo", Required}}));
}

TEST_F(SymbolLookupSetTest, MergeEntriesRequiredWinsRequiredFirst) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("foo"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"foo", Required}}));
}

// Merging must not invent a requirement: all-weak duplicates stay weak.
TEST_F(SymbolLookupSetTest, MergeEntriesAllWeakStaysWeak) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Weak);
  LS.add(intern("foo"), Weak);
  LS.add(intern("foo"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"foo", Weak}}));
}

// Several distinct names, each duplicated a different number of times and with
// mixed flags, all merge in one pass.
TEST_F(SymbolLookupSetTest, MergeEntriesMultipleRuns) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Weak);
  LS.add(intern("bar"), Required);
  LS.add(intern("foo"), Weak);
  LS.add(intern("baz"), Weak);
  LS.add(intern("bar"), Weak);
  LS.add(intern("foo"), Required);
  LS.add(intern("qux"), Required);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"bar", Required},
                                   {"baz", Weak},
                                   {"foo", Required},
                                   {"qux", Required}}));
}
