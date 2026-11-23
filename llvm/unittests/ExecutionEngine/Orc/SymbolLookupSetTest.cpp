//===- SymbolLookupSetTest.cpp - Test SymbolLookupSet ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SymbolLookupSet.h"

#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Testing/Support/Error.h"

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

  using Entries = std::vector<std::pair<std::string, SymbolLookupFlags>>;
  using Names = std::vector<std::string>;

  /// Collect a lookup set into a name-sorted vector of (name, flags).
  ///
  /// A vector rather than a map so that a name appearing more than once stays
  /// visible, and name-sorted because most operations here leave an order that
  /// depends either on allocation order or on swap-with-back removal, neither
  /// of which is part of the contract.
  static Entries contents(const SymbolLookupSet &LS) {
    Entries Result;
    for (const auto &[Name, Flags] : LS)
      Result.emplace_back(std::string(*Name), Flags);
    llvm::sort(Result);
    return Result;
  }

  /// Collect a lookup set's entries in iteration order. Only for the operations
  /// whose resulting order is actually specified.
  static Entries entriesInOrder(const SymbolLookupSet &LS) {
    Entries Result;
    for (const auto &[Name, Flags] : LS)
      Result.emplace_back(std::string(*Name), Flags);
    return Result;
  }

  static Names namesInOrder(const SymbolLookupSet &LS) {
    Names Result;
    for (const auto &[Name, Flags] : LS)
      Result.push_back(std::string(*Name));
    return Result;
  }

  static Names toNames(const SymbolNameVector &V) {
    Names Result;
    for (const auto &Name : V)
      Result.push_back(std::string(*Name));
    return Result;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

TEST_F(SymbolLookupSetTest, DefaultConstructedIsEmpty) {
  SymbolLookupSet LS;
  EXPECT_TRUE(LS.empty());
  EXPECT_EQ(LS.size(), 0U);
  EXPECT_EQ(LS.begin(), LS.end());
}

TEST_F(SymbolLookupSetTest, ConstructFromSingleName) {
  SymbolLookupSet Defaulted(intern("foo"));
  EXPECT_EQ(contents(Defaulted), (Entries{{"foo", Required}}));

  SymbolLookupSet Weakly(intern("foo"), Weak);
  EXPECT_EQ(contents(Weakly), (Entries{{"foo", Weak}}));
}

TEST_F(SymbolLookupSetTest, ConstructFromEntryList) {
  SymbolLookupSet LS({{intern("foo"), Required}, {intern("bar"), Weak}});
  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Required}}));
}

TEST_F(SymbolLookupSetTest, ConstructFromNameList) {
  SymbolLookupSet Defaulted({intern("foo"), intern("bar")});
  EXPECT_EQ(contents(Defaulted),
            (Entries{{"bar", Required}, {"foo", Required}}));

  SymbolLookupSet Weakly({intern("foo"), intern("bar")}, Weak);
  EXPECT_EQ(contents(Weakly), (Entries{{"bar", Weak}, {"foo", Weak}}));
}

TEST_F(SymbolLookupSetTest, ConstructFromSymbolNameSet) {
  SymbolNameSet S;
  S.insert(intern("foo"));
  S.insert(intern("bar"));

  SymbolLookupSet LS(S, Weak);
  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Weak}}));
}

TEST_F(SymbolLookupSetTest, ConstructFromArrayRef) {
  SymbolNameVector V{intern("foo"), intern("bar")};

  SymbolLookupSet LS(ArrayRef<SymbolStringPtr>(V), Weak);
  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Weak}}));
}

TEST_F(SymbolLookupSetTest, FromMapKeys) {
  DenseMap<SymbolStringPtr, int> M;
  M[intern("foo")] = 1;
  M[intern("bar")] = 2;

  auto LS = SymbolLookupSet::fromMapKeys(M, Weak);
  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Weak}}));
}

//===----------------------------------------------------------------------===//
// add / append
//===----------------------------------------------------------------------===//

// add returns *this so that calls can be chained.
TEST_F(SymbolLookupSetTest, AddIsChainable) {
  SymbolLookupSet LS;
  LS.add(intern("foo")).add(intern("bar"), Weak);
  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Required}}));
}

TEST_F(SymbolLookupSetTest, Append) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);

  SymbolLookupSet Other;
  Other.add(intern("bar"), Weak);
  Other.add(intern("baz"), Required);

  LS.append(std::move(Other));

  EXPECT_EQ(contents(LS),
            (Entries{{"bar", Weak}, {"baz", Required}, {"foo", Required}}));
}

// append does not merge: a name already present is simply added again.
TEST_F(SymbolLookupSetTest, AppendDoesNotMerge) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Weak);

  SymbolLookupSet Other;
  Other.add(intern("foo"), Required);

  LS.append(std::move(Other));

  EXPECT_EQ(contents(LS), (Entries{{"foo", Required}, {"foo", Weak}}));
}

//===----------------------------------------------------------------------===//
// remove / remove_if
//===----------------------------------------------------------------------===//

// remove(iterator) drops the element it points at. Removal swaps the last
// element into the vacated slot, so the surviving order is unspecified.
TEST_F(SymbolLookupSetTest, RemoveByIterator) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);
  LS.add(intern("baz"), Required);

  LS.remove(LS.begin());

  EXPECT_EQ(LS.size(), 2U);
  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"baz", Required}}));
}

TEST_F(SymbolLookupSetTest, RemoveByIndex) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);

  LS.remove(static_cast<SymbolLookupSet::UnderlyingVector::size_type>(1));

  EXPECT_EQ(contents(LS), (Entries{{"foo", Required}}));
}

TEST_F(SymbolLookupSetTest, RemoveLastRemainingElement) {
  SymbolLookupSet LS;
  LS.add(intern("foo"));

  LS.remove(LS.begin());

  EXPECT_TRUE(LS.empty());
}

// remove_if drops exactly the elements its predicate selects.
//
// The predicate reads both the name and the flags, and records what it saw:
// removal swaps the last element into the slot the loop is on without
// advancing, so an element being skipped or visited twice is the real hazard
// here.
TEST_F(SymbolLookupSetTest, RemoveIf) {
  SymbolLookupSet LS;
  LS.add(intern("keep1"), Required);
  LS.add(intern("dropWeak"), Weak);
  LS.add(intern("keep2"), Required);
  LS.add(intern("dropNamed"), Required);
  LS.add(intern("keep3"), Required);

  Entries Visited;
  LS.remove_if([&](const SymbolStringPtr &Name, SymbolLookupFlags Flags) {
    Visited.emplace_back(std::string(*Name), Flags);
    return Flags == SymbolLookupFlags::WeaklyReferencedSymbol ||
           *Name == "dropNamed";
  });

  llvm::sort(Visited);
  EXPECT_EQ(Visited, (Entries{{"dropNamed", Required},
                              {"dropWeak", Weak},
                              {"keep1", Required},
                              {"keep2", Required},
                              {"keep3", Required}}));
  EXPECT_EQ(
      contents(LS),
      (Entries{{"keep1", Required}, {"keep2", Required}, {"keep3", Required}}));
}

//===----------------------------------------------------------------------===//
// forEachWithRemoval
//===----------------------------------------------------------------------===//

// The bool overload removes on true and retains on false, and visits every
// element exactly once despite removal shuffling the vector under the loop. The
// body sees both the name and the flags.
TEST_F(SymbolLookupSetTest, ForEachWithRemoval) {
  SymbolLookupSet LS;
  LS.add(intern("keep1"), Required);
  LS.add(intern("dropWeak"), Weak);
  LS.add(intern("keep2"), Required);
  LS.add(intern("dropNamed"), Required);

  Entries Visited;
  LS.forEachWithRemoval(
      [&](const SymbolStringPtr &Name, SymbolLookupFlags Flags) {
        Visited.emplace_back(std::string(*Name), Flags);
        return Flags == SymbolLookupFlags::WeaklyReferencedSymbol ||
               *Name == "dropNamed";
      });

  llvm::sort(Visited);
  EXPECT_EQ(Visited, (Entries{{"dropNamed", Required},
                              {"dropWeak", Weak},
                              {"keep1", Required},
                              {"keep2", Required}}));
  EXPECT_EQ(contents(LS), (Entries{{"keep1", Required}, {"keep2", Required}}));
}

// The Expected<bool> overload removes on true and retains on false, as the bool
// overload does, when no error is returned.
TEST_F(SymbolLookupSetTest, ForEachWithRemovalExpectedSuccess) {
  SymbolLookupSet LS;
  LS.add(intern("keep"), Required);
  LS.add(intern("drop"), Weak);

  EXPECT_THAT_ERROR(
      LS.forEachWithRemoval([](const SymbolStringPtr &Name, SymbolLookupFlags)
                                -> Expected<bool> { return *Name == "drop"; }),
      Succeeded());

  EXPECT_EQ(contents(LS), (Entries{{"keep", Required}}));
}

// An error exits the loop immediately and propagates to the caller.
TEST_F(SymbolLookupSetTest, ForEachWithRemovalExpectedError) {
  SymbolLookupSet LS;
  LS.add(intern("foo"));
  LS.add(intern("bar"));

  unsigned Visits = 0;
  EXPECT_THAT_ERROR(
      LS.forEachWithRemoval(
          [&](const SymbolStringPtr &, SymbolLookupFlags) -> Expected<bool> {
            ++Visits;
            return make_error<StringError>("boom", inconvertibleErrorCode());
          }),
      Failed());

  EXPECT_EQ(Visits, 1U);
}

//===----------------------------------------------------------------------===//
// getSymbolNames
//===----------------------------------------------------------------------===//

// getSymbolNames drops the flags and preserves iteration order.
TEST_F(SymbolLookupSetTest, GetSymbolNames) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);

  EXPECT_EQ(toNames(LS.getSymbolNames()), namesInOrder(LS));

  auto Sorted = toNames(LS.getSymbolNames());
  llvm::sort(Sorted);
  EXPECT_EQ(Sorted, (Names{"bar", "foo"}));
}

TEST_F(SymbolLookupSetTest, GetSymbolNamesOnEmptySet) {
  SymbolLookupSet LS;
  EXPECT_TRUE(LS.getSymbolNames().empty());
}

//===----------------------------------------------------------------------===//
// Sorting
//===----------------------------------------------------------------------===//

// sortByName is lexicographic, so its resulting order is specified.
TEST_F(SymbolLookupSetTest, SortByName) {
  SymbolLookupSet LS;
  LS.add(intern("charlie"), Weak);
  LS.add(intern("alpha"), Required);
  LS.add(intern("bravo"), Weak);

  LS.sortByName();

  EXPECT_EQ(entriesInOrder(LS),
            (Entries{{"alpha", Required}, {"bravo", Weak}, {"charlie", Weak}}));
}

// sortByAddress orders by pointer value, which depends on allocation order. All
// that can be checked portably is that the contents survive and that the result
// really is non-decreasing by pointer.
TEST_F(SymbolLookupSetTest, SortByAddress) {
  SymbolLookupSet LS;
  LS.add(intern("charlie"), Weak);
  LS.add(intern("alpha"), Required);
  LS.add(intern("bravo"), Weak);

  LS.sortByAddress();

  EXPECT_EQ(contents(LS),
            (Entries{{"alpha", Required}, {"bravo", Weak}, {"charlie", Weak}}));

  for (auto I = LS.begin(), E = LS.end(); I != E && std::next(I) != E; ++I)
    EXPECT_FALSE(std::next(I)->first < I->first);
}

//===----------------------------------------------------------------------===//
// mergeEntries
//===----------------------------------------------------------------------===//

TEST_F(SymbolLookupSetTest, MergeEntriesTrivialSizes) {
  SymbolLookupSet Empty;
  Empty.mergeEntries();
  EXPECT_TRUE(Empty.empty());

  SymbolLookupSet One;
  One.add(intern("foo"), Weak);
  One.mergeEntries();
  EXPECT_EQ(contents(One), (Entries{{"foo", Weak}}));
}

// A set that is already duplicate-free is left alone, flags included.
TEST_F(SymbolLookupSetTest, MergeEntriesNoDuplicates) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Required}}));
}

// Entries that agree on flags collapse to a single entry.
TEST_F(SymbolLookupSetTest, MergeEntriesSameFlags) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("foo"), Required);
  LS.add(intern("bar"), Weak);
  LS.add(intern("bar"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"bar", Weak}, {"foo", Required}}));
}

// A name requested both ways merges to RequiredSymbol: if any entry required
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

// Merging must not invent a requirement: all-weak entries stay weak.
TEST_F(SymbolLookupSetTest, MergeEntriesAllWeakStaysWeak) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Weak);
  LS.add(intern("foo"), Weak);
  LS.add(intern("foo"), Weak);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"foo", Weak}}));
}

// A set holding entries for a single name only, so that the merged run reaches
// the end of the vector.
TEST_F(SymbolLookupSetTest, MergeEntriesSingleNameOnly) {
  SymbolLookupSet LS;
  for (unsigned I = 0; I != 5; ++I)
    LS.add(intern("foo"), Weak);
  LS.add(intern("foo"), Required);

  LS.mergeEntries();

  EXPECT_EQ(contents(LS), (Entries{{"foo", Required}}));
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

#ifndef NDEBUG
//===----------------------------------------------------------------------===//
// containsDuplicates
//===----------------------------------------------------------------------===//

TEST_F(SymbolLookupSetTest, ContainsDuplicatesFalseCases) {
  SymbolLookupSet Empty;
  EXPECT_FALSE(Empty.containsDuplicates());

  SymbolLookupSet One;
  One.add(intern("foo"));
  EXPECT_FALSE(One.containsDuplicates());

  SymbolLookupSet Distinct;
  Distinct.add(intern("foo"), Required);
  Distinct.add(intern("bar"), Weak);
  EXPECT_FALSE(Distinct.containsDuplicates());
}

// containsDuplicates compares names only, so entries differing in flags still
// count as duplicates.
TEST_F(SymbolLookupSetTest, ContainsDuplicatesIgnoresFlags) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("foo"), Weak);
  EXPECT_TRUE(LS.containsDuplicates());
}

// mergeEntries establishes the invariant that containsDuplicates checks: the
// two must agree on what a duplicate is.
TEST_F(SymbolLookupSetTest, MergeEntriesSatisfiesContainsDuplicates) {
  SymbolLookupSet LS;
  LS.add(intern("foo"), Required);
  LS.add(intern("foo"), Weak);
  LS.add(intern("bar"), Weak);
  LS.add(intern("bar"), Weak);
  LS.add(intern("baz"), Required);

  ASSERT_TRUE(LS.containsDuplicates());
  LS.mergeEntries();
  EXPECT_FALSE(LS.containsDuplicates());
}
#endif
