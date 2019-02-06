//===- unittests/ADT/FallibleIteratorTest.cpp - fallible_iterator.h tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/fallible_iterator.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

#include <utility>
#include <vector>

using namespace llvm;

namespace {

using ItemValid = enum { ValidItem, InvalidItem };
using LinkValid = enum { ValidLink, InvalidLink };

class Item {
public:
  Item(ItemValid V) : V(V) {}
  bool isValid() const { return V == ValidItem; }

private:
  ItemValid V;
};

// A utility to mock "bad collections". It supports both invalid items,
// where the dereference operator may return an Error, and bad links
// where the inc/dec operations may return an Error.
// Each element of the mock collection contains a pair of a (possibly broken)
// item and link.
using FallibleCollection = std::vector<std::pair<Item, LinkValid>>;

class FallibleCollectionWalker {
public:
  FallibleCollectionWalker(FallibleCollection &C, unsigned Idx)
      : C(C), Idx(Idx) {}

  Item &operator*() { return C[Idx].first; }

  const Item &operator*() const { return C[Idx].first; }

  Error inc() {
    assert(Idx != C.size() && "Walking off end of (mock) collection");
    if (C[Idx].second == ValidLink) {
      ++Idx;
      return Error::success();
    }
    return make_error<StringError>("cant get next object in (mock) collection",
                                   inconvertibleErrorCode());
  }

  Error dec() {
    assert(Idx != 0 && "Walking off start of (mock) collection");
    --Idx;
    if (C[Idx].second == ValidLink)
      return Error::success();
    return make_error<StringError>("cant get prev object in (mock) collection",
                                   inconvertibleErrorCode());
  }

  friend bool operator==(const FallibleCollectionWalker &LHS,
                         const FallibleCollectionWalker &RHS) {
    assert(&LHS.C == &RHS.C && "Comparing iterators across collectionss.");
    return LHS.Idx == RHS.Idx;
  }

private:
  FallibleCollection &C;
  unsigned Idx;
};

class FallibleCollectionWalkerWithStructDeref
    : public FallibleCollectionWalker {
public:
  using FallibleCollectionWalker::FallibleCollectionWalker;

  Item *operator->() { return &this->operator*(); }

  const Item *operator->() const { return &this->operator*(); }
};

class FallibleCollectionWalkerWithFallibleDeref
    : public FallibleCollectionWalker {
public:
  using FallibleCollectionWalker::FallibleCollectionWalker;

  Expected<Item &> operator*() {
    auto &I = FallibleCollectionWalker::operator*();
    if (!I.isValid())
      return make_error<StringError>("bad item", inconvertibleErrorCode());
    return I;
  }

  Expected<const Item &> operator*() const {
    const auto &I = FallibleCollectionWalker::operator*();
    if (!I.isValid())
      return make_error<StringError>("bad item", inconvertibleErrorCode());
    return I;
  }
};

TEST(FallibleIteratorTest, BasicSuccess) {

  // Check that a basic use-case involing successful iteration over a
  // "FallibleCollection" works.

  FallibleCollection C({{ValidItem, ValidLink}, {ValidItem, ValidLink}});

  FallibleCollectionWalker begin(C, 0);
  FallibleCollectionWalker end(C, 2);

  Error Err = Error::success();
  for (auto &Elem :
       make_fallible_range<FallibleCollectionWalker>(begin, end, Err))
    EXPECT_TRUE(Elem.isValid());
  cantFail(std::move(Err));
}

TEST(FallibleIteratorTest, BasicFailure) {

  // Check that a iteration failure (due to the InvalidLink state on element one
  // of the fallible collection) breaks out of the loop and raises an Error.

  FallibleCollection C({{ValidItem, ValidLink}, {ValidItem, InvalidLink}});

  FallibleCollectionWalker begin(C, 0);
  FallibleCollectionWalker end(C, 2);

  Error Err = Error::success();
  for (auto &Elem :
       make_fallible_range<FallibleCollectionWalker>(begin, end, Err))
    EXPECT_TRUE(Elem.isValid());

  EXPECT_THAT_ERROR(std::move(Err), Failed()) << "Expected failure value";
}

TEST(FallibleIteratorTest, NoRedundantErrorCheckOnEarlyExit) {

  // Check that an early return from the loop body does not require a redundant
  // check of Err.

  FallibleCollection C({{ValidItem, ValidLink}, {ValidItem, ValidLink}});

  FallibleCollectionWalker begin(C, 0);
  FallibleCollectionWalker end(C, 2);

  Error Err = Error::success();
  for (auto &Elem :
       make_fallible_range<FallibleCollectionWalker>(begin, end, Err)) {
    (void)Elem;
    return;
  }
  // Err not checked, but should be ok because we exit from the loop
  // body.
}

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
TEST(FallibleIteratorTest, RegularLoopExitRequiresErrorCheck) {

  // Check that Err must be checked after a normal (i.e. not early) loop exit
  // by failing to check and expecting program death (due to the unchecked
  // error).

  EXPECT_DEATH(
      {
        FallibleCollection C({{ValidItem, ValidLink}, {ValidItem, ValidLink}});

        FallibleCollectionWalker begin(C, 0);
        FallibleCollectionWalker end(C, 2);

        Error Err = Error::success();
        for (auto &Elem :
             make_fallible_range<FallibleCollectionWalker>(begin, end, Err))
          (void)Elem;
      },
      "Program aborted due to an unhandled Error:")
      << "Normal (i.e. not early) loop exit should require an error check";
}
#endif

TEST(FallibleIteratorTest, RawIncrementAndDecrementBehavior) {

  // Check the exact behavior of increment / decrement.

  FallibleCollection C({{ValidItem, ValidLink},
                        {ValidItem, InvalidLink},
                        {ValidItem, ValidLink},
                        {ValidItem, InvalidLink}});

  {
    // One increment from begin succeeds.
    Error Err = Error::success();
    auto I = make_fallible_itr(FallibleCollectionWalker(C, 0), Err);
    ++I;
    EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  }

  {
    // Two increments from begin fail.
    Error Err = Error::success();
    auto I = make_fallible_itr(FallibleCollectionWalker(C, 0), Err);
    ++I;
    EXPECT_THAT_ERROR(std::move(Err), Succeeded());
    ++I;
    EXPECT_THAT_ERROR(std::move(Err), Failed()) << "Expected failure value";
  }

  {
    // One decement from element three succeeds.
    Error Err = Error::success();
    auto I = make_fallible_itr(FallibleCollectionWalker(C, 3), Err);
    --I;
    EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  }

  {
    // One decement from element three succeeds.
    Error Err = Error::success();
    auto I = make_fallible_itr(FallibleCollectionWalker(C, 3), Err);
    --I;
    EXPECT_THAT_ERROR(std::move(Err), Succeeded());
    --I;
    EXPECT_THAT_ERROR(std::move(Err), Failed());
  }
}

TEST(FallibleIteratorTest, CheckStructDerefOperatorSupport) {
  // Check that the fallible_iterator wrapper forwards through to the
  // underlying iterator's structure dereference operator if present.

  FallibleCollection C({{ValidItem, ValidLink},
                        {ValidItem, ValidLink},
                        {InvalidItem, InvalidLink}});

  FallibleCollectionWalkerWithStructDeref begin(C, 0);

  {
    Error Err = Error::success();
    auto I = make_fallible_itr(begin, Err);
    EXPECT_TRUE(I->isValid());
    cantFail(std::move(Err));
  }

  {
    Error Err = Error::success();
    const auto I = make_fallible_itr(begin, Err);
    EXPECT_TRUE(I->isValid());
    cantFail(std::move(Err));
  }
}

TEST(FallibleIteratorTest, CheckDerefToExpectedSupport) {

  // Check that the fallible_iterator wrapper forwards value types, in
  // particular llvm::Expected, correctly.

  FallibleCollection C({{ValidItem, ValidLink},
                        {InvalidItem, ValidLink},
                        {ValidItem, ValidLink}});

  FallibleCollectionWalkerWithFallibleDeref begin(C, 0);
  FallibleCollectionWalkerWithFallibleDeref end(C, 3);

  Error Err = Error::success();
  auto I = make_fallible_itr(begin, Err);
  auto E = make_fallible_end(end);

  Expected<Item> V1 = *I;
  EXPECT_THAT_ERROR(V1.takeError(), Succeeded());
  ++I;
  EXPECT_NE(I, E); // Implicitly check error.
  Expected<Item> V2 = *I;
  EXPECT_THAT_ERROR(V2.takeError(), Failed());
  ++I;
  EXPECT_NE(I, E); // Implicitly check error.
  Expected<Item> V3 = *I;
  EXPECT_THAT_ERROR(V3.takeError(), Succeeded());
  ++I;
  EXPECT_EQ(I, E);
  cantFail(std::move(Err));
}

} // namespace
