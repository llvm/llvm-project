//===- IteratorTest.cpp - Unit tests for iterator utilities ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/iterator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ilist.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>
#include <type_traits>
#include <vector>

using namespace llvm;
using testing::ElementsAre;

namespace {

template <int> struct Shadow;

struct WeirdIter
    : llvm::iterator_facade_base<WeirdIter, std::input_iterator_tag, Shadow<0>,
                                 Shadow<1>, Shadow<2>, Shadow<3>> {};

struct AdaptedIter : iterator_adaptor_base<AdaptedIter, WeirdIter> {};

// Test that iterator_adaptor_base forwards typedefs, if value_type is
// unchanged.
static_assert(std::is_same_v<typename AdaptedIter::value_type, Shadow<0>>, "");
static_assert(std::is_same_v<typename AdaptedIter::difference_type, Shadow<1>>,
              "");
static_assert(std::is_same_v<typename AdaptedIter::pointer, Shadow<2>>, "");
static_assert(std::is_same_v<typename AdaptedIter::reference, Shadow<3>>, "");

// Ensure that pointe{e,r}_iterator adaptors correctly forward the category of
// the underlying iterator.

using RandomAccessIter = SmallVectorImpl<int*>::iterator;
using BidiIter = ilist<int*>::iterator;

template<class T>
using pointee_iterator_defaulted = pointee_iterator<T>;
template<class T>
using pointer_iterator_defaulted = pointer_iterator<T>;

// Ensures that an iterator and its adaptation have the same iterator_category.
template<template<typename> class A, typename It>
using IsAdaptedIterCategorySame =
  std::is_same<typename std::iterator_traits<It>::iterator_category,
               typename std::iterator_traits<A<It>>::iterator_category>;

// Check that dereferencing works correctly adapting pointers and proxies.
template <class T>
struct PointerWrapper : public iterator_adaptor_base<PointerWrapper<T>, T *> {
  PointerWrapper(T *I) : PointerWrapper::iterator_adaptor_base(I) {}
};
struct IntProxy {
  int &I;
  IntProxy(int &I) : I(I) {}
  void operator=(int NewValue) { I = NewValue; }
};
struct ConstIntProxy {
  const int &I;
  ConstIntProxy(const int &I) : I(I) {}
};
template <class T, class ProxyT>
struct PointerProxyWrapper
    : public iterator_adaptor_base<PointerProxyWrapper<T, ProxyT>, T *,
                                   std::random_access_iterator_tag, T,
                                   ptrdiff_t, T *, ProxyT> {
  PointerProxyWrapper(T *I) : PointerProxyWrapper::iterator_adaptor_base(I) {}
};
using IntIterator = PointerWrapper<int>;
using ConstIntIterator = PointerWrapper<const int>;
using IntProxyIterator = PointerProxyWrapper<int, IntProxy>;
using ConstIntProxyIterator = PointerProxyWrapper<const int, ConstIntProxy>;

// There should only be a single (const-qualified) operator*, operator->, and
// operator[]. This test confirms that there isn't a non-const overload. Rather
// than adding those, users should double-check that T, PointerT, and ReferenceT
// have the right constness, and/or make fields mutable.
static_assert(&IntIterator::operator* == &IntIterator::operator*, "");
static_assert(&IntIterator::operator-> == &IntIterator::operator->, "");
static_assert(&IntIterator::operator[] == &IntIterator::operator[], "");

template <class T, std::enable_if_t<std::is_assignable_v<T, int>, bool> = false>
constexpr bool canAssignFromInt(T &&) {
  return true;
}
template <class T,
          std::enable_if_t<!std::is_assignable_v<T, int>, bool> = false>
constexpr bool canAssignFromInt(T &&) {
  return false;
}

TEST(IteratorAdaptorTest, Dereference) {
  int Number = 1;

  // Construct some iterators and check whether they can be assigned to.
  IntIterator I(&Number);
  const IntIterator IC(&Number);
  ConstIntIterator CI(&Number);
  const ConstIntIterator CIC(&Number);
  EXPECT_EQ(true, canAssignFromInt(*I));    // int *
  EXPECT_EQ(true, canAssignFromInt(*IC));   // int *const
  EXPECT_EQ(false, canAssignFromInt(*CI));  // const int *
  EXPECT_EQ(false, canAssignFromInt(*CIC)); // const int *const

  // Prove that dereference and assignment work.
  EXPECT_EQ(1, *I);
  EXPECT_EQ(1, *IC);
  EXPECT_EQ(1, *CI);
  EXPECT_EQ(1, *CIC);
  *I = 2;
  EXPECT_EQ(2, Number);
  *IC = 3;
  EXPECT_EQ(3, Number);

  // Construct some proxy iterators and check whether they can be assigned to.
  IntProxyIterator P(&Number);
  const IntProxyIterator PC(&Number);
  ConstIntProxyIterator CP(&Number);
  const ConstIntProxyIterator CPC(&Number);
  EXPECT_EQ(true, canAssignFromInt(*P));    // int *
  EXPECT_EQ(true, canAssignFromInt(*PC));   // int *const
  EXPECT_EQ(false, canAssignFromInt(*CP));  // const int *
  EXPECT_EQ(false, canAssignFromInt(*CPC)); // const int *const

  // Prove that dereference and assignment work.
  EXPECT_EQ(3, (*P).I);
  EXPECT_EQ(3, (*PC).I);
  EXPECT_EQ(3, (*CP).I);
  EXPECT_EQ(3, (*CPC).I);
  *P = 4;
  EXPECT_EQ(4, Number);
  *PC = 5;
  EXPECT_EQ(5, Number);
}

// pointeE_iterator
static_assert(IsAdaptedIterCategorySame<pointee_iterator_defaulted,
                                        RandomAccessIter>::value, "");
static_assert(IsAdaptedIterCategorySame<pointee_iterator_defaulted,
                                        BidiIter>::value, "");
// pointeR_iterator
static_assert(IsAdaptedIterCategorySame<pointer_iterator_defaulted,
                                        RandomAccessIter>::value, "");
static_assert(IsAdaptedIterCategorySame<pointer_iterator_defaulted,
                                        BidiIter>::value, "");

TEST(PointeeIteratorTest, Basic) {
  int arr[4] = {1, 2, 3, 4};
  SmallVector<int *, 4> V;
  V.push_back(&arr[0]);
  V.push_back(&arr[1]);
  V.push_back(&arr[2]);
  V.push_back(&arr[3]);

  typedef pointee_iterator<SmallVectorImpl<int *>::const_iterator>
      test_iterator;

  test_iterator Begin, End;
  Begin = V.begin();
  End = test_iterator(V.end());

  test_iterator I = Begin;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(*V[i], *I);

    EXPECT_EQ(I, Begin + i);
    EXPECT_EQ(I, std::next(Begin, i));
    test_iterator J = Begin;
    J += i;
    EXPECT_EQ(I, J);
    EXPECT_EQ(*V[i], Begin[i]);

    EXPECT_NE(I, End);
    EXPECT_GT(End, I);
    EXPECT_LT(I, End);
    EXPECT_GE(I, Begin);
    EXPECT_LE(Begin, I);

    EXPECT_EQ(i, I - Begin);
    EXPECT_EQ(i, std::distance(Begin, I));
    EXPECT_EQ(Begin, I - i);

    test_iterator K = I++;
    EXPECT_EQ(K, std::prev(I));
  }
  EXPECT_EQ(End, I);
}

TEST(PointeeIteratorTest, SmartPointer) {
  SmallVector<std::unique_ptr<int>, 4> V;
  V.push_back(std::make_unique<int>(1));
  V.push_back(std::make_unique<int>(2));
  V.push_back(std::make_unique<int>(3));
  V.push_back(std::make_unique<int>(4));

  typedef pointee_iterator<
      SmallVectorImpl<std::unique_ptr<int>>::const_iterator>
      test_iterator;

  test_iterator Begin, End;
  Begin = V.begin();
  End = test_iterator(V.end());

  test_iterator I = Begin;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(*V[i], *I);

    EXPECT_EQ(I, Begin + i);
    EXPECT_EQ(I, std::next(Begin, i));
    test_iterator J = Begin;
    J += i;
    EXPECT_EQ(I, J);
    EXPECT_EQ(*V[i], Begin[i]);

    EXPECT_NE(I, End);
    EXPECT_GT(End, I);
    EXPECT_LT(I, End);
    EXPECT_GE(I, Begin);
    EXPECT_LE(Begin, I);

    EXPECT_EQ(i, I - Begin);
    EXPECT_EQ(i, std::distance(Begin, I));
    EXPECT_EQ(Begin, I - i);

    test_iterator K = I++;
    EXPECT_EQ(K, std::prev(I));
  }
  EXPECT_EQ(End, I);
}

TEST(PointeeIteratorTest, Range) {
  int A[] = {1, 2, 3, 4};
  SmallVector<int *, 4> V{&A[0], &A[1], &A[2], &A[3]};

  int I = 0;
  for (int II : make_pointee_range(V))
    EXPECT_EQ(A[I++], II);
}

TEST(PointeeIteratorTest, PointeeType) {
  struct S {
    int X;
    bool operator==(const S &RHS) const { return X == RHS.X; };
  };
  S A[] = {S{0}, S{1}};
  SmallVector<S *, 2> V{&A[0], &A[1]};

  pointee_iterator<SmallVectorImpl<S *>::const_iterator, const S> I = V.begin();
  for (int j = 0; j < 2; ++j, ++I) {
    EXPECT_EQ(*V[j], *I);
  }
}

TEST(FilterIteratorTest, Lambda) {
  auto IsOdd = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(A, IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, Enumerate) {
  auto IsOdd = [](auto N) { return N.value() % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Enumerate = llvm::enumerate(A);
  SmallVector<int> Actual;
  for (auto IndexedValue : make_filter_range(Enumerate, IsOdd))
    Actual.push_back(IndexedValue.value());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, CallableObject) {
  int Counter = 0;
  struct Callable {
    int &Counter;

    Callable(int &Counter) : Counter(Counter) {}

    bool operator()(int N) {
      Counter++;
      return N % 2 == 1;
    }
  };
  Callable IsOdd(Counter);
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(A, IsOdd);
  EXPECT_EQ(2, Counter);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_GE(Counter, 7);
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, FunctionPointer) {
  bool (*IsOdd)(int) = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(A, IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, Composition) {
  auto IsOdd = [](int N) { return N % 2 == 1; };
  std::unique_ptr<int> A[] = {std::make_unique<int>(0), std::make_unique<int>(1),
                              std::make_unique<int>(2), std::make_unique<int>(3),
                              std::make_unique<int>(4), std::make_unique<int>(5),
                              std::make_unique<int>(6)};
  using PointeeIterator = pointee_iterator<std::unique_ptr<int> *>;
  auto Range = make_filter_range(
      make_range(PointeeIterator(std::begin(A)), PointeeIterator(std::end(A))),
      IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, InputIterator) {
  struct InputIterator
      : iterator_adaptor_base<InputIterator, int *, std::input_iterator_tag> {
    InputIterator(int *It) : InputIterator::iterator_adaptor_base(It) {}
  };

  auto IsOdd = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};
  auto Range = make_filter_range(
      make_range(InputIterator(std::begin(A)), InputIterator(std::end(A))),
      IsOdd);
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual);
}

TEST(FilterIteratorTest, ReverseFilterRange) {
  auto IsOdd = [](int N) { return N % 2 == 1; };
  int A[] = {0, 1, 2, 3, 4, 5, 6};

  // Check basic reversal.
  auto Range = reverse(make_filter_range(A, IsOdd));
  SmallVector<int, 3> Actual(Range.begin(), Range.end());
  EXPECT_EQ((SmallVector<int, 3>{5, 3, 1}), Actual);

  // Check that the reverse of the reverse is the original.
  auto Range2 = reverse(reverse(make_filter_range(A, IsOdd)));
  SmallVector<int, 3> Actual2(Range2.begin(), Range2.end());
  EXPECT_EQ((SmallVector<int, 3>{1, 3, 5}), Actual2);

  // Check empty ranges.
  auto Range3 = reverse(make_filter_range(ArrayRef<int>(), IsOdd));
  SmallVector<int, 0> Actual3(Range3.begin(), Range3.end());
  EXPECT_EQ((SmallVector<int, 0>{}), Actual3);

  // Check that we don't skip the first element, provided it isn't filtered
  // away.
  auto IsEven = [](int N) { return N % 2 == 0; };
  auto Range4 = reverse(make_filter_range(A, IsEven));
  SmallVector<int, 4> Actual4(Range4.begin(), Range4.end());
  EXPECT_EQ((SmallVector<int, 4>{6, 4, 2, 0}), Actual4);
}

TEST(PointerIterator, Basic) {
  int A[] = {1, 2, 3, 4};
  pointer_iterator<int *> Begin(std::begin(A)), End(std::end(A));
  EXPECT_EQ(A, *Begin);
  ++Begin;
  EXPECT_EQ(A + 1, *Begin);
  ++Begin;
  EXPECT_EQ(A + 2, *Begin);
  ++Begin;
  EXPECT_EQ(A + 3, *Begin);
  ++Begin;
  EXPECT_EQ(Begin, End);
}

TEST(PointerIterator, Const) {
  int A[] = {1, 2, 3, 4};
  const pointer_iterator<int *> Begin(std::begin(A));
  EXPECT_EQ(A, *Begin);
  EXPECT_EQ(A + 1, std::next(*Begin, 1));
  EXPECT_EQ(A + 2, std::next(*Begin, 2));
  EXPECT_EQ(A + 3, std::next(*Begin, 3));
  EXPECT_EQ(A + 4, std::next(*Begin, 4));
}

TEST(PointerIterator, Range) {
  int A[] = {1, 2, 3, 4};
  int I = 0;
  for (int *P : make_pointer_range(A))
    EXPECT_EQ(A + I++, P);
}

TEST(ZipIteratorTest, Basic) {
  using namespace std;
  const SmallVector<unsigned, 6> pi{3, 1, 4, 1, 5, 9};
  SmallVector<bool, 6> odd{1, 1, 0, 1, 1, 1};
  const char message[] = "yynyyy\0";
  std::array<int, 2> shortArr = {42, 43};

  for (auto tup : zip(pi, odd, message)) {
    EXPECT_EQ(get<0>(tup) & 0x01, get<1>(tup));
    EXPECT_EQ(get<0>(tup) & 0x01 ? 'y' : 'n', get<2>(tup));
  }

  // Note the rvalue.
  for (auto tup : zip(pi, SmallVector<bool, 0>{1, 1, 0, 1, 1})) {
    EXPECT_EQ(get<0>(tup) & 0x01, get<1>(tup));
  }

  // Iterate until we run out elements in the *shortest* range.
  for (auto [idx, elem] : enumerate(zip(odd, shortArr))) {
    EXPECT_LT(idx, static_cast<size_t>(2));
  }
  for (auto [idx, elem] : enumerate(zip(shortArr, odd))) {
    EXPECT_LT(idx, static_cast<size_t>(2));
  }
}

TEST(ZipIteratorTest, ZipEqualBasic) {
  const SmallVector<unsigned, 6> pi = {3, 1, 4, 1, 5, 8};
  const SmallVector<bool, 6> vals = {1, 1, 0, 1, 1, 0};
  unsigned iters = 0;

  for (auto [lhs, rhs] : zip_equal(vals, pi)) {
    EXPECT_EQ(lhs, rhs & 0x01);
    ++iters;
  }

  EXPECT_EQ(iters, 6u);
}

template <typename T>
constexpr bool IsConstRef =
    std::is_reference_v<T> && std::is_const_v<std::remove_reference_t<T>>;

template <typename T>
constexpr bool IsBoolConstRef =
    std::is_same_v<llvm::remove_cvref_t<T>, std::vector<bool>::const_reference>;

/// Returns a `const` copy of the passed value. The `const` on the returned
/// value is intentional here so that `MakeConst` can be used in range-for
/// loops.
template <typename T> const T MakeConst(T &&value) {
  return std::forward<T>(value);
}

TEST(ZipIteratorTest, ZipEqualConstCorrectness) {
  const std::vector<unsigned> c_first = {3, 1, 4};
  std::vector<unsigned> first = c_first;
  const SmallVector<bool> c_second = {1, 1, 0};
  SmallVector<bool> second = c_second;

  for (auto [a, b, c, d] : zip_equal(c_first, first, c_second, second)) {
    b = 0;
    d = true;
    static_assert(IsConstRef<decltype(a)>);
    static_assert(!IsConstRef<decltype(b)>);
    static_assert(IsConstRef<decltype(c)>);
    static_assert(!IsConstRef<decltype(d)>);
  }

  EXPECT_THAT(first, ElementsAre(0, 0, 0));
  EXPECT_THAT(second, ElementsAre(true, true, true));

  std::vector<bool> nemesis = {true, false, true};
  const std::vector<bool> c_nemesis = nemesis;

  for (auto &&[a, b, c, d] : zip_equal(first, c_first, nemesis, c_nemesis)) {
    a = 2;
    c = true;
    static_assert(!IsConstRef<decltype(a)>);
    static_assert(IsConstRef<decltype(b)>);
    static_assert(!IsBoolConstRef<decltype(c)>);
    static_assert(IsBoolConstRef<decltype(d)>);
  }

  EXPECT_THAT(first, ElementsAre(2, 2, 2));
  EXPECT_THAT(nemesis, ElementsAre(true, true, true));

  unsigned iters = 0;
  for (const auto &[a, b, c, d] :
       zip_equal(first, c_first, nemesis, c_nemesis)) {
    static_assert(!IsConstRef<decltype(a)>);
    static_assert(IsConstRef<decltype(b)>);
    static_assert(!IsBoolConstRef<decltype(c)>);
    static_assert(IsBoolConstRef<decltype(d)>);
    ++iters;
  }
  EXPECT_EQ(iters, 3u);
  iters = 0;

  for (const auto &[a, b, c, d] :
       MakeConst(zip_equal(first, c_first, nemesis, c_nemesis))) {
    static_assert(!IsConstRef<decltype(a)>);
    static_assert(IsConstRef<decltype(b)>);
    static_assert(!IsBoolConstRef<decltype(c)>);
    static_assert(IsBoolConstRef<decltype(d)>);
    ++iters;
  }
  EXPECT_EQ(iters, 3u);
}

TEST(ZipIteratorTest, ZipEqualTemporaries) {
  unsigned iters = 0;

  // These temporary ranges get moved into the `tuple<...> storage;` inside
  // `zippy`. From then on, we can use references obtained from this storage to
  // access them. This does not rely on any lifetime extensions on the
  // temporaries passed to `zip_equal`.
  for (auto [a, b, c] : zip_equal(SmallVector<int>{1, 2, 3}, std::string("abc"),
                                  std::vector<bool>{true, false, true})) {
    a = 3;
    b = 'c';
    c = false;
    static_assert(!IsConstRef<decltype(a)>);
    static_assert(!IsConstRef<decltype(b)>);
    static_assert(!IsBoolConstRef<decltype(c)>);
    ++iters;
  }
  EXPECT_EQ(iters, 3u);
  iters = 0;

  for (auto [a, b, c] :
       MakeConst(zip_equal(SmallVector<int>{1, 2, 3}, std::string("abc"),
                           std::vector<bool>{true, false, true}))) {
    static_assert(IsConstRef<decltype(a)>);
    static_assert(IsConstRef<decltype(b)>);
    static_assert(IsBoolConstRef<decltype(c)>);
    ++iters;
  }
  EXPECT_EQ(iters, 3u);
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
// Check that an assertion is triggered when ranges passed to `zip_equal` differ
// in length.
TEST(ZipIteratorTest, ZipEqualNotEqual) {
  const SmallVector<unsigned, 6> pi = {3, 1, 4, 1, 5, 8};
  const SmallVector<bool, 2> vals = {1, 1};

  EXPECT_DEATH(zip_equal(pi, vals), "Iteratees do not have equal length");
  EXPECT_DEATH(zip_equal(vals, pi), "Iteratees do not have equal length");
  EXPECT_DEATH(zip_equal(pi, pi, vals), "Iteratees do not have equal length");
  EXPECT_DEATH(zip_equal(vals, vals, pi), "Iteratees do not have equal length");
}
#endif

TEST(ZipIteratorTest, ZipFirstBasic) {
  using namespace std;
  const SmallVector<unsigned, 6> pi{3, 1, 4, 1, 5, 9};
  unsigned iters = 0;

  for (auto tup : zip_first(SmallVector<bool, 0>{1, 1, 0, 1}, pi)) {
    EXPECT_EQ(get<0>(tup), get<1>(tup) & 0x01);
    iters += 1;
  }

  EXPECT_EQ(iters, 4u);
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
// Make sure that we can detect when the first range is not the shortest.
TEST(ZipIteratorTest, ZipFirstNotShortest) {
  const std::array<unsigned, 6> longer = {};
  const std::array<unsigned, 4> shorter = {};

  EXPECT_DEATH(zip_first(longer, shorter),
               "First iteratee is not the shortest");
  EXPECT_DEATH(zip_first(longer, shorter, longer),
               "First iteratee is not the shortest");
  EXPECT_DEATH(zip_first(longer, longer, shorter),
               "First iteratee is not the shortest");
}
#endif

TEST(ZipIteratorTest, ZipLongestBasic) {
  using namespace std;
  const vector<unsigned> pi{3, 1, 4, 1, 5, 9};
  const vector<StringRef> e{"2", "7", "1", "8"};

  {
    // Check left range longer than right.
    const vector<tuple<optional<unsigned>, optional<StringRef>>> expected{
        make_tuple(3, StringRef("2")), make_tuple(1, StringRef("7")),
        make_tuple(4, StringRef("1")), make_tuple(1, StringRef("8")),
        make_tuple(5, std::nullopt),   make_tuple(9, std::nullopt)};
    size_t iters = 0;
    for (auto tup : zip_longest(pi, e)) {
      EXPECT_EQ(tup, expected[iters]);
      iters += 1;
    }
    EXPECT_EQ(iters, expected.size());
  }

  {
    // Check right range longer than left.
    const vector<tuple<optional<StringRef>, optional<unsigned>>> expected{
        make_tuple(StringRef("2"), 3), make_tuple(StringRef("7"), 1),
        make_tuple(StringRef("1"), 4), make_tuple(StringRef("8"), 1),
        make_tuple(std::nullopt, 5),   make_tuple(std::nullopt, 9)};
    size_t iters = 0;
    for (auto tup : zip_longest(e, pi)) {
      EXPECT_EQ(tup, expected[iters]);
      iters += 1;
    }
    EXPECT_EQ(iters, expected.size());
  }
}

TEST(ZipIteratorTest, Mutability) {
  using namespace std;
  const SmallVector<unsigned, 4> pi{3, 1, 4, 1, 5, 9};
  char message[] = "hello zip\0";

  for (auto tup : zip(pi, message, message)) {
    EXPECT_EQ(get<1>(tup), get<2>(tup));
    get<2>(tup) = get<0>(tup) & 0x01 ? 'y' : 'n';
  }

  // note the rvalue
  for (auto tup : zip(message, "yynyyyzip\0")) {
    EXPECT_EQ(get<0>(tup), get<1>(tup));
  }
}

TEST(ZipIteratorTest, ZipFirstMutability) {
  using namespace std;
  vector<unsigned> pi{3, 1, 4, 1, 5, 9};
  unsigned iters = 0;

  for (auto tup : zip_first(SmallVector<bool, 0>{1, 1, 0, 1}, pi)) {
    get<1>(tup) = get<0>(tup);
    iters += 1;
  }

  EXPECT_EQ(iters, 4u);

  for (auto tup : zip_first(SmallVector<bool, 0>{1, 1, 0, 1}, pi)) {
    EXPECT_EQ(get<0>(tup), get<1>(tup));
  }
}

TEST(ZipIteratorTest, Filter) {
  using namespace std;
  vector<unsigned> pi{3, 1, 4, 1, 5, 9};

  unsigned iters = 0;
  // pi is length 6, but the zip RHS is length 7.
  auto zipped = zip_first(pi, vector<bool>{1, 1, 0, 1, 1, 1, 0});
  for (auto tup : make_filter_range(
           zipped, [](decltype(zipped)::value_type t) { return get<1>(t); })) {
    EXPECT_EQ(get<0>(tup) & 0x01, get<1>(tup));
    get<0>(tup) += 1;
    iters += 1;
  }

  // Should have skipped pi[2].
  EXPECT_EQ(iters, 5u);

  // Ensure that in-place mutation works.
  EXPECT_TRUE(all_of(pi, [](unsigned n) { return (n & 0x01) == 0; }));
}

TEST(ZipIteratorTest, Reverse) {
  using namespace std;
  vector<unsigned> ascending{0, 1, 2, 3, 4, 5};

  auto zipped = zip_first(ascending, vector<bool>{0, 1, 0, 1, 0, 1});
  unsigned last = 6;
  for (auto tup : reverse(zipped)) {
    // Check that this is in reverse.
    EXPECT_LT(get<0>(tup), last);
    last = get<0>(tup);
    EXPECT_EQ(get<0>(tup) & 0x01, get<1>(tup));
  }

  auto odds = [](decltype(zipped)::value_type tup) { return get<1>(tup); };
  last = 6;
  for (auto tup : make_filter_range(reverse(zipped), odds)) {
    EXPECT_LT(get<0>(tup), last);
    last = get<0>(tup);
    EXPECT_TRUE(get<0>(tup) & 0x01);
    get<0>(tup) += 1;
  }

  // Ensure that in-place mutation works.
  EXPECT_TRUE(all_of(ascending, [](unsigned n) { return (n & 0x01) == 0; }));
}

TEST(RangeTest, Distance) {
  std::vector<int> v1;
  std::vector<int> v2{1, 2, 3};

  EXPECT_EQ(std::distance(v1.begin(), v1.end()), size(v1));
  EXPECT_EQ(std::distance(v2.begin(), v2.end()), size(v2));
}

} // anonymous namespace
