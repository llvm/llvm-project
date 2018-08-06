//===- IteratorTest.cpp - Unit tests for iterator utilities ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/iterator.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

template <int> struct Shadow;

struct WeirdIter : std::iterator<std::input_iterator_tag, Shadow<0>, Shadow<1>,
                                 Shadow<2>, Shadow<3>> {};

struct AdaptedIter : iterator_adaptor_base<AdaptedIter, WeirdIter> {};

// Test that iterator_adaptor_base forwards typedefs, if value_type is
// unchanged.
static_assert(std::is_same<typename AdaptedIter::value_type, Shadow<0>>::value,
              "");
static_assert(
    std::is_same<typename AdaptedIter::difference_type, Shadow<1>>::value, "");
static_assert(std::is_same<typename AdaptedIter::pointer, Shadow<2>>::value,
              "");
static_assert(std::is_same<typename AdaptedIter::reference, Shadow<3>>::value,
              "");

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
  V.push_back(make_unique<int>(1));
  V.push_back(make_unique<int>(2));
  V.push_back(make_unique<int>(3));
  V.push_back(make_unique<int>(4));

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
  std::unique_ptr<int> A[] = {make_unique<int>(0), make_unique<int>(1),
                              make_unique<int>(2), make_unique<int>(3),
                              make_unique<int>(4), make_unique<int>(5),
                              make_unique<int>(6)};
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
    using BaseT =
        iterator_adaptor_base<InputIterator, int *, std::input_iterator_tag>;

    InputIterator(int *It) : BaseT(It) {}
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

  for (auto tup : zip(pi, odd, message)) {
    EXPECT_EQ(get<0>(tup) & 0x01, get<1>(tup));
    EXPECT_EQ(get<0>(tup) & 0x01 ? 'y' : 'n', get<2>(tup));
  }

  // note the rvalue
  for (auto tup : zip(pi, SmallVector<bool, 0>{1, 1, 0, 1, 1})) {
    EXPECT_EQ(get<0>(tup) & 0x01, get<1>(tup));
  }
}

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

TEST(IteratorRangeTest, DropBegin) {
  SmallVector<int, 5> vec{0, 1, 2, 3, 4};

  for (int n = 0; n < 5; ++n) {
    int i = n;
    for (auto &v : drop_begin(vec, n)) {
      EXPECT_EQ(v, i);
      i += 1;
    }
    EXPECT_EQ(i, 5);
  }
}

} // anonymous namespace
