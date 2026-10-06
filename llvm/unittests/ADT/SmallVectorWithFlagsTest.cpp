//===- llvm/unittest/ADT/SmallVectorWithFlagsTest.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVectorWithFlags.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <type_traits>

using namespace llvm;
using testing::ElementsAre;

namespace {

template <typename T, unsigned N, unsigned Bits>
constexpr bool SameSize =
    sizeof(SmallVectorWithFlags<T, N, Bits>) == sizeof(SmallVector<T, N>);
static_assert(SameSize<int, 0, 1> && SameSize<int, 1, 1> &&
              SameSize<int, 4, 2>);
static_assert(SameSize<char, 0, 2> && SameSize<char, 8, 1>);
static_assert(SameSize<void *, 4, 2>);
static_assert(SameSize<std::string, 1, 2>);
static_assert(!std::is_convertible_v<SmallVectorWithFlags<int, 4> &,
                                     SmallVectorImpl<int> &>);
// Copying from the N-independent interface would have to choose between
// copying the flags and preserving them; it is not provided, like SmallVector.
static_assert(!std::is_constructible_v<SmallVectorWithFlags<int, 4>,
                                       const SmallVectorWithFlagsImpl<int> &>);

struct alignas(32) Aligned {
  int Value;
};
static_assert(SameSize<Aligned, 0, 2> && SameSize<Aligned, 2, 4>);

template <typename T> T element(unsigned I) {
  if constexpr (std::is_same_v<T, std::string>) {
    return std::to_string(I);
  } else if constexpr (std::is_same_v<T, void *>) {
    static int Pool[64];
    return &Pool[I];
  } else {
    return static_cast<T>(I);
  }
}

template <typename V>
class SmallVectorWithFlagsTypedTest : public testing::Test {};
using VectorTypes = testing::Types<
    SmallVectorWithFlags<int, 0>, SmallVectorWithFlags<int, 1>,
    SmallVectorWithFlags<int, 4, 2>, SmallVectorWithFlags<int, 4, 4>,
    SmallVectorWithFlags<char, 0, 2>, SmallVectorWithFlags<char, 4>,
    SmallVectorWithFlags<char, 2, 4>, SmallVectorWithFlags<void *, 2, 2>,
    SmallVectorWithFlags<std::string, 0>,
    SmallVectorWithFlags<std::string, 1, 2>>;
TYPED_TEST_SUITE(SmallVectorWithFlagsTypedTest, VectorTypes);

TYPED_TEST(SmallVectorWithFlagsTypedTest, FlagsDoNotChangeElementsOrPointer) {
  TypeParam V;
  using T = typename TypeParam::value_type;
  // The flag bits come out of the maximum capacity, and nothing else.
  EXPECT_EQ((SmallVector<T, 0>().max_size()) >> TypeParam::NumFlagBits,
            V.max_size());
  EXPECT_EQ(0u, V.getFlags());
  EXPECT_FALSE(V.getFlag(0));
  auto *Data = V.data();
  auto Capacity = V.capacity();
  V.setFlag(0, true);
  EXPECT_TRUE(V.getFlag(0));
  EXPECT_EQ(Data, V.data());
  EXPECT_EQ(Capacity, V.capacity());
  EXPECT_TRUE(V.empty());
  if constexpr (TypeParam::NumFlagBits >= 2) {
    V.setFlag(1, true);
    EXPECT_EQ(3u, V.getFlags());
    V.setFlag(0, false);
    EXPECT_EQ(2u, V.getFlags());
  }
  V.push_back(element<T>(7));
  unsigned Flags = V.getFlags();
  Data = V.data();
  V.setFlags(0);
  EXPECT_EQ(Data, V.data());
  EXPECT_EQ(element<T>(7), V.front());
  V.setFlags(Flags);
  EXPECT_EQ(Data, V.data());
  EXPECT_EQ(element<T>(7), V.front());
}

TYPED_TEST(SmallVectorWithFlagsTypedTest, EveryFlagValueKeepsCapacity) {
  using T = typename TypeParam::value_type;
  for (unsigned Size : {0u, 1u, 16u}) {
    TypeParam V(Size, element<T>(3));
    size_t Capacity = V.capacity();
    for (unsigned Flags = 0; Flags < (1u << TypeParam::NumFlagBits); ++Flags) {
      V.setFlags(Flags);
      EXPECT_EQ(Flags, V.getFlags());
      EXPECT_EQ(Capacity, V.capacity());
      EXPECT_EQ(Size, V.size());
      for (unsigned Bit = 0; Bit < TypeParam::NumFlagBits; ++Bit)
        EXPECT_EQ((Flags >> Bit) & 1, V.getFlag(Bit));
    }
    V.push_back(element<T>(4));
    EXPECT_EQ((1u << TypeParam::NumFlagBits) - 1, V.getFlags());
    EXPECT_EQ(element<T>(4), V.back());
  }
}

TYPED_TEST(SmallVectorWithFlagsTypedTest, ElementOperationsPreserveFlags) {
  TypeParam V;
  using T = typename TypeParam::value_type;
  unsigned Flags = (1u << TypeParam::NumFlagBits) - 1;
  V.setFlags(Flags);
  for (unsigned I = 0; I < 32; ++I) {
    V.push_back(element<T>(I));
    EXPECT_EQ(Flags, V.getFlags());
    EXPECT_LE(V.size(), V.capacity());
    for (unsigned J = 0; J <= I; ++J)
      EXPECT_EQ(element<T>(J), V[J]);
  }
  V.reserve(100);
  EXPECT_EQ(Flags, V.getFlags());
  // The reported capacity is the real one, without the flag bits.
  EXPECT_LE(100u, V.capacity());
  EXPECT_GT(1000u, V.capacity());
  V.insert(V.begin() + 1, element<T>(42));
  EXPECT_EQ(element<T>(42), V[1]);
  EXPECT_EQ(Flags, V.getFlags());
  V.erase(V.begin() + 1);
  EXPECT_EQ(Flags, V.getFlags());
  V.pop_back();
  EXPECT_EQ(Flags, V.getFlags());
  V.resize(2);
  EXPECT_EQ(Flags, V.getFlags());
  V.emplace_back(element<T>(8));
  EXPECT_EQ(Flags, V.getFlags());
  V.append({element<T>(1), element<T>(2)});
  EXPECT_EQ(Flags, V.getFlags());
  V.truncate(3);
  EXPECT_EQ(Flags, V.getFlags());
  V.clear();
  EXPECT_EQ(Flags, V.getFlags());
  V.assign(3, element<T>(9));
  EXPECT_EQ(Flags, V.getFlags());
  EXPECT_EQ(3u, V.size());
  V = {element<T>(4)};
  EXPECT_EQ(Flags, V.getFlags());
  EXPECT_EQ(element<T>(4), V.front());
  // Exercise aliased arguments during both inline-to-heap and heap growth.
  V.assign(V.capacity(), element<T>(5));
  V.push_back(V.front());
  EXPECT_EQ(element<T>(5), V.back());
  EXPECT_EQ(Flags, V.getFlags());
}

TYPED_TEST(SmallVectorWithFlagsTypedTest, CopyAndMoveIncludeFlags) {
  using T = typename TypeParam::value_type;
  for (unsigned Size : {0u, 1u, 16u}) {
    TypeParam Source(Size, element<T>(7));
    unsigned Flags = (1u << TypeParam::NumFlagBits) - 1;
    Source.setFlags(Flags);
    // Sixteen elements exceed every inline capacity in the type list.
    bool OnHeap = Size == 16;
    T *SourceData = Source.data();
    TypeParam Copy(Source);
    EXPECT_EQ(Source, Copy);
    EXPECT_EQ(Flags, Copy.getFlags());
    TypeParam Assigned(3, element<T>(9));
    Assigned = Source;
    EXPECT_EQ(Source, Assigned);
    EXPECT_EQ(Flags, Assigned.getFlags());
    TypeParam Moved(std::move(Source));
    EXPECT_EQ(Copy, Moved);
    EXPECT_EQ(Flags, Moved.getFlags());
    if (OnHeap)
      EXPECT_EQ(SourceData, Moved.data());
    EXPECT_TRUE(Source.empty());
    EXPECT_EQ(Flags, Source.getFlags());
    Source.push_back(element<T>(8));
    EXPECT_EQ(Flags, Source.getFlags());
    EXPECT_EQ(element<T>(8), Source.front());
    Assigned.setFlags(0);
    T *MovedData = Moved.data();
    Assigned = std::move(Moved);
    EXPECT_EQ(Copy, Assigned);
    EXPECT_EQ(Flags, Assigned.getFlags());
    if (OnHeap)
      EXPECT_EQ(MovedData, Assigned.data());
    EXPECT_TRUE(Moved.empty());
    EXPECT_EQ(Flags, Moved.getFlags());
    // Aliased base references exercise self assignment without warnings.
    const TypeParam &Self = Assigned;
    Assigned = Self;
    TypeParam &MoveSelf = Assigned;
    Assigned = std::move(MoveSelf);
    EXPECT_EQ(Copy, Assigned);
    EXPECT_EQ(Flags, Assigned.getFlags());
  }
}

TYPED_TEST(SmallVectorWithFlagsTypedTest, SwapIncludesFlagsInEveryStorageMode) {
  using T = typename TypeParam::value_type;
  for (unsigned LeftSize : {0u, 1u, 16u}) {
    for (unsigned RightSize : {0u, 1u, 16u}) {
      TypeParam L(LeftSize, element<T>(7)), R(RightSize, element<T>(9));
      L.setFlag(0, true);
      L.swap(R);
      EXPECT_FALSE(L.getFlag(0));
      EXPECT_TRUE(R.getFlag(0));
      EXPECT_EQ(RightSize, L.size());
      EXPECT_EQ(LeftSize, R.size());
      for (const T &Value : L)
        EXPECT_EQ(element<T>(9), Value);
      for (const T &Value : R)
        EXPECT_EQ(element<T>(7), Value);
      L.swap(L);
      EXPECT_FALSE(L.getFlag(0));
      using std::swap;
      swap(L, R);
      EXPECT_TRUE(L.getFlag(0));
      EXPECT_FALSE(R.getFlag(0));
      std::swap(L, R);
      EXPECT_FALSE(L.getFlag(0));
      EXPECT_TRUE(R.getFlag(0));
    }
  }
}

TEST(SmallVectorWithFlagsTest, BaseOperationsPreserveOwnerFlags) {
  SmallVectorWithFlags<int, 1, 2> L{1}, R{2, 3};
  L.setFlags(1);
  R.setFlags(2);
  SmallVectorWithFlagsImpl<int, 2> &LB = L, &RB = R;
  LB.swap(RB);
  EXPECT_THAT(L, ElementsAre(2, 3));
  EXPECT_THAT(R, ElementsAre(1));
  EXPECT_EQ(1u, L.getFlags());
  EXPECT_EQ(2u, R.getFlags());
  LB = RB;
  EXPECT_EQ(1u, L.getFlags());
  LB = std::move(RB);
  EXPECT_THAT(L, ElementsAre(1));
  EXPECT_EQ(1u, L.getFlags());
  EXPECT_EQ(2u, R.getFlags());
}

TEST(SmallVectorWithFlagsTest, BaseMoveAndSwapTransferHeapBuffers) {
  SmallVectorWithFlags<int, 1, 2> L(16, 7), R(16, 9);
  L.setFlags(1);
  R.setFlags(2);
  int *LD = L.data(), *RD = R.data();
  SmallVectorWithFlagsImpl<int, 2> &LB = L, &RB = R;
  LB.swap(RB);
  EXPECT_EQ(RD, L.data());
  EXPECT_EQ(LD, R.data());
  EXPECT_EQ(1u, L.getFlags());
  EXPECT_EQ(2u, R.getFlags());
  EXPECT_EQ(9, L.front());
  LB = std::move(RB);
  EXPECT_EQ(LD, L.data());
  EXPECT_EQ(7, L.front());
  EXPECT_EQ(1u, L.getFlags());
  EXPECT_EQ(2u, R.getFlags());
  EXPECT_TRUE(R.empty());
}

TEST(SmallVectorWithFlagsTest, AssignAndSwapWithImplUpdateOnlyElements) {
  SmallVectorWithFlags<int, 4, 2> L{1}, R{2, 3};
  L.setFlags(1);
  R.setFlags(2);
  SmallVectorWithFlagsImpl<int, 2> &RB = R;
  L = RB;
  EXPECT_THAT(L, ElementsAre(2, 3));
  EXPECT_EQ(1u, L.getFlags());
  R.assign({4});
  L.swap(RB);
  EXPECT_THAT(L, ElementsAre(4));
  EXPECT_THAT(R, ElementsAre(2, 3));
  EXPECT_EQ(1u, L.getFlags());
  EXPECT_EQ(2u, R.getFlags());
  L = std::move(RB);
  EXPECT_THAT(L, ElementsAre(2, 3));
  EXPECT_EQ(1u, L.getFlags());
  EXPECT_EQ(2u, R.getFlags());
  EXPECT_TRUE(R.empty());
}

TEST(SmallVectorWithFlagsTest, DifferentInlineCapacities) {
  SmallVectorWithFlags<int, 1, 2> L{1};
  SmallVectorWithFlags<int, 4, 2> R{2, 3};
  L.setFlags(1);
  R.setFlags(2);
  L.swap(R);
  EXPECT_THAT(L, ElementsAre(2, 3));
  EXPECT_THAT(R, ElementsAre(1));
  EXPECT_EQ(2u, L.getFlags());
  EXPECT_EQ(1u, R.getFlags());
  L = R;
  EXPECT_EQ(1u, L.getFlags());
  R.setFlags(3);
  L = std::move(R);
  EXPECT_THAT(L, ElementsAre(1));
  EXPECT_EQ(3u, L.getFlags());
  EXPECT_EQ(3u, R.getFlags());
}

TEST(SmallVectorWithFlagsTest, ArrayRefAndConstructors) {
  int Values[] = {1, 2, 3};
  SmallVectorWithFlags<int, 4> V{1, 2, 3};
  SmallVectorWithFlags<int, 1> Range(std::begin(Values), std::end(Values));
  SmallVectorWithFlags<int, 0> FromArray{ArrayRef<int>(Values)};
  SmallVectorWithFlags<int, 2> FromIteratorRange(
      make_range(std::begin(Values), std::end(Values)));
  SmallVectorWithFlags<int, 2> Sized(3);
  SmallVectorWithFlags<int, 2> FromLong{ArrayRef<long>({1L, 2L, 3L})};
  EXPECT_THAT(Range, ElementsAre(1, 2, 3));
  EXPECT_THAT(FromArray, ElementsAre(1, 2, 3));
  EXPECT_THAT(FromIteratorRange, ElementsAre(1, 2, 3));
  EXPECT_THAT(Sized, ElementsAre(0, 0, 0));
  EXPECT_THAT(FromLong, ElementsAre(1, 2, 3));
  V.setFlag(0, true);
  ArrayRef Read = V;
  MutableArrayRef Write = V;
  static_assert(std::is_same_v<decltype(Read), ArrayRef<int>>);
  static_assert(std::is_same_v<decltype(Write), MutableArrayRef<int>>);
  EXPECT_EQ(V.data(), Read.data());
  EXPECT_EQ(ArrayRef<int>(V), Read);
  EXPECT_EQ(V.capacity() * sizeof(int), capacity_in_bytes(V));
  Write[0] = 4;
  EXPECT_TRUE(V.getFlag(0));
  EXPECT_EQ(4, V.front());
  SmallVectorWithFlagsImpl<int> &Base = V;
  ArrayRef BaseRead = Base;
  MutableArrayRef BaseWrite = Base;
  EXPECT_EQ(V.data(), BaseRead.data());
  EXPECT_EQ(V.data(), BaseWrite.data());
}

template <typename L, typename R, typename = void>
struct HasEquality : std::false_type {};

template <typename L, typename R>
struct HasEquality<L, R,
                   std::void_t<decltype(std::declval<const L &>() ==
                                        std::declval<const R &>())>>
    : std::true_type {};

// A view cannot represent flags: discarding them for comparison is explicit.
static_assert(!HasEquality<SmallVectorWithFlags<int, 4>, ArrayRef<int>>::value);
static_assert(!HasEquality<ArrayRef<int>, SmallVectorWithFlags<int, 4>>::value);

TEST(SmallVectorWithFlagsTest, EqualityIncludesFlagsAcrossInlineCapacities) {
  SmallVectorWithFlags<int, 1, 2> L;
  SmallVectorWithFlags<int, 4, 2> R;
  EXPECT_EQ(L, R);
  for (unsigned Flags = 1; Flags < 4; ++Flags) {
    R.setFlags(Flags);
    EXPECT_NE(L, R);
    EXPECT_NE(R, L);
    L.setFlags(Flags);
    EXPECT_EQ(L, R);
    L.setFlags(0);
  }
  L = {1, 2};
  R = {1, 2};
  L.setFlags(3);
  R.setFlags(3);
  EXPECT_EQ(L, R);
  R.setFlags(1);
  EXPECT_NE(L, R);
  EXPECT_EQ(ArrayRef<int>(L), ArrayRef<int>(R));
  const SmallVectorWithFlagsImpl<int, 2> &LB = L, &RB = R;
  EXPECT_NE(LB, RB);
  R.setFlags(3);
  EXPECT_EQ(LB, RB);
  R.back() = 4;
  EXPECT_NE(LB, RB);
}

TEST(SmallVectorWithFlagsTest, OrderingIncludesFlagsAfterElements) {
  SmallVectorWithFlags<int, 1, 2> L{1, 2};
  SmallVectorWithFlags<int, 4, 2> R{1, 2};
  L.setFlags(1);
  R.setFlags(2);
  EXPECT_LT(L, R);
  EXPECT_LE(L, R);
  EXPECT_GT(R, L);
  EXPECT_GE(R, L);
  EXPECT_FALSE(R < L);
  EXPECT_FALSE(L > R);
  EXPECT_FALSE(R <= L);
  EXPECT_FALSE(L >= R);
  R.setFlags(1);
  EXPECT_EQ(L, R);
  EXPECT_FALSE(L < R);
  EXPECT_FALSE(L > R);
  EXPECT_LE(L, R);
  EXPECT_GE(L, R);
  L.setFlags(3);
  R.back() = 3;
  EXPECT_LT(L, R); // Elements take precedence over flags.
  R = {1, 2, 0};
  EXPECT_LT(L, R); // A shorter equal prefix sorts first.
}

TEST(SmallVectorWithFlagsTest, EnclosingClassEqualityIncludesFlags) {
  struct Owner {
    SmallVectorWithFlags<int, 4, 2> Values;
    bool operator==(const Owner &RHS) const { return Values == RHS.Values; }
  };
  Owner L{{1, 2}}, R{{1, 2}};
  EXPECT_EQ(L, R);
  R.Values.setFlags(2);
  EXPECT_FALSE(L == R);
  L.Values.setFlags(2);
  EXPECT_EQ(L, R);
}

// Defaulted comparisons require C++20; this test only runs in such builds.
#if defined(__cpp_impl_three_way_comparison) &&                                \
    __cpp_impl_three_way_comparison >= 201907L
TEST(SmallVectorWithFlagsTest, DefaultedEnclosingClassEqualityIncludesFlags) {
  struct Owner {
    SmallVectorWithFlags<int, 4, 2> Values;
    bool operator==(const Owner &) const = default;
  };
  Owner L{{1, 2}}, R{{1, 2}};
  EXPECT_EQ(L, R);
  R.Values.setFlags(2);
  EXPECT_NE(L, R);
  L.Values.setFlags(2);
  EXPECT_EQ(L, R);
}
#endif

struct MoveConstructOnly {
  explicit MoveConstructOnly(int Value) : Value(Value) {}
  MoveConstructOnly(MoveConstructOnly &&) = default;
  MoveConstructOnly &operator=(MoveConstructOnly &&) = delete;
  int Value;
};

TEST(SmallVectorWithFlagsTest, MoveConstructNonMoveAssignableElements) {
  for (unsigned Size : {0u, 1u, 16u}) {
    SmallVectorWithFlags<MoveConstructOnly, 1> Source;
    for (unsigned I = 0; I < Size; ++I)
      Source.emplace_back(I);
    Source.setFlag(0, true);
    bool OnHeap = Size > 1;
    MoveConstructOnly *SourceData = Source.data();
    SmallVectorWithFlags<MoveConstructOnly, 1> Dest(std::move(Source));
    EXPECT_TRUE(Dest.getFlag(0));
    EXPECT_TRUE(Source.getFlag(0));
    EXPECT_TRUE(Source.empty());
    if (OnHeap)
      EXPECT_EQ(SourceData, Dest.data());
    ASSERT_EQ(Size, Dest.size());
    for (unsigned I = 0; I < Size; ++I)
      EXPECT_EQ(static_cast<int>(I), Dest[I].Value);
    MoveConstructOnly *DestData = Dest.data();
    SmallVectorWithFlags<MoveConstructOnly, 0> FromBase(std::move(
        static_cast<SmallVectorWithFlagsImpl<MoveConstructOnly> &>(Dest)));
    EXPECT_TRUE(FromBase.getFlag(0));
    EXPECT_EQ(Size, FromBase.size());
    if (OnHeap)
      EXPECT_EQ(DestData, FromBase.data());
  }
}

TEST(SmallVectorWithFlagsTest, OwningElements) {
  SmallVectorWithFlags<std::unique_ptr<int>, 1> V;
  V.setFlag(0, true);
  for (int I = 0; I < 16; ++I)
    V.push_back(std::make_unique<int>(I));
  SmallVectorWithFlags<std::unique_ptr<int>, 1> Moved(std::move(V));
  EXPECT_TRUE(Moved.getFlag(0));
  for (int I = 0; I < 16; ++I)
    EXPECT_EQ(I, *Moved[I]);
  EXPECT_TRUE(V.getFlag(0));
}

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
TEST(SmallVectorWithFlagsTest, InvalidFlags) {
  SmallVectorWithFlags<int, 0, 2> V;
  EXPECT_DEATH(V.setFlags(4), "flags exceed FlagBits");
  EXPECT_DEATH(V.setFlag(2, true), "flag index out of range");
  EXPECT_DEATH(V.getFlag(2), "flag index out of range");
}
#endif

} // namespace
