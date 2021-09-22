//===- llvm/unittest/Support/HashBuilderTest.cpp - HashBuilder unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/HashBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/SHA256.h"
#include "gtest/gtest.h"

#include <list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// gtest utilities and macros rely on using a single type. So wrap both the
// hasher type and endianness.
template <typename _HasherT, llvm::support::endianness _Endianness>
struct HasherTAndEndianness {
  using HasherT = _HasherT;
  static constexpr llvm::support::endianness Endianness = _Endianness;
};
using HasherTAndEndiannessToTest =
    ::testing::Types<HasherTAndEndianness<llvm::MD5, llvm::support::big>,
                     HasherTAndEndianness<llvm::MD5, llvm::support::little>,
                     HasherTAndEndianness<llvm::MD5, llvm::support::native>,
                     HasherTAndEndianness<llvm::SHA1, llvm::support::big>,
                     HasherTAndEndianness<llvm::SHA1, llvm::support::little>,
                     HasherTAndEndianness<llvm::SHA1, llvm::support::native>,
                     HasherTAndEndianness<llvm::SHA256, llvm::support::big>,
                     HasherTAndEndianness<llvm::SHA256, llvm::support::little>,
                     HasherTAndEndianness<llvm::SHA256, llvm::support::native>>;
template <typename HasherT> class HashBuilderTest : public testing::Test {};
TYPED_TEST_SUITE(HashBuilderTest, HasherTAndEndiannessToTest, );

template <typename HasherTAndEndianness>
using HashBuilder = llvm::HashBuilder<typename HasherTAndEndianness::HasherT,
                                      HasherTAndEndianness::Endianness>;

template <typename HasherTAndEndianness, typename... Ts>
static std::string hashWithBuilder(const Ts &...Args) {
  return HashBuilder<HasherTAndEndianness>().add(Args...).final().str();
}

template <typename HasherTAndEndianness, typename... Ts>
static std::string hashRangeWithBuilder(const Ts &...Args) {
  return HashBuilder<HasherTAndEndianness>().addRange(Args...).final().str();
}

// All the test infrastructure relies on the variadic helpers. Test them first.
TYPED_TEST(HashBuilderTest, VariadicHelpers) {
  {
    HashBuilder<TypeParam> HBuilder;

    HBuilder.add(100);
    HBuilder.add('c');
    HBuilder.add("string");

    EXPECT_EQ(HBuilder.final(), hashWithBuilder<TypeParam>(100, 'c', "string"));
  }

  {
    HashBuilder<TypeParam> HBuilder;

    std::vector<int> Vec{100, 101, 102};
    HBuilder.addRange(Vec);

    EXPECT_EQ(HBuilder.final(), hashRangeWithBuilder<TypeParam>(Vec));
  }

  {
    HashBuilder<TypeParam> HBuilder;

    std::vector<int> Vec{200, 201, 202};
    HBuilder.addRange(Vec.begin(), Vec.end());

    EXPECT_EQ(HBuilder.final(),
              hashRangeWithBuilder<TypeParam>(Vec.begin(), Vec.end()));
  }
}

TYPED_TEST(HashBuilderTest, AddRangeElements) {
  HashBuilder<TypeParam> HBuilder;
  int Values[] = {1, 2, 3};
  HBuilder.addRangeElements(llvm::ArrayRef<int>(Values));
  EXPECT_EQ(HBuilder.final(), hashWithBuilder<TypeParam>(1, 2, 3));
}

TYPED_TEST(HashBuilderTest, AddHashableData) {
  using HE = TypeParam;

  auto ByteSwapAndHashWithHasher = [](auto Data) {
    using H = typename HE::HasherT;
    constexpr auto E = HE::Endianness;
    H Hasher;
    auto SwappedData = llvm::support::endian::byte_swap(Data, E);
    Hasher.update(llvm::makeArrayRef(
        reinterpret_cast<const uint8_t *>(&SwappedData), sizeof(Data)));
    return static_cast<std::string>(Hasher.final());
  };

  char C = 'c';
  int32_t I = 0x12345678;
  uint64_t UI64 = static_cast<uint64_t>(1) << 50;
  enum TestEnumeration : uint16_t { TE_One = 1, TE_Two = 2 };
  TestEnumeration Enum = TE_Two;

  EXPECT_EQ(ByteSwapAndHashWithHasher(C), hashWithBuilder<HE>(C));
  EXPECT_EQ(ByteSwapAndHashWithHasher(I), hashWithBuilder<HE>(I));
  EXPECT_EQ(ByteSwapAndHashWithHasher(UI64), hashWithBuilder<HE>(UI64));
  EXPECT_EQ(ByteSwapAndHashWithHasher(Enum), hashWithBuilder<HE>(Enum));
}

struct SimpleStruct {
  char C;
  int I;
};

template <typename HasherT, llvm::support::endianness Endianness>
void addHash(llvm::HashBuilderImpl<HasherT, Endianness> &HBuilder,
             const SimpleStruct &Value) {
  HBuilder.add(Value.C);
  HBuilder.add(Value.I);
}

struct StructWithoutCopyOrMove {
  int I;
  StructWithoutCopyOrMove() = default;
  StructWithoutCopyOrMove(const StructWithoutCopyOrMove &) = delete;
  StructWithoutCopyOrMove &operator=(const StructWithoutCopyOrMove &) = delete;

  template <typename HasherT, llvm::support::endianness Endianness>
  friend void addHash(llvm::HashBuilderImpl<HasherT, Endianness> &HBuilder,
                      const StructWithoutCopyOrMove &Value) {
    HBuilder.add(Value.I);
  }
};

// The struct and associated tests are simplified to avoid failures caused by
// different alignments on different platforms.
struct /* __attribute__((packed)) */ StructWithFastHash {
  int I;
  // char C;

  // If possible, we want to hash both `I` and `C` in a single `update`
  // call for performance concerns.
  template <typename HasherT, llvm::support::endianness Endianness>
  friend void addHash(llvm::HashBuilderImpl<HasherT, Endianness> &HBuilder,
                      const StructWithFastHash &Value) {
    if (Endianness == llvm::support::endian::system_endianness()) {
      HBuilder.update(llvm::makeArrayRef(
          reinterpret_cast<const uint8_t *>(&Value), sizeof(Value)));
    } else {
      // Rely on existing `add` methods to handle endianness.
      HBuilder.add(Value.I);
      // HBuilder.add(Value.C);
    }
  }
};

struct CustomContainer {
private:
  size_t Size;
  int Elements[100];

public:
  CustomContainer(size_t Size) : Size(Size) {
    for (size_t I = 0; I != Size; ++I)
      Elements[I] = I;
  }
  template <typename HasherT, llvm::support::endianness Endianness>
  friend void addHash(llvm::HashBuilderImpl<HasherT, Endianness> &HBuilder,
                      const CustomContainer &Value) {
    if (Endianness == llvm::support::endian::system_endianness()) {
      HBuilder.update(llvm::makeArrayRef(
          reinterpret_cast<const uint8_t *>(&Value.Size),
          sizeof(Value.Size) + Value.Size * sizeof(Value.Elements[0])));
    } else {
      HBuilder.addRange(&Value.Elements[0], &Value.Elements[0] + Value.Size);
    }
  }
};

TYPED_TEST(HashBuilderTest, HashUserDefinedStruct) {
  using HE = TypeParam;
  EXPECT_EQ(hashWithBuilder<HE>(SimpleStruct{'c', 123}),
            hashWithBuilder<HE>('c', 123));
  EXPECT_EQ(hashWithBuilder<HE>(StructWithoutCopyOrMove{1}),
            hashWithBuilder<HE>(1));
  EXPECT_EQ(hashWithBuilder<HE>(StructWithFastHash{123}),
            hashWithBuilder<HE>(123));
  EXPECT_EQ(hashWithBuilder<HE>(CustomContainer(3)),
            hashWithBuilder<HE>(static_cast<size_t>(3), 0, 1, 2));
}

TYPED_TEST(HashBuilderTest, HashArrayRefHashableDataTypes) {
  using HE = TypeParam;
  int Values[] = {1, 20, 0x12345678};
  llvm::ArrayRef<int> Array(Values);
  EXPECT_NE(hashWithBuilder<HE>(Array), hashWithBuilder<HE>(1, 20, 0x12345678));
  EXPECT_EQ(hashWithBuilder<HE>(Array),
            hashRangeWithBuilder<HE>(Array.begin(), Array.end()));
  EXPECT_EQ(
      hashWithBuilder<HE>(Array),
      hashRangeWithBuilder<HE>(Array.data(), Array.data() + Array.size()));
}

TYPED_TEST(HashBuilderTest, HashArrayRef) {
  using HE = TypeParam;
  int Values[] = {1, 2, 3};
  llvm::ArrayRef<int> Array123(&Values[0], 3);
  llvm::ArrayRef<int> Array12(&Values[0], 2);
  llvm::ArrayRef<int> Array1(&Values[0], 1);
  llvm::ArrayRef<int> Array23(&Values[1], 2);
  llvm::ArrayRef<int> Array3(&Values[2], 1);
  llvm::ArrayRef<int> ArrayEmpty(&Values[0], static_cast<size_t>(0));

  auto Hash123andEmpty = hashWithBuilder<HE>(Array123, ArrayEmpty);
  auto Hash12And3 = hashWithBuilder<HE>(Array12, Array3);
  auto Hash1And23 = hashWithBuilder<HE>(Array1, Array23);
  auto HashEmptyAnd123 = hashWithBuilder<HE>(ArrayEmpty, Array123);

  EXPECT_NE(Hash123andEmpty, Hash12And3);
  EXPECT_NE(Hash123andEmpty, Hash1And23);
  EXPECT_NE(Hash123andEmpty, HashEmptyAnd123);
  EXPECT_NE(Hash12And3, Hash1And23);
  EXPECT_NE(Hash12And3, HashEmptyAnd123);
  EXPECT_NE(Hash1And23, HashEmptyAnd123);
}

TYPED_TEST(HashBuilderTest, HashArrayRefNonHashableDataTypes) {
  using HE = TypeParam;
  SimpleStruct Values[] = {{'a', 100}, {'b', 200}};
  llvm::ArrayRef<SimpleStruct> Array(Values);
  EXPECT_NE(
      hashWithBuilder<HE>(Array),
      hashWithBuilder<HE>(SimpleStruct{'a', 100}, SimpleStruct{'b', 200}));
}

TYPED_TEST(HashBuilderTest, HashStringRef) {
  using HE = TypeParam;
  llvm::StringRef SEmpty("");
  llvm::StringRef S1("1");
  llvm::StringRef S12("12");
  llvm::StringRef S123("123");
  llvm::StringRef S23("23");
  llvm::StringRef S3("3");

  auto Hash123andEmpty = hashWithBuilder<HE>(S123, SEmpty);
  auto Hash12And3 = hashWithBuilder<HE>(S12, S3);
  auto Hash1And23 = hashWithBuilder<HE>(S1, S23);
  auto HashEmptyAnd123 = hashWithBuilder<HE>(SEmpty, S123);

  EXPECT_NE(Hash123andEmpty, Hash12And3);
  EXPECT_NE(Hash123andEmpty, Hash1And23);
  EXPECT_NE(Hash123andEmpty, HashEmptyAnd123);
  EXPECT_NE(Hash12And3, Hash1And23);
  EXPECT_NE(Hash12And3, HashEmptyAnd123);
  EXPECT_NE(Hash1And23, HashEmptyAnd123);
}

TYPED_TEST(HashBuilderTest, HashStdString) {
  using HE = TypeParam;
  EXPECT_EQ(hashWithBuilder<HE>(std::string("123")),
            hashWithBuilder<HE>(llvm::StringRef("123")));
}

TYPED_TEST(HashBuilderTest, HashStdPair) {
  using HE = TypeParam;
  EXPECT_EQ(hashWithBuilder<HE>(std::make_pair(1, "string")),
            hashWithBuilder<HE>(1, "string"));

  std::pair<StructWithoutCopyOrMove, std::string> Pair;
  Pair.first.I = 1;
  Pair.second = "string";
  EXPECT_EQ(hashWithBuilder<HE>(Pair), hashWithBuilder<HE>(1, "string"));
}

TYPED_TEST(HashBuilderTest, HashStdTuple) {
  using HE = TypeParam;

  EXPECT_EQ(hashWithBuilder<HE>(std::make_tuple(1)), hashWithBuilder<HE>(1));
  EXPECT_EQ(hashWithBuilder<HE>(std::make_tuple(2ULL)),
            hashWithBuilder<HE>(2ULL));
  EXPECT_EQ(hashWithBuilder<HE>(std::make_tuple("three")),
            hashWithBuilder<HE>("three"));
  EXPECT_EQ(hashWithBuilder<HE>(std::make_tuple(1, 2ULL)),
            hashWithBuilder<HE>(1, 2ULL));
  EXPECT_EQ(hashWithBuilder<HE>(std::make_tuple(1, 2ULL, "three")),
            hashWithBuilder<HE>(1, 2ULL, "three"));

  std::tuple<StructWithoutCopyOrMove, std::string> Tuple;
  std::get<0>(Tuple).I = 1;
  std::get<1>(Tuple) = "two";

  EXPECT_EQ(hashWithBuilder<HE>(Tuple), hashWithBuilder<HE>(1, "two"));
}

TYPED_TEST(HashBuilderTest, HashRangeWithForwardIterator) {
  using HE = TypeParam;
  std::list<int> List;
  List.push_back(1);
  List.push_back(2);
  List.push_back(3);
  EXPECT_NE(hashRangeWithBuilder<HE>(List), hashWithBuilder<HE>(1, 2, 3));
}

TEST(CustomHasher, CustomHasher) {
  struct SumHash {
    explicit SumHash(uint8_t Seed1, uint8_t Seed2) : Hash(Seed1 + Seed2) {}
    void update(llvm::ArrayRef<uint8_t> Data) {
      for (uint8_t C : Data)
        Hash += C;
    }
    uint8_t Hash;
  };

  {
    llvm::HashBuilder<SumHash, llvm::support::endianness::little> HBuilder(0,
                                                                           1);
    EXPECT_EQ(HBuilder.add(0x02, 0x03, 0x400).getHasher().Hash, 0xa);
  }
  {
    llvm::HashBuilder<SumHash, llvm::support::endianness::little> HBuilder(2,
                                                                           3);
    EXPECT_EQ(HBuilder.add("ab", 'c').getHasher().Hash,
              static_cast<uint8_t>(/*seeds*/ 2 + 3 + /*range size*/ 2 +
                                   /*characters*/ 'a' + 'b' + 'c'));
  }
}
