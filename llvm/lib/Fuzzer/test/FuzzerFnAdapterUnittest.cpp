// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

#include "FuzzerFnAdapter.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace fuzzer {
namespace impl {

template <typename... Args>
bool Unpack(std::tuple<Args...> *Tuple, std::initializer_list<uint8_t> data) {
  std::vector<uint8_t> V(data);
  return Unpack(V.data(), V.size(), Tuple);
}

TEST(Unpack, Bool) {
  std::tuple<bool> T;
  EXPECT_TRUE(Unpack(&T, {1}));
  EXPECT_TRUE(std::get<0>(T));

  EXPECT_TRUE(Unpack(&T, {0}));
  EXPECT_FALSE(std::get<0>(T));

  EXPECT_FALSE(Unpack(&T, {}));
}

TEST(Unpack, BoolBool) {
  std::tuple<bool, bool> T;
  EXPECT_TRUE(Unpack(&T, {1, 0}));
  EXPECT_TRUE(std::get<0>(T));
  EXPECT_FALSE(std::get<1>(T));

  EXPECT_TRUE(Unpack(&T, {0, 1}));
  EXPECT_FALSE(std::get<0>(T));
  EXPECT_TRUE(std::get<1>(T));

  EXPECT_FALSE(Unpack(&T, {}));
  EXPECT_FALSE(Unpack(&T, {10}));
}

TEST(Unpack, BoolInt) {
  std::tuple<bool, int> T;
  EXPECT_TRUE(Unpack(&T, {1, 16, 2, 0, 0}));
  EXPECT_TRUE(std::get<0>(T));
  EXPECT_EQ(528, std::get<1>(T));

  EXPECT_FALSE(Unpack(&T, {1, 2}));
}

TEST(Unpack, Vector) {
  std::tuple<std::vector<uint8_t>> T;
  const auto &V = std::get<0>(T);

  EXPECT_FALSE(Unpack(&T, {}));

  EXPECT_TRUE(Unpack(&T, {0}));
  EXPECT_EQ(0ul, V.size());

  EXPECT_TRUE(Unpack(&T, {0, 1, 2, 3}));
  EXPECT_EQ(0ul, V.size());

  EXPECT_TRUE(Unpack(&T, {2}));
  EXPECT_EQ(0ul, V.size());

  EXPECT_TRUE(Unpack(&T, {2, 3}));
  EXPECT_EQ(1ul, V.size());
  EXPECT_EQ(3, V[0]);

  EXPECT_TRUE(Unpack(&T, {2, 9, 8}));
  EXPECT_EQ(2ul, V.size());
  EXPECT_EQ(9, V[0]);
  EXPECT_EQ(8, V[1]);
}

TEST(Unpack, String) {
  std::tuple<std::string> T;
  const auto &S = std::get<0>(T);

  EXPECT_TRUE(Unpack(&T, {2, 3}));
  EXPECT_EQ(1ul, S.size());
  EXPECT_EQ(3, S[0]);
}

template <typename Fn>
bool UnpackAndApply(Fn F, std::initializer_list<uint8_t> Data) {
  std::vector<uint8_t> V(Data);
  return UnpackAndApply(F, V.data(), V.size());
}

static void fnBool(bool b) { EXPECT_TRUE(b); }

TEST(Apply, Bool) {
  EXPECT_FALSE(UnpackAndApply(fnBool, {}));
  EXPECT_TRUE(UnpackAndApply(fnBool, {1}));
  EXPECT_NONFATAL_FAILURE(UnpackAndApply(fnBool, {0}),
                          "Actual: false\nExpected: true");
}

static void fnInt(int i) { EXPECT_EQ(42, i); }

TEST(Apply, Int) {
  EXPECT_FALSE(UnpackAndApply(fnInt, {}));
  EXPECT_TRUE(UnpackAndApply(fnInt, {42, 0, 0, 0}));
  EXPECT_NONFATAL_FAILURE(UnpackAndApply(fnInt, {10, 0, 0, 0}),
                          "Actual: 10\nExpected: 42");
}

} // namespace impl
} // namespace fuzzer
