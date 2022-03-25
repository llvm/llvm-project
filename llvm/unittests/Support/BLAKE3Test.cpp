//===- llvm/unittest/Support/BLAKE3Test.cpp - BLAKE3 tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the BLAKE3 functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/BLAKE3.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

/// Tests an arbitrary set of bytes passed as \p Input.
void TestBLAKE3Sum(ArrayRef<uint8_t> Input, StringRef Final) {
  BLAKE3 Hash;
  Hash.update(Input);
  auto hash = Hash.final();
  auto hashStr = toHex(hash);
  EXPECT_EQ(hashStr, Final);
}

using KV = std::pair<const char *, const char *>;

TEST(BLAKE3Test, BLAKE3) {
  std::array<KV, 5> testvectors{
      KV{"",
         "AF1349B9F5F9A1A6A0404DEA36DCC9499BCB25C9ADC112B7CC9A93CAE41F3262"},
      KV{"a",
         "17762FDDD969A453925D65717AC3EEA21320B66B54342FDE15128D6CAF21215F"},
      KV{"abc",
         "6437B3AC38465133FFB63B75273A8DB548C558465D79DB03FD359C6CD5BD9D85"},
      KV{"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "C19012CC2AAF0DC3D8E5C45A1B79114D2DF42ABB2A410BF54BE09E891AF06FF8"},
      KV{"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklm"
         "nopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
         "553E1AA2A477CB3166E6AB38C12D59F6C5017F0885AAF079F217DA00CFCA363F"}};

  for (auto input_expected : testvectors) {
    auto str = std::get<0>(input_expected);
    auto expected = std::get<1>(input_expected);
    TestBLAKE3Sum({reinterpret_cast<const uint8_t *>(str), strlen(str)},
                  expected);
  }

  std::string rep(1000, 'a');
  BLAKE3 Hash;
  for (int i = 0; i < 1000; ++i) {
    Hash.update({reinterpret_cast<const uint8_t *>(rep.data()), rep.size()});
  }
  auto hash = Hash.final();
  auto hashStr = toHex(hash);
  EXPECT_EQ(hashStr,
            "616F575A1B58D4C9797D4217B9730AE5E6EB319D76EDEF6549B46F4EFE31FF8B");
}

TEST(BLAKE3Test, SmallerHashSize) {
  const char *InputStr = "abc";
  ArrayRef<uint8_t> Input(reinterpret_cast<const uint8_t *>(InputStr),
                          strlen(InputStr));
  BLAKE3 Hash;
  Hash.update(Input);
  auto hash1 = Hash.final<16>();
  auto hash2 = BLAKE3::hash<16>(Input);
  auto hashStr1 = toHex(hash1);
  auto hashStr2 = toHex(hash2);
  EXPECT_EQ(hashStr1, hashStr2);
  EXPECT_EQ(hashStr1, "6437B3AC38465133FFB63B75273A8DB5");
}

} // namespace
