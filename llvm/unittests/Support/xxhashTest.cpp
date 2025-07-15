//===- llvm/unittest/Support/xxhashTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/xxhash.h"
#include "gtest/gtest.h"

using namespace llvm;

/* use #define to make them constant, required for initialization */
#define PRIME32 2654435761U
#define PRIME64 11400714785074694797ULL

/*
 * Fills a test buffer with pseudorandom data.
 *
 * This is used in the sanity check - its values must not be changed.
 */
static void fillTestBuffer(uint8_t *buffer, size_t len) {
  uint64_t byteGen = PRIME32;

  assert(buffer != NULL);

  for (size_t i = 0; i < len; i++) {
    buffer[i] = (uint8_t)(byteGen >> 56);
    byteGen *= PRIME64;
  }
}

TEST(xxhashTest, Basic) {
  EXPECT_EQ(0xef46db3751d8e999U, xxHash64(StringRef()));
  EXPECT_EQ(0x33bf00a859c4ba3fU, xxHash64("foo"));
  EXPECT_EQ(0x48a37c90ad27a659U, xxHash64("bar"));
  EXPECT_EQ(0x69196c1b3af0bff9U,
            xxHash64("0123456789abcdefghijklmnopqrstuvwxyz"));
}

TEST(xxhashTest, xxh3) {
  constexpr size_t size = 2243;
  uint8_t a[size];
  uint64_t x = 1;
  for (size_t i = 0; i < size; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    a[i] = uint8_t(x);
  }

#define F(len, expected)                                                       \
  EXPECT_EQ(uint64_t(expected), xxh3_64bits(ArrayRef(a, size_t(len))))
  F(0, 0x2d06800538d394c2);
  F(1, 0xd0d496e05c553485);
  F(2, 0x84d625edb7055eac);
  F(3, 0x6ea2d59aca5c3778);
  F(4, 0xbf65290914e80242);
  F(5, 0xc01fd099ad4fc8e4);
  F(6, 0x9e3ea8187399caa5);
  F(7, 0x9da8b60540644f5a);
  F(8, 0xabc1413da6cd0209);
  F(9, 0x8bc89400bfed51f6);
  F(16, 0x7e46916754d7c9b8);
  F(17, 0xed4be912ba5f836d);
  F(32, 0xf59b59b58c304fd1);
  F(33, 0x9013fb74ca603e0c);
  F(64, 0xfa5271fcce0db1c3);
  F(65, 0x79c42431727f1012);
  F(96, 0x591ee0ddf9c9ccd1);
  F(97, 0x8ffc6a3111fe19da);
  F(128, 0x06a146ee9a2da378);
  F(129, 0xbc7138129bf065da);
  F(403, 0xcefeb3ffa532ad8c);
  F(512, 0xcdfa6b6268e3650f);
  F(513, 0x4bb5d42742f9765f);
  F(2048, 0x330ce110cbb79eae);
  F(2049, 0x3ba6afa0249fef9a);
  F(2240, 0xd61d4d2a94e926a8);
  F(2243, 0x0979f786a24edde7);
#undef F
}

TEST(xxhashTest, xxh3_128bits) {
#define SANITY_BUFFER_SIZE 2367
  uint8_t sanityBuffer[SANITY_BUFFER_SIZE];

  fillTestBuffer(sanityBuffer, sizeof(sanityBuffer));

#define F(len, expected)                                                       \
  EXPECT_EQ(XXH128_hash_t(expected),                                           \
            xxh3_128bits(ArrayRef(sanityBuffer, size_t(len))))

  F(0, (XXH128_hash_t{0x6001C324468D497FULL,
                      0x99AA06D3014798D8ULL})); /* empty string */
  F(1, (XXH128_hash_t{0xC44BDFF4074EECDBULL,
                      0xA6CD5E9392000F6AULL})); /*  1 -  3 */
  F(6, (XXH128_hash_t{0x3E7039BDDA43CFC6ULL,
                      0x082AFE0B8162D12AULL})); /*  4 -  8 */
  F(12, (XXH128_hash_t{0x061A192713F69AD9ULL,
                       0x6E3EFD8FC7802B18ULL})); /*  9 - 16 */
  F(24, (XXH128_hash_t{0x1E7044D28B1B901DULL,
                       0x0CE966E4678D3761ULL})); /* 17 - 32 */
  F(48, (XXH128_hash_t{0xF942219AED80F67BULL,
                       0xA002AC4E5478227EULL})); /* 33 - 64 */
  F(81, (XXH128_hash_t{0x5E8BAFB9F95FB803ULL,
                       0x4952F58181AB0042ULL})); /* 65 - 96 */
  F(222, (XXH128_hash_t{0xF1AEBD597CEC6B3AULL,
                        0x337E09641B948717ULL})); /* 129-240 */
  F(403,
    (XXH128_hash_t{
        0xCDEB804D65C6DEA4ULL,
        0x1B6DE21E332DD73DULL})); /* one block, last stripe is overlapping */
  F(512,
    (XXH128_hash_t{
        0x617E49599013CB6BULL,
        0x18D2D110DCC9BCA1ULL})); /* one block, finishing at stripe boundary */
  F(2048,
    (XXH128_hash_t{
        0xDD59E2C3A5F038E0ULL,
        0xF736557FD47073A5ULL})); /* 2 blocks, finishing at block boundary */
  F(2240,
    (XXH128_hash_t{
        0x6E73A90539CF2948ULL,
        0xCCB134FBFA7CE49DULL})); /* 3 blocks, finishing at stripe boundary */
  F(2367,
    (XXH128_hash_t{
        0xCB37AEB9E5D361EDULL,
        0xE89C0F6FF369B427ULL})); /* 3 blocks, last stripe is overlapping */
#undef F
}
