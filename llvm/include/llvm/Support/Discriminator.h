//===---- llvm/Support/Discriminator.h -- Discriminator Utils ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the constants and utility functions for discriminators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DISCRIMINATOR_H
#define LLVM_SUPPORT_DISCRIMINATOR_H

// Utility functions for encoding / decoding discriminators.
/// With a given unsigned int \p U, use up to 13 bits to represent it.
/// old_bit 1~5  --> new_bit 1~5
/// old_bit 6~12 --> new_bit 7~13
/// new_bit_6 is 0 if higher bits (7~13) are all 0
static inline unsigned getPrefixEncodingFromUnsigned(unsigned U) {
  U &= 0xfff;
  return U > 0x1f ? (((U & 0xfe0) << 1) | (U & 0x1f) | 0x20) : U;
}

/// Reverse transformation as getPrefixEncodingFromUnsigned.
static inline unsigned getUnsignedFromPrefixEncoding(unsigned U) {
  if (U & 1)
    return 0;
  U >>= 1;
  return (U & 0x20) ? (((U >> 1) & 0xfe0) | (U & 0x1f)) : (U & 0x1f);
}

/// Returns the next component stored in discriminator.
static inline unsigned getNextComponentInDiscriminator(unsigned D) {
  if ((D & 1) == 0)
    return D >> ((D & 0x40) ? 14 : 7);
  else
    return D >> 1;
}

static inline unsigned encodeComponent(unsigned C) {
  return (C == 0) ? 1U : (getPrefixEncodingFromUnsigned(C) << 1);
}

static inline unsigned encodingBits(unsigned C) {
  return (C == 0) ? 1 : (C > 0x1f ? 14 : 7);
}

// Some constants used in FS Discriminators.
#define BASE_DIS_BIT_BEG 0
#define BASE_DIS_BIT_END 7

#define PASS_1_DIS_BIT_BEG 8
#define PASS_1_DIS_BIT_END 13

#define PASS_2_DIS_BIT_BEG 14
#define PASS_2_DIS_BIT_END 19

#define PASS_3_DIS_BIT_BEG 20
#define PASS_3_DIS_BIT_END 25

#define PASS_LAST_DIS_BIT_BEG 26
#define PASS_LAST_DIS_BIT_END 31

// Set bits range [0 .. n] to 1. Used in FS Discriminators.
static inline unsigned getN1Bits(int N) {
  if (N >= 31)
    return 0xFFFFFFFF;
  return (1 << (N + 1)) - 1;
}

#endif /* LLVM_SUPPORT_DISCRIMINATOR_H */
