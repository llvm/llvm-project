//===- LEB128.cpp - LEB128 utility functions implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some utility functions for encoding SLEB128 and
// ULEB128 values.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LEB128.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::support;

namespace llvm {

/// Utility function to get the size of the ULEB128-encoded value.
unsigned getULEB128Size(uint64_t Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

/// Utility function to get the size of the SLEB128-encoded value.
unsigned getSLEB128Size(int64_t Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;

  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}
}  // namespace llvm

void llvm::encodeUCLeb128(uint64_t x, raw_ostream &os) {
  // Fast path for n == 1
  if (x < 128) {
    os.write((x << 1) | 1);
    return;
  }

  unsigned significantBits = 64 - countl_zero(x);
  unsigned n = (significantBits + 6) / 7;
  if (n > 8) {
    // 9 bytes: 00000000 xxxxxxxx ...
    os.write(0);
    endian::write(os, x, endianness::little);
    return;
  }

  uint64_t tagged = endian::byte_swap((x << n) | ((uint64_t)1 << (n - 1)),
                                      endianness::little);
  os.write((const char *)&tagged, n);
}

template <int n>
static inline uint64_t getUCLeb128Case(const uint8_t *&p, uint8_t byte) {
  uint64_t val = byte >> n;
  int shift = 8 - n;
  for (int i = 1; i < n; ++i) {
    val |= (uint64_t)p[i] << shift;
    shift += 8;
  }
  p += n;
  return val;
}

template <bool CheckBounds>
static uint64_t getUCLeb128Impl(const uint8_t *&p, const uint8_t *end) {
  if constexpr (CheckBounds) {
    if (p >= end)
      return 0;
  }
  // Fast path for n == 1
  uint8_t b0 = p[0];
  if (b0 & 1) {
    ++p;
    return b0 >> 1;
  }

  unsigned n = llvm::countr_zero(b0) + 1;
  if constexpr (CheckBounds) {
    if (end - p < n)
      return 0;
  }
  // Note: If n < 9 and we allow out-of-bounds read, we can use read64le(p) <<
  // (64-8*n) >> (64-7*n) instead of the following switch statement.
  switch (n) {
  case 1:
    return getUCLeb128Case<1>(p, b0);
  case 2:
    return getUCLeb128Case<2>(p, b0);
  case 3:
    return getUCLeb128Case<3>(p, b0);
  case 4:
    return getUCLeb128Case<4>(p, b0);
  case 5:
    return getUCLeb128Case<5>(p, b0);
  case 6:
    return getUCLeb128Case<6>(p, b0);
  case 7:
    return getUCLeb128Case<7>(p, b0);
  case 8:
    return getUCLeb128Case<8>(p, b0);
  default:
    // 9 bytes: 00000000 xxxxxxxx ...
    p += 9;
    return endian::read64le(p - 8);
  }
}

uint64_t llvm::getUCLeb128(const uint8_t *&p, const uint8_t *end) {
  return getUCLeb128Impl<true>(p, end);
}

uint64_t llvm::getUCLeb128Unsafe(const uint8_t *&p) {
  return getUCLeb128Impl<false>(p, nullptr);
}
