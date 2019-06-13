//==-------------- half_type.cpp --- SYCL half type ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/half_type.hpp>
// This is included to enable __builtin_expect()
#include <CL/sycl/detail/platform_util.hpp>
#include <iostream>
#include <cstring>

namespace cl {
namespace sycl {
namespace detail {

static uint16_t float2Half(const float &Val) {
  const uint32_t Bits = *reinterpret_cast<const uint32_t *>(&Val);

  // Extract the sign from the float value
  const uint16_t Sign = (Bits & 0x80000000) >> 16;
  // Extract the fraction from the float value
  const uint32_t Frac32 = Bits & 0x7fffff;
  // Extract the exponent from the float value
  const uint8_t Exp32 = (Bits & 0x7f800000) >> 23;
  const int8_t Exp32Diff = Exp32 - 127;

  uint16_t Exp16 = 0;
  uint16_t Frac16 = Frac32 >> 13;

  if (__builtin_expect(Exp32 == 0xff || Exp32Diff > 15, 0)) {
    Exp16 = 0x1f;
  } else if (__builtin_expect(Exp32 == 0 || Exp32Diff < -14, 0)) {
    Exp16 = 0;
  } else {
    Exp16 = Exp32Diff + 15;
  }

  if (__builtin_expect(Exp32 == 0xff && Frac32 != 0 && Frac16 == 0, 0)) {
    // corner case 1: NaN
    // This case happens when FP32 value is NaN whose the fraction part
    // transformed to FP16 counterpart is truncated to 0. We need to flip the
    // high bit to 1 to make it distinguished from inf.
    Frac16 = 0x200;
  } else if (__builtin_expect(Exp32 == 0 || (Exp16 == 0x1f && Exp32 != 0xff),
                              0)) {
    // corner case 2: subnormal
    // All FP32 subnormal values are under the range of FP16 so the fraction
    // part is set to 0.
    // corner case 3: overflow
    Frac16 = 0;
  } else if (__builtin_expect(Exp16 == 0 && Exp32 != 0, 0)) {
    // corner case 4: underflow
    // We use `truncate` mode here.
    Frac16 = 0x100 | (Frac16 >> 2);
  }

  // Compose the final FP16 binary
  uint16_t Ret = 0;
  Ret |= Sign;
  Ret |= Exp16 << 10;
  Ret |= Frac16;

  return Ret;
}

static float half2Float(const uint16_t &Val) {
  // Extract the sign from the bits
  const uint32_t Sign = static_cast<uint32_t>(Val & 0x8000) << 16;
  // Extract the exponent from the bits
  const uint8_t Exp16 = (Val & 0x7c00) >> 10;
  // Extract the fraction from the bits
  uint16_t Frac16 = Val & 0x3ff;

  uint32_t Exp32 = 0;
  if (__builtin_expect(Exp16 == 0x1f, 0)) {
    Exp32 = 0xff;
  } else if (__builtin_expect(Exp16 == 0, 0)) {
    Exp32 = 0;
  } else {
    Exp32 = static_cast<uint32_t>(Exp16) + 112;
  }

  // corner case: subnormal -> normal
  // The denormal number of FP16 can be represented by FP32, therefore we need
  // to recover the exponent and recalculate the fration.
  if (__builtin_expect(Exp16 == 0 && Frac16 != 0, 0)) {
    uint8_t OffSet = 0;
    do {
      ++OffSet;
      Frac16 <<= 1;
    } while ((Frac16 & 0x400) != 0x400);
    // mask the 9th bit
    Frac16 &= 0x3ff;
    Exp32 = 113 - OffSet;
  }

  uint32_t Frac32 = Frac16 << 13;

  // Compose the final FP32 binary
  uint32_t Bits = 0;

  Bits |= Sign;
  Bits |= (Exp32 << 23);
  Bits |= Frac32;

  float Result;
  std::memcpy(&Result, &Bits, sizeof(Result));
  return Result;
}

std::ostream &operator<<(std::ostream &O, const half_impl::half &Val) {
  O << static_cast<float>(Val);
  return O;
}

std::istream &operator>>(std::istream &I, half_impl::half &ValHalf) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  ValHalf = ValFloat;
  return I;
}

namespace half_impl {

half::half(const float &RHS) : Buf(float2Half(RHS)) {}

half &half::operator+=(const half &RHS) {
  *this = operator float() + static_cast<float>(RHS);
  return *this;
}

half &half::operator-=(const half &RHS) {
  *this = operator float() - static_cast<float>(RHS);
  return *this;
}

half &half::operator*=(const half &RHS) {
  *this = operator float() * static_cast<float>(RHS);
  return *this;
}

half &half::operator/=(const half &RHS) {
  *this = operator float() / static_cast<float>(RHS);
  return *this;
}

half::operator float() const { return half2Float(Buf); }

// Operator +, -, *, /
half operator+(half LHS, const half &RHS) {
  LHS += RHS;
  return LHS;
}

half operator-(half LHS, const half &RHS) {
  LHS -= RHS;
  return LHS;
}

half operator*(half LHS, const half &RHS) {
  LHS *= RHS;
  return LHS;
}

half operator/(half LHS, const half &RHS) {
  LHS /= RHS;
  return LHS;
}

// Operator <, >, <=, >=
bool operator<(const half &LHS, const half &RHS) {
  return static_cast<float>(LHS) < static_cast<float>(RHS);
}

bool operator>(const half &LHS, const half &RHS) { return RHS < LHS; }

bool operator<=(const half &LHS, const half &RHS) { return !(LHS > RHS); }

bool operator>=(const half &LHS, const half &RHS) { return !(LHS < RHS); }

// Operator ==, !=
bool operator==(const half &LHS, const half &RHS) {
  return static_cast<float>(LHS) == static_cast<float>(RHS);
}

bool operator!=(const half &LHS, const half &RHS) { return !(LHS == RHS); }
} // namespace half_impl

} // namespace detail
} // namespace sycl
} // namespace cl
