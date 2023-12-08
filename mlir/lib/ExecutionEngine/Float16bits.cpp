//===--- Float16bits.cpp - supports 2-byte floats  ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements f16 and bf16 to support the compilation and execution
// of programs using these types.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/Float16bits.h"

#ifdef MLIR_FLOAT16_DEFINE_FUNCTIONS // We are building this library

#include <cmath>
#include <cstring>

namespace {

// Union used to make the int/float aliasing explicit so we can access the raw
// bits.
union Float32Bits {
  uint32_t u;
  float f;
};

const uint32_t kF32MantiBits = 23;
const uint32_t kF32HalfMantiBitDiff = 13;
const uint32_t kF32HalfBitDiff = 16;
const Float32Bits kF32Magic = {113 << kF32MantiBits};
const uint32_t kF32HalfExpAdjust = (127 - 15) << kF32MantiBits;

// Constructs the 16 bit representation for a half precision value from a float
// value. This implementation is adapted from Eigen.
uint16_t float2half(float floatValue) {
  const Float32Bits inf = {255 << kF32MantiBits};
  const Float32Bits f16max = {(127 + 16) << kF32MantiBits};
  const Float32Bits denormMagic = {((127 - 15) + (kF32MantiBits - 10) + 1)
                                   << kF32MantiBits};
  uint32_t signMask = 0x80000000u;
  uint16_t halfValue = static_cast<uint16_t>(0x0u);
  Float32Bits f;
  f.f = floatValue;
  uint32_t sign = f.u & signMask;
  f.u ^= sign;

  if (f.u >= f16max.u) {
    const uint32_t halfQnan = 0x7e00;
    const uint32_t halfInf = 0x7c00;
    // Inf or NaN (all exponent bits set).
    halfValue = (f.u > inf.u) ? halfQnan : halfInf; // NaN->qNaN and Inf->Inf
  } else {
    // (De)normalized number or zero.
    if (f.u < kF32Magic.u) {
      // The resulting FP16 is subnormal or zero.
      //
      // Use a magic value to align our 10 mantissa bits at the bottom of the
      // float. As long as FP addition is round-to-nearest-even this works.
      f.f += denormMagic.f;

      halfValue = static_cast<uint16_t>(f.u - denormMagic.u);
    } else {
      uint32_t mantOdd =
          (f.u >> kF32HalfMantiBitDiff) & 1; // Resulting mantissa is odd.

      // Update exponent, rounding bias part 1. The following expressions are
      // equivalent to `f.u += ((unsigned int)(15 - 127) << kF32MantiBits) +
      // 0xfff`, but without arithmetic overflow.
      f.u += 0xc8000fffU;
      // Rounding bias part 2.
      f.u += mantOdd;
      halfValue = static_cast<uint16_t>(f.u >> kF32HalfMantiBitDiff);
    }
  }

  halfValue |= static_cast<uint16_t>(sign >> kF32HalfBitDiff);
  return halfValue;
}

// Converts the 16 bit representation of a half precision value to a float
// value. This implementation is adapted from Eigen.
float half2float(uint16_t halfValue) {
  const uint32_t shiftedExp =
      0x7c00 << kF32HalfMantiBitDiff; // Exponent mask after shift.

  // Initialize the float representation with the exponent/mantissa bits.
  Float32Bits f = {
      static_cast<uint32_t>((halfValue & 0x7fff) << kF32HalfMantiBitDiff)};
  const uint32_t exp = shiftedExp & f.u;
  f.u += kF32HalfExpAdjust; // Adjust the exponent

  // Handle exponent special cases.
  if (exp == shiftedExp) {
    // Inf/NaN
    f.u += kF32HalfExpAdjust;
  } else if (exp == 0) {
    // Zero/Denormal?
    f.u += 1 << kF32MantiBits;
    f.f -= kF32Magic.f;
  }

  f.u |= (halfValue & 0x8000) << kF32HalfBitDiff; // Sign bit.
  return f.f;
}

const uint32_t kF32BfMantiBitDiff = 16;

// Constructs the 16 bit representation for a bfloat value from a float value.
// This implementation is adapted from Eigen.
uint16_t float2bfloat(float floatValue) {
  if (std::isnan(floatValue))
    return std::signbit(floatValue) ? 0xFFC0 : 0x7FC0;

  Float32Bits floatBits;
  floatBits.f = floatValue;
  uint16_t bfloatBits;

  // Least significant bit of resulting bfloat.
  uint32_t lsb = (floatBits.u >> kF32BfMantiBitDiff) & 1;
  uint32_t roundingBias = 0x7fff + lsb;
  floatBits.u += roundingBias;
  bfloatBits = static_cast<uint16_t>(floatBits.u >> kF32BfMantiBitDiff);
  return bfloatBits;
}

// Converts the 16 bit representation of a bfloat value to a float value. This
// implementation is adapted from Eigen.
float bfloat2float(uint16_t bfloatBits) {
  Float32Bits floatBits;
  floatBits.u = static_cast<uint32_t>(bfloatBits) << kF32BfMantiBitDiff;
  return floatBits.f;
}

} // namespace

f16::f16(float f) : bits(float2half(f)) {}

bf16::bf16(float f) : bits(float2bfloat(f)) {}

std::ostream &operator<<(std::ostream &os, const f16 &f) {
  os << half2float(f.bits);
  return os;
}

std::ostream &operator<<(std::ostream &os, const bf16 &d) {
  os << bfloat2float(d.bits);
  return os;
}

// Mark these symbols as weak so they don't conflict when compiler-rt also
// defines them.
#define ATTR_WEAK
#ifdef __has_attribute
#if __has_attribute(weak) && !defined(__MINGW32__) && !defined(__CYGWIN__) &&  \
    !defined(_WIN32)
#undef ATTR_WEAK
#define ATTR_WEAK __attribute__((__weak__))
#endif
#endif

#if defined(__x86_64__)
// On x86 bfloat16 is passed in SSE registers. Since both float and __bf16
// are passed in the same register we can use the wider type and careful casting
// to conform to x86_64 psABI. This only works with the assumption that we're
// dealing with little-endian values passed in wider registers.
// Ideally this would directly use __bf16, but that type isn't supported by all
// compilers.
using BF16ABIType = float;
#else
// Default to uint16_t if we have nothing else.
using BF16ABIType = uint16_t;
#endif

// Provide a float->bfloat conversion routine in case the runtime doesn't have
// one.
extern "C" BF16ABIType ATTR_WEAK __truncsfbf2(float f) {
  uint16_t bf = float2bfloat(f);
  // The output can be a float type, bitcast it from uint16_t.
  BF16ABIType ret = 0;
  std::memcpy(&ret, &bf, sizeof(bf));
  return ret;
}

// Provide a double->bfloat conversion routine in case the runtime doesn't have
// one.
extern "C" BF16ABIType ATTR_WEAK __truncdfbf2(double d) {
  // This does a double rounding step, but it's precise enough for our use
  // cases.
  return __truncsfbf2(static_cast<float>(d));
}

// Provide these to the CRunner with the local float16 knowledge.
extern "C" void printF16(uint16_t bits) {
  f16 f;
  std::memcpy(&f, &bits, sizeof(f16));
  std::cout << f;
}
extern "C" void printBF16(uint16_t bits) {
  bf16 f;
  std::memcpy(&f, &bits, sizeof(bf16));
  std::cout << f;
}

#endif // MLIR_FLOAT16_DEFINE_FUNCTIONS
