//===- NarrowFloatExpansion.cpp - Narrow float truncf/extf ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/NarrowFloatExpansion.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include <climits>

using namespace mlir;

namespace {
struct BitsBuilder {
  BitsBuilder(ImplicitLocOpBuilder &b, Type carrier)
      : b(b), carrierTy(carrier),
        width(getElementTypeOrSelf(carrier).getIntOrFloatBitWidth()) {}

  Value cst(Type ty, int64_t v) {
    if (auto shapedTy = dyn_cast<ShapedType>(ty)) {
      Attribute elem = b.getIntegerAttr(shapedTy.getElementType(), v);
      return arith::ConstantOp::create(
          b, ty, DenseElementsAttr::get(shapedTy, ArrayRef<Attribute>(elem)));
    }
    return arith::ConstantOp::create(b, ty, b.getIntegerAttr(ty, v));
  }
  Value carrierConst(int64_t v) { return cst(carrierTy, v); }

  Value andi(Value a, Value c) { return arith::AndIOp::create(b, a, c); }
  Value ori(Value a, Value c) { return arith::OrIOp::create(b, a, c); }
  Value shli(Value v, Value a) { return arith::ShLIOp::create(b, v, a); }
  Value shli(Value v, int amt) { return shli(v, cst(v.getType(), amt)); }
  Value lshri(Value v, Value a) { return arith::ShRUIOp::create(b, v, a); }
  Value lshri(Value v, int amt) { return lshri(v, cst(v.getType(), amt)); }
  Value addi(Value a, Value c) { return arith::AddIOp::create(b, a, c); }
  Value subi(Value a, Value c) { return arith::SubIOp::create(b, a, c); }
  Value muli(Value a, Value c) { return arith::MulIOp::create(b, a, c); }
  Value minui(Value a, Value c) { return arith::MinUIOp::create(b, a, c); }
  Value maxui(Value a, Value c) { return arith::MaxUIOp::create(b, a, c); }
  Value maxsi(Value a, Value c) { return arith::MaxSIOp::create(b, a, c); }

  Value icmp(arith::CmpIPredicate p, Value a, Value c) {
    return arith::CmpIOp::create(b, p, a, c);
  }
  Value eq(Value a, Value c) { return icmp(arith::CmpIPredicate::eq, a, c); }
  Value ult(Value a, Value c) { return icmp(arith::CmpIPredicate::ult, a, c); }
  Value ule(Value a, Value c) { return icmp(arith::CmpIPredicate::ule, a, c); }
  Value ugt(Value a, Value c) { return icmp(arith::CmpIPredicate::ugt, a, c); }
  Value uge(Value a, Value c) { return icmp(arith::CmpIPredicate::uge, a, c); }
  Value slt(Value a, Value c) { return icmp(arith::CmpIPredicate::slt, a, c); }
  Value sle(Value a, Value c) { return icmp(arith::CmpIPredicate::sle, a, c); }
  Value sge(Value a, Value c) { return icmp(arith::CmpIPredicate::sge, a, c); }
  Value select(Value cond, Value t, Value f) {
    return arith::SelectOp::create(b, cond, t, f);
  }

  ImplicitLocOpBuilder &b;
  Type carrierTy;
  unsigned width;
};
} // namespace

static uint64_t lowBitsMask(unsigned n) {
  assert(n < 64 && "mask wider than the widest carrier");
  return (uint64_t(1) << n) - 1;
}

static uint64_t bitAt(unsigned n) {
  assert(n < 64 && "bit outside the widest carrier");
  return uint64_t(1) << n;
}

static Type checkCarrierType(Value srcBits, const arith::FloatFormatSpec &src,
                             const arith::FloatFormatSpec &dst) {
  Type carrier = srcBits.getType();
  assert((!isa<ShapedType>(carrier) ||
          cast<ShapedType>(carrier).hasStaticShape()) &&
         "a shaped carrier must have a static shape");
  [[maybe_unused]] unsigned width =
      cast<IntegerType>(getElementTypeOrSelf(carrier)).getWidth();
  assert((width == 32 || width == 64) && "the carrier must be i32 or i64");
  assert(width >= src.sizeInBits && width >= dst.sizeInBits &&
         "the carrier must hold both formats");
  return carrier;
}

arith::FloatFormatSpec
mlir::arith::floatFormatSpecFromSemantics(const llvm::fltSemantics &sem) {
  assert(sem.hasZero && sem.hasDenormals &&
         "the format needs a zero and subnormals (f8E8M0FNU has neither)");
  assert(!sem.hasExplicitIntegerBit && "the integer bit must be implicit");
  FloatFormatSpec spec;
  spec.sizeInBits = llvm::APFloat::semanticsSizeInBits(sem);
  assert(spec.sizeInBits <= 64 && "the encoding must fit in 64 bits");
  spec.mantBits = llvm::APFloat::semanticsPrecision(sem) - 1;
  spec.hasInf = llvm::APFloat::semanticsHasInf(sem);
  spec.isSigned = llvm::APFloat::semanticsHasSignedRepr(sem);
  if (spec.isSigned) {
    spec.signBitPos = spec.sizeInBits - 1;
    spec.expBits = spec.sizeInBits - 1 - spec.mantBits;
  } else {
    spec.signBitPos = UINT_MAX;
    spec.expBits = spec.sizeInBits - spec.mantBits;
  }
  assert(spec.expBits < 32 && "the exponent field must fit in 32 bits");
  spec.maxStoredExp = (1u << spec.expBits) - 1u;
  spec.bias = 1 - llvm::APFloat::semanticsMinExponent(sem);
  bool nanOnly = sem.nonFiniteBehavior == llvm::fltNonfiniteBehavior::NanOnly;
  spec.hasAllOnesNaN =
      nanOnly && sem.nanEncoding == llvm::fltNanEncoding::AllOnes;
  spec.isFNUZ =
      nanOnly && sem.nanEncoding == llvm::fltNanEncoding::NegativeZero;
  spec.maxFiniteBitsAbs =
      llvm::APFloat::getLargest(sem).bitcastToAPInt().getZExtValue();
  // getNaN asserts on a format without a NaN.
  if (sem.nonFiniteBehavior == llvm::fltNonfiniteBehavior::FiniteOnly)
    spec.quietNaNBits = spec.maxFiniteBitsAbs;
  else
    spec.quietNaNBits =
        llvm::APFloat::getNaN(sem).bitcastToAPInt().getZExtValue();
  return spec;
}

arith::FloatFormatSpec
mlir::arith::floatFormatSpecFromFloatType(FloatType floatTy) {
  return floatFormatSpecFromSemantics(floatTy.getFloatSemantics());
}

// The truncation rounds once. With sm / dm the mantissa bits, minNormalExp =
// srcBias - dstBias + 1 and zeroExp = minNormalExp - dm - 2, a source with
// exponent field e and significand sig (implicit bit included) gives
//   code = round(sig * 2^t / 2^(sm + 2)) + ((e - e1) << dm),
//   e1 = min(e, minNormalExp), t = max(e1, zeroExp) - zeroExp,
// with round() in `rnd`, for normal and subnormal results alike (a mantissa
// carry bumps the exponent field). Codes above max-finite are resolved later.
// E.g. f32 0x3F880008 to f8E4M3FN gives 0x39 (1.125), not 1.0 as via f16.

Value mlir::arith::emitNarrowFloatTrunc(ImplicitLocOpBuilder &b, Value srcBits,
                                        const FloatFormatSpec &src,
                                        const FloatFormatSpec &dst,
                                        RoundingMode rnd) {
  assert(src.hasInf && src.isSigned && src.sizeInBits >= 16 &&
         "the source must be an IEEE f16, bf16, f32 or f64");
  assert(dst.bias <= src.bias &&
         "a source subnormal must map to a destination subnormal or zero");
  assert(int(src.maxStoredExp) - src.bias >=
             int(dst.maxStoredExp) - dst.bias + (dst.hasInf ? 0 : 1) &&
         "a source Inf must lie above the destination's finite range");
  BitsBuilder bb(b, checkCarrierType(srcBits, src, dst));

  const int srcMantBits = src.mantBits, dstMantBits = dst.mantBits;
  const int minNormalExp = src.bias - dst.bias + 1;
  const int roundShift = srcMantBits + 2;
  const int zeroExp = minNormalExp - dstMantBits - 2;
  assert(srcMantBits + dstMantBits + 4 <= int(bb.width) &&
         "the rounding add must not carry out of the carrier");
  const uint64_t infBits = uint64_t(src.maxStoredExp) << srcMantBits;
  const uint64_t srcSignBit = bitAt(src.signBitPos);
  const uint64_t maxFinite = dst.maxFiniteBitsAbs;
  const uint64_t dstInf = uint64_t(dst.maxStoredExp) << dstMantBits;
  // The carrier bits above the source are ignored, so sign tests mask them.
  const bool srcFillsCarrier = src.sizeInBits == bb.width;
  const bool directed =
      rnd == RoundingMode::upward || rnd == RoundingMode::downward;

  Value zero, one, maxFiniteV, maskedSrc;
  auto getZero = [&]() {
    if (!zero)
      zero = bb.carrierConst(0);
    return zero;
  };
  auto getOne = [&]() {
    if (!one)
      one = bb.carrierConst(1);
    return one;
  };
  auto getMaxFinite = [&]() {
    if (!maxFiniteV)
      maxFiniteV = bb.carrierConst(maxFinite);
    return maxFiniteV;
  };
  auto srcInFormat = [&]() {
    if (srcFillsCarrier)
      return srcBits;
    if (!maskedSrc)
      maskedSrc =
          bb.andi(srcBits, bb.carrierConst(lowBitsMask(src.sizeInBits)));
    return maskedSrc;
  };

  // A directed mode rounds the magnitude up for one sign, also into an
  // unsigned `dst` (APFloat rounds x, then drops the sign).
  Value magUp;
  if (directed) {
    if (srcFillsCarrier) {
      magUp = rnd == RoundingMode::upward ? bb.sge(srcBits, getZero())
                                          : bb.slt(srcBits, getZero());
    } else {
      Value signBitV = bb.carrierConst(srcSignBit);
      magUp = rnd == RoundingMode::upward ? bb.ult(srcInFormat(), signBitV)
                                          : bb.uge(srcInFormat(), signBitV);
    }
  }

  // Returns v >> shift rounded in `rnd`, the mode being an addend to v.
  auto roundAt = [&](Value v, int shift) {
    const uint64_t half = bitAt(shift - 1);
    Value rounded;
    switch (rnd) {
    case RoundingMode::to_nearest_even: {
      // half - 1 plus the kept LSB: a tie rounds to even.
      Value kept = bb.lshri(v, shift);
      Value lsb = bb.andi(kept, getOne());
      Value biased = bb.addi(v, bb.carrierConst(half - 1));
      rounded = bb.lshri(bb.addi(biased, lsb), shift);
      break;
    }
    case RoundingMode::to_nearest_away:
      rounded = bb.lshri(bb.addi(v, bb.carrierConst(half)), shift);
      break;
    case RoundingMode::toward_zero:
      rounded = bb.lshri(v, shift);
      break;
    case RoundingMode::upward:
    case RoundingMode::downward: {
      Value allOnes = bb.carrierConst(bitAt(shift) - 1);
      Value addend = bb.select(magUp, allOnes, getZero());
      rounded = bb.lshri(bb.addi(v, addend), shift);
      break;
    }
    }
    return rounded;
  };

  Value mag =
      bb.andi(srcBits, bb.carrierConst(lowBitsMask(src.sizeInBits - 1)));
  Value exp = bb.lshri(mag, srcMantBits);

  // The implicit bit must be exact only if a subnormal can round to nonzero.
  const bool dynImplicit = directed || zeroExp <= 0;
  Value implicit = dynImplicit ? bb.shli(bb.minui(exp, getOne()), srcMantBits)
                               : bb.carrierConst(bitAt(srcMantBits));
  Value mantissa = bb.andi(mag, bb.carrierConst(lowBitsMask(srcMantBits)));
  Value sig = bb.ori(mantissa, implicit);

  Value hi, shift;
  if (zeroExp <= 0) {
    // A subnormal has exponent 1 and no implicit bit; e1 >= 1 > zeroExp.
    Value ee = bb.maxui(exp, getOne());
    Value e1 = bb.minui(ee, bb.carrierConst(minNormalExp));
    hi = bb.subi(ee, e1);
    shift = zeroExp == 0 ? e1 : bb.subi(e1, bb.carrierConst(zeroExp));
  } else {
    Value e1 = bb.minui(exp, bb.carrierConst(minNormalExp));
    hi = bb.subi(exp, e1);
    // smax equals umax here; LLVM miscompiled umax(x, c) - c on zero vectors.
    Value clampedExp = bb.maxsi(e1, bb.carrierConst(zeroExp));
    shift = bb.subi(clampedExp, bb.carrierConst(zeroExp));
  }
  Value v = bb.shli(sig, shift);
  Value rounded = roundAt(v, roundShift);
  Value code = bb.addi(rounded, bb.shli(hi, dstMantBits));

  // Resolve codes above maxFinite into `clamped` and, if set, `isNaNResult`.
  Value clamped, isNaNResult;
  if (!dst.hasNaN()) {
    clamped = bb.minui(code, getMaxFinite());
  } else if (dst.hasInf) {
    Value infBitsV = bb.carrierConst(infBits);
    Value dstInfV = bb.carrierConst(dstInf);
    switch (rnd) {
    case RoundingMode::to_nearest_even:
    case RoundingMode::to_nearest_away:
      // A source Inf lands on the Inf code too.
      clamped = bb.minui(code, dstInfV);
      break;
    case RoundingMode::toward_zero:
      // A finite value stays finite; a source Inf does not.
      clamped = bb.minui(code, getMaxFinite());
      clamped = bb.select(bb.eq(mag, infBitsV), dstInfV, clamped);
      break;
    case RoundingMode::upward:
    case RoundingMode::downward: {
      Value limit = bb.select(magUp, dstInfV, getMaxFinite());
      clamped = bb.minui(code, limit);
      clamped = bb.select(bb.eq(mag, infBitsV), dstInfV, clamped);
      break;
    }
    }
    isNaNResult = bb.ugt(mag, infBitsV);
  } else if (rnd == RoundingMode::to_nearest_even ||
             rnd == RoundingMode::to_nearest_away) {
    // NaN-only: an overflow, a source Inf and a NaN all give the NaN.
    clamped = code;
    isNaNResult = bb.ugt(code, getMaxFinite());
  } else if (rnd == RoundingMode::toward_zero ||
             (rnd == RoundingMode::downward && !dst.isSigned)) {
    // Downward, only an x < 0 overflows to the NaN, which an unsigned `dst`
    // lacks (APFloat asserts): it saturates instead.
    clamped = bb.minui(code, getMaxFinite());
    isNaNResult = bb.uge(mag, bb.carrierConst(infBits));
  } else {
    clamped = bb.minui(code, getMaxFinite());
    Value overflowsUp = bb.andi(bb.ugt(code, getMaxFinite()), magUp);
    isNaNResult = bb.ori(overflowsUp, bb.uge(mag, bb.carrierConst(infBits)));
  }

  Value dstSignBit, srcSign;
  if (dst.isSigned) {
    dstSignBit = bb.carrierConst(bitAt(dst.signBitPos));
    if (dst.hasNaN()) {
      const int signShift = int(src.sizeInBits) - int(dst.sizeInBits);
      srcSign = bb.andi(bb.lshri(srcBits, signShift), dstSignBit);
    }
  }

  // The NaN keeps the source sign where `dst` has a NaN of each sign and, in an
  // IEEE-encoded `dst`, the payload bits that fit.
  Value nanOut;
  if (dst.hasNaN()) {
    nanOut = bb.carrierConst(dst.quietNaNBits);
    if (dst.isSigned && !dst.isFNUZ)
      nanOut = bb.ori(nanOut, srcSign);
    if (dst.hasInf) {
      const int payloadShift = int(src.mantBits) - int(dst.mantBits);
      Value payload = bb.lshri(srcBits, payloadShift);
      Value nanPayload =
          bb.andi(payload, bb.carrierConst(lowBitsMask(dst.mantBits)));
      nanOut = bb.ori(nanOut, nanPayload);
    }
  }
  if (!dst.isSigned)
    return isNaNResult ? bb.select(isNaNResult, nanOut, clamped) : clamped;

  Value sign = srcSign;
  if (!dst.hasNaN()) {
    // A NaN gives +max-finite here: only a negative non-NaN sets the sign.
    Value isNegative;
    if (srcFillsCarrier) {
      isNegative = bb.sle(srcBits, bb.carrierConst(srcSignBit | infBits));
    } else {
      Value srcValue = srcInFormat();
      Value offset = bb.subi(srcValue, bb.carrierConst(srcSignBit));
      isNegative = bb.ule(offset, bb.carrierConst(infBits));
    }
    sign = bb.select(isNegative, dstSignBit, getZero());
  }
  Value res = bb.ori(clamped, sign);
  // An FNUZ format's -0 encoding is its NaN: a finite result there is +0.
  if (dst.isFNUZ) {
    Value isNegZero = bb.eq(res, dstSignBit);
    res = bb.select(isNegZero, getZero(), res);
  }
  return isNaNResult ? bb.select(isNaNResult, nanOut, res) : res;
}

// The extension is exact and linear on each source binade: the magnitude code
// m maps to m * mult + add. With ms / md the mantissa bits and bs / bd the
// biases, a normal shifts by md - ms and adds (bd - bs) << md, a subnormal
// shifts its leading one onto the implicit bit, a binade among the destination
// subnormals loses its exponent field, and one above the range or Inf gives
// Inf. Neighboring pieces that agree on every non-NaN code are merged.
// E.g. f8E4M3FN to f16 maps normal m to (m << 7) + 0x2000: 0x39 to 0x3C80.

namespace {
/// One linear piece of the map: for m in [lo, hi], r = m * mult + add,
/// wrapping in the carrier. A constant piece (mult == 0) is the Inf.
struct ExtPiece {
  uint64_t lo, hi;
  uint64_t mult;
  int64_t add;
};
} // namespace

/// Whether the map must produce code `code`'s value; a NaN's is replaced.
static bool extCodeIsLive(const arith::FloatFormatSpec &src, uint64_t code) {
  uint64_t e = code >> src.mantBits, m = code & lowBitsMask(src.mantBits);
  if (src.hasInf)
    return !(e == src.maxStoredExp && m != 0);
  if (src.hasAllOnesNaN)
    return code != lowBitsMask(src.expBits + src.mantBits);
  return true;
}

/// Returns the destination encoding of 2^k, a subnormal below its range.
static uint64_t encodePow2(const arith::FloatFormatSpec &dst, int k) {
  int biased = k + dst.bias;
  assert(biased <= int(dst.maxStoredExp) - 1 && "2^k above the dst range");
  if (biased >= 1)
    return uint64_t(biased) << dst.mantBits;
  int sh = int(dst.mantBits) - 1 + biased;
  assert(sh >= 0 && "2^k below the dst subnormal step: the ext is inexact");
  return uint64_t(1) << sh;
}

/// Returns the merged pieces of the map from `src` to `dst` (`width`-bit).
static SmallVector<ExtPiece, 6>
buildExtPieceTable(const arith::FloatFormatSpec &src,
                   const arith::FloatFormatSpec &dst, unsigned width) {
  const int ms = src.mantBits, md = dst.mantBits;
  const int bs = src.bias, bd = dst.bias;
  const int ns = md - ms;
  const int k = 1 - bs - ms;
  const int64_t dstInf = int64_t(dst.maxStoredExp) << md;
  SmallVector<ExtPiece, 40> pieces;
  pieces.push_back({0, 1, encodePow2(dst, k), 0});
  for (int p = 1; p < ms; ++p) {
    const uint64_t lo = uint64_t(1) << p, hi = (uint64_t(2) << p) - 1;
    int dstExp = p + k + bd;
    if (dstExp >= 1) {
      pieces.push_back(
          {lo, hi, uint64_t(1) << (md - p), int64_t(dstExp - 1) << md});
      continue;
    }
    int sh = ns + bd - bs;
    assert(sh >= 0 && "a source subnormal below the dst subnormal step");
    pieces.push_back({lo, hi, uint64_t(1) << sh, 0});
  }
  for (unsigned e = 1; e <= src.maxStoredExp; ++e) {
    const uint64_t lo = uint64_t(e) << ms, hi = (uint64_t(e + 1) << ms) - 1;
    if (src.hasInf && e == src.maxStoredExp) {
      pieces.push_back({lo, hi, 0, dstInf});
      continue;
    }
    int dstExp = int(e) - bs + bd;
    if (dstExp > int(dst.maxStoredExp) - 1) {
      assert(dst.hasInf && "an ext that can overflow needs a dst with an Inf");
      pieces.push_back({lo, hi, 0, dstInf});
    } else if (dstExp >= 1) {
      // (bd - bs) << md, as a multiply: bd - bs may be negative.
      pieces.push_back(
          {lo, hi, uint64_t(1) << ns, int64_t(bd - bs) * (int64_t(1) << md)});
    } else {
      int sh = ns + dstExp - 1;
      assert(sh >= 0 && "a source normal below the dst subnormal step");
      pieces.push_back(
          {lo, hi, uint64_t(1) << sh, -(int64_t(e - 1) << (md + dstExp - 1))});
    }
  }
  const uint64_t widthMask = width == 64 ? ~uint64_t(0) : lowBitsMask(width);
  auto eval = [widthMask](const ExtPiece &q, uint64_t code) {
    return (q.mult * code + uint64_t(q.add)) & widthMask;
  };
  SmallVector<ExtPiece, 6> table;
  for (const ExtPiece &q : pieces) {
    if (!table.empty()) {
      ExtPiece &prev = table.back();
      bool agrees = true;
      for (uint64_t code = q.lo; code <= q.hi && agrees; ++code)
        if (extCodeIsLive(src, code))
          agrees = eval(prev, code) == eval(q, code);
      if (agrees) {
        prev.hi = q.hi;
        continue;
      }
    }
    table.push_back(q);
  }
  return table;
}

Value mlir::arith::emitNarrowFloatExt(ImplicitLocOpBuilder &b, Value srcBits,
                                      const FloatFormatSpec &src,
                                      const FloatFormatSpec &dst) {
  assert(dst.hasInf && dst.isSigned && dst.sizeInBits >= 16 &&
         "the destination must be an IEEE f16, bf16, f32 or f64");
  assert(dst.mantBits >= src.mantBits &&
         "the destination needs at least the source's mantissa bits");
  BitsBuilder bb(b, checkCarrierType(srcBits, src, dst));
  const int ms = src.mantBits, md = dst.mantBits;
  const unsigned magBits = src.expBits + ms;

  Value mag = bb.andi(srcBits, bb.carrierConst(lowBitsMask(magBits)));
  SmallVector<ExtPiece, 6> table = buildExtPieceTable(src, dst, bb.width);
  auto linear = [&](const ExtPiece &q) -> Value {
    assert(q.mult != 0 && "a constant piece has no linear form");
    Value t = llvm::isPowerOf2_64(q.mult)
                  ? bb.shli(mag, int(llvm::Log2_64(q.mult)))
                  : bb.muli(mag, bb.carrierConst(int64_t(q.mult)));
    return q.add ? bb.addi(t, bb.carrierConst(q.add)) : t;
  };
  Value fin = linear(table.front());
  for (const ExtPiece &q : llvm::drop_begin(table)) {
    Value inPiece = bb.uge(mag, bb.carrierConst(q.lo));
    Value pieceBits = q.mult ? linear(q) : bb.carrierConst(q.add);
    fin = bb.select(inPiece, pieceBits, fin);
  }

  Value isNaN;
  if (src.hasInf) {
    isNaN = bb.ugt(mag, bb.carrierConst(uint64_t(src.maxStoredExp) << ms));
  } else if (src.hasAllOnesNaN) {
    isNaN = bb.eq(mag, bb.carrierConst(lowBitsMask(magBits)));
  } else if (src.isFNUZ) {
    Value srcCode =
        bb.andi(srcBits, bb.carrierConst(lowBitsMask(src.sizeInBits)));
    isNaN = bb.eq(srcCode, bb.carrierConst(bitAt(src.signBitPos)));
  }

  Value res = fin, sign;
  if (src.isSigned) {
    const int signShift = int(dst.signBitPos) - int(src.signBitPos);
    Value signBit = bb.andi(srcBits, bb.carrierConst(bitAt(src.signBitPos)));
    sign = bb.shli(signBit, signShift);
    res = bb.ori(res, sign);
  }

  if (isNaN) {
    Value nanRes = bb.carrierConst(dst.quietNaNBits);
    if (!src.isFNUZ) {
      Value payload = bb.andi(srcBits, bb.carrierConst(lowBitsMask(ms)));
      nanRes = bb.ori(nanRes, bb.shli(payload, md - ms));
      if (sign)
        nanRes = bb.ori(nanRes, sign);
    }
    res = bb.select(isNaN, nanRes, res);
  }
  return res;
}
