//===- SlowMPInt.cpp - MLIR SlowMPInt Class -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/SlowMPInt.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace presburger;
using namespace detail;

SlowMPInt::SlowMPInt(int64_t val) : val(64, val, /*isSigned=*/true) {}
SlowMPInt::SlowMPInt() : SlowMPInt(0) {}
SlowMPInt::SlowMPInt(const llvm::APInt &val) : val(val) {}
SlowMPInt &SlowMPInt::operator=(int64_t val) { return *this = SlowMPInt(val); }
SlowMPInt::operator int64_t() const { return val.getSExtValue(); }

llvm::hash_code detail::hash_value(const SlowMPInt &x) {
  return hash_value(x.val);
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
void SlowMPInt::print(llvm::raw_ostream &os) const { os << val; }

void SlowMPInt::dump() const { print(llvm::errs()); }

llvm::raw_ostream &detail::operator<<(llvm::raw_ostream &os,
                                      const SlowMPInt &x) {
  x.print(os);
  return os;
}

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
SlowMPInt &detail::operator+=(SlowMPInt &a, int64_t b) {
  return a += SlowMPInt(b);
}
SlowMPInt &detail::operator-=(SlowMPInt &a, int64_t b) {
  return a -= SlowMPInt(b);
}
SlowMPInt &detail::operator*=(SlowMPInt &a, int64_t b) {
  return a *= SlowMPInt(b);
}
SlowMPInt &detail::operator/=(SlowMPInt &a, int64_t b) {
  return a /= SlowMPInt(b);
}
SlowMPInt &detail::operator%=(SlowMPInt &a, int64_t b) {
  return a %= SlowMPInt(b);
}

bool detail::operator==(const SlowMPInt &a, int64_t b) {
  return a == SlowMPInt(b);
}
bool detail::operator!=(const SlowMPInt &a, int64_t b) {
  return a != SlowMPInt(b);
}
bool detail::operator>(const SlowMPInt &a, int64_t b) {
  return a > SlowMPInt(b);
}
bool detail::operator<(const SlowMPInt &a, int64_t b) {
  return a < SlowMPInt(b);
}
bool detail::operator<=(const SlowMPInt &a, int64_t b) {
  return a <= SlowMPInt(b);
}
bool detail::operator>=(const SlowMPInt &a, int64_t b) {
  return a >= SlowMPInt(b);
}
SlowMPInt detail::operator+(const SlowMPInt &a, int64_t b) {
  return a + SlowMPInt(b);
}
SlowMPInt detail::operator-(const SlowMPInt &a, int64_t b) {
  return a - SlowMPInt(b);
}
SlowMPInt detail::operator*(const SlowMPInt &a, int64_t b) {
  return a * SlowMPInt(b);
}
SlowMPInt detail::operator/(const SlowMPInt &a, int64_t b) {
  return a / SlowMPInt(b);
}
SlowMPInt detail::operator%(const SlowMPInt &a, int64_t b) {
  return a % SlowMPInt(b);
}

bool detail::operator==(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) == b;
}
bool detail::operator!=(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) != b;
}
bool detail::operator>(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) > b;
}
bool detail::operator<(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) < b;
}
bool detail::operator<=(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) <= b;
}
bool detail::operator>=(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) >= b;
}
SlowMPInt detail::operator+(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) + b;
}
SlowMPInt detail::operator-(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) - b;
}
SlowMPInt detail::operator*(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) * b;
}
SlowMPInt detail::operator/(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) / b;
}
SlowMPInt detail::operator%(int64_t a, const SlowMPInt &b) {
  return SlowMPInt(a) % b;
}

static unsigned getMaxWidth(const APInt &a, const APInt &b) {
  return std::max(a.getBitWidth(), b.getBitWidth());
}

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------

// TODO: consider instead making APInt::compare available and using that.
bool SlowMPInt::operator==(const SlowMPInt &o) const {
  unsigned width = getMaxWidth(val, o.val);
  return val.sext(width) == o.val.sext(width);
}
bool SlowMPInt::operator!=(const SlowMPInt &o) const {
  unsigned width = getMaxWidth(val, o.val);
  return val.sext(width) != o.val.sext(width);
}
bool SlowMPInt::operator>(const SlowMPInt &o) const {
  unsigned width = getMaxWidth(val, o.val);
  return val.sext(width).sgt(o.val.sext(width));
}
bool SlowMPInt::operator<(const SlowMPInt &o) const {
  unsigned width = getMaxWidth(val, o.val);
  return val.sext(width).slt(o.val.sext(width));
}
bool SlowMPInt::operator<=(const SlowMPInt &o) const {
  unsigned width = getMaxWidth(val, o.val);
  return val.sext(width).sle(o.val.sext(width));
}
bool SlowMPInt::operator>=(const SlowMPInt &o) const {
  unsigned width = getMaxWidth(val, o.val);
  return val.sext(width).sge(o.val.sext(width));
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------

/// Bring a and b to have the same width and then call op(a, b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call op(a, b, overflow), returning its result. The operation with double
/// widths should not also overflow.
APInt runOpWithExpandOnOverflow(
    const APInt &a, const APInt &b,
    llvm::function_ref<APInt(const APInt &, const APInt &, bool &overflow)>
        op) {
  bool overflow;
  unsigned width = getMaxWidth(a, b);
  APInt ret = op(a.sext(width), b.sext(width), overflow);
  if (!overflow)
    return ret;

  width *= 2;
  ret = op(a.sext(width), b.sext(width), overflow);
  assert(!overflow && "double width should be sufficient to avoid overflow!");
  return ret;
}

SlowMPInt SlowMPInt::operator+(const SlowMPInt &o) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(val, o.val, std::mem_fn(&APInt::sadd_ov)));
}
SlowMPInt SlowMPInt::operator-(const SlowMPInt &o) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(val, o.val, std::mem_fn(&APInt::ssub_ov)));
}
SlowMPInt SlowMPInt::operator*(const SlowMPInt &o) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(val, o.val, std::mem_fn(&APInt::smul_ov)));
}
SlowMPInt SlowMPInt::operator/(const SlowMPInt &o) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(val, o.val, std::mem_fn(&APInt::sdiv_ov)));
}
SlowMPInt detail::abs(const SlowMPInt &x) { return x >= 0 ? x : -x; }
SlowMPInt detail::ceilDiv(const SlowMPInt &lhs, const SlowMPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return SlowMPInt(
      llvm::APIntOps::RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::UP));
}
SlowMPInt detail::floorDiv(const SlowMPInt &lhs, const SlowMPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return SlowMPInt(
      llvm::APIntOps::RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::DOWN));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
SlowMPInt detail::mod(const SlowMPInt &lhs, const SlowMPInt &rhs) {
  assert(rhs >= 1 && "mod is only supported for positive divisors!");
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

SlowMPInt detail::gcd(const SlowMPInt &a, const SlowMPInt &b) {
  assert(a >= 0 && b >= 0 && "operands must be non-negative!");
  return SlowMPInt(llvm::APIntOps::GreatestCommonDivisor(a.val, b.val));
}

/// Returns the least common multiple of 'a' and 'b'.
SlowMPInt detail::lcm(const SlowMPInt &a, const SlowMPInt &b) {
  SlowMPInt x = abs(a);
  SlowMPInt y = abs(b);
  return (x * y) / gcd(x, y);
}

/// This operation cannot overflow.
SlowMPInt SlowMPInt::operator%(const SlowMPInt &o) const {
  unsigned width = std::max(val.getBitWidth(), o.val.getBitWidth());
  return SlowMPInt(val.sext(width).srem(o.val.sext(width)));
}

SlowMPInt SlowMPInt::operator-() const {
  if (val.isMinSignedValue()) {
    /// Overflow only occurs when the value is the minimum possible value.
    APInt ret = val.sext(2 * val.getBitWidth());
    return SlowMPInt(-ret);
  }
  return SlowMPInt(-val);
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
SlowMPInt &SlowMPInt::operator+=(const SlowMPInt &o) {
  *this = *this + o;
  return *this;
}
SlowMPInt &SlowMPInt::operator-=(const SlowMPInt &o) {
  *this = *this - o;
  return *this;
}
SlowMPInt &SlowMPInt::operator*=(const SlowMPInt &o) {
  *this = *this * o;
  return *this;
}
SlowMPInt &SlowMPInt::operator/=(const SlowMPInt &o) {
  *this = *this / o;
  return *this;
}
SlowMPInt &SlowMPInt::operator%=(const SlowMPInt &o) {
  *this = *this % o;
  return *this;
}
SlowMPInt &SlowMPInt::operator++() {
  *this += 1;
  return *this;
}

SlowMPInt &SlowMPInt::operator--() {
  *this -= 1;
  return *this;
}
