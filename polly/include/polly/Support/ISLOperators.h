//===------ IslOperators.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Operator overloads for isl C++ objects.
//
//===----------------------------------------------------------------------===//

#include "isl/isl-noexceptions.h"
namespace polly {

inline isl::pw_aff operator+(isl::pw_aff A, isl::pw_aff B) { return A.add(B); }

inline isl::pw_aff operator+(isl::val V, isl::pw_aff A) {
  isl::pw_aff AV(A.domain(), V);
  return A.add(AV);
}

inline isl::pw_aff operator+(isl::pw_aff A, isl::val V) { return V + A; }

inline isl::pw_aff operator+(int i, isl::pw_aff A) {
  isl::ctx ctx = A.get_ctx();
  return A + isl::val(ctx, i);
}

inline isl::pw_aff operator+(isl::pw_aff A, int i) { return i + A; }

inline isl::pw_aff operator*(isl::pw_aff A, isl::pw_aff B) { return A.mul(B); }

inline isl::pw_aff operator*(isl::val V, isl::pw_aff A) {
  isl::pw_aff AV(A.domain(), V);
  return A.add(AV);
}

inline isl::pw_aff operator*(isl::pw_aff A, isl::val V) { return V * A; }

inline isl::pw_aff operator*(int i, isl::pw_aff A) {
  isl::ctx ctx = A.get_ctx();
  return A * isl::val(ctx, i);
}

inline isl::pw_aff operator*(isl::pw_aff A, int i) { return i * A; }

inline isl::pw_aff operator-(isl::pw_aff A, isl::pw_aff B) { return A.sub(B); }

inline isl::pw_aff operator-(isl::val V, isl::pw_aff A) {
  isl::pw_aff AV(A.domain(), V);
  return AV - A;
}

inline isl::pw_aff operator-(isl::pw_aff A, isl::val V) {
  isl::pw_aff AV(A.domain(), V);
  return A - AV;
}

inline isl::pw_aff operator-(int i, isl::pw_aff A) {
  isl::ctx ctx = A.get_ctx();
  return isl::val(ctx, i) - A;
}

inline isl::pw_aff operator-(isl::pw_aff A, int i) {
  isl::ctx ctx = A.get_ctx();
  return A - isl::val(ctx, i);
}

inline isl::pw_aff operator/(isl::pw_aff A, isl::pw_aff B) {
  return A.tdiv_q(B);
}

inline isl::pw_aff operator/(isl::val V, isl::pw_aff A) {
  isl::pw_aff AV(A.domain(), V);
  return AV / A;
}

inline isl::pw_aff operator/(isl::pw_aff A, isl::val V) {
  isl::pw_aff AV(A.domain(), V);
  return A / AV;
}

inline isl::pw_aff operator/(int i, isl::pw_aff A) {
  isl::ctx ctx = A.get_ctx();
  return isl::val(ctx, i) / A;
}

inline isl::pw_aff operator/(isl::pw_aff A, int i) {
  isl::ctx ctx = A.get_ctx();
  return A / isl::val(ctx, i);
}

inline isl::pw_aff operator%(isl::pw_aff A, isl::pw_aff B) {
  return A.tdiv_r(B);
}

inline isl::pw_aff operator%(isl::val V, isl::pw_aff A) {
  isl::pw_aff AV(A.domain(), V);
  return AV % A;
}

inline isl::pw_aff operator%(isl::pw_aff A, isl::val V) {
  isl::pw_aff AV(A.domain(), V);
  return A % AV;
}

inline isl::pw_aff operator%(int i, isl::pw_aff A) {
  isl::ctx ctx = A.get_ctx();
  return isl::val(ctx, i) % A;
}

inline isl::pw_aff operator%(isl::pw_aff A, int i) {
  isl::ctx ctx = A.get_ctx();
  return A % isl::val(ctx, i);
}
} // namespace polly
