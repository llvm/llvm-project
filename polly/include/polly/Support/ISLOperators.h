//===------ ISLOperators.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operator overloads for isl C++ objects.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISLOPERATORS_H
#define POLLY_ISLOPERATORS_H

#include "isl/isl-noexceptions.h"

namespace polly {

/// Addition
/// @{
inline isl::pw_aff operator+(isl::pw_aff Left, isl::pw_aff Right) {
  return Left.add(Right);
}

inline isl::pw_aff operator+(isl::val ValLeft, isl::pw_aff Right) {
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.add(Right);
}

inline isl::pw_aff operator+(isl::pw_aff Left, isl::val ValRight) {
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.add(Right);
}

inline isl::pw_aff operator+(long IntLeft, isl::pw_aff Right) {
  isl::ctx Ctx = Right.get_ctx();
  isl::val ValLeft(Ctx, IntLeft);
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.add(Right);
}

inline isl::pw_aff operator+(isl::pw_aff Left, long IntRight) {
  isl::ctx Ctx = Left.get_ctx();
  isl::val ValRight(Ctx, IntRight);
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.add(Right);
}
/// @}

/// Multiplication
/// @{
inline isl::pw_aff operator*(isl::pw_aff Left, isl::pw_aff Right) {
  return Left.mul(Right);
}

inline isl::pw_aff operator*(isl::val ValLeft, isl::pw_aff Right) {
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.mul(Right);
}

inline isl::pw_aff operator*(isl::pw_aff Left, isl::val ValRight) {
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.mul(Right);
}

inline isl::pw_aff operator*(long IntLeft, isl::pw_aff Right) {
  isl::ctx Ctx = Right.get_ctx();
  isl::val ValLeft(Ctx, IntLeft);
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.mul(Right);
}

inline isl::pw_aff operator*(isl::pw_aff Left, long IntRight) {
  isl::ctx Ctx = Left.get_ctx();
  isl::val ValRight(Ctx, IntRight);
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.mul(Right);
}
/// @}

/// Subtraction
/// @{
inline isl::pw_aff operator-(isl::pw_aff Left, isl::pw_aff Right) {
  return Left.sub(Right);
}

inline isl::pw_aff operator-(isl::val ValLeft, isl::pw_aff Right) {
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.sub(Right);
}

inline isl::pw_aff operator-(isl::pw_aff Left, isl::val ValRight) {
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.sub(Right);
}

inline isl::pw_aff operator-(long IntLeft, isl::pw_aff Right) {
  isl::ctx Ctx = Right.get_ctx();
  isl::val ValLeft(Ctx, IntLeft);
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.sub(Right);
}

inline isl::pw_aff operator-(isl::pw_aff Left, long IntRight) {
  isl::ctx Ctx = Left.get_ctx();
  isl::val ValRight(Ctx, IntRight);
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.sub(Right);
}
/// @}

/// Division
///
/// This division rounds towards zero. This follows the semantics of C/C++.
///
/// @{
inline isl::pw_aff operator/(isl::pw_aff Left, isl::pw_aff Right) {
  return Left.tdiv_q(Right);
}

inline isl::pw_aff operator/(isl::val ValLeft, isl::pw_aff Right) {
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.tdiv_q(Right);
}

inline isl::pw_aff operator/(isl::pw_aff Left, isl::val ValRight) {
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.tdiv_q(Right);
}

inline isl::pw_aff operator/(long IntLeft, isl::pw_aff Right) {
  isl::ctx Ctx = Right.get_ctx();
  isl::val ValLeft(Ctx, IntLeft);
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.tdiv_q(Right);
}

inline isl::pw_aff operator/(isl::pw_aff Left, long IntRight) {
  isl::ctx Ctx = Left.get_ctx();
  isl::val ValRight(Ctx, IntRight);
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.tdiv_q(Right);
}
/// @}

/// Remainder
///
/// This is the remainder of a division which rounds towards zero. This follows
/// the semantics of C/C++.
///
/// @{
inline isl::pw_aff operator%(isl::pw_aff Left, isl::pw_aff Right) {
  return Left.tdiv_r(Right);
}

inline isl::pw_aff operator%(isl::val ValLeft, isl::pw_aff Right) {
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.tdiv_r(Right);
}

inline isl::pw_aff operator%(isl::pw_aff Left, isl::val ValRight) {
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.tdiv_r(Right);
}

inline isl::pw_aff operator%(long IntLeft, isl::pw_aff Right) {
  isl::ctx Ctx = Right.get_ctx();
  isl::val ValLeft(Ctx, IntLeft);
  isl::pw_aff Left(Right.domain(), ValLeft);
  return Left.tdiv_r(Right);
}

inline isl::pw_aff operator%(isl::pw_aff Left, long IntRight) {
  isl::ctx Ctx = Left.get_ctx();
  isl::val ValRight(Ctx, IntRight);
  isl::pw_aff Right(Left.domain(), ValRight);
  return Left.tdiv_r(Right);
}
/// @}

} // namespace polly

#endif // POLLY_ISLOPERATORS_H
