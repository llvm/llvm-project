//==-- llvm/Support/CheckedArithmetic.h - Safe arithmetical operations *- C++ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains generic functions for operating on integers which
// give the indication on whether the operation has overflown.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CHECKEDARITHMETIC_H
#define LLVM_SUPPORT_CHECKEDARITHMETIC_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Optional.h"

#include <type_traits>

namespace {

/// Utility function to apply a given method of \c APInt \p F to \p LHS and
/// \p RHS.
/// \return Empty optional if the operation overflows, or result otherwise.
template <typename T, typename F>
std::enable_if_t<std::is_integral<T>::value && sizeof(T) * 8 <= 64,
                 llvm::Optional<T>>
checkedOp(T LHS, T RHS, F Op, bool Signed = true) {
  llvm::APInt ALHS(/*BitSize=*/sizeof(T) * 8, LHS, Signed);
  llvm::APInt ARHS(/*BitSize=*/sizeof(T) * 8, RHS, Signed);
  bool Overflow;
  llvm::APInt Out = (ALHS.*Op)(ARHS, Overflow);
  if (Overflow)
    return llvm::None;
  return Signed ? Out.getSExtValue() : Out.getZExtValue();
}
}

namespace llvm {

/// Add two signed integers \p LHS and \p RHS.
/// \return Optional of sum if no signed overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_signed<T>::value, llvm::Optional<T>>
checkedAdd(T LHS, T RHS) {
  return checkedOp(LHS, RHS, &llvm::APInt::sadd_ov);
}

/// Subtract two signed integers \p LHS and \p RHS.
/// \return Optional of sum if no signed overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_signed<T>::value, llvm::Optional<T>>
checkedSub(T LHS, T RHS) {
  return checkedOp(LHS, RHS, &llvm::APInt::ssub_ov);
}

/// Multiply two signed integers \p LHS and \p RHS.
/// \return Optional of product if no signed overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_signed<T>::value, llvm::Optional<T>>
checkedMul(T LHS, T RHS) {
  return checkedOp(LHS, RHS, &llvm::APInt::smul_ov);
}

/// Multiply A and B, and add C to the resulting product.
/// \return Optional of result if no signed overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_signed<T>::value, llvm::Optional<T>>
checkedMulAdd(T A, T B, T C) {
  if (auto Product = checkedMul(A, B))
    return checkedAdd(*Product, C);
  return llvm::None;
}

/// Add two unsigned integers \p LHS and \p RHS.
/// \return Optional of sum if no unsigned overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_unsigned<T>::value, llvm::Optional<T>>
checkedAddUnsigned(T LHS, T RHS) {
  return checkedOp(LHS, RHS, &llvm::APInt::uadd_ov, /*Signed=*/false);
}

/// Multiply two unsigned integers \p LHS and \p RHS.
/// \return Optional of product if no unsigned overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_unsigned<T>::value, llvm::Optional<T>>
checkedMulUnsigned(T LHS, T RHS) {
  return checkedOp(LHS, RHS, &llvm::APInt::umul_ov, /*Signed=*/false);
}

/// Multiply unsigned integers A and B, and add C to the resulting product.
/// \return Optional of result if no unsigned overflow occurred,
/// \c None otherwise.
template <typename T>
std::enable_if_t<std::is_unsigned<T>::value, llvm::Optional<T>>
checkedMulAddUnsigned(T A, T B, T C) {
  if (auto Product = checkedMulUnsigned(A, B))
    return checkedAddUnsigned(*Product, C);
  return llvm::None;
}

} // End llvm namespace

#endif
