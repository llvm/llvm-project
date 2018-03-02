//==-- llvm/Support/CheckedArithmetic.h - Safe arithmetical operations *- C++ //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include <type_traits>

namespace {

/// Utility function to apply a given method of \c APInt \p F to \p LHS and
/// \p RHS, and write the output into \p Res.
/// \return Whether the operation overflows.
template <typename T, typename F>
typename std::enable_if<std::is_integral<T>::value && sizeof(T) * 8 <= 64,
                        bool>::type
checkedOp(T LHS, T RHS, F Op, T *Res = nullptr, bool Signed = true) {
  llvm::APInt ALHS(/*BitSize=*/sizeof(T) * 8, LHS, Signed);
  llvm::APInt ARHS(/*BitSize=*/sizeof(T) * 8, RHS, Signed);
  bool Overflow;
  llvm::APInt Out = (ALHS.*Op)(ARHS, Overflow);
  if (Res)
    *Res = Signed ? Out.getSExtValue() : Out.getZExtValue();
  return Overflow;
}
}

namespace llvm {

/// Add two signed integers \p LHS and \p RHS, write into \p Res if non-null.
/// Does not guarantee saturating arithmetic.
/// \return Whether the result overflows.
template <typename T>
typename std::enable_if<std::is_signed<T>::value, bool>::type
checkedAdd(T LHS, T RHS, T *Res = nullptr) {
  return checkedOp(LHS, RHS, &llvm::APInt::sadd_ov, Res);
}

/// Multiply two signed integers \p LHS and \p RHS, write into \p Res if
/// non-null.
/// Does not guarantee saturating arithmetic.
/// \return Whether the result overflows.
template <typename T>
typename std::enable_if<std::is_signed<T>::value, bool>::type
checkedMul(T LHS, T RHS, T *Res = nullptr) {
  return checkedOp(LHS, RHS, &llvm::APInt::smul_ov, Res);
}

/// Add two unsigned integers \p LHS and \p RHS, write into \p Res if non-null.
/// Does not guarantee saturating arithmetic.
/// \return Whether the result overflows.
template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, bool>::type
checkedAddUnsigned(T LHS, T RHS, T *Res = nullptr) {
  return checkedOp(LHS, RHS, &llvm::APInt::uadd_ov, Res, /*Signed=*/false);
}

/// Multiply two unsigned integers \p LHS and \p RHS, write into \p Res if
/// non-null.
/// Does not guarantee saturating arithmetic.
/// \return Whether the result overflows.
template <typename T>
typename std::enable_if<std::is_unsigned<T>::value, bool>::type
checkedMulUnsigned(T LHS, T RHS, T *Res = nullptr) {
  return checkedOp(LHS, RHS, &llvm::APInt::umul_ov, Res, /*Signed=*/false);
}

} // End llvm namespace

#endif
