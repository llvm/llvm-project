//===- APFloatWrappers.cpp - Software Implementation of FP Arithmetics --- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the APFloat infrastructure to MLIR programs as a runtime
// library. APFloat is a software implementation of floating point arithmetics.
//
// On the MLIR side, floating-point values must be bitcasted to 64-bit integers
// before calling a runtime function. If a floating-point type has less than
// 64 bits, it must be zero-extended to 64 bits after bitcasting it to an
// integer.
//
// Runtime functions receive the floating-point operands of the arithmeic
// operation in the form of 64-bit integers, along with the APFloat semantics
// in the form of a 32-bit integer, which will be interpreted as an
// APFloatBase::Semantics enum value.
//
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"

#ifdef _WIN32
#ifndef MLIR_APFLOAT_WRAPPERS_EXPORT
#ifdef mlir_apfloat_wrappers_EXPORTS
// We are building this library
#define MLIR_APFLOAT_WRAPPERS_EXPORT __declspec(dllexport)
#else
// We are using this library
#define MLIR_APFLOAT_WRAPPERS_EXPORT __declspec(dllimport)
#endif // mlir_apfloat_wrappers_EXPORTS
#endif // MLIR_APFLOAT_WRAPPERS_EXPORT
#else
// Non-windows: use visibility attributes.
#define MLIR_APFLOAT_WRAPPERS_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

/// Binary operations without rounding mode.
#define APFLOAT_BINARY_OP(OP)                                                  \
  MLIR_APFLOAT_WRAPPERS_EXPORT int64_t _mlir_apfloat_##OP(                     \
      int32_t semantics, uint64_t a, uint64_t b) {                             \
    const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(        \
        static_cast<llvm::APFloatBase::Semantics>(semantics));                 \
    unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);           \
    llvm::APFloat lhs(sem, llvm::APInt(bitWidth, a));                          \
    llvm::APFloat rhs(sem, llvm::APInt(bitWidth, b));                          \
    lhs.OP(rhs);                                                               \
    return lhs.bitcastToAPInt().getZExtValue();                                \
  }

/// Binary operations with rounding mode.
#define APFLOAT_BINARY_OP_ROUNDING_MODE(OP, ROUNDING_MODE)                     \
  MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t _mlir_apfloat_##OP(                    \
      int32_t semantics, uint64_t a, uint64_t b) {                             \
    const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(        \
        static_cast<llvm::APFloatBase::Semantics>(semantics));                 \
    unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);           \
    llvm::APFloat lhs(sem, llvm::APInt(bitWidth, a));                          \
    llvm::APFloat rhs(sem, llvm::APInt(bitWidth, b));                          \
    lhs.OP(rhs, ROUNDING_MODE);                                                \
    return lhs.bitcastToAPInt().getZExtValue();                                \
  }

extern "C" {

#define BIN_OPS_WITH_ROUNDING(X)                                               \
  X(add, llvm::RoundingMode::NearestTiesToEven)                                \
  X(subtract, llvm::RoundingMode::NearestTiesToEven)                           \
  X(multiply, llvm::RoundingMode::NearestTiesToEven)                           \
  X(divide, llvm::RoundingMode::NearestTiesToEven)

BIN_OPS_WITH_ROUNDING(APFLOAT_BINARY_OP_ROUNDING_MODE)
#undef BIN_OPS_WITH_ROUNDING
#undef APFLOAT_BINARY_OP_ROUNDING_MODE

APFLOAT_BINARY_OP(remainder)

#undef APFLOAT_BINARY_OP

MLIR_APFLOAT_WRAPPERS_EXPORT void printApFloat(int32_t semantics, uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  double d = x.convertToDouble();
  fprintf(stdout, "%lg", d);
}

MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t
_mlir_apfloat_convert(int32_t inSemantics, int32_t outSemantics, uint64_t a) {
  const llvm::fltSemantics &inSem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(inSemantics));
  const llvm::fltSemantics &outSem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(outSemantics));
  unsigned bitWidthIn = llvm::APFloatBase::semanticsSizeInBits(inSem);
  llvm::APFloat val(inSem, llvm::APInt(bitWidthIn, a));
  // TODO: Custom rounding modes are not supported yet.
  bool losesInfo;
  val.convert(outSem, llvm::RoundingMode::NearestTiesToEven, &losesInfo);
  llvm::APInt result = val.bitcastToAPInt();
  return result.getZExtValue();
}

MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t _mlir_apfloat_convert_to_int(
    int32_t semantics, int32_t resultWidth, bool isUnsigned, uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned inputWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat val(sem, llvm::APInt(inputWidth, a));
  llvm::APSInt result(resultWidth, isUnsigned);
  bool isExact;
  // TODO: Custom rounding modes are not supported yet.
  val.convertToInteger(result, llvm::RoundingMode::NearestTiesToEven, &isExact);
  // This function always returns uint64_t, regardless of the desired result
  // width. It does not matter whether we zero-extend or sign-extend the APSInt
  // to 64 bits because the generated IR in arith-to-apfloat will truncate the
  // result to the desired result width.
  return result.getZExtValue();
}

MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t _mlir_apfloat_convert_from_int(
    int32_t semantics, int32_t inputWidth, bool isUnsigned, uint64_t a) {
  llvm::APInt val(inputWidth, a, /*isSigned=*/!isUnsigned);
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  llvm::APFloat result(sem);
  // TODO: Custom rounding modes are not supported yet.
  result.convertFromAPInt(val, /*IsSigned=*/!isUnsigned,
                          llvm::RoundingMode::NearestTiesToEven);
  return result.bitcastToAPInt().getZExtValue();
}

MLIR_APFLOAT_WRAPPERS_EXPORT int8_t _mlir_apfloat_compare(int32_t semantics,
                                                          uint64_t a,
                                                          uint64_t b) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  llvm::APFloat y(sem, llvm::APInt(bitWidth, b));
  return static_cast<int8_t>(x.compare(y));
}

MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t _mlir_apfloat_neg(int32_t semantics,
                                                        uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  x.changeSign();
  return x.bitcastToAPInt().getZExtValue();
}

MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t _mlir_apfloat_abs(int32_t semantics,
                                                        uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  return abs(x).bitcastToAPInt().getZExtValue();
}

MLIR_APFLOAT_WRAPPERS_EXPORT bool _mlir_apfloat_isfinite(int32_t semantics,
                                                         uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  return x.isFinite();
}

MLIR_APFLOAT_WRAPPERS_EXPORT bool _mlir_apfloat_isinfinite(int32_t semantics,
                                                           uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  return x.isInfinity();
}

MLIR_APFLOAT_WRAPPERS_EXPORT bool _mlir_apfloat_isnormal(int32_t semantics,
                                                         uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  return x.isNormal();
}

MLIR_APFLOAT_WRAPPERS_EXPORT bool _mlir_apfloat_isnan(int32_t semantics,
                                                      uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  return x.isNaN();
}

MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t
_mlir_apfloat_fused_multiply_add(int32_t semantics, uint64_t operand,
                                 uint64_t multiplicand, uint64_t addend) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat operand_(sem, llvm::APInt(bitWidth, operand));
  llvm::APFloat multiplicand_(sem, llvm::APInt(bitWidth, multiplicand));
  llvm::APFloat addend_(sem, llvm::APInt(bitWidth, addend));
  llvm::detail::opStatus stat = operand_.fusedMultiplyAdd(
      multiplicand_, addend_, llvm::RoundingMode::NearestTiesToEven);
  assert(stat == llvm::APFloatBase::opOK &&
         "expected fusedMultiplyAdd status to be OK");
  (void)stat;
  return operand_.bitcastToAPInt().getZExtValue();
}

/// Min/max operations.
#define APFLOAT_MIN_MAX_OP(OP)                                                 \
  MLIR_APFLOAT_WRAPPERS_EXPORT uint64_t _mlir_apfloat_##OP(                    \
      int32_t semantics, uint64_t a, uint64_t b) {                             \
    const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(        \
        static_cast<llvm::APFloatBase::Semantics>(semantics));                 \
    unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);           \
    llvm::APFloat lhs(sem, llvm::APInt(bitWidth, a));                          \
    llvm::APFloat rhs(sem, llvm::APInt(bitWidth, b));                          \
    llvm::APFloat result = llvm::OP(lhs, rhs);                                 \
    return result.bitcastToAPInt().getZExtValue();                             \
  }

APFLOAT_MIN_MAX_OP(minimum)
APFLOAT_MIN_MAX_OP(maximum)
APFLOAT_MIN_MAX_OP(minnum)
APFLOAT_MIN_MAX_OP(maxnum)

#undef APFLOAT_MIN_MAX_OP
}
