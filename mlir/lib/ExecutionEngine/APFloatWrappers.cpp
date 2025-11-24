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
  MLIR_APFLOAT_WRAPPERS_EXPORT int64_t _mlir_apfloat_##OP(                     \
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
}
