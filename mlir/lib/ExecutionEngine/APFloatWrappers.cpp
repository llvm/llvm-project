//===- ArmRunnerUtils.cpp - Utilities for configuring architecture properties //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"

#include <iostream>

#if (defined(_WIN32) || defined(__CYGWIN__))
#define MLIR_APFLOAT_WRAPPERS_EXPORTED __declspec(dllexport)
#else
#define MLIR_APFLOAT_WRAPPERS_EXPORTED __attribute__((visibility("default")))
#endif

#define APFLOAT_BINARY_OP(OP)                                                  \
  int64_t MLIR_APFLOAT_WRAPPERS_EXPORTED APFloat_##OP(                         \
      int32_t semantics, uint64_t a, uint64_t b) {                             \
    const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(        \
        static_cast<llvm::APFloatBase::Semantics>(semantics));                 \
    unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);           \
    llvm::APFloat lhs(sem, llvm::APInt(bitWidth, a));                          \
    llvm::APFloat rhs(sem, llvm::APInt(bitWidth, b));                          \
    llvm::APFloatBase::opStatus status = lhs.OP(rhs);                          \
    assert(status == llvm::APFloatBase::opOK && "expected " #OP                \
                                                " opstatus to be OK");         \
    return lhs.bitcastToAPInt().getZExtValue();                                \
  }

#define APFLOAT_BINARY_OP_ROUNDING_MODE(OP, ROUNDING_MODE)                     \
  int64_t MLIR_APFLOAT_WRAPPERS_EXPORTED APFloat_##OP(                         \
      int32_t semantics, uint64_t a, uint64_t b) {                             \
    const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(        \
        static_cast<llvm::APFloatBase::Semantics>(semantics));                 \
    unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);           \
    llvm::APFloat lhs(sem, llvm::APInt(bitWidth, a));                          \
    llvm::APFloat rhs(sem, llvm::APInt(bitWidth, b));                          \
    llvm::APFloatBase::opStatus status = lhs.OP(rhs, ROUNDING_MODE);           \
    assert(status == llvm::APFloatBase::opOK && "expected " #OP                \
                                                " opstatus to be OK");         \
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
APFLOAT_BINARY_OP(mod)

#undef APFLOAT_BINARY_OP

void MLIR_APFLOAT_WRAPPERS_EXPORTED printApFloat(int32_t semantics,
                                                 uint64_t a) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat x(sem, llvm::APInt(bitWidth, a));
  double d = x.convertToDouble();
  std::cout << d << std::endl;
}
}
