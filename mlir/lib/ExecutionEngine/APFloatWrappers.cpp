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

extern "C" {

int64_t MLIR_APFLOAT_WRAPPERS_EXPORTED APFloat_add(int32_t semantics,
                                                   uint64_t a, uint64_t b) {
  const llvm::fltSemantics &sem = llvm::APFloatBase::EnumToSemantics(
      static_cast<llvm::APFloatBase::Semantics>(semantics));
  unsigned bitWidth = llvm::APFloatBase::semanticsSizeInBits(sem);
  llvm::APFloat lhs(sem, llvm::APInt(bitWidth, a));
  llvm::APFloat rhs(sem, llvm::APInt(bitWidth, b));
  auto status = lhs.add(rhs, llvm::RoundingMode::NearestTiesToEven);
  return lhs.bitcastToAPInt().getZExtValue();
}

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
