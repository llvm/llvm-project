//===- AVX512Dialect.h - MLIR Dialect for AVX512 ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for AVX512 in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AVX512_AVX512DIALECT_H_
#define MLIR_DIALECT_AVX512_AVX512DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace avx512 {

#define GET_OP_CLASSES
#include "mlir/Dialect/AVX512/AVX512.h.inc"

#include "mlir/Dialect/AVX512/AVX512Dialect.h.inc"

} // namespace avx512
} // namespace mlir

#endif // MLIR_DIALECT_AVX512_AVX512DIALECT_H_
