//===- ArithEmulateF64.h - Double precision on 64-bit integers --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_ARITHEMULATEF64_ARITHEMULATEF64_H
#define MLIR_CONVERSION_ARITHEMULATEF64_ARITHEMULATEF64_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

#define GEN_PASS_DECL_ARITHEMULATEF64
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHEMULATEF64_ARITHEMULATEF64_H
