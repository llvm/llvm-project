//===- MemRefToEmitCPass.h - A Pass to convert MemRef to EmitC ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITCPASS_H
#define MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITCPASS_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTMEMREFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITCPASS_H
