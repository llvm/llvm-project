//===- ArithToAPFloat.h - Arith to APFloat impl conversion ---*- C++ ----*-===//
//
// Part of the APFloat Project, under the Apache License v2.0 with APFloat
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH APFloat-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOAPFLOAT_ARITHTOAPFLOAT_H
#define MLIR_CONVERSION_ARITHTOAPFLOAT_ARITHTOAPFLOAT_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_ARITHTOAPFLOATCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOAPFLOAT_ARITHTOAPFLOAT_H
