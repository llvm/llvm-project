//===- MathBase.td - Base definitions for math dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MATH_BASE
#define MATH_BASE
include "mlir/IR/OpBase.td"
def Math_Dialect : Dialect {
  let name = "math";
  let cppNamespace = "::mlir::math";
  let description = [{
    The math dialect is intended to hold mathematical operations on integer and
    floating type beyond simple arithmetics.
  }];
}
#endif // MATH_BASE
