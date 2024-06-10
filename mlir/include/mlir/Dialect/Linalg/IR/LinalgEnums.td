//===- LinalgEnums.td - Linalg dialect base support ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for enums used in linear algebra operations.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_ENUMS
#define LINALG_ENUMS

include "mlir/IR/EnumAttr.td"

// Define the function attribute enums matching the OpDSL functions.
def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"exp", 0>,
  I32EnumAttrCase<"log", 1>,
  I32EnumAttrCase<"abs", 2>,
  I32EnumAttrCase<"ceil", 3>,
  I32EnumAttrCase<"floor", 4>,
  I32EnumAttrCase<"negf", 5>,
  I32EnumAttrCase<"reciprocal", 6>,
  I32EnumAttrCase<"round", 7>,
  I32EnumAttrCase<"sqrt", 8>,
  I32EnumAttrCase<"rsqrt", 9>,
  I32EnumAttrCase<"square", 10>,
  I32EnumAttrCase<"tanh", 11>,
  I32EnumAttrCase<"erf", 12>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}
def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"add", 0>,
  I32EnumAttrCase<"sub", 1>,
  I32EnumAttrCase<"mul", 2>,
  I32EnumAttrCase<"div", 3>,
  I32EnumAttrCase<"div_unsigned", 4>,
  I32EnumAttrCase<"max_signed", 5>,
  I32EnumAttrCase<"min_signed", 6>,
  I32EnumAttrCase<"max_unsigned", 7>,
  I32EnumAttrCase<"min_unsigned", 8>,
  I32EnumAttrCase<"powf", 9>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}
def TernaryFn : I32EnumAttr<"TernaryFn", "", [
  I32EnumAttrCase<"select", 0>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}
def TypeFn : I32EnumAttr<"TypeFn", "", [
  I32EnumAttrCase<"cast_signed", 0>,
  I32EnumAttrCase<"cast_unsigned", 1>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}

#endif // LINALG_ENUMS
