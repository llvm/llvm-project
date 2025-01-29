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


class ConvDimEnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I8, sym, str, val>;

def ConvDimEnumAttr :
    IntEnumAttr<I8, "ConvDimEnum", "summary", [
      /// Batch is a dimension of input and output, indexed from a parallel loop.
      ConvDimEnumAttrCase<"BATCH", 0, "N">,
      /// Input channel is a dimension in all tensors, indexed from a reduction loop.
      /// Depthwise convolutions perform no reduction across channels and therefore
      /// do not use this.
      ConvDimEnumAttrCase<"INPUT_CHANNEL", 1, "C">,
      /// Output channel is a dimension in filter and output, index from a parallel loop.
      ConvDimEnumAttrCase<"OUTPUT_CHANNEL", 2, "F">,
      /// Group is a dimension in all tensors and indexed from a parallel loop.
      ConvDimEnumAttrCase<"GROUP", 3, "G">,
      /// Spatial dimensions occur in all tensors. Output is indexed from a parallel
      /// loop, filter from a reduction loop and input from both.
      ConvDimEnumAttrCase<"SPATIAL_0", 4, "0">,
      ConvDimEnumAttrCase<"SPATIAL_1", 5, "1">,
      ConvDimEnumAttrCase<"SPATIAL_2", 6, "2">,
    ]> {
  let underlyingType = "uint8_t";
  let cppNamespace = "::mlir::linalg";
}

#endif // LINALG_ENUMS
