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

// Join two I32EnumAttrCase lists. This joining takes care that the
// 'int enum values' in the combined list do not overlap. It does this
// by adding to each element of second list the offset '!size(a)'.
class JoinTwoI32EnumAttrCaseList< list<I32EnumAttrCase> a,
                                  list<I32EnumAttrCase> b> {
  int aSize = !size(a);
  list<I32EnumAttrCase> result =
             !foldl(a, b, acc, var,
                    acc # [I32EnumAttrCase<var.symbol,
                                           !add(var.value, aSize)
                                           >]);
}

// Flatten 'list of list of I32EnumAttrCase' to 'list of I32EnumAttrCase'.
// The flattening (via call to 'join') ensures no overlap in enum values.
class ConcatI32EnumAtrCaseList< list<list<I32EnumAttrCase>> l> {
  list<I32EnumAttrCase> result =
             !foldl([]<I32EnumAttrCase>, l, acc, var,
                    JoinTwoI32EnumAttrCaseList<acc, var>.result);
}

// Define a unified `enum class : i32` for all element-wise op functions.
def ElementwiseKind :
            I32EnumAttr<"ElementwiseKind",
                        "",
                        ConcatI32EnumAtrCaseList<[UnaryFn.enumerants,
                                                  BinaryFn.enumerants,
                                                  TernaryFn.enumerants]>.result
                      > {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}

// Define an `enum class : i32` that marks where each individual enum class
// e.g. UnaryFn, BinaryFn, etc. end in the unified enum class ElementwiseKind.
def ElementwiseCaseLimits : I32EnumAttr<"ElementwiseCaseLimits", "", []> {
  int last_unary = !size(UnaryFn.enumerants);
  int last_binary = !add(last_unary, !size(BinaryFn.enumerants));
  int last_ternary = !add(last_binary, !size(TernaryFn.enumerants));

  let enumerants =  [
         I32EnumAttrCase<"LastUnary", last_unary>,
         I32EnumAttrCase<"LastBinary", last_binary>,
         I32EnumAttrCase<"LastTernary", last_ternary>];
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::linalg";
}

// Define an `enum class : i32` to categorise arity elementwise ops.
def ElementwiseArityGroup : I32EnumAttr<"ElementwiseArityGroup", "", [
  I32EnumAttrCase<"Unary", 1>,
  I32EnumAttrCase<"Binary", 2>,
  I32EnumAttrCase<"Ternary", 3>
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

/// We use F(m, r) to define the size of minimal filtering algorithms.
/// m is the output dimension and r is the filter dimension. We can get
/// the input dimension, alpha, from the formula, alpha = m + r - 1.
///
/// For example, when m = 2 and r = 3, we know its input size is 4.
/// The Conv2D will operate on 4x4 input data with 3x3 filter and get
/// 2x2 output result.
def WinogradConv2DFmr : I32EnumAttr<"WinogradConv2DFmr", "", [
      I32EnumAttrCase<"F_2_3", 0>,
      I32EnumAttrCase<"F_4_3", 1>,
      I32EnumAttrCase<"F_2_5", 2>,
]>{
  let cppNamespace = "mlir::linalg";
}

#endif // LINALG_ENUMS
