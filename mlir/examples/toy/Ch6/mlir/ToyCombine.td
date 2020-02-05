//===- ToyCombine.td - Pattern Match Optimizations for Toy -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines language-specific pattern match optimizations for Toy using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_COMBINE
#define TOY_COMBINE

include "toy/Ops.td"

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;

//===----------------------------------------------------------------------===//
// Basic Pattern-Match and Rewrite
//===----------------------------------------------------------------------===//

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite using Native Code Call
//===----------------------------------------------------------------------===//

// Native Code Calls may be used for more complex transformations using inline
// C++ and C++ helper functions.

// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

//===----------------------------------------------------------------------===//
// Pattern-Match and Rewrite with Constraints
//===----------------------------------------------------------------------===//

// DRR allows for constraint checking when the transformation is conditional
// on operand properties.

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE
