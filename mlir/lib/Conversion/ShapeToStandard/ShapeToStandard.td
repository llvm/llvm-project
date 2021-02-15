//==-- ShapeToStandard.td - Shape to Standard Patterns -------*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower Shape ops to Std.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SHAPETOSTANDARD_TD
#define MLIR_CONVERSION_SHAPETOSTANDARD_TD

include "mlir/Dialect/Shape/IR/ShapeOps.td"

def BroadcastableStringAttr : NativeCodeCall<[{
  $_builder.getStringAttr("required broadcastable shapes")
}]>;

def CstrBroadcastableToRequire : Pat<(Shape_CstrBroadcastableOp $shapes),
            (Shape_CstrRequireOp
              (Shape_IsBroadcastableOp $shapes),
              (BroadcastableStringAttr))>;

#endif // MLIR_CONVERSION_SHAPETOSTANDARD_TD
