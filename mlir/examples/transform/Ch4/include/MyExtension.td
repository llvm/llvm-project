//===-- MyExtension.td - Transform dialect tutorial --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 4 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#ifndef MY_EXTENSION
#define MY_EXTENSION

include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the new operation. By convention, prefix its name with `match`
// followed by the name of the dialect extension.
def HasOperandSatisfyingOp : TransformDialectOp<"match.my.has_operand_satisfying",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<TransformOpInterface>,
     // Indicate that the operation implements MatchOpInterface in addition to
     // the TransformOpInterface. This interface is only used as a tag at this
     // point and has no methods that are mandatory to implement.
     MatchOpInterface,
     SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Succeed if any of the operands matches all nested criteria";
  let arguments = (ins TransformHandleTypeInterface:$op);
  let results = (outs TransformParamTypeInterface:$position,
                      Variadic<Transform_AnyHandleOrParamType>:$results);

  // Match operations can be arbitrarily complex, e.g., containing regions.
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $op `:` functional-type($op, results) attr-dict-with-keyword $body
  }];
}

#endif // MY_EXTENSION
