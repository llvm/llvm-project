//===-- TosaUtilOps.td - TOSA dialect utility operations ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines codegen utility operators for the TOSA dialect. 
// These operators are not part of the formal TOSA specification and 
// are intended to aid code generation from TOSA. 
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_UTIL_OPS
#define TOSA_UTIL_OPS

include "mlir/IR/OpBase.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Dialect/Tosa/IR/TosaInterfaces.td"

include "mlir/Dialect/Tosa/IR/TosaTypesBase.td"
include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

def Tosa_ApplyScaleOp: Tosa_Op<"apply_scale", [NoSideEffect] # ElementwiseMappable.traits> {
  let summary = "Rescale scalar operator for Tosa tensor operators";

  let description = [{
    Applies rescaling for fixed point values. This behavior is replicated in
    multiple quantized operations (mul, convolution, rescale, matmul, pooling).

    The commonplace implementation is to use i64 operations to avoid integer
    overflow with target specific implementations can use native operations to
    avoid wider than necessary types.
  }];

  let arguments = (ins
    Tosa_IntLike:$value,
    Tosa_IntLike:$multiplier,
    Tosa_Int8Like:$shift,
    BoolAttr:$double_round
  );

  let results = (outs
    Tosa_IntLike:$output
  );
}

//===----------------------------------------------------------------------===//
// Operator: yield
//===----------------------------------------------------------------------===//
def Tosa_YieldOp : Tosa_Op<"yield", [
       Terminator,
       NoSideEffect]> {
  let summary = "yield operator";

  let description = [{
    return operation within the conditional and body of
    structured control flow. Operation takes variadic operands
    but produces no results of its own.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$inputs
  );
}

#endif // TOSA_UTIL_OPS
