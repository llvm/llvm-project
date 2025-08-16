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
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Dialect/Tosa/IR/TosaInterfaces.td"

include "mlir/Dialect/Tosa/IR/TosaTypesBase.td"
include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

def Tosa_ApplyScaleOp :
  Tosa_Op<"apply_scale",
          [Pure, DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
          ElementwiseMappable.traits> {
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
    Tosa_RoundingModeAttr:$rounding_mode
  );

  let results = (outs
    Tosa_IntLike:$output
  );

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Operator: yield
//===----------------------------------------------------------------------===//
def Tosa_YieldOp : Tosa_Op<"yield", [
       Terminator,
       Pure]> {
  let summary = "yield operator";

  let description = [{
    return operation within the conditional and body of
    structured control flow. Operation takes variadic operands
    but produces no results of its own.
  }];

  let arguments = (ins
    Variadic<Tosa_Tensor>:$inputs
  );

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
}

//===----------------------------------------------------------------------===//
// Operator: variable
//===----------------------------------------------------------------------===//
def Tosa_VariableOp : Tosa_Op<"variable", []> {
  let summary = "Defines a variable";

  let description = [{
    Defines a new TOSA variable. This is a persistent mutable value across multiple
    TOSA graph invocations. Modifications are expressed using read/write semantics.
  }];

  let arguments = (ins
    SymbolNameAttr:$name,
    IndexElementsAttr:$var_shape,
    TypeAttr:$type,
    OptionalAttr<AnyAttr>:$initial_value
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_VARIABLE]>,
  ];

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $name
    attr-dict
    custom<VariableOpTypeOrInitialValue>($var_shape, $type, $initial_value)
  }];

  let builders = [Tosa_VariableOpBuilder];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: variable_write
//===----------------------------------------------------------------------===//
def Tosa_VariableWriteOp : Tosa_Op<"variable_write", []> {
  let summary = "write_buffer operator";

  let description = [{
    Assigns a value to the pseudo-buffer resource holding a persistent mutable tensor.
  }];

  let arguments = (ins
    SymbolNameAttr:$name,
    Tosa_Tensor:$input1
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_VARIABLE]>,
  ];

  let assemblyFormat = [{
    $name attr-dict `,` $input1 `:` type($input1)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Operator: variable_read
//===----------------------------------------------------------------------===//
def Tosa_VariableReadOp : Tosa_Op<"variable_read", []> {
  let summary = "read_buffer operator";

  let description = [{
    Reads the value from a pseudo-buffer resource holding a persistent mutable tensor.
  }];

  let arguments = (ins
    SymbolNameAttr:$name
  );

  let results = (outs
    Tosa_Tensor:$output1
  );

  list<Availability> availability = [
    Profile<[Tosa_PRO_INT, Tosa_PRO_FP]>,
    Extension<[Tosa_EXT_VARIABLE]>,
  ];

  let assemblyFormat = [{
    $name attr-dict `:` type($output1)
  }];

  let hasVerifier = 1;
}

#endif // TOSA_UTIL_OPS
