//===- EmitC.td - EmitC operations--------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITC
#define MLIR_DIALECT_EMITC_IR_EMITC

include "mlir/Dialect/EmitC/IR/EmitCAttributes.td"
include "mlir/Dialect/EmitC/IR/EmitCTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// EmitC op definitions
//===----------------------------------------------------------------------===//

// Base class for EmitC dialect ops.
class EmitC_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitC_Dialect, mnemonic, traits>;

def EmitC_ApplyOp : EmitC_Op<"apply", []> {
  let summary = "Apply operation";
  let description = [{
    With the `apply` operation the operators & (address of) and * (contents of)
    can be applied to a single operand.

    Example:

    ```mlir
    // Custom form of applying the & operator.
    %0 = emitc.apply "&"(%arg0) : (i32) -> !emitc.opaque<"int32_t*">

    // Generic form of the same operation.
    %0 = "emitc.apply"(%arg0) {applicableOperator = "&"}
        : (i32) -> !emitc.opaque<"int32_t*">

    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the operator to apply">:$applicableOperator,
    AnyType:$operand
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
  }];
  let hasVerifier = 1;
}

def EmitC_CallOp : EmitC_Op<"call", []> {
  let summary = "Call operation";
  let description = [{
    The `call` operation represents a C++ function call. The call allows
    specifying order of operands and attributes in the call as follows:

    - integer value of index type refers to an operand;
    - attribute which will get lowered to constant value in call;

    Example:

    ```mlir
    // Custom form defining a call to `foo()`.
    %0 = emitc.call "foo" () : () -> i32

    // Generic form of the same operation.
    %0 = "emitc.call"() {callee = "foo"} : () -> i32
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the C++ function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Arg<OptionalAttr<ArrayAttr>, "template arguments">:$template_args,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def EmitC_ConstantOp : EmitC_Op<"constant", [ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `constant` operation also supports the EmitC opaque
    attribute and the EmitC opaque type.

    Example:

    ```mlir
    // Integer constant
    %0 = "emitc.constant"(){value = 42 : i32} : () -> i32

    // Constant emitted as `int32_t* = NULL;`
    %1 = "emitc.constant"()
        {value = #emitc.opaque<"NULL"> : !emitc.opaque<"int32_t*">}
        : () -> !emitc.opaque<"int32_t*">
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def EmitC_IncludeOp
    : EmitC_Op<"include", [NoSideEffect, HasParent<"ModuleOp">]> {
  let summary = "Include operation";
  let description = [{
    The `include` operation allows to define a source file inclusion via the
    `#include` directive.

    Example:

    ```mlir
    // Custom form defining the inclusion of `<myheader>`.
    emitc.include <"myheader.h">

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h", is_standard_include} : () -> ()

    // Custom form defining the inclusion of `"myheader"`.
    emitc.include "myheader.h"

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h"} : () -> ()
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "source file to include">:$include,
    UnitAttr:$is_standard_include
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // MLIR_DIALECT_EMITC_IR_EMITC
