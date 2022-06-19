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

include "mlir/Interfaces/CastInterfaces.td"
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
  // TODO: Disallow to apply the operator & (address of) to emitc.constant operations.
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

def EmitC_CastOp : EmitC_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    SameOperandsAndResultShape
  ]> {
  let summary = "Cast operation";
  let description = [{
    The `cast` operation performs an explicit type conversion and is emitted
    as a C-style cast expression. It can be applied to integer, float, index
    and EmitC types.

    Example:

    ```mlir
    // Cast from `int32_t` to `float`
    %0 = emitc.cast %arg0: i32 to f32

    // Cast from `void` to `int32_t` pointer
    %1 = emitc.cast %arg1 :
        !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
    ```
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

def EmitC_ConstantOp : EmitC_Op<"constant", [ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `constant` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. Since folding is supported,
    it should not be used with pointers.

    Example:

    ```mlir
    // Integer constant
    %0 = "emitc.constant"(){value = 42 : i32} : () -> i32

    // Constant emitted as `char = CHAR_MIN;`
    %1 = "emitc.constant"()
        {value = #emitc.opaque<"CHAR_MIN"> : !emitc.opaque<"char">}
        : () -> !emitc.opaque<"char">
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let hasFolder = 1;
  // TODO: Disallow empty constants.
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

def EmitC_VariableOp : EmitC_Op<"variable", []> {
  let summary = "Variable operation";
  let description = [{
    The `variable` operation produces an SSA value equal to some value
    specified by an attribute. This can be used to form simple integer and
    floating point variables, as well as more exotic things like tensor
    variables. The `variable` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. If further supports the EmitC
    pointer type, whereas folding is not supported.
    The `variable` is emitted as a C/C++ local variable.

    Example:

    ```mlir
    // Integer variable
    %0 = "emitc.variable"(){value = 42 : i32} : () -> i32

    // Variable emitted as `int32_t* = NULL;`
    %1 = "emitc.variable"()
        {value = #emitc.opaque<"NULL"> : !emitc.opaque<"int32_t*">}
        : () -> !emitc.opaque<"int32_t*">
    ```

    Since folding is not supported, it can be used with pointers.
    As an example, it is valid to create pointers to `variable` operations
    by using `apply` operations and pass these to a `call` operation.
    ```mlir
    %0 = "emitc.variable"() {value = 0 : i32} : () -> i32
    %1 = "emitc.variable"() {value = 0 : i32} : () -> i32
    %2 = emitc.apply "&"(%0) : (i32) -> !emitc.ptr<i32>
    %3 = emitc.apply "&"(%1) : (i32) -> !emitc.ptr<i32>
    emitc.call "write"(%2, %3) : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> ()
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType);

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_EMITC_IR_EMITC
