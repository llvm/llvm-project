//===-- CIROps.td - CIR dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of the CIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_DIALECT_CIR_OPS
#define MLIR_CIR_DIALECT_CIR_OPS

include "mlir/Dialect/CIR/IR/CIRDialect.td"
include "mlir/Dialect/CIR/IR/CIRTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// CIR Ops
//===----------------------------------------------------------------------===//

class CIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<CIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def ConstantOp : CIR_Op<"cst",
    [ConstantLike, Pure]> {

  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = cir.cst 42 : i32
      %1 = cir.cst 4.2 : f32
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of AnyType.
  let results = (outs AnyType:$res);

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";

  let hasVerifier = 1;

  // TODO: hasFolder, etc
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

// FIXME: add alignment, bool attr on being param, automatic scope?.
def AllocaOp : CIR_Op<"alloca", []> {
  let summary = "define a local variable";
  let description = [{
    The `cir.alloca` operation defines a local variable. The `initial_value`
    can either be a unit attribute to represent a definition of an uninitialized
    local variable, or constant to represent the definition of a
    variable with an initial value. It can also be marked constant using the
    `constant` unit attribute.

    The result is a pointer type for the original input type.

    Example:

    ```mlir
    // Local variable with an initial value.
    %0 = cir.alloca i32 = 1, !cir.ptr<i32>

    // Uninitialized local variable.
    %0 = cir.alloca f32 = uninitialized, !cir.ptr<f32>

    // Constant local variable.
    %0 = cir.alloca constant i8 = 3, !cir.ptr<i8>
    ```
  }];

  let arguments = (ins
      TypeAttr:$type,
      AnyAttr:$initial_value,
      UnitAttr:$constant
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let assemblyFormat = [{
    (`constant` $constant^)?
    custom<AllocaOpTypeAndInitialValue>($type, $initial_value)
    attr-dict `,` `cir.ptr` type($addr)
  }];

  let extraClassDeclaration = [{
    bool isUninitialized() {
      return getInitialValue().isa<UnitAttr>();
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def LoadOp : CIR_Op<"load", [
  TypesMatchWith<"type of 'result' matches pointee type of 'addr'",
                 "addr", "result",
                 "$_self.cast<PointerType>().getPointee()">]> {

  let summary = "load operation";
  let description = [{
    `cir.load` reads a variable using a pointer type.

    Example:

    ```mlir

    // Read from local variable, address in %0.
    %1 = cir.load %0 : !cir.ptr<i32>, i32
    ```
  }];

  let arguments = (ins Arg<CIR_PointerType, "the address to load from",
                           [MemRead]>:$addr);
  let results = (outs AnyType:$result);

  let assemblyFormat =
      "$addr attr-dict `:` `cir.ptr` type($addr) `,` type($result)";
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def StoreOp : CIR_Op<"store", [
  TypesMatchWith<"type of 'value' matches pointee type of 'addr'",
                 "addr", "value",
                 "$_self.cast<PointerType>().getPointee()">]> {

  let summary = "store operation";
  let description = [{
    `cir.load` reads a variable using a pointer type.

    Example:

    ```mlir

    // Store to local variable, address in %0.
    cir.store %arg0, %0 : i32, !cir.ptr<i32>
  }];

  let arguments = (ins AnyType:$value,
                       Arg<CIR_PointerType, "the address to store the value",
                           [MemWrite]>:$addr);

  let assemblyFormat =
      "$value `,` $addr attr-dict `:` type($value) `,` `cir.ptr` type($addr)";
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : CIR_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> AnyType {
        ...
        cir.return %0 : AnyType
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<AnyType>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}


#endif // MLIR_CIR_DIALECT_CIR_OPS
