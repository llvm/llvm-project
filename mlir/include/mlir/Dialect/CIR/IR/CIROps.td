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
include "mlir/Dialect/CIR/IR/CIRAttrs.td"

include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
      %0 = cir.cst(42 : i32)
      %1 = cir.cst(4.2 : f32)
      %2 = cir.cst(nullptr : !cir.ptr<i32>)
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins TypedAttrInterface:$value);

  // The constant operation returns a single value of AnyType.
  let results = (outs AnyType:$res);

  let assemblyFormat = "`(` custom<ConstantValue>($value) `)` attr-dict `:` type($res)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
     bool isNullPtr() {
       return getValue().isa<UnitAttr>();
     }
  }];

  // TODO: hasFolder, etc
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

def InitStyle_None      : I32EnumAttrCase<"uninitialized", 1>;
def InitStyle_ParamInit : I32EnumAttrCase<"paraminit", 2>;
def InitStyle_CInit     : I32EnumAttrCase<"cinit", 3>;
def InitStyle_CallInit  : I32EnumAttrCase<"callinit", 4>;
def InitStyle_ListInit  : I32EnumAttrCase<"listinit", 5>;

def InitStyle : I32EnumAttr<
    "InitStyle",
    "variable initialization style",
    [InitStyle_None, InitStyle_ParamInit,
     InitStyle_CInit, InitStyle_CallInit,
     InitStyle_ListInit]> {
  let cppNamespace = "::mlir::cir";
}

class AllocaTypesMatchWith<string summary, string lhsArg, string rhsArg,
                     string transform, string comparator = "std::equal_to<>()">
  : PredOpTrait<summary, CPred<
      comparator # "(" #
      !subst("$_self", "$" # lhsArg # ".getType()", transform) #
             ", $" # rhsArg # ")">> {
  string lhs = lhsArg;
  string rhs = rhsArg;
  string transformer = transform;
}

// FIXME: add alignment, bool attr on being param, automatic scope?.
def AllocaOp : CIR_Op<"alloca", [
  AllocaTypesMatchWith<"'type' matches pointee type of 'addr'",
                 "addr", "type",
                 "$_self.cast<PointerType>().getPointee()">]> {
  let summary = "local variable";
  let description = [{
    The `cir.alloca` operation defines a local variable.

    Possible initialization styles are: uninitialized, paraminit,
    callinit, cinit and listinit.

    The result is a pointer type for the original input type.

    Example:

    ```mlir
    // Local variable with uninitialized value.
    %0 = cir.alloca i32, !cir.ptr<i32>, [cinit]
    ```
  }];

  let arguments = (ins
    TypeAttr:$type,
    // FIXME: add "uninitialzed" as default mode
    Arg<InitStyle, "initialization style">:$init
  );

  let results = (outs Res<CIR_PointerType, "",
                      [MemAlloc<AutomaticAllocationScopeResource>]>:$addr);

  let assemblyFormat = [{
    $type `,` `cir.ptr` type($addr) `,` `[` $init `]`  attr-dict
  }];

  let hasVerifier = 0;
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
                           [MemRead]>:$addr,
                       UnitAttr:$conv);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $addr (`lvalue_to_rvalue` $conv^)? attr-dict `:` `cir.ptr` type($addr) `,` type($result)
  }];
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
