//===- UBOps.td - UB operations definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UB_IR_UBOPS_TD
#define MLIR_DIALECT_UB_IR_UBOPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/Dialect/UB/IR/UBOpsInterfaces.td"

def UB_Dialect : Dialect {
  let name = "ub";
  let cppNamespace = "::mlir::ub";

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class for UB dialect attributes.
class UB_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
    AttrDef<UB_Dialect, name, traits> {
  let cppIncludes = ["mlir/Dialect/UB/IR/UBOps.h"];
  let mnemonic = attrMnemonic;
}

// Base class for UB dialect ops.
class UB_Op<string mnemonic, list<Trait> traits = []> :
    Op<UB_Dialect, mnemonic, traits>;

def PoisonAttr : UB_Attr<"Poison", "poison", [PoisonAttrInterface]> {
}

//===----------------------------------------------------------------------===//
// AssumeOp
//===----------------------------------------------------------------------===//

def AssumeOp : UB_Op<"assume", [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "Refine an SSA value with an assumed predicate";
  let description = [{
    Returns `value` if `predicate` holds for `value` and `predicateOperands`,
    and poison otherwise. Poison in the predicated value propagates to the
    result. The predicate defines how poison in additional operands is handled.
    Executing this operation does not itself cause immediate undefined behavior,
    even if the predicate is false.

    The predicate holds for the result only. It does not constrain other uses
    of `value`. Data flow analyses may use the predicate to reason about
    properties of the result.

    The predicate attribute must implement `AssumePredicateAttrInterface`,
    which verifies the predicated value type and the number and types of
    additional operands.

    Example:

    ```mlir
    %positive = ub.assume %x, #arith.assume_cmpi<sgt, 0 : i32>() : i32
    ```
  }];

  let arguments = (ins AnyType:$value,
                       AssumePredicateAttrInterface:$predicate,
                       Variadic<AnyType>:$predicateOperands);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $value `,` $predicate `(`
    ($predicateOperands^ `:` type($predicateOperands))? `)`
    attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PoisonOp
//===----------------------------------------------------------------------===//

def PoisonOp : UB_Op<"poison", [ConstantLike, Pure]> {
  let summary = "Poisoned constant operation.";
  let description = [{
    The `poison` operation materializes a compile-time poisoned constant value
    to indicate deferred undefined behavior.
    `value` attribute is needed to indicate an optional additional poison
    semantics (e.g. partially poisoned vectors), default value indicates results
    is fully poisoned.

    Examples:

    ```
    // Short form
    %0 = ub.poison : i32
    // Long form
    %1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
    ```
  }];

  let arguments = (ins DefaultValuedAttr<PoisonAttrInterface, "{}">:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict (`<` $value^ `>`)? `:` type($result)";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

def UnreachableOp : UB_Op<"unreachable", [Terminator]> {
  let summary = "Unreachable operation.";
  let description = [{
    The `unreachable` operation triggers immediate undefined behavior if
    executed.

    Example:

    ```
    ub.unreachable
    ```
  }];

  let assemblyFormat = "attr-dict";
}

#endif // MLIR_DIALECT_UB_IR_UBOPS_TD
