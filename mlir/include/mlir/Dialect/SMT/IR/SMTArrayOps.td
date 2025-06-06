//===- SMTArrayOps.td - SMT array operations ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTARRAYOPS_TD
#define MLIR_DIALECT_SMT_IR_SMTARRAYOPS_TD

include "mlir/Dialect/SMT/IR/SMTDialect.td"
include "mlir/Dialect/SMT/IR/SMTAttributes.td"
include "mlir/Dialect/SMT/IR/SMTTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SMTArrayOp<string mnemonic, list<Trait> traits = []> :
  SMTOp<"array." # mnemonic, traits>;

def ArrayStoreOp : SMTArrayOp<"store", [
  Pure,
  TypesMatchWith<"summary", "array", "index", 
                 "cast<ArrayType>($_self).getDomainType()">,
  TypesMatchWith<"summary", "array", "value",
                 "cast<ArrayType>($_self).getRangeType()">,
  AllTypesMatch<["array", "result"]>,
]> {
  let summary = "stores a value at a given index and returns the new array";
  let description = [{
    This operation returns a new array which is the same as the 'array' operand
    except that the value at the given 'index' is changed to the given 'value'.
    The semantics are equivalent to the 'store' operator described in the
    [SMT ArrayEx theory](https://smtlib.cs.uiowa.edu/Theories/ArraysEx.smt2) of
    the SMT-LIB standard 2.7.
  }];

  let arguments = (ins ArrayType:$array, AnySMTType:$index, AnySMTType:$value);
  let results = (outs ArrayType:$result);

  let assemblyFormat = [{
    $array `[` $index `]` `,` $value attr-dict `:` qualified(type($array))
  }];
}

def ArraySelectOp : SMTArrayOp<"select", [
  Pure,
  TypesMatchWith<"summary", "array", "index",
                 "cast<ArrayType>($_self).getDomainType()">,
  TypesMatchWith<"summary", "array", "result",
                 "cast<ArrayType>($_self).getRangeType()">,
]> {
  let summary = "get the value stored in the array at the given index";
  let description = [{
    This operation is retuns the value stored in the given array at the given
    index. The semantics are equivalent to the `select` operator defined in the
    [SMT ArrayEx theory](https://smtlib.cs.uiowa.edu/Theories/ArraysEx.smt2) of
    the SMT-LIB standard 2.7.
  }];

  let arguments = (ins ArrayType:$array, AnySMTType:$index);
  let results = (outs AnySMTType:$result);

  let assemblyFormat = [{
    $array `[` $index `]` attr-dict `:` qualified(type($array))
  }];
}

def ArrayBroadcastOp : SMTArrayOp<"broadcast", [
  Pure,
  TypesMatchWith<"summary", "result", "value",
                 "cast<ArrayType>($_self).getRangeType()">,
]> {
  let summary = "construct an array with the given value stored at every index";
  let description = [{
    This operation represents a broadcast of the 'value' operand to all indices
    of the array. It is equivalent to
    ```
    %0 = smt.declare_fun "array" : !smt.array<[!smt.int -> !smt.bool]>
    %1 = smt.forall ["idx"] {
    ^bb0(%idx: !smt.int):
      %2 = smt.array.select %0[%idx] : !smt.array<[!smt.int -> !smt.bool]>
      %3 = smt.eq %value, %2 : !smt.bool
      smt.yield %3 : !smt.bool
    }
    smt.assert %1
    // return %0
    ```

    In SMT-LIB, this is frequently written as
    `((as const (Array Int Bool)) value)`.
  }];

  let arguments = (ins AnySMTType:$value);
  let results = (outs ArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
}

#endif // MLIR_DIALECT_SMT_IR_SMTARRAYOPS_TD
