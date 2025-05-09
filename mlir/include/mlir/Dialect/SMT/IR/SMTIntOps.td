//===- SMTIntOps.td - SMT dialect int theory operations ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTINTOPS_TD
#define MLIR_DIALECT_SMT_IR_SMTINTOPS_TD

include "mlir/Dialect/SMT/IR/SMTDialect.td"
include "mlir/Dialect/SMT/IR/SMTAttributes.td"
include "mlir/Dialect/SMT/IR/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SMTIntOp<string mnemonic, list<Trait> traits = []> :
  SMTOp<"int." # mnemonic, traits>;

def IntConstantOp : SMTIntOp<"constant", [
  Pure,
  ConstantLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "produce a constant (infinite-precision) integer";
  let description = [{
    This operation represents (infinite-precision) integer literals of the `Int`
    sort. The set of values for the sort `Int` consists of all numerals and
    all terms of the form `-n`where n is a numeral other than 0. For more
    information refer to the 
    [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.7 standard.
  }];

  let arguments = (ins APIntAttr:$value);
  let results = (outs IntType:$result);

  let hasCustomAssemblyFormat = true;
  let hasFolder = true;
}

class VariadicIntOp<string mnemonic> : SMTIntOp<mnemonic, [Pure, Commutative]> {
  let description = [{
    This operation represents (infinite-precision) }] # summary # [{.
    The semantics are equivalent to the corresponding operator described in
    the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.7 standard.
  }];

  let arguments = (ins Variadic<IntType>:$inputs);
  let results = (outs IntType:$result);
  let assemblyFormat = "$inputs attr-dict";

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$inputs), [{
      build($_builder, $_state, $_builder.getType<smt::IntType>(), inputs);
    }]>,
  ];
}

class BinaryIntOp<string mnemonic> : SMTIntOp<mnemonic, [Pure]> {
  let description = [{
    This operation represents (infinite-precision) }] # summary # [{.
    The semantics are equivalent to the corresponding operator described in
    the [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.7 standard.
  }];

  let arguments = (ins IntType:$lhs, IntType:$rhs);
  let results = (outs IntType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def IntAbsOp : SMTIntOp<"abs", [Pure]> {
  let summary = "the absolute value of an Int";
  let description = [{
    This operation represents the absolute value function for the `Int` sort.
    The semantics are equivalent to the `abs` operator as described in the
    [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.7 standard.
  }];

  let arguments = (ins IntType:$input);
  let results = (outs IntType:$result);
  let assemblyFormat = "$input attr-dict";
}

def IntAddOp : VariadicIntOp<"add"> { let summary = "integer addition"; }
def IntMulOp : VariadicIntOp<"mul"> { let summary = "integer multiplication"; }
def IntSubOp : BinaryIntOp<"sub"> { let summary = "integer subtraction"; }
def IntDivOp : BinaryIntOp<"div"> { let summary = "integer division"; }
def IntModOp : BinaryIntOp<"mod"> { let summary = "integer remainder"; }

def IntPredicateLT : I64EnumAttrCase<"lt", 0>;
def IntPredicateLE : I64EnumAttrCase<"le", 1>;
def IntPredicateGT : I64EnumAttrCase<"gt", 2>;
def IntPredicateGE : I64EnumAttrCase<"ge", 3>;
let cppNamespace = "mlir::smt" in
def IntPredicate : I64EnumAttr<
    "IntPredicate",
    "smt comparison predicate for integers",
    [IntPredicateLT, IntPredicateLE, IntPredicateGT, IntPredicateGE]>;

def IntCmpOp : SMTIntOp<"cmp", [Pure]> {
  let summary = "integer comparison";
  let description = [{
    This operation represents the comparison of (infinite-precision) integers.
    The semantics are equivalent to the `<= (le)`, `< (lt)`, `>= (ge)`, or
    `> (gt)` operator depending on the predicate (indicated in parentheses) as
    described in the
    [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.7 standard.
  }];

  let arguments = (ins IntPredicate:$pred, IntType:$lhs, IntType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$pred $lhs `,` $rhs attr-dict";
}

def Int2BVOp : SMTOp<"int2bv", [Pure]> {
  let summary = "Convert an integer to an inferred-width bitvector.";
  let description = [{
    Designed to lower directly to an operation of the same name in Z3. The Z3
    C API describes the semantics as follows:
    Create an n bit bit-vector from the integer argument t1.
    The resulting bit-vector has n bits, where the i'th bit (counting from 0
    to n-1) is 1 if (t1 div 2^i) mod 2 is 1.
    The node t1 must have integer sort.
  }];
  let arguments = (ins IntType:$input);
  let results = (outs BitVectorType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";
}

#endif // MLIR_DIALECT_SMT_IR_SMTINTOPS_TD
