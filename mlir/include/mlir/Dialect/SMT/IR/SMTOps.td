//===- SMTOps.td - SMT dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTOPS_TD
#define MLIR_DIALECT_SMT_IR_SMTOPS_TD

include "mlir/Dialect/SMT/IR/SMTDialect.td"
include "mlir/Dialect/SMT/IR/SMTAttributes.td"
include "mlir/Dialect/SMT/IR/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

class SMTOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, mnemonic, traits>;

def DeclareFunOp : SMTOp<"declare_fun", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "declare a symbolic value of a given sort";
  let description = [{
    This operation declares a symbolic value just as the `declare-const` and
    `declare-fun` statements in SMT-LIB 2.7. The result type determines the SMT
    sort of the symbolic value. The returned value can then be used to refer to
    the symbolic value instead of using the identifier like in SMT-LIB.

    The optionally provided string will be used as a prefix for the newly
    generated identifier (useful for easier readability when exporting to
    SMT-LIB). Each `declare` will always provide a unique new symbolic value
    even if the identifier strings are the same.

    Note that there does not exist a separate operation equivalent to
    SMT-LIBs `define-fun` since
    ```
    (define-fun f (a Int) Int (-a))
    ```
    is only syntactic sugar for
    ```
    %f = smt.declare_fun : !smt.func<(!smt.int) !smt.int>
    %0 = smt.forall {
    ^bb0(%arg0: !smt.int):
      %1 = smt.apply_func %f(%arg0) : !smt.func<(!smt.int) !smt.int>
      %2 = smt.int.neg %arg0
      %3 = smt.eq %1, %2 : !smt.int
      smt.yield %3 : !smt.bool
    }
    smt.assert %0
    ```

    Note that this operation cannot be marked as Pure since two operations (even
    with the same identifier string) could then be CSEd, leading to incorrect
    behavior.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$namePrefix);
  let results = (outs Res<AnySMTType, "a symbolic value", [MemAlloc]>:$result);

  let assemblyFormat = [{
    ($namePrefix^)? attr-dict `:` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "mlir::Type":$type), [{
      build($_builder, $_state, type, nullptr);
    }]>
  ];
}

def BoolConstantOp : SMTOp<"constant", [
  Pure,
  ConstantLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "Produce a constant boolean";
  let description = [{
    Produces the constant expressions 'true' and 'false' as described in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
    Standard 2.7.
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$value attr-dict";

  let hasFolder = true;
}

def SolverOp : SMTOp<"solver", [
  IsolatedFromAbove,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let summary = "create a solver instance within a lifespan";
  let description = [{
    This operation defines an SMT context with a solver instance. SMT operations
    are only valid when being executed between the start and end of the region
    of this operation. Any invocation outside is undefined. However, they do not
    have to be direct children of this operation. For example, it is allowed to
    have SMT operations in a `func.func` which is only called from within this
    region. No SMT value may enter or exit the lifespan of this region (such
    that no value created from another SMT context can be used in this scope and
    the solver can deallocate all state required to keep track of SMT values at
    the end).

    As a result, the region is comparable to an entire SMT-LIB script, but
    allows for concrete operations and control-flow. Concrete values may be
    passed in and returned to influence the computations after the `smt.solver`
    operation.

    Example:
    ```mlir
    %0:2 = smt.solver (%in) {smt.some_attr} : (i8) -> (i8, i32) {
    ^bb0(%arg0: i8):
      %c = smt.declare_fun "c" : !smt.bool
      smt.assert %c
      %1 = smt.check sat {
        %c1_i32 = arith.constant 1 : i32
        smt.yield %c1_i32 : i32
      } unknown {
        %c0_i32 = arith.constant 0 : i32
        smt.yield %c0_i32 : i32
      } unsat {
        %c-1_i32 = arith.constant -1 : i32
        smt.yield %c-1_i32 : i32
      } -> i32
      smt.yield %arg0, %1 : i8, i32
    }
    ```
  }];

  let arguments = (ins Variadic<AnyNonSMTType>:$inputs);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let results = (outs Variadic<AnyNonSMTType>:$results);

  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, $results) $bodyRegion
  }];

  let hasRegionVerifier = true;
}

def SetLogicOp : SMTOp<"set_logic", [
  HasParent<"smt::SolverOp">,
]> {
  let summary = "set the logic for the SMT solver";
  let arguments = (ins StrAttr:$logic);
  let assemblyFormat = "$logic attr-dict";
}

def AssertOp : SMTOp<"assert", []> {
  let summary = "assert that a boolean expression holds";
  let arguments = (ins BoolType:$input);
  let assemblyFormat = "$input attr-dict";
}

def ResetOp : SMTOp<"reset", []> {
  let summary = "reset the solver";
  let assemblyFormat = "attr-dict";
}

def PushOp : SMTOp<"push", []> {
  let summary = "push a given number of levels onto the assertion stack";
  let arguments = (ins ConfinedAttr<I32Attr, [IntNonNegative]>:$count);
  let assemblyFormat = "$count attr-dict";
}

def PopOp : SMTOp<"pop", []> {
  let summary = "pop a given number of levels from the assertion stack";
  let arguments = (ins ConfinedAttr<I32Attr, [IntNonNegative]>:$count);
  let assemblyFormat = "$count attr-dict";
}

def CheckOp : SMTOp<"check", [
  NoRegionArguments,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let summary = "check if the current set of assertions is satisfiable";
  let description = [{
    This operation checks if all the assertions in the solver defined by the
    nearest ancestor operation of type `smt.solver` are consistent. The outcome
    an be 'satisfiable', 'unknown', or 'unsatisfiable' and the corresponding
    region will be executed. It is the corresponding construct to the
    `check-sat` in SMT-LIB.

    Example:
    ```mlir
    %0 = smt.check sat {
      %c1_i32 = arith.constant 1 : i32
      smt.yield %c1_i32 : i32
    } unknown {
      %c0_i32 = arith.constant 0 : i32
      smt.yield %c0_i32 : i32
    } unsat {
      %c-1_i32 = arith.constant -1 : i32
      smt.yield %c-1_i32 : i32
    } -> i32
    ```
  }];

  let regions = (region SizedRegion<1>:$satRegion,
                        SizedRegion<1>:$unknownRegion,
                        SizedRegion<1>:$unsatRegion);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    attr-dict `sat` $satRegion `unknown` $unknownRegion `unsat` $unsatRegion
    (`->` qualified(type($results))^ )?
  }];

  let hasRegionVerifier = true;
}

def YieldOp : SMTOp<"yield", [
  Pure,
  Terminator,
  ReturnLike,
]> {
  let summary = "terminator operation for various regions of SMT operations";
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` qualified(type($values)))? attr-dict";
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, {});
  }]>];
}

def ApplyFuncOp : SMTOp<"apply_func", [
  Pure,
  TypesMatchWith<"summary", "func", "result",
                 "cast<SMTFuncType>($_self).getRangeType()">,
  RangedTypesMatchWith<"summary", "func", "args",
                       "cast<SMTFuncType>($_self).getDomainTypes()">
]> {
  let summary = "apply a function";
  let description = [{
    This operation performs a function application as described in the
    [SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
    It is part of the language itself rather than a theory or logic.
  }];

  let arguments = (ins SMTFuncType:$func,
                       Variadic<AnyNonFuncSMTType>:$args);
  let results = (outs AnyNonFuncSMTType:$result);

  let assemblyFormat = [{
    $func `(` $args `)` attr-dict `:` qualified(type($func))
  }];
}

def EqOp : SMTOp<"eq", [Pure, SameTypeOperands]> {
  let summary = "returns true iff all operands are identical";
  let description = [{
    This operation compares the operands and returns true iff all operands are
    identical. The semantics are equivalent to the `=` operator defined in the
    SMT-LIB Standard 2.7 in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).

    Any SMT sort/type is allowed for the operands and it supports a variadic
    number of operands, but requires at least two. This is because the `=`
    operator is annotated with `:chainable` which means that `= a b c d` is
    equivalent to `and (= a b) (= b c) (= c d)` where `and` is annotated
    `:left-assoc`, i.e., it can be further rewritten to
    `and (and (= a b) (= b c)) (= c d)`.
  }];

  let arguments = (ins Variadic<AnyNonFuncSMTType>:$inputs);
  let results = (outs BoolType:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, ValueRange{lhs, rhs});
    }]>
  ];

  let hasCustomAssemblyFormat = true;
  let hasVerifier = true;
}

def DistinctOp : SMTOp<"distinct", [Pure, SameTypeOperands]> {
  let summary = "returns true iff all operands are not identical to any other";
  let description = [{
    This operation compares the operands and returns true iff all operands are
    not identical to any of the other operands. The semantics are equivalent to
    the `distinct` operator defined in the SMT-LIB Standard 2.7 in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).

    Any SMT sort/type is allowed for the operands and it supports a variadic
    number of operands, but requires at least two. This is because the
    `distinct` operator is annotated with `:pairwise` which means that
    `distinct a b c d` is equivalent to
    ```
    and (distinct a b) (distinct a c) (distinct a d)
        (distinct b c) (distinct b d)
        (distinct c d)
    ```
    where `and` is annotated `:left-assoc`, i.e., it can be further rewritten to
    ```
    (and (and (and (and (and (distinct a b)
                             (distinct a c))
                        (distinct a d))
                   (distinct b c))
              (distinct b d))
         (distinct c d)
    ```
  }];

  let arguments = (ins Variadic<AnyNonFuncSMTType>:$inputs);
  let results = (outs BoolType:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, ValueRange{lhs, rhs});
    }]>
  ];

  let hasCustomAssemblyFormat = true;
  let hasVerifier = true;
}

def IteOp : SMTOp<"ite", [
  Pure,
  AllTypesMatch<["thenValue", "elseValue", "result"]>
]> {
  let summary = "an if-then-else function";
  let description = [{
    This operation returns its second operand or its third operand depending on
    whether its first operand is true or not. The semantics are equivalent to
    the `ite` operator defined in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
    2.7 standard.
  }];

  let arguments = (ins BoolType:$cond,
                       AnySMTType:$thenValue,
                       AnySMTType:$elseValue);
  let results = (outs AnySMTType:$result);

  let assemblyFormat = [{
    $cond `,` $thenValue `,` $elseValue attr-dict `:` qualified(type($result))
  }];
}

def NotOp : SMTOp<"not", [Pure]> {
  let summary = "a boolean negation";
  let description = [{
    This operation performs a boolean negation. The semantics are equivalent to
    the 'not' operator in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
    Standard 2.7.
  }];

  let arguments = (ins BoolType:$input);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$input attr-dict";
}

class VariadicBoolOp<string mnemonic, string desc> : SMTOp<mnemonic, [Pure]> {
  let summary = desc;
  let description = "This operation performs " # desc # [{.
    The semantics are equivalent to the '}] # mnemonic # [{' operator in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2).
    of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    This is because the operator is annotated with the `:left-assoc` attribute
    which means that `op a b c` is equivalent to `(op (op a b) c)`.
  }];

  let arguments = (ins Variadic<BoolType>:$inputs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$inputs attr-dict";

  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs), [{
      build($_builder, $_state, ValueRange{lhs, rhs});
    }]>
  ];
}

def AndOp : VariadicBoolOp<"and", "a boolean conjunction">;
def OrOp  : VariadicBoolOp<"or", "a boolean disjunction">;
def XOrOp : VariadicBoolOp<"xor", "a boolean exclusive OR">;

def ImpliesOp : SMTOp<"implies", [Pure]> {
  let summary = "boolean implication";
  let description = [{
    This operation performs a boolean implication. The semantics are equivalent
    to the '=>' operator in the
    [Core theory](https://smtlib.cs.uiowa.edu/Theories/Core.smt2) of the SMT-LIB
    Standard 2.7.
  }];

  let arguments = (ins BoolType:$lhs, BoolType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

class QuantifierOp<string mnemonic> : SMTOp<mnemonic, [
  RecursivelySpeculatable,
  RecursiveMemoryEffects,
  SingleBlockImplicitTerminator<"smt::YieldOp">,
]> {
  let description = [{
    This operation represents the }] # summary # [{ as described in the
    [SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
    It is part of the language itself rather than a theory or logic.

    The operation specifies the name prefixes (as an optional attribute) and
    types (as the types of the block arguments of the regions) of bound
    variables that may be used in the 'body' of the operation. If a 'patterns'
    region is specified, the block arguments must match the ones of the 'body'
    region and (other than there) must be used at least once in the 'patterns'
    region. It may also not contain any operations that bind variables, such as
    quantifiers. While the 'body' region must always yield exactly one
    `!smt.bool`-typed value, the 'patterns' region can yield an arbitrary number
    (but at least one) of SMT values.

    The bound variables can be any SMT type except of functions, since SMT only
    supports first-order logic.

    The 'no_patterns' attribute is only allowed when no 'patterns' region is
    specified and forbids the solver to generate and use patterns for this
    quantifier.

    The 'weight' attribute indicates the importance of this quantifier being
    instantiated compared to other quantifiers that may be present. The default
    value is zero.

    Both the 'no_patterns' and 'weight' attributes are annotations to the
    quantifiers body term. Annotations and attributes are described in the
    standard in sections 3.4, and 3.6 (specifically 3.6.5). SMT-LIB allows
    adding custom attributes to provide solvers with additional metadata, e.g.,
    hints such as above mentioned attributes. They are not part of the standard
    themselves, but supported by common SMT solvers (e.g., Z3). 
  }];

  let arguments = (ins DefaultValuedAttr<I32Attr, "0">:$weight,
                       UnitAttr:$noPattern,
                       OptionalAttr<StrArrayAttr>:$boundVarNames);
  let regions = (region SizedRegion<1>:$body,
                        VariadicRegion<SizedRegion<1>>:$patterns);
  let results = (outs BoolType:$result);

  let builders = [
    OpBuilder<(ins
      "TypeRange":$boundVarTypes,
      "function_ref<Value(OpBuilder &, Location, ValueRange)>":$bodyBuilder,
      CArg<"std::optional<ArrayRef<StringRef>>", "std::nullopt">:$boundVarNames,
      CArg<"function_ref<ValueRange(OpBuilder &, Location, ValueRange)>",
           "{}">:$patternBuilder,
      CArg<"uint32_t", "0">:$weight,
      CArg<"bool", "false">:$noPattern)>
  ];
  let skipDefaultBuilders = true;

  let assemblyFormat = [{
    ($boundVarNames^)? (`no_pattern` $noPattern^)? (`weight` $weight^)?
    attr-dict-with-keyword $body (`patterns` $patterns^)?
  }];

  let hasVerifier = true;
  let hasRegionVerifier = true;
}

def ForallOp : QuantifierOp<"forall"> { let summary = "forall quantifier"; }
def ExistsOp : QuantifierOp<"exists"> { let summary = "exists quantifier"; }

#endif // MLIR_DIALECT_SMT_IR_SMTOPS_TD
