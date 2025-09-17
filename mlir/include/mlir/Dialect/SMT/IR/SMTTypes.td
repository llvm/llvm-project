//===- SMTTypes.td - SMT dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTTYPES_TD
#define MLIR_DIALECT_SMT_IR_SMTTYPES_TD

include "mlir/Dialect/SMT/IR/SMTDialect.td"
include "mlir/IR/AttrTypeBase.td"

class SMTTypeDef<string name> : TypeDef<SMTDialect, name> { }

def BoolType : SMTTypeDef<"Bool"> {
  let mnemonic = "bool";
  let assemblyFormat = "";
}

def IntType : SMTTypeDef<"Int"> {
  let mnemonic = "int";
  let description = [{
    This type represents the `Int` sort as described in the
    [SMT Ints theory](https://smtlib.cs.uiowa.edu/Theories/Ints.smt2) of the
    SMT-LIB 2.7 standard.
  }];
  let assemblyFormat = "";
}

def BitVectorType : SMTTypeDef<"BitVector"> {
  let mnemonic = "bv";
  let description = [{
    This type represents the `(_ BitVec width)` sort as described in the
    [SMT bit-vector
    theory](https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml).

    The bit-width must be strictly greater than zero.
  }];

  let parameters = (ins "int64_t":$width);
  let assemblyFormat = "`<` $width `>`";

  let genVerifyDecl = true;
}

def ArrayType : SMTTypeDef<"Array"> {
  let mnemonic = "array";
  let description = [{
    This type represents the `(Array X Y)` sort, where X and Y are any
    sort/type, as described in the
    [SMT ArrayEx theory](https://smtlib.cs.uiowa.edu/Theories/ArraysEx.smt2) of
    the SMT-LIB standard 2.7.
  }];

  let parameters = (ins "mlir::Type":$domainType, "mlir::Type":$rangeType);
  let assemblyFormat = "`<` `[` $domainType `->` $rangeType `]` `>`";

  let genVerifyDecl = true;
}

def SMTFuncType : SMTTypeDef<"SMTFunc"> {
  let mnemonic = "func";
  let description = [{
    This type represents the SMT function sort as described in the
    [SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
    It is part of the language itself rather than a theory or logic.

    A function in SMT can have an arbitrary domain size, but always has exactly
    one range sort.

    Since SMT only supports first-order logic, it is not possible to nest
    function types.

    Example: `!smt.func<(!smt.bool, !smt.int) !smt.bool>` is equivalent to
    `((Bool Int) Bool)` in SMT-LIB.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::Type", "domain types">:$domainTypes,
    "mlir::Type":$rangeType
  );

  // Note: We are not printing the parentheses when no domain type is present
  // because the default MLIR parser thinks it is a builtin function type
  // otherwise.
  let assemblyFormat = "`<` `(` $domainTypes `)` ` ` $rangeType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "llvm::ArrayRef<mlir::Type>":$domainTypes,
      "mlir::Type":$rangeType), [{
      return $_get(rangeType.getContext(), domainTypes, rangeType);
    }]>,
    TypeBuilderWithInferredContext<(ins "mlir::Type":$rangeType), [{
      return $_get(rangeType.getContext(),
                   llvm::ArrayRef<mlir::Type>{}, rangeType);
    }]>
  ];

  let genVerifyDecl = true;
}

def SortType : SMTTypeDef<"Sort"> {
  let mnemonic = "sort";
  let description = [{
    This type represents uninterpreted sorts. The usage of a type like
    `!smt.sort<"sort_name"[!smt.bool, !smt.sort<"other_sort">]>` implies a
    `declare-sort sort_name 2` and a `declare-sort other_sort 0` in SMT-LIB.
    This type represents concrete use-sites of such declared sorts, in this
    particular case it would be equivalent to `(sort_name Bool other_sort)` in
    SMT-LIB. More details about the semantics can be found in the
    [SMT-LIB 2.7 standard](https://smt-lib.org/papers/smt-lib-reference-v2.7-r2025-02-05.pdf).
  }];

  let parameters = (ins
    "mlir::StringAttr":$identifier,
    OptionalArrayRefParameter<"mlir::Type", "sort parameters">:$sortParams
  );

  let assemblyFormat = "`<` $identifier (`[` $sortParams^ `]`)? `>`";

  let builders = [
    TypeBuilder<(ins "llvm::StringRef":$identifier,
                     "llvm::ArrayRef<mlir::Type>":$sortParams), [{
      return $_get($_ctxt, mlir::StringAttr::get($_ctxt, identifier),
                           sortParams);
    }]>,
    TypeBuilder<(ins "llvm::StringRef":$identifier), [{
      return $_get($_ctxt, mlir::StringAttr::get($_ctxt, identifier),
                    llvm::ArrayRef<mlir::Type>{});
    }]>,
  ];

  let genVerifyDecl = true;
}

def AnySMTType : Type<CPred<"smt::isAnySMTValueType($_self)">,
                      "any SMT value type">;
def AnyNonFuncSMTType : Type<CPred<"smt::isAnyNonFuncSMTValueType($_self)">,
                             "any non-function SMT value type">;
def AnyNonSMTType : Type<Neg<AnySMTType.predicate>, "any non-smt type">;

#endif // MLIR_DIALECT_SMT_IR_SMTTYPES_TD
