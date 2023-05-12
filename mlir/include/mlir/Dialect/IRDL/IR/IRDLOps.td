//===- IRDLOps.td - IR Definition Language Dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the IRDL dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLOPS
#define MLIR_DIALECT_IRDL_IR_IRDLOPS

include "IRDL.td"
include "IRDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class IRDL_Op<string mnemonic, list<Trait> traits = []>
    : Op<IRDL_Dialect, mnemonic, traits>;

class AtMostOneChildOf<string op> : ParamNativeOpTrait<"AtMostOneChildOf", op>;

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def IRDL_DialectOp : IRDL_Op<"dialect",
    [IsolatedFromAbove, NoTerminator, Symbol, SymbolTable]> {
  let summary = "Define a new dialect";
  let description = [{
    The `irdl.dialect` operation defines a dialect. All operations, attributes,
    and types defined inside its region will be part of the dialect.

    Example:

    ```mlir
    irdl.dialect @cmath {
      ...
    }
    ```

    The above program defines a `cmath` dialect.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
    "$sym_name attr-dict-with-keyword custom<SingleBlockRegion>($body)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Type and Attribute definition
//===----------------------------------------------------------------------===//

def IRDL_TypeOp : IRDL_Op<"type",
    [HasParent<"DialectOp">, NoTerminator, NoRegionArguments,
     AtMostOneChildOf<"ParametersOp">, Symbol]> {
  let summary = "Define a new type";
  let description = [{
    `irdl.type` defines a new type belonging to the `irdl.dialect` parent.

    The type parameters can be defined with an `irdl.parameters` operation in
    the optional region.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is i32
        %1 = irdl.is i64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
    "$sym_name attr-dict-with-keyword custom<SingleBlockRegion>($body)";
}

def IRDL_AttributeOp : IRDL_Op<"attribute",
    [HasParent<"DialectOp">, NoTerminator, NoRegionArguments,
     AtMostOneChildOf<"ParametersOp">, Symbol]> {
  let summary = "Define a new attribute";
  let description = [{
    `irdl.attribute` defines a new attribute belonging to the `irdl.dialect`
    parent.

    The attribute parameters can be defined with an `irdl.parameters` operation
    in the optional region.

    Example:

    ```mlir
    irdl.dialect @testd {
      irdl.attribute @enum_attr {
        %0 = irdl.is "foo"
        %1 = irdl.is "bar"
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```

    The above program defines an `enum_attr` attribute inside the `testd`
    dialect. The attribute has one `StringAttr` parameter that should be
    either a `"foo"` or a `"bar"`.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
    "$sym_name attr-dict-with-keyword custom<SingleBlockRegion>($body)";
}

def IRDL_ParametersOp : IRDL_Op<"parameters",
    [ParentOneOf<["AttributeOp", "TypeOp"]>]> {
  let summary =
    "Define the constraints on parameters of a type/attribute definition";
  let description = [{
    `irdl.parameters` defines the constraints on parameters of a type or
    attribute definition.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.is i32
        %1 = irdl.is i64
        %2 = irdl.any_of(%0, %1)
        irdl.parameters(%2)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath`. The
    type has a single parameter that should be either `i32` or `i64`.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args);
  let assemblyFormat = " `(` $args `)` attr-dict ";
}

//===----------------------------------------------------------------------===//
// IRDL Operation definition
//===----------------------------------------------------------------------===//

def IRDL_OperationOp : IRDL_Op<"operation",
    [HasParent<"DialectOp">, NoTerminator, NoRegionArguments,
    AtMostOneChildOf<"OperandsOp, ResultsOp">, Symbol]> {
  let summary = "Define a new operation";
  let description = [{
    `irdl.operation` defines a new operation belonging to the `irdl.dialect`
    parent.

    Operations can define constraints on their operands and results with the
    `irdl.results` and `irdl.operands` operations. If these operations are not
    present in the region, the results or operands are expected to be empty.

    Example:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @norm {
        %0 = irdl.any
        %1 = irdl.parametric @complex<%0>
        irdl.results(%0)
        irdl.operands(%1)
      }
    }
    ```

    The above program defines an operation `norm` inside the dialect `cmath`.
    The operation expects a single operand of base type `cmath.complex`, and
    returns a single result of the element type of the operand.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
    "$sym_name attr-dict-with-keyword custom<SingleBlockRegion>($body)";
}

def IRDL_OperandsOp : IRDL_Op<"operands", [HasParent<"OperationOp">]> {
  let summary = "Define the operands of an operation";
  let description = [{
    `irdl.operands` define the operands of the `irdl.operation` parent operation
    definition.

    In the following example, `irdl.operands` defines the operands of the
    `norm` operation:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @mul {
        %0 = irdl.any
        %1 = irdl.parametric @complex<%0>
        irdl.results(%1)
        irdl.operands(%1, %1)
      }
    }
    ```

    The `mul` operation will expect two operands of type `cmath.complex`, that
    have the same type, and return a result of the same type.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args);
  let assemblyFormat = " `(` $args `)` attr-dict ";
}

def IRDL_ResultsOp : IRDL_Op<"results", [HasParent<"OperationOp">]> {
  let summary = "Define the results of an operation";
  let description = [{
    `irdl.results` define the results of the `irdl.operation` parent operation
    definition.

    In the following example, `irdl.results` defines the results of the
    `norm` operation:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @get_values {
        %0 = irdl.any
        %1 = irdl.parametric @complex<%0>
        irdl.results(%0, %0)
        irdl.operands(%1)
      }
    }
    ```

    The operation will expect one operand of the `cmath.complex` type, and two
    results that have the underlying type of the `cmath.complex`.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args);
  let assemblyFormat = " `(` $args `)` attr-dict ";
}

//===----------------------------------------------------------------------===//
// IRDL Constraint operations
//===----------------------------------------------------------------------===//

class IRDL_ConstraintOp<string mnemonic, list<Trait> traits = []>
    : IRDL_Op<mnemonic, traits> {
}

def IRDL_Is : IRDL_ConstraintOp<"is",
    [ParentOneOf<["TypeOp", "AttributeOp", "OperationOp"]>, Pure]> {
  let summary = "Constraints an attribute/type to be a specific attribute instance";
  let description = [{
    `irdl.is` defines a constraint that only accepts a specific instance of a
    type or attribute.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex_i32 {
        %0 = irdl.is i32
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a `complex_i32` type inside the dialect `cmath`
    that can only have a `i32` as its parameter.
  }];

  let arguments = (ins AnyAttr:$expected);
  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = " $expected ` ` attr-dict ";
}

def IRDL_Parametric : IRDL_ConstraintOp<"parametric",
    [ParentOneOf<["TypeOp", "AttributeOp", "OperationOp"]>, Pure]> {
  let summary = "Constraints an attribute/type base and its parameters";
  let description = [{
    `irdl.parametric` defines a constraint that accepts only a single type
    or attribute base. The attribute base is defined by a symbolic reference
    to the corresponding definition. It will additionally constraint the
    parameters of the type/attribute.

    Example:

    ```mlir
    irdl.dialect @cmath {

      irdl.type @complex { /* ... */ }

      irdl.operation @norm {
        %0 = irdl.any
        %1 = irdl.parametric @complex<%0>
        irdl.operands(%1)
        irdl.results(%0)
      }
    }
    ```

    The above program defines an operation `norm` inside the dialect `cmath` that
    for any `T` takes a `cmath.complex` with parameter `T` and returns a `T`.
  }];

  let arguments = (ins SymbolRefAttr:$base_type,
                       Variadic<IRDL_AttributeType>:$args);
  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = " $base_type `<` $args `>` ` ` attr-dict ";
}

def IRDL_Any : IRDL_ConstraintOp<"any",
    [ParentOneOf<["TypeOp", "AttributeOp", "OperationOp"]>]> {
  let summary = "Accept any type or attribute";
  let description = [{
    `irdl.any` defines a constraint that accepts any type or attribute.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex_flexible {
        %0 = irdl.any
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a type `complex_flexible` inside the dialect
    `cmath` that has a single parameter that can be any attribute.
  }];

  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = " attr-dict ";
}


#endif // MLIR_DIALECT_IRDL_IR_IRDLOPS
