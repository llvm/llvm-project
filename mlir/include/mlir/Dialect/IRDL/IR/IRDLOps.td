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
include "IRDLAttributes.td"
include "IRDLTypes.td"
include "IRDLInterfaces.td"
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
    AtMostOneChildOf<"OperandsOp, ResultsOp, AttributesOp, RegionsOp">,
    Symbol]> {
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

    The operands can also be marked as variadic or optional:
    ```mlir
    irdl.operands(%0, single %1, optional %2, variadic %3)
    ```
    
    Here, %0 and %1 are required single operands, %2 is an optional operand,
    and %3 is a variadic operand.

    When more than one operand is marked as optional or variadic, the operation
    will expect a 'operandSegmentSizes' attribute that defines the number of
    operands in each segment.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args, 
                        VariadicityArrayAttr:$variadicity);
  let assemblyFormat =
    "`` custom<ValuesWithVariadicity>($args, $variadicity) attr-dict";
  let hasVerifier = true;
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

    The results can also be marked as variadic or optional:
    ```mlir
    irdl.results(%0, single %1, optional %2, variadic %3)
    ```
    
    Here, %0 and %1 are required single results, %2 is an optional result,
    and %3 is a variadic result.

    When more than one result is marked as optional or variadic, the operation
    will expect a 'resultSegmentSizes' attribute that defines the number of
    results in each segment.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args, 
                        VariadicityArrayAttr:$variadicity);
  let assemblyFormat =
    " `` custom<ValuesWithVariadicity>($args, $variadicity) attr-dict";
  let hasVerifier = true;
}

def IRDL_AttributesOp : IRDL_Op<"attributes", [HasParent<"OperationOp">]> {
  let summary = "Define the attributes of an operation";

  let description = [{
    `irdl.attributes` defines the attributes of the `irdl.operation` parent 
    operation definition.

    In the following example, `irdl.attributes` defines the attributes of the
    `attr_op` operation:

    ```mlir
    irdl.dialect @example {

      irdl.operation @attr_op {
        %0 = irdl.any
        %1 = irdl.is i64
        irdl.attibutes {
          "attr1" = %0,
          "attr2" = %1
        }
      }
    }
    ```

    The operation will expect an arbitrary attribute "attr1" and an
    attribute "attr2" with value `i64`.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$attributeValues, 
                        StrArrayAttr:$attributeValueNames);
  let assemblyFormat = [{
    custom<AttributesOp>($attributeValues, $attributeValueNames) attr-dict 
  }];

  let hasVerifier = true;
}

def IRDL_RegionOp : IRDL_Op<"region",
    [HasParent<"OperationOp">, VerifyRegionInterface,
    DeclareOpInterfaceMethods<VerifyRegionInterface>]> {
  let summary = "Define a region of an operation";
  let description = [{
    The irdl.region construct defines a set of characteristics
    that a region of an operation should satify.

    These characteristics include constraints for the entry block arguments
    of the region and the total number of blocks it contains.
    The number of blocks must be a non-zero and non-negative integer,
    and it is optional by default.
    The set of constraints for the entry block arguments may be optional or
    empty. If no parentheses are provided, the set is assumed to be optional,
    and the arguments are not constrained in any way. If parentheses are
    provided with no arguments, it means that the region must have
    no entry block arguments


    Example:

    ```mlir
    irdl.dialect @example {
      irdl.operation @op_with_regions {
          %r0 = irdl.region
          %r1 = irdl.region()
          %v0 = irdl.is i32
          %v1 = irdl.is i64
          %r2 = irdl.region(%v0, %v1)
          %r3 = irdl.region with size 3

          irdl.regions(%r0, %r1, %r2, %r3)
      }
    }
    ```

    The above snippet demonstrates an operation named `@op_with_regions`,
    which is constrained to have four regions.

    * Region `%r0` doesn't have any constraints on the arguments
      or the number of blocks.
    * Region `%r1` should have an empty set of arguments.
    * Region `%r2` should have two arguments of types `i32` and `i64`.
    * Region `%r3` should contain exactly three blocks.
  }];
  let arguments = (ins Variadic<IRDL_AttributeType>:$entryBlockArgs,
                    OptionalAttr<I32Attr>:$numberOfBlocks,
                    UnitAttr:$constrainedArguments);
  let results = (outs IRDL_RegionType:$output);

  let assemblyFormat = [{
    ``(`(` $entryBlockArgs $constrainedArguments^ `)`)?
    ``(` ` `with` `size` $numberOfBlocks^)? attr-dict
  }];

  let hasVerifier = true;
}

def IRDL_RegionsOp : IRDL_Op<"regions", [HasParent<"OperationOp">]> {
  let summary = "Define the regions of an operation";
  let description = [{
    `irdl.regions` defines the regions of an operation by accepting
    values produced by `irdl.region` operation as arguments.

    Example:

    ```mlir
    irdl.dialect @example {
      irdl.operation @op_with_regions {
        %r1 = irdl.region with size 3
        %0 = irdl.any
        %r2 = irdl.region(%0)
        irdl.regions(%r1, %r2)
      }
    }
    ```

    In the snippet above the operation is constrained to have two regions.
    The first region should contain three blocks.
    The second region should have one region with one argument.
  }];

  let arguments = (ins Variadic<IRDL_RegionType>:$args);
  let assemblyFormat = " `(` $args `)` attr-dict ";
}

//===----------------------------------------------------------------------===//
// IRDL Constraint operations
//===----------------------------------------------------------------------===//

class IRDL_ConstraintOp<string mnemonic, list<Trait> traits = []>
    : IRDL_Op<mnemonic, [VerifyConstraintInterface,
        DeclareOpInterfaceMethods<VerifyConstraintInterface>] # traits> {
}

def IRDL_IsOp : IRDL_ConstraintOp<"is",
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

def IRDL_BaseOp : IRDL_ConstraintOp<"base",
    [ParentOneOf<["TypeOp", "AttributeOp", "OperationOp"]>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Constraints an attribute/type base";
  let description = [{
    `irdl.base` defines a constraint that only accepts a single type
    or attribute base, e.g. an `IntegerType`. The attribute base is defined
    either by a symbolic reference to the corresponding IRDL definition,
    or by the name of the base. Named bases are prefixed with `!` or `#`
    respectively for types and attributes.

    Example:

    ```mlir
    irdl.dialect @cmath {
      irdl.type @complex {
        %0 = irdl.base "!builtin.integer"
        irdl.parameters(%0)
      }

      irdl.type @complex_wrapper {
        %0 = irdl.base @complex
        irdl.parameters(%0)
      }
    }
    ```

    The above program defines a `cmath.complex` type that expects a single
    parameter, which is a type with base name `builtin.integer`, which is the
    name of an `IntegerType` type.
    It also defines a `cmath.complex_wrapper` type that expects a single
    parameter, which is a type of base type `cmath.complex`.
  }];

  let arguments = (ins OptionalAttr<SymbolRefAttr>:$base_ref,
                       OptionalAttr<StrAttr>:$base_name);
  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = " ($base_ref^)? ($base_name^)? ` ` attr-dict";
  
  let builders = [
    OpBuilder<(ins "SymbolRefAttr":$base_ref), [{
      build($_builder, $_state, base_ref, {});
    }]>,
    OpBuilder<(ins "StringAttr":$base_name), [{
      build($_builder, $_state, {}, base_name);
    }]>,
  ];

  let hasVerifier = 1;
}

def IRDL_ParametricOp : IRDL_ConstraintOp<"parametric",
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

def IRDL_AnyOp : IRDL_ConstraintOp<"any",
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

def IRDL_AnyOfOp : IRDL_ConstraintOp<"any_of",
                  [ParentOneOf<["TypeOp", "AttributeOp", "OperationOp"]>,
                   SameOperandsAndResultType]> {
  let summary = "Constraints to the union of the provided constraints";
  let description = [{
    `irdl.any_of` defines a constraint that accepts any type or attribute that
    satisfies at least one of its provided type constraints.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex {
        %0 = irdl.is i32
        %1 = irdl.is i64
        %2 = irdl.is f32
        %3 = irdl.is f64
        %4 = irdl.any_of(%0, %1, %2, %3)
        irdl.parameters(%4)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    can have a single type parameter that can be either `i32`, `i64`, `f32` or
    `f32`.
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args);
  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = [{ `(` $args `)` ` ` attr-dict }];
}

def IRDL_AllOfOp : IRDL_ConstraintOp<"all_of",
                 [ParentOneOf<["TypeOp", "AttributeOp", "OperationOp"]>,
                  SameOperandsAndResultType]> {
  let summary = "Constraints to the intersection of the provided constraints";
  let description = [{
    `irdl.all_of` defines a constraint that accepts any type or attribute that
    satisfies all of its provided constraints.

    Example:

    ```mlir
    irdl.dialect cmath {
      irdl.type complex_f32 {
        %0 = irdl.is i32
        %1 = irdl.is f32
        %2 = irdl.any_of(%0, %1) // is 32-bit

        %3 = irdl.is f32
        %4 = irdl.is f64
        %5 = irdl.any_of(%3, %4) // is a float

        %6 = irdl.all_of(%2, %5) // is a 32-bit float
        irdl.parameters(%6)
      }
    }
    ```

    The above program defines a type `complex` inside the dialect `cmath` that
    can has one parameter that must be 32-bit long and a float (in other
    words, that must be `f32`).
  }];

  let arguments = (ins Variadic<IRDL_AttributeType>:$args);
  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = [{ `(` $args `)` ` ` attr-dict }];
}

def IRDL_CPredOp : IRDL_Op<"c_pred"> {
  let summary = "Constraints an attribute using a C++ predicate";
  let description = [{
    `irdl.c_pred` defines a constraint that is written in C++.

    Dialects using this operation cannot be registered at runtime, as it relies
    on C++ code.

    Special placeholders can be used to refer to entities in the context where
    this predicate is used. They serve as "hooks" to the enclosing environment.
    The following special placeholders are supported in constraints for an op:

    * `$_builder` will be replaced by a mlir::Builder instance.
    * `$_op` will be replaced by the current operation.
    * `$_self` will be replaced with the entity this predicate is attached to.
       Compared to ODS, `$_self` is always of type `mlir::Attribute`, and types
       are manipulated as `TypeAttr` attributes.

    Example:
    ```mlir
    irdl.type @op_with_attr {
      %0 = irdl.c_pred "::llvm::isa<::mlir::IntegerAttr>($_self)"
      irdl.parameters(%0)
    }
    ```

    In this example, @op_with_attr is defined as a type with a single
    parameter, which is an `IntegerAttr`, as constrained by the C++ predicate.
  }];

  let arguments = (ins StrAttr:$pred);
  let results = (outs IRDL_AttributeType:$output);
  let assemblyFormat = "$pred ` ` attr-dict";
}

#endif // MLIR_DIALECT_IRDL_IR_IRDLOPS
