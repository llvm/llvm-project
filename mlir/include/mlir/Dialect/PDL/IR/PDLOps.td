//===- PDLOps.td - Pattern descriptor operations -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Pattern Descriptor Language dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_PDLOPS
#define MLIR_DIALECT_PDL_IR_PDLOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// PDL Ops
//===----------------------------------------------------------------------===//

class PDL_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<PDL_Dialect, mnemonic, traits> {
  let printer = [{ ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// pdl::ApplyConstraintOp
//===----------------------------------------------------------------------===//

def PDL_ApplyConstraintOp
    : PDL_Op<"apply_constraint", [HasParent<"pdl::PatternOp">]> {
  let summary = "Apply a generic constraint to a set of provided entities";
  let description = [{
    `apply_constraint` operations apply a generic constraint, that has been
    registered externally with the consumer of PDL, to a given set of entities.
    The constraint is permitted to accept any number of constant valued
    parameters.

    Example:

    ```mlir
    // Apply `myConstraint` to the entities defined by `input`, `attr`, and
    // `op`. `42`, `"abc"`, and `i32` are constant parameters passed to the
    // constraint.
    pdl.apply_constraint "myConstraint"[42, "abc", i32](%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation)
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_AnyType>:$args,
                       OptionalAttr<ArrayAttr>:$constParams);
  let assemblyFormat = [{
    $name ($constParams^)? `(` $args `:` type($args) `)` attr-dict
  }];

  let builders = [
    OpBuilderDAG<(ins "StringRef":$name, CArg<"ValueRange", "{}">:$args,
      CArg<"ArrayRef<Attribute>", "{}">:$params), [{
      build($_builder, $_state, $_builder.getStringAttr(name), args,
            params.empty() ? ArrayAttr() : $_builder.getArrayAttr(params));
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl::AttributeOp
//===----------------------------------------------------------------------===//

def PDL_AttributeOp : PDL_Op<"attribute"> {
  let summary = "Define an input attribute in a pattern";
  let description = [{
    `pdl.attribute` operations capture named attribute edges into an operation.
    Instances of this operation define, and partially constrain, attributes of a
    given operation. A `pdl.attribute` may partially constrain the input by
    specifying an expected attribute value type (via a `pdl.type` operation), or
    a constant value for the attribute (via `val`). Only one of these may be set
    for a given input, as the type of the constant value provides the type. When
    defined within a `pdl.rewrite` region, the constant value must be specified.

    Example:

    ```mlir
    // Define an attribute:
    %attr = pdl.attribute

    // Define an attribute with an expected type:
    %type = pdl.type : i32
    %attr = pdl.attribute : %type

    // Define an attribute with a constant value:
    %attr = pdl.attribute "hello"
    ```
  }];

  let arguments = (ins Optional<PDL_Type>:$type,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs PDL_Attribute:$attr);
  let assemblyFormat = "attr-dict (`:` $type^)? ($value^)?";

  let builders = [
    OpBuilderDAG<(ins CArg<"Value", "Value()">:$type), [{
      build($_builder, $_state, $_builder.getType<AttributeType>(), type,
            Attribute());
    }]>,
    OpBuilderDAG<(ins "Attribute":$attr), [{
      build($_builder, $_state, $_builder.getType<AttributeType>(), Value(), attr);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl::CreateNativeOp
//===----------------------------------------------------------------------===//

def PDL_CreateNativeOp
    : PDL_Op<"create_native", [HasParent<"pdl::RewriteOp">]> {
  let summary = "Call a native creation method to construct an `Attribute`, "
                "`Operation`, `Type`, or `Value`";
  let description = [{
    `pdl.create_native` operations invoke a native C++ function, that has been
    registered externally with the consumer of PDL, to create an `Attribute`,
    `Operation`, `Type`, or `Value`. The native function must produce a value
    of the specified return type, and may accept any number of positional
    arguments and constant attribute parameters.

    Example:

    ```mlir
    %ret = pdl.create_native "myNativeFunc"[42, "gt"](%arg0, %arg1) : !pdl.attribute
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_AnyType>:$args,
                       OptionalAttr<ArrayAttr>:$constParams);
  let results = (outs PDL_AnyType:$result);
  let assemblyFormat = [{
    $name ($constParams^)? (`(` $args^ `:` type($args) `)`)? `:` type($result)
    attr-dict
  }];
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// pdl::EraseOp
//===----------------------------------------------------------------------===//

def PDL_EraseOp : PDL_Op<"erase", [HasParent<"pdl::RewriteOp">]> {
  let summary = "Mark an input operation as `erased`";
  let description = [{
    `pdl.erase` operations are used within `pdl.rewrite` regions to specify that
    an input operation should be marked as erased. The semantics of this
    operation correspond with the `eraseOp` method on a `PatternRewriter`.

    Example:

    ```mlir
    pdl.erase %root
    ```
  }];
  let arguments = (ins PDL_Operation:$operation);
  let assemblyFormat = "$operation attr-dict";
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// pdl::OperandOp
//===----------------------------------------------------------------------===//

def PDL_OperandOp : PDL_Op<"operand", [HasParent<"pdl::PatternOp">]> {
  let summary = "Define an external input operand in a pattern";
  let description = [{
    `pdl.operand` operations capture external operand edges into an operation
    node that originate from operations or block arguments not otherwise
    specified within the pattern (e.g. via `pdl.operation`). These operations
    define individual operands of a given operation. A `pdl.operand` may
    partially constrain an operand by specifying an expected value type
    (via a `pdl.type` operation).

    Example:

    ```mlir
    // Define an external operand:
    %operand = pdl.operand

    // Define an external operand with an expected type:
    %type = pdl.type : i32
    %operand = pdl.operand : %type
    ```
  }];

  let arguments = (ins Optional<PDL_Type>:$type);
  let results = (outs PDL_Value:$val);
  let assemblyFormat = "(`:` $type^)? attr-dict";

  let builders = [
    OpBuilderDAG<(ins), [{
      build($_builder, $_state, $_builder.getType<ValueType>(), Value());
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

def PDL_OperationOp
    : PDL_Op<"operation", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = "Define an operation within a pattern";
  let description = [{
    `pdl.operation` operations define operation nodes within a pattern. Within
    a match sequence, i.e. when directly nested within a `pdl.pattern`, these
    operations correspond to input operations, or those that already existing
    within the MLIR module. Inside of a `pdl.rewrite`, these operations
    correspond to operations that should be created as part of the replacement
    sequence.

    `pdl.operation`s are composed of a name, and a set of attribute, operand,
    and result type values, that map to what those that would be on a
    constructed instance of that operation. The results of a `pdl.operation` are
    a handle to the operation itself, and a handle to each of the operation
    result values.

    When used within a matching context, the name of the operation may be
    omitted.

    When used within a rewriting context, i.e. when defined within a
    `pdl.rewrite`, all of the result types must be "inferable". This means that
    the type must be attributable to either a constant type value or the result
    type of another entity, such as an attribute, the result of a
    `createNative`, or the result type of another operation. If the result type
    value does not meet any of these criteria, the operation must provide the
    `InferTypeOpInterface` to ensure that the result types can be inferred.

    Example:

    ```mlir
    // Define an instance of a `foo.op` operation.
    %op, %results:4 = pdl.operation "foo.op"(%arg0, %arg1) {"attrA" = %attr0} -> %type, %type, %type, %type
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$name,
                       Variadic<PDL_Value>:$operands,
                       Variadic<PDL_Attribute>:$attributes,
                       StrArrayAttr:$attributeNames,
                       Variadic<PDL_Type>:$types);
  let results = (outs PDL_Operation:$op,
                      Variadic<PDL_Value>:$results);
  let builders = [
    OpBuilderDAG<(ins CArg<"Optional<StringRef>", "llvm::None">:$name,
      CArg<"ValueRange", "llvm::None">:$operandValues,
      CArg<"ArrayRef<StringRef>", "llvm::None">:$attrNames,
      CArg<"ValueRange", "llvm::None">:$attrValues,
      CArg<"ValueRange", "llvm::None">:$resultTypes), [{
      auto nameAttr = name ? StringAttr() : $_builder.getStringAttr(*name);
      build($_builder, $_state, $_builder.getType<OperationType>(), {}, nameAttr,
            operandValues, attrValues, $_builder.getStrArrayAttr(attrNames),
            resultTypes);
      $_state.types.append(resultTypes.size(), $_builder.getType<ValueType>());
    }]>,
  ];
  let extraClassDeclaration = [{
    /// Returns true if the operation type referenced supports result type
    /// inference.
    bool hasTypeInference();
  }];
}

//===----------------------------------------------------------------------===//
// pdl::PatternOp
//===----------------------------------------------------------------------===//

def PDL_PatternOp : PDL_Op<"pattern", [IsolatedFromAbove, Symbol]> {
  let summary = "Define a rewrite pattern";
  let description = [{
    `pdl.pattern` operations provide a transformable representation for a
    `RewritePattern`. The attributes on this operation correspond to the various
    metadata on a `RewritePattern`, such as the benefit. The match section of
    the pattern is specified within the region body, with the rewrite provided
    by a terminating `pdl.rewrite`.

    Example:

    ```mlir
    // Provide a pattern matching "foo.op" that replaces the root with its
    // operand.
    pdl.pattern : benefit(1) {
      %resultType = pdl.type
      %inputOperand = pdl.operand
      %root, %results = pdl.operation "foo.op"(%inputOperand) -> (%resultType)
      pdl.rewrite %root {
        pdl.replace %root with (%inputOperand)
      }
    }
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$rootKind,
                       Confined<I16Attr, [IntNonNegative]>:$benefit,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ($sym_name^)? `:` `benefit` `(` $benefit `)`
    (`,` `root` `(` $rootKind^ `)`)? attr-dict-with-keyword $body
  }];

  let builders = [
    OpBuilderDAG<(ins CArg<"Optional<StringRef>", "llvm::None">:$rootKind,
      CArg<"Optional<uint16_t>", "1">:$benefit,
      CArg<"Optional<StringRef>", "llvm::None">:$name)>,
  ];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// A PatternOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

    /// Returns the rewrite operation of this pattern.
    RewriteOp getRewriter();

    /// Return the root operation kind that this pattern matches, or None if
    /// there isn't a specific root.
    Optional<StringRef> getRootKind();
  }];
}

//===----------------------------------------------------------------------===//
// pdl::ReplaceOp
//===----------------------------------------------------------------------===//

def PDL_ReplaceOp : PDL_Op<"replace", [
    AttrSizedOperandSegments, HasParent<"pdl::RewriteOp">
  ]> {
  let summary = "Mark an input operation as `replaced`";
  let description = [{
    `pdl.replace` operations are used within `pdl.rewrite` regions to specify
    that an input operation should be marked as replaced. The semantics of this
    operation correspond with the `replaceOp` method on a `PatternRewriter`. The
    set of replacement values can be either:
    * a single `Operation` (`replOperation` should be populated)
      - The operation will be replaced with the results of this operation.
    * a set of `Value`s (`replValues` should be populated)
      - The operation will be replaced with these values.

    Example:

    ```mlir
    // Replace root node with 2 values:
    pdl.replace %root with (%val0, %val1)

    // Replace root with another operation:
    pdl.replace %root with %otherOp
    ```
  }];
  let arguments = (ins PDL_Operation:$operation,
                       Optional<PDL_Operation>:$replOperation,
                       Variadic<PDL_Value>:$replValues);
  let assemblyFormat = [{
    $operation `with` (`(` $replValues^ `)`)? ($replOperation^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// pdl::RewriteOp
//===----------------------------------------------------------------------===//

def PDL_RewriteOp : PDL_Op<"rewrite", [
     Terminator, HasParent<"pdl::PatternOp">,
     SingleBlockImplicitTerminator<"pdl::RewriteEndOp">
  ]> {
  let summary = "Specify the rewrite of a matched pattern";
  let description = [{
    `pdl.rewrite` operations terminate the region of a `pdl.pattern` and specify
    the rewrite of a `pdl.pattern`, on the specified root operation. The
    rewrite is specified either via a string name (`name`) to an external
    rewrite function, or via the region body. The rewrite region, if specified,
    must contain a single block and terminate via the `pdl.rewrite_end`
    operation. If the rewrite is external, it also takes a set of constant
    parameters and a set of additional positional values defined within the
    matcher as arguments.

    Example:

    ```mlir
    // Specify an external rewrite function:
    pdl.rewrite %root with "myExternalRewriter"(%value : !pdl.value)

    // Specify the rewrite inline using PDL:
    pdl.rewrite %root {
      %op = pdl.operation "foo.op"(%arg0, %arg1)
      pdl.replace %root with %op
    }
    ```
  }];

  let arguments = (ins PDL_Operation:$root,
                       OptionalAttr<StrAttr>:$name,
                       Variadic<PDL_AnyType>:$externalArgs,
                       OptionalAttr<ArrayAttr>:$externalConstParams);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $root (`with` $name^ ($externalConstParams^)?
                         (`(` $externalArgs^ `:` type($externalArgs) `)`)?)?
          ($body^)?
    attr-dict-with-keyword
  }];
}

def PDL_RewriteEndOp : PDL_Op<"rewrite_end", [Terminator,
                                              HasParent<"pdl::RewriteOp">]> {
  let summary = "Implicit terminator of a `pdl.rewrite` region";
  let description = [{
    `pdl.rewrite_end` operations terminate the region of a `pdl.rewrite`.
  }];
  let assemblyFormat = "attr-dict";
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

def PDL_TypeOp : PDL_Op<"type"> {
  let summary = "Define a type handle within a pattern";
  let description = [{
    `pdl.type` operations capture result type constraints of an `Attributes`,
    `Values`, and `Operations`. Instances of this operation define, and
    partially constrain, results types of a given entity. A `pdl.type` may
    partially constrain the result by specifying a constant `Type`.

    Example:

    ```mlir
    // Define a type:
    %attr = pdl.type

    // Define a type with a constant value:
    %attr = pdl.type : i32
    ```
  }];

  let arguments = (ins OptionalAttr<TypeAttr>:$type);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "attr-dict (`:` $type^)?";

  let builders = [
    OpBuilderDAG<(ins CArg<"Type", "Type()">:$ty), [{
      build($_builder, $_state, $_builder.getType<AttributeType>(),
            ty ? TypeAttr::get(ty) : TypeAttr());
    }]>,
  ];
}

#endif // MLIR_DIALECT_PDL_IR_PDLOPS
