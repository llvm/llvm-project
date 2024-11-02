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
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PDL Ops
//===----------------------------------------------------------------------===//

class PDL_Op<string mnemonic, list<Trait> traits = []>
    : Op<PDL_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeConstraintOp
//===----------------------------------------------------------------------===//

def PDL_ApplyNativeConstraintOp
    : PDL_Op<"apply_native_constraint", [HasParent<"pdl::PatternOp">]> {
  let summary = "Apply a native constraint to a set of provided entities";
  let description = [{
    `pdl.apply_native_constraint` operations apply a native C++ constraint, that
    has been registered externally with the consumer of PDL, to a given set of
    entities.

    Example:

    ```mlir
    // Apply `myConstraint` to the entities defined by `input`, `attr`, and `op`.
    pdl.apply_native_constraint "myConstraint"(%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation)
    ```
  }];

  let arguments = (ins StrAttr:$name, Variadic<PDL_AnyType>:$args);
  let assemblyFormat = "$name `(` $args `:` type($args) `)` attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeRewriteOp
//===----------------------------------------------------------------------===//

def PDL_ApplyNativeRewriteOp
    : PDL_Op<"apply_native_rewrite", [HasParent<"pdl::RewriteOp">]> {
  let summary = "Apply a native rewrite method inside of pdl.rewrite region";
  let description = [{
    `pdl.apply_native_rewrite` operations apply a native C++ function, that has
    been registered externally with the consumer of PDL, to perform a rewrite
    and optionally return a number of values. The native function may accept any
    number of arguments. This operation is used within a pdl.rewrite region to enable
    the interleaving of native rewrite methods with other pdl constructs.

    Example:

    ```mlir
    // Apply a native rewrite method that returns an attribute.
    %ret = pdl.apply_native_rewrite "myNativeFunc"(%arg0, %attr1) : !pdl.attribute
    ```

    ```c++
    // The native rewrite as defined in C++:
    static Attribute myNativeFunc(PatternRewriter &rewriter, Value arg0, Attribute arg1) {
      // Just return the second arg.
      return arg1;
    }

    void registerNativeRewrite(PDLPatternModule &pdlModule) {
      pdlModule.registerRewriteFunction("myNativeFunc", myNativeFunc);
    }
    ```
  }];

  let arguments = (ins StrAttr:$name, Variadic<PDL_AnyType>:$args);
  let results = (outs Variadic<PDL_AnyType>:$results);
  let assemblyFormat = [{
    $name (`(` $args^ `:` type($args) `)`)? (`:` type($results)^)? attr-dict
  }];
  let hasVerifier = 1;
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
    %attr = pdl.attribute = "hello"
    ```
  }];

  let arguments = (ins Optional<PDL_Type>:$type,
                       OptionalAttr<AnyAttr>:$value);
  let results = (outs PDL_Attribute:$attr);
  let assemblyFormat = "(`:` $type^)? (`=` $value^)? attr-dict-with-keyword";

  let builders = [
    OpBuilder<(ins CArg<"Value", "Value()">:$type), [{
      build($_builder, $_state, $_builder.getType<AttributeType>(), type,
            Attribute());
    }]>,
    OpBuilder<(ins "Attribute":$attr), [{
      build($_builder, $_state, $_builder.getType<AttributeType>(), Value(), attr);
    }]>,
  ];
  let hasVerifier = 1;
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
}

//===----------------------------------------------------------------------===//
// pdl::OperandOp
//===----------------------------------------------------------------------===//

def PDL_OperandOp
    : PDL_Op<"operand", [HasParent<"pdl::PatternOp">]> {
  let summary = "Define an external input operand in a pattern";
  let description = [{
    `pdl.operand` operations capture external operand edges into an operation
    node that originate from operations or block arguments not otherwise
    specified within the pattern (i.e. via `pdl.result` or `pdl.results`). These
    operations define individual operands of a given operation. A `pdl.operand`
    may partially constrain an operand by specifying an expected value type
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
    OpBuilder<(ins), [{
      build($_builder, $_state, $_builder.getType<ValueType>(), Value());
    }]>,
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::OperandsOp
//===----------------------------------------------------------------------===//

def PDL_OperandsOp
    : PDL_Op<"operands", [HasParent<"pdl::PatternOp">]> {
  let summary = "Define a range of input operands in a pattern";
  let description = [{
    `pdl.operands` operations capture external operand range edges into an
    operation node that originate from operations or block arguments not
    otherwise specified within the pattern (i.e. via `pdl.result` or
    `pdl.results`). These operations define groups of input operands into a
    given operation. A `pdl.operands` may partially constrain a set of input
    operands by specifying expected value types (via `pdl.types` operations).

    Example:

    ```mlir
    // Define a range of input operands:
    %operands = pdl.operands

    // Define a range of input operands with expected types:
    %types = pdl.types : [i32, i64, i32]
    %typed_operands = pdl.operands : %types
    ```
  }];

  let arguments = (ins Optional<PDL_RangeOf<PDL_Type>>:$type);
  let results = (outs PDL_RangeOf<PDL_Value>:$val);
  let assemblyFormat = "(`:` $type^)? attr-dict";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, RangeType::get($_builder.getType<ValueType>()),
            Value());
    }]>,
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

def PDL_OperationOp : PDL_Op<"operation", [AttrSizedOperandSegments]> {
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
    a handle to the operation itself. Handles to the results of the operation
    can be extracted via `pdl.result`.

    Example:

    ```mlir
    // Define an instance of a `foo.op` operation.
    %op = pdl.operation "foo.op"(%arg0, %arg1 : !pdl.value, !pdl.value)
      {"attrA" = %attr0} -> (%type, %type : !pdl.type, !pdl.type)
    ```

    When used within a matching context, the name of the operation may be
    omitted.

    When used within a rewriting context, i.e. when defined within a
    `pdl.rewrite`, all of the result types must be "inferable". This means that
    the type must be attributable to either a constant type value or the result
    type of another entity, such as an attribute, the result of a
    `apply_native_rewrite`, or the result type of another operation. If the
    result type value does not meet any of these criteria, the operation must
    override the `InferTypeOpInterface` to ensure that the result types can be
    inferred.

    The operands of the operation are interpreted in the following ways:

    1) A single !pdl.range<value>:

    In this case, the single range is treated as all of the operands of the
    operation.

    ```mlir
    // Define an instance with single range of operands.
    %op = pdl.operation "func.return"(%allArgs : !pdl.range<value>)
    ```

    2) A variadic number of either !pdl.value or !pdl.range<value>:

    In this case, the inputs are expected to correspond with the operand groups
    defined on the operation in ODS.

    ```tablgen
    // Given the following operation definition in ODS:
    def MyIndirectCallOp {
      let results = (outs FunctionType:$call, Variadic<AnyType>:$args);
    }
    ```

    ```mlir
    // We can match the operands as so:
    %op = pdl.operation "my.indirect_call"(%call, %args : !pdl.value, !pdl.range<value>)
    ```

    The results of the operation are interpreted in the following ways:

    1) A single !pdl.range<type>:

    In this case, the single range is treated as all of the result types of the
    operation.

    ```mlir
    // Define an instance with single range of types.
    %allResultTypes = pdl.types
    %op = pdl.operation "builtin.unrealized_conversion_cast" -> (%allResultTypes : !pdl.types)
    ```

    2) A variadic number of either !pdl.type or !pdl.range<type>:

    In this case, the inputs are expected to correspond with the result groups
    defined on the operation in ODS.

    ```tablgen
    // Given the following operation definition in ODS:
    def MyOp {
      let results = (outs SomeType:$result, Variadic<SomeType>:$otherResults);
    }
    ```

    ```mlir
    // We can match the results as so:
    %result = pdl.type
    %otherResults = pdl.types
    %op = pdl.operation "foo.op" -> (%result, %otherResults : !pdl.type, !pdl.range<type>)
    ```
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$name,
                       Variadic<PDL_InstOrRangeOf<PDL_Value>>:$operands,
                       Variadic<PDL_Attribute>:$attributes,
                       StrArrayAttr:$attributeNames,
                       Variadic<PDL_InstOrRangeOf<PDL_Type>>:$types);
  let results = (outs PDL_Operation:$op);
  let assemblyFormat = [{
    ($name^)? (`(` $operands^ `:` type($operands) `)`)?
    custom<OperationOpAttributes>($attributes, $attributeNames)
    (`->` `(` $types^ `:` type($types) `)`)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins CArg<"Optional<StringRef>", "llvm::None">:$name,
      CArg<"ValueRange", "llvm::None">:$operandValues,
      CArg<"ArrayRef<StringRef>", "llvm::None">:$attrNames,
      CArg<"ValueRange", "llvm::None">:$attrValues,
      CArg<"ValueRange", "llvm::None">:$resultTypes), [{
      auto nameAttr = name ? $_builder.getStringAttr(*name) : StringAttr();
      build($_builder, $_state, $_builder.getType<OperationType>(), nameAttr,
            operandValues, attrValues, $_builder.getStrArrayAttr(attrNames),
            resultTypes);
    }]>,
  ];
  let extraClassDeclaration = [{
    /// Returns true if the operation type referenced supports result type
    /// inference.
    bool hasTypeInference();
    
    /// Returns true if the operation type referenced might support result type
    /// inference, i.e. it supports type reference or is currently not
    /// registered in the context. Returns false if the root operation name
    /// has not been set.
    bool mightHaveTypeInference();
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::PatternOp
//===----------------------------------------------------------------------===//

def PDL_PatternOp : PDL_Op<"pattern", [
    IsolatedFromAbove, SingleBlock, Symbol,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>
  ]> {
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
      %root = pdl.operation "foo.op"(%inputOperand) -> (%resultType)
      pdl.rewrite %root {
        pdl.replace %root with (%inputOperand)
      }
    }
    ```
  }];

  let arguments = (ins Confined<I16Attr, [IntNonNegative]>:$benefit,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    ($sym_name^)? `:` `benefit` `(` $benefit `)` attr-dict-with-keyword $body
  }];

  let builders = [
    OpBuilder<(ins CArg<"Optional<uint16_t>", "1">:$benefit,
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
  }];
  let hasRegionVerifier = 1;
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
    pdl.replace %root with (%val0, %val1 : !pdl.value, !pdl.value)

    // Replace root node with a range of values:
    pdl.replace %root with (%vals : !pdl.range<value>)

    // Replace root with another operation:
    pdl.replace %root with %otherOp
    ```
  }];
  let arguments = (ins PDL_Operation:$operation,
                       Optional<PDL_Operation>:$replOperation,
                       Variadic<PDL_InstOrRangeOf<PDL_Value>>:$replValues);
  let assemblyFormat = [{
    $operation `with` (`(` $replValues^ `:` type($replValues) `)`)?
    ($replOperation^)? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::ResultOp
//===----------------------------------------------------------------------===//

def PDL_ResultOp : PDL_Op<"result", [NoSideEffect]> {
  let summary = "Extract a result from an operation";
  let description = [{
    `pdl.result` operations extract result edges from an operation node within
    a pattern or rewrite region. The provided index is zero-based, and
    represents the concrete result to extract, i.e. this is not the result index
    as defined by the ODS definition of the operation.

    Example:

    ```mlir
    // Extract a result:
    %operation = pdl.operation ...
    %pdl_result = pdl.result 1 of %operation

    // Imagine the following IR being matched:
    %result_0, %result_1 = foo.op ...

    // If the example pattern snippet above were matching against `foo.op` in
    // the IR snippet, `%pdl_result` would correspond to `%result_1`.
    ```
  }];

  let arguments = (ins PDL_Operation:$parent, I32Attr:$index);
  let results = (outs PDL_Value:$val);
  let assemblyFormat = "$index `of` $parent attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl::ResultsOp
//===----------------------------------------------------------------------===//

def PDL_ResultsOp : PDL_Op<"results", [NoSideEffect]> {
  let summary = "Extract a result group from an operation";
  let description = [{
    `pdl.results` operations extract a result group from an operation within a
    pattern or rewrite region. If an index is provided, this operation extracts
    a result group as defined by the ODS definition of the operation. In this
    case the result of this operation may be either a single `pdl.value` or
    a `pdl.range<value>`, depending on the constraint of the result in ODS. If
    no index is provided, this operation extracts the full result range of the
    operation.

    Example:

    ```mlir
    // Extract all of the results of an operation:
    %operation = pdl.operation ...
    %results = pdl.results of %operation

    // Extract the results in the first result group of an operation, which is
    // variadic:
    %operation = pdl.operation ...
    %results = pdl.results 0 of %operation -> !pdl.range<value>

    // Extract the results in the second result group of an operation, which is
    // not variadic:
    %operation = pdl.operation ...
    %results = pdl.results 1 of %operation -> !pdl.value
    ```
  }];

  let arguments = (ins PDL_Operation:$parent, OptionalAttr<I32Attr>:$index);
  let results = (outs PDL_InstOrRangeOf<PDL_Value>:$val);
  let assemblyFormat = [{
    ($index^)? `of` $parent custom<ResultsValueType>(ref($index), type($val))
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::RewriteOp
//===----------------------------------------------------------------------===//

def PDL_RewriteOp : PDL_Op<"rewrite", [
     Terminator, HasParent<"pdl::PatternOp">, NoTerminator, NoRegionArguments,
     SingleBlock, AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>
  ]> {
  let summary = "Specify the rewrite of a matched pattern";
  let description = [{
    `pdl.rewrite` operations terminate the region of a `pdl.pattern` and specify
    the main rewrite of a `pdl.pattern`, on the optional root operation. The
    rewrite is specified either via a string name (`name`) to a native
    rewrite function, or via the region body. The rewrite region, if specified,
    must contain a single block. If the rewrite is external it functions
    similarly to `pdl.apply_native_rewrite`, and takes a set of additional
    positional values defined within the matcher as arguments. If the rewrite is
    external, the root operation is passed to the native function as the leading
    arguments. The root operation, if provided, specifies the starting point in
    the pattern for the subgraph isomorphism search. Pattern matching will proceed
    from this node downward (towards the defining operation) or upward
    (towards the users) until all the operations in the pattern have been matched.
    If the root is omitted, the pdl_interp lowering will automatically select
    the best root of the pdl.rewrite among all the operations in the pattern.

    Example:

    ```mlir
    // Specify an external rewrite function:
    pdl.rewrite %root with "myExternalRewriter"(%value : !pdl.value)

    // Specify a rewrite inline using PDL with the given root:
    pdl.rewrite %root {
      %op = pdl.operation "foo.op"(%arg0, %arg1)
      pdl.replace %root with %op
    }

    // Specify a rewrite inline using PDL, automatically selecting root:
    pdl.rewrite {
      %op1 = pdl.operation "foo.op"(%arg0, %arg1)
      %op2 = pdl.operation "bar.op"(%arg0, %arg1)
      pdl.replace %root1 with %op1
      pdl.replace %root2 with %op2
    }
    ```
  }];

  let arguments = (ins Optional<PDL_Operation>:$root,
                       OptionalAttr<StrAttr>:$name,
                       Variadic<PDL_AnyType>:$externalArgs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    ($root^)? (`with` $name^ (`(` $externalArgs^ `:` type($externalArgs) `)`)?)?
              ($body^)?
    attr-dict-with-keyword
  }];
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

def PDL_TypeOp : PDL_Op<"type"> {
  let summary = "Define a type handle within a pattern";
  let description = [{
    `pdl.type` operations capture result type constraints of `Attributes`,
    `Values`, and `Operations`. Instances of this operation define, and
    partially constrain, results types of a given entity. A `pdl.type` may
    partially constrain the result by specifying a constant `Type`.

    Example:

    ```mlir
    // Define a type:
    %type = pdl.type

    // Define a type with a constant value:
    %type = pdl.type : i32
    ```
  }];

  let arguments = (ins OptionalAttr<TypeAttr>:$type);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "attr-dict (`:` $type^)?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl::TypesOp
//===----------------------------------------------------------------------===//

def PDL_TypesOp : PDL_Op<"types"> {
  let summary = "Define a range of type handles within a pattern";
  let description = [{
    `pdl.types` operations capture result type constraints of `Value`s, and
    `Operation`s. Instances of this operation define results types of a given
    entity. A `pdl.types` may partially constrain the results by specifying
    an array of `Type`s.

    Example:

    ```mlir
    // Define a range of types:
    %types = pdl.types

    // Define a range of types with a range of constant values:
    %types = pdl.types : [i32, i64, i32]
    ```
  }];

  let arguments = (ins OptionalAttr<TypeArrayAttr>:$types);
  let results = (outs PDL_RangeOf<PDL_Type>:$result);
  let assemblyFormat = "attr-dict (`:` $types^)?";
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_PDL_IR_PDLOPS
