//===- PDLInterpOps.td - Pattern Interpreter Dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the PDL interpreter dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDLINTERP_IR_PDLINTERPOPS
#define MLIR_DIALECT_PDLINTERP_IR_PDLINTERPOPS

include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PDLInterp Dialect
//===----------------------------------------------------------------------===//

def PDLInterp_Dialect : Dialect {
  let summary = "Interpreted pattern execution dialect";
  let description = [{
    The PDL Interpreter dialect provides a lower level abstraction compared to
    the PDL dialect, and is targeted towards low level optimization and
    interpreter code generation. The dialect operations encapsulates
    low-level pattern match and rewrite "primitives", such as navigating the
    IR (Operation::getOperand), creating new operations (OpBuilder::create),
    etc. Many of the operations within this dialect also fuse branching control
    flow with some form of a predicate comparison operation. This type of fusion
    reduces the amount of work that an interpreter must do when executing.
  }];

  let name = "pdl_interp";
  let cppNamespace = "::mlir::pdl_interp";
  let dependentDialects = ["pdl::PDLDialect"];
  let extraClassDeclaration = [{
    /// Returns the name of the function containing the matcher code. This
    /// function is called by the interpreter when matching an operation.
    static StringRef getMatcherFunctionName() { return "matcher"; }

    /// Returns the name of the module containing the rewrite functions. These
    /// functions are invoked by distinct patterns within the matcher function
    /// to rewrite the IR after a successful match.
    static StringRef getRewriterModuleName() { return "rewriters"; }
  }];
}

//===----------------------------------------------------------------------===//
// PDLInterp Operations
//===----------------------------------------------------------------------===//

// Generic interpreter operation.
class PDLInterp_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PDLInterp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// PDLInterp_PredicateOp

// Check operations evaluate a predicate on a positional value and then
// conditionally branch on the result.
class PDLInterp_PredicateOp<string mnemonic, list<OpTrait> traits = []> :
    PDLInterp_Op<mnemonic, !listconcat([Terminator], traits)> {
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
}

//===----------------------------------------------------------------------===//
// PDLInterp_SwitchOp

// Switch operations evaluate a predicate on a positional value and then
// conditionally branch on the result.
class PDLInterp_SwitchOp<string mnemonic, list<OpTrait> traits = []> :
    PDLInterp_Op<mnemonic, !listconcat([Terminator], traits)> {
  let successors = (successor AnySuccessor:$defaultDest,
                              VariadicSuccessor<AnySuccessor>:$cases);

  let verifier = [{
    // Verify that the number of case destinations matches the number of case
    // values.
    size_t numDests = cases().size();
    size_t numValues = caseValues().size();
    if (numDests != numValues) {
      return emitOpError("expected number of cases to match the number of case "
                         "values, got ")
          << numDests << " but expected " << numValues;
    }
    return success();
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyConstraintOp
//===----------------------------------------------------------------------===//

def PDLInterp_ApplyConstraintOp : PDLInterp_PredicateOp<"apply_constraint"> {
  let summary = "Apply a constraint to a set of positional values";
  let description = [{
    `pdl_interp.apply_constraint` operations apply a generic constraint, that
    has been registered with the interpreter, with a given set of positional
    values. The constraint may have any number of constant parameters. On
    success, this operation branches to the true destination, otherwise the
    false destination is taken.

    Example:

    ```mlir
    // Apply `myConstraint` to the entities defined by `input`, `attr`, and
    // `op`.
    pdl_interp.apply_constraint "myConstraint"[42, "abc", i32](%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation) -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_PositionalValue>:$args,
                       OptionalAttr<ArrayAttr>:$constParams);
  let assemblyFormat = [{
    $name ($constParams^)? `(` $args `:` type($args) `)` attr-dict `->`
    successors
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyRewriteOp
//===----------------------------------------------------------------------===//

def PDLInterp_ApplyRewriteOp : PDLInterp_Op<"apply_rewrite"> {
  let summary = "Invoke and apply an externally registered rewrite method";
  let description = [{
    `pdl_interp.apply_rewrite` operations invoke an external rewriter that has
    been registered with the interpreter to perform the rewrite after a
    successful match. The rewrite is passed the root operation being matched, a
    set of additional positional arguments generated within the matcher, and a
    set of constant parameters.

    Example:

    ```mlir
    // Rewriter operating solely on the root operation.
    pdl_interp.apply_rewrite "rewriter" on %root

    // Rewriter operating on the root operation along with additional arguments
    // from the matcher.
    pdl_interp.apply_rewrite "rewriter"(%value : !pdl.value) on %root

    // Rewriter operating on the root operation along with additional arguments
    // and constant parameters.
    pdl_interp.apply_rewrite "rewriter"[42](%value : !pdl.value) on %root
    ```
  }];
  let arguments = (ins StrAttr:$name,
                       PDL_Operation:$root,
                       Variadic<PDL_PositionalValue>:$args,
                       OptionalAttr<ArrayAttr>:$constParams);
  let assemblyFormat = [{
    $name ($constParams^)? (`(` $args^ `:` type($args) `)`)? `on` $root
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::AreEqualOp
//===----------------------------------------------------------------------===//

def PDLInterp_AreEqualOp
    : PDLInterp_PredicateOp<"are_equal", [NoSideEffect, SameTypeOperands]> {
  let summary = "Check if two positional values are equivalent";
  let description = [{
    `pdl_interp.are_equal` operations compare two positional values for
    equality. On success, this operation branches to the true destination,
    otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.are_equal %result1, %result2 : !pdl.value -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_PositionalValue:$lhs,
                       PDL_PositionalValue:$rhs);
  let assemblyFormat = "operands `:` type($lhs) attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::BranchOp
//===----------------------------------------------------------------------===//

def PDLInterp_BranchOp : PDLInterp_Op<"branch", [NoSideEffect, Terminator]> {
  let summary = "General branch operation";
  let description = [{
    `pdl_interp.branch` operations expose general branch functionality to the
    interpreter, and are generally used to branch from one pattern match
    sequence to another.

    Example:

    ```mlir
    pdl_interp.branch ^dest
    ```
  }];

  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckAttributeOp
    : PDLInterp_PredicateOp<"check_attribute", [NoSideEffect]> {
  let summary = "Check the value of an `Attribute`";
  let description = [{
    `pdl_interp.check_attribute` operations compare the value of a given
    attribute with a constant value. On success, this operation branches to the
    true destination, otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.check_attribute %attr is 10 -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Attribute:$attribute, AnyAttr:$constantValue);
  let assemblyFormat = [{
    $attribute `is` $constantValue attr-dict `->` successors
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckOperandCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckOperandCountOp
    : PDLInterp_PredicateOp<"check_operand_count", [NoSideEffect]> {
  let summary = "Check the number of operands of an `Operation`";
  let description = [{
    `pdl_interp.check_operand_count` operations compare the number of operands
    of a given operation value with a constant. On success, this operation
    branches to the true destination, otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.check_operand_count of %op is 2 -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Operation:$operation,
                       Confined<I32Attr, [IntNonNegative]>:$count);
  let assemblyFormat = "`of` $operation `is` $count attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckOperationNameOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckOperationNameOp
    : PDLInterp_PredicateOp<"check_operation_name", [NoSideEffect]> {
  let summary = "Check the OperationName of an `Operation`";
  let description = [{
    `pdl_interp.check_operation_name` operations compare the name of a given
    operation with a known name. On success, this operation branches to the true
    destination, otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.check_operation_name of %op is "foo.op" -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Operation:$operation, StrAttr:$name);
  let assemblyFormat = "`of` $operation `is` $name attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckResultCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckResultCountOp
    : PDLInterp_PredicateOp<"check_result_count", [NoSideEffect]> {
  let summary = "Check the number of results of an `Operation`";
  let description = [{
    `pdl_interp.check_result_count` operations compare the number of results
    of a given operation value with a constant. On success, this operation
    branches to the true destination, otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.check_result_count of %op is 0 -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Operation:$operation,
                       Confined<I32Attr, [IntNonNegative]>:$count);
  let assemblyFormat = "`of` $operation `is` $count attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckTypeOp
    : PDLInterp_PredicateOp<"check_type", [NoSideEffect]> {
  let summary = "Compare a type to a known value";
  let description = [{
    `pdl_interp.check_type` operations compare a type with a statically known
    type. On success, this operation branches to the true destination, otherwise
    the false destination is taken.

    Example:

    ```mlir
    pdl_interp.check_type %type is i32 -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Type:$value, TypeAttr:$type);
  let assemblyFormat = "$value `is` $type attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateAttributeOp
    : PDLInterp_Op<"create_attribute", [NoSideEffect]> {
  let summary = "Create an interpreter handle to a constant `Attribute`";
  let description = [{
    `pdl_interp.create_attribute` operations generate a handle within the
    interpreter for a specific constant attribute value.

    Example:

    ```mlir
    %attr = pdl_interp.create_attribute 10 : i64
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs PDL_Attribute:$attribute);
  let assemblyFormat = "$value attr-dict";

  let builders = [
    OpBuilderDAG<(ins "Attribute":$value), [{
      build($_builder, $_state, $_builder.getType<pdl::AttributeType>(), value);
    }]>];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateNativeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateNativeOp : PDLInterp_Op<"create_native"> {
  let summary = "Call a native creation method to construct an `Attribute`, "
                "`Operation`, `Type`, or `Value`";
  let description = [{
    `pdl_interp.create_native` operations invoke a native C++ function, that has
    been registered externally with the consumer of PDL, to create an
    `Attribute`, `Operation`, `Type`, or `Value`. The native function must
    produce a value of the specified return type, and may accept any number of
    positional arguments and constant attribute parameters.

    Example:

    ```mlir
    %ret = pdl_interp.create_native "myNativeFunc"[42, "gt"](%arg0, %arg1 : !pdl.value, !pdl.value) : !pdl.attribute
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_PositionalValue>:$args,
                       OptionalAttr<ArrayAttr>:$constParams);
  let results = (outs PDL_PositionalValue:$result);
  let assemblyFormat = [{
    $name ($constParams^)? (`(` $args^ `:` type($args) `)`)? `:` type($result)
    attr-dict
  }];
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateOperationOp
    : PDLInterp_Op<"create_operation", [AttrSizedOperandSegments]> {
  let summary = "Create an instance of a specific `Operation`";
  let description = [{
    `pdl_interp.create_operation` operations create an `Operation` instance with
    the specified attributes, operands, and result types.

    Example:

    ```mlir
    // Create an instance of a `foo.op` operation.
    %op = pdl_interp.create_operation "foo.op"(%arg0) {"attrA" = %attr0} -> %type, %type
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_Value>:$operands,
                       Variadic<PDL_Attribute>:$attributes,
                       StrArrayAttr:$attributeNames,
                       Variadic<PDL_Type>:$types);
  let results = (outs PDL_Operation:$operation);

  let builders = [
    OpBuilderDAG<(ins "StringRef":$name, "ValueRange":$types,
      "ValueRange":$operands, "ValueRange":$attributes,
      "ArrayAttr":$attributeNames), [{
      build($_builder, $_state, $_builder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types);
    }]>];
  let parser = [{ return ::parseCreateOperationOp(parser, result); }];
  let printer = [{ ::print(p, *this); }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateTypeOp : PDLInterp_Op<"create_type", [NoSideEffect]> {
  let summary = "Create an interpreter handle to a constant `Type`";
  let description = [{
    `pdl_interp.create_type` operations generate a handle within the interpreter
    for a specific constant type value.

    Example:

    ```mlir
    pdl_interp.create_type i64
    ```
  }];

  let arguments = (ins TypeAttr:$value);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "$value attr-dict";

  let builders = [
    OpBuilderDAG<(ins "TypeAttr":$type), [{
      build($_builder, $_state, $_builder.getType<pdl::TypeType>(), type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::EraseOp
//===----------------------------------------------------------------------===//

def PDLInterp_EraseOp : PDLInterp_Op<"erase"> {
  let summary = "Mark an operation as `erased`";
  let description = [{
    `pdl.erase` operations are used to specify that an operation should be
    marked as erased. The semantics of this operation correspond with the
    `eraseOp` method on a `PatternRewriter`.

    Example:

    ```mlir
    pdl_interp.erase %root
    ```
  }];

  let arguments = (ins PDL_Operation:$operation);
  let assemblyFormat = "$operation attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::FinalizeOp
//===----------------------------------------------------------------------===//

def PDLInterp_FinalizeOp
    : PDLInterp_Op<"finalize", [NoSideEffect, Terminator]> {
  let summary = "Finalize a pattern match or rewrite sequence";
  let description = [{
    `pdl_interp.finalize` is used to denote the termination of a match or
    rewrite sequence.

    Example:

    ```mlir
    pdl_interp.finalize
    ```
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetAttributeOp : PDLInterp_Op<"get_attribute", [NoSideEffect]> {
  let summary = "Get a specified attribute value from an `Operation`";
  let description = [{
    `pdl_interp.get_attribute` operations try to get a specific attribute from
    an operation. If the operation does not have that attribute, a null value is
    returned.

    Example:

    ```mlir
    %attr = pdl_interp.get_attribute "attr" of %op
    ```
  }];

  let arguments = (ins PDL_Operation:$operation,
                       StrAttr:$name);
  let results = (outs PDL_Attribute:$attribute);
  let assemblyFormat = "$name `of` $operation attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetAttributeTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetAttributeTypeOp
    : PDLInterp_Op<"get_attribute_type", [NoSideEffect]> {
  let summary = "Get the result type of a specified `Attribute`";
  let description = [{
    `pdl_interp.get_attribute_type` operations get the resulting type of a
    specific attribute.

    Example:

    ```mlir
    %type = pdl_interp.get_attribute_type of %attr
    ```
  }];

  let arguments = (ins PDL_Attribute:$value);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "`of` $value attr-dict";

  let builders = [
    OpBuilderDAG<(ins "Value":$value), [{
      build($_builder, $_state, $_builder.getType<pdl::TypeType>(), value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetDefiningOpOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetDefiningOpOp
    : PDLInterp_Op<"get_defining_op", [NoSideEffect]> {
  let summary = "Get the defining operation of a `Value`";
  let description = [{
    `pdl_interp.get_defining_op` operations try to get the defining operation
    of a specific value. If the value is not an operation result, null is
    returned.

    Example:

    ```mlir
    %op = pdl_interp.get_defining_op of %value
    ```
  }];

  let arguments = (ins PDL_Value:$value);
  let results = (outs PDL_Operation:$operation);
  let assemblyFormat = "`of` $value attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetOperandOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetOperandOp : PDLInterp_Op<"get_operand", [NoSideEffect]> {
  let summary = "Get a specified operand from an `Operation`";
  let description = [{
    `pdl_interp.get_operand` operations try to get a specific operand from an
    operation If the operation does not have an operand for the given index, a
    null value is returned.

    Example:

    ```mlir
    %operand = pdl_interp.get_operand 1 of %op
    ```
  }];

  let arguments = (ins PDL_Operation:$operation,
                       Confined<I32Attr, [IntNonNegative]>:$index);
  let results = (outs PDL_Value:$value);
  let assemblyFormat = "$index `of` $operation attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetResultOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetResultOp : PDLInterp_Op<"get_result", [NoSideEffect]> {
  let summary = "Get a specified result from an `Operation`";
  let description = [{
    `pdl_interp.get_result` operations try to get a specific result from an
    operation. If the operation does not have a result for the given index, a
    null value is returned.

    Example:

    ```mlir
    %result = pdl_interp.get_result 1 of %op
    ```
  }];

  let arguments = (ins PDL_Operation:$operation,
                       Confined<I32Attr, [IntNonNegative]>:$index);
  let results = (outs PDL_Value:$value);
  let assemblyFormat = "$index `of` $operation attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

// Get a type from the root operation, held in the rewriter context.
def PDLInterp_GetValueTypeOp : PDLInterp_Op<"get_value_type", [NoSideEffect]> {
  let summary = "Get the result type of a specified `Value`";
  let description = [{
    `pdl_interp.get_value_type` operations get the resulting type of a specific
    value.

    Example:

    ```mlir
    %type = pdl_interp.get_value_type of %value
    ```
  }];

  let arguments = (ins PDL_Value:$value);
  let results = (outs PDL_Type:$result);
  let assemblyFormat = "`of` $value attr-dict";

  let builders = [
    OpBuilderDAG<(ins "Value":$value), [{
      build($_builder, $_state, $_builder.getType<pdl::TypeType>(), value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::InferredTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_InferredTypeOp : PDLInterp_Op<"inferred_type"> {
  let summary = "Generate a handle to a Type that is \"inferred\"";
  let description = [{
    `pdl_interp.inferred_type` operations generate a handle to a type that
    should be inferred. This signals to other operations, such as
    `pdl_interp.create_operation`, that this type should be inferred.

    Example:

    ```mlir
    pdl_interp.inferred_type
    ```
  }];
  let results = (outs PDL_Type:$type);
  let assemblyFormat = "attr-dict";

  let builders = [
    OpBuilderDAG<(ins), [{
      build($_builder, $_state, $_builder.getType<pdl::TypeType>());
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::IsNotNullOp
//===----------------------------------------------------------------------===//

def PDLInterp_IsNotNullOp
    : PDLInterp_PredicateOp<"is_not_null", [NoSideEffect]> {
  let summary = "Check if a positional value is non-null";
  let description = [{
    `pdl_interp.is_not_null` operations check that a positional value exists. On
    success, this operation branches to the true destination. Otherwise, the
    false destination is taken.

    Example:

    ```mlir
    pdl_interp.is_not_null %value : !pdl.value -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_PositionalValue:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::RecordMatchOp
//===----------------------------------------------------------------------===//

def PDLInterp_RecordMatchOp
    : PDLInterp_Op<"record_match", [AttrSizedOperandSegments, Terminator]> {
  let summary = "Record the metadata for a successful pattern match";
  let description = [{
    `pdl_interp.record_match` operations record a successful pattern match with
    the interpreter and branch to the next part of the matcher. The metadata
    recorded by these operations correspond to a specific `pdl.pattern`, as well
    as what values were used during that match that should be propagated to the
    rewriter.

    Example:

    ```mlir
    pdl_interp.record_match @rewriters::myRewriter(%root : !pdl.operation) : benefit(1), loc([%root, %op1]), root("foo.op") -> ^nextDest
    ```
  }];

  let arguments = (ins Variadic<PDL_PositionalValue>:$inputs,
                       Variadic<PDL_Operation>:$matchedOps,
                       SymbolRefAttr:$rewriter,
                       OptionalAttr<StrAttr>:$rootKind,
                       OptionalAttr<StrArrayAttr>:$generatedOps,
                       Confined<I16Attr, [IntNonNegative]>:$benefit);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $rewriter (`(` $inputs^ `:` type($inputs) `)`)? `:`
    `benefit` `(` $benefit `)` `,`
    (`generatedOps` `(` $generatedOps^ `)` `,`)?
    `loc` `(` `[` $matchedOps `]` `)`
    (`,` `root` `(` $rootKind^ `)`)? attr-dict `->` $dest
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::ReplaceOp
//===----------------------------------------------------------------------===//

def PDLInterp_ReplaceOp : PDLInterp_Op<"replace"> {
  let summary = "Mark an operation as `replace`d";
  let description = [{
    `pdl_interp.replaced` operations are used to specify that an operation
    should be marked as replaced. The semantics of this operation correspond
    with the `replaceOp` method on a `PatternRewriter`. The set of replacement
    values must match the number of results specified by the operation.

    Example:

    ```mlir
    // Replace root node with 2 values:
    pdl_interp.replace %root with (%val0, %val1)
    ```
  }];
  let arguments = (ins PDL_Operation:$operation,
                       Variadic<PDL_Value>:$replValues);
  let assemblyFormat = "$operation `with` `(` $replValues `)` attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchAttributeOp
    : PDLInterp_SwitchOp<"switch_attribute", [NoSideEffect]> {
  let summary = "Switch on the value of an `Attribute`";
  let description = [{
    `pdl_interp.switch_attribute` operations compare the value of a given
    attribute with a set of constant attributes. If the value matches one of the
    provided case values the destination for that case value is taken, otherwise
    the default destination is taken.

    Example:

    ```mlir
    pdl_interp.switch_attribute %attr to [10, true](^10Dest, ^trueDest) -> ^defaultDest
    ```
  }];
  let arguments = (ins PDL_Attribute:$attribute, ArrayAttr:$caseValues);
  let assemblyFormat = [{
    $attribute `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$attribute, "ArrayRef<Attribute>":$caseValues,
      "Block *":$defaultDest, "BlockRange":$dests), [{
    build($_builder, $_state, attribute, $_builder.getArrayAttr(caseValues),
          defaultDest, dests);
  }]>];
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperandCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchOperandCountOp
    : PDLInterp_SwitchOp<"switch_operand_count", [NoSideEffect]> {
  let summary = "Switch on the operand count of an `Operation`";
  let description = [{
    `pdl_interp.switch_operand_count` operations compare the operand count of a
    given operation with a set of potential counts. If the value matches one of
    the provided case values the destination for that case value is taken,
    otherwise the default destination is taken.

    Example:

    ```mlir
    pdl_interp.switch_operand_count of %op to [10, 2] -> ^10Dest, ^2Dest, ^defaultDest
    ```
  }];

  let arguments = (ins PDL_Operation:$operation, I32ElementsAttr:$caseValues);
  let assemblyFormat = [{
    `of` $operation `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$operation, "ArrayRef<int32_t>":$counts,
      "Block *":$defaultDest, "BlockRange":$dests), [{
    build($_builder, $_state, operation, $_builder.getI32VectorAttr(counts),
          defaultDest, dests);
  }]>];
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperationNameOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchOperationNameOp
    : PDLInterp_SwitchOp<"switch_operation_name", [NoSideEffect]> {
  let summary = "Switch on the OperationName of an `Operation`";
  let description = [{
    `pdl_interp.switch_operation_name` operations compare the name of a given
    operation with a set of known names. If the value matches one of the
    provided case values the destination for that case value is taken, otherwise
    the default destination is taken.

    Example:

    ```mlir
    pdl_interp.switch_operation_name of %op to ["foo.op", "bar.op"](^fooDest, ^barDest) -> ^defaultDest
    ```
  }];

  let arguments = (ins PDL_Operation:$operation,
                       StrArrayAttr:$caseValues);
  let assemblyFormat = [{
    `of` $operation `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$operation, "ArrayRef<OperationName>":$names,
      "Block *":$defaultDest, "BlockRange":$dests), [{
      auto stringNames = llvm::to_vector<8>(llvm::map_range(names,
          [](OperationName name) { return name.getStringRef(); }));
      build($_builder, $_state, operation, $_builder.getStrArrayAttr(stringNames),
            defaultDest, dests);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchResultCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchResultCountOp
    : PDLInterp_SwitchOp<"switch_result_count", [NoSideEffect]> {
  let summary = "Switch on the result count of an `Operation`";
  let description = [{
    `pdl_interp.switch_result_count` operations compare the result count of a
    given operation with a set of potential counts. If the value matches one of
    the provided case values the destination for that case value is taken,
    otherwise the default destination is taken.

    Example:

    ```mlir
    pdl_interp.switch_result_count of %op to [0, 2](^0Dest, ^2Dest) -> ^defaultDest
    ```
  }];

  let arguments = (ins PDL_Operation:$operation, I32ElementsAttr:$caseValues);
  let assemblyFormat = [{
    `of` $operation `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$operation, "ArrayRef<int32_t>":$counts,
      "Block *":$defaultDest, "BlockRange":$dests), [{
    build($_builder, $_state, operation, $_builder.getI32VectorAttr(counts),
          defaultDest, dests);
  }]>];
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchTypeOp : PDLInterp_SwitchOp<"switch_type", [NoSideEffect]> {
  let summary = "Switch on a `Type` value";
  let description = [{
    `pdl_interp.switch_type` operations compare a type with a set of statically
    known types. If the value matches one of the provided case values the
    destination for that case value is taken, otherwise the default destination
    is taken.

    Example:

    ```mlir
    pdl_interp.switch_type %type to [i32, i64] -> ^i32Dest, ^i64Dest, ^defaultDest
    ```
  }];

  let arguments = (ins PDL_Type:$value, TypeArrayAttr:$caseValues);
  let assemblyFormat = [{
    $value `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$edge, "TypeRange":$types, "Block *":$defaultDest,
      "BlockRange":$dests), [{
      build($_builder, $_state, edge, $_builder.getTypeArrayAttr(types),
            defaultDest, dests);
    }]>,
  ];

  let extraClassDeclaration = [{
    auto getCaseTypes() { return caseValues().getAsValueRange<TypeAttr>(); }
  }];
}

#endif // MLIR_DIALECT_PDLINTERP_IR_PDLINTERPOPS
