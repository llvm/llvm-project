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
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
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
class PDLInterp_Op<string mnemonic, list<Trait> traits = []> :
    Op<PDLInterp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// PDLInterp_PredicateOp

// Check operations evaluate a predicate on a positional value and then
// conditionally branch on the result.
class PDLInterp_PredicateOp<string mnemonic, list<Trait> traits = []> :
    PDLInterp_Op<mnemonic, !listconcat([Terminator], traits)> {
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);
}

//===----------------------------------------------------------------------===//
// PDLInterp_SwitchOp

// Switch operations evaluate a predicate on a positional value and then
// conditionally branch on the result.
class PDLInterp_SwitchOp<string mnemonic, list<Trait> traits = []> :
    PDLInterp_Op<mnemonic, !listconcat([Terminator], traits)> {
  let successors = (successor AnySuccessor:$defaultDest,
                              VariadicSuccessor<AnySuccessor>:$cases);
}

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyConstraintOp
//===----------------------------------------------------------------------===//

def PDLInterp_ApplyConstraintOp : PDLInterp_PredicateOp<"apply_constraint"> {
  let summary = "Apply a constraint to a set of positional values";
  let description = [{
    `pdl_interp.apply_constraint` operations apply a generic constraint, that
    has been registered with the interpreter, with a given set of positional
    values. On success, this operation branches to the true destination,
    otherwise the false destination is taken.

    Example:

    ```mlir
    // Apply `myConstraint` to the entities defined by `input`, `attr`, and
    // `op`.
    pdl_interp.apply_constraint "myConstraint"(%input, %attr, %op : !pdl.value, !pdl.attribute, !pdl.operation) -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins StrAttr:$name, Variadic<PDL_AnyType>:$args);
  let assemblyFormat = [{
    $name `(` $args `:` type($args) `)` attr-dict `->` successors
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
    successful match. The rewrite is passed a set of positional arguments. The
    rewrite function may return any number of results.

    Example:

    ```mlir
    // Rewriter operating solely on the root operation.
    pdl_interp.apply_rewrite "rewriter"(%root : !pdl.operation)

    // Rewriter operating solely on the root operation and return an attribute.
    %attr = pdl_interp.apply_rewrite "rewriter"(%root : !pdl.operation) : !pdl.attribute

    // Rewriter operating on the root operation along with additional arguments
    // from the matcher.
    pdl_interp.apply_rewrite "rewriter"(%root : !pdl.operation, %value : !pdl.value)
    ```
  }];
  let arguments = (ins StrAttr:$name, Variadic<PDL_AnyType>:$args);
  let results = (outs Variadic<PDL_AnyType>:$results);
  let assemblyFormat = [{
    $name (`(` $args^ `:` type($args) `)`)? (`:` type($results)^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::AreEqualOp
//===----------------------------------------------------------------------===//

def PDLInterp_AreEqualOp
    : PDLInterp_PredicateOp<"are_equal", [Pure, SameTypeOperands]> {
  let summary = "Check if two positional values or ranges are equivalent";
  let description = [{
    `pdl_interp.are_equal` operations compare two positional values for
    equality. On success, this operation branches to the true destination,
    otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.are_equal %result1, %result2 : !pdl.value -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_AnyType:$lhs, PDL_AnyType:$rhs);
  let assemblyFormat = "operands `:` type($lhs) attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::BranchOp
//===----------------------------------------------------------------------===//

def PDLInterp_BranchOp : PDLInterp_Op<"branch", [Pure, Terminator]> {
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
    : PDLInterp_PredicateOp<"check_attribute", [Pure]> {
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
    : PDLInterp_PredicateOp<"check_operand_count", [Pure]> {
  let summary = "Check the number of operands of an `Operation`";
  let description = [{
    `pdl_interp.check_operand_count` operations compare the number of operands
    of a given operation value with a constant. The comparison is either exact
    or at_least, with the latter used to compare against a minimum number of
    expected operands. On success, this operation branches to the true
    destination, otherwise the false destination is taken.

    Example:

    ```mlir
    // Check for exact equality.
    pdl_interp.check_operand_count of %op is 2 -> ^matchDest, ^failureDest

    // Check for at least N operands.
    pdl_interp.check_operand_count of %op is at_least 2 -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Operation:$inputOp,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$count,
                       UnitAttr:$compareAtLeast);
  let assemblyFormat = [{
    `of` $inputOp `is` (`at_least` $compareAtLeast^)? $count attr-dict
    `->` successors
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckOperationNameOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckOperationNameOp
    : PDLInterp_PredicateOp<"check_operation_name", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp, StrAttr:$name);
  let assemblyFormat = "`of` $inputOp `is` $name attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckResultCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckResultCountOp
    : PDLInterp_PredicateOp<"check_result_count", [Pure]> {
  let summary = "Check the number of results of an `Operation`";
  let description = [{
    `pdl_interp.check_result_count` operations compare the number of results
    of a given operation value with a constant. The comparison is either exact
    or at_least, with the latter used to compare against a minimum number of
    expected results. On success, this operation branches to the true
    destination, otherwise the false destination is taken.

    Example:

    ```mlir
    // Check for exact equality.
    pdl_interp.check_result_count of %op is 2 -> ^matchDest, ^failureDest

    // Check for at least N results.
    pdl_interp.check_result_count of %op is at_least 2 -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_Operation:$inputOp,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$count,
                       UnitAttr:$compareAtLeast);
  let assemblyFormat = [{
    `of` $inputOp `is` (`at_least` $compareAtLeast^)? $count attr-dict
    `->` successors
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CheckTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckTypeOp
    : PDLInterp_PredicateOp<"check_type", [Pure]> {
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
// pdl_interp::CheckTypesOp
//===----------------------------------------------------------------------===//

def PDLInterp_CheckTypesOp
    : PDLInterp_PredicateOp<"check_types", [Pure]> {
  let summary = "Compare a range of types to a range of known values";
  let description = [{
    `pdl_interp.check_types` operations compare a range of types with a
    statically known range of types. On success, this operation branches
    to the true destination, otherwise the false destination is taken.

    Example:

    ```mlir
    pdl_interp.check_types %type are [i32, i64] -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_RangeOf<PDL_Type>:$value,
                       TypeArrayAttr:$types);
  let assemblyFormat = "$value `are` $types attr-dict `->` successors";
}

//===----------------------------------------------------------------------===//
// pdl_interp::ContinueOp
//===----------------------------------------------------------------------===//

def PDLInterp_ContinueOp
    : PDLInterp_Op<"continue", [Pure, HasParent<"ForEachOp">,
                               Terminator]> {
  let summary = "Breaks the current iteration";
  let description = [{
    `pdl_interp.continue` operation breaks the current iteration within the
    `pdl_interp.foreach` region and continues with the next iteration from
    the beginning of the region.

    Example:

    ```mlir
    pdl_interp.continue
    ```
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateAttributeOp
    : PDLInterp_Op<"create_attribute", [Pure]> {
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
  let assemblyFormat = "$value attr-dict-with-keyword";

  let builders = [
    OpBuilder<(ins "Attribute":$value), [{
      build($_builder, $_state, $_builder.getType<pdl::AttributeType>(), value);
    }]>];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateOperationOp
    : PDLInterp_Op<"create_operation", [AttrSizedOperandSegments]> {
  let summary = "Create an instance of a specific `Operation`";
  let description = [{
    `pdl_interp.create_operation` operations create an `Operation` instance with
    the specified attributes, operands, and result types. See `pdl.operation`
    for a more detailed description on the general interpretation of the arguments
    to this operation.

    Example:

    ```mlir
    // Create an instance of a `foo.op` operation.
    %op = pdl_interp.create_operation "foo.op"(%arg0 : !pdl.value) {"attrA" = %attr0} -> (%type : !pdl.type)

    // Create an instance of a `foo.op` operation that has inferred result types
    // (using the InferTypeOpInterface).
    %op = pdl_interp.create_operation "foo.op"(%arg0 : !pdl.value) {"attrA" = %attr0} -> <inferred>
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       Variadic<PDL_InstOrRangeOf<PDL_Value>>:$inputOperands,
                       Variadic<PDL_Attribute>:$inputAttributes,
                       StrArrayAttr:$inputAttributeNames,
                       Variadic<PDL_InstOrRangeOf<PDL_Type>>:$inputResultTypes,
                       UnitAttr:$inferredResultTypes);
  let results = (outs PDL_Operation:$resultOp);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "ValueRange":$types,
      "bool":$inferredResultTypes, "ValueRange":$operands,
      "ValueRange":$attributes, "ArrayAttr":$attributeNames), [{
      build($_builder, $_state, $_builder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types, inferredResultTypes);
    }]>
  ];
  let assemblyFormat = [{
    $name (`(` $inputOperands^ `:` type($inputOperands) `)`)? ``
    custom<CreateOperationOpAttributes>($inputAttributes, $inputAttributeNames)
    custom<CreateOperationOpResults>($inputResultTypes, type($inputResultTypes),
                                     $inferredResultTypes)
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateTypeOp : PDLInterp_Op<"create_type", [Pure]> {
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
    OpBuilder<(ins "TypeAttr":$type), [{
      build($_builder, $_state, $_builder.getType<pdl::TypeType>(), type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateTypesOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateTypesOp : PDLInterp_Op<"create_types", [Pure]> {
  let summary = "Create an interpreter handle to a range of constant `Type`s";
  let description = [{
    `pdl_interp.create_types` operations generate a handle within the
    interpreter for a specific range of constant type values.

    Example:

    ```mlir
    pdl_interp.create_types [i64, i64]
    ```
  }];

  let arguments = (ins TypeArrayAttr:$value);
  let results = (outs PDL_RangeOf<PDL_Type>:$result);
  let assemblyFormat = "$value attr-dict";

  let builders = [
    OpBuilder<(ins "ArrayAttr":$type), [{
      build($_builder, $_state,
            pdl::RangeType::get($_builder.getType<pdl::TypeType>()), type);
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

  let arguments = (ins PDL_Operation:$inputOp);
  let assemblyFormat = "$inputOp attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::ExtractOp
//===----------------------------------------------------------------------===//

def PDLInterp_ExtractOp
    : PDLInterp_Op<"extract", [Pure,
     TypesMatchWith<
        "`range` is a PDL range whose element type matches type of `result`",
        "result", "range", "pdl::RangeType::get($_self)">]> {
  let summary = "Extract the item at the specified index in a range";
  let description = [{
    `pdl_interp.extract` operations are used to extract an item from a range
    at the specified index. If the index is out of range, returns null.

    Example:

    ```mlir
    // Extract the value at index 1 from a range of values.
    %ops = pdl_interp.extract 1 of %values : !pdl.value
    ```
  }];

  let arguments = (ins PDL_RangeOf<PDL_AnyType>:$range,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$index);
  let results = (outs PDL_AnyType:$result);
  let assemblyFormat = "$index `of` $range `:` type($result) attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$range, "unsigned":$index), [{
      build($_builder, $_state,
            range.getType().cast<pdl::RangeType>().getElementType(),
            range, index);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::FinalizeOp
//===----------------------------------------------------------------------===//

def PDLInterp_FinalizeOp
    : PDLInterp_Op<"finalize", [Pure, Terminator]> {
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
// pdl_interp::ForEachOp
//===----------------------------------------------------------------------===//

def PDLInterp_ForEachOp
    : PDLInterp_Op<"foreach", [Terminator]> {
  let summary = "Iterates over a range of values or ranges";
  let description = [{
    `pdl_interp.foreach` iteratively selects an element from a range of values
    and executes the region until pdl.continue is reached.

    In the bytecode interpreter, this operation is implemented by looping over
    the values and, for each selection, running the bytecode until we reach
    pdl.continue. This may result in multiple matches being reported. Note
    that the input range is mutated (popped from).

    Example:

    ```mlir
    pdl_interp.foreach %op : !pdl.operation in %ops {
      pdl_interp.continue
    } -> ^next
    ```
  }];

  let arguments = (ins PDL_RangeOf<PDL_AnyType>:$values);
  let regions = (region AnyRegion:$region);
  let successors = (successor AnySuccessor:$successor);

  let builders = [
    OpBuilder<(ins "Value":$range, "Block *":$successor, "bool":$initLoop)>
  ];

  let extraClassDeclaration = [{
    /// Returns the loop variable.
    BlockArgument getLoopVariable() { return getRegion().getArgument(0); }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::FuncOp
//===----------------------------------------------------------------------===//

def PDLInterp_FuncOp : PDLInterp_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, Symbol
  ]> {
  let summary = "PDL Interpreter Function Operation";
  let description = [{
    `pdl_interp.func` operations act as interpreter functions. These are
    callable SSA-region operations that contain other interpreter operations.
    Interpreter functions are used for both the matching and the rewriting
    portion of the interpreter.

    Example:

    ```mlir
    pdl_interp.func @rewriter(%root: !pdl.operation) {
      %op = pdl_interp.create_operation "foo.new_operation"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region MinSizedRegion<1>:$body);

  // Create the function with the given name and type. This also automatically
  // inserts the entry block for the function.
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetAttributeOp : PDLInterp_Op<"get_attribute", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp, StrAttr:$name);
  let results = (outs PDL_Attribute:$attribute);
  let assemblyFormat = "$name `of` $inputOp attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetAttributeTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetAttributeTypeOp
    : PDLInterp_Op<"get_attribute_type", [Pure]> {
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
    OpBuilder<(ins "Value":$value), [{
      build($_builder, $_state, $_builder.getType<pdl::TypeType>(), value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetDefiningOpOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetDefiningOpOp
    : PDLInterp_Op<"get_defining_op", [Pure]> {
  let summary = "Get the defining operation of a `Value`";
  let description = [{
    `pdl_interp.get_defining_op` operations try to get the defining operation
    of a specific value or range of values. In the case of range, the defining
    op of the first value is returned. If the value is not an operation result
    or range of operand results, null is returned.

    Example:

    ```mlir
    %op = pdl_interp.get_defining_op of %value : !pdl.value
    ```
  }];

  let arguments = (ins PDL_InstOrRangeOf<PDL_Value>:$value);
  let results = (outs PDL_Operation:$inputOp);
  let assemblyFormat = "`of` $value `:` type($value) attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetOperandOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetOperandOp : PDLInterp_Op<"get_operand", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$index);
  let results = (outs PDL_Value:$value);
  let assemblyFormat = "$index `of` $inputOp attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetOperandsOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetOperandsOp : PDLInterp_Op<"get_operands", [Pure]> {
  let summary = "Get a specified operand group from an `Operation`";
  let description = [{
    `pdl_interp.get_operands` operations try to get a specific operand
    group from an operation. If the expected result is a single Value, null is
    returned if the operand group is not of size 1. If a range is expected,
    null is returned if the operand group is invalid. If no index is provided,
    the returned operand group corresponds to all operands of the operation.

    Example:

    ```mlir
    // Get the first group of operands from an operation, and expect a single
    // element.
    %operand = pdl_interp.get_operands 0 of %op : !pdl.value

    // Get the first group of operands from an operation.
    %operands = pdl_interp.get_operands 0 of %op : !pdl.range<value>

    // Get all of the operands from an operation.
    %operands = pdl_interp.get_operands of %op : !pdl.range<value>
    ```
  }];

  let arguments = (ins
    PDL_Operation:$inputOp,
    OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$index
  );
  let results = (outs PDL_InstOrRangeOf<PDL_Value>:$value);
  let assemblyFormat = "($index^)? `of` $inputOp `:` type($value) attr-dict";
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$inputOp,
                   "Optional<unsigned>":$index), [{
      build($_builder, $_state, resultType, inputOp,
            index ? $_builder.getI32IntegerAttr(*index) : IntegerAttr());
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetResultOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetResultOp : PDLInterp_Op<"get_result", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp,
                       ConfinedAttr<I32Attr, [IntNonNegative]>:$index);
  let results = (outs PDL_Value:$value);
  let assemblyFormat = "$index `of` $inputOp attr-dict";
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetResultsOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetResultsOp : PDLInterp_Op<"get_results", [Pure]> {
  let summary = "Get a specified result group from an `Operation`";
  let description = [{
    `pdl_interp.get_results` operations try to get a specific result group
    from an operation. If the expected result is a single Value, null is
    returned if the result group is not of size 1. If a range is expected,
    null is returned if the result group is invalid. If no index is provided,
    the returned operand group corresponds to all results of the operation.

    Example:

    ```mlir
    // Get the first group of results from an operation, and expect a single
    // element.
    %result = pdl_interp.get_results 0 of %op : !pdl.value

    // Get the first group of results from an operation.
    %results = pdl_interp.get_results 0 of %op : !pdl.range<value>

    // Get all of the results from an operation.
    %results = pdl_interp.get_results of %op : !pdl.range<value>
    ```
  }];

  let arguments = (ins
    PDL_Operation:$inputOp,
    OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$index
  );
  let results = (outs PDL_InstOrRangeOf<PDL_Value>:$value);
  let assemblyFormat = "($index^)? `of` $inputOp `:` type($value) attr-dict";
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$inputOp,
                   "Optional<unsigned>":$index), [{
      build($_builder, $_state, resultType, inputOp,
            index ? $_builder.getI32IntegerAttr(*index) : IntegerAttr());
    }]>,
    OpBuilder<(ins "Value":$inputOp), [{
      build($_builder, $_state,
            pdl::RangeType::get($_builder.getType<pdl::ValueType>()), inputOp,
            IntegerAttr());
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetUsersOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetUsersOp
    : PDLInterp_Op<"get_users", [Pure]> {
  let summary = "Get the users of a `Value`";
  let description = [{
    `pdl_interp.get_users` extracts the users that accept this value. In the
    case of a range, the union of users of the all the values are returned,
    similarly to ResultRange::getUsers.

    Example:

    ```mlir
    // Get all the users of a single value.
    %ops = pdl_interp.get_users of %value : !pdl.value

    // Get all the users of the first value in a range.
    %ops = pdl_interp.get_users of %values : !pdl.range<value>
    ```
  }];

  let arguments = (ins PDL_InstOrRangeOf<PDL_Value>:$value);
  let results = (outs PDL_RangeOf<PDL_Operation>:$operations);
  let assemblyFormat = "`of` $value `:` type($value) attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$value), [{
      build($_builder, $_state,
            pdl::RangeType::get($_builder.getType<pdl::OperationType>()),
            value);
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_GetValueTypeOp : PDLInterp_Op<"get_value_type", [Pure,
     TypesMatchWith<"`value` type matches arity of `result`",
                    "result", "value", "getGetValueTypeOpValueType($_self)">]> {
  let summary = "Get the result type of a specified `Value`";
  let description = [{
    `pdl_interp.get_value_type` operations get the resulting type of a specific
    value or range thereof.

    Example:

    ```mlir
    // Get the type of a single value.
    %type = pdl_interp.get_value_type of %value : !pdl.type

    // Get the types of a value range.
    %type = pdl_interp.get_value_type of %values : !pdl.range<type>
    ```
  }];

  let arguments = (ins PDL_InstOrRangeOf<PDL_Value>:$value);
  let results = (outs PDL_InstOrRangeOf<PDL_Type>:$result);
  let assemblyFormat = "`of` $value `:` type($result) attr-dict";

  let builders = [
    OpBuilder<(ins "Value":$value), [{
      Type valType = value.getType();
      Type typeType = $_builder.getType<pdl::TypeType>();
      build($_builder, $_state,
            valType.isa<pdl::RangeType>() ? pdl::RangeType::get(typeType)
                                          : typeType,
            value);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// pdl_interp::IsNotNullOp
//===----------------------------------------------------------------------===//

def PDLInterp_IsNotNullOp
    : PDLInterp_PredicateOp<"is_not_null", [Pure]> {
  let summary = "Check if a positional value is non-null";
  let description = [{
    `pdl_interp.is_not_null` operations check that a positional value or range
    exists. For ranges, this does not mean that the range was simply empty. On
    success, this operation branches to the true destination. Otherwise, the
    false destination is taken.

    Example:

    ```mlir
    pdl_interp.is_not_null %value : !pdl.value -> ^matchDest, ^failureDest
    ```
  }];

  let arguments = (ins PDL_AnyType:$value);
  let assemblyFormat = "$value `:` type($value) attr-dict `->` successors";
}


//===----------------------------------------------------------------------===//
// pdl_interp::CreateRangeOp
//===----------------------------------------------------------------------===//

def PDLInterp_CreateRangeOp : PDLInterp_Op<"create_range", [Pure]> {
  let summary = "Construct a range of PDL entities";
  let description = [{
    `pdl_interp.create_range` operations construct a range from a given set of PDL
    entities, which all share the same underlying element type. For example, a
    `!pdl.range<value>` may be constructed from a list of `!pdl.value`
    or `!pdl.range<value>` entities.

    Example:

    ```mlir
    // Construct a range of values.
    %valueRange = pdl_interp.create_range %inputValue, %inputRange : !pdl.value, !pdl.range<value>

    // Construct a range of types.
    %typeRange = pdl_interp.create_range %inputType, %inputRange : !pdl.type, !pdl.range<type>

    // Construct an empty range of types.
    %valueRange = pdl_interp.create_range : !pdl.range<type>
    ```
  }];

  let arguments = (ins Variadic<PDL_AnyType>:$arguments);
  let results = (outs PDL_RangeOf<AnyTypeOf<[PDL_Type, PDL_Value]>>:$result);
  let assemblyFormat = [{
    ($arguments^ `:` type($arguments))?
    custom<RangeType>(ref(type($arguments)), type($result))
    attr-dict
  }];
  let hasVerifier = 1;
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

  let arguments = (ins Variadic<PDL_AnyType>:$inputs,
                       Variadic<PDL_Operation>:$matchedOps,
                       SymbolRefAttr:$rewriter,
                       OptionalAttr<StrAttr>:$rootKind,
                       OptionalAttr<StrArrayAttr>:$generatedOps,
                       ConfinedAttr<I16Attr, [IntNonNegative]>:$benefit);
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
    pdl_interp.replace %root with (%val0, %val1 : !pdl.type, !pdl.type)
    ```
  }];
  let arguments = (ins PDL_Operation:$inputOp,
                       Variadic<PDL_InstOrRangeOf<PDL_Value>>:$replValues);
  let assemblyFormat = [{
    $inputOp `with` ` ` `(` ($replValues^ `:` type($replValues))? `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchAttributeOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchAttributeOp
    : PDLInterp_SwitchOp<"switch_attribute", [Pure]> {
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
    OpBuilder<(ins "Value":$attribute, "ArrayRef<Attribute>":$caseValues,
      "Block *":$defaultDest, "BlockRange":$dests), [{
    build($_builder, $_state, attribute, $_builder.getArrayAttr(caseValues),
          defaultDest, dests);
  }]>];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperandCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchOperandCountOp
    : PDLInterp_SwitchOp<"switch_operand_count", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp, I32ElementsAttr:$caseValues);
  let assemblyFormat = [{
    `of` $inputOp `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilder<(ins "Value":$inputOp, "ArrayRef<int32_t>":$counts,
                   "Block *":$defaultDest, "BlockRange":$dests), [{
    build($_builder, $_state, inputOp, $_builder.getI32VectorAttr(counts),
          defaultDest, dests);
  }]>];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperationNameOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchOperationNameOp
    : PDLInterp_SwitchOp<"switch_operation_name", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp,
                       StrArrayAttr:$caseValues);
  let assemblyFormat = [{
    `of` $inputOp `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilder<(ins "Value":$inputOp, "ArrayRef<OperationName>":$names,
      "Block *":$defaultDest, "BlockRange":$dests), [{
      auto stringNames = llvm::to_vector<8>(llvm::map_range(names,
          [](OperationName name) { return name.getStringRef(); }));
      build($_builder, $_state, inputOp, $_builder.getStrArrayAttr(stringNames),
            defaultDest, dests);
    }]>,
  ];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchResultCountOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchResultCountOp
    : PDLInterp_SwitchOp<"switch_result_count", [Pure]> {
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

  let arguments = (ins PDL_Operation:$inputOp, I32ElementsAttr:$caseValues);
  let assemblyFormat = [{
    `of` $inputOp `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilder<(ins "Value":$inputOp, "ArrayRef<int32_t>":$counts,
                   "Block *":$defaultDest, "BlockRange":$dests), [{
    build($_builder, $_state, inputOp, $_builder.getI32VectorAttr(counts),
          defaultDest, dests);
  }]>];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypeOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchTypeOp : PDLInterp_SwitchOp<"switch_type", [Pure]> {
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
    OpBuilder<(ins "Value":$edge, "ArrayRef<Attribute>":$types,
                   "Block *":$defaultDest, "BlockRange":$dests), [{
      build($_builder, $_state, edge, $_builder.getArrayAttr(types),
            defaultDest, dests);
    }]>,
  ];

  let extraClassDeclaration = [{
    auto getCaseTypes() { return getCaseValues().getAsValueRange<TypeAttr>(); }
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypesOp
//===----------------------------------------------------------------------===//

def PDLInterp_SwitchTypesOp : PDLInterp_SwitchOp<"switch_types",
                                                 [Pure]> {
  let summary = "Switch on a range of `Type` values";
  let description = [{
    `pdl_interp.switch_types` operations compare a range of types with a set of
    statically known ranges. If the value matches one of the provided case
    values the destination for that case value is taken, otherwise the default
    destination is taken.

    Example:

    ```mlir
    pdl_interp.switch_types %type is [[i32], [i64, i64]] -> ^i32Dest, ^i64Dest, ^defaultDest
    ```
  }];

  let arguments = (ins
    PDL_RangeOf<PDL_Type>:$value,
    TypedArrayAttrBase<TypeArrayAttr, "type-array array attribute">:$caseValues
  );
  let assemblyFormat = [{
    $value `to` $caseValues `(` $cases `)` attr-dict `->` $defaultDest
  }];

  let builders = [
    OpBuilder<(ins "Value":$edge, "ArrayRef<Attribute>":$types,
                   "Block *":$defaultDest, "BlockRange":$dests), [{
      build($_builder, $_state, edge, $_builder.getArrayAttr(types),
            defaultDest, dests);
    }]>,
  ];

  let extraClassDeclaration = [{
    auto getCaseTypes() { return getCaseValues().getAsRange<ArrayAttr>(); }
  }];
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_PDLINTERP_IR_PDLINTERPOPS
