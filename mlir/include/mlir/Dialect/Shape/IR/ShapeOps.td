//===- Shape.td - Shape operations definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Shape dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef SHAPE_OPS
#define SHAPE_OPS

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Shape op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Shape_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ShapeDialect, mnemonic, traits>;

def Shape_AddOp : Shape_Op<"add", [Commutative, NoSideEffect]> {
  let summary = "Addition of sizes and indices";
  let description = [{
    Adds two sizes or indices. If either operand is an error it will be
    propagated to the result. The operands can be of type `size` or `index`. If
    at least one of the operands can hold an error, i.e. if it is of type `size`,
    then also the result must be of type `size`.
  }];

  let arguments = (ins Shape_SizeOrIndexType:$lhs, Shape_SizeOrIndexType:$rhs);
  let results = (outs Shape_SizeOrIndexType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let verifier = [{ return verifySizeOrIndexOp(*this); }];
}

def Shape_BroadcastOp : Shape_Op<"broadcast", [Commutative]> {
  let summary = "Returns the broadcasted output shape of two inputs";
  let description = [{
    Returns the broadcasted shape for two input shapes or extent tensors. Both
    operands can be of type `shape.shape` or `tensor<?xindex>`. The result is of
    type `shape.shape` and, if both operands are tensors, may be of type
    `tensor<?xindex>`.

    If the two operand shapes are of different rank the smaller one is padded
    with 1's from the left. The resulting broadcasted shape is then defined as

        result[i] = lhs[i] if lhs[i] == rhs[i]
                  = lhs[i] if rhs[i] == 1
                  = rhs[i] if lhs[i] == 1.

    In case the resulting shape is undefined, i.e. if corresponding extents are
    different from each other but none is 1, the result is an error shape.
    Likewise error values are propagated if any of the operands holds an error
    value. If the result type is an extent tensor (and can therefore not hold
    the error value) the behavior may be undefined. The optional string
    attribute can be used to describe the error case.
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$lhs,
                       Shape_ShapeOrExtentTensorType:$rhs,
                       OptionalAttr<StrAttr>:$error);
  let results = (outs Shape_ShapeOrExtentTensorType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let verifier = [{ return ::verifyShapeOrExtentTensorOp(*this); }];
  let hasFolder = 1;

  let verifier = [{ return ::verifyShapeOrExtentTensorOp(*this); }];
}

def Shape_ConstShapeOp : Shape_Op<"const_shape", [ConstantLike, NoSideEffect]> {
  let summary = "Creates a constant shape or extent tensor";
  let description = [{
    Creates a constant shape or extent tensor. The individual extents are given
    as the `shape` attribute. The number of these values equals the shape's
    rank.

    ```mlir
    %0 = shape.const_shape [] : !shape.shape
    %1 = shape.const_shape [1, 2, 3] : !shape.shape
    %2 = shape.const_shape [4, 5, 6] : tensor<?xindex>
    ```
  }];
  let arguments = (ins IndexElementsAttr:$shape);
  let results = (outs Shape_ShapeOrExtentTensorType:$result);

  // TODO: Move this to main so that all shape ops implement these.
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let hasFolder = 1;
}

def Shape_ConstSizeOp : Shape_Op<"const_size", [
    ConstantLike,
    NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface>
  ]> {
  let summary = "Creates a constant of type `shape.size`";
  let description = [{
    Creates a `shape.size` type representing the constant size given by `value`.

    ```mlir
    %x = shape.const_size 10
    ```
  }];

  let arguments = (ins IndexAttr:$value);
  let results = (outs Shape_SizeType:$result);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, int64_t value">
  ];

  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def Shape_ShapeEqOp : Shape_Op<"shape_eq", [Commutative, NoSideEffect]> {
  let summary = "Returns whether the input shapes or extent tensors are equal";
  let description = [{
    Takes two shape or extent tensor operands and determines whether they are
    equal. When extent tensors are compared to shapes they are regarded as their
    equivalent non-error shapes. Error shapes can be tested for equality like
    any other shape value, meaning that the error value is equal to itself.
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$lhs,
                       Shape_ShapeOrExtentTensorType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
  let hasFolder = 1;
}

def Shape_FromExtentsOp : Shape_Op<"from_extents", [NoSideEffect]> {
  let summary = "Creates a shape from extents";
  let description = [{
    Creates a shape from multiple SSA values representing the extents of
    the shape.

    ```mlir
    // Rank 2 shape.
    %s0 = shape.from_extents %a, %b
    // Rank 0 shape.
    %s1 = shape.from_extents
    ```
  }];
  let arguments = (ins Variadic<Index>:$extents);
  let results = (outs Shape_ShapeType:$shape);

  let assemblyFormat = "$extents attr-dict";

  let hasFolder = 1;
}

def Shape_FromExtentTensorOp : Shape_Op<"from_extent_tensor", [NoSideEffect]> {
  let summary = "Creates a shape from a tensor of extents";
  let description = [{
    Creates a shape from a 1D integral tensor of extents. The rank of the
    resulting shape equals the number of elements in the tensor, and the
    extents match the values of the elements.
  }];

  let arguments = (ins IndexTensor:$input);
  let results = (outs Shape_ShapeType:$result);

  let assemblyFormat = "attr-dict $input `:` type($input)";
}

def Shape_RankOp : Shape_Op<"rank", [NoSideEffect]> {
  let summary = "Gets the rank of a shape";
  let description = [{
    Returns the rank of the shape or extent tensor, i.e. the number of extents.
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$shape);
  let results = (outs Shape_SizeOrIndexType:$rank);

  let assemblyFormat = "$shape `:` type($shape) `->` type($rank) attr-dict";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let verifier = [{ return ::verifySizeOrIndexOp(*this); }];
}

def Shape_ToExtentTensorOp : Shape_Op<"to_extent_tensor", [NoSideEffect]> {
  let summary = "Creates a dimension tensor from a shape";
  let description = [{
    Converts a shape to a 1D integral tensor of extents. The number of elements
    in the tensor equals the rank of the shape, and the elements equal the
    extents of the shape.

    If the shape represents an error, this op's behavior is undefined.
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$input);
  let results = (outs IndexTensor:$result);

  let assemblyFormat = "attr-dict $input `:` type($input) `->` type($result)";

  let hasFolder = 1;
}

def Shape_GetExtentOp : Shape_Op<"get_extent", [NoSideEffect]> {
  let summary = "Gets the specified extent from a shape or extent tensor";
  let description = [{
    Gets the extent indexed by `dim` from the `shape` operand. If the shape is
    an error then it returns an error size.
  }];
  let arguments = (ins Shape_ShapeOrExtentTensorType:$shape,
                       Shape_SizeOrIndexType:$dim);
  let results = (outs Shape_SizeOrIndexType:$extent);
  let assemblyFormat = "$shape `,` $dim `:` type($shape) `,` type($dim) `->` "
                       "type($extent) attr-dict";

  let builders = [
    // Builder that allows passing a constant dimension as a simple integer.
    OpBuilder<"OpBuilder &builder, OperationState &result, Value shape, "
              "int64_t dim">
  ];

  let extraClassDeclaration = [{
    /// Get the `dim` value as integer if it is constant.
    Optional<int64_t> getConstantDim();
  }];

  let hasFolder = 1;
  let verifier = [{ return ::verifySizeOrIndexOp(*this); }];
}

def Shape_IndexToSizeOp : Shape_Op<"index_to_size", [NoSideEffect]> {
  let summary = "Converts a standard index to a shape size";
  let description = [{
    Converts a standard index to a `shape.size`. This operation and its
    inverse, `size_to_index`, facilitate index conversion between the standard
    and the shape dialect.

    The behavior is undefined for negative indices.
  }];

  let arguments = (ins Index:$arg);
  let results = (outs Shape_SizeType:$result);

  let assemblyFormat = "$arg attr-dict";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Shape_JoinOp : Shape_Op<"join", [Commutative]> {
  let summary = "Returns the least general shape.size of its operands";
  let description = [{
    An operation that computes the least general shape of input operands.
    This effectively asserts that corresponding static dimensions are equal.
    The behavior is to match each element of the `shape.shape` and propagate the
    most restrictive information, returning an invalid shape if there are
    contradictory requirements. E.g., using pseudo code

    ```
    shape.join([*], [*]) -> [*]
    shape.join([*], [1, ?]) -> [1, ?]
    shape.join([1, 2], [1, ?]) -> [1, 2]
    shape.join([*], [1, 2]) -> [1, 2]
    shape.join([], []) -> []
    shape.join([], [*]) -> []
    shape.join([], [?, ?]) -> [invalid]
    shape.join([1, ?], [2, ?, ?]) -> [invalid]
    ```

    `shape.join` also allows specifying an optional error string, that may be
    used to return an error to the user upon mismatch of dimensions.

    ```mlir
    %c = shape.join %a, %b, error="<reason>" : !shape.shape
    ```
  }];

  let arguments = (ins Shape_ShapeOrSizeType:$arg0, Shape_ShapeOrSizeType:$arg1,
                   OptionalAttr<StrAttr>:$error);
  let results = (outs Shape_ShapeOrSizeType:$result);
}

def Shape_MulOp : Shape_Op<"mul", [Commutative, NoSideEffect]> {
  let summary = "Multiplication of sizes and indices";
  let description = [{
    Multiplies two sizes or indices. If either operand is an error it will be
    propagated to the result. The operands can be of type `size` or `index`. If
    at least one of the operands can hold an error, i.e. if it is of type `size`,
    then also the result must be of type `size`. If error propagation is not
    possible because both operands are of type `index` then the result must also
    be of type `index`.
  }];

  let arguments = (ins Shape_SizeOrIndexType:$lhs, Shape_SizeOrIndexType:$rhs);
  let results = (outs Shape_SizeOrIndexType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict
  }];

  let verifier = [{ return ::verifySizeOrIndexOp(*this); }];
  let hasFolder = 1;
}

def Shape_NumElementsOp : Shape_Op<"num_elements", [NoSideEffect]> {
  let summary = "Returns the number of elements for a given shape";
  let description = [{
    Returns the number of elements for a given shape which is the product of its
    extents. If the argument is of type `shape` then the result will be of type
    `size` and potential errors will be propagated. Otherwise, if the argument
    is and extent tensor `tensor<?xindex>` then the result will be of type
    `index`.
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$shape);
  let results = (outs Shape_SizeOrIndexType:$result);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value shape">,
  ];

  let assemblyFormat = "$shape `:` type($shape) `->` type($result) attr-dict";

  let hasFolder = 1;
  let verifier = [{ return ::verifySizeOrIndexOp(*this); }];
}

def Shape_ReduceOp : Shape_Op<"reduce",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Returns an expression reduced over a shape or extent tensor";
  let description = [{
    An operation that takes as input a shape or extent tensor, and a number of
    initial values. This operation has a region/function that is applied
    repeatedly for every extent of the input. Starting with the initial values,
    the individual extents are then aggregated as defined by the associated
    region.

    Conceptually this op performs the following reduction:

    ```
    res[] = init;
    for (int i = 0, i < shape.rank(); i++) {
      res = fn(i, shape[i], res[0], ..., res[n]);
    }
    ```

    Where `fn` is provided by the user and the result of the reduce op is the
    last computed output of the reduce function. As an example, computing the
    number of elements can be defined as follows:

    ```mlir
    func @reduce(%shape : !shape.shape, %init : !shape.size) -> !shape.size {
      %num_elements = shape.reduce(%shape, %init) -> !shape.size  {
        ^bb0(%index: index, %dim: !shape.size, %acc: !shape.size):
          %updated_acc = "shape.mul"(%acc, %dim) :
            (!shape.size, !shape.size) -> !shape.size
          shape.yield %updated_acc : !shape.size
      }
      return %num_elements : !shape.size
    }
    ```
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$shape,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$result);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, "
              "Value shape, ValueRange initVals">,
  ];

  let verifier = [{ return ::verify(*this); }];
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def Shape_ShapeOfOp : Shape_Op<"shape_of", [NoSideEffect]> {
  let summary = "Returns shape of a value or shaped type operand";

  let description = [{
    The operation takes a value or a shaped operand as an argument and it
    returns a shape or extent tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyShaped, Shape_ValueShapeType]>:$arg);
  let results = (outs Shape_ShapeOrExtentTensorType:$result);

  let assemblyFormat = "$arg `:` type($arg) `->` type($result) attr-dict";

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, Value arg">
  ];

  let verifier = [{ return ::verifyShapeOrExtentTensorOp(*this); }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Shape_SizeToIndexOp : Shape_Op<"size_to_index", [NoSideEffect]> {
  let summary = "Casts between index types of the shape and standard dialect";
  let description = [{
    Converts a `shape.size` to a standard index. This operation and its
    inverse, `index_to_size`, facilitate index conversion between the standard
    and the shape dialect. The behavior is undefined for unknown and invalid
    arguments.
  }];

  let arguments = (ins Shape_SizeOrIndexType:$arg);
  let results = (outs Index:$result);

  let assemblyFormat = "$arg attr-dict `:` type($arg)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Shape_WithOp : Shape_Op<"with_shape", [NoSideEffect]> {
  let summary = "Returns ValueShape with given shape";
  let description = [{
    Returns ValueShape with the shape updated to match the shape operand. That
    is a new ValueShape tuple is created with value equal to `operand`'s
    value and shape equal to `shape`. If the ValueShape and given `shape` are
    non-conformant, then the returned ValueShape will represent an error of
    this mismatch. Similarly if either inputs are in an error state, then an
    error is popagated.

    Usage:
      %0 = shape.with_shape %1, %2 : tensor<...>, !shape.shape

    This is used, for example, where one combines shape function calculations
    and/or call one shape function from another. E.g.,

    ```mlir
    func @shape_foobah(%a: !shape.value_shape,
                       %b: !shape.value_shape,
                       %c: !shape.value_shape) -> !shape.shape {
      %0 = call @shape_foo(%a, %b) :
        (!shape.value_shape, !shape.value_shape) -> !shape.shape
      %1 = shape.with_shape %b, %0 : !shape.value_shape, !shape.shape
      %2 = call @shape_bah(%c, %1) :
        (!shape.value_shape, !shape.value_shape) -> !shape.shape
      return %2 : !shape.shape
    }
    ```

    This op need not be a refinement of the shape. In non-error cases the input
    ValueShape's value and shape are conformant and so too for the output, but
    the result may be less specified than `operand`'s shape as `shape` is
    merely used to construct the new ValueShape. If join behavior is desired
    then a join op should be used.
  }];

  let arguments = (ins AnyTypeOf<[AnyShaped, Shape_ValueShapeType]>:$operand,
                       Shape_ShapeType:$shape);
  let results = (outs Shape_ValueShapeType:$result);

  let assemblyFormat = "operands attr-dict `:` type($operand) `,` type($shape)";
}

def Shape_YieldOp : Shape_Op<"yield",
    [HasParent<"ReduceOp">,
     NoSideEffect,
     ReturnLike,
     Terminator]> {
  let summary = "Returns the value to parent op";

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let verifier = [{ return ::verify(*this); }];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// TODO: Add Ops: if_static, if_ranked

// For testing usage.
def Shape_DebugPrintOp : Shape_Op<"debug_print", []> {
  let summary = "Prints the input shape or size";
  let description = [{
    Prints the input dim or shape and passes through input.

    Note: This is intended for testing and debugging only.
  }];

  let arguments = (ins Shape_ShapeOrSizeType:$input);
  let results =  (outs Shape_ShapeOrSizeType:$output);
}

def Shape_SplitAtOp : Shape_Op<"split_at", []> {
  let summary = "Splits a shape at a given index";
  let description = [{
    Splits a shape at a given dimension `index`, returning two shapes.
    If `index` is negative, it is treated as indexing from the back of the
    shape. This negative-handling behavior is important when handling unranked
    shapes, where the positive index is not necessarily knowable due to a
    dynamic number of leading dimensions.

    Examples:
    - split_at([4,5,6], index=0) -> [], [4,5,6]
    - split_at([4,5,6], index=1) -> [4], [5,6]
    - split_at([4,5,6], index=2) -> [4,5], [6]
    - split_at([4,5,6], index=3) -> [4,5,6], []
    - split_at([4,5,6], index=4) -> error
    - split_at([4,5,6], index=-1) -> [4,5], [6]
    - split_at([4,5,6], index=-2) -> [4], [5,6]
    - split_at([4,5,6], index=-3) -> [], [4,5,6]
    - split_at([4,5,6], index=-4) -> error

    Requires:
    - `index` is in the range [-rank(operand),rank(operand)]
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$operand, I32:$index);
  let results = (outs Shape_ShapeType:$head, Shape_ShapeType:$tail);
  let hasFolder = 1;
}

def Shape_ConcatOp : Shape_Op<"concat", []> {
  let summary = "Concatenates two shapes";
  let description = [{
    Creates a shape whose dimensions consist of first the dimensions from `lhs`
    followed by the dimensions of `rhs`.

    Example:
    concat([2,3], [4,5]) -> [2,3,4,5]
    concat([], []) -> []
    concat([], [4,5,6]) -> [4,5,6]
  }];

  let arguments = (ins Shape_ShapeType:$lhs, Shape_ShapeType:$rhs);
  let results = (outs Shape_ShapeType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Shape constraint related ops.
//===----------------------------------------------------------------------===//

// TODO: Move the code below and witnesses to a different file.
def Shape_AnyOp : Shape_Op<"any", [Commutative,
                                   NoSideEffect]> {
  let summary = "Return any combination of the input shapes";
  let description = [{
    This operation takes multiple input shapes or extent tensors and returns
    some combination of their dimensions. This can be best seen with examples
    below.

    The result is undefined, but still side-effect free, in cases where the
    inputs have differing ranks or differ in extents of shared dimensions.

    Example:
    ```mlir
    %s0 = shape.any [2,?], [?,3] // [2,3]
    %s1 = shape.any [?,?], [1,2] // [1,2]
    ```
  }];

  let arguments = (ins Variadic<Shape_ShapeOrExtentTensorType>:$inputs);
  let results = (outs Shape_ShapeOrExtentTensorType:$result);

  let hasFolder = 1;
}

def Shape_AssumingAllOp : Shape_Op<"assuming_all", [Commutative, NoSideEffect]> {
  let summary = "Return a logical AND of all witnesses";
  let description = [{
    Used to simplify constraints as any single failing precondition is enough
    to prevent execution.

    "assuming" operations represent an execution order restriction to the
    compiler, information for dependent code to rely on (by assuming), and
    nothing else. They should not exist after a program is fully lowered and
    ready to execute.

    Example:
    ```mlir
    %w0 = shape.cstr_broadcastable [2,2], [3,1,2] // Passing
    %w1 = shape.cstr_broadcastable [2,2], [3,2] // Failure
    %w2 = shape.cstr_eq [1,2], [1,2], [1,2] // Passing
    %wf = shape.assuming_all %w0, %w1 // Failure
    %wt = shape.assuming_all %w0, %w2 // Passing
    ```
  }];

  let arguments = (ins Variadic<Shape_WitnessType>:$inputs);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "$inputs attr-dict";

  let hasFolder = 1;

  let verifier = [{ return ::verify(*this); }];
}

def Shape_AssumingOp : Shape_Op<"assuming",
                           [SingleBlockImplicitTerminator<"AssumingYieldOp">,
                            RecursiveSideEffects]> {
  let summary = "Execute the region";
  let description = [{
    Executes the region assuming all witnesses are true.

    "assuming" operations represent an execution order restriction to the
    compiler, information for dependent code to rely on (by assuming), and
    nothing else. They should not exist after a program is fully lowered and
    ready to execute.
  }];
  let arguments = (ins Shape_WitnessType:$witness);
  let regions = (region SizedRegion<1>:$doRegion);
  let results = (outs Variadic<AnyType>:$results);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    // Inline the region into the region containing the AssumingOp and delete
    // the AssumingOp.
    //
    // This does no checks on the inputs to the AssumingOp.
    static void inlineRegionIntoParent(AssumingOp &op, PatternRewriter &rewriter);
  }];

  let hasCanonicalizer = 1;
}

def Shape_AssumingYieldOp : Shape_Op<"assuming_yield",
                                     [NoSideEffect, ReturnLike, Terminator]> {
  let summary = "Yield operation";
  let description = [{
    This yield operation represents a return operation within the assert_and_exec
    region. The operation takes variable number of operands and produces no
    results. The operand number and types must match the return signature of
    the region that contains the operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result",
    [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Shape_CstrBroadcastableOp : Shape_Op<"cstr_broadcastable", [Commutative]> {
  let summary = "Determines if 2 shapes can be successfully broadcasted";
  let description = [{
    Given two input shapes or extent tensors, return a witness specifying if
    they are broadcastable. This broadcastable follows the same logic as what
    shape.broadcast documents.

    "cstr" operations represent runtime assertions.

    Example:
    ```mlir
    %w0 = shape.cstr_broadcastable [2,2], [3,1,2] // Passing
    %w1 = shape.cstr_broadcastable [2,2], [3,2] // Failure
    ```
  }];

  let arguments = (ins Shape_ShapeOrExtentTensorType:$lhs,
                       Shape_ShapeOrExtentTensorType:$rhs);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) attr-dict";

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Shape_CstrEqOp : Shape_Op<"cstr_eq", [Commutative]> {
  let summary = "Determines if all input shapes are equal";
  let description = [{
    Given 1 or more input shapes, determine if all shapes are the exact same.

    "cstr" operations represent runtime assertions.

    Example:
    ```mlir
    %w0 = shape.cstr_eq [1,2], [1,2], [1,2] // Passing
    %w1 = shape.cstr_eq [2,2], [1,2] // Failure
    ```
  }];
  let arguments = (ins Variadic<Shape_ShapeType>:$inputs);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "$inputs attr-dict";

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def Shape_ConstWitnessOp : Shape_Op<"const_witness", [ConstantLike, NoSideEffect]> {
  let summary = "An operation that returns a statically known witness value";
  let description = [{
  This operation represents a statically known witness result. This can be
  often used to canonicalize/fold constraint and assuming code that will always
  pass.

  ```mlir
  %0 = shape.const_shape [1,2,3]
  %1 = shape.const_shape [1, 2, 3]
  %w0 = shape.cstr_eq(%0, %1) // Can be folded to "const_witness true"
  %w1 = shape.const_witness true
  %w2 = shape.assuming_all(%w0, %w2) // Can be folded to "const_witness true"
  ```
  }];
  let arguments = (ins BoolAttr:$passing);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "$passing attr-dict";

  let hasFolder = 1;
}


#endif // SHAPE_OPS
