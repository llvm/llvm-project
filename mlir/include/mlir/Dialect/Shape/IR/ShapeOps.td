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

def Shape_WitnessType : DialectType<ShapeDialect,
    CPred<"$_self.isa<::mlir::shape::WitnessType>()">, "witness">,
    BuildableType<"$_builder.getType<::mlir::shape::WitnessType>()"> {
  let typeDescription = [{
    A witness is a structural device in the compiler to maintain ordering of
    code relying on information obtained from passing assertions. Witnesses do
    not represent any physical data.

    "cstr_" operations will return witnesses and be lowered into assertion logic
    when not resolvable at compile time.

    "assuming_" operations will take witnesses as input and represent only
    information to the compiler, so they do not exist in executing code. Code
    that is dependent on "assuming_" operations can assume all cstr operations
    transitively before are honored as true.

    These abstractions are intended to allow the compiler more freedom with
    assertions by merely showing the assertion through dataflow at this time
    rather than a side effecting operation that acts as a barrier. This can be
    viewed similarly to a compiler representation of promises from asynchronous,
    possibly crashing assertions. Reliant code will not be reordered to before
    the code and non-reliant code can be reordered freely, and there are no
    guarantees on the final ordering of the assertions or their related code.
  }];
}

//===----------------------------------------------------------------------===//
// Shape op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Shape_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ShapeDialect, mnemonic, traits>;

def Shape_AddOp : Shape_Op<"add", [SameOperandsAndResultType]> {
  let summary = "Addition of sizes";
  let description = [{
    Adds two valid sizes as follows:
    * lhs + rhs = unknown if either lhs or rhs unknown;
    * lhs + rhs = (int)lhs + (int)rhs if known;
  }];

  let arguments = (ins Shape_SizeType:$lhs, Shape_SizeType:$rhs);
  let results = (outs Shape_SizeType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Shape_BroadcastOp : Shape_Op<"broadcast", []> {
  let summary = "Returns the broadcasted output shape of two inputs";
  let description = [{
    Computes the broadcasted output shape following:
    1. If any inputs are unranked, output is unranked;
    2. Else the input array with number of dimensions smaller than the max
       input dimension, has 1’s prepended to its shapes and the output shape is
       calculated as follows:

           output[i] = lhs[i] if lhs[i] == rhs[i] or rhs[i] is unknown/undefined
                     = rhs[i] if lhs[i] is unknown/undefined
                     = lhs[i] if rhs[i] == 1
                     = rhs[i] if lhs[i] == 1
                     = error  if lhs[i] != rhs[i]

    Op has an optional string attribute for the error case where there is no
    broadcastable output shape possible for the given inputs.
  }];

  let arguments = (ins Shape_ShapeType:$lhs, Shape_ShapeType:$rhs,
                   OptionalAttr<StrAttr>:$error);
  let results = (outs Shape_ShapeType:$result);

  let hasFolder = 1;
}

def Shape_ConstShapeOp : Shape_Op<"const_shape", [ConstantLike, NoSideEffect]> {
  let summary = "Creates a constant of !shape.shape type";
  let description = [{
    Creates a !shape.shape with rank given by the length of `shape` and with
    dimension sizes given by the values of `shape`.

    ```mlir
    %0 = shape.const_shape []
    %1 = shape.const_shape [1, 2, 3]
    ```
  }];
  let arguments = (ins IndexElementsAttr:$shape);
  let results = (outs Shape_ShapeType:$result);

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
    Returns the rank of the shape, i.e. the number of extents.
  }];

  let arguments = (ins Shape_ShapeType:$shape);
  let results = (outs Shape_SizeType:$rank);

  let assemblyFormat = "attr-dict $shape";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def Shape_ToExtentTensorOp : Shape_Op<"to_extent_tensor", [NoSideEffect]> {
  let summary = "Creates a dimension tensor from a shape";
  let description = [{
    Converts a shape to a 1D integral tensor of extents. The number of elements
    in the tensor equals the rank of the shape, and the elements equal the
    extents of the shape.

    If the shape represents an error, this op's behavior is undefined.
  }];

  let arguments = (ins Shape_ShapeType:$input);
  let results = (outs IndexTensor:$result);

  let assemblyFormat = "attr-dict $input `:` type($result)";

  let hasFolder = 1;
}

def Shape_GetExtentOp : Shape_Op<"get_extent", [NoSideEffect]> {
  let summary = "Gets the specified extent from a shape";
  let description = [{
    Gets the extent indexed by `dim` from `shape`.
    If the shape is an error, it returns an error size.
  }];
  let arguments = (ins
    Shape_ShapeType:$shape,
    Shape_SizeType:$dim
  );
  let results = (outs Shape_SizeType:$extent);
  let assemblyFormat = "$shape `,` $dim attr-dict";

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

def Shape_JoinOp : Shape_Op<"join", []> {
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

def Shape_MulOp : Shape_Op<"mul", [SameOperandsAndResultType]> {
  let summary = "Multiplication of sizes";
  let description = [{
    Multiplies two valid sizes as follows:
    - lhs * rhs = unknown if either lhs or rhs unknown;
    - lhs * rhs = (int)lhs * (int)rhs if both known;
  }];

  let arguments = (ins Shape_SizeType:$lhs, Shape_SizeType:$rhs);
  let results = (outs Shape_SizeType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def Shape_NumElementsOp : Shape_Op<"num_elements", [NoSideEffect]> {
  let summary = "Returns the number of elements for a given shape";
  let description = [{
    Returns the number of elements for a given shape which is the product of its
    dimensions.

    ```mlir
    %product = shape.mul %lhs, %rhs
    ```
  }];

  let arguments = (ins Shape_ShapeType:$shape);
  let results = (outs Shape_SizeType:$result);

  let assemblyFormat = "$shape attr-dict";

  let hasFolder = 1;
}

def Shape_ReduceOp : Shape_Op<"reduce",
    [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Returns an expression reduced over a shape";
  let description = [{
    An operation that takes as input a shape, number of initial values and has a
    region/function that is applied repeatedly for every dimension of the shape.

    Conceptually this op performs the following reduction:

    ```
    res[] = init;
    for (int i = 0, e = shape.rank(); i != e; ++i) {
      res = fn(i, shape[i], res[0], ..., res[n]);
    }
    ```

    Where fn is provided by the user and the result of the reduce op is the
    last computed output of the reduce function. As an example, computing the
    number of elements

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

    If the shape is unranked, then the results of the op is also unranked.
  }];

  let arguments = (ins Shape_ShapeType:$shape, Variadic<AnyType>:$initVals);
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

  let arguments = (ins AnyTypeOf<[AnyShaped, Shape_ValueShapeType]>:$arg);
  let results = (outs Shape_ShapeType:$result);

  let assemblyFormat = "$arg `:` type($arg) attr-dict";

  let hasFolder = 1;
}

def Shape_SizeToIndexOp : Shape_Op<"size_to_index", [NoSideEffect]> {
  let summary = "Casts between index types of the shape and standard dialect";
  let description = [{
    Converts a `shape.size` to a standard index.
    This operation and its inverse, `index_to_size`, facilitate index conversion
    between the standard and the shape dialect.
    The behavior is undefined for unknown and invalid arguments.
  }];

  let arguments = (ins Shape_SizeType:$arg);
  let results = (outs Index:$result);

  let assemblyFormat = "$arg attr-dict";

  let hasFolder = 1;
  let hasCanonicalizer = 1;
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

  let arguments = (ins Shape_ShapeType:$operand, I32:$index);
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

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Shape constraint related ops.
//===----------------------------------------------------------------------===//

//TODO(tpopp): Move the code below and witnesses to a different file.
def Shape_AnyOp : Shape_Op<"any", [Commutative, NoSideEffect]> {
  let summary = "Return any combination of the input shapes";
  let description = [{
    This operation takes multiple input shapes and returns some combination of
    their dimensions. This can be best seen with examples below.

    The result is undefined, but still side-effect free, in cases where the
    inputs have differing ranks or differ in extents of shared dimensions.

    Example:
    ```mlir
    %s0 = shape.any [2,?], [?,3] // [2,3]
    %s1 = shape.any [?,?], [1,2] // [1,2]
    ```
  }];

  let arguments = (ins Variadic<Shape_ShapeType>:$inputs);
  let results = (outs Shape_ShapeType:$result);

  let assemblyFormat = "$inputs attr-dict";

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
    Given 2 input shapes, return a witness specifying if they are broadcastable.
    This broadcastable follows the same logic as what shape.broadcast documents.

    "cstr" operations represent runtime assertions.

    Example:
    ```mlir
    %w0 = shape.cstr_broadcastable [2,2], [3,1,2] // Passing
    %w1 = shape.cstr_broadcastable [2,2], [3,2] // Failure
    ```
  }];

  let arguments = (ins Shape_ShapeType:$lhs, Shape_ShapeType:$rhs);
  let results = (outs Shape_WitnessType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

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
