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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
}

def Shape_BroadcastOp : Shape_Op<"broadcast",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

def Shape_ConstShapeOp : Shape_Op<"const_shape",
    [ConstantLike,
     NoSideEffect,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Creates a constant of !shape.shape type.";
  let description = [{
    Creates a !shape.shape with rank given by the length of `shape` and with
    dimension sizes given by the values of `shape`.

    ```mlir
    %0 = shape.const_shape []
    %1 = shape.const_shape [1, 2, 3]
    ```
  }];
  let arguments = (ins I64ElementsAttr:$shape);
  let results = (outs Shape_ShapeType:$result);

  // TODO: Move this to main so that all shape ops implement these.
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let hasFolder = 1;
}

def Shape_ConstSizeOp : Shape_Op<"const_size",
    [ConstantLike,
     NoSideEffect,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Creates a constant of !shape.size type.";
  let description = [{
    Creates a !shape.size type representing the constant size given by `value`.

    ```mlir
    %x = shape.const_size 10
    ```
  }];

  let arguments = (ins IndexAttr:$value);
  let results = (outs Shape_SizeType:$result);

  let assemblyFormat = "attr-dict $value";
}

def Shape_FromExtentTensorOp : Shape_Op<"from_extent_tensor", []> {
  let summary = "Creates a shape from a tensor of extents";
  let description = [{
    Creates a shape from a 1D integral tensor of extents. The rank of the
    resulting shape equals the number of elements in the tensor, and the
    extents match the values of the elements.
  }];

  let arguments = (ins I32Tensor:$input);
  let results = (outs Shape_ShapeType:$result);
}

def Shape_ToExtentTensorOp : Shape_Op<"to_extent_tensor", []> {
  let summary = "Creates a dimension tensor from a shape";
  // TODO: Think more about the error situation. Perhaps factor out the
  // error detection into a separate op so downstream consumers can control
  // their error behavior. Then this op would assume that the input has
  // been properly checked to not be an error (and could thus be a
  // NoSideEffect op).
  let description = [{
    Converts a shape to a 1D integral tensor of extents. The number of elements
    in the tensor equals the rank of the shape, and the elements equal the
    extents of the shape.

    If the shape represents an error, then this op currently aborts the program.
  }];

  let arguments = (ins Shape_ShapeType:$input);
  let results = (outs IndexTensor:$result);

  let hasFolder = 1;
}

def Shape_JoinOp : Shape_Op<"join", []> {
  let summary = "Returns the least general shape.size of its operands";
  let description = [{
    An operation that computes the least general shape of input operands. This
    effectively asserts that corresponding static dimensions are equal. The
    behavior is to match each element of the `shape.type` and propagate the most
    restrictive information, returning an invalid shape if there are
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
    %c = shape.join %a, %b, error="<reason>" : !shape.type
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
}

def Shape_ReduceOp : Shape_Op<"reduce", []> {
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
    func @shape_num_elements(%shape : !shape.type) -> !shape.size {
      %0 = "shape.constant_dim"() {value = 1 : i32} : () -> !shape.size
      %1 = "shape.reduce"(%shape, %0) ( {
        ^bb0(%index: i32, %dim: !shape.size, %lci: !shape.size):
          %acc = "shape.mul"(%lci, %dim) :
            (!shape.size, !shape.size) -> !shape.size
          shape.yield %acc : !shape.size
        }) : (!shape.type, !shape.size) -> (!shape.size)
      return %1 : !shape.size
    }
    ```

    If the shape is unranked, then the results of the op is also unranked.
  }];

  let arguments = (ins Shape_ShapeType:$shape, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$result);

  let regions = (region SizedRegion<1>:$body);
}

def Shape_ShapeOfOp : Shape_Op<"shape_of",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Returns shape of a value or shaped type operand";

  let arguments = (ins AnyTypeOf<[AnyShaped, Shape_ValueShapeType]>:$arg);
  let results = (outs Shape_ShapeType:$result);

  let hasFolder = 1;
}

def Shape_YieldOp : Shape_Op<"yield", [NoSideEffect, Terminator]> {
  let summary = "Returns the value to parent op";

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<
    "OpBuilder &b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

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

def Shape_SplitAtOp : Shape_Op<"split_at",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Splits a shape at a given index.";
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

def Shape_ConcatOp : Shape_Op<"concat",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Concatenates two shapes.";
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

#endif // SHAPE_OPS
