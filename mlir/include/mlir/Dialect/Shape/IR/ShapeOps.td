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

include "mlir/IR/OpBase.td"

// TODO(jpienaar): Move to base.
def AnyShaped: ShapedContainerType<[AnyType], IsShapedTypePred, "shaped">;

//===----------------------------------------------------------------------===//
// Shape Inference dialect definitions
//===----------------------------------------------------------------------===//

def ShapeDialect : Dialect {
  let name = "shape";

  let summary = "Types and operations for shape dialect";
  let description = [{
    This dialect contains operations for shape inference.

    Note: Unless explicitly stated, all functions that return a shape and take
    shapes as input, return the invalid shape if one of its operands is an
    invalid shape. This avoids flagging multiple errors for one verification
    failure. The dialect itself does not specify how errors should be combined
    (there are multiple different options, from always chosing first operand,
    concatting etc. on how to combine them).
  }];

  let cppNamespace = "shape";
}

def Shape_SizeType : DialectType<ShapeDialect,
    CPred<"$_self.isa<DimType>()">, "dim"> {
  let typeDescription = [{
    `shape.size` represents a non-negative integer with support for being
    unknown and invalid.

    Operations on `shape.size` types are specialized to handle unknown/dynamic
    value. So, for example, `<unknown> + x == <unknown>` for all non-error `x :
    !shape.size` (e.g., an unknown value does not become known due to addition).
  }];
}

def Shape_ShapeType : DialectType<ShapeDialect,
    CPred<"$_self.isa<ShapeType>()">, "shape"> {
  let typeDescription = [{
    `shape.type` represents either an unranked shape, a ranked shape with
    possibly unknown dimensions or an invalid shape. The rank is of type
    `shape.size` and, if rank is known, the extent is a 1D tensor of type
    `shape.size`.

    Shape is printed:
    * `[*]` if it is an unranked shape
    * `[?, 2]` if a rank 2 tensor with one unknown dimension
    * `[3, 4]` is a rank 2 static tensor
    * `[]` is a scalar
    * `[1]` is a rank 1 tensor with 1 element
    * `[invalid]` for an invalid shape
  }];
}

def Shape_ElementType : DialectType<ShapeDialect,
    CPred<"$_self.isa<ElementType>()">, "element type"> {
  let typeDescription = [{
    `shape.element_type` represents the element type of the ShapedType. It may
    be unknown, error or regular element type supported by ShapedType.
  }];
}

def Shape_ComponentType : DialectType<ShapeDialect,
    CPred<"$_self.isa<ComponentType>()">, "component type"> {
  let typeDescription = [{
    `shape.element_type` represents the element type of the ShapedType. It may
    be unknown, error or regular element type supported by ShapedType.
  }];
}

def Shape_ValueShapeType : DialectType<ShapeDialect,
    CPred<"$_self.isa<ValueShapeType>()">, "value shape"> {
  let typeDescription = [{
    `shape.value_shape` represents the value produced by an operation (this
    corresponds to `Value` in the compiler) and a shape. Conceptually this is a
    tuple of a value (potentially unknown) and `shape.type`. The value and shape
    can either or both be unknown. If both the `value` and `shape` are known,
    then the shape of `value` is conformant with `shape`.
  }];
}

def Shape_ShapeOrSizeType: AnyTypeOf<[Shape_SizeType, Shape_ShapeType],
  "shape or size">;

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

  let arguments = (ins Shape_ShapeType:$lhs, Shape_ShapeType:$rhs);
  let results = (outs Shape_ShapeType:$result);
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
}

def Shape_ConstantOp : Shape_Op<"constant", []> {
  let summary = "Creates a shape constant";
  let description = [{
    An operation that builds a size or shape from integer or array attribute.
    It allows for creating dynamically valued shapes by using `?` for unknown
    values. A constant shape specified with `*` will return an unranked shape.

    ```mlir
    %x = shape.constant 10 : !shape.size
    ```
  }];

  // TODO(jpienaar): Change to a more specialized attribute that would
  // encapsulate the unknown parsing while using denser packing.
  let arguments = (ins ArrayAttr:$value);
  let results = (outs Shape_ShapeOrSizeType:$result);
}

def Shape_CreateShapeOp : Shape_Op<"create_shape", []> {
  let summary = "Creates a shape descriptor from a tensor";
  let description = [{
    Creates a shape from a 1D integral tensor. The rank equals the number of
    elements in the tensor, and extent matches the values of the elements.
  }];

  let arguments = (ins I32Tensor:$input);
  let results = (outs Shape_ShapeType:$result);
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

  let arguments = (ins Shape_ShapeType:$lhs, Shape_ShapeType:$rhs);
  let results = (outs Shape_ShapeType:$result);
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
          "shape.return"(%acc) : (!shape.size) -> ()
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

def Shape_ShapeOfOp : Shape_Op<"shape_of", []> {
  let summary = "Returns shape of a value or shaped type operand";

  let arguments = (ins AnyTypeOf<[AnyShaped, Shape_ValueShapeType]>:$arg);
  let results = (outs Shape_ShapeType:$result);
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

#endif // SHAPE_OPS
