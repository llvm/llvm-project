//===-- SPIRVCompositeOps.td - MLIR SPIR-V Composite Ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains composite ops for SPIR-V dialect. It corresponds
// to "3.32.12. Composite Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_COMPOSITE_OPS
#define MLIR_DIALECT_SPIRV_IR_COMPOSITE_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPIRV_CompositeConstructOp : SPIRV_Op<"CompositeConstruct", [Pure]> {
  let summary = [{
    Construct a new composite object from a set of constituent objects.
  }];

  let description = [{
    Result Type must be a composite type, whose top-level
    members/elements/components/columns have the same type as the types of
    the operands, with one exception. The exception is that for constructing
    a vector, the operands may also be vectors with the same component type
    as the Result Type component type. When constructing a vector, the total
    number of components in all the operands must equal the number of
    components in Result Type.

    Constituents will become members of a structure, or elements of an
    array, or components of a vector, or columns of a matrix. There must be
    exactly one Constituent for each top-level
    member/element/component/column of the result, with one exception. The
    exception is that for constructing a vector, a contiguous subset of the
    scalars consumed can be represented by a vector operand instead. The
    Constituents must appear in the order needed by the definition of the
    type of the result. When constructing a vector, there must be at least
    two Constituent operands.

    <!-- End of AutoGen section -->

    ```
    composite-construct-op ::= ssa-id `=` `spirv.CompositeConstruct`
                               (ssa-use (`,` ssa-use)* )? `:` composite-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.CompositeConstruct %1, %2, %3 : vector<3xf32>
    ```
  }];

  let arguments = (ins
    Variadic<SPIRV_Type>:$constituents
  );

  let results = (outs
    SPIRV_Composite:$result
  );

  let assemblyFormat = [{
    $constituents attr-dict `:` `(` type(operands) `)` `->` type($result)
  }];
}

// -----

def SPIRV_CompositeExtractOp : SPIRV_Op<"CompositeExtract",
                                        [Pure, UsableInSpecConstantOp]> {
  let summary = "Extract a part of a composite object.";

  let description = [{
    Result Type must be the type of object selected by the last provided
    index.  The instruction result is the extracted object.

    Composite is the composite to extract from.

    Indexes walk the type hierarchy, potentially down to component
    granularity, to select the part to extract. All indexes must be in
    bounds.  All composite constituents use zero-based numbering, as
    described by their OpType… instruction.

    <!-- End of AutoGen section -->

    ```
    composite-extract-op ::= ssa-id `=` `spirv.CompositeExtract` ssa-use
                             `[` integer-literal (',' integer-literal)* `]`
                             `:` composite-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.Variable : !spirv.ptr<!spirv.array<4x!spirv.array<4xf32>>, Function>
    %1 = spirv.Load "Function" %0 ["Volatile"] : !spirv.array<4x!spirv.array<4xf32>>
    %2 = spirv.CompositeExtract %1[1 : i32] : !spirv.array<4x!spirv.array<4xf32>>
    ```

  }];

  let arguments = (ins
    SPIRV_Composite:$composite,
    I32ArrayAttr:$indices
  );

  let results = (outs
    SPIRV_Type:$component
  );

  let builders = [
    OpBuilder<(ins "Value":$composite, "ArrayRef<int32_t>":$indices)>
  ];

  let hasFolder = 1;
}

// -----

def SPIRV_CompositeInsertOp : SPIRV_Op<"CompositeInsert",
                                       [Pure, UsableInSpecConstantOp]> {
  let summary = [{
    Make a copy of a composite object, while modifying one part of it.
  }];

  let description = [{
    Result Type must be the same type as Composite.

    Object is the object to use as the modified part.

    Composite is the composite to copy all but the modified part from.

    Indexes walk the type hierarchy of Composite to the desired depth,
    potentially down to component granularity, to select the part to modify.
    All indexes must be in bounds. All composite constituents use zero-based
    numbering, as described by their OpType… instruction. The type of the
    part selected to modify must match the type of Object.

    <!-- End of AutoGen section -->

    ```
    composite-insert-op ::= ssa-id `=` `spirv.CompositeInsert` ssa-use, ssa-use
                            `[` integer-literal (',' integer-literal)* `]`
                            `:` object-type `into` composite-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.CompositeInsert %object, %composite[1 : i32] : f32 into !spirv.array<4xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_Type:$object,
    SPIRV_Composite:$composite,
    I32ArrayAttr:$indices
  );

  let results = (outs
    SPIRV_Composite:$result
  );

  let builders = [
    OpBuilder<(ins "Value":$object, "Value":$composite,
                      "ArrayRef<int32_t>":$indices)>
  ];
}

// -----

def SPIRV_VectorExtractDynamicOp : SPIRV_Op<"VectorExtractDynamic", [
    Pure,
    TypesMatchWith<"type of 'result' matches element type of 'vector'",
                   "vector", "result",
                   "$_self.cast<mlir::VectorType>().getElementType()">]> {
  let summary = [{
    Extract a single, dynamically selected, component of a vector.
  }];

  let description = [{
    Result Type must be a scalar type.

    Vector must have a type OpTypeVector whose Component Type is Result
    Type.

    Index must be a scalar integer. It is interpreted as a 0-based index of
    which component of Vector to extract.

    Behavior is undefined if Index's value is less than zero or greater than
    or equal to the number of components in Vector.

    <!-- End of AutoGen section -->

    #### Example:

    ```
    %2 = spirv.VectorExtractDynamic %0[%1] : vector<8xf32>, i32
    ```
  }];

  let arguments = (ins
    SPIRV_Vector:$vector,
    SPIRV_Integer:$index
  );

  let results = (outs
    SPIRV_Scalar:$result
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $vector `[` $index `]` attr-dict `:` type($vector) `,` type($index)
  }];
}

// -----

def SPIRV_VectorInsertDynamicOp : SPIRV_Op<"VectorInsertDynamic", [
    Pure,
    TypesMatchWith<
      "type of 'component' matches element type of 'vector'",
      "vector", "component",
      "$_self.cast<mlir::VectorType>().getElementType()">,
    AllTypesMatch<["vector", "result"]>]> {
  let summary = [{
    Make a copy of a vector, with a single, variably selected, component
    modified.
  }];

  let description = [{
    Result Type must be an OpTypeVector.

    Vector must have the same type as Result Type and is the vector that the
    non-written components are copied from.

    Component is the value supplied for the component selected by Index. It
    must have the same type as the type of components in Result Type.

    Index must be a scalar integer. It is interpreted as a 0-based index of
    which component to modify.

    Behavior is undefined if Index's value is less than zero or greater than
    or equal to the number of components in Vector.

    <!-- End of AutoGen section -->

    ```
    scalar-type ::= integer-type | float-type | boolean-type
    vector-insert-dynamic-op ::= `spirv.VectorInsertDynamic ` ssa-use `,`
                                  ssa-use `[` ssa-use `]`
                                  `:` `vector<` integer-literal `x` scalar-type `>` `,`
                                  integer-type
    ```mlir

    #### Example:

    ```
    %scalar = ... : f32
    %2 = spirv.VectorInsertDynamic %scalar %0[%1] : f32, vector<8xf32>, i32
    ```
  }];

  let arguments = (ins
    SPIRV_Vector:$vector,
    SPIRV_Scalar:$component,
    SPIRV_Integer:$index
  );

  let results = (outs
    SPIRV_Vector:$result
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $component `,` $vector `[` $index `]` attr-dict `:` type($vector) `,` type($index)
  }];
}

// -----

def SPIRV_VectorShuffleOp : SPIRV_Op<"VectorShuffle", [
    Pure, AllElementTypesMatch<["vector1", "vector2", "result"]>]> {
  let summary = [{
    Select arbitrary components from two vectors to make a new vector.
  }];

  let description = [{
    Result Type must be an OpTypeVector. The number of components in Result
    Type must be the same as the number of Component operands.

    Vector 1 and Vector 2 must both have vector types, with the same
    Component Type as Result Type. They do not have to have the same number
    of components as Result Type or with each other. They are logically
    concatenated, forming a single vector with Vector 1’s components
    appearing before Vector 2’s. The components of this logical vector are
    logically numbered with a single consecutive set of numbers from 0 to N
    - 1, where N is the total number of components.

    Components are these logical numbers (see above), selecting which of the
    logically numbered components form the result. Each component is an
    unsigned 32-bit integer.  They can select the components in any order
    and can repeat components. The first component of the result is selected
    by the first Component operand,  the second component of the result is
    selected by the second Component operand, etc. A Component literal may
    also be FFFFFFFF, which means the corresponding result component has no
    source and is undefined. All Component literals must either be FFFFFFFF
    or in [0, N - 1] (inclusive).

    Note: A vector “swizzle” can be done by using the vector for both Vector
    operands, or using an OpUndef for one of the Vector operands.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.VectorShuffle [1: i32, 3: i32, 5: i32]
                           %vector1: vector<4xf32>, %vector2: vector<2xf32>
                        -> vector<3xf32>
    ```
  }];

  let arguments = (ins
    SPIRV_Vector:$vector1,
    SPIRV_Vector:$vector2,
    I32ArrayAttr:$components
  );

  let results = (outs
    SPIRV_Vector:$result
  );

  let assemblyFormat = [{
    attr-dict $components $vector1 `:` type($vector1) `,`
                          $vector2 `:` type($vector2) `->` type($result)
  }];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_COMPOSITE_OPS
