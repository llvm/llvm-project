//===-- SPIRVIntegerDotProductOps.td - MLIR SPIR-V IDP Ops -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains arithmetic ops for the SPIR-V dialect. It corresponds
// to instructions defined by the "SPV_KHR_integer_dot_product" SPIR-V
// extension.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_INTEGER_DOT_PRODUCT_OPS
#define MLIR_DIALECT_SPIRV_IR_INTEGER_DOT_PRODUCT_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SPIRV_IntegerDotProductOp<string mnemonic,
                                list<Trait> traits = []> :
      SPIRV_Op<mnemonic, !listconcat(traits, [Pure])> {
  let results = (outs
    SPIRV_Integer:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)` `->` type($result)
  }];

  // These ops require dynamic availability specification based on operand and
  // result types.
  bit autogenAvailability = 0;

  // These ops require a custom verifier.
  let hasVerifier = 1;
}

class SPIRV_IntegerDotProductBinaryOp<string mnemonic,
                                      list<Trait> traits = []> :
      SPIRV_IntegerDotProductOp<mnemonic, traits> {
  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$vector1,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$vector2,
    OptionalAttr<SPIRV_PackedVectorFormatAttr>:$format
  );
}

class SPIRV_IntegerDotProductTernaryOp<string mnemonic,
                                       list<Trait> traits = []> :
      SPIRV_IntegerDotProductOp<mnemonic, traits> {
  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$vector1,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$vector2,
    SPIRV_Integer:$accumulator,
    OptionalAttr<SPIRV_PackedVectorFormatAttr>:$format
  );
}

// -----

def SPIRV_SDotOp : SPIRV_IntegerDotProductBinaryOp<"SDot",
                                                   [SignedOp, Commutative]> {
  let summary = "Signed integer dot product of Vector 1 and Vector 2.";

  let description = [{
    Result Type must be an integer type whose Width must be greater than or
    equal to that of the components of Vector 1 and Vector 2.

    Vector 1 and Vector 2 must have the same type.

    Vector 1 and Vector 2 must be either 32-bit integers (enabled by the
    DotProductInput4x8BitPacked capability) or vectors of integer type
    (enabled by the DotProductInput4x8Bit or DotProductInputAll capability).

    When Vector 1 and Vector 2 are scalar integer types, Packed Vector
    Format must be specified to select how the integers are to be
    interpreted as vectors.

    All components of the input vectors are sign-extended to the bit width
    of the result's type. The sign-extended input vectors are then
    multiplied component-wise and all components of the vector resulting
    from the component-wise multiplication are added together. The resulting
    value will equal the low-order N bits of the correct result R, where N
    is the result width and R is computed with enough precision to avoid
    overflow and underflow.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
    %r = spirv.SDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i64
    %r = spirv.SDot %a, %b : (vector<4xi8>, vector<4xi8>) -> i32
    ```
  }];
}

// -----

def SPIRV_SUDotOp : SPIRV_IntegerDotProductBinaryOp<"SUDot",
                                                    [SignedOp, UnsignedOp]> {
  let summary = [{
    Mixed-signedness integer dot product of Vector 1 and Vector 2.
    Components of Vector 1 are treated as signed, components of Vector 2 are
    treated as unsigned.
  }];

  let description = [{
    Result Type must be an integer type whose Width must be greater than or
    equal to that of the components of Vector 1 and Vector 2.

    Vector 1 and Vector 2 must be either 32-bit integers (enabled by the
    DotProductInput4x8BitPacked capability) or vectors of integer type with
    the same number of components and same component Width (enabled by the
    DotProductInput4x8Bit or DotProductInputAll capability). When Vector 1
    and Vector 2 are vectors, the components of Vector 2 must have a
    Signedness of 0.

    When Vector 1 and Vector 2 are scalar integer types, Packed Vector
    Format must be specified to select how the integers are to be
    interpreted as vectors.

    All components of Vector 1 are sign-extended to the bit width of the
    result's type. All components of Vector 2 are zero-extended to the bit
    width of the result's type. The sign- or zero-extended input vectors are
    then multiplied component-wise and all components of the vector
    resulting from the component-wise multiplication are added together. The
    resulting value will equal the low-order N bits of the correct result R,
    where N is the result width and R is computed with enough precision to
    avoid overflow and underflow.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %r = spirv.SUDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
    %r = spirv.SUDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i64
    %r = spirv.SUDot %a, %b : (vector<4xi8>, vector<4xi8>) -> i32
    ```
  }];
}

// -----

def SPIRV_UDotOp : SPIRV_IntegerDotProductBinaryOp<"UDot",
                                                   [UnsignedOp, Commutative]> {
  let summary = "Unsigned integer dot product of Vector 1 and Vector 2.";

  let description = [{
    Result Type must be an integer type with Signedness of 0 whose Width
    must be greater than or equal to that of the components of Vector 1 and
    Vector 2.

    Vector 1 and Vector 2 must have the same type.

    Vector 1 and Vector 2 must be either 32-bit integers (enabled by the
    DotProductInput4x8BitPacked capability) or vectors of integer type with
    Signedness of 0 (enabled by the DotProductInput4x8Bit or
    DotProductInputAll capability).

    When Vector 1 and Vector 2 are scalar integer types, Packed Vector
    Format must be specified to select how the integers are to be
    interpreted as vectors.

    All components of the input vectors are zero-extended to the bit width
    of the result's type. The zero-extended input vectors are then
    multiplied component-wise and all components of the vector resulting
    from the component-wise multiplication are added together. The resulting
    value will equal the low-order N bits of the correct result R, where N
    is the result width and R is computed with enough precision to avoid
    overflow and underflow.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %r = spirv.UDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i32
    %r = spirv.UDot %a, %b {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32) -> i64
    %r = spirv.UDot %a, %b : (vector<4xi8>, vector<4xi8>) -> i32
    ```
  }];
}

// -----

def SPIRV_SDotAccSatOp : SPIRV_IntegerDotProductTernaryOp<"SDotAccSat",
                                                          [SignedOp]> {
  let summary = [{
    Signed integer dot product of Vector 1 and Vector 2 and signed
    saturating addition of the result with Accumulator.
  }];

  let description = [{
    Result Type must be an integer type whose Width must be greater than or
    equal to that of the components of Vector 1 and Vector 2.

    Vector 1 and Vector 2 must have the same type.

    Vector 1 and Vector 2 must be either 32-bit integers (enabled by the
    DotProductInput4x8BitPacked capability) or vectors of integer type
    (enabled by the DotProductInput4x8Bit or DotProductInputAll capability).

    The type of Accumulator must be the same as Result Type.

    When Vector 1 and Vector 2 are scalar integer types, Packed Vector
    Format must be specified to select how the integers are to be
    interpreted as vectors.

    All components of the input vectors are sign-extended to the bit width
    of the result's type. The sign-extended input vectors are then
    multiplied component-wise and all components of the vector resulting
    from the component-wise multiplication are added together. Finally, the
    resulting sum is added to the input accumulator. This final addition is
    saturating.

    If any of the multiplications or additions, with the exception of the
    final accumulation, overflow or underflow, the result of the instruction
    is undefined.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i32
    %r = spirv.SDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i64) -> i64
    %r = spirv.SDotAccSat %a, %b, %acc : (vector<4xi8>, vector<4xi8>, i32) -> i32
    ```
  }];
}

// -----

def SPIRV_SUDotAccSatOp : SPIRV_IntegerDotProductTernaryOp<"SUDotAccSat",
                                                           [SignedOp,
                                                            UnsignedOp]> {
  let summary = [{
    Mixed-signedness integer dot product of Vector 1 and Vector 2 and signed
    saturating addition of the result with Accumulator. Components of Vector
    1 are treated as signed, components of Vector 2 are treated as unsigned.
  }];

  let description = [{
    Result Type must be an integer type whose Width must be greater than or
    equal to that of the components of Vector 1 and Vector 2.

    Vector 1 and Vector 2 must be either 32-bit integers (enabled by the
    DotProductInput4x8BitPacked capability) or vectors of integer type with
    the same number of components and same component Width (enabled by the
    DotProductInput4x8Bit or DotProductInputAll capability). When Vector 1
    and Vector 2 are vectors, the components of Vector 2 must have a
    Signedness of 0.

    The type of Accumulator must be the same as Result Type.

    When Vector 1 and Vector 2 are scalar integer types, Packed Vector
    Format must be specified to select how the integers are to be
    interpreted as vectors.

    All components of Vector 1 are sign-extended to the bit width of the
    result's type. All components of Vector 2 are zero-extended to the bit
    width of the result's type. The sign- or zero-extended input vectors are
    then multiplied component-wise and all components of the vector
    resulting from the component-wise multiplication are added together.
    Finally, the resulting sum is added to the input accumulator. This final
    addition is saturating.

    If any of the multiplications or additions, with the exception of the
    final accumulation, overflow or underflow, the result of the instruction
    is undefined.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %r = spirv.SUDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i32
    %r = spirv.SUDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i64) -> i64
    %r = spirv.SUDotAccSat %a, %b, %acc : (vector<4xi8>, vector<4xi8>, i32) -> i32
    ```
  }];
}

// -----

def SPIRV_UDotAccSatOp :
    SPIRV_IntegerDotProductTernaryOp<"UDotAccSat", [UnsignedOp]> {
  let summary = [{
    Unsigned integer dot product of Vector 1 and Vector 2 and unsigned
    saturating addition of the result with Accumulator.
  }];

  let description = [{
    Result Type must be an integer type with Signedness of 0 whose Width
    must be greater than or equal to that of the components of Vector 1 and
    Vector 2.

    Vector 1 and Vector 2 must have the same type.

    Vector 1 and Vector 2 must be either 32-bit integers (enabled by the
    DotProductInput4x8BitPacked capability) or vectors of integer type with
    Signedness of 0 (enabled by the DotProductInput4x8Bit or
    DotProductInputAll capability).

    The type of Accumulator must be the same as Result Type.

    When Vector 1 and Vector 2 are scalar integer types, Packed Vector
    Format must be specified to select how the integers are to be
    interpreted as vectors.

    All components of the input vectors are zero-extended to the bit width
    of the result's type. The zero-extended input vectors are then
    multiplied component-wise and all components of the vector resulting
    from the component-wise multiplication are added together. Finally, the
    resulting sum is added to the input accumulator. This final addition is
    saturating.

    If any of the multiplications or additions, with the exception of the
    final accumulation, overflow or underflow, the result of the instruction
    is undefined.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %r = spirv.UDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i32) -> i32
    %r = spirv.UDotAccSat %a, %b, %acc {format = #spirv.packed_vector_format<PackedVectorFormat4x8Bit>}: (i32, i32, i64) -> i64
    %r = spirv.UDotAccSat %a, %b, %acc : (vector<4xi8>, vector<4xi8>, i32) -> i32
    ```
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_INTEGER_DOT_PRODUCT_OPS
