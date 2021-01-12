//===-- SPIRVBitOps.td - MLIR SPIR-V Bit Ops -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains bit ops for the SPIR-V dialect. It corresponds
// to "3.32.13. Bit Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_BIT_OPS
#define MLIR_DIALECT_SPIRV_IR_BIT_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SPV_BitBinaryOp<string mnemonic, list<OpTrait> traits = []> :
      // All the operands type used in bit instructions are SPV_Integer.
      SPV_BinaryOp<mnemonic, SPV_Integer, SPV_Integer,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

class SPV_BitFieldExtractOp<string mnemonic, list<OpTrait> traits = []> :
      SPV_Op<mnemonic, !listconcat(traits,
             [NoSideEffect, AllTypesMatch<["base", "result"]>])> {
  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Integer>:$base,
    SPV_Integer:$offset,
    SPV_Integer:$count
  );

  let results = (outs
    SPV_ScalarOrVectorOf<SPV_Integer>:$result
  );

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    operands attr-dict `:` type($base) `,` type($offset) `,` type($count)
  }];
}

class SPV_BitUnaryOp<string mnemonic, list<OpTrait> traits = []> :
      SPV_UnaryOp<mnemonic, SPV_Integer, SPV_Integer,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultType])>;

class SPV_ShiftOp<string mnemonic, list<OpTrait> traits = []> :
      SPV_BinaryOp<mnemonic, SPV_Integer, SPV_Integer,
                   !listconcat(traits,
                               [NoSideEffect, SameOperandsAndResultShape])> {
  let parser = [{ return ::parseShiftOp(parser, result); }];
  let printer = [{ ::printShiftOp(this->getOperation(), p); }];
  let verifier = [{ return ::verifyShiftOp(this->getOperation()); }];
}

// -----

def SPV_BitCountOp : SPV_BitUnaryOp<"BitCount", []> {
  let summary = "Count the number of set bits in an object.";

  let description = [{
     Results are computed per component.

    Result Type must be a scalar or vector of integer type.  The components
    must be wide enough to hold the unsigned Width of Base as an unsigned
    value. That is, no sign bit is needed or counted when checking for a
    wide enough result width.

    Base must be a scalar or vector of integer type.  It must have the same
    number of components as Result Type.

    The result is the unsigned value that is the number of bits in Base that
    are 1.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitcount-op ::= ssa-id `=` `spv.BitCount` ssa-use
                               `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitCount %0: i32
    %3 = spv.BitCount %1: vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_BitFieldInsertOp : SPV_Op<"BitFieldInsert",
    [NoSideEffect, AllTypesMatch<["base", "insert", "result"]>]> {
  let summary = [{
    Make a copy of an object, with a modified bit field that comes from
    another object.
  }];

  let description = [{
     Results are computed per component.

    Result Type must be a scalar or vector of integer type.

     The type of Base and Insert must be the same as Result Type.

    Any result bits numbered outside [Offset, Offset + Count -  1]
    (inclusive) will come from the corresponding bits in Base.

    Any result bits numbered in [Offset, Offset + Count -  1] come, in
    order, from the bits numbered [0, Count - 1] of Insert.

    Count  must be an integer type scalar. Count is the number of bits taken
    from Insert. It will be consumed as an unsigned value. Count can be 0,
    in which case the result will be Base.

    Offset  must be an integer type scalar. Offset is the lowest-order bit
    of the bit field.  It will be consumed as an unsigned value.

    The resulting value is undefined if Count or Offset or their sum is
    greater than the number of bits in the result.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitfield-insert-op ::= ssa-id `=` `spv.BitFieldInsert` ssa-use `,` ssa-use
                                      `,` ssa-use `,` ssa-use
                                      `:` integer-scalar-vector-type
                                      `,` integer-type `,` integer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.BitFieldInsert %base, %insert, %offset, %count : vector<3xi32>, i8, i8
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Shader]>
  ];

  let arguments = (ins
    SPV_ScalarOrVectorOf<SPV_Integer>:$base,
    SPV_ScalarOrVectorOf<SPV_Integer>:$insert,
    SPV_Integer:$offset,
    SPV_Integer:$count
  );

  let results = (outs
    SPV_ScalarOrVectorOf<SPV_Integer>:$result
  );

  let verifier = [{ return success(); }];

  let assemblyFormat = [{
    operands attr-dict `:` type($base) `,` type($offset) `,` type($count)
  }];
}

// -----

def SPV_BitFieldSExtractOp : SPV_BitFieldExtractOp<"BitFieldSExtract", []> {
  let summary = "Extract a bit field from an object, with sign extension.";

  let description = [{
     Results are computed per component.

    Result Type must be a scalar or vector of integer type.

     The type of Base must be the same as Result Type.

    If Count is greater than 0: The bits of Base numbered in [Offset, Offset
    + Count -  1] (inclusive) become the bits numbered [0, Count - 1] of the
    result. The remaining bits of the result will all be the same as bit
    Offset + Count -  1 of Base.

    Count  must be an integer type scalar. Count is the number of bits
    extracted from Base. It will be consumed as an unsigned value. Count can
    be 0, in which case the result will be 0.

    Offset  must be an integer type scalar. Offset is the lowest-order bit
    of the bit field to extract from Base.  It will be consumed as an
    unsigned value.

    The resulting value is undefined if Count or Offset or their sum is
    greater than the number of bits in the result.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitfield-extract-s-op ::= ssa-id `=` `spv.BitFieldSExtract` ssa-use
                                         `,` ssa-use `,` ssa-use
                                         `:` integer-scalar-vector-type
                                         `,` integer-type `,` integer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.BitFieldSExtract %base, %offset, %count : vector<3xi32>, i8, i8
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Shader]>
  ];
}

// -----

def SPV_BitFieldUExtractOp : SPV_BitFieldExtractOp<"BitFieldUExtract",
                                                   [UnsignedOp]> {
  let summary = "Extract a bit field from an object, without sign extension.";

  let description = [{
    The semantics are the same as with OpBitFieldSExtract with the exception
    that there is no sign extension. The remaining bits of the result will
    all be 0.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitfield-extract-u-op ::= ssa-id `=` `spv.BitFieldUExtract` ssa-use
                                         `,` ssa-use `,` ssa-use
                                         `:` integer-scalar-vector-type
                                         `,` integer-type `,` integer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.BitFieldUExtract %base, %offset, %count : vector<3xi32>, i8, i8
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Shader]>
  ];
}

// -----

def SPV_BitReverseOp : SPV_BitUnaryOp<"BitReverse", []> {
  let summary = "Reverse the bits in an object.";

  let description = [{
     Results are computed per component.

    Result Type must be a scalar or vector of integer type.

     The type of Base must be the same as Result Type.

    The bit-number n of the result will be taken from bit-number Width - 1 -
    n of Base, where Width is the OpTypeInt operand of the Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                   `vector<` integer-literal `x` integer-type `>`
    bitreverse-op ::= ssa-id `=` `spv.BitReverse` ssa-use
                                 `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitReverse %0 : i32
    %3 = spv.BitReverse %1 : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_Shader]>
  ];
}

// -----

def SPV_BitwiseAndOp : SPV_BitBinaryOp<"BitwiseAnd",
                                       [Commutative, UsableInSpecConstantOp]> {
  let summary = [{
    Result is 1 if both Operand 1 and Operand 2 are 1. Result is 0 if either
    Operand 1 or Operand 2 are 0.
  }];

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitwise-and-op ::= ssa-id `=` `spv.BitwiseAnd` ssa-use, ssa-use
                                  `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitwiseAnd %0, %1 : i32
    %2 = spv.BitwiseAnd %0, %1 : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_BitwiseOrOp : SPV_BitBinaryOp<"BitwiseOr",
                                      [Commutative, UsableInSpecConstantOp]> {
  let summary = [{
    Result is 1 if either Operand 1 or Operand 2 is 1. Result is 0 if both
    Operand 1 and Operand 2 are 0.
  }];

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitwise-or-op ::= ssa-id `=` `spv.BitwiseOr` ssa-use, ssa-use
                                  `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitwiseOr %0, %1 : i32
    %2 = spv.BitwiseOr %0, %1 : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_BitwiseXorOp : SPV_BitBinaryOp<"BitwiseXor",
                                       [Commutative, UsableInSpecConstantOp]> {
  let summary = [{
    Result is 1 if exactly one of Operand 1 or Operand 2 is 1. Result is 0
    if Operand 1 and Operand 2 have the same value.
  }];

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.  The type of
    Operand 1 and Operand 2  must be a scalar or vector of integer type.
    They must have the same number of components as Result Type. They must
    have the same component width as Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    bitwise-xor-op ::= ssa-id `=` `spv.BitwiseXor` ssa-use, ssa-use
                                  `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.BitwiseXor %0, %1 : i32
    %2 = spv.BitwiseXor %0, %1 : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_ShiftLeftLogicalOp : SPV_ShiftOp<"ShiftLeftLogical",
                                         [UsableInSpecConstantOp]> {
  let summary = [{
    Shift the bits in Base left by the number of bits specified in Shift.
    The least-significant bits will be zero filled.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of each Base and Shift must be a scalar or vector of integer
    type. Base and Shift must have the same number of components.  The
    number of components and bit width of the type of Base must be the same
    as in Result Type.

    Shift is treated as unsigned. The result is undefined if Shift is
    greater than or equal to the bit width of the components of Base.

    The number of components and bit width of Result Type must match those
    Base type. All types must be integer types.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    shift-left-logical-op ::= ssa-id `=` `spv.ShiftLeftLogical`
                                          ssa-use `,` ssa-use `:`
                                          integer-scalar-vector-type `,`
                                          integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.ShiftLeftLogical %0, %1 : i32, i16
    %5 = spv.ShiftLeftLogical %3, %4 : vector<3xi32>, vector<3xi16>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_ShiftRightArithmeticOp : SPV_ShiftOp<"ShiftRightArithmetic",
                                             [UsableInSpecConstantOp]> {
  let summary = [{
    Shift the bits in Base right by the number of bits specified in Shift.
    The most-significant bits will be filled with the sign bit from Base.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of each Base and Shift must be a scalar or vector of integer
    type. Base and Shift must have the same number of components.  The
    number of components and bit width of the type of Base must be the same
    as in Result Type.

    Shift is treated as unsigned. The result is undefined if Shift is
    greater than or equal to the bit width of the components of Base.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    shift-right-arithmetic-op ::= ssa-id `=` `spv.ShiftRightArithmetic`
                                              ssa-use `,` ssa-use `:`
                                              integer-scalar-vector-type `,`
                                              integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.ShiftRightArithmetic %0, %1 : i32, i16
    %5 = spv.ShiftRightArithmetic %3, %4 : vector<3xi32>, vector<3xi16>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_ShiftRightLogicalOp : SPV_ShiftOp<"ShiftRightLogical",
                                          [UsableInSpecConstantOp]> {
  let summary = [{
    Shift the bits in Base right by the number of bits specified in Shift.
    The most-significant bits will be zero filled.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

     The type of each Base and Shift must be a scalar or vector of integer
    type. Base and Shift must have the same number of components.  The
    number of components and bit width of the type of Base must be the same
    as in Result Type.

    Shift is consumed as an unsigned integer. The result is undefined if
    Shift is greater than or equal to the bit width of the components of
    Base.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    shift-right-logical-op ::= ssa-id `=` `spv.ShiftRightLogical`
                                           ssa-use `,` ssa-use `:`
                                           integer-scalar-vector-type `,`
                                           integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.ShiftRightLogical %0, %1 : i32, i16
    %5 = spv.ShiftRightLogical %3, %4 : vector<3xi32>, vector<3xi16>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[]>
  ];
}

// -----

def SPV_NotOp : SPV_BitUnaryOp<"Not", [UsableInSpecConstantOp]> {
  let summary = "Complement the bits of Operand.";

  let description = [{
     Results are computed per component, and within each component, per bit.

    Result Type must be a scalar or vector of integer type.

    Operand’s type  must be a scalar or vector of integer type.  It must
    have the same number of components as Result Type.  The component width
    must equal the component width in Result Type.

    <!-- End of AutoGen section -->

    ```
    integer-scalar-vector-type ::= integer-type |
                                  `vector<` integer-literal `x` integer-type `>`
    not-op ::= ssa-id `=` `spv.BitNot` ssa-use `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %2 = spv.Not %0 : i32
    %3 = spv.Not %1 : vector<4xi32>
    ```
  }];
}

#endif // MLIR_DIALECT_SPIRV_IR_BIT_OPS
