//===-- SPIRVCastOps.td - MLIR SPIR-V Cast Ops -------*- tablegen -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains cast ops for the SPIR-V dialect. It corresponds
// to "3.32.11. Conversion Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_CAST_OPS
#define MLIR_DIALECT_SPIRV_IR_CAST_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SPIRV_CastOp<string mnemonic, Type resultType, Type operandType,
                   list<Trait> traits = []> :
      SPIRV_Op<mnemonic,
             !listconcat(traits,
                         [Pure, SameOperandsAndResultShape])> {
  let arguments = (ins
    SPIRV_ScalarOrVectorOrCoopMatrixOf<operandType>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOrCoopMatrixOf<resultType>:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
}

// -----

def SPIRV_BitcastOp : SPIRV_Op<"Bitcast", [Pure]> {
  let summary = "Bit pattern-preserving type conversion.";

  let description = [{
    Result Type must be an OpTypePointer, or a scalar or vector of
    numerical-type.

    Operand must have a type of OpTypePointer, or a scalar or vector of
    numerical-type. It must be a different type than Result Type.

    If either Result Type or Operand is a pointer, the other must be a
    pointer (diverges from the SPIR-V spec).

    If Result Type has a different number of components than Operand, the
    total number of bits in Result Type must equal the total number of bits
    in Operand. Let L be the type, either Result Type or Operand's type,
    that has the larger number of components. Let S be the other type, with
    the smaller number of components. The number of components in L must be
    an integer multiple of the number of components in S. The first
    component (that is, the only or lowest-numbered component) of S maps to
    the first components of L, and so on,  up to the last component of S
    mapping to the last components of L. Within this mapping, any single
    component of S (mapping to multiple components of L) maps its lower-
    ordered bits to the lower-numbered components of L.

    #### Example:

    ```mlir
    %1 = spirv.Bitcast %0 : f32 to i32
    %1 = spirv.Bitcast %0 : vector<2xf32> to i64
    %1 = spirv.Bitcast %0 : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
    ```
  }];

  let arguments = (ins
    SPIRV_ScalarOrVectorOrPtr:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOrPtr:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
  let hasFolder = 1;
}

// -----

def SPIRV_ConvertFToSOp : SPIRV_CastOp<"ConvertFToS", SPIRV_Integer, SPIRV_AnyFloat, []> {
  let summary = [{
    Convert value numerically from floating point to signed integer, with
    round toward 0.0.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.ConvertFToS %0 : f32 to i32
    %3 = spirv.ConvertFToS %2 : vector<3xf32> to vector<3xi32>
    ```
  }];
}

// -----

def SPIRV_ConvertFToUOp : SPIRV_CastOp<"ConvertFToU", SPIRV_Integer, SPIRV_AnyFloat, []> {
  let summary = [{
    Convert value numerically from floating point to unsigned integer, with
    round toward 0.0.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.ConvertFToU %0 : f32 to i32
    %3 = spirv.ConvertFToU %2 : vector<3xf32> to vector<3xi32>
    ```
  }];
}

// -----

def SPIRV_ConvertSToFOp : SPIRV_CastOp<"ConvertSToF",
                                   SPIRV_AnyFloat,
                                   SPIRV_Integer,
                                   [SignedOp]> {
  let summary = [{
    Convert value numerically from signed integer to floating point.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Signed Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.ConvertSToF %0 : i32 to f32
    %3 = spirv.ConvertSToF %2 : vector<3xi32> to vector<3xf32>
    ```
  }];
}

// -----

def SPIRV_ConvertUToFOp : SPIRV_CastOp<"ConvertUToF",
                                   SPIRV_AnyFloat,
                                   SPIRV_Integer,
                                   [UnsignedOp]> {
  let summary = [{
    Convert value numerically from unsigned integer to floating point.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Unsigned Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.ConvertUToF %0 : i32 to f32
    %3 = spirv.ConvertUToF %2 : vector<3xi32> to vector<3xf32>
    ```
  }];
}

// -----

def SPIRV_FConvertOp : SPIRV_CastOp<"FConvert",
                                SPIRV_AnyFloat,
                                SPIRV_AnyFloat,
                                [UsableInSpecConstantOp]> {
  let summary = [{
    Convert value numerically from one floating-point width to another
    width.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.  The component width
    cannot equal the component width in Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.FConvertOp %0 : f32 to f64
    %3 = spirv.FConvertOp %2 : vector<3xf32> to vector<3xf64>
    ```
  }];
}

// -----

def SPIRV_SConvertOp : SPIRV_CastOp<"SConvert",
                                SPIRV_Integer,
                                SPIRV_Integer,
                                [UsableInSpecConstantOp, SignedOp]> {
  let summary = [{
    Convert signed width.  This is either a truncate or a sign extend.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Signed Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.  The component width
    cannot equal the component width in Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.SConvertOp %0 : i32 to i64
    %3 = spirv.SConvertOp %2 : vector<3xi32> to vector<3xi64>
    ```
  }];
}

// -----

def SPIRV_UConvertOp : SPIRV_CastOp<"UConvert",
                                SPIRV_Integer,
                                SPIRV_Integer,
                                [UnsignedOp, UsableInSpecConstantOp]> {
  let summary = [{
    Convert unsigned width. This is either a truncate or a zero extend.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type, whose Signedness
    operand is 0.

    Unsigned Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.  The component width
    cannot equal the component width in Result Type.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.UConvertOp %0 : i32 to i64
    %3 = spirv.UConvertOp %2 : vector<3xi32> to vector<3xi64>
    ```
  }];
}

// -----

def SPIRV_ConvertPtrToUOp : SPIRV_Op<"ConvertPtrToU", []> {
  let summary = [{
    Bit pattern-preserving conversion of a pointer to
    an unsigned scalar integer of possibly different bit width.
  }];

  let description = [{
    Result Type must be a scalar of integer type, whose Signedness operand is 0.

    Pointer must be a physical pointer type. If the bit width of Pointer is
    smaller than that of Result Type, the conversion zero extends Pointer.
    If the bit width of Pointer is larger than that of Result Type,
    the conversion truncates Pointer.

    For same bit width Pointer and Result Type, this is the same as OpBitcast.

    #### Example:

    ```mlir
    %1 = spirv.ConvertPtrToU %0 : !spirv.ptr<i32, Generic> to i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Addresses, SPIRV_C_PhysicalStorageBufferAddresses]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer
  );

  let results = (outs
    SPIRV_Integer:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];

  let hasVerifier = 1;
}


// -----

def SPIRV_ConvertUToPtrOp : SPIRV_Op<"ConvertUToPtr", [UnsignedOp]> {
  let summary = [{
    Bit pattern-preserving conversion of an unsigned scalar integer
    to a pointer.
  }];

  let description = [{
    Result Type must be a physical pointer type.

    Integer Value must be a scalar of integer type, whose Signedness
    operand is 0. If the bit width of Integer Value is smaller
    than that of Result Type, the conversion zero extends Integer Value.
    If the bit width of Integer Value is larger than that of Result Type,
    the conversion truncates Integer Value.

    For same-width Integer Value and Result Type, this is the same as OpBitcast.

    #### Example:

    ```mlir
    %1 = spirv.ConvertUToPtr %0 :  i32 to !spirv.ptr<i32, Generic>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Addresses, SPIRV_C_PhysicalStorageBufferAddresses]>
  ];

  let arguments = (ins
    SPIRV_Integer:$operand
  );

  let results = (outs
    SPIRV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasVerifier = 1;
}

// -----


def SPIRV_PtrCastToGenericOp : SPIRV_Op<"PtrCastToGeneric", [Pure]> {
  let summary = "Convert a pointer’s Storage Class to Generic.";

  let description = [{
    Result Type must be an OpTypePointer. Its Storage Class must be Generic.

    Pointer must point to the Workgroup, CrossWorkgroup, or Function Storage
    Class.

    Result Type and Pointer must point to the same type.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %1 = spirv.PtrCastToGenericOp %0 : !spirv.ptr<f32, CrossWorkGroup> to
         !spirv.ptr<f32, Generic>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Kernel]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer
  );

  let results = (outs
    SPIRV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];
}

// -----

def SPIRV_GenericCastToPtrOp : SPIRV_Op<"GenericCastToPtr", [Pure]> {
  let summary = "Convert a pointer’s Storage Class to a non-Generic class.";

  let description = [{
    Result Type must be an OpTypePointer. Its Storage Class must be
    Workgroup, CrossWorkgroup, or Function.

    Pointer must point to the Generic Storage Class.

    Result Type and Pointer must point to the same type.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
       %1 = spirv.GenericCastToPtrOp %0 : !spirv.ptr<f32, Generic> to
       !spirv.ptr<f32, CrossWorkGroup>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Kernel]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer
  );

  let results = (outs
    SPIRV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];
}

// -----

def SPIRV_GenericCastToPtrExplicitOp : SPIRV_Op<"GenericCastToPtrExplicit", [Pure]> {
  let summary = [{
    Attempts to explicitly convert Pointer to Storage storage-class pointer
    value.
  }];

  let description = [{
    Result Type must be an OpTypePointer. Its Storage Class must be Storage.

    Pointer must have a type of OpTypePointer whose Type is the same as the
    Type of Result Type.Pointer must point to the Generic Storage Class. If
    the cast fails, the instruction result is an OpConstantNull pointer in
    the Storage Storage Class.

    Storage must be one of the following literal values from Storage Class:
    Workgroup, CrossWorkgroup, or Function.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
       %1 = spirv.GenericCastToPtrExplicitOp %0 : !spirv.ptr<f32, Generic> to
       !spirv.ptr<f32, CrossWorkGroup>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Kernel]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer
  );

  let results = (outs
    SPIRV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];

  let autogenSerialization = 0;
}

#endif // MLIR_DIALECT_SPIRV_IR_CAST_OPS
