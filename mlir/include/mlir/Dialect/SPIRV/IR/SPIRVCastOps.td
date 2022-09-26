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

class SPV_CastOp<string mnemonic, Type resultType, Type operandType,
                 list<Trait> traits = []> :
      SPV_Op<mnemonic,
             !listconcat(traits,
                         [NoSideEffect, SameOperandsAndResultShape])> {
  let arguments = (ins
    SPV_ScalarOrVectorOrCoopMatrixOf<operandType>:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOrCoopMatrixOf<resultType>:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
}

// -----

def SPV_BitcastOp : SPV_Op<"Bitcast", [NoSideEffect]> {
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
    in Operand. Let L be the type, either Result Type or Operand’s type,
    that has the larger number of components. Let S be the other type, with
    the smaller number of components. The number of components in L must be
    an integer multiple of the number of components in S. The first
    component (that is, the only or lowest-numbered component) of S maps to
    the first components of L, and so on,  up to the last component of S
    mapping to the last components of L. Within this mapping, any single
    component of S (mapping to multiple components of L) maps its lower-
    ordered bits to the lower-numbered components of L.

    <!-- End of AutoGen section -->

    ```
    bitcast-op ::= ssa-id `=` `spirv.Bitcast` ssa-use
                   `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.Bitcast %0 : f32 to i32
    %1 = spirv.Bitcast %0 : vector<2xf32> to i64
    %1 = spirv.Bitcast %0 : !spirv.ptr<f32, Function> to !spirv.ptr<i32, Function>
    ```
  }];

  let arguments = (ins
    SPV_ScalarOrVectorOrPtr:$operand
  );

  let results = (outs
    SPV_ScalarOrVectorOrPtr:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
  let hasCanonicalizer = 1;
}

// -----

def SPV_ConvertFToSOp : SPV_CastOp<"ConvertFToS", SPV_Integer, SPV_Float, []> {
  let summary = [{
    Convert value numerically from floating point to signed integer, with
    round toward 0.0.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Float Value must be a scalar or vector of floating-point type.  It must
    have the same number of components as Result Type.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    convert-f-to-s-op ::= ssa-id `=` `spirv.ConvertFToSOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertFToS %0 : f32 to i32
    %3 = spirv.ConvertFToS %2 : vector<3xf32> to vector<3xi32>
    ```
  }];
}

// -----

def SPV_ConvertFToUOp : SPV_CastOp<"ConvertFToU", SPV_Integer, SPV_Float, []> {
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

    <!-- End of AutoGen section -->

    ```
    convert-f-to-u-op ::= ssa-id `=` `spirv.ConvertFToUOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertFToU %0 : f32 to i32
    %3 = spirv.ConvertFToU %2 : vector<3xf32> to vector<3xi32>
    ```
  }];
}

// -----

def SPV_ConvertSToFOp : SPV_CastOp<"ConvertSToF",
                                   SPV_Float,
                                   SPV_Integer,
                                   [SignedOp]> {
  let summary = [{
    Convert value numerically from signed integer to floating point.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Signed Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    convert-s-to-f-op ::= ssa-id `=` `spirv.ConvertSToFOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertSToF %0 : i32 to f32
    %3 = spirv.ConvertSToF %2 : vector<3xi32> to vector<3xf32>
    ```
  }];
}

// -----

def SPV_ConvertUToFOp : SPV_CastOp<"ConvertUToF",
                                   SPV_Float,
                                   SPV_Integer,
                                   [UnsignedOp]> {
  let summary = [{
    Convert value numerically from unsigned integer to floating point.
  }];

  let description = [{
    Result Type must be a scalar or vector of floating-point type.

    Unsigned Value must be a scalar or vector of integer type.  It must have
    the same number of components as Result Type.

     Results are computed per component.

    <!-- End of AutoGen section -->

    ```
    convert-u-to-f-op ::= ssa-id `=` `spirv.ConvertUToFOp` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertUToF %0 : i32 to f32
    %3 = spirv.ConvertUToF %2 : vector<3xi32> to vector<3xf32>
    ```
  }];
}

// -----

def SPV_FConvertOp : SPV_CastOp<"FConvert",
                                SPV_Float,
                                SPV_Float,
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

    <!-- End of AutoGen section -->

    ```
    f-convert-op ::= ssa-id `=` `spirv.FConvertOp` ssa-use
                     `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.FConvertOp %0 : f32 to f64
    %3 = spirv.FConvertOp %2 : vector<3xf32> to vector<3xf64>
    ```
  }];
}

// -----

def SPV_SConvertOp : SPV_CastOp<"SConvert",
                                SPV_Integer,
                                SPV_Integer,
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

    <!-- End of AutoGen section -->

    ```
    s-convert-op ::= ssa-id `=` `spirv.SConvertOp` ssa-use
                     `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.SConvertOp %0 : i32 to i64
    %3 = spirv.SConvertOp %2 : vector<3xi32> to vector<3xi64>
    ```
  }];
}

// -----

def SPV_UConvertOp : SPV_CastOp<"UConvert",
                                SPV_Integer,
                                SPV_Integer,
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

    <!-- End of AutoGen section -->

    ```
    u-convert-op ::= ssa-id `=` `spirv.UConvertOp` ssa-use
                 `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.UConvertOp %0 : i32 to i64
    %3 = spirv.UConvertOp %2 : vector<3xi32> to vector<3xi64>
    ```
  }];
}

// -----
def SPV_PtrCastToGenericOp : SPV_Op<"PtrCastToGeneric", [NoSideEffect]> {
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
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_6>,
    Extension<[]>,
    Capability<[SPV_C_Kernel]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer
  );

  let results = (outs
    SPV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];
}

// -----

def SPV_GenericCastToPtrOp : SPV_Op<"GenericCastToPtr", [NoSideEffect]> {
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
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_6>,
    Extension<[]>,
    Capability<[SPV_C_Kernel]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer
  );

  let results = (outs
    SPV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];
}

// -----

def SPV_GenericCastToPtrExplicitOp : SPV_Op<"GenericCastToPtrExplicit", [NoSideEffect]> {
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

    ```
    [TODO]
    ```mlir

    #### Example:

    ```mlir
       %1 = spirv.GenericCastToPtrExplicitOp %0 : !spirv.ptr<f32, Generic> to
       !spirv.ptr<f32, CrossWorkGroup>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_6>,
    Extension<[]>,
    Capability<[SPV_C_Kernel]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer
  );

  let results = (outs
    SPV_AnyPtr:$result
  );

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer) `to` type($result)
  }];

  let autogenSerialization = 0;
}

#endif // MLIR_DIALECT_SPIRV_IR_CAST_OPS
