//===- XeGPUTypes.td - XeGPU dialect types definition -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD
#define MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD

include "mlir/IR/BuiltinTypes.td"
include "mlir/Dialect/XeGPU/IR/XeGPUAttrs.td"
include "mlir/Dialect/XeGPU/IR/XeGPUDialect.td"

def XeGPU_IntType: AnyTypeOf<[I1, I8, I16, I32, I64, SI1, SI8, SI16, SI32, SI64, UI1, UI8, UI16, UI32, UI64]>;
def XeGPU_FloatType: AnyTypeOf<[F16, F32, F64, BF16, TF32]>;
def XeGPU_ScalarType: AnyTypeOf<[XeGPU_IntType, XeGPU_FloatType]>;
def XeGPU_BaseAddrType: AnyTypeOf<[MemRefRankOf<[XeGPU_ScalarType], [1, 2]>, UI64, UI32, I64, I32]>;
def XeGPU_DpasOpType: VectorOfRankAndType<[2, 3], [XeGPU_ScalarType]>;
def XeGPU_OffsetType: VectorOfRankAndType<[1], [Index]>;
def XeGPU_MaskType: AnyTypeOf<[VectorOfRankAndType<[1,2], [I1]>, I1]>;
def XeGPU_ValueType: AnyTypeOf<[VectorOfRankAndType<[1,2,3,4], [XeGPU_ScalarType]>, XeGPU_ScalarType]>;
def XeGPU_Vector2DType: VectorOfRankAndType<[2], [XeGPU_ScalarType]>;

// common base class for types in XeGPU dialect
class XeGPUTypeDef<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<XeGPU_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUTYPES_TD
