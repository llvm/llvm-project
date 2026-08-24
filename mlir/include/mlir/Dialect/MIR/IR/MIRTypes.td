//===- MIRTypes.td - MIR dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LLT (Low Level Type) modeling. The type of an SSA value in the MIR dialect
// is the LLT of the generic virtual register it represents.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIR_IR_MIRTYPES_TD
#define MLIR_DIALECT_MIR_IR_MIRTYPES_TD

include "mlir/Dialect/MIR/IR/MIRBase.td"

//===----------------------------------------------------------------------===//
// ScalarType : LLT sN
//===----------------------------------------------------------------------===//

def MIR_ScalarType : MIR_Type<"Scalar", "scalar"> {
  let summary = "LLT scalar type (sN)";
  let description = [{
    A scalar Low Level Type of a given bit width, e.g. `!mir.scalar<32>` for the
    LLT `s32`.
  }];
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

//===----------------------------------------------------------------------===//
// PointerType : LLT pN
//===----------------------------------------------------------------------===//

def MIR_PointerType : MIR_Type<"Pointer", "pointer"> {
  let summary = "LLT pointer type (pN)";
  let description = [{
    A pointer Low Level Type carrying an address space and a size in bits, e.g.
    `!mir.pointer<0, 64>` for the LLT `p0` on a 64-bit target.
  }];
  let parameters = (ins "unsigned":$addressSpace, "unsigned":$sizeInBits);
  let assemblyFormat = "`<` $addressSpace `,` $sizeInBits `>`";
}

//===----------------------------------------------------------------------===//
// VectorType : LLT <N x sM>
//===----------------------------------------------------------------------===//

def MIR_VectorType : MIR_Type<"Vector", "vector"> {
  let summary = "LLT fixed-length vector type";
  let description = [{
    A fixed-length vector Low Level Type of a scalar element, e.g.
    `!mir.vector<4 x !mir.scalar<32>>` for the LLT `<4 x s32>`.
  }];
  let parameters = (ins "unsigned":$numElements, "::mlir::mir::ScalarType":$elementType);
  let assemblyFormat = "`<` $numElements `x` qualified($elementType) `>`";
}

//===----------------------------------------------------------------------===//
// Aggregate constraint : any LLT-typed value
//===----------------------------------------------------------------------===//

def MIR_AnyLLT
    : AnyTypeOf<[MIR_ScalarType, MIR_PointerType, MIR_VectorType],
                "any MIR LLT type">;

#endif // MLIR_DIALECT_MIR_IR_MIRTYPES_TD
