//===- QuantOps.td - Quantization operation definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Quantization.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QUANT_QUANT_OPS_
#define DIALECT_QUANT_QUANT_OPS_

include "mlir/Dialect/Quant/QuantOpsBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class quant_Op<string mnemonic, list<Trait> traits> :
    Op<Quantization_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Quantization casts
//===----------------------------------------------------------------------===//
// A QuantizeCast (qcast) represents a potential type shift from a quantizable
// type to a quantized type.
//
// At runtime, a qcast will apply the transformation expressed by its
// operand and result type. For flexibility during transformation, it is also
// possible to have a qcast that performs no transformation (both its
// operand and result type are quantizable).
//
// A qcast will typically originate from either:
//   a) An expressed or implied constraint in the source dialect which signals
//      that a certain level of quantization is possible or required.
//   b) An inference made by a quantization algorithm indicating that a
//      quantized representation may be acceptable.
//
// Especially early in transformation, it is common to have pairs of
// qcast/dcast at points where a transition to a quantized type is
// required. In addition, it is also common to have an identity qcast
// (where the operand and result type are not quantized) at all points where
// it is legal to use a quantized representation (but is not known to be
// acceptable).
def quant_QuantizeCastOp : quant_Op<"qcast", [Pure]> {
  let arguments = (ins quant_RealValueType:$arg);
  let results = (outs quant_RealValueType);
}

// A DequantizeCast op (dcast) represents the inverse of a qcast,
// converting back from a quantized to quantizable (expressed) type.
//
// Like qcasts, a dcast is allowed to have both its operand and result
// as non quantized types. This facilitates transformations and marks edges
// where the computation must be carried out in the expressed type.
//
// Especially early in transformation, it is common to have dcasts on
// all operands to ops that must operate with the expressed type (typically
// math ops prior to lowering to target-specific, quantized kernels).
def quant_DequantizeCastOp : quant_Op<"dcast", [Pure]> {
  let arguments = (ins quant_RealValueType:$arg);
  let results = (outs quant_RealValueType);
}

// A StorageCast (scast) represents a cast from or to a type based on the
// storage type and a type based on a corresponding quantized type.
//
// This op exists to ensure type coherency for between parts of the computation
// which are operating directly on an underlying storage type and those which
// operate on quantized values.
//
// Examples from storage to quantized type:
//   i8 -> !quant<"uniform[i8:f32]{1.0}">
//   tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
//   vector<4xi8> -> vector<4x!quant<"uniform[i8:f32]{1.0}">>
def quant_StorageCastOp : quant_Op<"scast", [Pure]> {
  let arguments = (ins quant_RealOrStorageValueType:$arg);
  let results = (outs quant_RealOrStorageValueType);
  let hasFolder = 1;
}

#endif // DIALECT_QUANT_QUANT_OPS_
