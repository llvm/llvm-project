//==- StandardToSPIRV.td - Standard Ops to SPIR-V Patterns ---*- tablegen -*==//

// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower standard ops to SPIR-V.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSPIRV_TD
#define MLIR_CONVERSION_STANDARDTOSPIRV_TD

include "mlir/Dialect/StandardOps/Ops.td"
include "mlir/Dialect/SPIRV/SPIRVOps.td"

class BinaryOpPattern<Type type, Op src, Op tgt> :
      Pat<(src SPV_ScalarOrVectorOf<type>:$l, SPV_ScalarOrVectorOf<type>:$r),
          (tgt $l, $r)>;

def : BinaryOpPattern<SPV_Bool, AndOp, SPV_LogicalAndOp>;
def : BinaryOpPattern<SPV_Bool, OrOp, SPV_LogicalOrOp>;
def : BinaryOpPattern<SPV_Float, AddFOp, SPV_FAddOp>;
def : BinaryOpPattern<SPV_Float, DivFOp, SPV_FDivOp>;
def : BinaryOpPattern<SPV_Float, MulFOp, SPV_FMulOp>;
def : BinaryOpPattern<SPV_Float, RemFOp, SPV_FRemOp>;
def : BinaryOpPattern<SPV_Float, SubFOp, SPV_FSubOp>;

// Constant Op
// TODO(ravishankarm): Handle lowering other constant types.
def : Pat<(ConstantOp:$result $valueAttr),
          (SPV_ConstantOp $valueAttr),
          [(SPV_ScalarOrVector $result)]>;

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_TD
