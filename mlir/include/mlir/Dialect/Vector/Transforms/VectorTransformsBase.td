//===- VectorTransformBase.td - Vector transform ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_TRANSFORMS_BASE
#define VECTOR_TRANSFORMS_BASE

include "mlir/IR/EnumAttr.td"

// Lower transpose into element-wise extract and inserts.
def VectorTransposeLowering_Elementwise:
  I32EnumAttrCase<"EltWise",  0, "eltwise">;
// Lower 2-D transpose to `vector.flat_transpose`, maps 1-1 to LLVM matrix
// intrinsics.
def VectorTransposeLowering_FlatTranspose:
  I32EnumAttrCase<"Flat",  1, "flat_transpose">;
// Lower 2-D transpose to `vector.shuffle`.
def VectorTransposeLowering_Shuffle:
  I32EnumAttrCase<"Shuffle",  2, "shuffle">;
def VectorTransposeLoweringAttr : I32EnumAttr<
    "VectorTransposeLowering",
    "control the lowering of `vector.transpose` operations.",
    [VectorTransposeLowering_Elementwise, VectorTransposeLowering_FlatTranspose,
     VectorTransposeLowering_Shuffle]> {
  let cppNamespace = "::mlir::vector";
}

// Lower multi_reduction into outer-reduction and inner-parallel ops.
def VectorMultiReductionLowering_InnerParallel:
  I32EnumAttrCase<"InnerParallel", 0, "innerparallel">;
// Lower multi_reduction into outer-parallel and inner-reduction ops.
def VectorMultiReductionLowering_InnerReduction:
  I32EnumAttrCase<"InnerReduction", 1, "innerreduction">;
def VectorMultiReductionLoweringAttr: I32EnumAttr<
    "VectorMultiReductionLowering",
    "control the lowering of `vector.multi_reduction`.",
  [VectorMultiReductionLowering_InnerParallel,
   VectorMultiReductionLowering_InnerReduction]> {
  let cppNamespace = "::mlir::vector";
}

// Progressively lower to finer grained `vector.contract` and dot-products.
def VectorContractLowering_Dot: I32EnumAttrCase<"Dot", 0, "dot">;
// Lower to `vector.matrix_multiply`, maps 1-1 to LLVM matrix intrinsics.
def VectorContractLowering_Matmul:
  I32EnumAttrCase<"Matmul", 1, "matmulintrinsics">;
// Lower to `vector.outerproduct`.
def VectorContractLowering_OuterProduct:
  I32EnumAttrCase<"OuterProduct", 2, "outerproduct">;
// Lower contract with all reduction dimensions unrolled to 1 to a vector
// elementwise operations.
def VectorContractLowering_ParallelArith:
  I32EnumAttrCase<"ParallelArith", 3, "parallelarith">;
def VectorContractLoweringAttr: I32EnumAttr<
    "VectorContractLowering",
    "control the lowering of `vector.contract` operations.",
  [VectorContractLowering_Dot, VectorContractLowering_Matmul,
   VectorContractLowering_OuterProduct, VectorContractLowering_ParallelArith]> {
  let cppNamespace = "::mlir::vector";
}

// Do not split vector transfer operations.
def VectorTransferSplit_None: I32EnumAttrCase<"None", 0, "none">;
// Split using in-bounds + out-of-bounds vector.transfer operations.
def VectorTransferSplit_VectorTransfer:
  I32EnumAttrCase<"VectorTransfer", 1, "vector-transfer">;
// Split using an in-bounds vector.transfer + linalg.fill + linalg.copy
// operations.
def VectorTransferSplit_LinalgCopy:
  I32EnumAttrCase<"LinalgCopy", 2, "linalg-copy">;
// Do not split vector transfer operation but instead mark it as "in-bounds".
def VectorTransferSplit_ForceInBounds:
  I32EnumAttrCase<"ForceInBounds", 3, "force-in-bounds">;
def VectorTransferSplitAttr: I32EnumAttr<
    "VectorTransferSplit",
    "control the splitting of `vector.transfer` operations into in-bounds"
    " and out-of-bounds variants.",
  [VectorTransferSplit_None, VectorTransferSplit_VectorTransfer,
   VectorTransferSplit_LinalgCopy, VectorTransferSplit_ForceInBounds]> {
  let cppNamespace = "::mlir::vector";
}
#endif
