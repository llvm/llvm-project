//===- VectorTransformOps.td - Vector transform ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_TRANSFORM_OPS
#define VECTOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def ApplyVectorToLLVMConversionPatternsOp : Op<Transform_Dialect,
    "apply_conversion_patterns.vector.vector_to_llvm",
    [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface,
                               ["verifyTypeConverter"]>]> {
  let description = [{
    Collects patterns that convert vector dialect ops to LLVM dialect ops. These
    patterns require an "LLVMTypeConverter".

    The patterns can be customized as follows:
    - `reassociate_fp_reductions`: Allows LLVM to reassociate floating-point
      reductions for speed.
    - `force_32bit_vector_indices`: Allows the compiler to assume that vector
      indices fit in 32-bit if that yields faster code.
  }];

  let arguments = (ins
      DefaultValuedAttr<BoolAttr, "false">:$reassociate_fp_reductions,
      DefaultValuedAttr<BoolAttr, "true">:$force_32bit_vector_indices);
  let assemblyFormat = "attr-dict";
}


def ApplyCastAwayVectorLeadingOneDimPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.cast_away_vector_leading_one_dim",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect a set of leading one dimension removal patterns.

    These patterns insert vector.shape_cast to remove leading one dimensions
    to expose more canonical forms of read/write/insert/extract operations.
    With them, there are more chances that we can cancel out extract-insert
    pairs or forward write-read pairs.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyRankReducingSubviewPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.rank_reducing_subview_patterns",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Apply opt-in vector transfer permutation patterns that include:
      - TransferReadDropUnitDimsPattern
      - TransferWriteDropUnitDimsPattern

    These patterns have the effect of rewriting a vector.transfer with unit
    dimensions into a rank-reduced version thanks to subview operations.
    This is complemented by shape_cast folding patterns.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyTransferPermutationPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.transfer_permutation_patterns",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Apply opt-in vector transfer permutation patterns that include:
      - TransferReadPermutationLowering
      - TransferWritePermutationLowering
      - TransferOpReduceRank
      - TransferWriteNonPermutationLowering

    These patterns have the effect of rewriting a vector.transfer with an
    arbitrary permutation_map to a vector.transfer with a permutation_map that
    is a minor identity followed by a vector.transpose.

    In other words, this makes the vector.transfer contiguous on the most minor
    dimensions and materializes the permutation_map as a vector.transpose.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerBitCastPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_bitcast",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector bitcast operations should be lowered to
    finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerBroadcastPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_broadcast",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector broadcast operations should be lowered to
    finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerContractionPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_contraction",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contraction-like operations should be lowered to
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins DefaultValuedAttr<VectorContractLoweringAttr,
      "vector::VectorContractLowering::OuterProduct">:$lowering_strategy
  );
  let assemblyFormat = [{
    (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
  }];
}

def ApplyLowerCreateMaskPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_create_mask",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector create_mask-like operations should be lowered to
    finer-grained vector primitives.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerMasksPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_masks",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector.create_mask and vector.constant_mask operations
    should be lowered to finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerMaskedTransfersPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_masked_transfers",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Apply opt-in patterns that lower vector.mask operations surrounding
    side-effecting ops:
      - MaskedTransferReadOpPattern
      - MaskedTransferWriteOpPattern
      - MaskedGatherOpPattern

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyMaterializeMasksPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.materialize_masks",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that mask operations should be lowered to fine-grained arithemtic
    operations.

    This is usually the last step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerMultiReductionPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_multi_reduction",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector multi_reduction-like operations should be lowered to
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins DefaultValuedAttr<VectorMultiReductionLoweringAttr,
      "vector::VectorMultiReductionLowering::InnerParallel">:$lowering_strategy
  );

  let assemblyFormat = [{
    (`lowering_strategy` `=` $lowering_strategy^)? attr-dict
  }];
}

def ApplyLowerOuterProductPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_outerproduct",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that the vector outerproduct operations should be lowered to
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerGatherPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_gather",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector.gather operations should be lowered to
    finer-grained vector primitives.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerScanPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_scan",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector.scan operations should be lowered to
    finer-grained vector primitives.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerShapeCastPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_shape_cast",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector shape_cast operations should be lowered to
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyLowerTransferPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_transfer",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector transfer operations should be lowered to finer-grained
    vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "1">:$max_transfer_rank);

  let assemblyFormat = [{
    (`max_transfer_rank` `=` $max_transfer_rank^)? attr-dict
  }];
}

def ApplyLowerTransposePatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_transpose",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector transpose-like operations should be lowered to
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins
     DefaultValuedAttr<VectorTransposeLoweringAttr,
       "vector::VectorTransposeLowering::EltWise">:$lowering_strategy,
     DefaultValuedAttr<BoolAttr, "false">:$avx2_lowering_strategy
  );

  let assemblyFormat = [{
    oilist (
      `lowering_strategy` `=` $lowering_strategy
      | `avx2_lowering_strategy` `=` $avx2_lowering_strategy
    )
    attr-dict
  }];
}

def ApplyLowerInterleavePatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.lower_interleave",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector interleave operations should be lowered to
    finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyInterleaveToShufflePatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.interleave_to_shuffle",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that 1D vector interleave operations should be rewritten as
    vector shuffle operations.

    This is motivated by some current codegen backends not handling vector
    interleave operations.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyRewriteNarrowTypePatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.rewrite_narrow_types",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector narrow rewrite operations should be applied.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.

    Warning: these patterns currently only work for little endian targets.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplySplitTransferFullPartialPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.split_transfer_full_partial",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector transfer operations should be split to full and
    partial parts.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins
     DefaultValuedAttr<VectorTransferSplitAttr,
       "vector::VectorTransferSplit::LinalgCopy">:$split_transfer_strategy
  );

  let assemblyFormat = [{
    (`split_transfer_strategy` `=` $split_transfer_strategy^)? attr-dict
  }];
}

def ApplyTransferToScfPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.transfer_to_scf",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector transfer operations should be rewritten with scf.for
    loops over finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins
     DefaultValuedAttr<I64Attr, "1">:$max_transfer_rank,
     DefaultValuedAttr<BoolAttr, "false">:$full_unroll
  );

  let assemblyFormat = [{
    oilist (
        `max_transfer_rank` `=` $max_transfer_rank
      | `full_unroll` `=` $full_unroll
    )
    attr-dict
  }];
}

def ApplyFoldArithExtensionPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.fold_arith_extension",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect a set of patterns that fold arithmetic extension on floating point
    into vector contract for the backends with native support.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyFoldElementwiseToVectorPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.elementwise_to_vector",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect a set of patterns that fold elementwise op on vectors to the vector 
    dialect.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyVectorReductionToContractPatternsOp : Op<Transform_Dialect,
    "apply_patterns.vector.reduction_to_contract",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Apply opt-in patterns that convert reductions to contract:
      - MultiReduceToContract
      - CombineContractBroadcast
      - CombineContractABTranspose
      - CombineContractResultTranspose
      - ReorderCastOpsOnBroadcast
      - ReorderElementwiseOpsOnTranspose

    These patterns have the effect of rewriting a vector.multi_reduce into a
    vector.contract.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // VECTOR_TRANSFORM_OPS
