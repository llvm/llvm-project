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
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

// TODO: not isolated from above and better targetability of the op.
// TODO: not a functional-style transform.
class TransformWithPatternsOp<string opname, list<Trait> traits = []>
    : Op<Transform_Dialect, opname,
         !listconcat([TransformOpInterface,
                      TransformEachOpTrait,
                      TransformWithPatternsOpTrait,
                      MemoryEffectsOpInterface,
                      FunctionalStyleTransformOpTrait], traits)> {
  let extraClassDeclaration = [{
    void populatePatterns(RewritePatternSet &patterns);
  }];
}

def ApplyRankReducingSubviewPatternsOp : 
  TransformWithPatternsOp<"vector.apply_rank_reducing_subview_patterns"> {
  let description = [{
    Apply opt-in vector transfer permutation patterns that include:
      - TransferReadDropUnitDimsPattern
      - TransferWriteDropUnitDimsPattern
    
    These patterns have the effect of rewriting a vector.transfer with unit 
    dimensions into a rank-reduced version thanks to subview operations.
    This is complemented by shape_cast folding patterns.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

def ApplyTransferPermutationPatternsOp : 
  TransformWithPatternsOp<"vector.apply_transfer_permutation_patterns"> {
  let description = [{
    Apply opt-in vector transfer permutation patterns that include:
      - TransferReadPermutationLowering
      - TransferWritePermutationLowering
      - TransferOpReduceRank
      - TransferWriteNonPermutationLowering
    
    These patterns have the effect of rewriting a vector.transfer with an 
    arbitrary permutation_map to a vector.transfer with a permutation_map that is
    a minor identity followed by a vector.transpose.

    In other words, this makes the vector.transfer contiguous on the most minor
    dimensions and materializes the permutation_map as a vector.transpose.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

def LowerBroadcastOp : TransformWithPatternsOp<"vector.lower_broadcast"> {
  let description = [{
    Indicates that the vector outerproduct operations nested under the isolated
    from above op `target` should be lowered to finer-grained vector primitives.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

// TODO: evolve lowering_strategy to proper enums.
def LowerContractionOp : TransformWithPatternsOp<"vector.lower_contraction"> {
  let description = [{
    Indicates that the vector contraction-like operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorContractLoweringAttr,
       "vector::VectorContractLowering::OuterProduct">:$lowering_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`lowering_strategy` `=` $lowering_strategy^)?
    attr-dict
    `:` functional-type($target, results)
  }];
}

def LowerMasksOp : TransformWithPatternsOp<"vector.lower_masks"> {
  let description = [{
    Indicates that the vector.create_mask and vector.constant_mask operations
    nested under the isolated from above op `target` should be lowered to
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

def LowerMaskedTransfersOp : TransformWithPatternsOp<"vector.lower_masked_transfers"> {
  let description = [{
    Indicates that masked vector.transfer and vector.gather operations nested
    under the isolated from above op `target` should be lowered to finer-grained
    vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

// TODO: evolve lowering_strategy to proper enums.
def LowerMultiReductionOp
    : TransformWithPatternsOp<"vector.lower_multi_reduction"> {
  let description = [{
    Indicates that the vector multi_reduction-like operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorMultiReductionLoweringAttr,
       "vector::VectorMultiReductionLowering::InnerParallel">:
         $lowering_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`lowering_strategy` `=` $lowering_strategy^)?
    attr-dict
    `:` functional-type($target, results)
  }];
}

def LowerOuterProductOp : TransformWithPatternsOp<"vector.lower_outerproduct"> {
  let description = [{
    Indicates that the vector outerproduct operations nested under the isolated
    from above op `target` should be lowered to finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

def LowerShapeCastOp : TransformWithPatternsOp<"vector.lower_shape_cast"> {
  let description = [{
    Indicates that the vector shape_cast operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    attr-dict
    `:` functional-type($target, results)
  }];
}

def LowerTransferOp : TransformWithPatternsOp<"vector.lower_transfer"> {
  let description = [{
    Indicates that the vector transfer operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<I64Attr, "1">:$max_transfer_rank
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`max_transfer_rank` `=` $max_transfer_rank^)?
    attr-dict
    `:` functional-type($target, results)
  }];
}

// TODO: evolve lowering_strategy to proper enums.
def LowerTransposeOp : TransformWithPatternsOp<"vector.lower_transpose"> {
  let description = [{
    Indicates that the vector transpose-like operations nested under the 
    isolated from above op `target` should be lowered to finer-grained vector
    primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorTransposeLoweringAttr,
       "vector::VectorTransposeLowering::EltWise">:$lowering_strategy,
     DefaultValuedAttr<BoolAttr, "false">:$avx2_lowering_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    oilist (
      `lowering_strategy` `=` $lowering_strategy
      | `avx2_lowering_strategy` `=` $avx2_lowering_strategy
    )
    attr-dict
    `:` functional-type($target, results)
  }];
}

// TODO: evolve split_transfer_strategy to proper enums.
def SplitTransferFullPartialOp
    : TransformWithPatternsOp<"vector.split_transfer_full_partial"> {
  let description = [{
    Indicates that the vector transfer operations nested under the 
    isolated from above op `target` should be split to full and partial parts.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<VectorTransferSplitAttr,
       "vector::VectorTransferSplit::LinalgCopy">:$split_transfer_strategy
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    (`split_transfer_strategy` `=` $split_transfer_strategy^)?
    attr-dict
    `:` functional-type($target, results)
  }];
}

def TransferToScfOp : TransformWithPatternsOp<"vector.transfer_to_scf"> {
  let description = [{
    Indicates that the vector transfer operations nested under the 
    isolated from above op `target` should be rewritten with scf.for loops over
    finer-grained vector primitives.

    This is usually a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
     DefaultValuedAttr<I64Attr, "1">:$max_transfer_rank,
     DefaultValuedAttr<BoolAttr, "false">:$full_unroll
  );
  let results = (outs TransformHandleTypeInterface:$results);

  let assemblyFormat = [{
    $target
    oilist (
        `max_transfer_rank` `=` $max_transfer_rank
      | `full_unroll` `=` $full_unroll
    )
    attr-dict
    `:` functional-type($target, results)
  }];
}

#endif // VECTOR_TRANSFORM_OPS
