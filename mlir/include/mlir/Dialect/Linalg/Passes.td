//===-- Passes.td - Linalg pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_PASSES
#define MLIR_DIALECT_LINALG_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/IR/Constraints.td"

// ------------------ Begin of "form" conversions
//
// These conversions allow for the transformation of linalg ops between
// different forms. Structured ops can be represented in different forms,
// such as generic ops, category ops, and named ops.
//
// The operation tree is as follows:
//   generic     category      named
//  ---------|-------------|----------
//  generic ---> contract ----> matmul
//           |              \-> batch_matmul
//           |              \-> batch_reduce_matmul
//           |              \-> ...
//           \-> elementwise -> add
//                          \-> sub
//                          \-> ...
//
// Morphisms between representations can happen in the following 6 ways:
//  generic <---> category <---> named
//      \-------------------------/
//
// generic subsumes category which subsumes structured named (not softmax,
// convolutions, etc). The generalization path is guaranteed, the
// specialization path is not.

def LinalgMorphOpsPass : Pass<"linalg-morph-ops"> {
  let summary = "Convert linalg ops between forms";

  let description = [{
    Convert a linalg op from one representation to another equivalent.
    For example, a linalg named op `linalg.add` can also be written as an
    category op `linalg.elementwise`, and can also be re-written as
    a `linalg.generic`, giving the morphism:

      named-op <--> category_op (elementwise, contraction, ..) <--> generic

    Note that the set of `linalg.generic` subsumes named and category ops
    and therefore not all `linalg.genric` can be converted to  named or
    category op. Similarly, catgory ops subsume named ops.

    Note:
     Legacy converters:
     `--linalg-generalize-named-ops` is the path `named-op --> generic-op`
     `--linalg-specialize-generic-ops` is the path `named-op <-- generic-op`
  }];
  let dependentDialects = ["linalg::LinalgDialect"];

  let options = [
    // Generalization path is guaranteed.
    Option<"namedToCategory", "named-to-category", "bool", /*default=*/"false",
           "convert named ops to category op e.g. `linalg.elementwise`">,
    Option<"categoryToGeneric", "category-to-generic", "bool", /*default=*/"false",
           "convert category ops e.g. `linalg.elementwise` to `linalg.generic`">,
    Option<"namedToGeneric", "named-to-generic", "bool", /*default=*/"false",
           "convert named ops e.g. `linalg.add` to `linalg.generic`">,
    
    // Specialization path is not guaranteed.
    Option<"genericToNamed", "generic-to-named", "bool", /*default=*/"false",
           "convert linalg.generic to equivalent named ops"> ];
    //  TODOs: `generic-to-category`, `category-to-named`
}

def LinalgGeneralizeNamedOpsPass : Pass<"linalg-generalize-named-ops">,
                                   Deprecated<"Use 'linalg-morph-ops' instead."> {
  let summary = "Convert named ops into generic ops";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def LinalgSpecializeGenericOpsPass : Pass<"linalg-specialize-generic-ops">,
                                     Deprecated<"Use 'linalg-morph-ops' instead."> {
  let summary = "Convert generic ops back to named ops";
  let dependentDialects = ["linalg::LinalgDialect"];
}

// ------------------ End of "form" conversions

def SimplifyDepthwiseConvPass: Pass<"simplify-depthwise-conv"> {
  let summary = "Simplify depthwise convolution.";
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
}

def ConvertElementwiseToLinalgPass : Pass<"convert-elementwise-to-linalg", ""> {
  let summary = "Convert ElementwiseMappable ops to linalg";
  let description = [{
    Convert ops with the `ElementwiseMappable` trait to linalg parallel loops.

    This pass only converts ops that operate on ranked tensors. It can be
    run on op which contains linalg ops (most commonly a
    FunctionOpInterface op).
  }];
  let dependentDialects = ["linalg::LinalgDialect", "memref::MemRefDialect"];
}

def ConvertLinalgToAffineLoopsPass : Pass<"convert-linalg-to-affine-loops"> {
  let summary = "Lower the operations from the linalg dialect into affine "
                "loops";
  let dependentDialects = [
    "affine::AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"];
}

def ConvertLinalgToLoopsPass : Pass<"convert-linalg-to-loops"> {
  let summary = "Lower the operations from the linalg dialect into loops";
  let description = [{
    Lowers the `linalg` ops to loop nests using `scf.for`.

    Pre-condition: the operands used by the `linalg` ops have buffer semantics,
    i.e., tensor operands and results must be converted to memrefs via
    bufferization.
  }];
  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "affine::AffineDialect"
  ];
}

def ConvertLinalgToParallelLoopsPass
    : Pass<"convert-linalg-to-parallel-loops"> {
  let summary = "Lower the operations from the linalg dialect into parallel "
                "loops";
  let dependentDialects = [
    "affine::AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def LinalgFoldUnitExtentDimsPass : Pass<"linalg-fold-unit-extent-dims", ""> {
  let summary = "Remove unit-extent dimension in Linalg ops on tensors";
  let options = [
    Option<"useRankReducingSlices", "use-rank-reducing-slices", "bool",
           /*default=*/"false",
           "Generate rank-reducing slices instead of reassociative reshapes">
  ];
  let dependentDialects = [
    "linalg::LinalgDialect", "affine::AffineDialect", "memref::MemRefDialect"
  ];
}

def LinalgElementwiseOpFusionPass : Pass<"linalg-fuse-elementwise-ops"> {
  let summary = "Fuse elementwise operations on tensors";
  let dependentDialects = [
    "affine::AffineDialect", "linalg::LinalgDialect", "memref::MemRefDialect"
  ];
}

def LinalgInlineScalarOperandsPass : Pass<"linalg-inline-scalar-operands"> {
  let summary = "Inline scalar operands into linalg generic ops";
  let dependentDialects = [
    "linalg::LinalgDialect"
  ];
}

def LinalgFoldIntoElementwisePass : Pass<"linalg-fold-into-elementwise"> {
  let summary = "Fold transpose ops into elementwise";
  let dependentDialects = ["linalg::LinalgDialect"];

  let description = [{
    Fold transpose ops that feed `linalg.elementwise` into the elementwise op
    by updating its indexing maps. `linalg.transpose` producers whose consumer
    indexing map is the identity are absorbed, turning the permutation into
    the elementwise map itself. Other operands remain untouched.
  }];
}

def LinalgBlockPackMatmul : Pass<"linalg-block-pack-matmul"> {
  let summary = "Convert linalg matmul ops to block layout and back";
  let description = [{
    Pack a matmul operation into blocked layout with two levels of subdivision:
    - major 2D blocks - outer dimensions, consist of minor blocks
    - minor 2D blocks - inner dimensions, consist of scalar elements

    A 2D matmul MxNxK gets reshaped into blocked 4D representation
    as: [MB][NB][mb][nb] += [MB][KB][mb][kb] * [NB][KB][nb][kb]
    where the (MB, NB, KB) dimensions represent the major blocks,
    and the (mb, nb, kb) are the minor blocks of their respective
    original 2D dimensions (M, N, K).

    Depending on the initial operands' data layout and the specified
    packing options, the major blocks dimensions might get transposed
    e.g., [MB][KB] -> [KB][MB]. The minor blocks can also be transposed
    e.g., [mb][kb] -> [kb][mb].
    Any present batch dimensions remain unchanged.
    The final result is unpacked back to the original shape.

    For example, given a matmul operation:
    ```mlir
      %res = linalg.matmul ins(%A, %B) outs(%C)
    ```
    the default transformation result can be represented as:
    ```mlir
      %A_packed = pack %A : 2D <MxK> -> 4D <MBxKBxmbxkb>
      %B_packed = pack %B : 2D <KxN> -> 4D <NBxKBxnbxkb>
      %C_packed = pack %C : 2D <MxN> -> 4D <MBxNBxmbxnb>
      %res_packed = linalg.mmt4d ins(%A_packed, %B_packed) outs(%C_packed)
      %res = unpack %res_packed : 4D <MBxNBxmbxnb> -> 2D <MxN>
    ```
  }];
  let dependentDialects = ["linalg::LinalgDialect", "tensor::TensorDialect"];
  let options = [
    ListOption<"blockFactors", "block-factors", "int64_t",
               "Block factors (mb, nb, kb) for relayout">,
    Option<"allowPadding", "allow-padding", "bool",
           /*default=*/"true",
           "Allow packing padding">,
    ListOption<"mnkPaddedSizesNextMultipleOf", "mnk-padded-multiples", "int64_t",
               "Next multiples of the packing sizes">,
    ListOption<"mnkOrder", "mnk-order", "int64_t",
               "Permutation of matmul (M, N, K) dimensions order">,
    Option<"lhsTransposeOuterBlocks", "lhs-transpose-outer-blocks", "bool",
           /*default=*/"false",
           "Transpose LHS outer block layout [MB][KB] -> [KB][MB]">,
    Option<"lhsTransposeInnerBlocks", "lhs-transpose-inner-blocks", "bool",
           /*default=*/"false",
           "Transpose LHS inner block layout [mb][kb] -> [kb][mb]">,
    Option<"rhsTransposeOuterBlocks", "rhs-transpose-outer-blocks", "bool",
           /*default=*/"true",
           "Transpose RHS outer block layout [KB][NB] -> [NB][KB]">,
    Option<"rhsTransposeInnerBlocks", "rhs-transpose-inner-blocks", "bool",
           /*default=*/"true",
           "Transpose RHS inner block layout [kb][nb] -> [nb][kb]">
  ];
}

#endif // MLIR_DIALECT_LINALG_PASSES
