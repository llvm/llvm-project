//===-- Passes.td - SCF pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_PASSES
#define MLIR_DIALECT_SCF_PASSES

include "mlir/Pass/PassBase.td"

def SCFBufferize : Pass<"scf-bufferize"> {
  let summary = "Bufferize the scf dialect.";
  let constructor = "mlir::createSCFBufferizePass()";
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "memref::MemRefDialect"];
}

// Note: Making these canonicalization patterns would require a dependency
// of the SCF dialect on the Affine/Tensor/MemRef dialects or vice versa.
def SCFForLoopCanonicalization
    : Pass<"scf-for-loop-canonicalization"> {
  let summary = "Canonicalize operations within scf.for loop bodies";
  let constructor = "mlir::createSCFForLoopCanonicalizationPass()";
  let dependentDialects = ["affine::AffineDialect", "tensor::TensorDialect",
                           "memref::MemRefDialect"];
}

def SCFForLoopPeeling : Pass<"scf-for-loop-peeling"> {
  let summary = "Peel `for` loops at their upper bounds.";
  let constructor = "mlir::createForLoopPeelingPass()";
  let options = [
    Option<"skipPartial", "skip-partial", "bool",
           /*default=*/"true",
           "Do not peel loops inside of the last, partial iteration of another "
           "already peeled loop.">
  ];
  let dependentDialects = ["affine::AffineDialect"];
}

def SCFForLoopSpecialization : Pass<"scf-for-loop-specialization"> {
  let summary = "Specialize `for` loops for vectorization";
  let constructor = "mlir::createForLoopSpecializationPass()";
}

def SCFParallelLoopFusion : Pass<"scf-parallel-loop-fusion"> {
  let summary = "Fuse adjacent parallel loops";
  let constructor = "mlir::createParallelLoopFusionPass()";
}

def TestSCFParallelLoopCollapsing : Pass<"test-scf-parallel-loop-collapsing"> {
  let summary = "Test parallel loops collapsing transformation";
  let constructor = "mlir::createTestSCFParallelLoopCollapsingPass()";
  let description = [{
      This pass is purely for testing the scf::collapseParallelLoops
      transformation. The transformation does not have opinions on how a
      parallel loop should be collapsed, so this pass is structured for the
      common case on GPUs of collapsing to a 3d parallel loop. 3 lists can be
      provided to collapsed-indices-{0,1,2} to represent how the loop should be
      collapsed and must reference evrey iterator in the original parallel loop.

    ```mlir
    # Before:
    scf.parallel (%arg0, %arg1)
                 = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
      "test.sink"(%5, %3) : (index, index) -> ()
      scf.yield
    }

    # After:
    scf.parallel (%arg0) = (%c0) to (%c4) step (%c1) {
      %0 = arith.remsi %arg0, %c2 : index
      %1 = arith.divsi %arg0, %c2 : index
      %2 = arith.muli %0, %c7 : index
      %3 = arith.addi %2, %c3 : index
      %4 = arith.muli %1, %c7 : index
      %5 = arith.addi %4, %c3 : index
      "test.sink"(%5, %3) : (index, index) -> ()
    }
    ```
  }];

  let options = [
    ListOption<"clCollapsedIndices0", "collapsed-indices-0", "unsigned",
               "Which loop indices to combine 0th loop index">,
    ListOption<"clCollapsedIndices1", "collapsed-indices-1", "unsigned",
               "Which loop indices to combine into the position 1 loop index">,
    ListOption<"clCollapsedIndices2", "collapsed-indices-2", "unsigned",
               "Which loop indices to combine into the position 2 loop index">,
  ];
}

def SCFParallelLoopSpecialization
    : Pass<"scf-parallel-loop-specialization"> {
  let summary = "Specialize parallel loops for vectorization";
  let constructor = "mlir::createParallelLoopSpecializationPass()";
}

def SCFParallelLoopTiling : Pass<"scf-parallel-loop-tiling"> {
  let summary = "Tile parallel loops";
  let constructor = "mlir::createParallelLoopTilingPass()";
  let options = [
    ListOption<"tileSizes", "parallel-loop-tile-sizes", "int64_t",
               "Factors to tile parallel loops by">,
    Option<"noMinMaxBounds", "no-min-max-bounds", "bool",
           /*default=*/"false",
           "Perform tiling with fixed upper bound with inbound check "
           "inside the internal loops">
  ];
  let dependentDialects = ["affine::AffineDialect"];
}

def SCFForLoopRangeFolding : Pass<"scf-for-loop-range-folding"> {
  let summary = "Fold add/mul ops into loop range";
  let constructor = "mlir::createForLoopRangeFoldingPass()";
}

def SCFForToWhileLoop : Pass<"scf-for-to-while"> {
  let summary = "Convert SCF for loops to SCF while loops";
  let constructor = "mlir::createForToWhileLoopPass()";
  let description = [{
    This pass transforms SCF.ForOp operations to SCF.WhileOp. The For loop
    condition is placed in the 'before' region of the while operation, and the
    induction variable incrementation and loop body in the 'after' region.
    The loop carried values of the while op are the induction variable (IV) of
    the for-loop + any iter_args specified for the for-loop.
    Any 'yield' ops in the for-loop are rewritten to additionally yield the
    (incremented) induction variable.

    ```mlir
    # Before:
      scf.for %i = %c0 to %arg1 step %c1 {
        %0 = arith.addi %arg2, %arg2 : i32
        memref.store %0, %arg0[%i] : memref<?xi32>
      }

    # After:
      %0 = scf.while (%i = %c0) : (index) -> index {
        %1 = arith.cmpi slt, %i, %arg1 : index
        scf.condition(%1) %i : index
      } do {
      ^bb0(%i: index):
        %1 = arith.addi %i, %c1 : index
        %2 = arith.addi %arg2, %arg2 : i32
        memref.store %2, %arg0[%i] : memref<?xi32>
        scf.yield %1 : index
      }
    ```
  }];
}

#endif // MLIR_DIALECT_SCF_PASSES
