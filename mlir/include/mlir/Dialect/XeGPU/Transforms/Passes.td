//===-- Passes.td - XeGPU transformation definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def XeGPUFoldAliasOps : Pass<"xegpu-fold-alias-ops"> {
  let summary = "Fold alias ops into XeGPU ops";
  let description = [{
    The pass folds aliasing ops into XeGPU ops that they operate on the original
    source references.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect"];
}

def XeGPUSubgroupDistribute : Pass<"xegpu-subgroup-distribute"> {
  let summary = "Distribute XeGPU ops to work items";
  let description = [{
    The pass distributes subgroup level (SIMD) XeGPU ops to work items.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect"];
}

def XeGPUPropagateLayout : Pass<"xegpu-propagate-layout"> {
  let summary = "Propagate and assign XeGPU layout information";
  let description = [{
    This pass propagates the XeGPU layout information accross ops. Starting
    from a set of anchor operations (e.g. `dpas`, `store_nd`), this will
    propagate the layouts required for their operands to the producers. With
    this propagated layout information, pass will then update op result type
    with the layout information.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect"];
  let options = [Option<
    "printOnly", "print-analysis-only", "bool",
    /*default=*/"false",
    "Print the result of layout propagation analysis and exit.">];
}

def XeGPUWgToSgDistribute : Pass<"xegpu-wg-to-sg-distribute"> {
  let summary = "Transform WorkGroup level XeGPU code to SubGroup level";
  let description = [{
    This transform pass distributes the workgroup level computation to
    multiple subgroups based on the sg_layout and sg_data attributes.
  }];

  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect", "arith::ArithDialect",
                           "gpu::GPUDialect", "index::IndexDialect"];
}

def XeGPUBlocking: Pass<"xegpu-blocking"> {
  let summary = "Block XeGPU ops into smaller size.";
  let description = [{
    This pass partitions operations that process large shapes into multiple
    operations on smaller shapes, as specified by the inst_data in the layout
    attribute. This enables each resulting operation to be efficiently mapped
    to a hardware instruction.
  }];
  let dependentDialects = [
      "memref::MemRefDialect", "xegpu::XeGPUDialect", "vector::VectorDialect",
      "index::IndexDialect"];
}

def XeGPUVectorLinearize : Pass<"xegpu-vector-linearize"> {
  let summary = "Linearize n-D vectors to 1-D vectors";
  let description = [{
    This pass linearizes n-D vectors to 1-D vectors for lowering to XeVM.
  }];
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                           "scf::SCFDialect", "ub::UBDialect", "vector::VectorDialect"];
}

#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
