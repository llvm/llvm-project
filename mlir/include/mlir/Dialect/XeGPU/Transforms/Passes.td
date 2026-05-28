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

def XeGPUSubgroupDistribute : Pass<"xegpu-subgroup-distribute"> {
  let summary = "Distribute XeGPU ops to work items";
  let description = [{
    The pass distributes subgroup level (SIMD) XeGPU ops to work items.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect", "index::IndexDialect"];
}

def XeGPUPropagateLayout : Pass<"xegpu-propagate-layout"> {
  let summary = "Propagate and assign XeGPU layout information";
  let description = [{
    This pass propagates the XeGPU layout information accross ops. Starting
    from a set of anchor operations (e.g. `dpas`, `store_nd`), this will
    propagate the layouts required for their operands to the producers. With
    this propagated layout information, pass will then update op result type
    with the layout information.

    `layout-kind` option values:
    - `inst`
      Propagate the `inst_data` field of the layout attribute. The default is chosen to
       maximize instruction-level granularity so that the user shape can be processed
       with the fewest instructions. For N-D operations, this granularity depends on
       W (width) and H (height) of the instruction shape.
       The B (block) dimension (or array length) is not included in the default
       configuration and must be enabled via a separate optimization pass.

    - `lane`
      Propagate the `lane_layout` and `lane_data` fields of the layout attribute.
      Default values are selected to align with hardware.

    - `subgroup`
      Propagate the `sg_layout` and `sg_data` fields of the layout attribute.
      Default values are selected to align with hardware.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect"];
  let options = [Option<
    "printOnly", "print-analysis-only", "bool",
    /*default=*/"false",
    "Print the result of layout propagation analysis and exit.">,
    Option<
    "layoutKind", "layout-kind", "std::string",
    /*default=*/"\"lane\"",
    "Propagate `subgroup` / `inst` / `lane` level of xegpu layouts.">,
    Option<
    "indexBitWidth", "index-bitwidth", "unsigned",
    /*default=*/"64",
    "Vectors of `index` type should also be distributable, inst-data and lower levels need to know the index size.">
  ];
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

def XeGPUPeepHoleOptimizer : Pass<"xegpu-optimize-peephole"> {
  let summary = "Optimize XeGPU block load operations";
  let description = [{
    This pass rewrites XeGPU loadNd operations into more optimal forms
    to improve performance. This includes,
    - Rewriting transpose B loads into more optimal forms to use HW block
      transpose instructions for better performance.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect"];
}

def XeGPUSgToWiDistributeExperimental : Pass<"xegpu-sg-to-wi-distribute-experimental"> {
  let summary = "Distribute XeGPU ops to work items";
  let description = [{
    The pass distributes subgroup level XeGPU ops to work item level XeGPU ops.
  }];
  let dependentDialects = ["memref::MemRefDialect", "xegpu::XeGPUDialect",
                           "vector::VectorDialect", "index::IndexDialect"];
}

def XeGPUCoalesceGatherScatter : Pass<"xegpu-coalesce-gather-scatter"> {
  let summary = "Coalesce neighbouring lanes of xegpu.load/store into chunked accesses";
  let description = [{
    Rewrites `xegpu.load` / `xegpu.store` ops whose offsets vector describes a
    contiguous-per-lane access into an equivalent op with a smaller offsets
    vector and a larger `chunk_size`. The transformation reduces the number of
    memory messages by having each lane fetch / write multiple contiguous
    elements.

    Two index patterns are recognized:
    - Affine offsets of the form `base + i * stride` (e.g. `vector.step` or a
      dense constant arithmetic progression). The offsets are coalescible with
      a factor `N` when `stride` divides `N` and the original `chunk_size`
      multiplied by `N` does not exceed the configured maximum.
    - All-equal offsets (e.g. `dense<0>`). All lanes load from the same
      address; this is rewritten as a broadcast: a single load of one element
      is materialized and the result is splatted via `vector.broadcast`.

    The mask must also be uniformly `true` (a `dense<true>` constant) for the
    coalesced lanes; partial-mask coalescing is not yet supported.

    This pass is intended to run before `xegpu-propagate-layout` so that the
    coalesced shape participates in `inst_data` selection.

    Pass options:
    - `max-chunk-size`: upper bound on the produced `chunk_size`. Defaults to
      8, matching typical Xe scatter chunk-size limits.

    Ops that already declare an explicit `chunk_size > 1` are left alone:
    a downstream pass has already committed to a per-lane chunked access,
    and the verifier-imposed shape relationship between value, offsets,
    and mask makes coalescing unsafe without a deeper rewrite.

    The subgroup size is read per-op from the enclosing `gpu.module`'s
    `xevm.target` chip (the same lookup as `xegpu-propagate-layout`). If no
    target chip is found, the pass falls back to a subgroup size of 16.
  }];
  let dependentDialects = [
      "arith::ArithDialect", "memref::MemRefDialect", "xegpu::XeGPUDialect",
      "vector::VectorDialect"];
  let options = [Option<
    "maxChunkSize", "max-chunk-size", "unsigned",
    /*default=*/"8",
    "Upper bound on the produced chunk_size when coalescing.">];
}


#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
