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
  let dependentDialects = ["memref::MemRefDialect", "vector::VectorDialect"];
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

  let dependentDialects = ["memref::MemRefDialect", "vector::VectorDialect",
                           "arith::ArithDialect", "gpu::GPUDialect",
                           "index::IndexDialect"];
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
      "memref::MemRefDialect", "vector::VectorDialect",
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

def XeGPULegalizeVectorWidth : Pass<"xegpu-legalize-vector-width"> {
  let summary = "Split elementwise vector ops that exceed the target vector width";
  let description = [{
    Unrolls elementwise operations whose vectors have more components than the
    target can handle into a sequence of narrower operations, glued together
    with `vector.extract_strided_slice` / `vector.insert_strided_slice`.

    Only operations with `ElementwiseMappable` traits and a single result are
    considered. Data movement operations (`vector.shuffle`, length-changing
    `vector.bitcast`, `xegpu.load_nd`, `xegpu.store_nd`, `xegpu.dpas_mx`, ...)
    are deliberately left alone: their vectors are *packed payloads* whose
    component count is not a compute width. For example a
    `vector<64xf4E2M1FN>` DPAS operand is only 256 bits and becomes a
    `vector<8xi32>` once lowered, so splitting it would be a pessimization.
    This distinction is exactly what the `ElementwiseMappable` traits express,
    so no explicit op list is needed.

    Note that a bit-width based threshold would not work here: a
    `vector<32xbf16>` is exactly 512 bits yet is illegal as a compute operand,
    while the 256-bit `vector<64xf4E2M1FN>` payload above must be preserved.
    The limit is therefore expressed in vector components.

    This runs after `arith-expand-ops`, which is what materializes the wide
    arithmetic (for example the `f8E8M0FNU` scale expansion), and before the
    XeVM conversions.
  }];
  let options = [
    Option<"maxVectorComponents", "max-vector-components", "unsigned",
           /*default=*/"0",
           "Maximum number of components in a vector operand of an elementwise "
           "op. A value of 0 selects the target default.">
  ];
  let dependentDialects = ["arith::ArithDialect", "vector::VectorDialect"];
}

def XeGPUPeepHoleOptimizer : Pass<"xegpu-optimize-peephole"> {
  let summary = "Optimize XeGPU block load operations";
  let description = [{
    This pass rewrites XeGPU loadNd operations into more optimal forms
    to improve performance. This includes,
    - Rewriting transpose B loads into more optimal forms to use HW block
      transpose instructions for better performance.
  }];
  let dependentDialects = ["memref::MemRefDialect",
                           "vector::VectorDialect"];
}

def XeGPUSgToLaneDistribute : Pass<"xegpu-sg-to-lane-distribute"> {
  let summary = "Distribute XeGPU ops to lanes";
  let description = [{
    The pass distributes subgroup level XeGPU ops to lane level XeGPU ops.
  }];
  let dependentDialects = ["memref::MemRefDialect", "vector::VectorDialect",
                           "index::IndexDialect"];
}


#endif // MLIR_DIALECT_XEGPU_TRANSFORMS_PASSES_TD
