// RUN: mlir-opt %s -convert-gpu-to-nvvm='index-bitwidth=32 use-opaque-pointers=1' -split-input-file | FileCheck %s

// RUN: mlir-opt %s -test-transform-dialect-interpreter | FileCheck %s

gpu.module @test_module_0 {
  // CHECK-LABEL: func @gpu_index_ops()
  func.func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index,
          index) {
    %tIdX = gpu.thread_id x
    %tIdY = gpu.thread_id y
    %tIdZ = gpu.thread_id z

    %bDimX = gpu.block_dim x
    %bDimY = gpu.block_dim y
    %bDimZ = gpu.block_dim z

    %bIdX = gpu.block_id x
    %bIdY = gpu.block_id y
    %bIdZ = gpu.block_id z

    %gDimX = gpu.grid_dim x
    %gDimY = gpu.grid_dim y
    %gDimZ = gpu.grid_dim z

    // CHECK-NOT: = llvm.sext %{{.*}} : i32 to i64
    %laneId = gpu.lane_id

    func.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ,
               %laneId
        : index, index, index, index, index, index,
          index, index, index, index, index, index,
          index
  }
}



gpu.module @test_module_1 {
  // CHECK-LABEL: func @gpu_index_comp
  func.func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i32
    %0 = arith.addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i32
    func.return %0 : index
  }
}

transform.sequence failures(propagate) {
^bb1(%toplevel_module: !transform.any_op):
  %gpu_module = transform.structured.match ops{["gpu.module"]} in %toplevel_module
    : (!transform.any_op) -> !transform.any_op
  transform.apply_conversion_patterns to %gpu_module {
    transform.apply_conversion_patterns.dialect_to_llvm "arith"
    transform.apply_conversion_patterns.dialect_to_llvm "cf"
    transform.apply_conversion_patterns.vector.vector_to_llvm
    transform.apply_conversion_patterns.func.func_to_llvm
    transform.apply_conversion_patterns.dialect_to_llvm "memref"
    transform.apply_conversion_patterns.gpu.gpu_to_nvvm
    transform.apply_conversion_patterns.gpu.gpu_wmma_to_nvvm
    transform.apply_conversion_patterns.gpu.gpu_subgroup_reduce_to_nvvm {has_redux = true}
    transform.apply_conversion_patterns.nvgpu.nvgpu_to_nvvm
  } with type_converter {
    transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
      {index_bitwidth = 32, use_opaque_pointers = true}
  } {
    legal_dialects = ["llvm", "memref", "nvvm"],
    legal_ops = ["func.func", "gpu.module", "gpu.module_end", "gpu.yield"],
    illegal_dialects = ["gpu"],
    illegal_ops = ["llvm.cos", "llvm.exp", "llvm.exp2", "llvm.fabs", "llvm.fceil",
                   "llvm.ffloor", "llvm.log", "llvm.log10", "llvm.log2", "llvm.pow",
                   "llvm.sin", "llvm.sqrt"],
    partial_conversion
  } : !transform.any_op
}
