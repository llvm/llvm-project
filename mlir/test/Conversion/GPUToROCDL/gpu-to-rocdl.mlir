// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK32-NOT: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64

    // CHECK: rocdl.workitem.id.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workitem.id.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workitem.id.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.dim.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.dim.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.workgroup.id.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.workgroup.id.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.workgroup.id.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: rocdl.grid.dim.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: rocdl.grid.dim.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: rocdl.grid.dim.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %gDimZ = "gpu.grid_dim"() {dimension = "z"} : () -> (index)

    std.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ
        : index, index, index, index, index, index,
          index, index, index, index, index, index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_comp
  // CHECK32-LABEL: func @gpu_index_comp
  func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : !llvm.i64
    // CHECK32: = llvm.add %{{.*}}, %{{.*}} : !llvm.i32
    %0 = addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : !llvm.i64
    // CHECK32: llvm.return %{{.*}} : !llvm.i32
    std.return %0 : index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_sync()
  func @gpu_sync() {
    // CHECK: rocdl.barrier
    gpu.barrier
    std.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_fabs_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_fabs_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_fabs
  func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.absf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fabs_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.absf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fabs_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_ceil_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_ceil
  func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.ceilf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.ceilf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cos_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_cos_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_cos
  func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_exp_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_exp
  func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp_f32 = std.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result32 = std.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}


// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  "test.symbol_scope"() ({
    // CHECK: test.symbol_scope
    // CHECK: llvm.func @__ocml_exp_f32(!llvm.float) -> !llvm.float
    // CHECK: llvm.func @__ocml_exp_f64(!llvm.double) -> !llvm.double
    // CHECK-LABEL: func @gpu_exp
    func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %exp_f32 = std.exp %arg_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result32 = std.exp %exp_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result64 = std.exp %arg_f64 : f64
      // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
      std.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_log_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log
  func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log10_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_log10_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log10
  func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log2_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_log2_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log2
  func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log2_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log2_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__ocml_tanh_f64(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_tanh
  func @gpu_tanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}
