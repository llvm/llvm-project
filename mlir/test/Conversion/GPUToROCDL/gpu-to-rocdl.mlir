// RUN: mlir-opt %s -convert-gpu-to-rocdl -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  builtin.func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK32-NOT: = llvm.sext %{{.*}} : i32 to i64

    // CHECK: rocdl.workitem.id.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = gpu.thread_id x
    // CHECK: rocdl.workitem.id.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdY = gpu.thread_id y
    // CHECK: rocdl.workitem.id.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdZ = gpu.thread_id z

    // CHECK: rocdl.workgroup.dim.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimX = gpu.block_dim x
    // CHECK: rocdl.workgroup.dim.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimY = gpu.block_dim y
    // CHECK: rocdl.workgroup.dim.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimZ = gpu.block_dim z

    // CHECK: rocdl.workgroup.id.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdX = gpu.block_id x
    // CHECK: rocdl.workgroup.id.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdY = gpu.block_id y
    // CHECK: rocdl.workgroup.id.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdZ = gpu.block_id z

    // CHECK: rocdl.grid.dim.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimX = gpu.grid_dim x
    // CHECK: rocdl.grid.dim.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimY = gpu.grid_dim y
    // CHECK: rocdl.grid.dim.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimZ = gpu.grid_dim z

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
  builtin.func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i64
    // CHECK32: = llvm.add %{{.*}}, %{{.*}} : i32
    %0 = arith.addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i64
    // CHECK32: llvm.return %{{.*}} : i32
    std.return %0 : index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_sync()
  builtin.func @gpu_sync() {
    // CHECK: rocdl.barrier
    gpu.barrier
    std.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_fabs_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_fabs_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_fabs
  builtin.func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.abs %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fabs_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.abs %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fabs_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_ceil_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_ceil
  builtin.func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.ceil %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.ceil %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_floor_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_floor_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_floor
  builtin.func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.floor %arg_f32 : f32
    // CHECK: llvm.call @__ocml_floor_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.floor %arg_f64 : f64
    // CHECK: llvm.call @__ocml_floor_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cos_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cos_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_cos
  builtin.func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_exp
  builtin.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp_f32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp2_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_exp2
  builtin.func @gpu_exp2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp2_f32 = math.exp2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp2 %exp2_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp2_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  "test.symbol_scope"() ({
    // CHECK: test.symbol_scope
    // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
    // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
    // CHECK-LABEL: func @gpu_exp
    builtin.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %exp_f32 = math.exp %arg_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
      %result32 = math.exp %exp_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
      %result64 = math.exp %arg_f64 : f64
      // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
      std.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_expm1_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_expm1_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_expm1
  builtin.func @gpu_expm1(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %expm1_f32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.expm1 %expm1_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_expm1_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log
  builtin.func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log1p_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log1p_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log1p
  builtin.func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log1p_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log1p_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log10_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log10_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log10
  builtin.func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log2_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log2
  builtin.func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log2_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_rsqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_rsqrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_rsqrt
  builtin.func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_rsqrt_f64(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_sqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sqrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_sqrt
  builtin.func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sqrt_f64(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tanh_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_tanh
  builtin.func @gpu_tanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_atan_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_atan
  builtin.func @gpu_atan(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan_f64(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan2_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_atan2_f64(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_atan2
  builtin.func @gpu_atan2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan2_f32(%{{.*}}) : (f32, f32) -> f32
    %result64 = math.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan2_f64(%{{.*}}) : (f64, f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_pow_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_pow_f64(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_pow
  builtin.func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_pow_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_pow_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @kernel_func
  // CHECK: attributes
  // CHECK: gpu.kernel
  // CHECK: rocdl.kernel
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}
