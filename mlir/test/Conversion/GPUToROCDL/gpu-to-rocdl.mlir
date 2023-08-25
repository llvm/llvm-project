// RUN: mlir-opt %s -convert-gpu-to-rocdl='use-opaque-pointers=1' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl='index-bitwidth=32 use-opaque-pointers=1' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  func.func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index,
          index) {
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

    // CHECK: = rocdl.mbcnt.lo %{{.*}}, %{{.*}} : (i32, i32) -> i32
    // CHECK: = rocdl.mbcnt.hi %{{.*}}, %{{.*}} : (i32, i32) -> i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %laneId = gpu.lane_id

    func.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ,
               %laneId
        : index, index, index, index, index, index,
          index, index, index, index, index, index,
          index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops_range()
  // CHECK-SAME: rocdl.flat_work_group_size = "1536,1536"
  // CHECK-SAME: rocdl.reqd_work_group_size = array<i32: 8, 12, 16>
  func.func @gpu_index_ops_range()
      -> (index, index, index, index, index, index) attributes
      {gpu.known_block_size = array<i32: 8, 12, 16>,
       gpu.known_grid_size = array<i32: 20, 24, 28>} {

    // CHECK: rocdl.workitem.id.x {range = array<i32: 0, 8>} : i32
    %tIdX = gpu.thread_id x
    // CHECK: rocdl.workitem.id.y {range = array<i32: 0, 12>} : i32
    %tIdY = gpu.thread_id y
    // CHECK: rocdl.workitem.id.z {range = array<i32: 0, 16>} : i32
    %tIdZ = gpu.thread_id z

    // CHECK: rocdl.workgroup.id.x {range = array<i32: 0, 20>} : i32
    %bIdX = gpu.block_id x
    // CHECK: rocdl.workgroup.id.y {range = array<i32: 0, 24>} : i32
    %bIdY = gpu.block_id y
    // CHECK: rocdl.workgroup.id.z {range = array<i32: 0, 28>} : i32
    %bIdZ = gpu.block_id z

    func.return %tIdX, %tIdY, %tIdZ, %bIdX, %bIdY, %bIdZ
        : index, index, index, index, index, index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_comp
  // CHECK32-LABEL: func @gpu_index_comp
  func.func @gpu_index_comp(%idx : index) -> index {
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i64
    // CHECK32: = llvm.add %{{.*}}, %{{.*}} : i32
    %0 = arith.addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i64
    // CHECK32: llvm.return %{{.*}} : i32
    func.return %0 : index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_sync()
  func.func @gpu_sync() {
    // CHECK: rocdl.barrier
    gpu.barrier
    func.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_fabs_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_fabs_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_fabs
  func.func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.absf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_fabs_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.absf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_fabs_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cbrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cbrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_cbrt
  func.func @gpu_cbrt(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cbrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cbrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cbrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cbrt_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_ceil_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_ceil_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_ceil
  func.func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.ceil %arg_f32 : f32
    // CHECK: llvm.call @__ocml_ceil_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.ceil %arg_f64 : f64
    // CHECK: llvm.call @__ocml_ceil_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_floor_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_floor_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_floor
  func.func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.floor %arg_f32 : f32
    // CHECK: llvm.call @__ocml_floor_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.floor %arg_f64 : f64
    // CHECK: llvm.call @__ocml_floor_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_cos_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_cos_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_cos
  func.func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__ocml_cos_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__ocml_cos_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_exp
  func.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp_f32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp %exp_f32 : f32
    // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_exp2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_exp2_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_exp2
  func.func @gpu_exp2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %exp2_f32 = math.exp2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.exp2 %exp2_f32 : f32
    // CHECK: llvm.call @__ocml_exp2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_exp2_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
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
    func.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %exp_f32 = math.exp %arg_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
      %result32 = math.exp %exp_f32 : f32
      // CHECK: llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
      %result64 = math.exp %arg_f64 : f64
      // CHECK: llvm.call @__ocml_exp_f64(%{{.*}}) : (f64) -> f64
      func.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_expm1_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_expm1_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_expm1
  func.func @gpu_expm1(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %expm1_f32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result32 = math.expm1 %expm1_f32 : f32
    // CHECK: llvm.call @__ocml_expm1_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_expm1_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log
  func.func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log1p_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log1p_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log1p
  func.func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log1p_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log1p_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log10_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log10_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log10
  func.func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log10_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log10_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_log2_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_log2_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_log2
  func.func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK: llvm.call @__ocml_log2_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.log2 %arg_f64 : f64
    // CHECK: llvm.call @__ocml_log2_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_rsqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_rsqrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_rsqrt
  func.func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_rsqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_rsqrt_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_sqrt_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_sqrt_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_sqrt
  func.func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__ocml_sqrt_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__ocml_sqrt_f64(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tan_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_tan
  func.func @gpu_tan(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.tan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tan_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.tan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tan_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_tanh_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_tanh_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_tanh
  func.func @gpu_tanh(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__ocml_tanh_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__ocml_tanh_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_atan_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_atan
  func.func @gpu_atan(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_atan2_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_atan2_f64(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_atan2
  func.func @gpu_atan2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_atan2_f32(%{{.*}}) : (f32, f32) -> f32
    %result64 = math.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_atan2_f64(%{{.*}}) : (f64, f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_pow_f32(f32, f32) -> f32
  // CHECK: llvm.func @__ocml_pow_f64(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_pow
  func.func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__ocml_pow_f32(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__ocml_pow_f64(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__ocml_erf_f32(f32) -> f32
  // CHECK: llvm.func @__ocml_erf_f64(f64) -> f64
  // CHECK-LABEL: func @gpu_erf
  func.func @gpu_erf(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.erf %arg_f32 : f32
    // CHECK: llvm.call @__ocml_erf_f32(%{{.*}}) : (f32) -> f32
    %result64 = math.erf %arg_f64 : f64
    // CHECK: llvm.call @__ocml_erf_f64(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_unroll
  func.func @gpu_unroll(%arg0 : vector<4xf32>) -> vector<4xf32> {
    %result = math.exp %arg0 : vector<4xf32>
    // CHECK: %[[V0:.+]] = llvm.mlir.undef : vector<4xf32>
    // CHECK: %[[CL:.+]] = llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V1:.+]] = llvm.insertelement %[[CL]], %[[V0]]
    // CHECK: %[[CL:.+]] = llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V2:.+]] = llvm.insertelement %[[CL]], %[[V1]]
    // CHECK: %[[CL:.+]] = llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V3:.+]] = llvm.insertelement %[[CL]], %[[V2]]
    // CHECK: %[[CL:.+]] = llvm.call @__ocml_exp_f32(%{{.*}}) : (f32) -> f32
    // CHECK: %[[V4:.+]] = llvm.insertelement %[[CL]], %[[V3]]
    // CHECK: return %[[V4]]
    func.return %result : vector<4xf32>
  }
}

// -----

// Test that the bf16 type is lowered away on this target.

gpu.module @test_module {
  // CHECK-LABEL: func @bf16_id
  func.func @bf16_id(%arg0 : bf16) -> bf16 {
    // CHECK-SAME: (%[[ARG0:.+]]: i16)
    // CHECK-SAME: -> i16
    // CHECK: return %[[ARG0]] : i16
    func.return %arg0 : bf16
  }

  // CHECK-LABEL: func @bf16x4_id
  func.func @bf16x4_id(%arg0 : vector<4xbf16>) -> vector<4xbf16> {
    // CHECK-SAME: (%[[ARG0:.+]]: vector<4xi16>)
    // CHECK-SAME: -> vector<4xi16>
    // CHECK: return %[[ARG0]] : vector<4xi16>
    func.return %arg0 : vector<4xbf16>
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

// -----

gpu.module @module {
// CHECK-LABEL: @spirv_exp
// CHECK: llvm.call @__ocml_exp_f32
  spirv.func @spirv_exp(%arg0: vector<4xf32>) -> vector<4xf32> "None" {
    %0 = math.exp %arg0 : vector<4xf32>
    spirv.ReturnValue %0 : vector<4xf32>
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_all_reduce_op()
  gpu.func @gpu_all_reduce_op() {
    %arg0 = arith.constant 1.0 : f32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: llvm.add
    // CHECK: llvm.and
    // CHECK: llvm.xor
    // CHECK: llvm.icmp "slt"
    // CHECK: llvm.select
    // CHECK: llvm.shl
    // CHECK: rocdl.ds_bpermute {{.*}}
    // CHECK: rocdl.barrier
    // CHECK: llvm.bitcast
    // CHECK: llvm.fadd
    %result = gpu.all_reduce add %arg0 uniform {} : (f32) -> (f32)

    gpu.return
  }
}


// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_all_reduce_region()
  gpu.func @gpu_all_reduce_region() {
    %arg0 = arith.constant 1 : i32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: llvm.add
    // CHECK: llvm.and
    // CHECK: llvm.xor
    // CHECK: llvm.icmp "slt"
    // CHECK: llvm.select
    // CHECK: llvm.shl
    // CHECK: rocdl.ds_bpermute {{.*}}
    // CHECK: rocdl.barrier
    %result = gpu.all_reduce %arg0 uniform {
    ^bb(%lhs : i32, %rhs : i32):
      %xor = arith.xori %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    } : (i32) -> (i32)
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_shuffle()
  func.func @gpu_shuffle() -> (f32, f32) {
    // CHECK: %[[#VALUE:]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %arg0 = arith.constant 1.0 : f32
    // CHECK: %[[#OFFSET:]] = llvm.mlir.constant(4 : i32) : i32
    %arg1 = arith.constant 4 : i32
    // CHECK: %[[#WIDTH:]] = llvm.mlir.constant(23 : i32) : i32
    %arg2 = arith.constant 23 : i32
    // CHECK: %[[#LANE_ID:]] = rocdl.mbcnt.hi
    // CHECK: %[[#ZERO:]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[#NEG_WIDTH:]] = llvm.sub %[[#ZERO]], %[[#WIDTH]] : i32
    // CHECK: %[[#ADD:]] = llvm.add %[[#LANE_ID]], %[[#WIDTH]] : i32
    // CHECK: %[[#WARP_OR_ZERO:]] = llvm.and %[[#ADD]], %[[#NEG_WIDTH]] : i32
    // CHECK: %[[#XOR:]] = llvm.xor %[[#LANE_ID]], %{{.*}} : i32
    // CHECK: %[[#CMP:]] = llvm.icmp "slt" %[[#XOR]], %[[#WARP_OR_ZERO]] : i32
    // CHECK: %[[#DST_LANE:]] = llvm.select %[[#CMP]], %[[#XOR]], %{{.*}} : i1, i32
    // CHECK: %[[#TWO:]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[#ALIGNED_DST_LANE:]] = llvm.shl %[[#DST_LANE]], %[[#TWO]] : i32
    // CHECK: %[[#CAST_VALUE:]] = llvm.bitcast %[[#VALUE]] : f32 to i32
    // CHECK: %[[#PERMUTE:]] = rocdl.ds_bpermute %[[#ALIGNED_DST_LANE]], %[[#CAST_VALUE]] : (i32, i32) -> i32
    // CHECK: %[[#CAST_SHFL_VALUE:]] = llvm.bitcast %[[#PERMUTE]] : i32 to f32
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    // CHECK: %[[#LANE_ID:]] = rocdl.mbcnt.hi
    // CHECK: %[[#ZERO:]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %[[#NEG_WIDTH:]] = llvm.sub %[[#ZERO]], %[[#WIDTH]] : i32
    // CHECK: %[[#ADD:]] = llvm.add %[[#LANE_ID]], %[[#WIDTH]] : i32
    // CHECK: %[[#WARP_OR_ZERO:]] = llvm.and %[[#ADD]], %[[#NEG_WIDTH]] : i32
    // CHECK: %[[#CMP:]] = llvm.icmp "slt" %[[#OFFSET]], %[[#WARP_OR_ZERO]] : i32
    // CHECK: %[[#DST_LANE:]] = llvm.select %[[#CMP]], %[[#OFFSET]], %{{.*}} : i1, i32
    // CHECK: %[[#TWO:]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %[[#ALIGNED_DST_LANE:]] = llvm.shl %[[#DST_LANE]], %[[#TWO]] : i32
    // CHECK: %[[#CAST_VALUE:]] = llvm.bitcast %[[#VALUE]] : f32 to i32
    // CHECK: %[[#PERMUTE:]] = rocdl.ds_bpermute %[[#ALIGNED_DST_LANE]], %[[#CAST_VALUE]] : (i32, i32) -> i32
    // CHECK: %[[#CAST_SHFL_VALUE:]] = llvm.bitcast %[[#PERMUTE]] : i32 to f32
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32
    func.return %shfl, %shfli : f32, f32
  }
}