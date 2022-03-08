// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  func.func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK32-NOT: = llvm.sext %{{.*}} : i32 to i64

    // CHECK: = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = gpu.thread_id x
    // CHECK: = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdY = gpu.thread_id y
    // CHECK: = nvvm.read.ptx.sreg.tid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdZ = gpu.thread_id z

    // CHECK: = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimX = gpu.block_dim x
    // CHECK: = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimY = gpu.block_dim y
    // CHECK: = nvvm.read.ptx.sreg.ntid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimZ = gpu.block_dim z

    // CHECK: = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdX = gpu.block_id x
    // CHECK: = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdY = gpu.block_id y
    // CHECK: = nvvm.read.ptx.sreg.ctaid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdZ = gpu.block_id z

    // CHECK: = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimX = gpu.grid_dim x
    // CHECK: = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimY = gpu.grid_dim y
    // CHECK: = nvvm.read.ptx.sreg.nctaid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimZ = gpu.grid_dim z

    func.return %tIdX, %tIdY, %tIdZ, %bDimX, %bDimY, %bDimZ,
               %bIdX, %bIdY, %bIdZ, %gDimX, %gDimY, %gDimZ
        : index, index, index, index, index, index,
          index, index, index, index, index, index
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
  // CHECK-LABEL: func @gpu_all_reduce_op()
  gpu.func @gpu_all_reduce_op() {
    %arg0 = arith.constant 1.0 : f32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: nvvm.shfl.sync bfly {{.*}}
    // CHECK: nvvm.barrier0
    // CHECK: llvm.fadd
    %result = gpu.all_reduce add %arg0 {} : (f32) -> (f32)

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_all_reduce_region()
  gpu.func @gpu_all_reduce_region() {
    %arg0 = arith.constant 1 : i32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: nvvm.shfl.sync bfly {{.*}}
    // CHECK: nvvm.barrier0
    %result = gpu.all_reduce %arg0 {
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
  func.func @gpu_shuffle() -> (f32, f32, f32, f32) {
    // CHECK: %[[#VALUE:]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %arg0 = arith.constant 1.0 : f32
    // CHECK: %[[#OFFSET:]] = llvm.mlir.constant(4 : i32) : i32
    %arg1 = arith.constant 4 : i32
    // CHECK: %[[#WIDTH:]] = llvm.mlir.constant(23 : i32) : i32
    %arg2 = arith.constant 23 : i32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#MINUS_ONE:]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[#THIRTY_TWO:]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[#NUM_LANES:]] = llvm.sub %[[#THIRTY_TWO]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.lshr %[[#MINUS_ONE]], %[[#NUM_LANES]] : i32
    // CHECK: %[[#CLAMP:]] = llvm.sub %[[#WIDTH]], %[[#ONE]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync bfly %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#CLAMP]] {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][0 : index] : !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][1 : index] : !llvm.struct<(f32, i1)>
    %shfl, %pred = gpu.shuffle xor %arg0, %arg1, %arg2 : f32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#MINUS_ONE:]] = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %[[#THIRTY_TWO:]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[#NUM_LANES:]] = llvm.sub %[[#THIRTY_TWO]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.lshr %[[#MINUS_ONE]], %[[#NUM_LANES]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync up %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#NUM_LANES]] {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][0 : index] : !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][1 : index] : !llvm.struct<(f32, i1)>
    %shflu, %predu = gpu.shuffle up %arg0, %arg1, %arg2 : f32
    // CHECK: nvvm.shfl.sync down {{.*}} {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    %shfld, %predd = gpu.shuffle down %arg0, %arg1, %arg2 : f32
    // CHECK: nvvm.shfl.sync idx {{.*}} {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
    %shfli, %predi = gpu.shuffle idx %arg0, %arg1, %arg2 : f32

    func.return %shfl, %shflu, %shfld, %shfli : f32, f32,f32, f32
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_sync()
  func.func @gpu_sync() {
    // CHECK: nvvm.barrier0
    gpu.barrier
    func.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_fabsf(f32) -> f32
  // CHECK: llvm.func @__nv_fabs(f64) -> f64
  // CHECK-LABEL: func @gpu_fabs
  func.func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.abs %arg_f32 : f32
    // CHECK: llvm.call @__nv_fabsf(%{{.*}}) : (f32) -> f32
    %result64 = math.abs %arg_f64 : f64
    // CHECK: llvm.call @__nv_fabs(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_ceilf(f32) -> f32
  // CHECK: llvm.func @__nv_ceil(f64) -> f64
  // CHECK-LABEL: func @gpu_ceil
  func.func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.ceil %arg_f32 : f32
    // CHECK: llvm.call @__nv_ceilf(%{{.*}}) : (f32) -> f32
    %result64 = math.ceil %arg_f64 : f64
    // CHECK: llvm.call @__nv_ceil(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_floorf(f32) -> f32
  // CHECK: llvm.func @__nv_floor(f64) -> f64
  // CHECK-LABEL: func @gpu_floor
  func.func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.floor %arg_f32 : f32
    // CHECK: llvm.call @__nv_floorf(%{{.*}}) : (f32) -> f32
    %result64 = math.floor %arg_f64 : f64
    // CHECK: llvm.call @__nv_floor(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_cosf(f32) -> f32
  // CHECK: llvm.func @__nv_cos(f64) -> f64
  // CHECK-LABEL: func @gpu_cos
  func.func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__nv_cosf(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__nv_cos(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__nv_expf(f32) -> f32
  // CHECK: llvm.func @__nv_exp(f64) -> f64
  // CHECK-LABEL: func @gpu_exp
  func.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__nv_exp2f(f32) -> f32
  // CHECK: llvm.func @__nv_exp2(f64) -> f64
  // CHECK-LABEL: func @gpu_exp2
  func.func @gpu_exp2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.exp2 %arg_f32 : f32
    // CHECK: llvm.call @__nv_exp2f(%{{.*}}) : (f32) -> f32
    %result64 = math.exp2 %arg_f64 : f64
    // CHECK: llvm.call @__nv_exp2(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_logf(f32) -> f32
  // CHECK: llvm.func @__nv_log(f64) -> f64
  // CHECK-LABEL: func @gpu_log
  func.func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log %arg_f32 : f32
    // CHECK: llvm.call @__nv_logf(%{{.*}}) : (f32) -> f32
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__nv_log(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log10f(f32) -> f32
  // CHECK: llvm.func @__nv_log10(f64) -> f64
  // CHECK-LABEL: func @gpu_log10
  func.func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log10f(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log10(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log1pf(f32) -> f32
  // CHECK: llvm.func @__nv_log1p(f64) -> f64
  // CHECK-LABEL: func @gpu_log1p
  func.func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__nv_log1pf(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__nv_log1p(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log2f(f32) -> f32
  // CHECK: llvm.func @__nv_log2(f64) -> f64
  // CHECK-LABEL: func @gpu_log2
  func.func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log2f(%{{.*}}) : (f32) -> f32
    %result64 = math.log2 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log2(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_sinf(f32) -> f32
  // CHECK: llvm.func @__nv_sin(f64) -> f64
  // CHECK-LABEL: func @gpu_sin
  func.func @gpu_sin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sin %arg_f32 : f32
    // CHECK: llvm.call @__nv_sinf(%{{.*}}) : (f32) -> f32
    %result64 = math.sin %arg_f64 : f64
    // CHECK: llvm.call @__nv_sin(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_tanhf(f32) -> f32
  // CHECK: llvm.func @__nv_tanh(f64) -> f64
  // CHECK-LABEL: func @gpu_tanh
  func.func @gpu_tanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.tanh %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_tanhf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__nv_tanhf(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__nv_tanh(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_rsqrtf(f32) -> f32
  // CHECK: llvm.func @__nv_rsqrt(f64) -> f64
  // CHECK-LABEL: func @gpu_rsqrt
  func.func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_rsqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_rsqrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_rsqrt(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_sqrtf(f32) -> f32
  // CHECK: llvm.func @__nv_sqrt(f64) -> f64
  // CHECK-LABEL: func @gpu_sqrt
  func.func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_sqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_sqrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_sqrt(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_atanf(f32) -> f32
  // CHECK: llvm.func @__nv_atan(f64) -> f64
  // CHECK-LABEL: func @gpu_atan
  func.func @gpu_atan(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.atan %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_atanf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__nv_atanf(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__nv_atan(%{{.*}}) : (f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_atan2f(f32, f32) -> f32
  // CHECK: llvm.func @__nv_atan2(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_atan2
  func.func @gpu_atan2(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.atan2 %arg_f16, %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_atan2f(%{{.*}}) : (f32, f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.atan2 %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_atan2f(%{{.*}}) : (f32, f32) -> f32
    %result64 = math.atan2 %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_atan2(%{{.*}}) : (f64, f64) -> f64
    func.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  "test.symbol_scope"() ({
  // CHECK: test.symbol_scope
  // CHECK: llvm.func @__nv_expf(f32) -> f32
  // CHECK: llvm.func @__nv_exp(f64) -> f64
  // CHECK-LABEL: func @gpu_exp
    func.func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %result32 = math.exp %arg_f32 : f32
      // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
      %result64 = math.exp %arg_f64 : f64
      // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
      func.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_expm1f(f32) -> f32
  // CHECK: llvm.func @__nv_expm1(f64) -> f64
  // CHECK-LABEL: func @gpu_expm1
  func.func @gpu_expm1(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__nv_expm1f(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__nv_expm1(%{{.*}}) : (f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_powf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_pow(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_pow
  func.func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_powf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_pow(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    func.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @kernel_func
  // CHECK: attributes
  // CHECK: gpu.kernel
  // CHECK: nvvm.kernel
  gpu.func @kernel_func() kernel {
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: @async_cp(
  // CHECK-SAME: %[[IDX:[a-zA-Z0-9_]+]]: i64)
  gpu.func @async_cp(
    %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index) kernel {
    // CHECK-DAG: %[[BASEDST:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
    // CHECK-DAG: %[[S0:.*]] = llvm.mlir.constant(2048 : index) : i64
    // CHECK-DAG: %[[LI:.*]] = llvm.mul %[[IDX]], %[[S0]] : i64
    // CHECK-DAG: %[[S1:.*]] = llvm.mlir.constant(128 : index) : i64
    // CHECK-DAG: %[[FI0:.*]] = llvm.mul %[[IDX]], %[[S1]] : i64
    // CHECK-DAG: %[[FI1:.*]] = llvm.add %[[LI]], %[[FI0]] : i64
    // CHECK-DAG: %[[FI2:.*]] = llvm.add %[[FI1]], %[[IDX]] : i64
    // CHECK-DAG: %[[ADDRESSDST:.*]] = llvm.getelementptr %[[BASEDST]][%[[FI2]]] : (!llvm.ptr<f32, 3>, i64) -> !llvm.ptr<f32, 3>
    // CHECK-DAG: %[[CAST0:.*]] = llvm.bitcast %[[ADDRESSDST]] : !llvm.ptr<f32, 3> to !llvm.ptr<i8, 3>
    // CHECK-DAG: %[[BASESRC:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    // CHECK-DAG: %[[S3:.*]] = llvm.mlir.constant(128 : index) : i64
    // CHECK-DAG: %[[FI3:.*]] = llvm.mul %[[IDX]], %[[S3]]  : i64
    // CHECK-DAG: %[[FI4:.*]] = llvm.add %[[FI3]], %[[IDX]]  : i64
    // CHECK-DAG: %[[ADDRESSSRC:.*]] = llvm.getelementptr %[[BASESRC]][%[[FI4]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    // CHECK-DAG: %[[CAST1:.*]] = llvm.bitcast %[[ADDRESSSRC]] : !llvm.ptr<f32> to !llvm.ptr<i8>
    // CHECK-DAG: %[[CAST2:.*]] = llvm.addrspacecast %[[CAST1]] : !llvm.ptr<i8> to !llvm.ptr<i8, 1>
    // CHECK-DAG: nvvm.cp.async.shared.global %[[CAST0]], %[[CAST2]], 16
    %0 = gpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 4 : memref<128x128xf32> to memref<3x16x128xf32, 3>
    // CHECK: nvvm.cp.async.commit.group
    %1 = gpu.device_async_create_group %0
    // CHECK: nvvm.cp.async.wait.group 1
    gpu.device_async_wait %1 { numGroups = 1 : i32 }
    gpu.return
  }
}
