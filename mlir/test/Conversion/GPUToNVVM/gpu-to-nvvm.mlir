// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK32-NOT: = llvm.sext %{{.*}} : i32 to i64

    // CHECK: = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.tid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.tid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: = nvvm.read.ptx.sreg.ntid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ntid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ntid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: = nvvm.read.ptx.sreg.ctaid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ctaid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ctaid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: = nvvm.read.ptx.sreg.nctaid.x : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.nctaid.y : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.nctaid.z : i32
    // CHECK: = llvm.sext %{{.*}} : i32 to i64
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
    // CHECK: = llvm.add %{{.*}}, %{{.*}} : i64
    // CHECK32: = llvm.add %{{.*}}, %{{.*}} : i32
    %0 = addi %idx, %idx : index
    // CHECK: llvm.return %{{.*}} : i64
    // CHECK32: llvm.return %{{.*}} : i32
    std.return %0 : index
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_all_reduce_op()
  gpu.func @gpu_all_reduce_op() {
    %arg0 = constant 1.0 : f32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: nvvm.shfl.sync.bfly
    // CHECK: nvvm.barrier0
    // CHECK: llvm.fadd
    %result = "gpu.all_reduce"(%arg0) ({}) {op = "add"} : (f32) -> (f32)

    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_all_reduce_region()
  gpu.func @gpu_all_reduce_region() {
    %arg0 = constant 1 : i32
    // TODO: Check full IR expansion once lowering has settled.
    // CHECK: nvvm.shfl.sync.bfly
    // CHECK: nvvm.barrier0
    %result = "gpu.all_reduce"(%arg0) ({
    ^bb(%lhs : i32, %rhs : i32):
      %xor = xor %lhs, %rhs : i32
      "gpu.yield"(%xor) : (i32) -> ()
    }) : (i32) -> (i32)
    gpu.return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_shuffle()
  func @gpu_shuffle() -> (f32) {
    // CHECK: %[[#VALUE:]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %arg0 = constant 1.0 : f32
    // CHECK: %[[#OFFSET:]] = llvm.mlir.constant(4 : i32) : i32
    %arg1 = constant 4 : i32
    // CHECK: %[[#WIDTH:]] = llvm.mlir.constant(23 : i32) : i32
    %arg2 = constant 23 : i32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[#SHL:]] = llvm.shl %[[#ONE]], %[[#WIDTH]] : i32
    // CHECK: %[[#MASK:]] = llvm.sub %[[#SHL]], %[[#ONE]] : i32
    // CHECK: %[[#CLAMP:]] = llvm.sub %[[#WIDTH]], %[[#ONE]] : i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync.bfly %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#CLAMP]] : !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][0 : index] : !llvm.struct<(f32, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][1 : index] : !llvm.struct<(f32, i1)>
    %shfl, %pred = "gpu.shuffle"(%arg0, %arg1, %arg2) { mode = "xor" } : (f32, i32, i32) -> (f32, i1)

    std.return %shfl : f32
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_sync()
  func @gpu_sync() {
    // CHECK: nvvm.barrier0
    gpu.barrier
    std.return
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_fabsf(f32) -> f32
  // CHECK: llvm.func @__nv_fabs(f64) -> f64
  // CHECK-LABEL: func @gpu_fabs
  func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.absf %arg_f32 : f32
    // CHECK: llvm.call @__nv_fabsf(%{{.*}}) : (f32) -> f32
    %result64 = std.absf %arg_f64 : f64
    // CHECK: llvm.call @__nv_fabs(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_ceilf(f32) -> f32
  // CHECK: llvm.func @__nv_ceil(f64) -> f64
  // CHECK-LABEL: func @gpu_ceil
  func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.ceilf %arg_f32 : f32
    // CHECK: llvm.call @__nv_ceilf(%{{.*}}) : (f32) -> f32
    %result64 = std.ceilf %arg_f64 : f64
    // CHECK: llvm.call @__nv_ceil(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_floorf(f32) -> f32
  // CHECK: llvm.func @__nv_floor(f64) -> f64
  // CHECK-LABEL: func @gpu_floor
  func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.floorf %arg_f32 : f32
    // CHECK: llvm.call @__nv_floorf(%{{.*}}) : (f32) -> f32
    %result64 = std.floorf %arg_f64 : f64
    // CHECK: llvm.call @__nv_floor(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_cosf(f32) -> f32
  // CHECK: llvm.func @__nv_cos(f64) -> f64
  // CHECK-LABEL: func @gpu_cos
  func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.cos %arg_f32 : f32
    // CHECK: llvm.call @__nv_cosf(%{{.*}}) : (f32) -> f32
    %result64 = math.cos %arg_f64 : f64
    // CHECK: llvm.call @__nv_cos(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__nv_expf(f32) -> f32
  // CHECK: llvm.func @__nv_exp(f64) -> f64
  // CHECK-LABEL: func @gpu_exp
  func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.exp %arg_f32 : f32
    // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
    %result64 = math.exp %arg_f64 : f64
    // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_logf(f32) -> f32
  // CHECK: llvm.func @__nv_log(f64) -> f64
  // CHECK-LABEL: func @gpu_log
  func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log %arg_f32 : f32
    // CHECK: llvm.call @__nv_logf(%{{.*}}) : (f32) -> f32
    %result64 = math.log %arg_f64 : f64
    // CHECK: llvm.call @__nv_log(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log10f(f32) -> f32
  // CHECK: llvm.func @__nv_log10(f64) -> f64
  // CHECK-LABEL: func @gpu_log10
  func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log10 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log10f(%{{.*}}) : (f32) -> f32
    %result64 = math.log10 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log10(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log1pf(f32) -> f32
  // CHECK: llvm.func @__nv_log1p(f64) -> f64
  // CHECK-LABEL: func @gpu_log1p
  func @gpu_log1p(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log1p %arg_f32 : f32
    // CHECK: llvm.call @__nv_log1pf(%{{.*}}) : (f32) -> f32
    %result64 = math.log1p %arg_f64 : f64
    // CHECK: llvm.call @__nv_log1p(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log2f(f32) -> f32
  // CHECK: llvm.func @__nv_log2(f64) -> f64
  // CHECK-LABEL: func @gpu_log2
  func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.log2 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log2f(%{{.*}}) : (f32) -> f32
    %result64 = math.log2 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log2(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_sinf(f32) -> f32
  // CHECK: llvm.func @__nv_sin(f64) -> f64
  // CHECK-LABEL: func @gpu_sin
  func @gpu_sin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.sin %arg_f32 : f32
    // CHECK: llvm.call @__nv_sinf(%{{.*}}) : (f32) -> f32
    %result64 = math.sin %arg_f64 : f64
    // CHECK: llvm.call @__nv_sin(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_tanhf(f32) -> f32
  // CHECK: llvm.func @__nv_tanh(f64) -> f64
  // CHECK-LABEL: func @gpu_tanh
  func @gpu_tanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = math.tanh %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_tanhf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.tanh %arg_f32 : f32
    // CHECK: llvm.call @__nv_tanhf(%{{.*}}) : (f32) -> f32
    %result64 = math.tanh %arg_f64 : f64
    // CHECK: llvm.call @__nv_tanh(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_rsqrtf(f32) -> f32
  // CHECK: llvm.func @__nv_rsqrt(f64) -> f64
  // CHECK-LABEL: func @gpu_rsqrt
  func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_rsqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_rsqrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_rsqrt(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_sqrtf(f32) -> f32
  // CHECK: llvm.func @__nv_sqrt(f64) -> f64
  // CHECK-LABEL: func @gpu_sqrt
  func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_sqrtf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_sqrtf(%{{.*}}) : (f32) -> f32
    %result64 = math.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_sqrt(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_atanf(f32) -> f32
  // CHECK: llvm.func @__nv_atan(f64) -> f64
  // CHECK-LABEL: func @gpu_atan
  func @gpu_atan(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = math.atan %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : f16 to f32
    // CHECK-NEXT: llvm.call @__nv_atanf(%{{.*}}) : (f32) -> f32
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : f32 to f16
    %result32 = math.atan %arg_f32 : f32
    // CHECK: llvm.call @__nv_atanf(%{{.*}}) : (f32) -> f32
    %result64 = math.atan %arg_f64 : f64
    // CHECK: llvm.call @__nv_atan(%{{.*}}) : (f64) -> f64
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_atan2f(f32, f32) -> f32
  // CHECK: llvm.func @__nv_atan2(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_atan2
  func @gpu_atan2(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
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
    std.return %result16, %result32, %result64 : f16, f32, f64
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
    func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %result32 = math.exp %arg_f32 : f32
      // CHECK: llvm.call @__nv_expf(%{{.*}}) : (f32) -> f32
      %result64 = math.exp %arg_f64 : f64
      // CHECK: llvm.call @__nv_exp(%{{.*}}) : (f64) -> f64
      std.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_expm1f(f32) -> f32
  // CHECK: llvm.func @__nv_expm1(f64) -> f64
  // CHECK-LABEL: func @gpu_expm1
  func @gpu_expm1(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.expm1 %arg_f32 : f32
    // CHECK: llvm.call @__nv_expm1f(%{{.*}}) : (f32) -> f32
    %result64 = math.expm1 %arg_f64 : f64
    // CHECK: llvm.call @__nv_expm1(%{{.*}}) : (f64) -> f64
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_powf(f32, f32) -> f32
  // CHECK: llvm.func @__nv_pow(f64, f64) -> f64
  // CHECK-LABEL: func @gpu_pow
  func @gpu_pow(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = math.powf %arg_f32, %arg_f32 : f32
    // CHECK: llvm.call @__nv_powf(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
    %result64 = math.powf %arg_f64, %arg_f64 : f64
    // CHECK: llvm.call @__nv_pow(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
    std.return %result32, %result64 : f32, f64
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
