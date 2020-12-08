// RUN: mlir-opt %s -convert-gpu-to-nvvm -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-nvvm='index-bitwidth=32' -split-input-file | FileCheck --check-prefix=CHECK32 %s

gpu.module @test_module {
  // CHECK-LABEL: func @gpu_index_ops()
  // CHECK32-LABEL: func @gpu_index_ops()
  func @gpu_index_ops()
      -> (index, index, index, index, index, index,
          index, index, index, index, index, index) {
    // CHECK32-NOT: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64

    // CHECK: = nvvm.read.ptx.sreg.tid.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %tIdX = "gpu.thread_id"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.tid.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %tIdY = "gpu.thread_id"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.tid.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %tIdZ = "gpu.thread_id"() {dimension = "z"} : () -> (index)

    // CHECK: = nvvm.read.ptx.sreg.ntid.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bDimX = "gpu.block_dim"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ntid.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bDimY = "gpu.block_dim"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ntid.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bDimZ = "gpu.block_dim"() {dimension = "z"} : () -> (index)

    // CHECK: = nvvm.read.ptx.sreg.ctaid.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bIdX = "gpu.block_id"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ctaid.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bIdY = "gpu.block_id"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.ctaid.z : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %bIdZ = "gpu.block_id"() {dimension = "z"} : () -> (index)

    // CHECK: = nvvm.read.ptx.sreg.nctaid.x : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %gDimX = "gpu.grid_dim"() {dimension = "x"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.nctaid.y : !llvm.i32
    // CHECK: = llvm.sext %{{.*}} : !llvm.i32 to !llvm.i64
    %gDimY = "gpu.grid_dim"() {dimension = "y"} : () -> (index)
    // CHECK: = nvvm.read.ptx.sreg.nctaid.z : !llvm.i32
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
    // CHECK: %[[#VALUE:]] = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
    %arg0 = constant 1.0 : f32
    // CHECK: %[[#OFFSET:]] = llvm.mlir.constant(4 : i32) : !llvm.i32
    %arg1 = constant 4 : i32
    // CHECK: %[[#WIDTH:]] = llvm.mlir.constant(23 : i32) : !llvm.i32
    %arg2 = constant 23 : i32
    // CHECK: %[[#ONE:]] = llvm.mlir.constant(1 : i32) : !llvm.i32
    // CHECK: %[[#SHL:]] = llvm.shl %[[#ONE]], %[[#WIDTH]] : !llvm.i32
    // CHECK: %[[#MASK:]] = llvm.sub %[[#SHL]], %[[#ONE]] : !llvm.i32
    // CHECK: %[[#CLAMP:]] = llvm.sub %[[#WIDTH]], %[[#ONE]] : !llvm.i32
    // CHECK: %[[#SHFL:]] = nvvm.shfl.sync.bfly %[[#MASK]], %[[#VALUE]], %[[#OFFSET]], %[[#CLAMP]] : !llvm.struct<(float, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][0 : index] : !llvm.struct<(float, i1)>
    // CHECK: llvm.extractvalue %[[#SHFL]][1 : index] : !llvm.struct<(float, i1)>
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
  // CHECK: llvm.func @__nv_fabsf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_fabs(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_fabs
  func @gpu_fabs(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.absf %arg_f32 : f32
    // CHECK: llvm.call @__nv_fabsf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.absf %arg_f64 : f64
    // CHECK: llvm.call @__nv_fabs(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_ceilf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_ceil(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_ceil
  func @gpu_ceil(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.ceilf %arg_f32 : f32
    // CHECK: llvm.call @__nv_ceilf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.ceilf %arg_f64 : f64
    // CHECK: llvm.call @__nv_ceil(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_floorf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_floor(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_floor
  func @gpu_floor(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.floorf %arg_f32 : f32
    // CHECK: llvm.call @__nv_floorf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.floorf %arg_f64 : f64
    // CHECK: llvm.call @__nv_floor(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_cosf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_cos(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_cos
  func @gpu_cos(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.cos %arg_f32 : f32
    // CHECK: llvm.call @__nv_cosf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.cos %arg_f64 : f64
    // CHECK: llvm.call @__nv_cos(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----
gpu.module @test_module {
  // CHECK: llvm.func @__nv_expf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_exp(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_exp
  func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.exp %arg_f32 : f32
    // CHECK: llvm.call @__nv_expf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.exp %arg_f64 : f64
    // CHECK: llvm.call @__nv_exp(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_logf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_log(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log
  func @gpu_log(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log %arg_f32 : f32
    // CHECK: llvm.call @__nv_logf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log %arg_f64 : f64
    // CHECK: llvm.call @__nv_log(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log10f(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_log10(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log10
  func @gpu_log10(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log10 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log10f(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log10 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log10(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_log2f(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_log2(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_log2
  func @gpu_log2(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.log2 %arg_f32 : f32
    // CHECK: llvm.call @__nv_log2f(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.log2 %arg_f64 : f64
    // CHECK: llvm.call @__nv_log2(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_sinf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_sin(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_sin
  func @gpu_sin(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
    %result32 = std.sin %arg_f32 : f32
    // CHECK: llvm.call @__nv_sinf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.sin %arg_f64 : f64
    // CHECK: llvm.call @__nv_sin(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result32, %result64 : f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_tanhf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_tanh(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_tanh
  func @gpu_tanh(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64) -> (f16, f32, f64) {
    %result16 = std.tanh %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : !llvm.half to !llvm.float
    // CHECK-NEXT: llvm.call @__nv_tanhf(%{{.*}}) : (!llvm.float) -> !llvm.float
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : !llvm.float to !llvm.half
    %result32 = std.tanh %arg_f32 : f32
    // CHECK: llvm.call @__nv_tanhf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.tanh %arg_f64 : f64
    // CHECK: llvm.call @__nv_tanh(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_rsqrtf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_rsqrt(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_rsqrt
  func @gpu_rsqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = std.rsqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : !llvm.half to !llvm.float
    // CHECK-NEXT: llvm.call @__nv_rsqrtf(%{{.*}}) : (!llvm.float) -> !llvm.float
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : !llvm.float to !llvm.half
    %result32 = std.rsqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_rsqrtf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.rsqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_rsqrt(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

gpu.module @test_module {
  // CHECK: llvm.func @__nv_sqrtf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_sqrt(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_sqrt
  func @gpu_sqrt(%arg_f16 : f16, %arg_f32 : f32, %arg_f64 : f64)
      -> (f16, f32, f64) {
    %result16 = std.sqrt %arg_f16 : f16
    // CHECK: llvm.fpext %{{.*}} : !llvm.half to !llvm.float
    // CHECK-NEXT: llvm.call @__nv_sqrtf(%{{.*}}) : (!llvm.float) -> !llvm.float
    // CHECK-NEXT: llvm.fptrunc %{{.*}} : !llvm.float to !llvm.half
    %result32 = std.sqrt %arg_f32 : f32
    // CHECK: llvm.call @__nv_sqrtf(%{{.*}}) : (!llvm.float) -> !llvm.float
    %result64 = std.sqrt %arg_f64 : f64
    // CHECK: llvm.call @__nv_sqrt(%{{.*}}) : (!llvm.double) -> !llvm.double
    std.return %result16, %result32, %result64 : f16, f32, f64
  }
}

// -----

// Test that we handled properly operation with SymbolTable other than module op
gpu.module @test_module {
  "test.symbol_scope"() ({
  // CHECK: test.symbol_scope
  // CHECK: llvm.func @__nv_expf(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @__nv_exp(!llvm.double) -> !llvm.double
  // CHECK-LABEL: func @gpu_exp
    func @gpu_exp(%arg_f32 : f32, %arg_f64 : f64) -> (f32, f64) {
      %result32 = std.exp %arg_f32 : f32
      // CHECK: llvm.call @__nv_expf(%{{.*}}) : (!llvm.float) -> !llvm.float
      %result64 = std.exp %arg_f64 : f64
      // CHECK: llvm.call @__nv_exp(%{{.*}}) : (!llvm.double) -> !llvm.double
      std.return %result32, %result64 : f32, f64
    }
    "test.finish" () : () -> ()
  }) : () -> ()
}

