// RUN: mlir-opt %s -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 64 --wg 64 %t.mlir f32:iota*64 f32*64 | FileCheck %s

// Each thread scales its element and the SIMD-groups sum them: the two
// SIMD-groups of 32 threads get 2 * (0 + ... + 31) = 992 and
// 2 * (32 + ... + 63) = 3040.
// CHECK: 1: 992 992 {{.*}} 992 3040 {{.*}} 3040{{$}}

gpu.module @kernels {
  gpu.func @scale(%a: memref<64xf32, 1>, %out: memref<64xf32, 1>) kernel {
    %i = gpu.thread_id x
    %v = memref.load %a[%i] : memref<64xf32, 1>
    %c2 = arith.constant 2.0 : f32
    %w = arith.mulf %v, %c2 : f32
    %r = gpu.subgroup_reduce add %w : (f32) -> f32
    memref.store %r, %out[%i] : memref<64xf32, 1>
    gpu.return
  }
}
