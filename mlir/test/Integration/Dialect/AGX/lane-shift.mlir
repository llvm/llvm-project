// RUN: mlir-opt %s -convert-gpu-to-air='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir f32:iota*32 f32*32 | FileCheck %s

// shuffle_up(a, 1) * 1 + 0.5 in segments of 4 lanes: lane t reads t - 1,
// except the first lane of each segment, which reads 0 (a gpu.shuffle whose
// source is outside the segment, then fma).
// CHECK: 1: 0.5 0.5 1.5 2.5 0.5 4.5 5.5 6.5 0.5 8.5 9.5 10.5 0.5 {{.*}} 0.5 28.5 29.5 30.5{{$}}

gpu.module @kernels {
  gpu.func @shift(%a: memref<32xf32, 1>, %out: memref<32xf32, 1>) kernel {
    %t = gpu.lane_id
    %x = memref.load %a[%t] : memref<32xf32, 1>
    %one = arith.constant 1.0 : f32
    %half = arith.constant 0.5 : f32
    %zero = arith.constant 0.0 : f32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %s, %valid = gpu.shuffle up %x, %c1, %c4 : f32
    %up = arith.select %valid, %s, %zero : f32
    %d = math.fma %up, %one, %half : f32
    memref.store %d, %out[%t] : memref<32xf32, 1>
    gpu.return
  }
}
