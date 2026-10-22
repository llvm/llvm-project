// RUN: mlir-opt %s -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir f32:iota*32 f32*32 | FileCheck %s

// shuffle_up(a, 1) * 1 + 0.5 in segments of 4 lanes: lane t reads t - 1,
// except the first lane of each segment, which reads 0.
// CHECK: 1: 0.5 0.5 1.5 2.5 0.5 4.5 5.5 6.5 0.5 8.5 9.5 10.5 0.5 {{.*}} 0.5 28.5 29.5 30.5{{$}}

gpu.module @kernels {
  gpu.func @shift(%a: memref<32xf32, 1>, %out: memref<32xf32, 1>) kernel {
    %t = gpu.lane_id
    %x = memref.load %a[%t] : memref<32xf32, 1>
    %one = arith.constant 1.0 : f32
    %half = arith.constant 0.5 : f32
    %d = agx.fma_lane_shift %x, %one, %half shift 1 width 4 : f32
    memref.store %d, %out[%t] : memref<32xf32, 1>
    gpu.return
  }
}
