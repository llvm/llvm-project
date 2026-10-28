// RUN: mlir-opt %s -convert-gpu-to-air='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir f32:@%S/Inputs/identity16.txt \
// RUN:   f32:iota*64 f32*64 | FileCheck %s

// I * B + B = 2 B on 8x8 SIMD-group matrices: I is the top left of the 16x16
// identity, B is 0..63 in row-major order.
// CHECK: 2: 0 2 4 6 8 10 {{.*}} 122 124 126{{$}}

gpu.module @kernels {
  gpu.func @mma(%a: memref<16x16xf32, 1>, %b: memref<8x8xf32, 1>,
                %out: memref<8x8xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %x = air.simdgroup_matrix_load %a[%c0, %c0] : memref<16x16xf32, 1> -> vector<64xf32>
    %y = air.simdgroup_matrix_load %b[%c0, %c0] : memref<8x8xf32, 1> -> vector<64xf32>
    %d = air.simdgroup_matrix_multiply_accumulate %x, %y, %y : vector<64xf32>
    air.simdgroup_matrix_store %d, %out[%c0, %c0] : vector<64xf32> -> memref<8x8xf32, 1>
    gpu.return
  }
}
