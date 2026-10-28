// RUN: mlir-opt %s -convert-gpu-to-air='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir f32:iota*256 \
// RUN:   f32:@%S/Inputs/identity16.txt f32*256 | FileCheck %s
// RUN: %run_agx --cpu agx-g13g --grid 32 --wg 32 %t.mlir f32:iota*256 \
// RUN:   f32:@%S/Inputs/identity16.txt f32*256 | FileCheck %s

// A 16x16x16 tile product on one SIMD-group, as 2 x 2 8x8 matrices along two
// steps of K: C = A * I = A, so C is 0..255 in row-major order.
// CHECK: 2: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 {{.*}} 253 254 255{{$}}

gpu.module @kernels {
  gpu.func @tile(%a: memref<16x16xf32, 1>, %b: memref<16x16xf32, 1>,
                 %c: memref<16x16xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %fz = arith.constant 0.0 : f32
    %zero = air.simdgroup_matrix_fill %fz : vector<64xf32>
    %acc:4 = scf.for %k = %c0 to %c16 step %c8
        iter_args(%a00 = %zero, %a01 = %zero, %a10 = %zero, %a11 = %zero)
        -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>) {
      %m0 = air.simdgroup_matrix_load %a[%c0, %k] : memref<16x16xf32, 1> -> vector<64xf32>
      %m1 = air.simdgroup_matrix_load %a[%c8, %k] : memref<16x16xf32, 1> -> vector<64xf32>
      %n0 = air.simdgroup_matrix_load %b[%k, %c0] : memref<16x16xf32, 1> -> vector<64xf32>
      %n1 = air.simdgroup_matrix_load %b[%k, %c8] : memref<16x16xf32, 1> -> vector<64xf32>
      %d00 = air.simdgroup_matrix_multiply_accumulate %m0, %n0, %a00 : vector<64xf32>
      %d01 = air.simdgroup_matrix_multiply_accumulate %m0, %n1, %a01 : vector<64xf32>
      %d10 = air.simdgroup_matrix_multiply_accumulate %m1, %n0, %a10 : vector<64xf32>
      %d11 = air.simdgroup_matrix_multiply_accumulate %m1, %n1, %a11 : vector<64xf32>
      scf.yield %d00, %d01, %d10, %d11
          : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>
    }
    air.simdgroup_matrix_store %acc#0, %c[%c0, %c0] : vector<64xf32> -> memref<16x16xf32, 1>
    air.simdgroup_matrix_store %acc#1, %c[%c0, %c8] : vector<64xf32> -> memref<16x16xf32, 1>
    air.simdgroup_matrix_store %acc#2, %c[%c8, %c0] : vector<64xf32> -> memref<16x16xf32, 1>
    air.simdgroup_matrix_store %acc#3, %c[%c8, %c8] : vector<64xf32> -> memref<16x16xf32, 1>
    gpu.return
  }
}
