// RUN: mlir-opt %s -convert-gpu-to-agxir='use-bare-ptr-memref-call-conv=1' \
// RUN:   -reconcile-unrealized-casts -o %t.mlir
// RUN: %run_agx --grid 32 --wg 32 %t.mlir f32:iota*256 \
// RUN:   f32:@%S/Inputs/identity16.txt f32*256 | FileCheck %s
// RUN: %run_agx --cpu agx-g13g --grid 32 --wg 32 %t.mlir f32:iota*256 \
// RUN:   f32:@%S/Inputs/identity16.txt f32*256 | FileCheck %s

// A 16x16x16 tile product on one SIMD-group: C = A * I = A, so C is 0..255
// in row-major order; the four fragments are stored with agx.matrix_store.
// CHECK: 2: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 {{.*}} 253 254 255{{$}}

gpu.module @kernels {
  gpu.func @tile(%a: memref<16x16xf32, 1>, %b: memref<16x16xf32, 1>,
                 %c: memref<16x16xf32, 1>) kernel {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.0> : vector<8xf32>
    %acc = agx.mma_tile %a[%c0, %c0] * %b[%c0, %c0] + %zero tile(16, 16, 16)
        : memref<16x16xf32, 1>, memref<16x16xf32, 1>, vector<8xf32>
    %f0 = vector.extract_strided_slice %acc {offsets = [0], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
    %f1 = vector.extract_strided_slice %acc {offsets = [2], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
    %f2 = vector.extract_strided_slice %acc {offsets = [4], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
    %f3 = vector.extract_strided_slice %acc {offsets = [6], sizes = [2], strides = [1]} : vector<8xf32> to vector<2xf32>
    agx.matrix_store %f0, %c[%c0, %c0] : vector<2xf32> -> memref<16x16xf32, 1>
    agx.matrix_store %f1, %c[%c0, %c8] : vector<2xf32> -> memref<16x16xf32, 1>
    agx.matrix_store %f2, %c[%c8, %c0] : vector<2xf32> -> memref<16x16xf32, 1>
    agx.matrix_store %f3, %c[%c8, %c8] : vector<2xf32> -> memref<16x16xf32, 1>
    gpu.return
  }
}
