// RUN: mlir-opt %s -test-lower-to-llvm  | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_c_runner_utils | \
// RUN: FileCheck %s

func.func @transfer_write16_inbounds_1d(%A : memref<?xf32>, %base: index) {
  %f = arith.constant 16.0 : f32
  %v = vector.splat %f : vector<16xf32>
  vector.transfer_write %v, %A[%base]
    {permutation_map = affine_map<(d0) -> (d0)>, in_bounds = [true]}
    : vector<16xf32>, memref<?xf32>
  return
}

func.func @transfer_write13_1d(%A : memref<?xf32>, %base: index) {
  %f = arith.constant 13.0 : f32
  %v = vector.splat %f : vector<13xf32>
  vector.transfer_write %v, %A[%base]
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<13xf32>, memref<?xf32>
  return
}

func.func @transfer_write17_1d(%A : memref<?xf32>, %base: index) {
  %f = arith.constant 17.0 : f32
  %v = vector.splat %f : vector<17xf32>
  vector.transfer_write %v, %A[%base]
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<17xf32>, memref<?xf32>
  return
}

func.func @transfer_read_1d(%A : memref<?xf32>) -> vector<32xf32> {
  %z = arith.constant 0: index
  %f = arith.constant 0.0: f32
  %r = vector.transfer_read %A[%z], %f
    {permutation_map = affine_map<(d0) -> (d0)>}
    : memref<?xf32>, vector<32xf32>
  return %r : vector<32xf32>
}

func.func @transfer_write_inbounds_3d(%A : memref<4x4x4xf32>) {
  %c0 = arith.constant 0: index
  %f = arith.constant 0.0 : f32
  %v0 = vector.splat %f : vector<2x3x4xf32>
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32
  %f5 = arith.constant 5.0 : f32
  %f6 = arith.constant 6.0 : f32
  %f7 = arith.constant 7.0 : f32
  %f8 = arith.constant 8.0 : f32

  %v1 = vector.insert %f1, %v0[0, 0, 0] : f32 into vector<2x3x4xf32>
  %v2 = vector.insert %f2, %v1[0, 0, 3] : f32 into vector<2x3x4xf32>
  %v3 = vector.insert %f3, %v2[0, 2, 0] : f32 into vector<2x3x4xf32>
  %v4 = vector.insert %f4, %v3[0, 2, 3] : f32 into vector<2x3x4xf32>
  %v5 = vector.insert %f5, %v4[1, 0, 0] : f32 into vector<2x3x4xf32>
  %v6 = vector.insert %f6, %v5[1, 0, 3] : f32 into vector<2x3x4xf32>
  %v7 = vector.insert %f7, %v6[1, 2, 0] : f32 into vector<2x3x4xf32>
  %v8 = vector.insert %f8, %v7[1, 2, 3] : f32 into vector<2x3x4xf32>
  vector.transfer_write %v8, %A[%c0, %c0, %c0]
    {permutation_map = affine_map<(d0, d1, d2) -> (d2, d0, d1)>,
    in_bounds = [true, true, true]}
    : vector<2x3x4xf32>, memref<4x4x4xf32>
  return
}

func.func @entry() {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c32 = arith.constant 32: index
  %A = memref.alloc(%c32) {alignment=64} : memref<?xf32>
  scf.for %i = %c0 to %c32 step %c1 {
    %f = arith.constant 0.0: f32
    memref.store %f, %A[%i] : memref<?xf32>
  }

  // On input, memory contains all zeros.
  %0 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %0 : vector<32xf32>

  // Overwrite with 16 values of 16 at base 3.
  // Statically guaranteed to be in-bounds. Exercises proper alignment.
  %c3 = arith.constant 3: index
  call @transfer_write16_inbounds_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  %1 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %1 : vector<32xf32>

  // Overwrite with 13 values of 13 at base 3.
  call @transfer_write13_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  %2 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %2 : vector<32xf32>

  // Overwrite with 17 values of 17 at base 7.
  %c7 = arith.constant 7: index
  call @transfer_write17_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  %3 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %3 : vector<32xf32>

  // Overwrite with 13 values of 13 at base 8.
  %c8 = arith.constant 8: index
  call @transfer_write13_1d(%A, %c8) : (memref<?xf32>, index) -> ()
  %4 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %4 : vector<32xf32>

  // Overwrite with 17 values of 17 at base 14.
  %c14 = arith.constant 14: index
  call @transfer_write17_1d(%A, %c14) : (memref<?xf32>, index) -> ()
  %5 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %5 : vector<32xf32>

  // Overwrite with 13 values of 13 at base 19.
  %c19 = arith.constant 19: index
  call @transfer_write13_1d(%A, %c19) : (memref<?xf32>, index) -> ()
  %6 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %6 : vector<32xf32>

  memref.dealloc %A : memref<?xf32>

  // 3D case
  %c4 = arith.constant 4: index
  %A1 = memref.alloc() {alignment=64} : memref<4x4x4xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        %f = arith.constant 0.0: f32
        memref.store %f, %A1[%i, %j, %k] : memref<4x4x4xf32>
      }
    }
  }
  call @transfer_write_inbounds_3d(%A1) : (memref<4x4x4xf32>) -> ()
  %f = arith.constant 0.0: f32
  %r = vector.transfer_read %A1[%c0, %c0, %c0], %f
    : memref<4x4x4xf32>, vector<4x4x4xf32>
  vector.print %r : vector<4x4x4xf32>

  memref.dealloc %A1 : memref<4x4x4xf32>

  return
}

// CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13 )

// 3D case.
// CHECK: ( ( ( 1, 5, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 2, 6, 0, 0 ) ), ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) ),
// CHECK-SAME: ( ( 3, 7, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 4, 8, 0, 0 ) ), ( ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ), ( 0, 0, 0, 0 ) ) )
