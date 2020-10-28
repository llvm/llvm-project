// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @transfer_write16_unmasked_1d(%A : memref<?xf32>, %base: index) {
  %f = constant 16.0 : f32
  %v = splat %f : vector<16xf32>
  vector.transfer_write %v, %A[%base]
    {permutation_map = affine_map<(d0) -> (d0)>, masked = [false]}
    : vector<16xf32>, memref<?xf32>
  return
}

func @transfer_write13_1d(%A : memref<?xf32>, %base: index) {
  %f = constant 13.0 : f32
  %v = splat %f : vector<13xf32>
  vector.transfer_write %v, %A[%base]
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<13xf32>, memref<?xf32>
  return
}

func @transfer_write17_1d(%A : memref<?xf32>, %base: index) {
  %f = constant 17.0 : f32
  %v = splat %f : vector<17xf32>
  vector.transfer_write %v, %A[%base]
    {permutation_map = affine_map<(d0) -> (d0)>}
    : vector<17xf32>, memref<?xf32>
  return
}

func @transfer_read_1d(%A : memref<?xf32>) -> vector<32xf32> {
  %z = constant 0: index
  %f = constant 0.0: f32
  %r = vector.transfer_read %A[%z], %f
    {permutation_map = affine_map<(d0) -> (d0)>}
    : memref<?xf32>, vector<32xf32>
  return %r : vector<32xf32>
}

func @entry() {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c32 = constant 32: index
  %A = alloc(%c32) {alignment=64} : memref<?xf32>
  scf.for %i = %c0 to %c32 step %c1 {
    %f = constant 0.0: f32
    store %f, %A[%i] : memref<?xf32>
  }

  // On input, memory contains all zeros.
  %0 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %0 : vector<32xf32>

  // Overwrite with 16 values of 16 at base 3.
  // Statically guaranteed to be unmasked. Exercises proper alignment.
  %c3 = constant 3: index
  call @transfer_write16_unmasked_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  %1 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %1 : vector<32xf32>

  // Overwrite with 13 values of 13 at base 3.
  call @transfer_write13_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  %2 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %2 : vector<32xf32>

  // Overwrite with 17 values of 17 at base 7.
  %c7 = constant 7: index
  call @transfer_write17_1d(%A, %c3) : (memref<?xf32>, index) -> ()
  %3 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %3 : vector<32xf32>

  // Overwrite with 13 values of 13 at base 8.
  %c8 = constant 8: index
  call @transfer_write13_1d(%A, %c8) : (memref<?xf32>, index) -> ()
  %4 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %4 : vector<32xf32>

  // Overwrite with 17 values of 17 at base 14.
  %c14 = constant 14: index
  call @transfer_write17_1d(%A, %c14) : (memref<?xf32>, index) -> ()
  %5 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %5 : vector<32xf32>

  // Overwrite with 13 values of 13 at base 19.
  %c19 = constant 19: index
  call @transfer_write13_1d(%A, %c19) : (memref<?xf32>, index) -> ()
  %6 = call @transfer_read_1d(%A) : (memref<?xf32>) -> (vector<32xf32>)
  vector.print %6 : vector<32xf32>

  return
}

// CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 0 )
// CHECK: ( 0, 0, 0, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 17, 17, 17, 17, 17, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13 )
