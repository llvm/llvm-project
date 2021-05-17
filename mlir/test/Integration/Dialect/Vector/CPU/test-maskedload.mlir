// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @maskedload16(%base: memref<?xf32>, %mask: vector<16xi1>,
                   %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c0 = constant 0: index
  %ld = vector.maskedload %base[%c0], %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

func @maskedload16_at8(%base: memref<?xf32>, %mask: vector<16xi1>,
                       %pass_thru: vector<16xf32>) -> vector<16xf32> {
  %c8 = constant 8: index
  %ld = vector.maskedload %base[%c8], %mask, %pass_thru
    : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return %ld : vector<16xf32>
}

func @entry() {
  // Set up memory.
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c16 = constant 16: index
  %A = memref.alloc(%c16) : memref<?xf32>
  scf.for %i = %c0 to %c16 step %c1 {
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    memref.store %fi, %A[%i] : memref<?xf32>
  }

  // Set up pass thru vector.
  %u = constant -7.0: f32
  %pass = vector.broadcast %u : f32 to vector<16xf32>

  // Set up masks.
  %f = constant 0: i1
  %t = constant 1: i1
  %none = vector.constant_mask [0] : vector<16xi1>
  %all = vector.constant_mask [16] : vector<16xi1>
  %some = vector.constant_mask [8] : vector<16xi1>
  %0 = vector.insert %f, %some[0] : i1 into vector<16xi1>
  %1 = vector.insert %t, %0[13] : i1 into vector<16xi1>
  %2 = vector.insert %t, %1[14] : i1 into vector<16xi1>
  %other = vector.insert %t, %2[14] : i1 into vector<16xi1>

  //
  // Masked load tests.
  //

  %l1 = call @maskedload16(%A, %none, %pass)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> (vector<16xf32>)
  vector.print %l1 : vector<16xf32>
  // CHECK: ( -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7 )

  %l2 = call @maskedload16(%A, %all, %pass)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> (vector<16xf32>)
  vector.print %l2 : vector<16xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  %l3 = call @maskedload16(%A, %some, %pass)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> (vector<16xf32>)
  vector.print %l3 : vector<16xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, -7, -7, -7, -7, -7, -7, -7, -7 )

  %l4 = call @maskedload16(%A, %other, %pass)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> (vector<16xf32>)
  vector.print %l4 : vector<16xf32>
  // CHECK: ( -7, 1, 2, 3, 4, 5, 6, 7, -7, -7, -7, -7, -7, 13, 14, -7 )

  %l5 = call @maskedload16_at8(%A, %some, %pass)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> (vector<16xf32>)
  vector.print %l5 : vector<16xf32>
  // CHECK: ( 8, 9, 10, 11, 12, 13, 14, 15, -7, -7, -7, -7, -7, -7, -7, -7 )

  return
}

