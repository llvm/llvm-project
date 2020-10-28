// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @compress16(%base: memref<?xf32>,
                 %mask: vector<16xi1>, %value: vector<16xf32>) {
  vector.compressstore %base, %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func @printmem16(%A: memref<?xf32>) {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c16 = constant 16: index
  %z = constant 0.0: f32
  %m = vector.broadcast %z : f32 to vector<16xf32>
  %mem = scf.for %i = %c0 to %c16 step %c1
    iter_args(%m_iter = %m) -> (vector<16xf32>) {
    %c = load %A[%i] : memref<?xf32>
    %i32 = index_cast %i : index to i32
    %m_new = vector.insertelement %c, %m_iter[%i32 : i32] : vector<16xf32>
    scf.yield %m_new : vector<16xf32>
  }
  vector.print %mem : vector<16xf32>
  return
}

func @entry() {
  // Set up memory.
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c16 = constant 16: index
  %A = alloc(%c16) : memref<?xf32>
  %z = constant 0.0: f32
  %v = vector.broadcast %z : f32 to vector<16xf32>
  %value = scf.for %i = %c0 to %c16 step %c1
    iter_args(%v_iter = %v) -> (vector<16xf32>) {
    store %z, %A[%i] : memref<?xf32>
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    %v_new = vector.insertelement %fi, %v_iter[%i32 : i32] : vector<16xf32>
    scf.yield %v_new : vector<16xf32>
  }

  // Set up masks.
  %f = constant 0: i1
  %t = constant 1: i1
  %none = vector.constant_mask [0] : vector<16xi1>
  %all = vector.constant_mask [16] : vector<16xi1>
  %some1 = vector.constant_mask [4] : vector<16xi1>
  %0 = vector.insert %f, %some1[0] : i1 into vector<16xi1>
  %1 = vector.insert %t, %0[7] : i1 into vector<16xi1>
  %2 = vector.insert %t, %1[11] : i1 into vector<16xi1>
  %3 = vector.insert %t, %2[13] : i1 into vector<16xi1>
  %some2 = vector.insert %t, %3[15] : i1 into vector<16xi1>
  %some3 = vector.insert %f, %some2[2] : i1 into vector<16xi1>

  //
  // Expanding load tests.
  //

  call @compress16(%A, %none, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @compress16(%A, %all, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK-NEXT: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @compress16(%A, %some3, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK-NEXT: ( 1, 3, 7, 11, 13, 15, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @compress16(%A, %some2, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK-NEXT: ( 1, 2, 3, 7, 11, 13, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @compress16(%A, %some1, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK-NEXT: ( 0, 1, 2, 3, 11, 13, 15, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  return
}
