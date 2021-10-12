// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @compress16(%base: memref<?xf32>,
                 %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = arith.constant 0: index
  vector.compressstore %base[%c0], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func @compress16_at8(%base: memref<?xf32>,
                     %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c8 = arith.constant 8: index
  vector.compressstore %base[%c8], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func @printmem16(%A: memref<?xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c16 = arith.constant 16: index
  %z = arith.constant 0.0: f32
  %m = vector.broadcast %z : f32 to vector<16xf32>
  %mem = scf.for %i = %c0 to %c16 step %c1
    iter_args(%m_iter = %m) -> (vector<16xf32>) {
    %c = memref.load %A[%i] : memref<?xf32>
    %i32 = arith.index_cast %i : index to i32
    %m_new = vector.insertelement %c, %m_iter[%i32 : i32] : vector<16xf32>
    scf.yield %m_new : vector<16xf32>
  }
  vector.print %mem : vector<16xf32>
  return
}

func @entry() {
  // Set up memory.
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c16 = arith.constant 16: index
  %A = memref.alloc(%c16) : memref<?xf32>
  %z = arith.constant 0.0: f32
  %v = vector.broadcast %z : f32 to vector<16xf32>
  %value = scf.for %i = %c0 to %c16 step %c1
    iter_args(%v_iter = %v) -> (vector<16xf32>) {
    memref.store %z, %A[%i] : memref<?xf32>
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    %v_new = vector.insertelement %fi, %v_iter[%i32 : i32] : vector<16xf32>
    scf.yield %v_new : vector<16xf32>
  }

  // Set up masks.
  %f = arith.constant 0: i1
  %t = arith.constant 1: i1
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

  call @compress16_at8(%A, %some1, %value)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK-NEXT: ( 0, 1, 2, 3, 11, 13, 15, 7, 0, 1, 2, 3, 12, 13, 14, 15 )

  memref.dealloc %A : memref<?xf32>
  return
}
