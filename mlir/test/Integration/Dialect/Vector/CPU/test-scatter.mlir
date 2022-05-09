// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-memref-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @scatter8(%base: memref<?xf32>,
               %indices: vector<8xi32>,
               %mask: vector<8xi1>, %value: vector<8xf32>) {
  %c0 = arith.constant 0: index
  vector.scatter %base[%c0][%indices], %mask, %value
    : memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>
  return
}

func.func @printmem8(%A: memref<?xf32>) {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c8 = arith.constant 8: index
  %z = arith.constant 0.0: f32
  %m = vector.broadcast %z : f32 to vector<8xf32>
  %mem = scf.for %i = %c0 to %c8 step %c1
    iter_args(%m_iter = %m) -> (vector<8xf32>) {
    %c = memref.load %A[%i] : memref<?xf32>
    %i32 = arith.index_cast %i : index to i32
    %m_new = vector.insertelement %c, %m_iter[%i32 : i32] : vector<8xf32>
    scf.yield %m_new : vector<8xf32>
  }
  vector.print %mem : vector<8xf32>
  return
}

func.func @entry() {
  // Set up memory.
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c8 = arith.constant 8: index
  %A = memref.alloc(%c8) : memref<?xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %i32 = arith.index_cast %i : index to i32
    %fi = arith.sitofp %i32 : i32 to f32
    memref.store %fi, %A[%i] : memref<?xf32>
  }

  // Set up idx vector.
  %i0 = arith.constant 0: i32
  %i1 = arith.constant 1: i32
  %i2 = arith.constant 2: i32
  %i3 = arith.constant 3: i32
  %i4 = arith.constant 4: i32
  %i5 = arith.constant 5: i32
  %i6 = arith.constant 6: i32
  %i7 = arith.constant 7: i32
  %0 = vector.broadcast %i7 : i32 to vector<8xi32>
  %1 = vector.insert %i0, %0[1] : i32 into vector<8xi32>
  %2 = vector.insert %i1, %1[2] : i32 into vector<8xi32>
  %3 = vector.insert %i6, %2[3] : i32 into vector<8xi32>
  %4 = vector.insert %i2, %3[4] : i32 into vector<8xi32>
  %5 = vector.insert %i4, %4[5] : i32 into vector<8xi32>
  %6 = vector.insert %i5, %5[6] : i32 into vector<8xi32>
  %idx = vector.insert %i3, %6[7] : i32 into vector<8xi32>

  // Set up value vector.
  %f0 = arith.constant 0.0: f32
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32
  %f6 = arith.constant 6.0: f32
  %f7 = arith.constant 7.0: f32
  %7 = vector.broadcast %f0 : f32 to vector<8xf32>
  %8 = vector.insert %f1, %7[1] : f32 into vector<8xf32>
  %9 = vector.insert %f2, %8[2] : f32 into vector<8xf32>
  %10 = vector.insert %f3, %9[3] : f32 into vector<8xf32>
  %11 = vector.insert %f4, %10[4] : f32 into vector<8xf32>
  %12 = vector.insert %f5, %11[5] : f32 into vector<8xf32>
  %13 = vector.insert %f6, %12[6] : f32 into vector<8xf32>
  %val = vector.insert %f7, %13[7] : f32 into vector<8xf32>

  // Set up masks.
  %t = arith.constant 1: i1
  %none = vector.constant_mask [0] : vector<8xi1>
  %some = vector.constant_mask [4] : vector<8xi1>
  %more = vector.insert %t, %some[7] : i1 into vector<8xi1>
  %all = vector.constant_mask [8] : vector<8xi1>

  //
  // Scatter tests.
  //

  vector.print %idx : vector<8xi32>
  // CHECK: ( 7, 0, 1, 6, 2, 4, 5, 3 )

  call @printmem8(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )

  call @scatter8(%A, %idx, %none, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()
  call @printmem8(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )

  call @scatter8(%A, %idx, %some, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()
  call @printmem8(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 1, 2, 2, 3, 4, 5, 3, 0 )

  call @scatter8(%A, %idx, %more, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()
  call @printmem8(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 1, 2, 2, 7, 4, 5, 3, 0 )

  call @scatter8(%A, %idx, %all, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()
  call @printmem8(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 1, 2, 4, 7, 5, 6, 3, 0 )

  memref.dealloc %A : memref<?xf32>
  return
}
