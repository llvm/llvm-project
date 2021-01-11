// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Illustrates an 8x8 Sparse Matrix x Vector implemented with only operations
// of the vector dialect (and some std/scf). Essentially, this example performs
// the following multiplication:
//
//     0  1  2  3  4  5  6  7
//   +------------------------+
// 0 | 1  0  2  0  0  1  0  1 |   | 1 |   | 21 |
// 1 | 1  8  0  0  3  0  1  0 |   | 2 |   | 39 |
// 2 | 0  0  1  0  0  2  6  2 |   | 3 |   | 73 |
// 3 | 0  3  0  1  0  1  0  1 | x | 4 | = | 24 |
// 4 | 5  0  0  1  1  1  0  0 |   | 5 |   | 20 |
// 5 | 0  3  0  0  2  1  2  0 |   | 6 |   | 36 |
// 6 | 4  0  7  0  1  0  1  0 |   | 7 |   | 37 |
// 7 | 0  3  0  2  0  0  1  1 |   | 8 |   | 29 |
//   +------------------------+
//
// The sparse storage scheme used is an extended column scheme (also referred
// to as jagged diagonal, which is essentially a vector friendly variant of
// the general sparse row-wise scheme (also called compressed row storage),
// using fixed length vectors and no explicit pointer indexing into the
// value array to find the rows.
//
// The extended column storage for the matrix shown above is as follows.
//
//      VALUE           INDEX
//   +---------+     +---------+
// 0 | 1 2 1 1 |     | 0 2 5 7 |
// 1 | 1 8 3 1 |     | 0 1 4 6 |
// 2 | 1 2 6 2 |     | 2 5 6 7 |
// 3 | 3 1 1 1 |     | 1 3 5 7 |
// 4 | 5 1 1 1 |     | 0 3 4 5 |
// 5 | 3 2 1 2 |     | 1 4 5 6 |
// 6 | 4 7 1 1 |     | 0 2 4 6 |
// 7 | 3 2 1 1 |     | 1 3 6 7 |
//   +---------+     +---------+
//
// This example illustrates an effective SAXPY version that operates
// on the transposed jagged diagonal storage to obtain higher vector
// lengths. Another example in this directory illustrates a DOT
// version of the operation.

func @spmv8x8(%AVAL: memref<4xvector<8xf32>>,
              %AIDX: memref<4xvector<8xi32>>,
	      %X: memref<?xf32>, %B: memref<1xvector<8xf32>>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cn = constant 4 : index
  %f0 = constant 0.0 : f32
  %mask = vector.constant_mask [8] : vector<8xi1>
  %pass = vector.broadcast %f0 : f32 to vector<8xf32>
  %b = load %B[%c0] : memref<1xvector<8xf32>>
  %b_out = scf.for %k = %c0 to %cn step %c1 iter_args(%b_iter = %b) -> (vector<8xf32>) {
    %aval = load %AVAL[%k] : memref<4xvector<8xf32>>
    %aidx = load %AIDX[%k] : memref<4xvector<8xi32>>
    %0 = vector.gather %X[%aidx], %mask, %pass
       : memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
    %b_new = vector.fma %aval, %0, %b_iter : vector<8xf32>
    scf.yield %b_new : vector<8xf32>
  }
  store %b_out, %B[%c0] : memref<1xvector<8xf32>>
  return
}

func @entry() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %c5 = constant 5 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c8 = constant 8 : index

  %f0 = constant 0.0 : f32
  %f1 = constant 1.0 : f32
  %f2 = constant 2.0 : f32
  %f3 = constant 3.0 : f32
  %f4 = constant 4.0 : f32
  %f5 = constant 5.0 : f32
  %f6 = constant 6.0 : f32
  %f7 = constant 7.0 : f32
  %f8 = constant 8.0 : f32

  %i0 = constant 0 : i32
  %i1 = constant 1 : i32
  %i2 = constant 2 : i32
  %i3 = constant 3 : i32
  %i4 = constant 4 : i32
  %i5 = constant 5 : i32
  %i6 = constant 6 : i32
  %i7 = constant 7 : i32

  //
  // Allocate.
  //

  %AVAL = alloc()    {alignment = 64} : memref<4xvector<8xf32>>
  %AIDX = alloc()    {alignment = 64} : memref<4xvector<8xi32>>
  %X    = alloc(%c8) {alignment = 64} : memref<?xf32>
  %B    = alloc()    {alignment = 64} : memref<1xvector<8xf32>>

  //
  // Initialize.
  //

  %vf1 = vector.broadcast %f1 : f32 to vector<8xf32>

  %0 = vector.insert %f3, %vf1[3] : f32 into vector<8xf32>
  %1 = vector.insert %f5, %0[4] : f32 into vector<8xf32>
  %2 = vector.insert %f3, %1[5] : f32 into vector<8xf32>
  %3 = vector.insert %f4, %2[6] : f32 into vector<8xf32>
  %4 = vector.insert %f3, %3[7] : f32 into vector<8xf32>
  store %4, %AVAL[%c0] : memref<4xvector<8xf32>>

  %5 = vector.insert %f2, %vf1[0] : f32 into vector<8xf32>
  %6 = vector.insert %f8, %5[1] : f32 into vector<8xf32>
  %7 = vector.insert %f2, %6[2] : f32 into vector<8xf32>
  %8 = vector.insert %f2, %7[5] : f32 into vector<8xf32>
  %9 = vector.insert %f7, %8[6] : f32 into vector<8xf32>
  %10 = vector.insert %f2, %9[7] : f32 into vector<8xf32>
  store %10, %AVAL[%c1] : memref<4xvector<8xf32>>

  %11 = vector.insert %f3, %vf1[1] : f32 into vector<8xf32>
  %12 = vector.insert %f6, %11[2] : f32 into vector<8xf32>
  store %12, %AVAL[%c2] : memref<4xvector<8xf32>>

  %13 = vector.insert %f2, %vf1[2] : f32 into vector<8xf32>
  %14 = vector.insert %f2, %13[5] : f32 into vector<8xf32>
  store %14, %AVAL[%c3] : memref<4xvector<8xf32>>

  %vi0 = vector.broadcast %i0 : i32 to vector<8xi32>

  %20 = vector.insert %i2, %vi0[2] : i32 into vector<8xi32>
  %21 = vector.insert %i1, %20[3] : i32 into vector<8xi32>
  %22 = vector.insert %i1, %21[5] : i32 into vector<8xi32>
  %23 = vector.insert %i1, %22[7] : i32 into vector<8xi32>
  store %23, %AIDX[%c0] : memref<4xvector<8xi32>>

  %24 = vector.insert %i2, %vi0[0] : i32 into vector<8xi32>
  %25 = vector.insert %i1, %24[1] : i32 into vector<8xi32>
  %26 = vector.insert %i5, %25[2] : i32 into vector<8xi32>
  %27 = vector.insert %i3, %26[3] : i32 into vector<8xi32>
  %28 = vector.insert %i3, %27[4] : i32 into vector<8xi32>
  %29 = vector.insert %i4, %28[5] : i32 into vector<8xi32>
  %30 = vector.insert %i2, %29[6] : i32 into vector<8xi32>
  %31 = vector.insert %i3, %30[7] : i32 into vector<8xi32>
  store %31, %AIDX[%c1] : memref<4xvector<8xi32>>

  %32 = vector.insert %i5, %vi0[0] : i32 into vector<8xi32>
  %33 = vector.insert %i4, %32[1] : i32 into vector<8xi32>
  %34 = vector.insert %i6, %33[2] : i32 into vector<8xi32>
  %35 = vector.insert %i5, %34[3] : i32 into vector<8xi32>
  %36 = vector.insert %i4, %35[4] : i32 into vector<8xi32>
  %37 = vector.insert %i5, %36[5] : i32 into vector<8xi32>
  %38 = vector.insert %i4, %37[6] : i32 into vector<8xi32>
  %39 = vector.insert %i6, %38[7] : i32 into vector<8xi32>
  store %39, %AIDX[%c2] : memref<4xvector<8xi32>>

  %40 = vector.insert %i7, %vi0[0] : i32 into vector<8xi32>
  %41 = vector.insert %i6, %40[1] : i32 into vector<8xi32>
  %42 = vector.insert %i7, %41[2] : i32 into vector<8xi32>
  %43 = vector.insert %i7, %42[3] : i32 into vector<8xi32>
  %44 = vector.insert %i5, %43[4] : i32 into vector<8xi32>
  %45 = vector.insert %i6, %44[5] : i32 into vector<8xi32>
  %46 = vector.insert %i6, %45[6] : i32 into vector<8xi32>
  %47 = vector.insert %i7, %46[7] : i32 into vector<8xi32>
  store %47, %AIDX[%c3] : memref<4xvector<8xi32>>

  %vf0 = vector.broadcast %f0 : f32 to vector<8xf32>
  store %vf0, %B[%c0] : memref<1xvector<8xf32>>

  scf.for %i = %c0 to %c8 step %c1 {
    %ix = addi %i, %c1 : index
    %kx = index_cast %ix : index to i32
    %fx = sitofp %kx : i32 to f32
    store %fx, %X[%i] : memref<?xf32>
  }

  //
  // Multiply.
  //

  call @spmv8x8(%AVAL, %AIDX, %X, %B) : (memref<4xvector<8xf32>>,
                                         memref<4xvector<8xi32>>,
					 memref<?xf32>,
					 memref<1xvector<8xf32>>) -> ()

  //
  // Print and verify.
  //

  scf.for %i = %c0 to %c4 step %c1 {
    %aval = load %AVAL[%i] : memref<4xvector<8xf32>>
    vector.print %aval : vector<8xf32>
  }

  scf.for %i = %c0 to %c4 step %c1 {
    %aidx = load %AIDX[%i] : memref<4xvector<8xi32>>
    vector.print %aidx : vector<8xi32>
  }

  %ldb = load %B[%c0] : memref<1xvector<8xf32>>
  vector.print %ldb : vector<8xf32>

  //
  // CHECK:      ( 1, 1, 1, 3, 5, 3, 4, 3 )
  // CHECK-NEXT: ( 2, 8, 2, 1, 1, 2, 7, 2 )
  // CHECK-NEXT: ( 1, 3, 6, 1, 1, 1, 1, 1 )
  // CHECK-NEXT: ( 1, 1, 2, 1, 1, 2, 1, 1 )
  //
  // CHECK-NEXT: ( 0, 0, 2, 1, 0, 1, 0, 1 )
  // CHECK-NEXT: ( 2, 1, 5, 3, 3, 4, 2, 3 )
  // CHECK-NEXT: ( 5, 4, 6, 5, 4, 5, 4, 6 )
  // CHECK-NEXT: ( 7, 6, 7, 7, 5, 6, 6, 7 )
  //
  // CHECK-NEXT: ( 21, 39, 73, 24, 20, 36, 37, 29 )
  //

  //
  // Free.
  //

  dealloc %AVAL : memref<4xvector<8xf32>>
  dealloc %AIDX : memref<4xvector<8xi32>>
  dealloc %X    : memref<?xf32>
  dealloc %B    : memref<1xvector<8xf32>>

  return
}
