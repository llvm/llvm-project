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
// This example illustrates a DOT version for the operation. Another example
// in this directory illustrates an effective SAXPY version that operates on the
// transposed jagged diagonal storage to obtain higher vector lengths.

#contraction_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#dot_trait = {
  indexing_maps = #contraction_accesses,
  iterator_types = ["reduction"]
}

func @spmv8x8(%AVAL: memref<8xvector<4xf32>>,
              %AIDX: memref<8xvector<4xi32>>, %X: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %cn = constant 8 : index
  %f0 = constant 0.0 : f32
  %mask = vector.constant_mask [4] : vector<4xi1>
  %pass = vector.broadcast %f0 : f32 to vector<4xf32>
  scf.for %i = %c0 to %cn step %c1 {
    %aval = load %AVAL[%i] : memref<8xvector<4xf32>>
    %aidx = load %AIDX[%i] : memref<8xvector<4xi32>>
    %0 = vector.gather %X[%c0][%aidx], %mask, %pass
       : memref<?xf32>, vector<4xi32>, vector<4xi1>, vector<4xf32> into vector<4xf32>
    %1 = vector.contract #dot_trait %aval, %0, %f0 : vector<4xf32>, vector<4xf32> into f32
    store %1, %B[%i] : memref<?xf32>
  }
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

  %AVAL = alloc()    {alignment = 64} : memref<8xvector<4xf32>>
  %AIDX = alloc()    {alignment = 64} : memref<8xvector<4xi32>>
  %X    = alloc(%c8) {alignment = 64} : memref<?xf32>
  %B    = alloc(%c8) {alignment = 64} : memref<?xf32>

  //
  // Initialize.
  //

  %vf1 = vector.broadcast %f1 : f32 to vector<4xf32>

  %0 = vector.insert %f2, %vf1[1] : f32 into vector<4xf32>
  store %0, %AVAL[%c0] : memref<8xvector<4xf32>>

  %1 = vector.insert %f8, %vf1[1] : f32 into vector<4xf32>
  %2 = vector.insert %f3, %1[2]   : f32 into vector<4xf32>
  store %2, %AVAL[%c1] : memref<8xvector<4xf32>>

  %3 = vector.insert %f2, %vf1[1] : f32 into vector<4xf32>
  %4 = vector.insert %f6, %3[2]   : f32 into vector<4xf32>
  %5 = vector.insert %f2, %4[3]   : f32 into vector<4xf32>
  store %5, %AVAL[%c2] : memref<8xvector<4xf32>>

  %6 = vector.insert %f3, %vf1[0] : f32 into vector<4xf32>
  store %6, %AVAL[%c3] : memref<8xvector<4xf32>>

  %7 = vector.insert %f5, %vf1[0] : f32 into vector<4xf32>
  store %7, %AVAL[%c4] : memref<8xvector<4xf32>>

  %8 = vector.insert %f3, %vf1[0] : f32 into vector<4xf32>
  %9 = vector.insert %f2, %8[1]   : f32 into vector<4xf32>
  %10 = vector.insert %f2, %9[3]   : f32 into vector<4xf32>
  store %10, %AVAL[%c5] : memref<8xvector<4xf32>>

  %11 = vector.insert %f4, %vf1[0] : f32 into vector<4xf32>
  %12 = vector.insert %f7, %11[1]   : f32 into vector<4xf32>
  store %12, %AVAL[%c6] : memref<8xvector<4xf32>>

  %13 = vector.insert %f3, %vf1[0] : f32 into vector<4xf32>
  %14 = vector.insert %f2, %13[1]   : f32 into vector<4xf32>
  store %14, %AVAL[%c7] : memref<8xvector<4xf32>>

  %vi0 = vector.broadcast %i0 : i32 to vector<4xi32>

  %20 = vector.insert %i2, %vi0[1] : i32 into vector<4xi32>
  %21 = vector.insert %i5, %20[2] : i32 into vector<4xi32>
  %22 = vector.insert %i7, %21[3] : i32 into vector<4xi32>
  store %22, %AIDX[%c0] : memref<8xvector<4xi32>>

  %23 = vector.insert %i1, %vi0[1] : i32 into vector<4xi32>
  %24 = vector.insert %i4, %23[2] : i32 into vector<4xi32>
  %25 = vector.insert %i6, %24[3] : i32 into vector<4xi32>
  store %25, %AIDX[%c1] : memref<8xvector<4xi32>>

  %26 = vector.insert %i2, %vi0[0] : i32 into vector<4xi32>
  %27 = vector.insert %i5, %26[1] : i32 into vector<4xi32>
  %28 = vector.insert %i6, %27[2] : i32 into vector<4xi32>
  %29 = vector.insert %i7, %28[3] : i32 into vector<4xi32>
  store %29, %AIDX[%c2] : memref<8xvector<4xi32>>

  %30 = vector.insert %i1, %vi0[0] : i32 into vector<4xi32>
  %31 = vector.insert %i3, %30[1] : i32 into vector<4xi32>
  %32 = vector.insert %i5, %31[2] : i32 into vector<4xi32>
  %33 = vector.insert %i7, %32[3] : i32 into vector<4xi32>
  store %33, %AIDX[%c3] : memref<8xvector<4xi32>>

  %34 = vector.insert %i3, %vi0[1] : i32 into vector<4xi32>
  %35 = vector.insert %i4, %34[2] : i32 into vector<4xi32>
  %36 = vector.insert %i5, %35[3] : i32 into vector<4xi32>
  store %36, %AIDX[%c4] : memref<8xvector<4xi32>>

  %37 = vector.insert %i1, %vi0[0] : i32 into vector<4xi32>
  %38 = vector.insert %i4, %37[1] : i32 into vector<4xi32>
  %39 = vector.insert %i5, %38[2] : i32 into vector<4xi32>
  %40 = vector.insert %i6, %39[3] : i32 into vector<4xi32>
  store %40, %AIDX[%c5] : memref<8xvector<4xi32>>

  %41 = vector.insert %i2, %vi0[1] : i32 into vector<4xi32>
  %42 = vector.insert %i4, %41[2] : i32 into vector<4xi32>
  %43 = vector.insert %i6, %42[3] : i32 into vector<4xi32>
  store %43, %AIDX[%c6] : memref<8xvector<4xi32>>

  %44 = vector.insert %i1, %vi0[0] : i32 into vector<4xi32>
  %45 = vector.insert %i3, %44[1] : i32 into vector<4xi32>
  %46 = vector.insert %i6, %45[2] : i32 into vector<4xi32>
  %47 = vector.insert %i7, %46[3] : i32 into vector<4xi32>
  store %47, %AIDX[%c7] : memref<8xvector<4xi32>>

  scf.for %i = %c0 to %c8 step %c1 {
    %ix = addi %i, %c1 : index
    %kx = index_cast %ix : index to i32
    %fx = sitofp %kx : i32 to f32
    store %fx, %X[%i] : memref<?xf32>
    store %f0, %B[%i] : memref<?xf32>
  }

  //
  // Multiply.
  //

  call @spmv8x8(%AVAL, %AIDX, %X, %B) : (memref<8xvector<4xf32>>,
                                         memref<8xvector<4xi32>>,
                                         memref<?xf32>, memref<?xf32>) -> ()

  //
  // Print and verify.
  //

  scf.for %i = %c0 to %c8 step %c1 {
    %aval = load %AVAL[%i] : memref<8xvector<4xf32>>
    vector.print %aval : vector<4xf32>
  }

  scf.for %i = %c0 to %c8 step %c1 {
    %aidx = load %AIDX[%i] : memref<8xvector<4xi32>>
    vector.print %aidx : vector<4xi32>
  }

  scf.for %i = %c0 to %c8 step %c1 {
    %ldb = load %B[%i] : memref<?xf32>
    vector.print %ldb : f32
  }

  //
  // CHECK:      ( 1, 2, 1, 1 )
  // CHECK-NEXT: ( 1, 8, 3, 1 )
  // CHECK-NEXT: ( 1, 2, 6, 2 )
  // CHECK-NEXT: ( 3, 1, 1, 1 )
  // CHECK-NEXT: ( 5, 1, 1, 1 )
  // CHECK-NEXT: ( 3, 2, 1, 2 )
  // CHECK-NEXT: ( 4, 7, 1, 1 )
  // CHECK-NEXT: ( 3, 2, 1, 1 )
  //
  // CHECK-NEXT: ( 0, 2, 5, 7 )
  // CHECK-NEXT: ( 0, 1, 4, 6 )
  // CHECK-NEXT: ( 2, 5, 6, 7 )
  // CHECK-NEXT: ( 1, 3, 5, 7 )
  // CHECK-NEXT: ( 0, 3, 4, 5 )
  // CHECK-NEXT: ( 1, 4, 5, 6 )
  // CHECK-NEXT: ( 0, 2, 4, 6 )
  // CHECK-NEXT: ( 1, 3, 6, 7 )
  //
  // CHECK-NEXT: 21
  // CHECK-NEXT: 39
  // CHECK-NEXT: 73
  // CHECK-NEXT: 24
  // CHECK-NEXT: 20
  // CHECK-NEXT: 36
  // CHECK-NEXT: 37
  // CHECK-NEXT: 29
  //

  //
  // Free.
  //

  dealloc %AVAL : memref<8xvector<4xf32>>
  dealloc %AIDX : memref<8xvector<4xi32>>
  dealloc %X    : memref<?xf32>
  dealloc %B    : memref<?xf32>

  return
}
