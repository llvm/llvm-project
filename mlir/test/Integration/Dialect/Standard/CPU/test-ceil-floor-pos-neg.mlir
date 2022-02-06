// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-cf \
// RUN:   -memref-expand -arith-expand -convert-vector-to-llvm \
// RUN:   -convert-memref-to-llvm -convert-std-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @transfer_read_2d(%A : memref<40xi32>, %base1: index) {
  %i42 = arith.constant -42: i32
  %f = vector.transfer_read %A[%base1], %i42
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<40xi32>, vector<40xi32>
  vector.print %f: vector<40xi32>
  return
}

func @entry() {
  %c0 = arith.constant 0: index
  %c20 = arith.constant 20: i32
  %c10 = arith.constant 10: i32
  %cmin10 = arith.constant -10: i32
  %cmax_int = arith.constant 2147483647: i32
  %A = memref.alloc() : memref<40xi32>

  // print numerator
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %ii30 = arith.subi %ii, %c20 : i32
    memref.store %ii30, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()

  // test with ceil(*, 10)
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %ii30 = arith.subi %ii, %c20 : i32
    %val = arith.ceildivsi %ii30, %c10 : i32
    memref.store %val, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()

    // test with floor(*, 10)
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %ii30 = arith.subi %ii, %c20 : i32
    %val = arith.floordivsi %ii30, %c10 : i32
    memref.store %val, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()


  // test with ceil(*, -10)
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %ii30 = arith.subi %ii, %c20 : i32
    %val = arith.ceildivsi %ii30, %cmin10 : i32
    memref.store %val, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()

  // test with floor(*, -10)
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %ii30 = arith.subi %ii, %c20 : i32
    %val = arith.floordivsi %ii30, %cmin10 : i32
    memref.store %val, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()

  // test with ceildivui(*, 10)
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %val = arith.ceildivui %ii, %c10 : i32
    memref.store %val, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()

  // test with ceildivui(*, -1)
  affine.for %i = 0 to 40  {
    %ii = arith.index_cast %i: index to i32
    %ii30 = arith.subi %ii, %c20 : i32
    %val = arith.ceildivui %ii30, %cmax_int : i32
    memref.store %val, %A[%i] : memref<40xi32>
  }
  call @transfer_read_2d(%A, %c0) : (memref<40xi32>, index) -> ()

  memref.dealloc %A : memref<40xi32>
  return
}

// List below is aligned for easy manual check
// legend: num, signed_ceil(num, 10), floor(num, 10), signed_ceil(num, -10), floor(num, -10), unsigned_ceil(num, 10), unsigned_ceil(num, max_int)
//  ( -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 )
//  (  -2,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2 )
//  (  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 )
//  (   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
//  (   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2 )

// CHECK:( -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 )
// CHECK:( -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2 )
// CHECK:( -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
// CHECK:( 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 )
// CHECK:( 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2 )
// CHECK:( 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4 )
// CHECK:( 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )
