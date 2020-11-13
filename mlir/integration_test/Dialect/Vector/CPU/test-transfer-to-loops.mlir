// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_runner_utils%shlibext,%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d1)>

func private @print_memref_f32(memref<*xf32>)

func @alloc_2d_filled_f32(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %c100 = constant 100 : index
  %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
  scf.for %arg5 = %c0 to %arg0 step %c1 {
    scf.for %arg6 = %c0 to %arg1 step %c1 {
      %arg66 = muli %arg6, %c100 : index
      %tmp1 = addi %arg5, %arg66 : index
      %tmp2 = index_cast %tmp1 : index to i32
      %tmp3 = sitofp %tmp2 : i32 to f32
      store %tmp3, %0[%arg5, %arg6] : memref<?x?xf32>
    }
  }
  return %0 : memref<?x?xf32>
}

func @main() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %cst = constant -4.2e+01 : f32
  %0 = call @alloc_2d_filled_f32(%c6, %c6) : (index, index) -> memref<?x?xf32>
  %converted = memref_cast %0 : memref<?x?xf32> to memref<*xf32>
  call @print_memref_f32(%converted): (memref<*xf32>) -> ()
  // CHECK:      Unranked{{.*}}data =
  // CHECK:      [
  // CHECK-SAME:  [0,   100,   200,   300,   400,   500],
  // CHECK-NEXT:  [1,   101,   201,   301,   401,   501],
  // CHECK-NEXT:  [2,   102,   202,   302,   402,   502],
  // CHECK-NEXT:  [3,   103,   203,   303,   403,   503],
  // CHECK-NEXT:  [4,   104,   204,   304,   404,   504],
  // CHECK-NEXT:  [5,   105,   205,   305,   405,   505]]

  %init = vector.transfer_read %0[%c1, %c1], %cst : memref<?x?xf32>, vector<5x5xf32>
  vector.print %init : vector<5x5xf32>
  // 5x5 block rooted at {1, 1}
  // CHECK-NEXT:  ( ( 101, 201, 301, 401, 501 ),
  // CHECK-SAME:    ( 102, 202, 302, 402, 502 ),
  // CHECK-SAME:    ( 103, 203, 303, 403, 503 ),
  // CHECK-SAME:    ( 104, 204, 304, 404, 504 ),
  // CHECK-SAME:    ( 105, 205, 305, 405, 505 ) )

  %1 = vector.transfer_read %0[%c1, %c1], %cst {permutation_map = #map0} : memref<?x?xf32>, vector<5x5xf32>
  vector.print %1 : vector<5x5xf32>
  // Transposed 5x5 block rooted @{1, 1} in memory.
  // CHECK-NEXT:  ( ( 101, 102, 103, 104, 105 ),
  // CHECK-SAME:    ( 201, 202, 203, 204, 205 ),
  // CHECK-SAME:    ( 301, 302, 303, 304, 305 ),
  // CHECK-SAME:    ( 401, 402, 403, 404, 405 ),
  // CHECK-SAME:    ( 501, 502, 503, 504, 505 ) )

  // Transpose-write the transposed 5x5 block @{0, 0} in memory.
  vector.transfer_write %1, %0[%c0, %c0] {permutation_map = #map0} : vector<5x5xf32>, memref<?x?xf32>

  %2 = vector.transfer_read %0[%c1, %c1], %cst : memref<?x?xf32>, vector<5x5xf32>
  vector.print %2 : vector<5x5xf32>
  // New 5x5 block rooted @{1, 1} in memory.
  // Here we expect the boundaries from the original data
  //   (i.e. last row: 105 .. 505, last col: 501 .. 505)
  // and the 4x4 subblock 202 .. 505 rooted @{0, 0} in the vector
  // CHECK-NEXT:  ( ( 202, 302, 402, 502, 501 ),
  // CHECK-SAME:    ( 203, 303, 403, 503, 502 ),
  // CHECK-SAME:    ( 204, 304, 404, 504, 503 ),
  // CHECK-SAME:    ( 205, 305, 405, 505, 504 ),
  // CHECK-SAME:    ( 105, 205, 305, 405, 505 ) )

  %3 = vector.transfer_read %0[%c2, %c3], %cst : memref<?x?xf32>, vector<5x5xf32>
  vector.print %3 : vector<5x5xf32>
  // New 5x5 block rooted @{2, 3} in memory.
  // CHECK-NEXT: ( ( 403, 503, 502, -42, -42 ),
  // CHECK-SAME:   ( 404, 504, 503, -42, -42 ),
  // CHECK-SAME:   ( 405, 505, 504, -42, -42 ),
  // CHECK-SAME:   ( 305, 405, 505, -42, -42 ),
  // CHECK-SAME:   ( -42, -42, -42, -42, -42 ) )

  %4 = vector.transfer_read %0[%c2, %c3], %cst {permutation_map = #map0} : memref<?x?xf32>, vector<5x5xf32>
  vector.print %4 : vector<5x5xf32>
  // Transposed 5x5 block rooted @{2, 3} in memory.
  // CHECK-NEXT: ( ( 403, 404, 405, 305, -42 ),
  // CHECK-SAME:   ( 503, 504, 505, 405, -42 ),
  // CHECK-SAME:   ( 502, 503, 504, 505, -42 ),
  // CHECK-SAME:   ( -42, -42, -42, -42, -42 ),
  // CHECK-SAME:   ( -42, -42, -42, -42, -42 ) )

  %5 = vector.transfer_read %0[%c2, %c3], %cst {permutation_map = #map1} : memref<?x?xf32>, vector<5xf32>
  vector.print %5 : vector<5xf32>
  // CHECK-NEXT: ( 403, 503, 502, -42, -42 )

  dealloc %0 : memref<?x?xf32>
  return
}
