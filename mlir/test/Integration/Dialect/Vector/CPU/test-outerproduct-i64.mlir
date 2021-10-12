// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!vector_type_A = type vector<8xi64>
!vector_type_B = type vector<8xi64>
!vector_type_C = type vector<8x8xi64>

!vector_type_X = type vector<2xi64>
!vector_type_Y = type vector<3xi64>
!vector_type_Z = type vector<2x3xi64>

!vector_type_R = type vector<7xi64>

func @vector_outerproduct_splat_8x8(%ia: i64, %ib: i64, %ic: i64) -> !vector_type_C {
  %a = splat %ia: !vector_type_A
  %b = splat %ib: !vector_type_B
  %c = splat %ic: !vector_type_C
  %d = vector.outerproduct %a, %b, %c : !vector_type_A, !vector_type_B
  return %d: !vector_type_C
}

func @vector_outerproduct_vec_2x3(%x : !vector_type_X,
                                  %y : !vector_type_Y) -> !vector_type_Z {
  %o = vector.outerproduct %x, %y : !vector_type_X, !vector_type_Y
  return %o: !vector_type_Z
}

func @vector_outerproduct_vec_2x3_acc(%x : !vector_type_X,
                                      %y : !vector_type_Y,
                                      %z : !vector_type_Z) -> !vector_type_Z {
  %o = vector.outerproduct %x, %y, %z : !vector_type_X, !vector_type_Y
  return %o: !vector_type_Z
}

func @entry() {
  %i0 = arith.constant 0: i64
  %i1 = arith.constant 1: i64
  %i2 = arith.constant 2: i64
  %i3 = arith.constant 3: i64
  %i4 = arith.constant 4: i64
  %i5 = arith.constant 5: i64
  %i10 = arith.constant 10: i64

  // Simple case, splat scalars into vectors, then take outer product.
  %v = call @vector_outerproduct_splat_8x8(%i1, %i2, %i10)
      : (i64, i64, i64) -> (!vector_type_C)
  vector.print %v : !vector_type_C
  //
  // outer product 8x8:
  //
  // CHECK-COUNT-8: ( 12, 12, 12, 12, 12, 12, 12, 12 )

  // Direct outerproduct on vectors with different size.
  %0 = vector.broadcast %i1 : i64 to !vector_type_X
  %x = vector.insert %i2, %0[1] : i64 into !vector_type_X
  %1 = vector.broadcast %i3 : i64 to !vector_type_Y
  %2 = vector.insert %i4, %1[1] : i64 into !vector_type_Y
  %y = vector.insert %i5, %2[2] : i64 into !vector_type_Y

  %p = call @vector_outerproduct_vec_2x3(%x, %y)
      : (!vector_type_X, !vector_type_Y) -> (!vector_type_Z)
  vector.print %p : !vector_type_Z
  //
  // outer product 2x3:
  //
  // CHECK: ( ( 3, 4, 5 ), ( 6, 8, 10 ) )

  %q = call @vector_outerproduct_vec_2x3_acc(%x, %y, %p)
      : (!vector_type_X, !vector_type_Y, !vector_type_Z) -> (!vector_type_Z)
  vector.print %q : !vector_type_Z
  //
  // outer product 2x3:
  //
  // CHECK: ( ( 6, 8, 10 ), ( 12, 16, 20 ) )

  %3 = vector.broadcast %i0 : i64 to !vector_type_R
  %4 = vector.insert %i1,  %3[1] : i64 into !vector_type_R
  %5 = vector.insert %i2,  %4[2] : i64 into !vector_type_R
  %6 = vector.insert %i3,  %5[3] : i64 into !vector_type_R
  %7 = vector.insert %i4,  %6[4] : i64 into !vector_type_R
  %8 = vector.insert %i5,  %7[5] : i64 into !vector_type_R
  %9 = vector.insert %i10, %8[6] : i64 into !vector_type_R

  %o = vector.broadcast %i1 : i64 to !vector_type_R

  %axpy1 = vector.outerproduct %9, %i2     : !vector_type_R, i64
  %axpy2 = vector.outerproduct %9, %i2, %o : !vector_type_R, i64

  vector.print %axpy1 : !vector_type_R
  vector.print %axpy2 : !vector_type_R
  //
  // axpy operations:
  //
  // CHECK: ( 0, 2, 4, 6, 8, 10, 20 )
  // CHECK: ( 1, 3, 5, 7, 9, 11, 21 )

  return
}
