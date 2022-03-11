// RUN: mlir-opt %s -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!vector_type_A = type vector<8xf32>
!vector_type_B = type vector<8xf32>
!vector_type_C = type vector<8x8xf32>

!vector_type_X = type vector<2xf32>
!vector_type_Y = type vector<3xf32>
!vector_type_Z = type vector<2x3xf32>

!vector_type_R = type vector<7xf32>

func @vector_outerproduct_splat_8x8(%fa: f32, %fb: f32, %fc: f32) -> !vector_type_C {
  %a = vector.splat %fa: !vector_type_A
  %b = vector.splat %fb: !vector_type_B
  %c = vector.splat %fc: !vector_type_C
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
  %f0 = arith.constant 0.0: f32
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %f3 = arith.constant 3.0: f32
  %f4 = arith.constant 4.0: f32
  %f5 = arith.constant 5.0: f32
  %f10 = arith.constant 10.0: f32

  // Simple case, splat scalars into vectors, then take outer product.
  %v = call @vector_outerproduct_splat_8x8(%f1, %f2, %f10)
      : (f32, f32, f32) -> (!vector_type_C)
  vector.print %v : !vector_type_C
  //
  // outer product 8x8:
  //
  // CHECK-COUNT-8: ( 12, 12, 12, 12, 12, 12, 12, 12 )

  // Direct outerproduct on vectors with different size.
  %0 = vector.broadcast %f1 : f32 to !vector_type_X
  %x = vector.insert %f2, %0[1] : f32 into !vector_type_X
  %1 = vector.broadcast %f3 : f32 to !vector_type_Y
  %2 = vector.insert %f4, %1[1] : f32 into !vector_type_Y
  %y = vector.insert %f5, %2[2] : f32 into !vector_type_Y

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

  %3 = vector.broadcast %f0 : f32 to !vector_type_R
  %4 = vector.insert %f1,  %3[1] : f32 into !vector_type_R
  %5 = vector.insert %f2,  %4[2] : f32 into !vector_type_R
  %6 = vector.insert %f3,  %5[3] : f32 into !vector_type_R
  %7 = vector.insert %f4,  %6[4] : f32 into !vector_type_R
  %8 = vector.insert %f5,  %7[5] : f32 into !vector_type_R
  %9 = vector.insert %f10, %8[6] : f32 into !vector_type_R

  %o = vector.broadcast %f1 : f32 to !vector_type_R

  %axpy1 = vector.outerproduct %9, %f2     : !vector_type_R, f32
  %axpy2 = vector.outerproduct %9, %f2, %o : !vector_type_R, f32

  vector.print %axpy1 : !vector_type_R
  vector.print %axpy2 : !vector_type_R
  //
  // axpy operations:
  //
  // CHECK: ( 0, 2, 4, 6, 8, 10, 20 )
  // CHECK: ( 1, 3, 5, 7, 9, 11, 21 )

  return
}
