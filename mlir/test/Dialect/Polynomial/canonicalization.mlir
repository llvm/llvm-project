// RUN: mlir-opt -canonicalize %s | FileCheck %s
#ntt_poly = #polynomial.int_polynomial<-1 + x**8>
#ntt_ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256, polynomialModulus=#ntt_poly>
#root = #polynomial.primitive_root<value=31:i32, degree=8:index>
!ntt_poly_ty = !polynomial.polynomial<ring=#ntt_ring>
!tensor_ty = tensor<8xi32, #ntt_ring>

// CHECK-LABEL: @test_canonicalize_intt_after_ntt
// CHECK: (%[[P:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_intt_after_ntt(%p0 : !ntt_poly_ty) -> !ntt_poly_ty {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: %[[RESULT:.+]] = polynomial.add %[[P]], %[[P]]  : [[T]]
  %t0 = polynomial.ntt %p0 {root=#root} : !ntt_poly_ty -> !tensor_ty
  %p1 = polynomial.intt %t0 {root=#root} : !tensor_ty -> !ntt_poly_ty
  %p2 = polynomial.add %p1, %p1 : !ntt_poly_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %p2 : !ntt_poly_ty
}

// CHECK-LABEL: @test_canonicalize_ntt_after_intt
// CHECK: (%[[X:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_ntt_after_intt(%t0 : !tensor_ty) -> !tensor_ty {
  // CHECK-NOT: polynomial.intt
  // CHECK-NOT: polynomial.ntt
  // CHECK: %[[RESULT:.+]] = arith.addi %[[X]], %[[X]] : [[T]]
  %p0 = polynomial.intt %t0 {root=#root} : !tensor_ty -> !ntt_poly_ty
  %t1 = polynomial.ntt %p0 {root=#root} : !ntt_poly_ty -> !tensor_ty
  %t2 = arith.addi %t1, %t1 : !tensor_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %t2 : !tensor_ty
}

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256:i32, polynomialModulus=#cycl_2048>
!sub_ty = !polynomial.polynomial<ring=#ring>

// CHECK-LABEL: test_canonicalize_sub
// CHECK-SAME: (%[[p0:.*]]: [[T:.*]], %[[p1:.*]]: [[T]]) -> [[T]] {
func.func @test_canonicalize_sub(%poly0 : !sub_ty, %poly1 : !sub_ty) -> !sub_ty {
  %0 = polynomial.sub %poly0, %poly1  : !sub_ty
  // CHECK: %[[minus_one:.+]] = arith.constant -1 : i32
  // CHECK: %[[p1neg:.+]] = polynomial.mul_scalar %[[p1]], %[[minus_one]]
  // CHECK: [[ADD:%.+]] = polynomial.add %[[p0]], %[[p1neg]]
  return %0 : !sub_ty
}

