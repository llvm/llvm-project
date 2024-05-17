// RUN: mlir-opt --sccp --canonicalize %s | FileCheck %s

// Tests for folding

#poly_3t = #polynomial.int_polynomial<3t>
#poly_t3_plus_4t_plus_2 = #polynomial.int_polynomial<t**3 + 4t + 2>
#ring = #polynomial.ring<coefficientType=i32>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-LABEL: test_fold_add
// CHECK-NEXT: polynomial.constant {value = #polynomial.int_polynomial<2 + 7x + x**3>}
// CHECK-NEXT: return
func.func @test_fold_add() -> !poly_ty {
  %0 = polynomial.constant {value=#poly_3t} : !poly_ty
  %1 = polynomial.constant {value=#poly_t3_plus_4t_plus_2} : !poly_ty
  %2 = polynomial.add %0, %1 : !poly_ty
  return %2 : !poly_ty
}

// CHECK-LABEL: test_fold_add_elementwise
// CHECK-NEXT: polynomial.constant {value = dense<
// CHECK-SAME:  #polynomial.typed_int_polynomial<type=
// CHECK-SAME:     value = <2 + 7x + x**3>>,
// CHECK-SAME:  #polynomial.typed_int_polynomial<type=
// CHECK-SAME:     value = <2 + 7x + x**3>>,
// CHECK-SAME: ]>}
// CHECK-NEXT: return
#typed_poly1 = #polynomial.typed_int_polynomial<type=!poly_ty, value=<3t>>
#typed_poly2 = #polynomial.typed_int_polynomial<type=!poly_ty, value=<t**3 + 4t + 2>>
!tensor_ty = tensor<2x!poly_ty>
func.func @test_fold_add_elementwise() -> !tensor_ty {
  %0 = polynomial.constant {value=[#typed_poly1, #typed_poly2]} : !tensor_ty
  %1 = polynomial.constant {value=[#typed_poly2, #typed_poly1]} : !tensor_ty
  %2 = polynomial.add %0, %1 : !tensor_ty
  return %2 : !tensor_ty
}


#fpoly_1 = #polynomial.float_polynomial<3.5t>
#fpoly_2 = #polynomial.float_polynomial<1.0t**3 + 1.25t + 2.0>
#fring = #polynomial.ring<coefficientType=f32>
!fpoly_ty = !polynomial.polynomial<ring=#fring>

// CHECK-LABEL: test_fold_add_float
// CHECK-NEXT: polynomial.constant {value = #polynomial.float_polynomial<2 + 4.75x + x**3>}
// CHECK-NEXT: return
func.func @test_fold_add_float() -> !fpoly_ty {
  %0 = polynomial.constant {value=#fpoly_1} : !fpoly_ty
  %1 = polynomial.constant {value=#fpoly_2} : !fpoly_ty
  %2 = polynomial.add %0, %1 : !fpoly_ty
  return %2 : !fpoly_ty
}

// Test elementwise folding of add
