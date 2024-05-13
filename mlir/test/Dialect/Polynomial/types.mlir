// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: func @test_types
// CHECK-SAME:  !polynomial.polynomial<
// CHECK-SAME:    #polynomial.ring<
// CHECK-SAME:       coefficientType=i32,
// CHECK-SAME:       coefficientModulus=2837465 : i32,
// CHECK-SAME:       polynomialModulus=#polynomial.polynomial<1 + x**1024>>>
#my_poly = #polynomial.polynomial<1 + x**1024>
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<#ring1>
func.func @test_types(%0: !ty) -> !ty {
  return %0 : !ty
}


// CHECK-LABEL: func @test_non_x_variable_64_bit
// CHECK-SAME:  !polynomial.polynomial<
// CHECK-SAME:    #polynomial.ring<
// CHECK-SAME:       coefficientType=i64,
// CHECK-SAME:       coefficientModulus=2837465 : i64,
// CHECK-SAME:       polynomialModulus=#polynomial.polynomial<2 + 4x + x**3>>>
#my_poly_2 = #polynomial.polynomial<t**3 + 4t + 2>
#ring2 = #polynomial.ring<coefficientType=i64, coefficientModulus=2837465, polynomialModulus=#my_poly_2>
!ty2 = !polynomial.polynomial<#ring2>
func.func @test_non_x_variable_64_bit(%0: !ty2) -> !ty2 {
  return %0 : !ty2
}


// CHECK-LABEL: func @test_linear_poly
// CHECK-SAME:  !polynomial.polynomial<
// CHECK-SAME:    #polynomial.ring<
// CHECK-SAME:       coefficientType=i32,
// CHECK-SAME:       coefficientModulus=12 : i32,
// CHECK-SAME:       polynomialModulus=#polynomial.polynomial<4x>>
#my_poly_3 = #polynomial.polynomial<4x>
#ring3 = #polynomial.ring<coefficientType=i32, coefficientModulus=12, polynomialModulus=#my_poly_3>
!ty3 = !polynomial.polynomial<#ring3>
func.func @test_linear_poly(%0: !ty3) -> !ty3 {
  return %0 : !ty3
}

// CHECK-LABEL: func @test_negative_leading_1
// CHECK-SAME:  !polynomial.polynomial<
// CHECK-SAME:    #polynomial.ring<
// CHECK-SAME:       coefficientType=i32,
// CHECK-SAME:       coefficientModulus=2837465 : i32,
// CHECK-SAME:       polynomialModulus=#polynomial.polynomial<-1 + x**1024>>>
#my_poly_4 = #polynomial.polynomial<-1 + x**1024>
#ring4 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly_4>
!ty4 = !polynomial.polynomial<#ring4>
func.func @test_negative_leading_1(%0: !ty4) -> !ty4 {
  return %0 : !ty4
}

