// RUN: mlir-opt %s | FileCheck %s

#my_poly = #polynomial.polynomial<1 + x**1024>
#my_poly_2 = #polynomial.polynomial<2>
#my_poly_3 = #polynomial.polynomial<3x>
#my_poly_4 = #polynomial.polynomial<t**3 + 4t + 2>
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>

!ty = !polynomial.polynomial<#ring1>

// CHECK-LABEL: func @test_types
// CHECK-SAME:  !polynomial.polynomial<
// CHECK-SAME:    #polynomial.ring<
// CHECK-SAME:       coefficientType=i32,
// CHECK-SAME:       coefficientModulus=2837465 : i32,
// CHECK-SAME:       polynomialModulus=#polynomial.polynomial<1 + x**1024>>>
func.func @test_types(%0: !ty) -> !ty {
  return %0 : !ty
}
