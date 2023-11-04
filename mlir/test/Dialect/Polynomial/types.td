// RUN: mlir-opt %s | FileCheck %s

#my_poly = #polynomial.polynomial<1 + x**1024>
#my_poly_2 = #polynomial.polynomial<2>
#my_poly_3 = #polynomial.polynomial<3x>
#my_poly_4 = #polynomial.polynomial<t**3 + 4t + 2>
#ring1 = #polynomial.ring<ctype=i32, cmod=2837465, ideal=#my_poly>

!ty = !polynomial.polynomial<#ring1>

// CHECK-LABEL: func @test_types
// CHECK-SAME:  !polynomial.polynomial<
// CHECK-SAME:    #polynomial.ring<
// CHECK-SAME:       ctype=i32,
// CHECK-SAME:       cmod=2837465 : i32,
// CHECK-SAME:       ideal=#polynomial.polynomial<1 + x**1024>>>
func.func @test_types(%0: !ty) -> !ty {
  return %0 : !ty
}
