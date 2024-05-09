// RUN: mlir-opt --sccp --canonicalize %s | FileCheck %s

// Tests for folding

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#poly_3t = #polynomial.int_polynomial<3t>
#poly_t3_plus_4t_plus_2 = #polynomial.int_polynomial<t**3 + 4t + 2>
#modulus = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256, polynomialModulus=#modulus, primitiveRoot=193>
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

// Test elementwise folding of add
// Test float folding of add
