// RUN: mlir-opt %s | FileCheck %s

// This simply tests for syntax.

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#my_poly_2 = #polynomial.int_polynomial<2>
#my_poly_3 = #polynomial.int_polynomial<3x>
#my_poly_4 = #polynomial.int_polynomial<t**3 + 4t + 2>
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>
#ring2 = #polynomial.ring<coefficientType=f32>
#one_plus_x_squared = #polynomial.int_polynomial<1 + x**2>

#ideal = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256, polynomialModulus=#ideal, primitiveRoot=193>
!poly_ty = !polynomial.polynomial<ring=#ring>

#ntt_poly = #polynomial.int_polynomial<-1 + x**8>
#ntt_ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256, polynomialModulus=#ntt_poly, primitiveRoot=31>
!ntt_poly_ty = !polynomial.polynomial<ring=#ntt_ring>

module {
  func.func @test_multiply() -> !polynomial.polynomial<ring=#ring1> {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %five = arith.constant 5 : i16
    %coeffs1 = tensor.from_elements %two, %two, %five : tensor<3xi16>
    %coeffs2 = tensor.from_elements %five, %five, %two : tensor<3xi16>

    %poly1 = polynomial.from_tensor %coeffs1 : tensor<3xi16> -> !polynomial.polynomial<ring=#ring1>
    %poly2 = polynomial.from_tensor %coeffs2 : tensor<3xi16> -> !polynomial.polynomial<ring=#ring1>

    %3 = polynomial.mul %poly1, %poly2 : !polynomial.polynomial<ring=#ring1>

    return %3 : !polynomial.polynomial<ring=#ring1>
  }

  func.func @test_elementwise(%p0 : !polynomial.polynomial<ring=#ring1>, %p1: !polynomial.polynomial<ring=#ring1>) {
    %tp0 = tensor.from_elements %p0, %p1 : tensor<2x!polynomial.polynomial<ring=#ring1>>
    %tp1 = tensor.from_elements %p1, %p0 : tensor<2x!polynomial.polynomial<ring=#ring1>>

    %c = arith.constant 2 : i32
    %mul_const_sclr = polynomial.mul_scalar %tp0, %c : tensor<2x!polynomial.polynomial<ring=#ring1>>, i32

    %add = polynomial.add %tp0, %tp1 : tensor<2x!polynomial.polynomial<ring=#ring1>>
    %sub = polynomial.sub %tp0, %tp1 : tensor<2x!polynomial.polynomial<ring=#ring1>>
    %mul = polynomial.mul %tp0, %tp1 : tensor<2x!polynomial.polynomial<ring=#ring1>>

    return
  }

  func.func @test_to_from_tensor(%p0 : !polynomial.polynomial<ring=#ring1>) {
    %c0 = arith.constant 0 : index
    %two = arith.constant 2 : i16
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi16>
    // CHECK: from_tensor
    %poly = polynomial.from_tensor %coeffs1 : tensor<2xi16> -> !polynomial.polynomial<ring=#ring1>
    // CHECK: to_tensor
    %tensor = polynomial.to_tensor %poly : !polynomial.polynomial<ring=#ring1> -> tensor<1024xi16>

    return
  }

  func.func @test_degree(%p0 : !polynomial.polynomial<ring=#ring1>) {
    %0, %1 = polynomial.leading_term %p0 : !polynomial.polynomial<ring=#ring1> -> (index, i32)
    return
  }

  func.func @test_monomial() {
    %deg = arith.constant 1023 : index
    %five = arith.constant 5 : i16
    %0 = polynomial.monomial %five, %deg : (i16, index) -> !polynomial.polynomial<ring=#ring1>
    return
  }

  func.func @test_monic_monomial_mul() {
    %five = arith.constant 5 : index
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    %1 = polynomial.monic_monomial_mul %0, %five : (!polynomial.polynomial<ring=#ring1>, index) -> !polynomial.polynomial<ring=#ring1>
    return
  }

  func.func @test_constant() {
    %0 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    %1 = polynomial.constant int<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    %2 = polynomial.constant float<1.5 + 0.5 x**2> : !polynomial.polynomial<ring=#ring2>

    // Test verbose fallbacks
    %verb0 = polynomial.constant #polynomial.typed_int_polynomial<1 + x**2> : !polynomial.polynomial<ring=#ring1>
    %verb2 = polynomial.constant #polynomial.typed_float_polynomial<1.5 + 0.5 x**2> : !polynomial.polynomial<ring=#ring2>
    return
  }

  func.func @test_ntt(%0 : !ntt_poly_ty) {
    %1 = polynomial.ntt %0 : !ntt_poly_ty -> tensor<8xi32, #ntt_ring>
    return
  }

  func.func @test_intt(%0 : tensor<8xi32, #ntt_ring>) {
    %1 = polynomial.intt %0 : tensor<8xi32, #ntt_ring> -> !ntt_poly_ty
    return
  }
}
