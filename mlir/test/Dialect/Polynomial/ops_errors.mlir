// RUN: mlir-opt --split-input-file --verify-diagnostics %s

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<#ring>

func.func @test_from_tensor_too_large_coeffs() {
  %two = arith.constant 2 : i32
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi32>
  // expected-error@below {{is too large to fit in the coefficients}}
  // expected-note@below {{rescaled to fit}}
  %poly = polynomial.from_tensor %coeffs1 : tensor<2xi32> -> !ty
  return
}

// -----

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<#ring>

func.func @test_mul_scalar_wrong_type(%arg0: !ty) -> !ty {
  %scalar = arith.constant 2 : i32  // should be i16
  // expected-error@below {{polynomial coefficient type 'i16' does not match scalar type 'i32'}}
  %poly = polynomial.mul_scalar %arg0, %scalar : !ty, i32
  return %poly : !ty
}
