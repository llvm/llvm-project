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

#my_poly = #polynomial.polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<#ring>
func.func @test_from_tensor_wrong_tensor_type() {
  %two = arith.constant 2 : i32
  %coeffs1 = tensor.from_elements %two, %two, %two, %two, %two : tensor<5xi32>
  // expected-error@below {{input type 'tensor<5xi32>' does not match output type '!polynomial.polynomial<#polynomial.ring<coefficientType=i32, coefficientModulus=256 : i32, polynomialModulus=#polynomial.polynomial<1 + x**4>>>'}}
  // expected-note@below {{at most the degree of the polynomialModulus of the output type's ring attribute}}
  %poly = polynomial.from_tensor %coeffs1 : tensor<5xi32> -> !ty
  return
}

// -----

#my_poly = #polynomial.polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<#ring>
func.func @test_to_tensor_wrong_output_tensor_type(%arg0 : !ty) {
  // expected-error@below {{input type '!polynomial.polynomial<#polynomial.ring<coefficientType=i32, coefficientModulus=256 : i32, polynomialModulus=#polynomial.polynomial<1 + x**4>>>' does not match output type 'tensor<5xi32>'}}
  // expected-note@below {{at most the degree of the polynomialModulus of the input type's ring attribute}}
  %tensor = polynomial.to_tensor %arg0 : !ty -> tensor<5xi32>
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
