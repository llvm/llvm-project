// RUN: mlir-opt --split-input-file --verify-diagnostics %s

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16>
!ty = !polynomial.polynomial<ring=#ring>

func.func @test_from_tensor_too_large_coeffs() {
  %two = arith.constant 2 : i32
  %coeffs1 = tensor.from_elements %two, %two : tensor<2xi32>
  // expected-error@below {{is too large to fit in the coefficients}}
  // expected-note@below {{rescaled to fit}}
  %poly = polynomial.from_tensor %coeffs1 : tensor<2xi32> -> !ty
  return
}

// -----

#my_poly = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256:i32, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<ring=#ring>
func.func @test_from_tensor_wrong_tensor_type() {
  %two = arith.constant 2 : i32
  %coeffs1 = tensor.from_elements %two, %two, %two, %two, %two : tensor<5xi32>
  // expected-error@below {{input type 'tensor<5xi32>' does not match output type '!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 256 : i32, polynomialModulus = <1 + x**4>>>'}}
  // expected-note@below {{at most the degree of the polynomialModulus of the output type's ring attribute}}
  %poly = polynomial.from_tensor %coeffs1 : tensor<5xi32> -> !ty
  return
}

// -----

#my_poly = #polynomial.int_polynomial<1 + x**4>
#ring = #polynomial.ring<coefficientType=i32, coefficientModulus=256:i32, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<ring=#ring>
func.func @test_to_tensor_wrong_output_tensor_type(%arg0 : !ty) {
  // expected-error@below {{input type '!polynomial.polynomial<ring = <coefficientType = i32, coefficientModulus = 256 : i32, polynomialModulus = <1 + x**4>>>' does not match output type 'tensor<5xi32>'}}
  // expected-note@below {{at most the degree of the polynomialModulus of the input type's ring attribute}}
  %tensor = polynomial.to_tensor %arg0 : !ty -> tensor<5xi32>
  return
}

// -----

#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256:i32, polynomialModulus=#my_poly>
!ty = !polynomial.polynomial<ring=#ring>

func.func @test_mul_scalar_wrong_type(%arg0: !ty) -> !ty {
  %scalar = arith.constant 2 : i32  // should be i16
  // expected-error@below {{polynomial coefficient type 'i16' does not match scalar type 'i32'}}
  %poly = polynomial.mul_scalar %arg0, %scalar : !ty, i32
  return %poly : !ty
}

// -----

#my_poly = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256:i16, polynomialModulus=#my_poly>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-NOT: @test_invalid_ntt
// CHECK-NOT: polynomial.ntt
func.func @test_invalid_ntt(%0 : !poly_ty) {
  // expected-error@below {{expects a ring encoding to be provided to the tensor}}
  %1 = polynomial.ntt %0 {root=#polynomial.primitive_root<value=31:i32, degree=8:index>} : !poly_ty -> tensor<1024xi32>
  return
}

// -----

#my_poly = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256:i16, polynomialModulus=#my_poly>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-NOT: @test_invalid_ntt
// CHECK-NOT: polynomial.ntt
func.func @test_invalid_ntt(%0 : !poly_ty) {
  // expected-error@below {{tensor encoding is not a ring attribute}}
  %1 = polynomial.ntt %0 {root=#polynomial.primitive_root<value=31:i32, degree=8:index>} : !poly_ty -> tensor<1024xi32, #my_poly>
  return
}

// -----

#my_poly = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256:i16, polynomialModulus=#my_poly>
#ring1 = #polynomial.ring<coefficientType=i16, coefficientModulus=257:i16, polynomialModulus=#my_poly>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-NOT: @test_invalid_intt
// CHECK-NOT: polynomial.intt
func.func @test_invalid_intt(%0 : tensor<1024xi32, #ring1>) {
  // expected-error@below {{not equivalent to the polynomial ring}}
  %1 = polynomial.intt %0 {root=#polynomial.primitive_root<value=31:i32, degree=8:index>} : tensor<1024xi32, #ring1> -> !poly_ty
  return
}

// -----

#my_poly = #polynomial.int_polynomial<-1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256:i16, polynomialModulus=#my_poly>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-NOT: @test_invalid_intt
// CHECK-NOT: polynomial.intt
func.func @test_invalid_intt(%0 : tensor<1025xi32, #ring>) {
  // expected-error@below {{does not match output type}}
  // expected-note@below {{exactly the degree of the polynomialModulus of the polynomial type's ring attribute}}
  %1 = polynomial.intt %0 {root=#polynomial.primitive_root<value=31:i32, degree=8:index>} : tensor<1025xi32, #ring> -> !poly_ty
  return
}

// -----

#my_poly = #polynomial.int_polynomial<-1 + x**8>
// A valid root is 31
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256:i16, polynomialModulus=#my_poly>
!poly_ty = !polynomial.polynomial<ring=#ring>

// CHECK-NOT: @test_invalid_intt
// CHECK-NOT: polynomial.intt
func.func @test_invalid_intt(%0 : tensor<8xi32, #ring>) {
  // expected-error@below {{provided root 32 is not a primitive root of unity mod 256, with the specified degree 8}}
  %1 = polynomial.intt %0 {root=#polynomial.primitive_root<value=32:i16, degree=8:index>} : tensor<8xi32, #ring> -> !poly_ty
  return
}
