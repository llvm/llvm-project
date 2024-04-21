// RUN: mlir-opt --verify-diagnostics %s

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType=i16, coefficientModulus=256, polynomialModulus=#my_poly>
module {
  func.func @test_from_tensor_too_large_coeffs() {
    %two = arith.constant 2 : i32
    %coeffs1 = tensor.from_elements %two, %two : tensor<2xi32>
    // expected-error@below {{is too large to fit in the coefficients}}
    %poly = polynomial.from_tensor %coeffs1 : tensor<2xi32> -> !polynomial.polynomial<#ring>
    return
  }
}
