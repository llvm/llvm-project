// RUN: mlir-opt %s --split-input-file --verify-diagnostics

#my_poly = #polynomial.polynomial<y + x**1024>
// expected-error@below {{polynomials must have one indeterminate, but there were multiple: x, y}}
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>

// -----

// expected-error@below {{expected integer value}}
// expected-error@below {{expected a monomial}}
// expected-error@below {{found invalid integer exponent}}
#my_poly = #polynomial.polynomial<5 + x**f>
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>

// -----

#my_poly = #polynomial.polynomial<5 + x**2 + 3x**2>
// expected-error@below {{parsed polynomial must have unique exponents among monomials}}
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>

// -----

// expected-error@below {{expected + and more monomials, or > to end polynomial attribute}}
#my_poly = #polynomial.polynomial<5 + x**2 7>
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>

// -----

// expected-error@below {{expected a monomial}}
#my_poly = #polynomial.polynomial<5 + x**2 +>
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=2837465, polynomialModulus=#my_poly>


// -----

#my_poly = #polynomial.polynomial<5 + x**2>
// expected-error@below {{coefficientType must specify an integer type}}
#ring1 = #polynomial.ring<coefficientType=f64, coefficientModulus=2837465, polynomialModulus=#my_poly>

// -----

#my_poly = #polynomial.polynomial<5 + x**2>
// expected-error@below {{expected integer value}}
// expected-error@below {{invalid coefficient modulus}}
#ring1 = #polynomial.ring<coefficientType=i32, coefficientModulus=x, polynomialModulus=#my_poly>
