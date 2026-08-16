// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

func.func @elementsattr_non_tensor_type() -> () {
  "foo"(){bar = dense<[4]> : i32} : () -> () // expected-error {{elements literal must be a shaped type}}
}

// -----

func.func @elementsattr_non_ranked() -> () {
  "foo"(){bar = dense<[4]> : tensor<?xi32>} : () -> () // expected-error {{elements literal type must have static shape}}
}

// -----

func.func @elementsattr_shape_mismatch() -> () {
  "foo"(){bar = dense<[4]> : tensor<5xi32>} : () -> () // expected-error {{inferred shape of elements literal ([1]) does not match type ([5])}}
}

// -----

func.func @elementsattr_invalid() -> () {
  "foo"(){bar = dense<[4, [5]]> : tensor<2xi32>} : () -> () // expected-error {{tensor literal is invalid; ranks are not consistent between elements}}
}

// -----

func.func @elementsattr_badtoken() -> () {
  "foo"(){bar = dense<[tf_opaque]> : tensor<1xi32>} : () -> () // expected-error {{expected element literal of primitive type}}
}

// -----

func.func @elementsattr_floattype1() -> () {
  // expected-error@+1 {{expected integer elements, but parsed floating-point}}
  "foo"(){bar = dense<[4.0]> : tensor<1xi32>} : () -> ()
}

// -----

func.func @elementsattr_floattype1() -> () {
  // expected-error@+1 {{expected integer elements, but parsed floating-point}}
  "foo"(){bar = dense<4.0> : tensor<i32>} : () -> ()
}

// -----

func.func @elementsattr_floattype2() -> () {
  // expected-error@below {{unexpected decimal integer literal for a floating point value}}
  // expected-note@below {{add a trailing dot to make the literal a float}}
  "foo"(){bar = dense<[4]> : tensor<1xf32>} : () -> ()
}

// -----

// The diagnostic must point at the float literal, not the following token
// (the `}` is on the next line, exposing a stale location).
func.func @float_attr_non_float_type() -> () {
  // expected-error@below {{floating point value not valid for specified type}}
  "foo"(){bar = 1.0 : i32
  } : () -> ()
}

// -----

func.func @elementsattr_toolarge1() -> () {
  "foo"(){bar = dense<[777]> : tensor<1xi8>} : () -> () // expected-error {{integer constant out of range}}
}

// -----

// expected-error@+1 {{parsed zero elements, but type ('tensor<i64>') expected at least 1}}
#attr = dense<> : tensor<i64>

// -----

// expected-error@+1 {{parsed 1 elements, but type ('complex<i64>') expected 2 elements}}
#attr = dense<0> : tensor<2xcomplex<i64>>

// -----

// expected-error@+1 {{parsed 2 elements, but type ('tensor<2xcomplex<i64>>') expected 4 elements}}
#attr = dense<[0, 1]> : tensor<2xcomplex<i64>>

// -----

// expected-error@+1 {{parsed 3 elements, but type ('tensor<2xcomplex<i64>>') expected 4 elements}}
#attr = dense<[0, (0, 1)]> : tensor<2xcomplex<i64>>

// -----

func.func @elementsattr_toolarge2() -> () {
  "foo"(){bar = dense<[-777]> : tensor<1xi8>} : () -> () // expected-error {{integer constant out of range}}
}

// -----

"foo"(){bar = dense<[()]> : tensor<complex<i64>>} : () -> () // expected-error {{expected element literal of primitive type}}

// -----

"foo"(){bar = dense<[(10)]> : tensor<complex<i64>>} : () -> () // expected-error {{expected ',' between complex elements}}

// -----

"foo"(){bar = dense<[(10,)]> : tensor<complex<i64>>} : () -> () // expected-error {{expected element literal of primitive type}}

// -----

"foo"(){bar = dense<[(10,10]> : tensor<complex<i64>>} : () -> () // expected-error {{expected ')' after complex elements}}

// -----

func.func @mi() {
  // expected-error @+1 {{expected element literal of primitive type}}
  "fooi64"(){bar = sparse<vector<1xi64>,[,[,1]

// -----

func.func @invalid_tensor_literal() {
  // expected-error @+1 {{expected 1-d tensor for sparse element values}}
  "foof16"(){bar = sparse<[[0, 0, 0]],  [[-2.0]]> : vector<1x1x1xf16>} : () -> ()

// -----

func.func @invalid_tensor_literal() {
  // expected-error @+1 {{expected element literal of primitive type}}
  "fooi16"(){bar = sparse<[[1, 1, 0], [0, 1, 0], [0,, [[0, 0, 0]], [-2.0]> : tensor<2x2x2xi16>} : () -> ()

// -----

func.func @invalid_tensor_literal() {
  // expected-error @+1 {{sparse index #0 is not contained within the value shape, with index=[1, 1], and type='tensor<1x1xi16>'}}
  "fooi16"(){bar = sparse<1, 10> : tensor<1x1xi16>} : () -> ()

// -----

func.func @invalid_sparse_indices() {
  // expected-error @+1 {{expected integer elements, but parsed floating-point}}
  "foo"(){bar = sparse<0.5, 1> : tensor<1xi16>} : () -> ()
}

// -----

func.func @invalid_sparse_values() {
  // expected-error @+1 {{expected integer elements, but parsed floating-point}}
  "foo"(){bar = sparse<0, 1.1> : tensor<1xi16>} : () -> ()
}

// -----

func.func @hexadecimal_float_leading_minus() {
  // expected-error @+1 {{hexadecimal float literal should not have a leading minus}}
  "foo"() {value = -0x7fff : f16} : () -> ()
}

// -----

func.func @hexadecimal_float_literal_overflow() {
  // expected-error @+1 {{hexadecimal float constant out of range for type}}
  "foo"() {value = 0xffffffff : f16} : () -> ()
}

// -----

func.func @decimal_float_literal() {
  // expected-error @+2 {{unexpected decimal integer literal for a floating point value}}
  // expected-note @+1 {{add a trailing dot to make the literal a float}}
  "foo"() {value = 42 : f32} : () -> ()
}

// -----

func.func @float_in_int_tensor() {
  // expected-error @+1 {{expected integer elements, but parsed floating-point}}
  "foo"() {bar = dense<[42.0, 42]> : tensor<2xi32>} : () -> ()
}

// -----

func.func @float_in_bool_tensor() {
  // expected-error@below {{expected integer elements, but parsed floating-point}}
  "foo"() {bar = dense<[true, 42.0]> : tensor<2xi1>} : () -> ()
}

// -----

func.func @decimal_int_in_float_tensor() {
  // expected-error@below {{unexpected decimal integer literal for a floating point value}}
  // expected-note@below {{add a trailing dot to make the literal a float}}
  "foo"() {bar = dense<[42, 42.0]> : tensor<2xf32>} : () -> ()
}

// -----

func.func @bool_in_float_tensor() {
  // expected-error @+1 {{expected floating point literal}}
  "foo"() {bar = dense<[42.0, true]> : tensor<2xf32>} : () -> ()
}

// -----

func.func @hexadecimal_float_leading_minus_in_tensor() {
  // expected-error @+1 {{hexadecimal float literal should not have a leading minus}}
  "foo"() {bar = dense<-0x7FFFFFFF> : tensor<2xf32>} : () -> ()
}

// -----

// Check that we report an error when a value could be parsed, but does not fit
// into the specified type.
func.func @hexadecimal_float_too_wide_for_type_in_tensor() {
  // expected-error @+1 {{hexadecimal float constant out of range for type}}
  "foo"() {bar = dense<0x7FF0000000000000> : tensor<2xf32>} : () -> ()
}

// -----

// Check that we report an error when a value is too wide to be parsed.
func.func @hexadecimal_float_too_wide_in_tensor() {
  // expected-error @+1 {{hexadecimal float constant out of range for type}}
  "foo"() {bar = dense<0x7FFFFFF0000000000000> : tensor<2xf32>} : () -> ()
}

// -----

func.func @integer_too_wide_in_tensor() {
  // expected-error @+1 {{integer constant out of range for type}}
  "foo"() {bar = dense<0xFFFFFFFFFFFFFF> : tensor<2xi16>} : () -> ()
}

// -----

func.func @bool_literal_in_non_bool_tensor() {
  // expected-error @+1 {{expected i1 type for 'true' or 'false' values}}
  "foo"() {bar = dense<true> : tensor<2xi16>} : () -> ()
}

// -----

func.func @negative_value_in_unsigned_int_attr() {
  // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
  "foo"() {bar = -5 : ui32} : () -> ()
}

// -----

func.func @negative_value_in_unsigned_vector_attr() {
  // expected-error @+1 {{expected unsigned integer elements, but parsed negative value}}
  "foo"() {bar = dense<[5, -5]> : vector<2xui32>} : () -> ()
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -129 : i8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 256 : i8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -129 : si8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 129 : si8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 256 : ui8
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -32769 : i16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 65536 : i16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -32769 : si16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 32768 : si16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 65536: ui16
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -2147483649 : i32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 4294967296 : i32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -2147483649 : si32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 2147483648 : si32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 4294967296 : ui32
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -9223372036854775809 : i64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 18446744073709551616 : i64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = -9223372036854775809 : si64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 9223372036854775808 : si64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{negative integer literal not valid for unsigned integer type}}
    attr = -1 : ui64
  } : () -> ()
  return
}

// -----

func.func @large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 18446744073709551616 : ui64
  } : () -> ()
  return
}

// -----

func.func @really_large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 79228162514264337593543950336 : ui96
  } : () -> ()
  return
}

// -----

func.func @really_large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 79228162514264337593543950336 : i96
  } : () -> ()
  return
}

// -----

func.func @really_large_bound() {
  "test.out_of_range_attribute"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr = 39614081257132168796771975168 : si96
  } : () -> ()
  return
}

// -----

func.func @duplicate_dictionary_attr_key() {
  // expected-error @+1 {{duplicate key 'a' in dictionary attribute}}
  "foo.op"() {a, a} : () -> ()
}

// -----

// expected-error@+1 {{expected ',' or ']'}}
"f"() { b = [@m:

// -----

"       // expected-error {{expected}}
"

// -----

// expected-error@+1 {{expected '"' in string literal}}
"J// -----

"       // expected-error {{expected}}

// -----

// expected-error@+1 {{expected '<' after 'dense_resource'}}
#attr = dense_resource>

// -----

// expected-error@+1 {{expected '>'}}
#attr = dense_resource<resource

// -----

// expected-error@+1 {{expected ':'}}
#attr = dense_resource<resource>

// -----

// expected-error@+1 {{`dense_resource` expected a shaped type}}
#attr = dense_resource<resource> : i32

// -----

// expected-error@below {{expected '<' after 'array'}}
#attr = array

// -----

// expected-error@below {{expected integer or float type}}
#attr = array<vector<i32>>

// -----

// expected-error@below {{element type bitwidth must be a multiple of 8}}
#attr = array<i7>

// -----

// expected-error@below {{expected ':' after dense array type}}
#attr = array<i8)

// -----

// expected-error@below {{expected '>' to close an array attribute}}
#attr = array<i8: 1)

// -----

// expected-error@below {{expected i1 type for 'true' or 'false' values}}
#attr = array<i8: true>

// -----

// expected-error@below {{expected 'true' or 'false' values for i1 type}}
#attr = array<i1: 0>

// -----

// expected-error@below {{expected '[' after 'distinct'}}
#attr = distinct<

// -----

// expected-error@below {{expected distinct ID}}
#attr = distinct[i8

// -----

// expected-error@below {{expected an unsigned 64-bit integer}}
#attr = distinct[0xAAAABBBBEEEEFFFF1]

// -----

// expected-error@below {{expected ']' to close distinct ID}}
#attr = distinct[8)

// -----

// expected-error@below {{expected '<' after distinct ID}}
#attr = distinct[8](

// -----

// expected-error@below {{expected attribute}}
#attr = distinct[8]<attribute

// -----

// expected-error@below {{expected '>' to close distinct attribute}}
#attr = distinct[8]<@foo]

// -----

#attr = distinct[0]<42 : i32>
// expected-error@below {{referenced attribute does not match previous definition: 42 : i32}}
#attr1 = distinct[0]<43 : i32>

// -----

// Make sure the error is not printed on the return.
func.func @print_error_on_correct_line() {
  %0 = arith.constant
    // expected-error@below {{elements literal must be a shaped type}} 
    dense<[3]> : i32
  return
}

// -----

// Make sure the error is not printed on the return.
func.func @print_error_on_correct_line() {
  %0 = arith.constant 
    // expected-error@below {{elements literal must be a shaped type}}
    sparse<
     [
       [0, 1, 2, 3],
       [1, 1, 2, 3],
       [1, 2, 2, 3],
       [1, 2, 3, 4]
     ],
     [1, 1, 1, 1] > : i32
  return
}

// -----

// Make sure the error is not printed on the return.
func.func @print_error_on_correct_line() {
  %0 = arith.constant 
    // expected-error@below {{elements literal must be a shaped type}}
    sparse <> : i32
  return
}

// -----

// Prevent assertions when parsing a dense attribute expected to be a string 
// but encountering a different type. 
func.func @expect_to_parse_literal() {
  // expected-error@below {{expected string token, got 23}}
  %0 = arith.constant dense<[23]> : tensor<1x!unknown<>>
  return
}

// -----

func.func @hex_float_without_exponent() {
  // expected-error@below {{expected binary exponent in hexadecimal floating point literal}}
  %0 = arith.constant 0x1.8 : f64
  return
}

// -----

func.func @hex_float_missing_exponent_digits() {
  // expected-error@below {{expected binary exponent in hexadecimal floating point literal}}
  %0 = arith.constant 0x1.0p : f64
  return
}

// -----

func.func @unclosed_nan_literal() {
  // expected-error@below {{expected ')' in NaN literal}}
  %0 = arith.constant +nan(0x1 : f32
  return
}

// -----

func.func @invalid_nan_payload() {
  // expected-error@below {{invalid floating point literal}}
  %0 = arith.constant +nan(0xZZ) : f32
  return
}

// -----

func.func @inf_on_integer_type() {
  // expected-error@below {{floating point value not valid for specified type}}
  %0 = arith.constant +inf : i32
  return
}

// -----

func.func @doubly_signed_inf() {
  // Doubly-signed literal: a '-' token before an already-signed inf/NaN.
  // expected-error@below {{floating point literal has more than one sign}}
  %0 = arith.constant -+inf : f64
  return
}

// -----

func.func @inf_on_type_without_inf() {
  // f4E2M1FN has no Inf encoding: reject, do not crash APFloat.
  // expected-error@below {{floating point type does not support infinity}}
  %0 = arith.constant +inf : f4E2M1FN
  return
}

// -----

func.func @nan_on_type_without_nan() {
  // f6E2M3FN has no NaN encoding.
  // expected-error@below {{floating point type does not support NaN}}
  %0 = arith.constant +qnan : f6E2M3FN
  return
}

// -----

func.func @inf_on_nan_only_type() {
  // f8E4M3FN has NaN but no Inf.
  // expected-error@below {{floating point type does not support infinity}}
  %0 = arith.constant +inf : f8E4M3FN
  return
}

// -----

func.func @negative_on_unsigned_type() {
  // f8E8M0FNU is unsigned: reject negatives, do not crash the printer.
  // expected-error@below {{floating point type does not support negative values}}
  %0 = arith.constant -1.0 : f8E8M0FNU
  return
}

// -----

func.func @signaling_nan_on_all_ones_type() {
  // f8E4M3FN (AllOnes) has a single NaN with no signaling bit.
  // expected-error@below {{floating point type does not support signaling NaN}}
  %0 = arith.constant +snan(0x1) : f8E4M3FN
  return
}

// -----

func.func @payload_nan_on_all_ones_type() {
  // f8E4M3FN (AllOnes) has a single NaN with no payload.
  // expected-error@below {{floating point type does not support NaN payload}}
  %0 = arith.constant +nan(0x1) : f8E4M3FN
  return
}

// -----

func.func @signaling_nan_on_negative_zero_type() {
  // f8E4M3FNUZ (NegativeZero) has a single NaN with no signaling bit.
  // expected-error@below {{floating point type does not support signaling NaN}}
  %0 = arith.constant +snan(0x1) : f8E4M3FNUZ
  return
}

// -----

func.func @payload_nan_on_negative_zero_type() {
  // f8E4M3FNUZ (NegativeZero) has a single NaN with no payload.
  // expected-error@below {{floating point type does not support NaN payload}}
  %0 = arith.constant -nan(0x1) : f8E4M3FNUZ
  return
}

// -----

func.func @positive_qnan_on_negative_zero_type() {
  // f8E4M3FNUZ (NegativeZero) has only a negative NaN.
  // expected-error@below {{floating point type only supports negative NaN}}
  %0 = arith.constant +qnan : f8E4M3FNUZ
  return
}

// -----

func.func @positive_nan_on_negative_zero_type() {
  // f8E4M3FNUZ (NegativeZero) has only a negative NaN.
  // expected-error@below {{floating point type only supports negative NaN}}
  %0 = arith.constant +nan : f8E4M3FNUZ
  return
}

// -----

func.func @negative_nan_on_unsigned_type() {
  // f8E8M0FNU is unsigned: a negative NaN is rejected like any negative value.
  // expected-error@below {{floating point type does not support negative values}}
  %0 = arith.constant -nan : f8E8M0FNU
  return
}

// -----

func.func @signaling_nan_on_unsigned_type() {
  // f8E8M0FNU (AllOnes) has a single NaN with no signaling bit.
  // expected-error@below {{floating point type does not support signaling NaN}}
  %0 = arith.constant +snan : f8E8M0FNU
  return
}

// -----

func.func @payload_nan_on_unsigned_type() {
  // f8E8M0FNU (AllOnes) has a single NaN with no payload.
  // expected-error@below {{floating point type does not support NaN payload}}
  %0 = arith.constant +nan(0x1) : f8E8M0FNU
  return
}
