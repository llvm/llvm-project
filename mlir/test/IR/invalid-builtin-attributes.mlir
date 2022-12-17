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
  // expected-error@+1 {{expected floating-point elements, but parsed integer}}
  "foo"(){bar = dense<[4]> : tensor<1xf32>} : () -> ()
}

// -----

func.func @elementsattr_toolarge1() -> () {
  "foo"(){bar = dense<[777]> : tensor<1xi8>} : () -> () // expected-error {{integer constant out of range}}
}

// -----

// expected-error@+1 {{parsed zero elements, but type ('tensor<i64>') expected at least 1}}
#attr = dense<> : tensor<i64>

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
  // expected-error @+1 {{expected integer elements, but parsed floating-point}}
  "foo"() {bar = dense<[true, 42.0]> : tensor<2xi1>} : () -> ()
}

// -----

func.func @decimal_int_in_float_tensor() {
  // expected-error @+1 {{expected floating-point elements, but parsed integer}}
  "foo"() {bar = dense<[42, 42.0]> : tensor<2xf32>} : () -> ()
}

// -----

func.func @bool_in_float_tensor() {
  // expected-error @+1 {{expected floating-point elements, but parsed integer}}
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
