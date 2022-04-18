// RUN: mlir-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test AnyAttrOf attributes
//===----------------------------------------------------------------------===//

func @any_attr_of_pass() {
  "test.any_attr_of_i32_str"() {
    // CHECK: attr = 3 : i32
    attr = 3 : i32
  } : () -> ()

  "test.any_attr_of_i32_str"() {
    // CHECK: attr = "string_data"
    attr = "string_data"
  } : () -> ()

  return
}

// -----

func @any_attr_of_fail() {
  // expected-error @+1 {{'test.any_attr_of_i32_str' op attribute 'attr' failed to satisfy constraint: 32-bit signless integer attribute or string attribute}}
  "test.any_attr_of_i32_str"() {
    attr = 3 : i64
  } : () -> ()

  return
}

// -----

//===----------------------------------------------------------------------===//
// Test integer attributes
//===----------------------------------------------------------------------===//

func @int_attrs_pass() {
  "test.int_attrs"() {
    // CHECK: any_i32_attr = 5 : ui32
    any_i32_attr = 5 : ui32,
    // CHECK-SAME: index_attr = 8 : index
    index_attr = 8 : index,
    // CHECK-SAME: si32_attr = 7 : si32
    si32_attr = 7 : si32,
    // CHECK-SAME: ui32_attr = 6 : ui32
    ui32_attr = 6 : ui32
  } : () -> ()

  "test.int_attrs"() {
    // CHECK: any_i32_attr = 5 : si32
    any_i32_attr = 5 : si32,
    index_attr = 8 : index,
    si32_attr = 7 : si32,
    ui32_attr = 6 : ui32
  } : () -> ()

  "test.int_attrs"() {
    // CHECK: any_i32_attr = 5 : i32
    any_i32_attr = 5 : i32,
    index_attr = 8 : index,
    si32_attr = 7 : si32,
    ui32_attr = 6 : ui32
  } : () -> ()

  return
}

// -----

//===----------------------------------------------------------------------===//
// Check that the maximum and minimum integer attribute values are
// representable and preserved during a round-trip.
//===----------------------------------------------------------------------===//

func @int_attrs_pass() {
  "test.in_range_attrs"() {
    // CHECK: attr_00 = -128 : i8
    attr_00 = -128 : i8,
    // CHECK-SAME: attr_01 = 127 : i8
    attr_01 = 127 : i8,
    // CHECK-SAME: attr_02 = -128 : si8
    attr_02 = -128 : si8,
    // CHECK-SAME: attr_03 = 127 : si8
    attr_03 = 127 : si8,
    // CHECK-SAME: attr_04 = 255 : ui8
    attr_04 = 255 : ui8,
    // CHECK-SAME: attr_05 = -32768 : i16
    attr_05 = -32768 : i16,
    // CHECK-SAME: attr_06 = 32767 : i16
    attr_06 = 32767 : i16,
    // CHECK-SAME: attr_07 = -32768 : si16
    attr_07 = -32768 : si16,
    // CHECK-SAME: attr_08 = 32767 : si16
    attr_08 = 32767 : si16,
    // CHECK-SAME: attr_09 = 65535 : ui16
    attr_09 = 65535 : ui16,
    // CHECK-SAME: attr_10 = -2147483647 : i32
    attr_10 = -2147483647 : i32,
    // CHECK-SAME: attr_11 = 2147483646 : i32
    attr_11 = 2147483646 : i32,
    // CHECK-SAME: attr_12 = -2147483647 : si32
    attr_12 = -2147483647 : si32,
    // CHECK-SAME: attr_13 = 2147483646 : si32
    attr_13 = 2147483646 : si32,
    // CHECK-SAME: attr_14 = 4294967295 : ui32
    attr_14 = 4294967295 : ui32,
    // CHECK-SAME: attr_15 = -9223372036854775808 : i64
    attr_15 = -9223372036854775808 : i64,
    // CHECK-SAME: attr_16 = 9223372036854775807 : i64
    attr_16 = 9223372036854775807 : i64,
    // CHECK-SAME: attr_17 = -9223372036854775808 : si64
    attr_17 = -9223372036854775808 : si64,
    // CHECK-SAME: attr_18 = 9223372036854775807 : si64
    attr_18 = 9223372036854775807 : si64,
    // CHECK-SAME: attr_19 = 18446744073709551615 : ui64
    attr_19 = 18446744073709551615 : ui64,
    // CHECK-SAME: attr_20 = 1 : ui1
    attr_20 = 1 : ui1,
    // CHECK-SAME: attr_21 = -1 : si1
    attr_21 = -1 : si1,
    // CHECK-SAME: attr_22 = 79228162514264337593543950335 : ui96
    attr_22 = 79228162514264337593543950335 : ui96,
    // CHECK-SAME: attr_23 = -39614081257132168796771975168 : si96
    attr_23 = -39614081257132168796771975168 : si96
  } : () -> ()

  return
}

// -----

//===----------------------------------------------------------------------===//
// Check that positive values larger than 2^n-1 for signless integers
// are mapped to their negative signed counterpart. This behaviour is
// undocumented in the language specification, but it is what the
// parser currently does.
//===----------------------------------------------------------------------===//

func @int_attrs_pass() {
  "test.i8_attr"() {
    // CHECK: attr_00 = -1 : i8
    attr_00 = 255 : i8,
    // CHECK-SAME: attr_01 = -1 : i16
    attr_01 = 65535 : i16,
    // CHECK-SAME: attr_02 = -1 : i32
    attr_02 = 4294967295 : i32,
    // CHECK-SAME: attr_03 = -1 : i64
    attr_03 = 18446744073709551615 : i64
  } : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Check that i0 is parsed and verified correctly. It can only have value 0.
// We check it explicitly because there are various special cases for it that
// are good to verify.
//===----------------------------------------------------------------------===//

func @int0_attrs_pass() {
  "test.i0_attr"() {
    // CHECK: attr_00 = 0 : i0
    attr_00 = 0 : i0,
    // CHECK: attr_01 = 0 : si0
    attr_01 = 0 : si0,
    // CHECK: attr_02 = 0 : ui0
    attr_02 = 0 : ui0,
    // CHECK: attr_03 = 0 : i0
    attr_03 = 0x0000 : i0,
    // CHECK: attr_04 = 0 : si0
    attr_04 = 0x0000 : si0,
    // CHECK: attr_05 = 0 : ui0
    attr_05 = 0x0000 : ui0
  } : () -> ()
  return
}

// -----

func @int0_attrs_negative_fail() {
  "test.i0_attr"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr_00 = -1 : i0
  } : () -> ()
  return
}

// -----

func @int0_attrs_positive_fail() {
  "test.i0_attr"() {
    // expected-error @+1 {{integer constant out of range for attribute}}
    attr_00 = 1 : i0
  } : () -> ()
  return
}

// -----

func @wrong_int_attrs_signedness_fail() {
  // expected-error @+1 {{'si32_attr' failed to satisfy constraint: 32-bit signed integer attribute}}
  "test.int_attrs"() {
    any_i32_attr = 5 : i32,
    index_attr = 8 : index,
    si32_attr = 7 : ui32,
    ui32_attr = 6 : ui32
  } : () -> ()
  return
}

// -----

func @wrong_int_attrs_signedness_fail() {
  // expected-error @+1 {{'ui32_attr' failed to satisfy constraint: 32-bit unsigned integer attribute}}
  "test.int_attrs"() {
    any_i32_attr = 5 : i32,
    index_attr = 8 : index,
    si32_attr = 7 : si32,
    ui32_attr = 6 : si32
  } : () -> ()
  return
}

// -----

func @wrong_int_attrs_type_fail() {
  // expected-error @+1 {{'any_i32_attr' failed to satisfy constraint: 32-bit integer attribute}}
  "test.int_attrs"() {
    any_i32_attr = 5.0 : f32,
    si32_attr = 7 : si32,
    ui32_attr = 6 : ui32,
    index_attr = 1 : index
  } : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test Non-negative Int Attr
//===----------------------------------------------------------------------===//

func @non_negative_int_attr_pass() {
  // CHECK: test.non_negative_int_attr
  "test.non_negative_int_attr"() {i32attr = 5 : i32, i64attr = 10 : i64} : () -> ()
  // CHECK: test.non_negative_int_attr
  "test.non_negative_int_attr"() {i32attr = 0 : i32, i64attr = 0 : i64} : () -> ()
  return
}

// -----

func @negative_int_attr_fail() {
  // expected-error @+1 {{'i32attr' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative}}
  "test.non_negative_int_attr"() {i32attr = -5 : i32, i64attr = 10 : i64} : () -> ()
  return
}

// -----

func @negative_int_attr_fail() {
  // expected-error @+1 {{'i64attr' failed to satisfy constraint: 64-bit signless integer attribute whose value is non-negative}}
  "test.non_negative_int_attr"() {i32attr = 5 : i32, i64attr = -10 : i64} : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test Positive Int Attr
//===----------------------------------------------------------------------===//

func @positive_int_attr_pass() {
  // CHECK: test.positive_int_attr
  "test.positive_int_attr"() {i32attr = 5 : i32, i64attr = 10 : i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i32attr' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  "test.positive_int_attr"() {i32attr = 0 : i32, i64attr = 5: i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i64attr' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  "test.positive_int_attr"() {i32attr = 5 : i32, i64attr = 0: i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i32attr' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive}}
  "test.positive_int_attr"() {i32attr = -10 : i32, i64attr = 5 : i64} : () -> ()
  return
}

// -----

func @positive_int_attr_fail() {
  // expected-error @+1 {{'i64attr' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive}}
  "test.positive_int_attr"() {i32attr = 5 : i32, i64attr = -10 : i64} : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test TypeArrayAttr
//===----------------------------------------------------------------------===//

func @correct_type_array_attr_pass() {
  // CHECK: test.type_array_attr
  "test.type_array_attr"() {attr = [i32, f32]} : () -> ()
  return
}

// -----

func @non_type_in_type_array_attr_fail() {
  // expected-error @+1 {{'attr' failed to satisfy constraint: type array attribute}}
  "test.type_array_attr"() {attr = [i32, 5 : i64]} : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test StringAttr with custom type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @string_attr_custom_type
func @string_attr_custom_type() {
  // CHECK: "string_data" : !foo.string
  test.string_attr_with_type "string_data" : !foo.string
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test I32EnumAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @allowed_cases_pass
func @allowed_cases_pass() {
  // CHECK: test.i32_enum_attr
  %0 = "test.i32_enum_attr"() {attr = 5: i32} : () -> i32
  // CHECK: test.i32_enum_attr
  %1 = "test.i32_enum_attr"() {attr = 10: i32} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 32-bit signless integer cases: 5, 10}}
  %0 = "test.i32_enum_attr"() {attr = 7: i32} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 32-bit signless integer cases: 5, 10}}
  %0 = "test.i32_enum_attr"() {attr = 5: i64} : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test I64EnumAttr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @allowed_cases_pass
func @allowed_cases_pass() {
  // CHECK: test.i64_enum_attr
  %0 = "test.i64_enum_attr"() {attr = 5: i64} : () -> i32
  // CHECK: test.i64_enum_attr
  %1 = "test.i64_enum_attr"() {attr = 10: i64} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 64-bit signless integer cases: 5, 10}}
  %0 = "test.i64_enum_attr"() {attr = 7: i64} : () -> i32
  return
}

// -----

func @disallowed_case7_fail() {
  // expected-error @+1 {{allowed 64-bit signless integer cases: 5, 10}}
  %0 = "test.i64_enum_attr"() {attr = 5: i32} : () -> i32
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test FloatElementsAttr
//===----------------------------------------------------------------------===//

func @correct_type_pass() {
  "test.float_elements_attr"() {
    // CHECK: scalar_f32_attr = dense<5.000000e+00> : tensor<2xf32>
    // CHECK: tensor_f64_attr = dense<6.000000e+00> : tensor<4x8xf64>
    scalar_f32_attr = dense<5.0> : tensor<2xf32>,
    tensor_f64_attr = dense<6.0> : tensor<4x8xf64>
  } : () -> ()
  return
}

// -----

func @wrong_element_type_pass() {
  // expected-error @+1 {{failed to satisfy constraint: 32-bit float elements attribute of shape [2]}}
  "test.float_elements_attr"() {
    scalar_f32_attr = dense<5.0> : tensor<2xf64>,
    tensor_f64_attr = dense<6.0> : tensor<4x8xf64>
  } : () -> ()
  return
}

// -----

func @correct_type_pass() {
  // expected-error @+1 {{failed to satisfy constraint: 64-bit float elements attribute of shape [4, 8]}}
  "test.float_elements_attr"() {
    scalar_f32_attr = dense<5.0> : tensor<2xf32>,
    tensor_f64_attr = dense<6.0> : tensor<4xf64>
  } : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test StringElementsAttr
//===----------------------------------------------------------------------===//

func @simple_scalar_example() {
  "test.string_elements_attr"() {
    // CHECK: dense<"example">
    scalar_string_attr = dense<"example"> : tensor<2x!unknown<"">>
  } : () -> ()
  return
}

// -----

func @escape_string_example() {
  "test.string_elements_attr"() {
    // CHECK: dense<"new\0Aline">
    scalar_string_attr = dense<"new\nline"> : tensor<2x!unknown<"">>
  } : () -> ()
  return
}

// -----

func @simple_scalar_example() {
  "test.string_elements_attr"() {
    // CHECK: dense<["example1", "example2"]>
    scalar_string_attr = dense<["example1", "example2"]> : tensor<2x!unknown<"">>
  } : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test SymbolRefAttr
//===----------------------------------------------------------------------===//

func @fn() { return }

// CHECK: test.symbol_ref_attr
"test.symbol_ref_attr"() {symbol = @fn} : () -> ()

// -----

// expected-error @+1 {{referencing to a 'func::FuncOp' symbol}}
"test.symbol_ref_attr"() {symbol = @foo} : () -> ()

// -----

//===----------------------------------------------------------------------===//
// Test IntElementsAttr
//===----------------------------------------------------------------------===//

func @correct_int_elements_attr_pass() {
  "test.int_elements_attr"() {
    // CHECK: any_i32_attr = dense<5> : tensor<1x2x3x4xui32>,
    any_i32_attr = dense<5> : tensor<1x2x3x4xui32>,
    i32_attr = dense<5> : tensor<6xi32>
  } : () -> ()

  "test.int_elements_attr"() {
    // CHECK: any_i32_attr = dense<5> : tensor<1x2x3x4xsi32>,
    any_i32_attr = dense<5> : tensor<1x2x3x4xsi32>,
    i32_attr = dense<5> : tensor<6xi32>
  } : () -> ()

  "test.int_elements_attr"() {
    // CHECK: any_i32_attr = dense<5> : tensor<1x2x3x4xi32>,
    any_i32_attr = dense<5> : tensor<1x2x3x4xi32>,
    i32_attr = dense<5> : tensor<6xi32>
  } : () -> ()

  "test.index_elements_attr"() {
    // CHECK: any_index_attr = dense<5> : tensor<1x2x3x4xindex>,
    any_index_attr = dense<5> : tensor<1x2x3x4xindex>,
    index_attr = dense<5> : tensor<6xindex>
  } : () -> ()

  "test.hex_index_elements_attr"() {
    // CHECK: hex_index_attr = dense<"0x00000C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000080C0000000000001A150000000000005078000000000000B659010000000000ECBC010000000000FEC5010000000000342902000000000046320200000000007C950200000000008E9E020000000000C401030000000000D60A0300000000000C6E0300000000001E7703000000000054DA03000000000066E30300000000009C46040000000000AE4F040000000000E4B2040000000000F6BB0400000000002C1F050000000000628100000000000098E40000000000000E0C00000000000020150000000000005678000000000000BC59010000000000F2BC01000000000004C60100000000003A290200000000004C320200000000008295020000000000949E020000000000CA01030000000000DC0A030000000000126E03000000000024770300000000005ADA0300000000006CE3030000000000A246040000000000B44F040000000000EAB2040000000000FCBB040000000000321F05000000000068810000000000009EE40000000000"> : tensor<23x5xindex>
    hex_index_attr = dense<"0x00000C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000080C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000100C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000180C000000000000080C0000000000001A150000000000005078000000000000B659010000000000ECBC010000000000FEC5010000000000342902000000000046320200000000007C950200000000008E9E020000000000C401030000000000D60A0300000000000C6E0300000000001E7703000000000054DA03000000000066E30300000000009C46040000000000AE4F040000000000E4B2040000000000F6BB0400000000002C1F050000000000628100000000000098E40000000000000E0C00000000000020150000000000005678000000000000BC59010000000000F2BC01000000000004C60100000000003A290200000000004C320200000000008295020000000000949E020000000000CA01030000000000DC0A030000000000126E03000000000024770300000000005ADA0300000000006CE3030000000000A246040000000000B44F040000000000EAB2040000000000FCBB040000000000321F05000000000068810000000000009EE40000000000"> : tensor<23x5xindex>
  } : () -> ()

  return
}

// -----

func @wrong_int_elements_attr_type_fail() {
  // expected-error @+1 {{'any_i32_attr' failed to satisfy constraint: 32-bit integer elements attribute}}
  "test.int_elements_attr"() {
    any_i32_attr = dense<5.0> : tensor<1x2x3x4xf32>,
    i32_attr = dense<5> : tensor<6xi32>
  } : () -> ()
  return
}

// -----

func @wrong_int_elements_attr_signedness_fail() {
  // expected-error @+1 {{'i32_attr' failed to satisfy constraint: 32-bit signless integer elements attribute}}
  "test.int_elements_attr"() {
    any_i32_attr = dense<5> : tensor<1x2x3x4xi32>,
    i32_attr = dense<5> : tensor<6xsi32>
  } : () -> ()
  return
}

// -----

//===----------------------------------------------------------------------===//
// Test Ranked IntElementsAttr
//===----------------------------------------------------------------------===//

func @correct_type_pass() {
  "test.ranked_int_elements_attr"() {
    // CHECK: matrix_i64_attr = dense<6> : tensor<4x8xi64>
    // CHECK: vector_i32_attr = dense<5> : tensor<2xi32>
    matrix_i64_attr = dense<6> : tensor<4x8xi64>,
    vector_i32_attr = dense<5> : tensor<2xi32>
  } : () -> ()
  return
}

// -----

func @wrong_element_type_fail() {
  // expected-error @+1 {{failed to satisfy constraint: 32-bit signless int elements attribute of shape [2]}}
  "test.ranked_int_elements_attr"() {
    matrix_i64_attr = dense<6> : tensor<4x8xi64>,
    vector_i32_attr = dense<5> : tensor<2xi64>
  } : () -> ()
  return
}

// -----

func @wrong_shape_fail() {
  // expected-error @+1 {{failed to satisfy constraint: 64-bit signless int elements attribute of shape [4, 8]}}
  "test.ranked_int_elements_attr"() {
    matrix_i64_attr = dense<6> : tensor<4xi64>,
    vector_i32_attr = dense<5> : tensor<2xi32>
  } : () -> ()
  return
}

// -----

func @wrong_shape_fail() {
  // expected-error @+1 {{failed to satisfy constraint: 32-bit signless int elements attribute of shape [2]}}
  "test.ranked_int_elements_attr"() {
    matrix_i64_attr = dense<6> : tensor<4x8xi64>,
    vector_i32_attr = dense<5> : tensor<i32>
  } : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test StructAttr
//===----------------------------------------------------------------------===//

// -----

func @missing_fields() {
  // expected-error @+1 {{failed to satisfy constraint: DictionaryAttr with field(s): 'some_field', 'some_other_field' (each field having its own constraints)}}
  "test.struct_attr"() {the_struct_attr = {}} : () -> ()
  return
}

// -----

func @erroneous_fields() {
  // expected-error @+1 {{failed to satisfy constraint: DictionaryAttr with field(s): 'some_field', 'some_other_field' (each field having its own constraints)}}
  "test.struct_attr"() {the_struct_attr = {some_field = 1 : i8, some_other_field = 1}} : () -> ()
  return
}

// -----

// expected-error @+1 {{invalid dialect namespace '"string with space"'}}
#invalid_dialect = opaque<"string with space", "0xDEADBEEF"> : tensor<100xi32>
