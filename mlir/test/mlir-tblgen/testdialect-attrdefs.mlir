// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func private @compoundA()
// CHECK-SAME: #test.cmpnd_a<1, !test.smpla, [5, 6]>
func.func private @compoundA() attributes {foo = #test.cmpnd_a<1, !test.smpla, [5, 6]>}

// CHECK: test.result_has_same_type_as_attr #test.attr_with_self_type_param : i32 -> i32
%a = test.result_has_same_type_as_attr #test.attr_with_self_type_param : i32 -> i32

// CHECK: test.result_has_same_type_as_attr #test<attr_with_type_builder 10 : i16> : i16 -> i16
%b = test.result_has_same_type_as_attr #test<attr_with_type_builder 10 : i16> -> i16

// CHECK-LABEL: @qualifiedAttr()
// CHECK-SAME: #test.cmpnd_nested_outer_qual<i #test.cmpnd_nested_inner<42 <1, !test.smpla, [5, 6]>>>
func.func private @qualifiedAttr() attributes {foo = #test.cmpnd_nested_outer_qual<i #test.cmpnd_nested_inner<42 <1, !test.smpla, [5, 6]>>>}

// CHECK-LABEL: @overriddenAttr
// CHECK-SAME: foo = 5 : index
func.func private @overriddenAttr() attributes {
  foo = #test.override_builder<5>
}

// CHECK-LABEL: @decimalIntegerShapeEmpty
// CHECK-SAME: foo = #test.decimal_shape<>
func.func private @decimalIntegerShapeEmpty() attributes {
  foo = #test.decimal_shape<>
}

// CHECK-LABEL: @decimalIntegerShape
// CHECK-SAME: foo = #test.decimal_shape<5>
func.func private @decimalIntegerShape() attributes {
  foo = #test.decimal_shape<5>
}

// CHECK-LABEL: @decimalIntegerShapeMultiple
// CHECK-SAME: foo = #test.decimal_shape<0x3x7>
func.func private @decimalIntegerShapeMultiple() attributes {
  foo = #test.decimal_shape<0x3x7>
}

// -----

func.func private @hexdecimalInteger() attributes {
// expected-error @below {{expected an integer}}
  sdg = #test.decimal_shape<1x0xb>
}
