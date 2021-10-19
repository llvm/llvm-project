// RUN: mlir-opt %s | mlir-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func private @compoundA()
// CHECK-SAME: #test.cmpnd_a<1, !test.smpla, [5, 6]>
func private @compoundA() attributes {foo = #test.cmpnd_a<1, !test.smpla, [5, 6]>}

// CHECK: test.result_has_same_type_as_attr #test<"attr_with_self_type_param i32"> -> i32
%a = test.result_has_same_type_as_attr #test<"attr_with_self_type_param i32"> -> i32

// CHECK: test.result_has_same_type_as_attr #test<"attr_with_type_builder 10 : i16"> -> i16
%b = test.result_has_same_type_as_attr #test<"attr_with_type_builder 10 : i16"> -> i16
