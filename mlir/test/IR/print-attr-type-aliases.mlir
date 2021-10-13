// RUN: mlir-opt %s | FileCheck %s
// Verify printer of type & attr aliases.
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-DAG: #test2Ealias = "alias_test:dot_in_name"
"test.op"() {alias_test = "alias_test:dot_in_name"} : () -> ()

// CHECK-DAG: #test_alias0_ = "alias_test:trailing_digit"
"test.op"() {alias_test = "alias_test:trailing_digit"} : () -> ()

// CHECK-DAG: #_0_test_alias = "alias_test:prefixed_digit"
"test.op"() {alias_test = "alias_test:prefixed_digit"} : () -> ()

// CHECK-DAG: #test_alias_conflict0_0 = "alias_test:sanitize_conflict_a"
// CHECK-DAG: #test_alias_conflict0_1 = "alias_test:sanitize_conflict_b"
"test.op"() {alias_test = ["alias_test:sanitize_conflict_a", "alias_test:sanitize_conflict_b"]} : () -> ()

// CHECK-DAG: !tuple = type tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>
"test.op"() {alias_test = "alias_test:large_tuple"} : () -> (tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>)

// CHECK-DAG: !test_tuple = type tuple<!test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla>
"test.op"() {alias_test = "alias_test:large_tuple"} : () -> (tuple<!test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla>)

// CHECK-DAG: #test_encoding = "alias_test:tensor_encoding"
// CHECK-DAG: tensor<32xf32, #test_encoding>
"test.op"() : () -> tensor<32xf32, "alias_test:tensor_encoding">

// CHECK-DAG: !test_ui8_ = type !test.int<unsigned, 8>
// CHECK-DAG: tensor<32x!test_ui8_>
"test.op"() : () -> tensor<32x!test.int<unsigned, 8>>
