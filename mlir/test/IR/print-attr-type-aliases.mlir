// RUN: mlir-opt %s -split-input-file | FileCheck %s
// Verify printer of type & attr aliases.
// RUN: mlir-opt %s -split-input-file | mlir-opt -split-input-file | FileCheck %s

// CHECK-DAG: #test2Ealias = "alias_test:dot_in_name"
"test.op"() {alias_test = "alias_test:dot_in_name"} : () -> ()

// CHECK-DAG: #test_alias0_ = "alias_test:trailing_digit"
"test.op"() {alias_test = "alias_test:trailing_digit"} : () -> ()

// CHECK-DAG: #_0_test_alias = "alias_test:prefixed_digit"
"test.op"() {alias_test = "alias_test:prefixed_digit"} : () -> ()

// CHECK-DAG: #test_alias_conflict0_ = "alias_test:sanitize_conflict_a"
// CHECK-DAG: #test_alias_conflict0_1 = "alias_test:sanitize_conflict_b"
"test.op"() {alias_test = ["alias_test:sanitize_conflict_a", "alias_test:sanitize_conflict_b"]} : () -> ()

// CHECK-DAG: !tuple = tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>
"test.op"() {alias_test = "alias_test:large_tuple"} : () -> (tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>)

// CHECK-DAG: !test_tuple = tuple<!test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla>
"test.op"() {alias_test = "alias_test:large_tuple"} : () -> (tuple<!test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla, !test.smpla>)

// CHECK-DAG: #test_encoding = "alias_test:tensor_encoding"
// CHECK-DAG: tensor<32xf32, #test_encoding>
"test.op"() : () -> tensor<32xf32, "alias_test:tensor_encoding">

// CHECK-DAG: !test_ui8_ = !test.int<unsigned, 8>
// CHECK-DAG: tensor<32x!test_ui8_>
"test.op"() : () -> tensor<32x!test.int<unsigned, 8>>

// CHECK-DAG: #loc2 = loc("nested")
// CHECK-DAG: #loc3 = loc(fused<#loc2>["test.mlir":10:8])
"test.op"() {alias_test = loc(fused<loc("nested")>["test.mlir":10:8])} : () -> ()

// -----

// Check proper ordering of intermixed attribute/type aliases.
// CHECK: !tuple = tuple<
// CHECK: #loc1 = loc(fused<!tuple
"test.op"() {alias_test = loc(fused<tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>>["test.mlir":10:8])} : () -> ()
