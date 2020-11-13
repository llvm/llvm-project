// RUN: mlir-opt %s | FileCheck %s


// CHECK-DAG: #test2Ealias = "alias_test:dot_in_name"
"test.op"() {alias_test = "alias_test:dot_in_name"} : () -> ()

// CHECK-DAG: #test_alias0_ = "alias_test:trailing_digit"
"test.op"() {alias_test = "alias_test:trailing_digit"} : () -> ()

// CHECK-DAG: #_0_test_alias = "alias_test:prefixed_digit"
"test.op"() {alias_test = "alias_test:prefixed_digit"} : () -> ()

// CHECK-DAG: #test_alias_conflict0_0 = "alias_test:sanitize_conflict_a"
// CHECK-DAG: #test_alias_conflict0_1 = "alias_test:sanitize_conflict_b"
"test.op"() {alias_test = ["alias_test:sanitize_conflict_a", "alias_test:sanitize_conflict_b"]} : () -> ()
