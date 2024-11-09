// RUN: mlir-opt %s -split-input-file -mlir-print-debuginfo | FileCheck %s
// Verify printer of type & attr aliases.
// RUN: mlir-opt %s -split-input-file -mlir-print-debuginfo | mlir-opt -split-input-file -mlir-print-debuginfo | FileCheck %s

// CHECK-DAG: #test2Ealias = "alias_test:dot_in_name"
"test.op"() {alias_test = "alias_test:dot_in_name"} : () -> ()

// CHECK-DAG: #test_alias0_ = "alias_test:trailing_digit"
"test.op"() {alias_test = "alias_test:trailing_digit"} : () -> ()

// CHECK-DAG: #_0_test_alias = "alias_test:prefixed_digit"
"test.op"() {alias_test = "alias_test:prefixed_digit"} : () -> ()

// CHECK-DAG: #_25test = "alias_test:prefixed_symbol"
"test.op"() {alias_test = "alias_test:prefixed_symbol"} : () -> ()

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

// CHECK-DAG: #[[LOC_NESTED:.+]] = loc("nested")
// CHECK-DAG: #[[LOC_RAW:.+]] = loc("test.mlir":10:8)
// CHECK-DAG: = loc(fused<#[[LOC_NESTED]]>[#[[LOC_RAW]]])
"test.op"() {alias_test = loc(fused<loc("nested")>["test.mlir":10:8])} : () -> ()

// -----

// Check proper ordering of intermixed attribute/type aliases.
// CHECK: !tuple = tuple<
// CHECK: = loc(fused<!tuple
"test.op"() {alias_test = loc(fused<tuple<i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32>>["test.mlir":10:8])} : () -> ()

// -----

// Ensure self type parameters get considered for aliases.
// CHECK: !test_ui8_ = !test.int<unsigned, 8>
// CHECK: #test.attr_with_self_type_param : !test_ui8_
"test.op"() {alias_test = #test.attr_with_self_type_param : !test.int<unsigned, 8> } : () -> ()

// -----

// Check that we don't print aliases for things that aren't printed.
// CHECK: = loc(fused<memref<1xi32>
// CHECK-NOT: #map
"test.op"() {alias_test = loc(fused<memref<1xi32, affine_map<(d0) -> (d0)>>>["test.mlir":10:8])} : () -> ()

// -----

#unalias_me = "goodbye"
#keep_aliased = "alias_test:dot_in_name"

// CHECK: #test.conditional_alias<hello>
"test.op"() {attr = #test.conditional_alias<"hello">} : () -> ()
// CHECK-NEXT: #test.conditional_alias<#test_encoding>
"test.op"() {attr = #test.conditional_alias<"alias_test:tensor_encoding">} : () -> ()
// CHECK: #test.conditional_alias<goodbye>
"test.op"() {attr = #test.conditional_alias<#unalias_me>} : () -> ()
// CHECK-NEXT: #test.conditional_alias<#test2Ealias>
"test.op"() {attr = #test.conditional_alias<#keep_aliased>} : () -> ()

// -----

// Check that a deferred no_alias attr can be un-deferred.

#keep_aliased = "alias_test:dot_in_name"
#cond_alias = #test.conditional_alias<#keep_aliased>
#no_alias = loc(fused<#cond_alias>["test.mlir":1:1])

// CHECK: #[[TEST_ALIAS:.+]] = "alias_test:dot_in_name"
// CHECK: fused<#test.conditional_alias<#[[TEST_ALIAS]]>
// CHECK: "test.op"
"test.op"() {attr = #no_alias} : () -> () loc(fused<#no_alias>["test.mlir":0:0])
