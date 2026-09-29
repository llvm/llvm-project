// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: "test.type_producer"() : () -> !test.type_verification<i16>
"test.type_producer"() : () -> !test.type_verification<i16>

// -----

// expected-error @below{{failed to verify 'param': 16-bit signless integer or 32-bit signless integer}}
"test.type_producer"() : () -> !test.type_verification<f16>

// -----

// CHECK: "test.type_producer"() : () -> vector<!ptr.ptr<#test.const_memory_space>>
"test.type_producer"() : () -> vector<!ptr.ptr<#test.const_memory_space>>

// -----

// CHECK: "test.type_producer"() : () -> vector<!llvm.ptr<1>>
"test.type_producer"() : () -> vector<!llvm.ptr<1>>

// -----

// expected-error @below{{failed to verify 'elementType': VectorElementTypeInterface instance}}
"test.type_producer"() : () -> vector<memref<2xf32>>

// -----

// Test PredTypeTrait with single parameter - valid case.
// CHECK: "test.type_producer"() : () -> !test.type_pred_trait<5>
"test.type_producer"() : () -> !test.type_pred_trait<5>

// -----

// Test PredTypeTrait with single parameter - invalid case (zero is not positive).
// expected-error @below{{failed to verify that value must be positive}}
"test.type_producer"() : () -> !test.type_pred_trait<0>

// -----

// Test PredTypeTrait with multiple parameters - valid case (5 >= 3).
// CHECK: "test.type_producer"() : () -> !test.type_pred_trait_multi<5, 3>
"test.type_producer"() : () -> !test.type_pred_trait_multi<5, 3>

// -----

// Test PredTypeTrait with multiple parameters - edge case (3 >= 3).
// CHECK: "test.type_producer"() : () -> !test.type_pred_trait_multi<3, 3>
"test.type_producer"() : () -> !test.type_pred_trait_multi<3, 3>

// -----

// Test PredTypeTrait with multiple parameters - invalid case (2 < 5).
// expected-error @below{{failed to verify that value must be at least min}}
"test.type_producer"() : () -> !test.type_pred_trait_multi<2, 5>

// -----

// Test combined parameter constraint + PredTypeTrait - valid case.
// CHECK: "test.type_producer"() : () -> !test.type_pred_trait_combined<3, [1, 2, 3], i32>
"test.type_producer"() : () -> !test.type_pred_trait_combined<3, [1, 2, 3], i32>

// -----

// Test combined - parameter type constraint fails (f16 not in [I16, I32]).
// expected-error @below{{failed to verify 'elementType': 16-bit signless integer or 32-bit signless integer}}
"test.type_producer"() : () -> !test.type_pred_trait_combined<2, [1, 2], f16>

// -----

// Test combined - PredTypeTrait fails (count 2 != elements.size() 3).
// expected-error @below{{failed to verify that count must match number of elements}}
"test.type_producer"() : () -> !test.type_pred_trait_combined<2, [1, 2, 3], i16>
