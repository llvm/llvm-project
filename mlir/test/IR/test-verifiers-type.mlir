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
