// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// CHECK: "test.type_producer"() : () -> !test.type_verification<i16>
"test.type_producer"() : () -> !test.type_verification<i16>

// -----

// expected-error @below{{failed to verify 'param': 16-bit signless integer or 32-bit signless integer}}
"test.type_producer"() : () -> !test.type_verification<f16>
