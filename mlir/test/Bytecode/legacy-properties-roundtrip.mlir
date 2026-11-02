// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=0 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=1 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=2 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=3 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=4 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=5 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode -emit-bytecode-version=6 | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -emit-bytecode | mlir-opt | FileCheck %s

// Default native properties are reconstructed without being encoded.
// CHECK: test.with_versioned_properties 0 | 0
test.with_versioned_properties 0 | 0

// Native defaults need not be zero, and inherent fields may be non-default.
// CHECK: test.with_default_valued_properties 7
test.with_default_valued_properties 7 "" -1 unit_absent

// A discardable attribute must not be interpreted as a native property.
// CHECK: test.with_versioned_properties 0 | 0 {prop = {value1 = 1 : i32, value2 = 2 : i32}}
test.with_versioned_properties 0 | 0 {prop = {value1 = 1 : i32, value2 = 2 : i32}}

// Inherent attributes and discardable attributes are preserved.
// CHECK: "test.versionedA"() <{dims = 123 : i64, modifier = true}> {discardable = 42 : i64}
"test.versionedA"() <{dims = 123 : i64, modifier = true}> {discardable = 42 : i64} : () -> ()

// Segment sizes are native properties exposed through the legacy hooks.
func.func @segments(%arg0: i32) {
  // CHECK: operandSegmentSizes = array<i32: 0, 2, 1, 1>
  "test.attr_sized_operands"(%arg0, %arg0, %arg0, %arg0) <{operandSegmentSizes = array<i32: 0, 2, 1, 1>}> : (i32, i32, i32, i32) -> ()
  // CHECK: resultSegmentSizes = array<i32: 0, 2, 1, 1>
  %0:4 = "test.attr_sized_results"() <{resultSegmentSizes = array<i32: 0, 2, 1, 1>}> : () -> (i32, i32, i32, i32)
  return
}
