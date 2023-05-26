// # RUN: mlir-opt %s -split-input-file | mlir-opt  |FileCheck %s
// # RUN: mlir-opt %s -mlir-print-op-generic -split-input-file  | mlir-opt -mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC

// CHECK:   test.with_properties
// CHECK-SAME: <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, b = "foo"}>
// GENERIC:   "test.with_properties"()
// GENERIC-SAME: <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, b = "foo"}> : () -> ()
test.with_properties <{a = 32 : i64, array = array<i64: 1, 2, 3, 4>, b = "foo"}>

// CHECK:   test.with_nice_properties
// CHECK-SAME:    "foo bar" is -3
// GENERIC: "test.with_nice_properties"()
// GENERIC-SAME:  <{prop = {label = "foo bar", value = -3 : i32}}> : () -> ()
test.with_nice_properties "foo bar" is -3

// CHECK:   test.with_wrapped_properties
// CHECK-SAME:    "content for properties"
// GENERIC: "test.with_wrapped_properties"()
// GENERIC-SAME:  <{prop = "content for properties"}> : () -> ()
test.with_wrapped_properties <{prop = "content for properties"}>
