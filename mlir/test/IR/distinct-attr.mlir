// RUN: mlir-opt %s | FileCheck %s
// RUN: mlir-opt -mlir-print-local-scope %s | FileCheck %s --check-prefix=CHECK-GENERIC

// CHECK: #[[DISTINCT0:.*]] = distinct[0]<42 : i32>
// CHECK: #[[DISTINCT1:.*]] = distinct[1]<array<i32: 10, 42>>
// CHECK: #[[DISTINCT2:.*]] = distinct[2]<42 : i32>

// CHECK:         distinct.attr = #[[DISTINCT0]]
// CHECK-GENERIC: distinct.attr = distinct[0]<42 : i32>
"test.op"() {distinct.attr = distinct[0]<42 : i32>} : () -> ()

// CHECK:         distinct.attr = #[[DISTINCT1]]
// CHECK-GENERIC: distinct.attr = distinct[1]<array<i32: 10, 42>>
"test.op"() {distinct.attr = distinct[1]<array<i32: 10, 42>>} : () -> ()

// CHECK:         distinct.attr = #[[DISTINCT0]]
// CHECK-GENERIC: distinct.attr = distinct[0]<42 : i32>
"test.op"() {distinct.attr = distinct[0]<42 : i32>} : () -> ()

// CHECK:         distinct.attr = #[[DISTINCT2]]
// CHECK-GENERIC: distinct.attr = distinct[2]<42 : i32>
"test.op"() {distinct.attr = distinct[42]<42 : i32>} : () -> ()

// CHECK: distinct.attr = distinct[3]<>
"test.op"() {distinct.attr = distinct[3]<>} : () -> ()

// CHECK: distinct.attr = distinct[4]<>
"test.op"() {distinct.attr = distinct[4]<unit>} : () -> ()
