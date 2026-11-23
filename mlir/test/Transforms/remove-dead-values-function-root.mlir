// RUN: mlir-opt %s --pass-pipeline='builtin.module(func.func(remove-dead-values))' | FileCheck %s

// CHECK-LABEL: func.func private @callee
// CHECK-SAME: %[[ARG:.*]]: i32, %{{.*}}: i32) -> i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[ARG]], %[[ARG]] : i32
// CHECK-NEXT: return %[[SUM]] : i32
func.func private @callee(%arg: i32, %unused: i32) -> i32 {
  %sum = arith.addi %arg, %arg : i32
  return %sum : i32
}

// CHECK-LABEL: func.func @caller
// CHECK: call @callee(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
func.func @caller(%x: i32) -> i32 {
  %r = call @callee(%x, %x) : (i32, i32) -> i32
  return %r : i32
}
