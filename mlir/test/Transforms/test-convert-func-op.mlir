// RUN: mlir-opt %s -test-convert-func-op | FileCheck %s

// CHECK-LABEL: llvm.func @add
func.func @add(%arg0: i32, %arg1: i32) -> i32 attributes { llvm.emit_c_interface } {
  %res = arith.addi %arg0, %arg1 : i32
  return %res : i32
}
// CHECK-LABEL: llvm.func @_mlir_ciface_add
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]+]]: i32
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]+]]: i32
// CHECK-NEXT: [[RES:%.*]] = llvm.call @add([[ARG0]], [[ARG1]])
// CHECK-NEXT: llvm.return [[RES]]
