// RUN: mlir-query %s -c "m isConstantOp()" | FileCheck %s

// CHECK: {{.*}}.mlir:5:13: note: "root" binds here
func.func @simple1() {
  %c1_i32 = arith.constant 1 : i32
  return
}

// CHECK: {{.*}}.mlir:12:11: note: "root" binds here
// CHECK: {{.*}}.mlir:13:11: note: "root" binds here
func.func @simple2() {
  %cst1 = arith.constant 1.0 : f32
  %cst2 = arith.constant 2.0 : f32
  %add = arith.addf %cst1, %cst2 : f32
  return
}
