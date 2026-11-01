// RUN: mlir-opt %s -pass-pipeline="builtin.module(convert-math-to-funcs{set-llvm-linkage=false convert-ctlz=true})" | FileCheck %s --implicit-check-not=llvm.linkage

// With set-llvm-linkage=false, the outlined functions carry no LLVM linkage.

// CHECK: func.func private @__mlir_math_ipowi_i32(
// CHECK: func.func private @__mlir_math_fpowi_f32_i32(
// CHECK: func.func private @__mlir_math_ctlz_i32(
func.func @outlined(%a: i32, %b: i32, %x: f32) -> (i32, f32, i32) {
  %0 = math.ipowi %a, %b : i32
  %1 = math.fpowi %x, %b : f32, i32
  %2 = math.ctlz %a : i32
  return %0, %1, %2 : i32, f32, i32
}
