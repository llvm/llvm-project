// RUN: mlir-opt -convert-to-spirv="run-signature-conversion=false run-vector-unrolling=false" -split-input-file %s | FileCheck %s

// CHECK-LABEL: @basic
func.func @basic(%a: index, %b: index) {
  // CHECK: spirv.IAdd
  %0 = index.add %a, %b
  // CHECK: spirv.ISub
  %1 = index.sub %a, %b
  // CHECK: spirv.IMul
  %2 = index.mul %a, %b
  // CHECK: spirv.SDiv
  %3 = index.divs %a, %b
  // CHECK: spirv.UDiv
  %4 = index.divu %a, %b
  // CHECK: spirv.SRem
  %5 = index.rems %a, %b
  // CHECK: spirv.UMod
  %6 = index.remu %a, %b
  // CHECK: spirv.GL.SMax
  %7 = index.maxs %a, %b
  // CHECK: spirv.GL.UMax
  %8 = index.maxu %a, %b
  // CHECK: spirv.GL.SMin
  %9 = index.mins %a, %b
  // CHECK: spirv.GL.UMin
  %10 = index.minu %a, %b
  // CHECK: spirv.ShiftLeftLogical
  %11 = index.shl %a, %b
  // CHECK: spirv.ShiftRightArithmetic
  %12 = index.shrs %a, %b
  // CHECK: spirv.ShiftRightLogical
  %13 = index.shru %a, %b
  // CHECK: spirv.BitwiseAnd
  %14 = index.and %a, %b
  // CHECK: spirv.BitwiseOr
  %15 = index.or %a, %b
  // CHECK: spirv.BitwiseXor
  %16 = index.xor %a, %b
  return
}

// CHECK-LABEL: @cmp
func.func @cmp(%a : index, %b : index) {
  // CHECK: spirv.IEqual
  %0 = index.cmp eq(%a, %b)
  return
}

// CHECK-LABEL: @ceildivs
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK:    spirv.ReturnValue %{{.*}} : i32
func.func @ceildivs(%n: index, %m: index) -> index {
  %result = index.ceildivs %n, %m
  return %result : index
}

// CHECK-LABEL: @ceildivu
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK:    spirv.ReturnValue %{{.*}} : i32
func.func @ceildivu(%n: index, %m: index) -> index {
  %result = index.ceildivu %n, %m
  return %result : index
}
