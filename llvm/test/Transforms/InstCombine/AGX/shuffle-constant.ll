; RUN: opt -mtriple=agx -passes=instcombine -S %s | FileCheck %s

; A shuffle of a constant is the constant.
; CHECK-LABEL: @constant
; CHECK-NOT: call
; CHECK: ret i32 7
define i32 @constant(i32 %l) {
  %a = call i32 @llvm.agx.simd.shuffle.i32(i32 0, i32 %l)
  %b = call i32 @llvm.agx.quad.shuffle.xor.i32(i32 7, i32 %l)
  %c = call i32 @llvm.agx.simd.shuffle.down.i32(i32 %a, i32 1)
  %r = or i32 %b, %c
  ret i32 %r
}

; CHECK-LABEL: @float
; CHECK: ret float 1.000000e+00
define float @float(i32 %l) {
  %a = call float @llvm.agx.simd.shuffle.up.f32(float 1.0, i32 %l)
  ret float %a
}

; A shuffle of a varying value stays.
; CHECK-LABEL: @value
; CHECK: call i32 @llvm.agx.simd.shuffle.i32(i32 %v, i32 %l)
define i32 @value(i32 %v, i32 %l) {
  %a = call i32 @llvm.agx.simd.shuffle.i32(i32 %v, i32 %l)
  ret i32 %a
}
