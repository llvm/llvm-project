; Rename LLVM math intrinsic declarations to their AIR equivalents and inline
; the software implementation of __mulhi.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: llvm.sin.f32
; CHECK-NOT: llvm.sqrt.f32
; CHECK-NOT: __mulhi
; CHECK: call float @air.fast_sin.f32
; CHECK: call float @air.fast_sqrt.f32
; CHECK: zext i32 {{.*}} to i64
; CHECK: mul i64
; CHECK: lshr i64 {{.*}}, 32
; CHECK: trunc i64 {{.*}} to i32
; CHECK: ret void

declare float @llvm.sin.f32(float)
declare float @llvm.sqrt.f32(float)
declare i32 @__mulhi(i32, i32)

define void @kernel(float %x, i32 %a, i32 %b, ptr %fout, ptr %iout) {
entry:
  %s = call float @llvm.sin.f32(float %x)
  %r = call float @llvm.sqrt.f32(float %s)
  store float %r, ptr %fout
  %h = call i32 @__mulhi(i32 %a, i32 %b)
  store i32 %h, ptr %iout
  ret void
}
