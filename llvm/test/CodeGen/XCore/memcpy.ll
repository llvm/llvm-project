; RUN: llc < %s -mtriple=xcore | FileCheck %s

; Optimize memcpy to __memcpy_4 if src, dst and size are all 4 byte aligned.
define void @f1(ptr %dst, ptr %src, i32 %n) nounwind {
; CHECK-LABEL: f1:
; CHECK: bl __memcpy_4
entry:
  %0 = shl i32 %n, 2
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 %0, i1 false)
  ret void
}

; Can't optimize - size is not a multiple of 4.
define void @f2(ptr %dst, ptr %src, i32 %n) nounwind {
; CHECK-LABEL: f2:
; CHECK: bl memcpy
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %dst, ptr align 4 %src, i32 %n, i1 false)
  ret void
}

; Can't optimize - alignment is not a multiple of 4.
define void @f3(ptr %dst, ptr %src, i32 %n) nounwind {
; CHECK-LABEL: f3:
; CHECK: bl memcpy
entry:
  %0 = shl i32 %n, 2
  call void @llvm.memcpy.p0.p0.i32(ptr align 2 %dst, ptr align 2 %src, i32 %0, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
