; RUN: llc < %s -march=xcore | FileCheck %s

; Optimize memcpy to __memcpy_4 if src, dst and size are all 4 byte aligned.
define void @f1(i8* %dst, i8* %src, i32 %n) nounwind {
; CHECK-LABEL: f1:
; CHECK: bl __memcpy_4
entry:
  %0 = shl i32 %n, 2
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %0, i32 4, i1 false)
  ret void
}

; Can't optimize - size is not a multiple of 4.
define void @f2(i8* %dst, i8* %src, i32 %n) nounwind {
; CHECK-LABEL: f2:
; CHECK: bl memcpy
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %n, i32 4, i1 false)
  ret void
}

; Can't optimize - alignment is not a multiple of 4.
define void @f3(i8* %dst, i8* %src, i32 %n) nounwind {
; CHECK-LABEL: f3:
; CHECK: bl memcpy
entry:
  %0 = shl i32 %n, 2
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %src, i32 %0, i32 2, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
