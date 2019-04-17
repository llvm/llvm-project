; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

; %x and %y share a stack slot between them, but not with the stack guard.
define void @f() safestack sspreq {
; CHECK-LABEL: define void @f
entry:
; CHECK:  %[[USP:.*]] = load i8*, i8** @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, i8* %[[USP]], i32 -16

; CHECK:  %[[A:.*]] = getelementptr i8, i8* %[[USP]], i32 -8
; CHECK:  %[[StackGuardSlot:.*]] = bitcast i8* %[[A]] to i8**
; CHECK:  store i8* %{{.*}}, i8** %[[StackGuardSlot]]

  %x = alloca i64, align 8
  %y = alloca i64, align 8
  %x0 = bitcast i64* %x to i8*
  %y0 = bitcast i64* %y to i8*

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %x0)
; CHECK:  getelementptr i8, i8* %[[USP]], i32 -16
  call void @capture64(i64* %x)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %x0)

  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %y0)
; CHECK:  getelementptr i8, i8* %[[USP]], i32 -16
  call void @capture64(i64* %y)
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %y0)

  ret void
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)
declare void @capture64(i64*)
