; RUN: opt -safe-stack -safe-stack-coloring=1 -S -mtriple=x86_64-pc-linux-gnu < %s -o - | FileCheck %s

; %x and %y share a stack slot between them, but not with the stack guard.
define void @f() safestack sspreq {
; CHECK-LABEL: define void @f
entry:
; CHECK:  %[[USP:.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK:   getelementptr i8, ptr %[[USP]], i32 -16

; CHECK:  %[[A:.*]] = getelementptr i8, ptr %[[USP]], i32 -8
; CHECK:  store ptr %{{.*}}, ptr %[[A]]

  %x = alloca i64, align 8
  %y = alloca i64, align 8

  call void @llvm.lifetime.start.p0(i64 -1, ptr %x)
; CHECK:  getelementptr i8, ptr %[[USP]], i32 -16
  call void @capture64(ptr %x)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %x)

  call void @llvm.lifetime.start.p0(i64 -1, ptr %y)
; CHECK:  getelementptr i8, ptr %[[USP]], i32 -16
  call void @capture64(ptr %y)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %y)

  ret void
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @capture64(ptr)
