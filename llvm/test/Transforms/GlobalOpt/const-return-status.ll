; RUN: opt -passes=globalopt < %s -S -o - | FileCheck %s

; When simplifying users of a global variable, the pass could incorrectly
; return false if there were still some uses left, and no further optimizations
; was done. This was caught by the pass return status check that is hidden
; under EXPENSIVE_CHECKS.

; CHECK: @src = internal unnamed_addr constant

; CHECK: entry:
; CHECK-NEXT: %call = call i32 @f(i32 0)
; CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 @dst, ptr align 4 @src, i64 1, i1 false)
; CHECK-NEXT: ret void

@src = internal unnamed_addr global [1 x i32] zeroinitializer, align 4
@dst = external dso_local local_unnamed_addr global i32, align 4

define dso_local void @d() local_unnamed_addr {
entry:
  %0 = load i32, ptr @src, align 4
  %call = call i32 @f(i32 %0)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 @dst, ptr align 4 @src, i64 1, i1 false)
  ret void
}

declare dso_local i32 @f(i32) local_unnamed_addr

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
