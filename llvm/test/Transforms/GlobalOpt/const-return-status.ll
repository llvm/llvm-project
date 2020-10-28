; RUN: opt -globalopt < %s -S -o - | FileCheck %s

; When simplifying users of a global variable, the pass could incorrectly
; return false if there were still some uses left, and no further optimizations
; was done. This was caught by the pass return status check that is hidden
; under EXPENSIVE_CHECKS.

; CHECK: @src = internal unnamed_addr constant

; CHECK: entry:
; CHECK-NEXT: %call = call i32 @f(i32 0)
; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 bitcast (i32* @dst to i8*), i8* align 4 bitcast ([1 x i32]* @src to i8*), i64 1, i1 false)
; CHECK-NEXT: ret void

@src = internal unnamed_addr global [1 x i32] zeroinitializer, align 4
@dst = external dso_local local_unnamed_addr global i32, align 4

define dso_local void @d() local_unnamed_addr {
entry:
  %0 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @src, i64 0, i64 0), align 4
  %call = call i32 @f(i32 %0)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 bitcast (i32* @dst to i8*), i8* align 4 bitcast ([1 x i32]* @src to i8*), i64 1, i1 false)
  ret void
}

declare dso_local i32 @f(i32) local_unnamed_addr

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
