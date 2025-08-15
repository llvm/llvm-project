; RUN: opt -S -passes=correlated-propagation -debug-only=lazy-value-info <%s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Shortcut in Correlated Value Propagation ensures not to take Lazy Value Info
; analysis for %a.i and %tmp because %a.i is defined by alloca and %tmp is
; defined by alloca + bitcast. We know the ret value of alloca is nonnull.
;
; CHECK-NOT: LVI Getting edge value   %a.i = alloca i64, align 8 at 'for.body'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [8 x i8] c"a = %l\0A\00", align 1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(ptr nocapture)

declare void @hoo(ptr)

declare i32 @printf(ptr nocapture readonly, ...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(ptr nocapture)

define void @goo(i32 %N, ptr %b) {
entry:
  %a.i = alloca i64, align 8
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0(ptr %a.i)
  call void @hoo(ptr %a.i)
  call void @hoo(ptr %b)
  %tmp1 = load volatile i64, ptr %a.i, align 8
  %call.i = call i32 (ptr, ...) @printf(ptr @.str, i64 %tmp1)
  call void @llvm.lifetime.end.p0(ptr %a.i)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
