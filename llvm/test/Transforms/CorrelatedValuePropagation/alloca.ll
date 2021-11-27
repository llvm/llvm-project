; RUN: opt -S -passes=correlated-propagation -debug-only=lazy-value-info <%s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Shortcut in Correlated Value Propagation ensures not to take Lazy Value Info
; analysis for %a.i and %tmp because %a.i is defined by alloca and %tmp is
; defined by alloca + bitcast. We know the ret value of alloca is nonnull.
;
; CHECK-NOT: LVI Getting edge value   %a.i = alloca i64, align 8 at 'for.body'
; CHECK-NOT: LVI Getting edge value   %tmp = bitcast i64* %a.i to i8* from 'for.cond' to 'for.body'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [8 x i8] c"a = %l\0A\00", align 1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)

declare void @hoo(i64*)

declare i32 @printf(i8* nocapture readonly, ...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

define void @goo(i32 %N, i64* %b) {
entry:
  %a.i = alloca i64, align 8
  %tmp = bitcast i64* %a.i to i8*
  %c = getelementptr inbounds i64, i64* %b, i64 0
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %tmp)
  call void @hoo(i64* %a.i)
  call void @hoo(i64* %c)
  %tmp1 = load volatile i64, i64* %a.i, align 8
  %call.i = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0), i64 %tmp1)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %tmp)
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
