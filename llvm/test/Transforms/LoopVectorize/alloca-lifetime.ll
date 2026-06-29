; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S 2>&1 | FileCheck %s

; The alloca is live-out of the loop. The lifetime intrinsics can be removed
; before vectorization, to avoid SSA violations.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

declare void @llvm.lifetime.start.p0(ptr nocapture) nounwind
declare void @llvm.lifetime.end.p0(ptr nocapture) nounwind

; CHECK-LABEL: @live_alloca
; CHECK: 4 x i64
; CHECK-NOT: call {{.*}} @llvm.lifetime
define i32 @live_alloca(ptr %a, i64 %n) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %r = phi i32 [ %tmp3, %for.body ], [ 0, %entry ]
  %alloca = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %alloca)
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  %tmp0 = select i1 %cond, i64 %i.next, i64 0
  %tmp1 = getelementptr inbounds i32, ptr %a, i64 %tmp0
  %tmp2 = load i32, ptr %tmp1, align 8
  %tmp3 = add i32 %r, %tmp2
  br i1 %cond, label %for.body, label %for.end

for.end:
  %tmp4 = phi i32 [ %tmp3, %for.body ]
  call void @llvm.lifetime.end.p0(ptr %alloca)
  ret i32 %tmp4
}

; CHECK-LABEL: @live_alloca2
; CHECK-NOT: call {{.*}} @llvm.lifetime
define void @live_alloca2(ptr %ptr) {
entry:
  br label %loop

loop:                       ; preds = %loop, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %alloca = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %alloca)
  %cond0 = icmp ult i64 %iv, 13
  %s = select i1 %cond0, i32 10, i32 20
  %gep = getelementptr inbounds i32, ptr %ptr, i64 %iv
  store i32 %s, ptr %gep
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 14
  br i1 %exitcond, label %exit, label %loop

exit:
  call void @llvm.lifetime.end.p0(ptr %alloca)
  ret void
}
