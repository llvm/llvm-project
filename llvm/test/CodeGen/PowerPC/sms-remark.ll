; RUN: llc < %s -ppc-vsr-nums-as-vr -mtriple=powerpc64-unknown-linux-gnu \
; RUN:       -verify-machineinstrs -ppc-asm-full-reg-names -mcpu=pwr9 --ppc-enable-pipeliner \
; RUN:       -pass-remarks-analysis=pipeliner -pass-remarks=pipeliner -o /dev/null 2>&1 \
; RUN:       | FileCheck %s --check-prefix=ENABLED
; RUN: llc < %s -ppc-vsr-nums-as-vr -mtriple=powerpc64-unknown-linux-gnu \
; RUN:       -verify-machineinstrs -ppc-asm-full-reg-names -mcpu=pwr8 --ppc-enable-pipeliner \
; RUN:       -pass-remarks-analysis=pipeliner -pass-remarks=pipeliner -o /dev/null 2>&1 \
; RUN:       | FileCheck %s --allow-empty --check-prefix=DISABLED

@x = dso_local local_unnamed_addr global <{ i32, i32, i32, i32, [1020 x i32] }> <{ i32 1, i32 2, i32 3, i32 4, [1020 x i32] zeroinitializer }>, align 4
@y = dso_local global [1024 x i32] zeroinitializer, align 4

define dso_local ptr @foo() local_unnamed_addr {
;ENABLED: Schedule found with Initiation Interval
;ENABLED: Pipelined succesfully!
;DISABLED-NOT: remark
entry:
  %.pre = load i32, ptr @y, align 4
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret ptr @y

for.body:                                         ; preds = %for.body, %entry
  %0 = phi i32 [ %.pre, %entry ], [ %add.2, %for.body ]
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next.2, %for.body ]
  %arrayidx2 = getelementptr inbounds [1024 x i32], ptr @x, i64 0, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %1, %1
  %add = add nsw i32 %mul, %0
  %arrayidx6 = getelementptr inbounds [1024 x i32], ptr @y, i64 0, i64 %indvars.iv
  store i32 %add, ptr %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx2.1 = getelementptr inbounds [1024 x i32], ptr @x, i64 0, i64 %indvars.iv.next
  %2 = load i32, ptr %arrayidx2.1, align 4
  %mul.1 = mul nsw i32 %2, %2
  %add.1 = add nsw i32 %mul.1, %add
  %arrayidx6.1 = getelementptr inbounds [1024 x i32], ptr @y, i64 0, i64 %indvars.iv.next
  store i32 %add.1, ptr %arrayidx6.1, align 4
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2
  %arrayidx2.2 = getelementptr inbounds [1024 x i32], ptr @x, i64 0, i64 %indvars.iv.next.1
  %3 = load i32, ptr %arrayidx2.2, align 4
  %mul.2 = mul nsw i32 %3, %3
  %add.2 = add nsw i32 %mul.2, %add.1
  %arrayidx6.2 = getelementptr inbounds [1024 x i32], ptr @y, i64 0, i64 %indvars.iv.next.1
  store i32 %add.2, ptr %arrayidx6.2, align 4
  %indvars.iv.next.2 = add nuw nsw i64 %indvars.iv, 3
  %exitcond.2 = icmp eq i64 %indvars.iv.next.2, 1024
  br i1 %exitcond.2, label %for.cond.cleanup, label %for.body
}
