; RUN: opt -S -indvars < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f_sadd(i8* %a) {
; CHECK-LABEL: @f_sadd(
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
  %i.04 = phi i32 [ 0, %entry ], [ %2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
; CHECK: for.body:
; CHECK-NOT: @llvm.sadd.with.overflow
; CHECK:  br i1 false, label %trap, label %cont, !nosanitize !0
  br i1 %1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap() #2, !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp = icmp slt i32 %2, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @f_uadd(i8* %a) {
; CHECK-LABEL: @f_uadd(
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
  %i.04 = phi i32 [ 0, %entry ], [ %2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %i.04, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
; CHECK: for.body:
; CHECK-NOT: @llvm.uadd.with.overflow
; CHECK: br i1 false, label %trap, label %cont, !nosanitize !0
  br i1 %1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp = icmp slt i32 %2, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @f_ssub(i8* nocapture %a) {
; CHECK-LABEL: @f_ssub(
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
  %i.04 = phi i32 [ 15, %entry ], [ %2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %i.04, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1
; CHECK: for.body:
; CHECK-NOT: @llvm.ssub.with.overflow.i32
; CHECK: br i1 false, label %trap, label %cont, !nosanitize !0
  br i1 %1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp = icmp sgt i32 %2, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @f_usub(i8* nocapture %a) {
; CHECK-LABEL: @f_usub(
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
  %i.04 = phi i32 [ 15, %entry ], [ %2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %i.04, i32 1)
  %1 = extractvalue { i32, i1 } %0, 1

; It is theoretically possible to prove this, but SCEV cannot
; represent non-unsigned-wrapping subtraction operations.

; CHECK: for.body:
; CHECK:  [[COND:%[^ ]+]] = extractvalue { i32, i1 } %1, 1
; CHECK-NEXT:  br i1 [[COND]], label %trap, label %cont, !nosanitize !0
  br i1 %1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %2 = extractvalue { i32, i1 } %0, 0
  %cmp = icmp sgt i32 %2, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

declare void @llvm.trap() #2
