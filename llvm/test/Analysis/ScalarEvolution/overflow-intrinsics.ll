; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f_sadd_0(i8* %a) {
; CHECK-LABEL: Classifying expressions for: @f_sadd_0
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%for.body> U: [0,16) S: [0,16)

  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap() #2, !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  %cmp = icmp slt i32 %tmp2, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK: Loop %for.body: max backedge-taken count is 15
}

define void @f_sadd_1(i8* %a) {
; CHECK-LABEL: Classifying expressions for: @f_sadd_1
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
; CHECK-NEXT:  -->  {0,+,1}<%for.body> U: [0,16) S: [0,16)

; SCEV can prove <nsw> for the above induction variable; but it does
; not bother so before it sees the sext below since it is not a 100%
; obvious.

  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body

  br label %cont

cont:                                             ; preds = %for.body
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  %cmp = icmp slt i32 %tmp2, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK: Loop %for.body: max backedge-taken count is 15
}

define void @f_sadd_2(i8* %a, i1* %c) {
; CHECK-LABEL: Classifying expressions for: @f_sadd_2
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
; CHECK-NEXT:  -->  {0,+,1}<%for.body>

  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body

  br label %cont

cont:                                             ; preds = %for.body
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  %cond = load volatile i1, i1* %c
  br i1 %cond, label %for.body, label %for.cond.cleanup
}

define void @f_sadd_3(i8* %a, i1* %c) {
; CHECK-LABEL: Classifying expressions for: @f_sadd_3
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %for.body ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%for.body>

  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %for.body ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  br i1 %tmp1, label %trap, label %for.body, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap() #2, !nosanitize !{}
  unreachable, !nosanitize !{}
}

define void @f_sadd_4(i8* %a, i1* %c) {
; CHECK-LABEL: Classifying expressions for: @f_sadd_4
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %merge ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%for.body>

  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %merge ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  br i1 %tmp1, label %notrap, label %merge

notrap:
  br label %merge

merge:
  %tmp3 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp3, label %trap, label %for.body, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap() #2, !nosanitize !{}
  unreachable, !nosanitize !{}
}

define void @f_sadd_may_overflow(i8* %a, i1* %c) {
; CHECK-LABEL: Classifying expressions for: @f_sadd_may_overflow
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp1, %cont ]
; CHECK-NEXT:  -->  {0,+,1}<%for.body> U: full-set S: full-set

  %i.04 = phi i32 [ 0, %entry ], [ %tmp1, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %i.04, i32 1)
  %cond1 = load volatile i1, i1* %c
  br i1 %cond1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap() #2, !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %tmp1 = extractvalue { i32, i1 } %tmp0, 0
  %cond = load volatile i1, i1* %c
  br i1 %cond, label %for.body, label %for.cond.cleanup
}

define void @f_uadd(i8* %a) {
; CHECK-LABEL: Classifying expressions for: @f_uadd
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><%for.body> U: [0,16) S: [0,16)

  %i.04 = phi i32 [ 0, %entry ], [ %tmp2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  %cmp = icmp slt i32 %tmp2, 16
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK: Loop %for.body: max backedge-taken count is 15
}

define void @f_ssub(i8* nocapture %a) {
; CHECK-LABEL: Classifying expressions for: @f_ssub
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 15, %entry ], [ %tmp2, %cont ]
; CHECK-NEXT:  -->  {15,+,-1}<%for.body> U: [0,16) S: [0,16)

  %i.04 = phi i32 [ 15, %entry ], [ %tmp2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  %cmp = icmp sgt i32 %tmp2, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK: Loop %for.body: max backedge-taken count is 15
}

define void @f_usub(i8* nocapture %a) {
; CHECK-LABEL: Classifying expressions for: @f_usub
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont
  ret void

for.body:                                         ; preds = %entry, %cont
; CHECK:  %i.04 = phi i32 [ 15, %entry ], [ %tmp2, %cont ]
; CHECK-NEXT:  -->  {15,+,-1}<%for.body> U: [0,16) S: [0,16)

  %i.04 = phi i32 [ 15, %entry ], [ %tmp2, %cont ]
  %idxprom = sext i32 %i.04 to i64
  %arrayidx = getelementptr inbounds i8, i8* %a, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %tmp0 = tail call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %i.04, i32 1)
  %tmp1 = extractvalue { i32, i1 } %tmp0, 1
  br i1 %tmp1, label %trap, label %cont, !nosanitize !{}

trap:                                             ; preds = %for.body
  tail call void @llvm.trap(), !nosanitize !{}
  unreachable, !nosanitize !{}

cont:                                             ; preds = %for.body
  %tmp2 = extractvalue { i32, i1 } %tmp0, 0
  %cmp = icmp sgt i32 %tmp2, -1
  br i1 %cmp, label %for.body, label %for.cond.cleanup
; CHECK: Loop %for.body: max backedge-taken count is 15
}

define i32 @f_smul(i32 %val_a, i32 %val_b) {
; CHECK-LABEL: Classifying expressions for: @f_smul
  %agg = tail call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %val_a, i32 %val_b)
; CHECK:   %mul = extractvalue { i32, i1 } %agg, 0
; CHECK-NEXT:  -->  (%val_a * %val_b) U: full-set S: full-set
  %mul = extractvalue { i32, i1 } %agg, 0
  ret i32 %mul
}

define i32 @f_umul(i32 %val_a, i32 %val_b) {
; CHECK-LABEL: Classifying expressions for: @f_umul
  %agg = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %val_a, i32 %val_b)
; CHECK:   %mul = extractvalue { i32, i1 } %agg, 0
; CHECK-NEXT:  -->  (%val_a * %val_b) U: full-set S: full-set
  %mul = extractvalue { i32, i1 } %agg, 0
  ret i32 %mul
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) nounwind readnone
declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32) nounwind readnone

declare void @llvm.trap() #2
