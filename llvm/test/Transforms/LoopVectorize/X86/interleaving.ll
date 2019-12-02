; RUN: opt -S -mtriple=x86_64-pc_linux -loop-vectorize -instcombine < %s | FileCheck %s --check-prefix=NORMAL
; RUN: opt -S -mtriple=x86_64-pc_linux -loop-vectorize -instcombine -mcpu=slm < %s | FileCheck %s --check-prefix=SLOW
; RUN: opt -S -mtriple=x86_64-pc_linux -loop-vectorize -instcombine -mcpu=atom < %s | FileCheck %s --check-prefix=SLOW

; NORMAL-LABEL: foo
; NORMAL: %[[WIDE:.*]] = load <8 x i32>, <8 x i32>* %{{.*}}, align 4
; NORMAL: %[[STRIDED1:.*]] = shufflevector <8 x i32> %[[WIDE]], <8 x i32> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; NORMAL: %[[STRIDED2:.*]] = shufflevector <8 x i32> %wide.vec, <8 x i32> undef, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; NORMAL: add nsw <4 x i32> %[[STRIDED2]], %[[STRIDED1]]

; SLOW-LABEL: foo
; SLOW: load i32
; SLOW: load i32
; SLOW: store i32
define void @foo(i32* noalias nocapture %a, i32* noalias nocapture readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = shl nsw i64 %indvars.iv, 1
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %0
  %1 = load i32, i32* %arrayidx, align 4
  %2 = or i64 %0, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %b, i64 %2
  %3 = load i32, i32* %arrayidx3, align 4
  %add4 = add nsw i32 %3, %1
  %arrayidx6 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %add4, i32* %arrayidx6, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}
