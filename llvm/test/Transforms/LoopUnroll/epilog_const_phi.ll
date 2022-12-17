; RUN: opt -S -passes=loop-unroll -unroll-runtime < %s | FileCheck %s

; Epilog unroll allows to keep PHI constant value.
; For the test this means that after unroll XOR could be deleted.
; Check that we do epilogue reminder here.

; CHECK-LABEL: const_phi_val
; CHECK:  for.body.epil

; Function Attrs: norecurse nounwind uwtable
define void @const_phi_val(i32 %i0, i32* nocapture %a) {
entry:
  %cmp6 = icmp slt i32 %i0, 1000
  br i1 %cmp6, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %tmp = sext i32 %i0 to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ %tmp, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %s.08 = phi i32 [ 0, %for.body.preheader ], [ %xor, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %s.08, i32* %arrayidx, align 4
  %xor = xor i32 %s.08, 1
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}

; When there is no phi with const coming from preheader,
; there is no need to do epilogue unrolling.

; CHECK-LABEL: var_phi_val
; CHECK:  for.body.prol

; Function Attrs: norecurse nounwind uwtable
define void @var_phi_val(i32 %i0, i32* nocapture %a) {
entry:
  %cmp6 = icmp slt i32 %i0, 1000
  br i1 %cmp6, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  %tmp = sext i32 %i0 to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %indvars.iv = phi i64 [ %tmp, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret void
}
