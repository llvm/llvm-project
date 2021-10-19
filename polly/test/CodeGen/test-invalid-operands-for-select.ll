; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s
;
; Check that we do not crash as described here: http://llvm.org/PR21167
;
; In case the pieceweise affine function used to create an isl_ast_expr
; had empty cases (e.g., with contradicting constraints on the
; parameters), it was possible that the condition of the isl_ast_expr
; select was not a comparison but a constant (thus of type i64).
; However, we shouldn't crash in such a case :)
;
; CHECK: polly.split_new_and_old
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @dradb4(i32 %ido, i32 %l1, float* %cc, float* %ch, float* %wa1, float* %wa3) #0 {
entry:
  %mul = mul nsw i32 %l1, %ido
  br i1 undef, label %for.end256, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %L105, label %for.cond45.preheader

for.cond45.preheader:                             ; preds = %if.end
  br i1 undef, label %for.body47, label %for.end198

for.body47:                                       ; preds = %for.inc096, %for.cond45.preheader
  br i1 undef, label %for.body53.lr.ph, label %for.inc096

for.body53.lr.ph:                                 ; preds = %for.body47
  br label %for.body53

for.body53:                                       ; preds = %for.body53, %for.body53.lr.ph
  %t7.014 = phi i32 [ 0, %for.body53.lr.ph ], [ %add58, %for.body53 ]
  %i.013 = phi i32 [ 2, %for.body53.lr.ph ], [ %add193, %for.body53 ]
  %add58 = add nsw i32 %t7.014, 2
  %arrayidx70 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx72 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx77 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx81 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx84 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx95 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx105 = getelementptr inbounds float, float* %cc, i64 0
  %arrayidx110 = getelementptr inbounds float, float* %ch, i64 0
  store float undef, float* %arrayidx110, align 4
  %arrayidx122 = getelementptr inbounds float, float* %wa1, i64 0
  %add129 = add nsw i32 %add58, %mul
  %idxprom142 = sext i32 %add129 to i64
  %arrayidx143 = getelementptr inbounds float, float* %ch, i64 %idxprom142
  store float undef, float* %arrayidx143, align 4
  %add153 = add nsw i32 %add129, %mul
  %arrayidx170 = getelementptr inbounds float, float* %wa3, i64 0
  %arrayidx174 = getelementptr inbounds float, float* %wa3, i64 0
  %add177 = add nsw i32 %add153, %mul
  %sub178 = add nsw i32 %add177, -1
  %idxprom179 = sext i32 %sub178 to i64
  %arrayidx180 = getelementptr inbounds float, float* %ch, i64 %idxprom179
  store float undef, float* %arrayidx180, align 4
  %arrayidx183 = getelementptr inbounds float, float* %wa3, i64 0
  %0 = load float, float* %arrayidx183, align 4
  %mul184 = fmul float undef, %0
  %add189 = fadd float %mul184, 0.000000e+00
  %idxprom190 = sext i32 %add177 to i64
  %arrayidx191 = getelementptr inbounds float, float* %ch, i64 %idxprom190
  store float %add189, float* %arrayidx191, align 4
  %add193 = add nsw i32 %i.013, 2
  %cmp52 = icmp slt i32 %add193, %ido
  br i1 %cmp52, label %for.body53, label %for.inc096

for.inc096:                                       ; preds = %for.body53, %for.body47
  br i1 undef, label %for.body47, label %for.end198

for.end198:                                       ; preds = %for.inc096, %for.cond45.preheader
  br i1 false, label %for.end256, label %L105

L105:                                             ; preds = %for.end198, %if.end
  br label %for.end256

for.end256:                                       ; preds = %L105, %for.end198, %entry
  ret void
}
