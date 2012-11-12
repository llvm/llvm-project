; RUN: opt < %s -loop-vectorize -force-vector-width=4 -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK: @start_at_nonzero
;CHECK: mul nuw <4 x i32>
;CHECK: ret i32
define i32 @start_at_nonzero(i32* nocapture %a, i32 %start, i32 %end) nounwind uwtable ssp {
entry:
  %cmp3 = icmp slt i32 %start, %end
  br i1 %cmp3, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  %0 = sext i32 %start to i64
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvars.iv = phi i64 [ %0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv
  %1 = load i32* %arrayidx, align 4, !tbaa !0
  %mul = mul nuw i32 %1, 333
  store i32 %mul, i32* %arrayidx, align 4, !tbaa !0
  %indvars.iv.next = add i64 %indvars.iv, 1
  %2 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %2, %end
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret i32 4
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
