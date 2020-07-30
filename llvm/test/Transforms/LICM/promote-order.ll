; RUN: opt -tbaa -basic-aa -licm -enable-mssa-loop-dependency=false -S < %s | FileCheck %s --check-prefixes=CHECK,AST
; RUN: opt -tbaa -basic-aa -licm -enable-mssa-loop-dependency=true -S < %s | FileCheck %s --check-prefixes=CHECK,MSSA
; RUN: opt -aa-pipeline=type-based-aa,basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop(licm)' -S %s | FileCheck %s  --check-prefixes=CHECK,AST
; RUN: opt -aa-pipeline=type-based-aa,basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' -S %s | FileCheck %s --check-prefixes=CHECK,MSSA

; LICM should keep the stores in their original order when it sinks/promotes them.
; rdar://12045203

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

@p = external global i8*

define i32* @_Z4doiti(i32 %n, float* %tmp1, i32* %tmp3) nounwind {
; CHECK-LABEL: for.body.lr.ph:
; CHECK: store float 1.000000e+00, float* %tmp1
; AST-LABEL: for.cond.for.end_crit_edge:
; CHECK: store i32 1, i32* %tmp3
; MSSA-LABEL: for.cond.for.end_crit_edge:

entry:
  %cmp1 = icmp slt i32 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  store float 1.000000e+00, float* %tmp1, align 4, !tbaa !1
  store i32 1, i32* %tmp3, align 4, !tbaa !2
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %split = phi i32* [ %tmp3, %for.body ]
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %r.0.lcssa = phi i32* [ %split, %for.cond.for.end_crit_edge ], [ undef, %entry ]
  ret i32* %r.0.lcssa
}

!0 = !{!"minimal TBAA"}
!1 = !{!3, !3, i64 0}
!2 = !{!4, !4, i64 0}
!3 = !{!"float", !0}
!4 = !{!"int", !0}
