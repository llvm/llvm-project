; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -vectorize-vector-loops \
; RUN:   -mtriple=aarch64 -mattr=+sve2p1 -debug-only=loop-vectorize \
; RUN:   -disable-output -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,VECTORIZE
; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=2 \
; RUN:   -vectorize-vector-loops -mtriple=aarch64 -mattr=+sve2p1 \
; RUN:   -debug-only=loop-vectorize -disable-output -S < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,INTERLEAVE

; CHECK-LABEL: test_store_fcmp
; VECTORIZE: Cost of Invalid for VF vscale x 1: WIDEN store vp<%6>, ir<%resi1>
; INTERLEAVE: Executing best plan with VF=1, UF=2
define void @test_store_fcmp(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <8 x half>, ptr %b, i64 %indvars.iv
  %0 = load <8 x half>, ptr %arrayidx, align 16
  %resi1 = fcmp fast ule <8 x half> %0, zeroinitializer
  %arrayidx2 = getelementptr inbounds <8 x i1>, ptr %a, i64 %indvars.iv
  store <8 x i1> %resi1, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: insertelement_cost
; VECTORIZE: Cost of Invalid for VF vscale x 1: REPLICATE ir<%result> = insertelement ir<zeroinitializer>, ir<%0>, ir<0>
; INTERLEAVE: Executing best plan with VF=1, UF=2
define void @insertelement_cost(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i16, ptr %b, i64 %indvars.iv
  %0 = load i16, ptr %arrayidx, align 16
  %result = insertelement <8 x i16> zeroinitializer, i16 %0, i32 0
  %arrayidx2 = getelementptr inbounds <8 x i16>, ptr %a, i64 %indvars.iv
  store <8 x i16> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: extractelement_cost
; CHECK: LV: Not vectorizing: Found unvectorizable type %result = extractelement <4 x i32> %0, i32 1
define void @extractelement_cost(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %result = extractelement <4 x i32> %0, i32 1
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
