; RUN: opt -S -passes=loop-vectorize -mtriple=aarch64 -mattr=+sve \
; RUN:     -scalable-vectorization=on -force-vector-interleave=1 \
; RUN:     -vectorize-vector-loops < %s | FileCheck %s --check-prefix=IR
; RUN: opt -disable-output -passes=loop-vectorize -mtriple=aarch64 -mattr=+sve \
; RUN:     -scalable-vectorization=on -force-vector-interleave=1 \
; RUN:     -vectorize-vector-loops -pass-remarks-analysis=loop-vectorize \
; RUN:     < %s 2>&1 | FileCheck %s --check-prefix=REMARKS

; IR-LABEL: @insertelement_cost(
; IR-NOT: vector.body:
; REMARKS: loop not vectorized: instruction return type cannot be vectorized
define void @insertelement_cost(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
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

; IR-LABEL: @extractelement_cost(
; IR-NOT: vector.body:
; REMARKS: loop not vectorized: instruction return type cannot be vectorized
define void @extractelement_cost(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
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

; IR-LABEL: @shufflevector_cost(
; IR-NOT: vector.body:
; REMARKS: loop not vectorized: instruction return type cannot be vectorized
define void @shufflevector_cost(ptr noalias nocapture noundef writeonly %a, ptr nocapture noundef readonly %b) {
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds <4 x i32>, ptr %b, i64 %indvars.iv
  %0 = load <4 x i32>, ptr %arrayidx, align 16
  %result = shufflevector <4 x i32> %0, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %arrayidx2 = getelementptr inbounds <4 x i32>, ptr %a, i64 %indvars.iv
  store <4 x i32> %result, ptr %arrayidx2, align 16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
