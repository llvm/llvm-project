; RUN: opt %loadPolly -scalar-evolution-max-value-compare-depth=3 -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -scalar-evolution-max-value-compare-depth=3 -polly-codegen -polly-invariant-load-hoisting=true -disable-output < %s
;
; Stress test for the code generation of invariant accesses.
;
;    void f(int *I0, int *I1, int *I2, int *V, long p0, long p1, long p2, long p3) {
;      *V = *I1;
;      for (int i = 0; i < 1000; i++) {
;        long n0 = p0 * *I1 + p1 * *I1;
;        V[i] = I0[n0];
;        long m0 = p0 * (I2[0]);
;        long m1 = p1 * (I2[1]);
;        long m2 = p2 * (I2[2]);
;        long m3 = p3 * (I2[3]);
;        int j = 0;
;        do {
;          if (j > 0) {
;            V[i] += I1[m0 + m2];
;            V[i] += I1[n0];
;          }
;        } while (j++ < m1 + m3 * n0);
;      }
;    }
;
; CHECK: p0: ((sext i32 %tmp6 to i64) * %p1)
; CHECK: p1: ((sext i32 %tmp3 to i64) * (sext i32 %tmp8 to i64) * (%p0 + %p1) * %p3)
; CHECK: p2: ((sext i32 %tmp3 to i64) * (%p0 + %p1))
; CHECK: p3: ((sext i32 %tmp5 to i64) * %p0)
; CHECK: p4: ((sext i32 %tmp7 to i64) * %p2)
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %I0, i32* %I1, i32* %I2, i32* %V, i64 %p0, i64 %p1, i64 %p2, i64 %p3) {
entry:
  %tmp = load i32, i32* %I1, align 4
  store i32 %tmp, i32* %V, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv1, 1000
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp3 = load i32, i32* %I1, align 4
  %conv = sext i32 %tmp3 to i64
  %mul = mul nsw i64 %conv, %p0
  %conv1 = sext i32 %tmp3 to i64
  %mul2 = mul nsw i64 %conv1, %p1
  %add = add nsw i64 %mul, %mul2
  %arrayidx = getelementptr inbounds i32, i32* %I0, i64 %add
  %tmp4 = load i32, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %V, i64 %indvars.iv1
  store i32 %tmp4, i32* %arrayidx3, align 4
  %tmp5 = load i32, i32* %I2, align 4
  %conv5 = sext i32 %tmp5 to i64
  %mul6 = mul nsw i64 %conv5, %p0
  %arrayidx7 = getelementptr inbounds i32, i32* %I2, i64 1
  %tmp6 = load i32, i32* %arrayidx7, align 4
  %conv8 = sext i32 %tmp6 to i64
  %mul9 = mul nsw i64 %conv8, %p1
  %arrayidx10 = getelementptr inbounds i32, i32* %I2, i64 2
  %tmp7 = load i32, i32* %arrayidx10, align 4
  %conv11 = sext i32 %tmp7 to i64
  %mul12 = mul nsw i64 %conv11, %p2
  %arrayidx13 = getelementptr inbounds i32, i32* %I2, i64 3
  %tmp8 = load i32, i32* %arrayidx13, align 4
  %conv14 = sext i32 %tmp8 to i64
  %mul15 = mul nsw i64 %conv14, %p3
  br label %do.body

do.body:                                          ; preds = %do.cond, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.cond ], [ 0, %for.body ]
  %cmp16 = icmp sgt i64 %indvars.iv, 0
  br i1 %cmp16, label %if.then, label %if.end

if.then:                                          ; preds = %do.body
  %add18 = add nsw i64 %mul6, %mul12
  %arrayidx19 = getelementptr inbounds i32, i32* %I1, i64 %add18
  %tmp9 = load i32, i32* %arrayidx19, align 4
  %arrayidx21 = getelementptr inbounds i32, i32* %V, i64 %indvars.iv1
  %tmp10 = load i32, i32* %arrayidx21, align 4
  %add22 = add nsw i32 %tmp10, %tmp9
  store i32 %add22, i32* %arrayidx21, align 4
  %arrayidx23 = getelementptr inbounds i32, i32* %I1, i64 %add
  %tmp11 = load i32, i32* %arrayidx23, align 4
  %arrayidx25 = getelementptr inbounds i32, i32* %V, i64 %indvars.iv1
  %tmp12 = load i32, i32* %arrayidx25, align 4
  %add26 = add nsw i32 %tmp12, %tmp11
  store i32 %add26, i32* %arrayidx25, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %do.body
  br label %do.cond

do.cond:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %mul28 = mul nsw i64 %mul15, %add
  %add29 = add nsw i64 %mul9, %mul28
  %cmp30 = icmp slt i64 %indvars.iv, %add29
  br i1 %cmp30, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  br label %for.inc

for.inc:                                          ; preds = %do.end
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
