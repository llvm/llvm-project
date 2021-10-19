; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-target-throughput-vector-fma=1 \
; RUN: -polly-target-latency-vector-fma=8 \
; RUN: -analyze -polly-ast -polly-target-1st-cache-level-associativity=8 \
; RUN: -polly-target-2nd-cache-level-associativity=8 \
; RUN: -polly-target-1st-cache-level-size=32768 \
; RUN: -polly-target-vector-register-bitwidth=256 \
; RUN: -polly-target-2nd-cache-level-size=262144 < %s
;
; Test whether isolation works as expected.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @kernel_gemm(i32 %ni, i32 %nj, i32 %nk, i8 signext %alpha, i8 signext %beta, [1020 x i8]* %C, [1020 x i8]* %A, [1020 x i8]* %B) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.inc23, %entry.split
  %indvars.iv45 = phi i64 [ 0, %entry.split ], [ %indvars.iv.next46, %for.inc23 ]
  br label %for.body3

for.body3:                                        ; preds = %for.inc20, %for.body
  %indvars.iv42 = phi i64 [ 0, %for.body ], [ %indvars.iv.next43, %for.inc20 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.body3
  %indvars.iv = phi i64 [ 0, %for.body3 ], [ %indvars.iv.next, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [1020 x i8], [1020 x i8]* %A, i64 %indvars.iv45, i64 %indvars.iv
  %tmp = load i8, i8* %arrayidx8, align 1
  %arrayidx12 = getelementptr inbounds [1020 x i8], [1020 x i8]* %B, i64 %indvars.iv, i64 %indvars.iv42
  %tmp1 = load i8, i8* %arrayidx12, align 1
  %mul = mul i8 %tmp1, %tmp
  %arrayidx17 = getelementptr inbounds [1020 x i8], [1020 x i8]* %C, i64 %indvars.iv45, i64 %indvars.iv42
  %tmp2 = load i8, i8* %arrayidx17, align 1
  %add = add i8 %mul, %tmp2
  store i8 %add, i8* %arrayidx17, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1020
  br i1 %exitcond, label %for.body6, label %for.inc20

for.inc20:                                        ; preds = %for.body6
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond44 = icmp ne i64 %indvars.iv.next43, 1020
  br i1 %exitcond44, label %for.body3, label %for.inc23

for.inc23:                                        ; preds = %for.inc20
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next46, 1020
  br i1 %exitcond47, label %for.body, label %for.end25

for.end25:                                        ; preds = %for.inc23
  ret void
}
