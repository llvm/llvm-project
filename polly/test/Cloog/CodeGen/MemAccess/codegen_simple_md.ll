;RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+withconst -polly-codegen < %s -S | FileCheck -check-prefix=WITHCONST %s
;RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+withoutconst -polly-codegen < %s -S | FileCheck -check-prefix=WITHOUTCONST %s

;int A[1040];
;
;int codegen_simple_md() {
;  for (int i = 0; i < 32; ++i)
;    for (int j = 0; j < 32; ++j)
;      A[32*i+j] = 100;
;
;  return 0;
;}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-pc-linux-gnu"

@A = common global [1040 x i32] zeroinitializer, align 4

define i32 @codegen_simple_md() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc4, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc5, %for.inc4 ]
  %exitcond1 = icmp ne i32 %i.0, 32
  br i1 %exitcond1, label %for.body, label %for.end6

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 32
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %mul = shl nsw i32 %i.0, 5
  %add = add nsw i32 %mul, %j.0
  %arrayidx = getelementptr inbounds [1040 x i32]* @A, i32 0, i32 %add
  store i32 100, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc4

for.inc4:                                         ; preds = %for.end
  %inc5 = add nsw i32 %i.0, 1
  br label %for.cond

for.end6:                                         ; preds = %for.cond
  ret i32 0
}

; WITHCONST:  [[REG1:%[0-9]+]] = sext i32 %{{[0-9]+}} to i64
; WITHCONST:  %p_mul_coeff = mul i64 16, [[REG1]]
; WITHCONST:  %p_sum_coeff = add i64 5, %p_mul_coeff
; WITHCONST:  [[REG2:%[0-9]+]] = sext i32 %{{[0-9]+}} to i64
; WITHCONST:  %p_mul_coeff6 = mul i64 2, [[REG2]]
; WITHCONST:  %p_sum_coeff7 = add i64 %p_sum_coeff, %p_mul_coeff6
; WITHCONST:  %p_newarrayidx_ = getelementptr [1040 x i32]* @A, i64 0, i64 %p_sum_coeff7
; WITHCONST:  store i32 100, i32* %p_newarrayidx_

; WITHOUTCONST:  [[REG1:%[0-9]+]] = sext i32 %{{[0-9]+}} to i64
; WITHOUTCONST:  %p_mul_coeff = mul i64 16, [[REG1]]
; WITHOUTCONST:  %p_sum_coeff = add i64 0, %p_mul_coeff
; WITHOUTCONST:  [[REG2:%[0-9]+]] = sext i32 %{{[0-9]+}} to i64
; WITHOUTCONST:  %p_mul_coeff6 = mul i64 2, [[REG2]]
; WITHOUTCONST:  %p_sum_coeff7 = add i64 %p_sum_coeff, %p_mul_coeff6
; WITHOUTCONST:  %p_newarrayidx_ = getelementptr [1040 x i32]* @A, i64 0, i64 %p_sum_coeff7
; WITHOUTCONST:  store i32 100, i32* %p_newarrayidx_
