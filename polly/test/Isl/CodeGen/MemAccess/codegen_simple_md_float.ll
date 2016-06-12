;RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+withconst -polly-codegen < %s -S | FileCheck -check-prefix=WITHCONST %s
;RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+withoutconst -polly-codegen < %s -S | FileCheck -check-prefix=WITHOUTCONST %s
;
;float A[1040];
;
;int codegen_simple_md() {
;  for (int i = 0; i < 32; ++i)
;    for (int j = 0; j < 32; ++j)
;      A[32*i+j] = 100;
;}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

@A = common global [1040 x float] zeroinitializer, align 4

define void @codegen_simple_md() nounwind {
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
  %arrayidx = getelementptr inbounds [1040 x float], [1040 x float]* @A, i32 0, i32 %add
  store float 100.0, float* %arrayidx, align 4
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
  ret void
}

; WITHCONST:  %[[IVOut:polly.indvar[0-9]*]] = phi i64 [ 0, %polly.loop_preheader{{[0-9]*}} ], [ %polly.indvar_next{{[0-9]*}}, %polly.{{[._a-zA-Z0-9]*}} ]
; WITHCONST:  %[[IVIn:polly.indvar[0-9]*]] = phi i64 [ 0, %polly.loop_preheader{{[0-9]*}} ], [ %polly.indvar_next{{[0-9]*}}, %polly.{{[._a-zA-Z0-9]*}} ]
; WITHCONST:  %[[MUL1:[._a-zA-Z0-9]+]] = mul nsw i64 16, %[[IVOut]]
; WITHCONST:  %[[MUL2:[._a-zA-Z0-9]+]] = mul nsw i64 2, %[[IVIn]]
; WITHCONST:  %[[SUM1:[._a-zA-Z0-9]+]] = add nsw i64 %[[MUL1]], %[[MUL2]]
; WITHCONST:  %[[SUM2:[._a-zA-Z0-9]+]] = add nsw i64 %[[SUM1]], 5
; WITHCONST:  %[[ACC:[._a-zA-Z0-9]*]] = getelementptr float, float* getelementptr inbounds ([1040 x float], [1040 x float]* @A, i{{(32|64)}} 0, i{{(32|64)}} 0), i64 %[[SUM2]]
; WITHCONST:  store float 1.000000e+02, float* %[[ACC]]

; WITHOUTCONST:  %[[IVOut:polly.indvar[0-9]*]] = phi i64 [ 0, %polly.loop_preheader{{[0-9]*}} ], [ %polly.indvar_next{{[0-9]*}}, %polly.{{[._a-zA-Z0-9]*}} ]
; WITHOUTCONST:  %[[IVIn:polly.indvar[0-9]*]] = phi i64 [ 0, %polly.loop_preheader{{[0-9]*}} ], [ %polly.indvar_next{{[0-9]*}}, %polly.{{[._a-zA-Z0-9]*}} ]
; WITHOUTCONST:  %[[MUL1:[._a-zA-Z0-9]+]] = mul nsw i64 16, %[[IVOut]]
; WITHOUTCONST:  %[[MUL2:[._a-zA-Z0-9]+]] = mul nsw i64 2, %[[IVIn]]
; WITHOUTCONST:  %[[SUM1:[._a-zA-Z0-9]+]] = add nsw i64 %[[MUL1]], %[[MUL2]]
; WITHOUTCONST:  %[[ACC:[._a-zA-Z0-9]*]] = getelementptr float, float* getelementptr inbounds ([1040 x float], [1040 x float]* @A, i{{(32|64)}} 0, i{{(32|64)}} 0), i64 %[[SUM1]]
; WITHOUTCONST:  store float 1.000000e+02, float* %[[ACC]]
