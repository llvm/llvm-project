;RUN: opt %loadPolly -polly-print-import-jscop -polly-import-jscop-postfix=transformed -disable-output < %s | FileCheck %s
;RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-postfix=transformed -polly-codegen -polly-vectorizer=polly -S < %s | FileCheck %s --check-prefix=JSCOPVEC
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"

@A = common global [100 x i32] zeroinitializer, align 4
@B = common global [100 x i32] zeroinitializer, align 4

define i32 @simple() nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %arrayidx = getelementptr [100 x i32], ptr @A, i32 0, i32 %0
  %exitcond1 = icmp ne i32 %0, 12
  br i1 %exitcond1, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 %0, ptr %arrayidx
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  fence seq_cst
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc11, %for.end
  %1 = phi i32 [ 0, %for.end ], [ %inc13, %for.inc11 ]
  %arrayidx10 = getelementptr [100 x i32], ptr @B, i32 0, i32 %1
  %exitcond = icmp ne i32 %1, 12
  br i1 %exitcond, label %for.body7, label %for.end14

for.body7:                                        ; preds = %for.cond4
  store i32 %1, ptr %arrayidx10
  br label %for.inc11

for.inc11:                                        ; preds = %for.body7
  %inc13 = add nsw i32 %1, 1
  br label %for.cond4

for.end14:                                        ; preds = %for.cond4
  ret i32 0
}
; CHECK-DAG: New access function '{ Stmt_for_body7[i0] -> MemRef_B[0] }' detected in JSCOP file
; CHECK-DAG: New access function '{ Stmt_for_body[i0] -> MemRef_A[0] }' detected in JSCOP file

; Verify that the new access function (see above) is actually used during vector code generation.

; JSCOPVEC:  store i32 0, ptr @B
; JSCOPVEC:  store i32 1, ptr @B
; JSCOPVEC:  store i32 2, ptr @B
; JSCOPVEC:  store i32 3, ptr @B
; JSCOPVEC:  store i32 4, ptr @B
; JSCOPVEC:  store i32 5, ptr @B
; JSCOPVEC:  store i32 6, ptr @B
; JSCOPVEC:  store i32 7, ptr @B
; JSCOPVEC:  store i32 8, ptr @B
; JSCOPVEC:  store i32 9, ptr @B
; JSCOPVEC:  store i32 10, ptr @B
; JSCOPVEC:  store i32 11, ptr @B

; JSCOPVEC:  store i32 0, ptr @A
; JSCOPVEC:  store i32 1, ptr @A
; JSCOPVEC:  store i32 2, ptr @A
; JSCOPVEC:  store i32 3, ptr @A
; JSCOPVEC:  store i32 4, ptr @A
; JSCOPVEC:  store i32 5, ptr @A
; JSCOPVEC:  store i32 6, ptr @A
; JSCOPVEC:  store i32 7, ptr @A
; JSCOPVEC:  store i32 8, ptr @A
; JSCOPVEC:  store i32 9, ptr @A
; JSCOPVEC:  store i32 10, ptr @A
; JSCOPVEC:  store i32 11, ptr @A
