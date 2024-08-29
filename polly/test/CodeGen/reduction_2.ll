; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -polly-invariant-load-hoisting=true '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s --allow-empty

;#include <string.h>
;#include <stdio.h>
;#define N 1021
;
;int main () {
;  int i;
;  int A[N];
;  int RED[1];
;
;  memset(A, 0, sizeof(int) * N);
;
;  A[0] = 1;
;  A[1] = 1;
;  RED[0] = 0;
;
;  for (i = 2; i < N; i++) {
;    A[i] = A[i-1] + A[i-2];
;    RED[0] += A[i-2];
;  }
;
;  if (RED[0] != 382399368)
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @main() nounwind {
entry:
  %A = alloca [1021 x i32], align 4               ; <ptr> [#uses=6]
  %RED = alloca [1 x i32], align 4                ; <ptr> [#uses=3]
  %arraydecay = getelementptr inbounds [1021 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  call void @llvm.memset.p0.i64(ptr %arraydecay, i8 0, i64 4084, i32 1, i1 false)
  %arraydecay1 = getelementptr inbounds [1021 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx = getelementptr inbounds i32, ptr %arraydecay1, i64 0 ; <ptr> [#uses=1]
  store i32 1, ptr %arrayidx
  %arraydecay2 = getelementptr inbounds [1021 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx3 = getelementptr inbounds i32, ptr %arraydecay2, i64 1 ; <ptr> [#uses=1]
  store i32 1, ptr %arrayidx3
  %arraydecay4 = getelementptr inbounds [1 x i32], ptr %RED, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx5 = getelementptr inbounds i32, ptr %arraydecay4, i64 0 ; <ptr> [#uses=1]
  store i32 0, ptr %arrayidx5
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ] ; <i64> [#uses=5]
  %arrayidx15 = getelementptr [1021 x i32], ptr %A, i64 0, i64 %indvar ; <ptr> [#uses=2]
  %tmp = add i64 %indvar, 2                       ; <i64> [#uses=1]
  %arrayidx20 = getelementptr [1021 x i32], ptr %A, i64 0, i64 %tmp ; <ptr> [#uses=1]
  %tmp1 = add i64 %indvar, 1                      ; <i64> [#uses=1]
  %arrayidx9 = getelementptr [1021 x i32], ptr %A, i64 0, i64 %tmp1 ; <ptr> [#uses=1]
  %exitcond = icmp ne i64 %indvar, 1019           ; <i1> [#uses=1]
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp10 = load i32, ptr %arrayidx9                   ; <i32> [#uses=1]
  %tmp16 = load i32, ptr %arrayidx15                  ; <i32> [#uses=1]
  %add = add nsw i32 %tmp10, %tmp16               ; <i32> [#uses=1]
  store i32 %add, ptr %arrayidx20
  %tmp26 = load i32, ptr %arrayidx15                  ; <i32> [#uses=1]
  %arraydecay27 = getelementptr inbounds [1 x i32], ptr %RED, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx28 = getelementptr inbounds i32, ptr %arraydecay27, i64 0 ; <ptr> [#uses=2]
  %tmp29 = load i32, ptr %arrayidx28                  ; <i32> [#uses=1]
  %add30 = add nsw i32 %tmp29, %tmp26             ; <i32> [#uses=1]
  store i32 %add30, ptr %arrayidx28
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %arraydecay32 = getelementptr inbounds [1 x i32], ptr %RED, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx33 = getelementptr inbounds i32, ptr %arraydecay32, i64 0 ; <ptr> [#uses=1]
  %tmp34 = load i32, ptr %arrayidx33                  ; <i32> [#uses=1]
  %cmp35 = icmp ne i32 %tmp34, 382399368          ; <i1> [#uses=1]
  br i1 %cmp35, label %if.then, label %if.end

if.then:                                          ; preds = %for.end
  br label %if.end

if.end:                                           ; preds = %if.then, %for.end
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %for.end ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i32, i1) nounwind

; This is a negative test. We can prove that RED[0] in the conditional after
; the loop is dereferencable and consequently expand the SCoP from the
; loop to include the conditional. However, during SCoP generation we realize
; that, while RED[0] is invariant, it is written to as part of the same scop
; and can consequently not be hoisted. Hence, we invalidate the scop.
;
; CHECK-NOT: for (int c0 = 0; c0 <= 1018; c0 += 1)
