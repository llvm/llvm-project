; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s
; XFAIL: *

;#define N 20
;#include "limits.h"
;
;int main () {
;  int i;
;  int A[N];
;
;  A[0] = 1;
;
;  __sync_synchronize();
;
;  i = 0;
;
;  do {
;    A[0] = 0;
;    ++i;
;  } while (i < 1);
;
;  __sync_synchronize();
;
;  if (A[0] == 0)
;    return 0;
;  else
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @main() nounwind {
entry:
  %A = alloca [20 x i32], align 4                 ; <ptr> [#uses=3]
  %arraydecay = getelementptr inbounds [20 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx = getelementptr inbounds i32, ptr %arraydecay, i64 0 ; <ptr> [#uses=1]
  store i32 1, ptr %arrayidx
  fence seq_cst
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %arraydecay1 = getelementptr inbounds [20 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx2 = getelementptr inbounds i32, ptr %arraydecay1, i64 0 ; <ptr> [#uses=1]
  store i32 0, ptr %arrayidx2
  br label %do.cond

do.cond:                                          ; preds = %do.body
  br i1 false, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  fence seq_cst
  %arraydecay4 = getelementptr inbounds [20 x i32], ptr %A, i32 0, i32 0 ; <ptr> [#uses=1]
  %arrayidx5 = getelementptr inbounds i32, ptr %arraydecay4, i64 0 ; <ptr> [#uses=1]
  %tmp6 = load i32, ptr %arrayidx5                    ; <i32> [#uses=1]
  %cmp7 = icmp eq i32 %tmp6, 0                    ; <i1> [#uses=1]
  br i1 %cmp7, label %if.then, label %if.else

if.then:                                          ; preds = %do.end
  br label %return

if.else:                                          ; preds = %do.end
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

; CHECK: S0(0)
