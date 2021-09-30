; RUN: opt %loadPolly -polly-ast -analyze  -S < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen < %s

;#define N 20
;#include "limits.h"
;long long A[N];
;
;int main () {
;  long long i;
;
;  A[0] = 0;
;
;  __sync_synchronize();
;
;  i = 0;
;
;  do {
;    A[0] = i;
;    ++i;
;  } while (i < LLONG_MAX);
;
;  __sync_synchronize();
;
;  if (A[0] == LLONG_MAX - 1)
;    return 0;
;  else
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [20 x i64] zeroinitializer, align 8 ; <[20 x i64]*> [#uses=1]

define i32 @main() nounwind {
entry:
  store i64 0, i64* getelementptr inbounds ([20 x i64], [20 x i64]* @A, i32 0, i32 0)
  fence seq_cst
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %0 = phi i64 [ 0, %entry ], [ %inc, %do.cond ]  ; <i64> [#uses=2]
  store i64 %0, i64* getelementptr inbounds ([20 x i64], [20 x i64]* @A, i32 0, i32 0)
  %inc = add nsw i64 %0, 1                        ; <i64> [#uses=2]
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %exitcond = icmp ne i64 %inc, 9223372036854775807 ; <i1> [#uses=1]
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  fence seq_cst
  %tmp3 = load i64, i64* getelementptr inbounds ([20 x i64], [20 x i64]* @A, i32 0, i32 0) ; <i64> [#uses=1]
  %cmp4 = icmp eq i64 %tmp3, 9223372036854775806  ; <i1> [#uses=1]
  br i1 %cmp4, label %if.then, label %if.else

if.then:                                          ; preds = %do.end
  br label %return

if.else:                                          ; preds = %do.end
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

; CHECK: for (int c0 = 0; c0 <= 9223372036854775806; c0 += 1)
; CHECK:   Stmt_do_body(c0);
