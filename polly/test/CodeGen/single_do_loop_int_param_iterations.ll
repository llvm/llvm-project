; RUN: opt %loadPolly -polly-ast -S -analyze  < %s | FileCheck %s
; XFAIL: *

;define N 20
;#include "limits.h"
;int A[N];
;
;void bar (int n) {
;  int i;
;  __sync_synchronize();
;  i = 0;
;
;  do {
;    A[0] = i;
;    ++i;
;  } while (i < 2 * n);
;  __sync_synchronize();
;}
;
;int main () {
;  A[0] = 0;
;  bar (N/2);
;
;  if (A[0] == N - 1 )
;    return 0;
;  else
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [20 x i32] zeroinitializer, align 4 ; <[20 x i32]*> [#uses=1]

define void @bar(i32 %n) nounwind {
entry:
  fence seq_cst
  %tmp = mul i32 %n, 2                            ; <i32> [#uses=2]
  %tmp1 = icmp sgt i32 %tmp, 1                    ; <i1> [#uses=1]
  %smax = select i1 %tmp1, i32 %tmp, i32 1        ; <i32> [#uses=1]
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %do.cond ]  ; <i32> [#uses=2]
  store i32 %0, i32* getelementptr inbounds ([20 x i32], [20 x i32]* @A, i32 0, i32 0)
  %inc = add nsw i32 %0, 1                        ; <i32> [#uses=2]
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %exitcond = icmp ne i32 %inc, %smax             ; <i1> [#uses=1]
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
entry:
  store i32 0, i32* getelementptr inbounds ([20 x i32], [20 x i32]* @A, i32 0, i32 0)
  call void @bar(i32 10)
  %tmp = load i32, i32* getelementptr inbounds ([20 x i32], [20 x i32]* @A, i32 0, i32 0) ; <i32> [#uses=1]
  %cmp = icmp eq i32 %tmp, 19                     ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %return

if.else:                                          ; preds = %entry
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}
; CHECK: Scop: do.body => do.end

