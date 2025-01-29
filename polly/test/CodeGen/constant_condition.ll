;RUN: opt %loadNPMPolly '-passes=polly-prepare,scop(print<polly-ast>)' -disable-output < %s 2>&1 | FileCheck %s

;#include <string.h>
;int A[1];
;
;void constant_condition () {
;  int a = 0;
;  int b = 0;
;
;  if (a == b)
;    A[0] = 0;
;  else
;    A[0] = 1;
;}
;
;int main () {
;  int i;
;
;  A[0] = 2;
;
;  constant_condition();
;
;  return A[0];
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
@A = common global [1 x i32] zeroinitializer, align 4 ; <ptr> [#uses=1]

define void @constant_condition() nounwind {
bb:
  %tmp = icmp eq i32 0, 0                         ; <i1> [#uses=0]
  br i1 true, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  store i32 0, ptr @A
  br label %bb3

bb2:                                              ; preds = %bb
  store i32 1, ptr @A
  br label %bb3

bb3:                                              ; preds = %bb2, %bb1
  ret void
}

define i32 @main() nounwind {
bb:
  store i32 2, ptr @A
  call void @constant_condition()
  %tmp = load i32, ptr @A ; <i32> [#uses=1]
  ret i32 %tmp
}


; CHECK: Stmt_bb1();
