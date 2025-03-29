; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s

;#define N 20
;#include "limits.h"
;#include <stdio.h>
;int A[N];
;
;void single_do_loop_int_max_iterations() {
;  int i;
;
;  __sync_synchronize();
;
;  i = 0;
;
;  do {
;    A[0] = i;
;    ++i;
;  } while (i < INT_MAX);
;
;  __sync_synchronize();
;}
;
;int main () {
;  int i;
;
;  A[0] = 0;
;
;  single_do_loop_int_max_iterations();
;
;  fprintf(stdout, "Output %d\n", A[0]);
;
;  if (A[0] == INT_MAX - 1)
;    return 0;
;  else
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct._IO_FILE = type { i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32, i64, i16, i8, [1 x i8], ptr, i64, ptr, ptr, ptr, ptr, i64, i32, [20 x i8] }
%struct._IO_marker = type { ptr, ptr, i32 }

@A = common global [20 x i32] zeroinitializer, align 4 ; <ptr> [#uses=1]
@stdout = external global ptr       ; <ptr> [#uses=1]
@.str = private constant [11 x i8] c"Output %d\0A\00" ; <ptr> [#uses=1]

define void @single_do_loop_int_max_iterations() nounwind {
entry:
  fence seq_cst
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc, %do.cond ]  ; <i32> [#uses=2]
  store i32 %0, ptr @A
  %inc = add nsw i32 %0, 1                        ; <i32> [#uses=2]
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %exitcond = icmp ne i32 %inc, 2147483647        ; <i1> [#uses=1]
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
entry:
  store i32 0, ptr @A
  call void @single_do_loop_int_max_iterations()
  %tmp = load ptr, ptr @stdout          ; <ptr> [#uses=1]
  %tmp1 = load i32, ptr @A ; <i32> [#uses=1]
  %call = call i32 (ptr, ptr, ...) @fprintf(ptr %tmp, ptr @.str, i32 %tmp1) ; <i32> [#uses=0]
  %tmp2 = load i32, ptr @A ; <i32> [#uses=1]
  %cmp = icmp eq i32 %tmp2, 2147483646            ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %return

if.else:                                          ; preds = %entry
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

declare i32 @fprintf(ptr, ptr, ...)

; CHECK: for (int c0 = 0; c0 <= 2147483646; c0 += 1)
; CHECK:   Stmt_do_body(c0);
