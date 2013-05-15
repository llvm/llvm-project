; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze < %s | FileCheck %s

;#define N 20
;#include "limits.h"
;#include <stdio.h>
;int A[2 * N];
;
;void single_do_loop_scev_replace() {
;  int i;
;
;  __sync_synchronize();
;
;  i = 0;
;
;  do {
;    A[2 * i] = i;
;    ++i;
;  } while (i < N);
;
;  __sync_synchronize();
;}
;
;int main () {
;  int i;
;
;  single_do_loop_scev_replace();
;
;  fprintf(stdout, "Output %d\n", A[0]);
;
;  if (A[2 * N - 2] == N - 1)
;    return 0;
;  else
;    return 1;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }

@A = common global [40 x i32] zeroinitializer, align 4 ; <[40 x i32]*> [#uses=3]
@stdout = external global %struct._IO_FILE*       ; <%struct._IO_FILE**> [#uses=1]
@.str = private constant [11 x i8] c"Output %d\0A\00" ; <[11 x i8]*> [#uses=1]

define void @single_do_loop_scev_replace() nounwind {
entry:
  fence seq_cst
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %indvar = phi i64 [ %indvar.next, %do.cond ], [ 0, %entry ] ; <i64> [#uses=3]
  %tmp = mul i64 %indvar, 2                       ; <i64> [#uses=1]
  %arrayidx = getelementptr [40 x i32]* @A, i64 0, i64 %tmp ; <i32*> [#uses=1]
  %i.0 = trunc i64 %indvar to i32                 ; <i32> [#uses=1]
  br label %do.cond

do.cond:                                          ; preds = %do.body
  store i32 %i.0, i32* %arrayidx
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp ne i64 %indvar.next, 20        ; <i1> [#uses=1]
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  fence seq_cst
  ret void
}

define i32 @main() nounwind {
entry:
  call void @single_do_loop_scev_replace()
  %tmp = load %struct._IO_FILE** @stdout          ; <%struct._IO_FILE*> [#uses=1]
  %tmp1 = load i32* getelementptr inbounds ([40 x i32]* @A, i32 0, i32 0) ; <i32> [#uses=1]
  %call = call i32 (%struct._IO_FILE*, i8*, ...)* @fprintf(%struct._IO_FILE* %tmp, i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i32 %tmp1) ; <i32> [#uses=0]
  %tmp2 = load i32* getelementptr inbounds ([40 x i32]* @A, i32 0, i64 38) ; <i32> [#uses=1]
  %cmp = icmp eq i32 %tmp2, 19                    ; <i1> [#uses=1]
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %return

if.else:                                          ; preds = %entry
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ 0, %if.then ], [ 1, %if.else ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...)

; CHECK:   for (c2=0;c2<=19;c2++) {
; CHECK:     Stmt_do_cond(c2);
; CHECK:   }

