; RUN: opt %loadPolly %defaultOpts -polly-cloog -analyze < %s | FileCheck %s

;#include <string.h>
;#define N 1024
;
;int main () {
;  int i;
;  int A[N];
;
;  memset(A, 0, sizeof(int) * N);
;
;  for (i = 0; i < N; i++) {
;    A[i] = 1;
;  }
;
;  for (i = 0; i < N; i++)
;    if (A[i] != 1)
;      return 1;
;
;  return 0;
;}
;

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() nounwind {
entry:
  %A = alloca [1024 x i32], align 4               ; <[1024 x i32]*> [#uses=3]
  %arraydecay = getelementptr inbounds [1024 x i32]* %A, i32 0, i32 0 ; <i32*> [#uses=1]
  %conv = bitcast i32* %arraydecay to i8*         ; <i8*> [#uses=1]
  call void @llvm.memset.p0i8.i64(i8* %conv, i8 0, i64 4096, i32 1, i1 false)
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar1 = phi i64 [ %indvar.next2, %for.inc ], [ 0, %entry ] ; <i64> [#uses=3]
  %arrayidx = getelementptr [1024 x i32]* %A, i64 0, i64 %indvar1 ; <i32*> [#uses=1]
  %exitcond = icmp ne i64 %indvar1, 1024          ; <i1> [#uses=1]
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 1, i32* %arrayidx
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvar.next2 = add i64 %indvar1, 1             ; <i64> [#uses=1]
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond5

for.cond5:                                        ; preds = %for.inc17, %for.end
  %indvar = phi i64 [ %indvar.next, %for.inc17 ], [ 0, %for.end ] ; <i64> [#uses=3]
  %arrayidx13 = getelementptr [1024 x i32]* %A, i64 0, i64 %indvar ; <i32*> [#uses=1]
  %i.1 = trunc i64 %indvar to i32                 ; <i32> [#uses=1]
  %cmp7 = icmp slt i32 %i.1, 1024                 ; <i1> [#uses=1]
  br i1 %cmp7, label %for.body9, label %for.end20

for.body9:                                        ; preds = %for.cond5
  %tmp14 = load i32* %arrayidx13                  ; <i32> [#uses=1]
  %cmp15 = icmp ne i32 %tmp14, 1                  ; <i1> [#uses=1]
  br i1 %cmp15, label %if.then, label %if.end

if.then:                                          ; preds = %for.body9
  br label %return

if.end:                                           ; preds = %for.body9
  br label %for.inc17

for.inc17:                                        ; preds = %if.end
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %for.cond5

for.end20:                                        ; preds = %for.cond5
  br label %return

return:                                           ; preds = %for.end20, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %for.end20 ] ; <i32> [#uses=1]
  ret i32 %retval.0
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind

; CHECK:for (c2=0;c2<=1023;c2++) {
; CHECK:    Stmt_for_body(c2);
; CHECK:}
