; RUN: opt %loadPolly -polly-prepare -S < %s | FileCheck %s
; RUN: opt %loadPolly -polly-prepare -S -polly-codegen-scev < %s | FileCheck %s

; void f(long A[], long N) {
;   long i;
;   for (i = 0; i < N; ++i)
;     A[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i64* %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %for.i
; CHECK: entry:
; CHECK: %value.reg2mem = alloca i64
; CHECK: br label %entry.split

for.i:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %merge ]
  %scevgep = getelementptr i64* %A, i64 %indvar
  %cmp = icmp eq i64 %indvar, 3
  br i1 %cmp, label %then, label %else

then:
  %add_two = add i64 %indvar, 2
  br label %merge
; CHECK: then:
; CHECK:   %add_two = add i64 %indvar, 2
; CHECK:   store i64 %add_two, i64* %value.reg2mem
; CHECK:   br label %merge

else:
  %add_three = add i64 %indvar, 4
  br label %merge
; CHECK: else:
; CHECK:   %add_three = add i64 %indvar, 4
; CHECK:   store i64 %add_three, i64* %value.reg2mem
; CHECK:   br label %merge

merge:
  %value = phi i64 [ %add_two, %then ], [ %add_three, %else ]
  store i64 %value, i64* %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}
