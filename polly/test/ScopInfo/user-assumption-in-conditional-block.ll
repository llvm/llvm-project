; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<ast>' -polly-print-ast -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AST
;
; https://github.com/llvm/llvm-project/issues/226419
;
; The user assumption itself has no preconditions, but it is located in a
; block whose domain is derived from (trunc i64 %n to i8), which is only
; modeled correctly if the truncation does not overflow. Adding the assumption
; to the context therefore relies on the 'n <= -129 or n >= 128' runtime
; check, but the context is used to gist the invalid context, which removes
; 'n <= -129' from it.
; For n = -200, (signed char)n == 56, hence neither the assumption nor the
; store is executed. The optimized code must not execute Stmt_then.
;
; void f(int *A, long n) {
;   for (long i = 0; i < 4; i++)
;     if ((signed char)n < 10) {
;       __builtin_assume(n >= 0);
;       A[i] = 1;
;     }
; }

; CHECK:      Context:
; CHECK-NEXT:   [n] -> {  : -9223372036854775808 <= n <= 9223372036854775807 }
; CHECK:      Invalid Context:
; CHECK-NEXT:   [n] -> {  : n <= -129 or n >= 128 }
; CHECK:      Defined Behavior Context:
; CHECK-NEXT:   [n] -> {  : 0 <= n <= 127 }
; CHECK:      Domain :=
; CHECK-NEXT:   [n] -> { Stmt_then[i0] : n <= 9 and 0 <= i0 <= 3 };

; AST: if (1 && 0 == (n <= -129 || n >= 128))

define void @f(ptr %A, i64 %n) {
entry:
  br label %for

for:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %t = trunc i64 %n to i8
  %small = icmp slt i8 %t, 10
  br i1 %small, label %then, label %latch

then:
  %c = icmp sge i64 %n, 0
  call void @llvm.assume(i1 %c)
  %gep = getelementptr inbounds i32, ptr %A, i64 %i
  store i32 1, ptr %gep
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.next, 4
  br i1 %cmp, label %for, label %exit

exit:
  ret void
}

declare void @llvm.assume(i1)
