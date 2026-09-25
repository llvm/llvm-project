; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<ast>' -polly-print-ast -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AST
;
; Both user assumptions are derived from (trunc i64 %n to i8), which is only
; modeled correctly if the truncation does not overflow. SCEVAffinator caches
; the translation of the truncation, so the precondition is recorded only for
; the first assumption. The second assumption must nevertheless not be added
; to the context: this would remove the 'n >= 128' runtime check from the
; invalid context and the 'n <= 9' condition from the domain of Stmt_then.
; For n = 261, both assumptions hold but Stmt_then must not be executed.
;
; void f(int *A, long n) {
;   __builtin_assume((signed char)n >= 0);
;   __builtin_assume((signed char)n < 10);
;   for (long i = 0; i < 4; i++)
;     if (n < 10)
;       A[i] = 1;
; }

; CHECK:      Context:
; CHECK-NEXT:   [n] -> {  : -9223372036854775808 <= n <= 9223372036854775807 }
; CHECK:      Invalid Context:
; CHECK-NEXT:   [n] -> {  : n <= -129 or n >= 128 }
; CHECK:      Defined Behavior Context:
; CHECK-NEXT:   [n] -> {  : 0 <= n <= 9 }
; CHECK:      Domain :=
; CHECK-NEXT:   [n] -> { Stmt_then[i0] : n <= 9 and 0 <= i0 <= 3 };

; AST:      if (1 && 0 == (n <= -129 || n >= 128))
; AST-EMPTY:
; AST-NEXT:     if (n <= 9)
; AST-NEXT:       for (int c0 = 0; c0 <= 3; c0 += 1)
; AST-NEXT:         Stmt_then(c0);

define void @f(ptr %A, i64 %n) {
entry:
  %t = trunc i64 %n to i8
  %c1 = icmp sge i8 %t, 0
  call void @llvm.assume(i1 %c1)
  %c2 = icmp slt i8 %t, 10
  call void @llvm.assume(i1 %c2)
  br label %for

for:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %small = icmp slt i64 %n, 10
  br i1 %small, label %then, label %latch

then:
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
