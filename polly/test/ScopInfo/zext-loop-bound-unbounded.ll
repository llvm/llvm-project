; RUN: opt %loadNPMPolly '-passes=polly-custom<scops>' -polly-print-scops -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=polly-custom<ast>' -polly-print-ast -disable-output < %s 2>&1 | FileCheck %s --check-prefix=AST
;
; https://github.com/llvm/llvm-project/issues/192616
; https://github.com/llvm/llvm-project/issues/192618
;
; The loop bound (zext i16 %n to i64) is modeled as 'n' under the assumption
; that n is non-negative. For n < 0, the model's exit condition is never
; satisfied and the loop appears to be unbounded. Since the induction variable
; is <nsw>, an unbounded loop would be undefined behavior, which removes
; n < 0 from the loop's domain without a runtime check. In reality, the loop
; executes 65536 + n iterations, i.e. the non-negativity assumption must be
; checked at runtime.
;
; void f(char *A, unsigned short *N) {
;   unsigned short n = *N;
;   if (n == 0)
;     return;
;   long i = 0;
;   do
;     A[i] = 1;
;   while (++i != n);
; }

; CHECK:      Invalid Context:
; CHECK-NEXT:   [n] -> {  : n < 0 }
; CHECK:      Domain :=
; CHECK-NEXT:   [n] -> { Stmt_for[i0] : 0 <= i0 < n };

; AST:      if (1 && 0 == n <= -1)
; AST-EMPTY:
; AST-NEXT:     for (int c0 = 0; c0 < n; c0 += 1)
; AST-NEXT:       Stmt_for(c0);

define void @f(ptr %A, ptr %N) {
entry:
  %n = load i16, ptr %N
  %nz = zext i16 %n to i64
  %nonzero = icmp ne i16 %n, 0
  br i1 %nonzero, label %for, label %exit

for:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for ]
  %gep = getelementptr inbounds i8, ptr %A, i64 %i
  store i8 1, ptr %gep
  %i.next = add nuw nsw i64 %i, 1
  %done = icmp eq i64 %i.next, %nz
  br i1 %done, label %exit, label %for

exit:
  ret void
}
