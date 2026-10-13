; RUN: opt %loadNPMPolly '-passes=polly<no-default-opts>' -S < %s | FileCheck %s

; Verify that Polly narrows the loop induction variable to i32 on a 32-bit
; target (e.g. Hexagon) when all loop bounds fit in 32 bits, and that it does
; NOT narrow when the lower bound is a wide non-constant i64 variable.
;
;   void narrow(int *A, int n) {
;     for (int i = 0; i < n; i++)
;       A[i] = i;
;   }

; 32-bit Hexagon-like target: pointer size = 32 bits.
target datalayout = "e-m:e-p:32:32:32-i64:64:64-i128:128:128-n32-S128"
target triple = "hexagon-unknown-linux-musl"

; CHECK-LABEL: @narrow
define void @narrow(ptr noalias %A, i32 %n) {
entry:
  br label %for.header

for.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %for.body ]
  %exitcond = icmp slt i32 %i, %n
  br i1 %exitcond, label %for.body, label %exit

for.body:
  %gep = getelementptr inbounds i32, ptr %A, i32 %i
  store i32 %i, ptr %gep
  %i.next = add nsw i32 %i, 1
  br label %for.header

exit:
  ret void
}

; The IV must be i32 and all loop arithmetic must stay in i32.
; CHECK:      polly.loop_header:
; CHECK-NEXT:   %polly.indvar = phi i32
; CHECK:        %polly.indvar_next = add nsw i32 %polly.indvar
; CHECK:        %polly.loop_cond = icmp slt i32 %polly.indvar_next


; When the lower bound is a non-constant i64 variable the guard must block
; narrowing to avoid an unsafe truncation of %start.
;
;   void no_narrow_wide_lb(int *A, long start, int n) {
;     for (long i = start; i < n; i++)
;       A[i] = (int)i;
;   }

; CHECK-LABEL: @no_narrow_wide_lb
define void @no_narrow_wide_lb(ptr noalias %A, i64 %start, i32 %n) {
entry:
  br label %for.header

for.header:
  %i = phi i64 [ %start, %entry ], [ %i.next, %for.body ]
  %n64 = sext i32 %n to i64
  %exitcond = icmp slt i64 %i, %n64
  br i1 %exitcond, label %for.body, label %exit

for.body:
  %gep = getelementptr inbounds i32, ptr %A, i64 %i
  %ival = trunc i64 %i to i32
  store i32 %ival, ptr %gep
  %i.next = add nsw i64 %i, 1
  br label %for.header

exit:
  ret void
}

; The IV must remain i64 because %start is a wide non-constant variable.
; Polly normalises the loop to start from 0, but keeps the type as i64.
; CHECK:      polly.loop_header:
; CHECK-NEXT:   %polly.indvar = phi i64
; CHECK:        %polly.indvar_next = add nsw i64 %polly.indvar
