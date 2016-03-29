; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
; Test comes from a bug (15771) or better a feature request. It was not allowed
; in Polly in the old domain generation as ScalarEvolution cannot figure out the
; loop bounds. However, the new domain generation will detect the SCoP.

; CHECK:      Context:
; CHECK-NEXT: [n] -> {  : -2147483648 <= n <= 2147483647 }
;
; CHECK:      p0: %n
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_next
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [n] -> { Stmt_for_next[i0] : i0 >= 0 and 2i0 <= -3 + n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [n] -> { Stmt_for_next[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_next[i0] -> MemRef_A[i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:             [n] -> { Stmt_for_next[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }

@A = common global [100 x i32] zeroinitializer, align 16

define void @foo(i32 %n) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp2 = icmp sgt i32 %n, 0
  br i1 %cmp2, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry.split
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.next ]
  %arrayidx = getelementptr [100 x i32], [100 x i32]* @A, i64 0, i64 %indvar
  %0 = mul i64 %indvar, 2
  %1 = add i64 %0, 2
  %add1 = trunc i64 %1 to i32
  %cmp = icmp slt i32 %add1, %n
  %indvar.next = add i64 %indvar, 1
  br i1 %cmp, label %for.next, label %for.cond.for.end_crit_edge

for.next:
  %2 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %2, 1
  store i32 %add, i32* %arrayidx, align 4
  br label %for.body

for.cond.for.end_crit_edge:                       ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry.split
  ret void
}
