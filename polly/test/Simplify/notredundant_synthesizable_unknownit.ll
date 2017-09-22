; RUN: opt %loadPolly -polly-simplify -analyze < %s | FileCheck %s -match-full-lines
;
; Do not remove the scalar value write of %i.trunc in inner.for.
; It is used by body.
; %i.trunc is synthesizable in inner.for, so some code might think it is
; synthesizable everywhere such that no scalar write would be needed.
;
; Note that -polly-simplify rightfully removes %inner.cond. It should
; not have been added to the instruction list in the first place.
;
define void @func(i32 %n, i32* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  %zero = sext i32 0 to i64
  br i1 %j.cmp, label %inner.for, label %exit


    ; This loop has some unusual properties:
    ; * It has a known iteration count (8), therefore SCoP-compatible.
    ; * %i.trunc is synthesizable within the loop ({1,+,1}<%while.body>).
    ; * %i.trunc is not synthesizable outside of the loop, because its value is
    ;   unknown when exiting.
    ;   (should be 8, but ScalarEvolution currently seems unable to derive that)
    ;
    ; ScalarEvolution currently seems to not able to handle the %zero.
    ; If it becomes more intelligent, there might be other such loop constructs.
    inner.for:
      %i = phi i64 [%zero, %for], [%i.inc, %inner.for]
      %i.inc = add nuw nsw i64 %i, 1
      %i.trunc = trunc i64 %i.inc to i32
      %i.and = and i32 %i.trunc, 7
      %inner.cond = icmp eq i32 %i.and, 0
      br i1 %inner.cond, label %body, label %inner.for

    body:
      store i32 %i.trunc, i32* %A
      br label %inc


inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK:      After accesses {
; CHECK-NEXT:     Stmt_inner_for
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_inner_for[i0, i1] -> MemRef_i_trunc[] };
; CHECK-NEXT:     Stmt_body
; CHECK-NEXT:             MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_A[0] };
; CHECK-NEXT:             ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [n] -> { Stmt_body[i0] -> MemRef_i_trunc[] };
; CHECK-NEXT: }
