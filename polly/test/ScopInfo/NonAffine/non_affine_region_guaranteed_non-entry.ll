; RUN: opt %loadPolly -polly-allow-nonaffine-loops -polly-detect -polly-print-scops -disable-output < %s | FileCheck %s

; The SCoP contains a loop with multiple exit blocks (BBs after leaving
; the loop). The current implementation of deriving their domain derives
; only a common domain for all of the exit blocks. We disabled loops with
; multiple exit blocks until this is fixed.
; XFAIL: *

; The BasicBlock "guaranteed" is always executed inside the non-affine subregion
; region_entry->region_exit. As such, writes accesses in blocks that always
; execute are MustWriteAccesses. Before Polly commit r255473, we only assumed
; that the subregion's entry block is guaranteed to execute.

; CHECK-NOT: MayWriteAccess
; CHECK:      MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_region_entry__TO__region_exit[i0] -> MemRef_A[0] };
; CHECK-NOT: MayWriteAccess

define void @f(i32* %A, i32* %B, i32* %C, float %b) {
entry:
  br label %for.cond

for.cond:
  %indvar = phi i32 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i32 %indvar, 1024
  br i1 %exitcond, label %region_entry, label %return

region_entry:
  br label %bb2

bb2:
  br label %guaranteed

bb3:
  br label %bb3

guaranteed:
  %ptr = getelementptr i32, i32* %B, i32 %indvar
  %val = load i32, i32* %ptr
  %cmp = icmp eq i32 %val, 0
  store i32 0, i32* %A
  br i1 %cmp, label %bb5, label %bb6

bb5:
  br label %region_exit

bb6:
  %ptr2 = getelementptr i32, i32* %C, i32 %indvar
  %val2 = load i32, i32* %ptr2
  %cmp2 = icmp eq i32 %val2, 0
  br i1 %cmp2, label %region_exit, label %region_entry

region_exit:
  br label %for.inc

for.inc:
  %indvar.next = add i32 %indvar, 1
  br label %for.cond

return:
  ret void
}
