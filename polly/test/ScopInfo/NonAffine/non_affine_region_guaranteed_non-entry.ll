; RUN: opt %loadPolly -polly-detect -polly-scops -analyze < %s | FileCheck %s

; The BasicBlock "guaranteed" is always executed inside the non-affine subregion
; region_entry->region_exit. As such, writes accesses in blocks that always
; execute are MustWriteAccesses. Before Polly commit r255473, we only assumed
; that the subregion's entry block is guaranteed to execute.

; CHECK-NOT: MayWriteAccess
; CHECK:      MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:     { Stmt_region_entry__TO__region_exit[i0] -> MemRef_A[0] };
; CHECK-NOT: MayWriteAccess

define void @f(i32* %A, float %b) {
entry:
  br label %for.cond

for.cond:
  %indvar = phi i32 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i32 %indvar, 1024
  br i1 %exitcond, label %region_entry, label %return

region_entry:
  %cond_entry = fcmp oeq float %b, 3.0
  br i1 %cond_entry, label %bb2, label %bb3

bb2:
  br label %guaranteed

bb3:
  br label %guaranteed

guaranteed:
  store i32 0, i32* %A
  br i1 %cond_entry, label %bb5, label %bb6

bb5:
  br label %region_exit

bb6:
  br i1 %cond_entry, label %region_exit, label %bb3

region_exit:
  br label %for.inc

for.inc:
  %indvar.next = add i32 %indvar, 1
  br label %for.cond

return:
  ret void
}
