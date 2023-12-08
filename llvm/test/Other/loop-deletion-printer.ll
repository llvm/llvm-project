; Make sure that Loop which was invalidated by loop-deletion
; does not lead to problems for -print-after-all and is just skipped.
;
; RUN: opt < %s -disable-output \
; RUN:     -passes=loop-instsimplify -print-after-all  2>&1 | FileCheck %s -check-prefix=SIMPLIFY
; RUN: opt < %s -disable-output \
; RUN:     -passes=loop-deletion,loop-instsimplify -print-after-all  2>&1 | FileCheck %s -check-prefix=DELETED
; RUN: opt < %s -disable-output \
; RUN:     -passes=loop-deletion,loop-instsimplify -print-after-all -print-module-scope  2>&1 | FileCheck %s -check-prefix=DELETED
;
; SIMPLIFY: IR Dump {{.*}} LoopInstSimplifyPass
; DELETED: IR Dump {{.*}}LoopDeletionPass {{.*}}(invalidated)
; DELETED-NOT: IR Dump {{.*}}LoopInstSimplifyPass

define void @deleteme() {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  %check = icmp ult i32 %iv.next, 3
  br i1 %check, label %loop, label %exit
exit:
  ret void
}

