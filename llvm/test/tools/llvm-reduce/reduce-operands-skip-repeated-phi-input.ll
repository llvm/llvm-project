; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-skip --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK %s < %t

; Make sure if we're replacing the value in a phi, it's replaced for
; all repeats of the same incoming block.

; CHECK-INTERESTINGNESS: switch
; CHECK-INTERESTINGNESS: phi
; CHECK-INTERESTINGNESS-SAME: [ %gep1, %bb1 ]

; CHECK: %phi.ptr = phi ptr [ %arg1, %entry ], [ %arg1, %entry ], [ %gep1, %bb1 ]
define void @foo(i32 %arg0, ptr %arg1, ptr %arg2) {
entry:
  %gep0 = getelementptr i32, ptr %arg1, i32 10
  %gep1 = getelementptr i32, ptr %arg2, i32 12
  switch i32 %arg0, label %ret [
    i32 3, label %bb1
    i32 4, label %bb2
    i32 12, label %bb2
  ]

bb1:
  br label %bb2

bb2:
  %phi.ptr = phi ptr [ %gep0, %entry ], [ %gep0, %entry ], [ %gep1, %bb1 ]
  store volatile i32 0, ptr %phi.ptr
  br label %ret

ret:
  ret void
}
