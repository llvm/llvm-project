; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=basic-blocks,simplify-cfg --test %python --test-arg %p/remove-bbs-sequence.py %s -o %t
; RUN: FileCheck %s < %t

; The interestingness test is that the CFG contains a loop. Verify that the
; unnecessary bb2 and bb3 are removed while still maintaining a loop.

define void @main() {
  bb0:
    br label %bb1
  bb1:
    br label %bb2
  bb2:
    br label %bb3
  bb3:
    %phi = phi i32 [ undef, %bb2 ]
    br label %bb4
  bb4:
    br label %bb1
}

; CHECK:define void @main() {
; CHECK-NEXT: bb0:
; CHECK-NEXT:   br label %bb4
; CHECK-EMPTY:
; CHECK-NEXT: bb4:
; CHECK-NEXT: %phi = phi i32 [ undef, %bb0 ], [ undef, %bb4 ]
; CHECK-NEXT: br label %bb4
; CHECK-NEXT:}
