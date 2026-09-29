; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t


; Check that no invalid reduction is produced. Remaining unreachable
; blocks can leave instructions that dominate uses. When %bb3 was made
; unreachable it produced this verifier error:
; Instruction does not dominate all uses!
;   %i4 = icmp eq i32 0, 0
;   %i6 = select i1 %i4, i1 false, i1 false


; CHECK: define void @func() {
; CHECK-NEXT: bb:
; CHECK-NEXT: br label %bb1

; CHECK: bb1:
; CHECK-NEXT: label %bb3

; CHECK: bb2:
; CHECK-NEXT: br i1 false, label %bb1, label %bb2

; CHECK: bb3:
; CHECK-NEXT: %i = phi i32 [ 0, %bb1 ]
; CHECK: %i4 = icmp eq i32 0, 0
; CHECK-NEXT: br label %bb5

; CHECK: bb5:
; CHECK-NEXT: %i6 = select i1 %i4, i1 false, i1 false
; CHECK-NEXT: store i32 0
; CHECK-NEXT: ret void
define void @func() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb2, %bb
  br label %bb3

bb2:                                              ; preds = %bb2
  br i1 false, label %bb1, label %bb2

bb3:                                              ; preds = %bb1
  %i = phi i32 [ 0, %bb1 ]
  %i4 = icmp eq i32 0, 0
  br label %bb5

bb5:                                              ; preds = %bb3
  %i6 = select i1 %i4, i1 false, i1 false
  ; CHECK-INTERESTINGNESS: store
  store i32 0, ptr undef, align 4
  ret void
}
