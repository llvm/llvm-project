; Check that verification doesn't fail when reducing a function with
; unreachable blocks.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=basic-blocks --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck %s < %t

; CHECK-INTERESTINGNESS: test

; CHECK: define void @test() {
; CHECK-NEXT:   unreachable:
; CHECK-NEXT:     ret void

define void @test() {
entry:
  br label %exit

unreachable:                                        ; No predecessors!
  br label %exit

exit:
  ret void
}
