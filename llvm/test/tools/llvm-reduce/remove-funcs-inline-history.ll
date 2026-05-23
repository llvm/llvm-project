; Test that llvm-reduce doesn't cause verifier errors with !inline_history
; metadata when removing functions. The metadata references to removed functions
; should become null (not ptr null) because we use replaceNonMetadataUsesWith.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=functions --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: define void @interesting()
define void @interesting() {
  ; CHECK-INTERESTINGNESS: call void @interesting()
  call void @interesting(), !inline_history !{ptr @uninteresting}
  call void @interesting(), !inline_history !{ptr @interesting, ptr @uninteresting}
  ret void
}

; CHECK-FINAL: define void @interesting()
; The metadata operand for @uninteresting becomes null.
; CHECK-FINAL: call void @interesting(), !inline_history ![[MD1:[0-9]+]]
; CHECK-FINAL: call void @interesting(), !inline_history ![[MD2:[0-9]+]]
; CHECK-FINAL: ret void

; CHECK-FINAL: ![[MD1]] = distinct !{null}
; CHECK-FINAL: ![[MD2]] = distinct !{ptr @interesting, null}

define void @uninteresting() {
  ret void
}
