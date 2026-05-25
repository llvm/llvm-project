; RUN: llvm-profdata merge %S/Inputs/indirect-call-vp-zeros.ll -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.profdata -S

;; Check that if we have a profile with VP metadat that has only zero values
;; with zero counts, we do not emit invalid VP metadata.

define void @test_call(ptr %fptr) {
entry:
  call void %fptr()
  ret void
}
