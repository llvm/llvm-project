; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=functions --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; Test handling of ifunc. Make sure function reduction doesn't create
; invalid ifunc


; CHECK-INTERESTINGNESS: define void @no_ifunc_interesting

; CHECK-FINAL: @ifunc1 = ifunc void (), ptr @has_ifunc
; CHECK-FINAL: define void @no_ifunc_interesting() {
; CHECK-FINAL-NOT: define

@ifunc1 = ifunc void (), ptr @has_ifunc


define ptr @has_ifunc() {
  ret ptr inttoptr (i64 124 to ptr)
}

define void @no_ifunc_interesting() {
  ret void
}

define void @no_ifunc_boring() {
  ret void
}
