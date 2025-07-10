; RUN: llc -O0 < %s -mtriple=s390x-linux-gnu 2>&1 | FileCheck %s

;; Tests that we can handle FAKE_USE instructions, emitting a comment for them
;; in the resulting assembly.

; CHECK:      .type   idd,@function
; CHECK:      # %bb.0:
; CHECK-NEXT: # fake_use:

define double @idd(double %d) {
entry:
  notail call void (...) @llvm.fake.use(double %d)
  ret double %d
}
