; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-LOG
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; CHECK-INTERESTINGNESS: ret i32
; CHECK-FINAL: ret i32 0

; Test that we don't invoke the oracle more than necessary (e.g. check the
; oracle then perform some failable/redundant reduction, as opposed to check if
; a reduction will fail/be redundant before invoking the oracle). This prevents
; overestimation of the number of possible reductions and the number of times we
; attempt to reduce.

; IR passes
; CHECK-LOG: Saved new best reduction
; Module data
; CHECK-LOG: Saved new best reduction
; SimplifyCFG
; CHECK-LOG: Saved new best reduction
; CHECK-LOG-NOT: Saved new best reduction

define i32 @f() {
  ret i32 0
}