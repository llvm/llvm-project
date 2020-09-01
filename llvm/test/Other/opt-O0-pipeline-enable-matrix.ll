; RUN: opt -enable-new-pm=0 -O0 -enable-matrix -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK:      Pass Arguments:
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Module Verifier
; CHECK-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; CHECK-NEXT:     Lower the matrix intrinsics (minimal)


define void @f() {
  ret void
}
