; RUN: opt -O0 -enable-matrix -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s

; REQUIRES: asserts

; CHECK:      Pass Arguments:
; CHECK-NEXT: Target Transform Information
; CHECK-NEXT: Target Library Information
; CHECK-NEXT: Assumption Cache Tracker
; CHECK-NEXT:   FunctionPass Manager
; CHECK-NEXT:     Module Verifier
; CHECK-NEXT:     Instrument function entry/exit with calls to e.g. mcount() (pre inlining)
; CHECK-NEXT:     Dominator Tree Construction
; CHECK-NEXT:     Natural Loop Information
; CHECK-NEXT:     Lazy Branch Probability Analysis
; CHECK-NEXT:     Lazy Block Frequency Analysis
; CHECK-NEXT:     Optimization Remark Emitter
; CHECK-NEXT:     Basic Alias Analysis (stateless AA impl)
; CHECK-NEXT:     Function Alias Analysis Results
; CHECK-NEXT:     Lower the matrix intrinsics


define void @f() {
  ret void
}
