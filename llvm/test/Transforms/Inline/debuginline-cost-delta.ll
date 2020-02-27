; Require asserts for -debug-only
; REQUIRES: asserts

; RUN: opt < %s -inline -debug-only=inline-cost -disable-output -print-instruction-deltas 2>&1 | FileCheck %s

; CHECK:       Analyzing call of callee1... (caller:foo)
; CHECK: define i32 @callee1(i32 %x) {
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = 5
; CHECK:   %x1 = add i32 %x, 1
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = 5
; CHECK:   %x2 = add i32 %x1, 1
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = 5
; CHECK:   %x3 = add i32 %x2, 1
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = 0
; CHECK:   ret i32 %x3
; CHECK: }
; CHECK:      NumConstantArgs: 0
; CHECK:      NumConstantOffsetPtrArgs: 0
; CHECK:      NumAllocaArgs: 0
; CHECK:      NumConstantPtrCmps: 0
; CHECK:      NumConstantPtrDiffs: 0
; CHECK:      NumInstructionsSimplified: 1
; CHECK:      NumInstructions: 4
; CHECK:      SROACostSavings: 0
; CHECK:      SROACostSavingsLost: 0
; CHECK:      LoadEliminationCost: 0
; CHECK:      ContainsNoDuplicateCall: 0
; CHECK:      Cost: {{.*}}
; CHECK:      Threshold: {{.*}}

define i32 @foo(i32 %y) {
  %x = call i32 @callee1(i32 %y)
  ret i32 %x
}

define i32 @callee1(i32 %x) {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  ret i32 %x3
}
