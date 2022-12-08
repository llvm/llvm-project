; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK:       Analyzing call of foo... (caller:main)
; CHECK: define ptr @foo() {
; CHECK:  cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:  %1 = inttoptr i64 754974720 to ptr
; CHECK:  cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:  ret ptr %1
; CHECK: }
; CHECK:       NumConstantArgs: {{.*}}
; CHECK:       NumConstantOffsetPtrArgs: {{.*}}
; CHECK:       NumAllocaArgs: {{.*}}
; CHECK:       NumConstantPtrCmps: {{.*}}
; CHECK:       NumConstantPtrDiffs: {{.*}}
; CHECK:       NumInstructionsSimplified: {{.*}}
; CHECK:       NumInstructions: {{.*}}
; CHECK:       SROACostSavings: {{.*}}
; CHECK:       SROACostSavingsLost: {{.*}}
; CHECK:       LoadEliminationCost: {{.*}}
; CHECK:       ContainsNoDuplicateCall: {{.*}}
; CHECK:       Cost: {{.*}}
; CHECK:       Threshold: {{.*}}
; CHECK-EMPTY:
; CHECK:  Analyzing call of foo... (caller:main)

define ptr @foo() {
  %1 = inttoptr i64 754974720 to ptr
  ret ptr %1
}

define ptr @main() {
  %1 = call ptr @foo()
  %2 = call ptr @foo()
  ret ptr %1
}
