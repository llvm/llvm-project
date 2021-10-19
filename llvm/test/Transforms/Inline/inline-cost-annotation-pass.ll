; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK:       Analyzing call of foo... (caller:main)
; CHECK: define i8 addrspace(1)** @foo() {
; CHECK:  cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:  %1 = inttoptr i64 754974720 to i8 addrspace(1)**
; CHECK:  cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:  ret i8 addrspace(1)** %1
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

define i8 addrspace(1)** @foo() {
  %1 = inttoptr i64 754974720 to i8 addrspace(1)**
  ret i8 addrspace(1)** %1
}

define i8 addrspace(1)** @main() {
  %1 = call i8 addrspace(1)** @foo()
  %2 = call i8 addrspace(1)** @foo()
  ret i8 addrspace(1)** %1
}
