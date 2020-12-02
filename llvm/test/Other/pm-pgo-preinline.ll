; RUN: opt -disable-verify -enable-new-pm=0 -pgo-kind=pgo-instr-gen-pipeline -mtriple=x86_64-- -Os -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-Osz
; RUN: opt -disable-verify -enable-new-pm=0 -pgo-kind=pgo-instr-gen-pipeline -mtriple=x86_64-- -Oz -debug-pass=Structure < %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-Osz


; CHECK-Osz: CallGraph Construction
; CHECK-Osz-NEXT: Call Graph SCC Pass Manager
; CHECK-Osz-NEXT: Function Integration/Inlining
; CHECK-Osz-NEXT: FunctionPass Manager
; CHECK-Osz-NEXT: Dominator Tree Construction
; CHECK-Osz-NEXT: SROA
; CHECK-Osz-NEXT: Early CSE
; CHECK-Osz-NEXT: Simplify the CFG
; CHECK-Osz-NEXT: Dominator Tree Construction
; CHECK-Osz-NEXT: Basic Alias Analysis (stateless AA impl)
; CHECK-Osz-NEXT: Function Alias Analysis Results
; CHECK-Osz-NEXT: Natural Loop Information
; CHECK-Osz-NEXT: Lazy Branch Probability Analysis
; CHECK-Osz-NEXT: Lazy Block Frequency Analysis
; CHECK-Osz-NEXT: Optimization Remark Emitter
; CHECK-Osz-NEXT: Combine redundant instructions

define void @foo() {
  ret void
}
