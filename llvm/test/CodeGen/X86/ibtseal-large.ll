; RUN: llc < %s -O2 -mtriple=x86_64-unknown-linux-gnu -x86-indirect-branch-tracking --code-model=large | FileCheck %s --check-prefix=CHECK-LARGE-IBTSEAL

; CHECK-LARGE-IBTSEAL: foo:
; CHECK-LARGE-IBTSEAL: endbr
; CHECK-LARGE-IBTSEAL: bar:
; CHECK-LARGE-IBTSEAL: endbr

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo() {
  ret void
}

define dso_local i8* @bar() {
  ret i8* bitcast (void ()* @foo to i8*)
}

!llvm.module.flags = !{!1}
!1 = !{i32 4, !"ibt-seal", i32 1}
