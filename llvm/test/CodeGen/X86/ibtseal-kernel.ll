; RUN: llc < %s -O2 -mtriple=x86_64-unknown-linux-gnu -x86-indirect-branch-tracking --code-model=kernel | FileCheck %s --check-prefix=CHECK-KERNEL-IBTSEAL

; CHECK-KERNEL-IBTSEAL: foo:
; CHECK-KERNEL-IBTSEAL: endbr
; CHECK-KERNEL-IBTSEAL: bar:
; CHECK-KERNEL-IBTSEAL-NOT: endbr

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @foo() {
  ret void
}

define dso_local i8* @bar() {
  ret i8* bitcast (void ()* @foo to i8*)
}

!llvm.module.flags = !{!1}
!1 = !{i32 4, !"ibt-seal", i32 1}
