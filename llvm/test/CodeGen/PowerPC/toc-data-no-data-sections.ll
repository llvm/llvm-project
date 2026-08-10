; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -data-sections=false -verify-machineinstrs < %s | FileCheck %s

@a1 = global i32 0, align 4 #0

define void @foo() {
entry:
  store i32 1, ptr @a1, align 4
  ret void
}

attributes #0 = { "toc-data" }

; CHECK: .toc
; CHECK-NEXT: .csect a1[TD],2
; CHECK-NEXT: .globl  a1[TD]
; CHECK-NEXT: .align  2
; CHECK-NOT: a1[TD]:
; CHECK-NEXT: .vbyte  4, 0
